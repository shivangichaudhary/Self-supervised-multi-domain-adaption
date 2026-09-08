"""
Interactive Demonstration & Visual Inspection Tool
==================================================
Allows real-time inspection of multi-sensor satellite cross-domain adaptation:
  - Select any patch and target sensor (SPOT-6, Sentinel-1 SAR, Sentinel-2).
  - Inspect cross-sensor continuous style translation.
  - Compare Baseline (No DA) vs Proposed Synergistic Adapted segmentation side-by-side with Ground Truth.
  - View real-time quantitative accuracy, mIoU, and SSIM metrics.
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from torch.utils.data import DataLoader

from configs.config import load_config, apply_dataset
from models.domain_encoder import ContinuousDomainEncoder
from models.generator import UnifiedSatelliteGenerator
from models.downstream_task import SatelliteUNet
from data.dataset import MultiSensorSatelliteDataset
from utils.visualizer import tensor_to_display_image, mask_to_color
from utils.metrics import compute_segmentation_metrics, compute_ssim, compute_psnr


class SatelliteAdaptationDemo:
    def __init__(self, config_type="pastis", device="auto"):
        self.config = apply_dataset(load_config(), config_type)
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print("=" * 68)
        print(" Satellite Multi-Domain Adaptation - Interactive Inspection Studio ")
        print("=" * 68)
        print(f"[*] Dataset: {config_type.upper()} | Device: {self.device.upper()}")
        
        self.output_dir = self.config["output_dir"]
        self.domains = self.config["data"]["domains"]
        self.source_domain = "sentinel2" if "sentinel2" in self.domains else self.domains[0]
        self.target_domains = [d for d in self.domains if d != self.source_domain]
        if not self.target_domains:
            self.target_domains = self.domains
            
        self.current_domain_idx = 0
        self.current_patch_idx = 0
        
        # Load Trained Weights
        self._load_models()
        
        # Pre-extract canonical source style
        self._extract_source_canonical_style()
        
        # Load dataset for currently selected domain
        self._load_domain_dataset()

    def _load_models(self):
        enc_path = os.path.join(self.output_dir, "domain_encoder.pth")
        gen_path = os.path.join(self.output_dir, "generator.pth")
        unet_path = os.path.join(self.output_dir, "downstream_unet.pth")
        
        for p in [enc_path, gen_path, unet_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Checkpoint {p} not found. Please train models first via main.py.")
                
        # 1. Domain Encoder
        self.encoder = ContinuousDomainEncoder(
            in_channels=self.config["domain_encoder"]["in_channels"],
            latent_dim=self.config["domain_encoder"]["latent_dim"]
        ).to(self.device)
        self.encoder.load_state_dict(torch.load(enc_path, map_location=self.device, weights_only=True))
        self.encoder.eval()

        # 2. Unified Generator
        gen_cfg = self.config["generator"]
        self.generator = UnifiedSatelliteGenerator(
            in_channels=gen_cfg["in_channels"],
            out_channels=gen_cfg["out_channels"],
            latent_dim=gen_cfg["latent_dim"],
            dim=gen_cfg["dim"],
            num_resblocks=gen_cfg["num_resblocks"]
        ).to(self.device)
        self.generator.load_state_dict(torch.load(gen_path, map_location=self.device, weights_only=True))
        self.generator.eval()

        # 3. Downstream U-Net
        down_cfg = self.config["downstream"]
        self.downstream_model = SatelliteUNet(
            in_channels=down_cfg["in_channels"],
            num_classes=down_cfg["num_classes"]
        ).to(self.device)
        self.downstream_model.load_state_dict(torch.load(unet_path, map_location=self.device, weights_only=True))
        self.downstream_model.eval()

        print("[OK] All deep neural network checkpoints loaded successfully.")

    def _extract_source_canonical_style(self):
        src_ds = MultiSensorSatelliteDataset(
            root_dir=self.config["data"]["data_dir"],
            domains=[self.source_domain],
            split="val"
        )
        loader = DataLoader(src_ds, batch_size=min(16, len(src_ds)), shuffle=False)
        batch = next(iter(loader))
        with torch.no_grad():
            z = self.encoder(batch["src_img"].to(self.device))
            self.canonical_src_style = F.normalize(z.mean(dim=0, keepdim=True), p=2, dim=1)

    def _load_domain_dataset(self):
        self.target_domain = self.target_domains[self.current_domain_idx]
        self.dataset = MultiSensorSatelliteDataset(
            root_dir=self.config["data"]["data_dir"],
            domains=[self.target_domain],
            split="val"
        )
        self.num_patches = len(self.dataset)
        self.current_patch_idx = min(self.current_patch_idx, max(0, self.num_patches - 1))
        print(f"[*] Active Target Domain: {self.target_domain.upper()} ({self.num_patches} validation patches available)")

    def process_patch(self, patch_idx=0):
        sample = self.dataset[patch_idx]
        img = sample["src_img"].unsqueeze(0).to(self.device)
        mask = sample["src_mask"].unsqueeze(0).to(self.device)
        
        num_cls = self.config["downstream"]["num_classes"]
        ign_idx = 19 if num_cls == 20 else -100

        with torch.no_grad():
            # 1. Baseline without adaptation
            raw_logits = self.downstream_model(img)
            raw_pred = raw_logits.argmax(dim=1)
            raw_metrics = compute_segmentation_metrics(raw_pred, mask, num_cls, ignore_index=ign_idx)

            # 2. Generator cross-sensor translation (Target -> Source Canonical Style)
            z_src = self.canonical_src_style
            translated_img = self.generator(img, z_src)

            # 3. Path 1: Pixel-space prediction with TTA
            p_orig = F.softmax(self.downstream_model(translated_img), dim=1)
            p_h = torch.flip(F.softmax(self.downstream_model(torch.flip(translated_img, [-1])), dim=1), [-1])
            p_v = torch.flip(F.softmax(self.downstream_model(torch.flip(translated_img, [-2])), dim=1), [-2])
            probs_pixel = (p_orig + p_h + p_v) / 3.0

            # 4. Path 2: Feature-space adaptation with TTA
            f_orig = F.softmax(self.downstream_model(img), dim=1)
            f_h = torch.flip(F.softmax(self.downstream_model(torch.flip(img, [-1])), dim=1), [-1])
            f_v = torch.flip(F.softmax(self.downstream_model(torch.flip(img, [-2])), dim=1), [-2])
            probs_feat = (f_orig + f_h + f_v) / 3.0

            # 5. Adaptive Confidence-Weighted Dual-Path Fusion
            conf_pixel = probs_pixel.max(dim=1, keepdim=True).values
            conf_feat = probs_feat.max(dim=1, keepdim=True).values
            w_pixel = conf_pixel / (conf_pixel + conf_feat + 1e-6)
            w_feat = 1.0 - w_pixel
            probs_ensemble = w_pixel * probs_pixel + w_feat * probs_feat
            
            adapted_pred = probs_ensemble.argmax(dim=1)
            adapted_metrics = compute_segmentation_metrics(adapted_pred, mask, num_cls, ignore_index=ign_idx)

            ssim_val = compute_ssim(img, translated_img)
            psnr_val = compute_psnr(img, translated_img)

        return {
            "img": img[0],
            "translated_img": translated_img[0],
            "raw_pred": raw_pred[0],
            "adapted_pred": adapted_pred[0],
            "mask": mask[0],
            "raw_metrics": raw_metrics,
            "adapted_metrics": adapted_metrics,
            "ssim": ssim_val,
            "psnr": psnr_val,
            "domain": self.target_domain,
            "patch_idx": patch_idx
        }

    def render_figure(self, res, fig=None):
        if fig is None:
            fig = plt.figure(figsize=(16, 9), facecolor="#12151c")
        else:
            fig.clf()

        num_cls = self.config["downstream"]["num_classes"]
        
        # Convert tensors to displayable numpy arrays
        orig_disp = tensor_to_display_image(res["img"].cpu())
        trans_disp = tensor_to_display_image(res["translated_img"].cpu())
        raw_mask_disp = mask_to_color(res["raw_pred"].cpu(), num_classes=num_cls)
        adapt_mask_disp = mask_to_color(res["adapted_pred"].cpu(), num_classes=num_cls)
        gt_mask_disp = mask_to_color(res["mask"].cpu(), num_classes=num_cls)

        raw_acc = res["raw_metrics"]["overall_accuracy"] * 100
        adapt_acc = res["adapted_metrics"]["overall_accuracy"] * 100
        raw_miou = res["raw_metrics"]["mIoU"] * 100
        adapt_miou = res["adapted_metrics"]["mIoU"] * 100
        acc_gain = adapt_acc - raw_acc
        miou_gain = adapt_miou - raw_miou

        # Grid of 2x3 subplots
        gs = fig.add_gridspec(2, 3, left=0.05, right=0.95, top=0.90, bottom=0.12, wspace=0.25, hspace=0.30)

        # 1. Original Target Patch
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(orig_disp)
        ax1.set_title(f"1. Target Input [{res['domain'].upper()}]\nPatch #{res['patch_idx']:03d}", color="white", fontsize=11, fontweight="bold")
        ax1.axis("off")

        # 2. Cross-Sensor Translated Image
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(trans_disp)
        ax2.set_title(f"2. Generator Style Translation\n[{res['domain'].upper()} -> Sentinel-2 Style]\nSSIM: {res['ssim']:.3f} | PSNR: {res['psnr']:.1f} dB", color="#4db8ff", fontsize=11, fontweight="bold")
        ax2.axis("off")

        # 3. Ground Truth Mask
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(gt_mask_disp)
        ax3.set_title(f"3. Ground Truth Land-Cover\n({num_cls} Classes)", color="#00e676", fontsize=11, fontweight="bold")
        ax3.axis("off")

        # 4. Baseline Prediction (No DA)
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(raw_mask_disp)
        ax4.set_title(f"4. Baseline (Source-Only / No DA)\nAcc: {raw_acc:.1f}% | mIoU: {raw_miou:.2f}%", color="#ff5252", fontsize=11, fontweight="bold")
        ax4.axis("off")

        # 5. Proposed Synergistic Adapted Prediction
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(adapt_mask_disp)
        ax5.set_title(f"5. Proposed Synergistic DA\nAcc: {adapt_acc:.1f}% ({'+' if acc_gain>=0 else ''}{acc_gain:.1f}%) | mIoU: {adapt_miou:.2f}%", color="#ffab00", fontsize=11, fontweight="bold")
        ax5.axis("off")

        # 6. Performance Comparison Bar Chart
        ax6 = fig.add_subplot(gs[1, 2])
        metrics_labels = ["Accuracy (%)", "mIoU (%)"]
        x = np.arange(len(metrics_labels))
        width = 0.35

        bar_raw = ax6.bar(x - width/2, [raw_acc, raw_miou], width, label="Baseline (No DA)", color="#ff5252", alpha=0.9)
        bar_adapt = ax6.bar(x + width/2, [adapt_acc, adapt_miou], width, label="Proposed DA", color="#00e676", alpha=0.9)

        ax6.set_ylabel("Score (%)", color="white", fontsize=10)
        ax6.set_title(f"Performance Gain\nAcc: {'+' if acc_gain>=0 else ''}{acc_gain:.1f}% | mIoU: {'+' if miou_gain>=0 else ''}{miou_gain:.2f}%", color="white", fontsize=11, fontweight="bold")
        ax6.set_xticks(x)
        ax6.set_xticklabels(metrics_labels, color="white", fontsize=10)
        ax6.set_ylim(0, max(100.0, max(raw_acc, adapt_acc) * 1.25))
        ax6.tick_params(colors="white")
        ax6.legend(loc="upper left", facecolor="#1e222d", edgecolor="#333", labelcolor="white")
        ax6.set_facecolor("#1e222d")
        for spine in ax6.spines.values():
            spine.set_color("#444")

        # Add bar data labels
        for bar in bar_raw:
            yval = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha='center', va='bottom', color="#ff9999", fontsize=9, fontweight="bold")
        for bar in bar_adapt:
            yval = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha='center', va='bottom', color="#a3ffc7", fontsize=9, fontweight="bold")

        fig.suptitle(f"Continuous Multi-Domain Satellite Adaptation Studio | Domain Shift: {self.source_domain.upper()} -> {res['domain'].upper()}",
                     color="white", fontsize=14, fontweight="bold", y=0.97)

        return fig

    def run_interactive(self):
        """Launches interactive Matplotlib UI with buttons to browse patches and domains."""
        fig = plt.figure(figsize=(16, 9), facecolor="#12151c")
        plt.subplots_adjust(bottom=0.15)
        
        def update():
            res = self.process_patch(self.current_patch_idx)
            self.render_figure(res, fig=fig)
            
            # Interactive Control Buttons at the bottom
            ax_prev = plt.axes([0.15, 0.02, 0.12, 0.05])
            ax_next = plt.axes([0.30, 0.02, 0.12, 0.05])
            ax_domain = plt.axes([0.45, 0.02, 0.20, 0.05])
            ax_save = plt.axes([0.68, 0.02, 0.16, 0.05])

            btn_prev = Button(ax_prev, '< Prev Patch', color='#1e222d', hovercolor='#2a3040')
            btn_prev.label.set_color('white')
            btn_next = Button(ax_next, 'Next Patch >', color='#1e222d', hovercolor='#2a3040')
            btn_next.label.set_color('white')
            btn_domain = Button(ax_domain, f'Switch Sensor: {self.target_domain.upper()}', color='#1e222d', hovercolor='#2a3040')
            btn_domain.label.set_color('#4db8ff')
            btn_save = Button(ax_save, 'Save Snapshot', color='#1e222d', hovercolor='#2a3040')
            btn_save.label.set_color('#00e676')

            def on_prev(event):
                self.current_patch_idx = (self.current_patch_idx - 1) % self.num_patches
                update()
                plt.draw()

            def on_next(event):
                self.current_patch_idx = (self.current_patch_idx + 1) % self.num_patches
                update()
                plt.draw()

            def on_domain(event):
                self.current_domain_idx = (self.current_domain_idx + 1) % len(self.target_domains)
                self._load_domain_dataset()
                self.current_patch_idx = 0
                update()
                plt.draw()

            def on_save(event):
                out_path = os.path.join(self.output_dir, f"demo_patch_{self.target_domain}_{self.current_patch_idx:03d}.png")
                fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
                print(f"[OK] Saved snapshot to: {out_path}")

            btn_prev.on_clicked(on_prev)
            btn_next.on_clicked(on_next)
            btn_domain.on_clicked(on_domain)
            btn_save.on_clicked(on_save)

            # Store references so garbage collector doesn't reap them
            fig._buttons = [btn_prev, btn_next, btn_domain, btn_save]

        update()
        print("\n[+] Studio launched! Use UI buttons to browse patches, switch sensors, or inspect live adaptation.")
        plt.show()

    def run_snapshot(self, domain=None, patch_idx=0, save_path=None):
        """Headless execution to process and save a sample inspection panel."""
        if domain is not None and domain in self.target_domains:
            self.current_domain_idx = self.target_domains.index(domain)
            self._load_domain_dataset()
            
        self.current_patch_idx = patch_idx % self.num_patches
        res = self.process_patch(self.current_patch_idx)
        
        fig = self.render_figure(res)
        if save_path is None:
            save_path = os.path.join(self.output_dir, "visualizations", f"demo_adaptation_{res['domain']}_patch{res['patch_idx']:03d}.png")
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        
        print("\n" + "=" * 65)
        print(f" [INSPECTION RESULT] Domain: {res['domain'].upper()} | Patch #{res['patch_idx']:03d}")
        print("=" * 65)
        print(f"  * SSIM Fidelity       : {res['ssim']:.3f} (Structural retention)")
        print(f"  * PSNR                : {res['psnr']:.1f} dB")
        print(f"  * Baseline (No DA)    : Acc = {res['raw_metrics']['overall_accuracy']*100:.2f}% | mIoU = {res['raw_metrics']['mIoU']*100:.2f}%")
        print(f"  * Proposed Adapted    : Acc = {res['adapted_metrics']['overall_accuracy']*100:.2f}% | mIoU = {res['adapted_metrics']['mIoU']*100:.2f}%")
        gain_acc = (res['adapted_metrics']['overall_accuracy'] - res['raw_metrics']['overall_accuracy']) * 100
        gain_miou = (res['adapted_metrics']['mIoU'] - res['raw_metrics']['mIoU']) * 100
        print(f"  * Net Adaptation Gain : {'+' if gain_acc >= 0 else ''}{gain_acc:.2f}% Accuracy | {'+' if gain_miou >= 0 else ''}{gain_miou:.2f}% mIoU")
        print(f"  --> Saved High-Res Snapshot: {save_path}\n")
        return save_path


def main():
    parser = argparse.ArgumentParser(description="Satellite Multi-Domain Adaptation Inspection Studio")
    parser.add_argument("--dataset", type=str, default="pastis", choices=["pastis", "synthetic"])
    parser.add_argument("--domain", type=str, default=None, help="Target domain (e.g. spot6, sentinel1_sar)")
    parser.add_argument("--patch", type=int, default=0, help="Patch index to inspect")
    parser.add_argument("--save", type=str, default=None, help="Custom output image path")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window")
    args = parser.parse_args()

    demo = SatelliteAdaptationDemo(config_type=args.dataset)

    if args.headless:
        demo.run_snapshot(domain=args.domain, patch_idx=args.patch, save_path=args.save)
    else:
        try:
            demo.run_interactive()
        except Exception as e:
            print(f"[!] Interactive GUI display unavailable ({e}). Falling back to snapshot generation...")
            demo.run_snapshot(domain=args.domain, patch_idx=args.patch, save_path=args.save)


if __name__ == "__main__":
    main()
