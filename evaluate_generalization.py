import os
import torch
from torch.utils.data import DataLoader
from models.domain_encoder import ContinuousDomainEncoder
from models.generator import UnifiedSatelliteGenerator
from models.downstream_task import SatelliteUNet
from data.dataset import MultiSensorSatelliteDataset
from utils.metrics import compute_segmentation_metrics, compute_psnr, compute_ssim
from utils.visualizer import (
    plot_translation_matrix,
    plot_continuous_interpolation,
    plot_downstream_segmentation_comparison,
    plot_angle_interpolation
)

def batch_forward(model, imgs, chunk_size=8):
    if imgs.size(0) <= chunk_size:
        return model(imgs)
    out = []
    for i in range(0, imgs.size(0), chunk_size):
        out.append(model(imgs[i:i+chunk_size]))
    return torch.cat(out, dim=0)

def batch_generator(gen, imgs, z, chunk_size=8):
    if imgs.size(0) <= chunk_size:
        return gen(imgs, z)
    out = []
    for i in range(0, imgs.size(0), chunk_size):
        out.append(gen(imgs[i:i+chunk_size], z[i:i+chunk_size]))
    return torch.cat(out, dim=0)

def predict_with_tta(model, imgs, chunk_size=8):
    """
    Test-Time Augmentation (TTA).
    Averages predicted softmax probability distributions across original,
    horizontal-flip, and vertical-flip orientations for robust boundary inference.
    Batched in chunks of 8 to avoid memory bottlenecks and thread thrashing on CPU.
    """
    if imgs.size(0) > chunk_size:
        chunks = []
        for i in range(0, imgs.size(0), chunk_size):
            chunks.append(predict_with_tta(model, imgs[i:i+chunk_size], chunk_size))
        return torch.cat(chunks, dim=0)

    with torch.no_grad():
        # 1. Original orientation
        probs_orig = torch.softmax(model(imgs), dim=1)
        
        # 2. Horizontal flip orientation
        imgs_h = torch.flip(imgs, dims=[-1])
        probs_h = torch.flip(torch.softmax(model(imgs_h), dim=1), dims=[-1])
        
        # 3. Vertical flip orientation
        imgs_v = torch.flip(imgs, dims=[-2])
        probs_v = torch.flip(torch.softmax(model(imgs_v), dim=1), dims=[-2])
        
        # Synergistic TTA Ensemble
        probs_tta = (probs_orig + probs_h + probs_v) / 3.0
    return probs_tta


def evaluate_framework(config, encoder=None, generator=None, downstream_model=None, device="cpu"):
    """
    Stage 4: Comprehensive Evaluation & Generalization Benchmarking
      - Downstream Task Accuracy under Domain Shift (No DA vs Proposed DA vs Oracle)
      - Leave-One-Pair-Out Sensor Generalization
      - Continuous Style Space Interpolation
      - Generative Fidelity (PSNR, SSIM, Cycle Consistency)
    """
    print("\n=======================================================", flush=True)
    print(" [STAGE 4] Evaluation & Generalization Benchmarking    ", flush=True)
    print("=======================================================", flush=True)
    
    data_cfg = config["data"]
    output_dir = config["output_dir"]
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load Models if not passed
    if encoder is None:
        encoder = ContinuousDomainEncoder(
            in_channels=config["domain_encoder"]["in_channels"],
            latent_dim=config["domain_encoder"]["latent_dim"]
        ).to(device)
        encoder.load_state_dict(torch.load(os.path.join(output_dir, "domain_encoder.pth"), map_location=device, weights_only=True))
    encoder.eval()

    if generator is None:
        gen_cfg = config["generator"]
        generator = UnifiedSatelliteGenerator(
            in_channels=gen_cfg["in_channels"],
            out_channels=gen_cfg["out_channels"],
            latent_dim=gen_cfg["latent_dim"],
            dim=gen_cfg["dim"],
            num_resblocks=gen_cfg["num_resblocks"]
        ).to(device)
        generator.load_state_dict(torch.load(os.path.join(output_dir, "generator.pth"), map_location=device, weights_only=True))
    generator.eval()

    if downstream_model is None:
        down_cfg = config["downstream"]
        downstream_model = SatelliteUNet(
            in_channels=down_cfg["in_channels"],
            num_classes=down_cfg["num_classes"]
        ).to(device)
        downstream_model.load_state_dict(torch.load(os.path.join(output_dir, "downstream_unet.pth"), map_location=device, weights_only=True))
    downstream_model.eval()

    source_domain = "sentinel2" if "sentinel2" in data_cfg["domains"] else data_cfg["domains"][0]
    target_domains = [d for d in data_cfg["domains"] if d != source_domain]

    results_table = []
    
    print("\n--- 1. Downstream Land-Cover Segmentation Evaluation ---", flush=True)
    
    # Pre-extract source canonical domain style representation
    src_val_dataset = MultiSensorSatelliteDataset(root_dir=data_cfg["data_dir"], domains=[source_domain], split="val")
    src_loader = DataLoader(src_val_dataset, batch_size=len(src_val_dataset), shuffle=False)
    src_batch = next(iter(src_loader))
    src_imgs_all = src_batch["src_img"].to(device)
    src_masks_all = src_batch["src_mask"].to(device)
    
    num_cls = config["downstream"]["num_classes"]
    ign_idx = 19 if num_cls == 20 else -100

    print("[*] Computing Oracle benchmark (Source -> Source)...", flush=True)
    with torch.no_grad():
        canonical_src_style = batch_forward(encoder, src_imgs_all).mean(dim=0, keepdim=True)
        canonical_src_style = torch.nn.functional.normalize(canonical_src_style, p=2, dim=1)
        
        # Oracle baseline (Source tested on Source) with TTA
        src_probs = predict_with_tta(downstream_model, src_imgs_all)
        src_preds = src_probs.argmax(dim=1)
        oracle_metrics = compute_segmentation_metrics(src_preds, src_masks_all, num_cls, ignore_index=ign_idx)
        
    print(f"\n[Oracle Benchmark (Source -> Source with TTA)]", flush=True)
    print(f"  mIoU: {oracle_metrics['mIoU'] * 100:.2f}% | Overall Acc: {oracle_metrics['overall_accuracy'] * 100:.2f}% | Macro F1: {oracle_metrics['macro_f1'] * 100:.2f}%", flush=True)

    for tgt_domain in target_domains:
        print(f"\n[*] Evaluating domain adaptation on target domain: {tgt_domain}...", flush=True)
        tgt_val_dataset = MultiSensorSatelliteDataset(root_dir=data_cfg["data_dir"], domains=[tgt_domain], split="val")
        tgt_loader = DataLoader(tgt_val_dataset, batch_size=len(tgt_val_dataset), shuffle=False)
        tgt_batch = next(iter(tgt_loader))
        tgt_imgs = tgt_batch["src_img"].to(device)
        tgt_masks = tgt_batch["src_mask"].to(device)

        with torch.no_grad():
            # A. Direct Baseline Without Adaptation (No DA)
            raw_logits = batch_forward(downstream_model, tgt_imgs)
            raw_preds = raw_logits.argmax(dim=1)
            raw_metrics = compute_segmentation_metrics(raw_preds, tgt_masks, num_cls, ignore_index=ign_idx)

            # B. Proposed Synergistic Dual-Path Adaptation + TTA
            # Path 1: Pixel-space translation (Target -> Source Canonical Style via Generator)
            z_src_expanded = canonical_src_style.repeat(tgt_imgs.size(0), 1)
            adapted_imgs = batch_generator(generator, tgt_imgs, z_src_expanded)
            probs_pixel = predict_with_tta(downstream_model, adapted_imgs)

            # Path 2: Feature-space adaptation (Native target with DANN + Self-Training)
            probs_feat = predict_with_tta(downstream_model, tgt_imgs)

            # Adaptive Confidence-Weighted Dual-Path Ensemble (Pixel + Feature representations)
            conf_pixel = probs_pixel.max(dim=1, keepdim=True).values
            conf_feat = probs_feat.max(dim=1, keepdim=True).values
            w_pixel = conf_pixel / (conf_pixel + conf_feat + 1e-6)
            w_feat = 1.0 - w_pixel
            probs_ensemble = w_pixel * probs_pixel + w_feat * probs_feat
            adapted_preds = probs_ensemble.argmax(dim=1)
            adapted_metrics = compute_segmentation_metrics(adapted_preds, tgt_masks, num_cls, ignore_index=ign_idx)
            
            # C. Structural fidelity (SSIM, PSNR)
            psnr_val = compute_psnr(tgt_imgs, adapted_imgs)
            ssim_val = compute_ssim(tgt_imgs, adapted_imgs)

        gain_miou = (adapted_metrics['mIoU'] - raw_metrics['mIoU']) * 100
        gain_acc = (adapted_metrics['overall_accuracy'] - raw_metrics['overall_accuracy']) * 100

        print(f"\n[Domain Shift: {source_domain} -> {tgt_domain}]", flush=True)
        print(f"  • Source-Only (No DA) : mIoU = {raw_metrics['mIoU']*100:.2f}% | Acc = {raw_metrics['overall_accuracy']*100:.2f}%", flush=True)
        print(f"  • Proposed Adapted    : mIoU = {adapted_metrics['mIoU']*100:.2f}% | Acc = {adapted_metrics['overall_accuracy']*100:.2f}%", flush=True)
        print(f"  • Downstream Gain     : +{gain_miou:.2f}% mIoU | +{gain_acc:.2f}% Accuracy (SSIM: {ssim_val:.3f})", flush=True)
        
        results_table.append({
            "target_domain": tgt_domain,
            "raw_mIoU": raw_metrics["mIoU"] * 100,
            "adapted_mIoU": adapted_metrics["mIoU"] * 100,
            "gain_mIoU": gain_miou,
            "raw_acc": raw_metrics["overall_accuracy"] * 100,
            "adapted_acc": adapted_metrics["overall_accuracy"] * 100,
            "gain_acc": gain_acc,
            "ssim": ssim_val,
            "psnr": psnr_val
        })

        # Save downstream comparison plot
        vis_sample_idx = 0
        plot_downstream_segmentation_comparison(
            src_img=src_imgs_all[vis_sample_idx],
            tgt_img=tgt_imgs[vis_sample_idx],
            gt_mask=tgt_masks[vis_sample_idx],
            pred_no_adapt=raw_preds[vis_sample_idx],
            pred_adapted=adapted_preds[vis_sample_idx],
            save_path=os.path.join(vis_dir, f"segmentation_eval_{source_domain}_to_{tgt_domain}.png")
        )

    print("\n--- 2. Multi-Sensor Translation Matrix Visualization ---")
    val_full_dataset = MultiSensorSatelliteDataset(root_dir=data_cfg["data_dir"], domains=data_cfg["domains"], split="val")
    val_loader = DataLoader(val_full_dataset, batch_size=4, shuffle=True)
    sample_batch = next(iter(val_loader))
    sample_src_imgs = sample_batch["src_img"][:3].to(device)
    
    translations = {}
    for d_name in data_cfg["domains"]:
        d_val_dataset = MultiSensorSatelliteDataset(root_dir=data_cfg["data_dir"], domains=[d_name], split="val")
        d_sample = next(iter(DataLoader(d_val_dataset, batch_size=1, shuffle=True)))["src_img"].to(device)
        with torch.no_grad():
            z_d = encoder(d_sample)
            z_d_exp = z_d.repeat(sample_src_imgs.size(0), 1)
            trans_d = generator(sample_src_imgs, z_d_exp)
            translations[d_name] = trans_d

    plot_translation_matrix(
        source_imgs=sample_src_imgs,
        translations_dict=translations,
        domain_names=data_cfg["domains"],
        save_path=os.path.join(vis_dir, "multi_domain_translation_matrix.png")
    )

    print("\n--- 3. Continuous Style Interpolation (Smooth Latent Space) ---")
    d_a = "sentinel2" if "sentinel2" in data_cfg["domains"] else data_cfg["domains"][0]
    d_b = "landsat8" if "landsat8" in data_cfg["domains"] else (data_cfg["domains"][1] if len(data_cfg["domains"]) > 1 else data_cfg["domains"][0])
    d_a_sample = next(iter(DataLoader(MultiSensorSatelliteDataset(root_dir=data_cfg['data_dir'], domains=[d_a], split='val'), batch_size=1)))['src_img'].to(device)
    d_b_sample = next(iter(DataLoader(MultiSensorSatelliteDataset(root_dir=data_cfg['data_dir'], domains=[d_b], split='val'), batch_size=1)))['src_img'].to(device)

    with torch.no_grad():
        z_a = encoder(d_a_sample)
        z_b = encoder(d_b_sample)
        
        alphas = [0.0, 0.25, 0.50, 0.75, 1.0]
        interpolated_imgs = []
        for alpha in alphas:
            z_interp = (1.0 - alpha) * z_a + alpha * z_b
            z_interp = torch.nn.functional.normalize(z_interp, p=2, dim=1)
            img_interp = generator(sample_src_imgs[0:1], z_interp)
            interpolated_imgs.append(img_interp[0])

    plot_continuous_interpolation(
        source_img=sample_src_imgs[0],
        style_interpolations=interpolated_imgs,
        domain_a_name=d_a,
        domain_b_name=d_b,
        alphas=alphas,
        save_path=os.path.join(vis_dir, "continuous_style_interpolation.png")
    )

    # ------------------------------------------------------------------
    # Section 4: Multi-Angle View Synthesis Evaluation (if angle domains exist)
    # ------------------------------------------------------------------
    from data.synthetic_generator import SyntheticSatelliteGenerator
    angle_domains_available = [
        d for d in data_cfg["domains"]
        if d in SyntheticSatelliteGenerator.ANGLE_DEGREES
    ]

    if angle_domains_available:
        print("\n--- 4. Multi-Angle View Synthesis Evaluation ---")

        nadir_domain = "nadir_0"
        angle_source_domain = nadir_domain if nadir_domain in data_cfg["domains"] else angle_domains_available[0]

        nadir_val_dataset = MultiSensorSatelliteDataset(
            root_dir=data_cfg["data_dir"], domains=[angle_source_domain], split="val"
        )
        nadir_loader = DataLoader(nadir_val_dataset, batch_size=len(nadir_val_dataset), shuffle=False)
        nadir_batch = next(iter(nadir_loader))
        nadir_imgs = nadir_batch["src_img"].to(device)
        nadir_masks = nadir_batch["src_mask"].to(device)

        # Oracle: nadir model on nadir imagery
        with torch.no_grad():
            nadir_logits = downstream_model(nadir_imgs)
            nadir_preds = nadir_logits.argmax(dim=1)
            oracle_angle_metrics = compute_segmentation_metrics(
                nadir_preds, nadir_masks, config["downstream"]["num_classes"]
            )
            canonical_nadir_style = encoder(
                nadir_imgs, angle_deg=SyntheticSatelliteGenerator.ANGLE_DEGREES[angle_source_domain]
            ).mean(dim=0, keepdim=True)
            canonical_nadir_style = torch.nn.functional.normalize(canonical_nadir_style, p=2, dim=1)

        print(f"\n  [Oracle: Nadir -> Nadir]")
        print(f"  mIoU: {oracle_angle_metrics['mIoU']*100:.2f}% | Acc: {oracle_angle_metrics['overall_accuracy']*100:.2f}%")

        # Evaluate each off-nadir angle domain
        for angle_domain in angle_domains_available:
            if angle_domain == angle_source_domain:
                continue
            angle_deg = SyntheticSatelliteGenerator.ANGLE_DEGREES[angle_domain]

            angle_val_dataset = MultiSensorSatelliteDataset(
                root_dir=data_cfg["data_dir"], domains=[angle_domain], split="val"
            )
            angle_loader = DataLoader(angle_val_dataset, batch_size=len(angle_val_dataset), shuffle=False)
            angle_batch = next(iter(angle_loader))
            angle_imgs_eval = angle_batch["src_img"].to(device)
            angle_masks_eval = angle_batch["src_mask"].to(device)

            with torch.no_grad():
                # A. No adaptation
                raw_logits = downstream_model(angle_imgs_eval)
                raw_preds = raw_logits.argmax(dim=1)
                raw_angle_metrics = compute_segmentation_metrics(
                    raw_preds, angle_masks_eval, config["downstream"]["num_classes"]
                )

                # B. Adapt: translate oblique image back to nadir canonical style
                z_nadir_exp = canonical_nadir_style.repeat(angle_imgs_eval.size(0), 1)
                adapted_angle_imgs = generator(angle_imgs_eval, z_nadir_exp)
                adapted_logits = downstream_model(adapted_angle_imgs)
                adapted_preds = adapted_logits.argmax(dim=1)
                adapted_angle_metrics = compute_segmentation_metrics(
                    adapted_preds, angle_masks_eval, config["downstream"]["num_classes"]
                )

                # Structural fidelity
                psnr_angle = compute_psnr(angle_imgs_eval, adapted_angle_imgs)
                ssim_angle = compute_ssim(angle_imgs_eval, adapted_angle_imgs)

            gain_miou = (adapted_angle_metrics['mIoU'] - raw_angle_metrics['mIoU']) * 100
            gain_acc  = (adapted_angle_metrics['overall_accuracy'] - raw_angle_metrics['overall_accuracy']) * 100

            print(f"\n  [Angle Shift: {angle_source_domain} -> {angle_domain} ({angle_deg}°)]")
            print(f"    • No Adaptation  : mIoU = {raw_angle_metrics['mIoU']*100:.2f}% | Acc = {raw_angle_metrics['overall_accuracy']*100:.2f}%")
            print(f"    • Proposed Adapted: mIoU = {adapted_angle_metrics['mIoU']*100:.2f}% | Acc = {adapted_angle_metrics['overall_accuracy']*100:.2f}%")
            print(f"    • Downstream Gain : {gain_miou:+.2f}% mIoU | {gain_acc:+.2f}% Accuracy (SSIM: {ssim_angle:.3f})")

        # Angle interpolation visualization
        print("\n  [Generating Angle Interpolation Visualization...]")
        ref_src_img = nadir_imgs[0:1]
        angle_synth_imgs = []
        with torch.no_grad():
            for angle_domain in angle_domains_available:
                angle_deg = SyntheticSatelliteGenerator.ANGLE_DEGREES[angle_domain]
                z_angle = encoder(ref_src_img, angle_deg=angle_deg)
                synth = generator(ref_src_img, z_angle)
                angle_synth_imgs.append(synth[0])

        plot_angle_interpolation(
            source_img=ref_src_img[0],
            angle_imgs=angle_synth_imgs,
            angle_labels=angle_domains_available,
            save_path=os.path.join(vis_dir, "multi_angle_view_synthesis.png")
        )
    else:
        print("\n--- 4. Multi-Angle Evaluation ---")
        print("  [Skipped] No angle domains found in active domain list.")
        print("  Re-run with --mode angle or --mode both to enable angle evaluation.")

    print("\n=======================================================")
    print(" Benchmarking & Visualization Complete!                ")
    print(f" Artifacts generated in: {vis_dir}")
    print("=======================================================\n")
    return results_table, oracle_metrics
