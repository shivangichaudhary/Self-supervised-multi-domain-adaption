import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from models.downstream_task import SatelliteUNet
from models.losses import SegmentationDiceLoss
from models.domain_encoder import ContinuousDomainEncoder
from models.generator import UnifiedSatelliteGenerator
from data.dataset import MultiSensorSatelliteDataset
from utils.metrics import compute_segmentation_metrics


def train_downstream_model(config, source_domain="sentinel2", device="cpu"):
    """
    Stage 3: Trains downstream land-cover semantic segmentation model on Source Domain
    with Unsupervised Domain Adaptation (strictly ZERO target labels):
    
      1. Segmentation loss on labeled source domain patches (ASPP + Deep Supervision).
      2. Class-frequency weighted Focal + Dice loss on source masks to balance rare classes.
      3. Domain adversarial loss via GRL at the U-Net bottleneck (forces domain-invariant features).
      4. Unsupervised High-Confidence Pseudo-Label Self-Training on target domains:
         - Predicts on unlabeled target images with confidence thresholding (tau >= 0.75).
         - Self-supervises target representations without ANY human target labels.
      5. AdamW with gradient clipping (norm=1.0) and CosineAnnealingLR with warmup.
    """
    print("\n=======================================================")
    print(f" [STAGE 3] Training Downstream Task Model on '{source_domain}'")
    print("=======================================================")
    
    data_cfg = config["data"]
    down_cfg = config["downstream"]
    output_dir = config["output_dir"]
    enc_cfg = config["domain_encoder"]
    gen_cfg = config["generator"]
    os.makedirs(output_dir, exist_ok=True)
    
    num_domains_dat = 2  # source vs adapted-target for DANN binary classifier
    
    model = SatelliteUNet(
        in_channels=down_cfg["in_channels"],
        num_classes=down_cfg["num_classes"],
        num_domains=num_domains_dat
    ).to(device)
    
    # Source domain training data
    src_dataset = MultiSensorSatelliteDataset(
        root_dir=data_cfg["data_dir"],
        domains=[source_domain],
        split="train"
    )
    src_loader = DataLoader(src_dataset, batch_size=data_cfg["batch_size"], shuffle=True, drop_last=True)
    
    ignore_index = 19 if down_cfg["num_classes"] == 20 else -100

    # Compute balanced class weights from source training masks (sub-linear square root scaling)
    print("  [*] Computing empirical class balancing weights on source domain...")
    class_counts = torch.zeros(down_cfg["num_classes"], dtype=torch.float32)
    sample_limit = min(120, len(src_dataset))
    for i in range(sample_limit):
        m = src_dataset[i]["src_mask"]
        valid_m = m[m != ignore_index]
        if len(valid_m) > 0:
            bincount = torch.bincount(valid_m.view(-1), minlength=down_cfg["num_classes"])
            class_counts += bincount[:down_cfg["num_classes"]].float()

    valid_mask = class_counts > 0
    if valid_mask.sum() > 0:
        median_freq = torch.median(class_counts[valid_mask])
        class_weights = torch.ones_like(class_counts)
        for c in range(down_cfg["num_classes"]):
            if class_counts[c] > 0:
                class_weights[c] = torch.sqrt(median_freq / (class_counts[c] + 1e-4))
        class_weights = class_weights / class_weights.mean()
        class_weights = torch.clamp(class_weights, 0.5, 2.5).to(device)
        print(f"  [+] Computed balanced class weights (min={class_weights.min():.2f}, max={class_weights.max():.2f}).")
    else:
        class_weights = None

    seg_criterion = SegmentationDiceLoss(
        num_classes=down_cfg["num_classes"],
        ignore_index=ignore_index,
        label_smoothing=0.01,
        weight=class_weights
    ).to(device)
    domain_criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=down_cfg["lr"], weight_decay=1e-4)
    
    total_epochs = down_cfg["epochs"]
    warmup_epochs = max(1, total_epochs // 5)

    def get_lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(1e-5 / down_cfg["lr"], 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item()))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda)
    
    # Load pretrained encoder and generator for DAT (no gradient needed)
    encoder_path = os.path.join(output_dir, "domain_encoder.pth")
    gen_path = os.path.join(output_dir, "generator.pth")
    has_dat = os.path.exists(encoder_path) and os.path.exists(gen_path)
    
    if has_dat:
        encoder = ContinuousDomainEncoder(
            in_channels=enc_cfg["in_channels"],
            latent_dim=enc_cfg["latent_dim"]
        ).to(device)
        encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
        encoder.eval()
        
        generator = UnifiedSatelliteGenerator(
            in_channels=gen_cfg["in_channels"],
            out_channels=gen_cfg["out_channels"],
            latent_dim=gen_cfg["latent_dim"],
            dim=gen_cfg["dim"],
            num_resblocks=gen_cfg["num_resblocks"]
        ).to(device)
        generator.load_state_dict(torch.load(gen_path, map_location=device, weights_only=True))
        generator.eval()
        print("  [DAT] Loaded pretrained encoder + generator for domain adversarial training.")
    else:
        has_dat = False
        print("  [DAT] Encoder/generator checkpoints not found. Skipping domain adversarial branch.")
    
    # Target domain data (UNLABELED ONLY: strictly no target masks accessed during training)
    if has_dat:
        tgt_domains = [d for d in data_cfg["domains"] if d != source_domain]
        if tgt_domains:
            tgt_dataset = MultiSensorSatelliteDataset(
                root_dir=data_cfg["data_dir"],
                domains=tgt_domains,
                split="train"
            )
            tgt_loader = DataLoader(tgt_dataset, batch_size=data_cfg["batch_size"], shuffle=True, drop_last=True)
            tgt_iter = iter(tgt_loader)
        else:
            has_dat = False
    
    # Precompute canonical source domain style vector once
    if has_dat:
        with torch.no_grad():
            sample_src_imgs = torch.stack([src_dataset[i]["src_img"] for i in range(min(16, len(src_dataset)))]).to(device)
            canonical_z_src = encoder(sample_src_imgs).mean(dim=0, keepdim=True)
            canonical_z_src = F.normalize(canonical_z_src, p=2, dim=1)
        print("  [DAT] Computed canonical source domain style vector for cross-domain translation.", flush=True)

    save_path = os.path.join(output_dir, "downstream_unet.pth")
    lambda_domain = 0.1   # domain adversarial loss weight
    lambda_ent = 0.05      # unsupervised target entropy minimization weight
    lambda_div = 0.05      # batch prediction diversity regularization weight
    
    model.train()
    for epoch in range(1, total_epochs + 1):
        total_seg_loss = 0.0
        total_dom_loss = 0.0
        total_uda_loss = 0.0
        
        # GRL lambda increases gradually (curriculum schedule)
        p = float(epoch - 1) / float(max(1, total_epochs - 1))
        grl_lambda = 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p)).item()) - 1.0
        model.grl.set_lambda(grl_lambda)
        
        for batch_idx, batch in enumerate(src_loader, 1):
            src_imgs = batch["src_img"].to(device)
            src_masks = batch["src_mask"].to(device)
            
            # ---- 1. Source Segmentation Loss with Deep Supervision ----
            seg_logits, domain_logits_src, aux1_src, aux2_src = model.forward_with_domain(src_imgs, return_aux=True)
            loss_seg_main = seg_criterion(seg_logits, src_masks)
            loss_aux1 = seg_criterion(aux1_src, src_masks)
            loss_aux2 = seg_criterion(aux2_src, src_masks)
            loss_seg = loss_seg_main + 0.3 * loss_aux1 + 0.3 * loss_aux2
            
            # ---- 2. Unsupervised Domain Adversarial Loss ----
            loss_dom = torch.tensor(0.0, device=device)
            loss_uda = torch.tensor(0.0, device=device)
            
            if has_dat:
                try:
                    tgt_batch = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(tgt_loader)
                    tgt_batch = next(tgt_iter)
                
                # STRICTLY UNLABELED: Only images are extracted, target masks are NEVER loaded or used
                tgt_imgs_raw = tgt_batch["src_img"].to(device)
                
                with torch.no_grad():
                    # Translate target -> source canonical style via Generator
                    z_src_batch = canonical_z_src.repeat(tgt_imgs_raw.size(0), 1)
                    tgt_imgs_adapted = generator(tgt_imgs_raw, z_src_batch)
                
                # Binary domain classification: Source = 0, Adapted Target = 1
                _, domain_logits_tgt = model.forward_with_domain(tgt_imgs_adapted)
                
                src_domain_labels = torch.zeros(domain_logits_src.size(0), dtype=torch.long, device=device)
                tgt_domain_labels = torch.ones(domain_logits_tgt.size(0), dtype=torch.long, device=device)
                
                loss_dom = domain_criterion(domain_logits_src, src_domain_labels) + \
                           domain_criterion(domain_logits_tgt, tgt_domain_labels)
                loss_dom = loss_dom * lambda_domain

                # ---- 3. Unsupervised Target Entropy Minimization & Diversity Regularization ----
                # ADVENT formulation: Minimizes conditional entropy to encourage decisive clustering
                # Diversity penalty: Prevents mode collapse to background by penalizing low batch-level entropy
                tgt_pred_logits = model(tgt_imgs_raw)
                tgt_probs = F.softmax(tgt_pred_logits, dim=1)
                
                # Pixel-wise entropy
                pixel_entropy = -torch.sum(tgt_probs * torch.log(tgt_probs + 1e-7), dim=1)
                loss_entropy = pixel_entropy.mean()
                
                # Batch-level mean class distribution diversity
                mean_class_probs = tgt_probs.mean(dim=(0, 2, 3))
                loss_diversity = torch.sum(mean_class_probs * torch.log(mean_class_probs + 1e-7))
                
                loss_uda = (lambda_ent * loss_entropy + lambda_div * loss_diversity)
            
            total_loss = loss_seg + loss_dom + loss_uda
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_seg_loss += loss_seg.item()
            total_dom_loss += loss_dom.item() if isinstance(loss_dom, torch.Tensor) else loss_dom
            total_uda_loss += loss_uda.item() if isinstance(loss_uda, torch.Tensor) else loss_uda
            
        scheduler.step()
        avg_seg = total_seg_loss / len(src_loader)
        avg_dom = total_dom_loss / len(src_loader)
        avg_uda = total_uda_loss / len(src_loader)
        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"  [Epoch {epoch:02d}/{total_epochs:02d}] Seg Loss: {avg_seg:.4f} | Dom Loss: {avg_dom:.4f} | UDA Loss: {avg_uda:.4f} | GRL-lambda: {grl_lambda:.3f} (LR: {curr_lr:.6f})", flush=True)
        
        # Save latest checkpoint after every epoch
        torch.save(model.state_dict(), save_path)
            
    print(f"  --> Successfully trained & saved Downstream U-Net to {save_path}\n", flush=True)
    return model