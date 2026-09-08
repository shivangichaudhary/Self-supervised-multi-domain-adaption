import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.generator import UnifiedSatelliteGenerator
from models.discriminator import MultiDomainProjectionDiscriminator
from models.domain_encoder import ContinuousDomainEncoder
from models.losses import MultiDomainLosses
from data.dataset import MultiSensorSatelliteDataset

def train_generator(config, encoder=None, device="cpu"):
    """
    Stage 2: Training Unified Multi-Domain Satellite Generator with Continuous AdaIN modulation.
    """
    print("\n=======================================================")
    print(" [STAGE 2] Training Unified Multi-Domain Generator     ")
    print("=======================================================")
    
    data_cfg = config["data"]
    gen_cfg = config["generator"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Domain Encoder
    if encoder is None:
        encoder = ContinuousDomainEncoder(
            in_channels=gen_cfg["in_channels"],
            latent_dim=gen_cfg["latent_dim"]
        ).to(device)
        enc_path = os.path.join(output_dir, "domain_encoder.pth")
        if os.path.exists(enc_path):
            encoder.load_state_dict(torch.load(enc_path, map_location=device, weights_only=True))
    encoder.eval()
    
    # Build Generator & Discriminator
    generator = UnifiedSatelliteGenerator(
        in_channels=gen_cfg["in_channels"],
        out_channels=gen_cfg["out_channels"],
        latent_dim=gen_cfg["latent_dim"],
        dim=gen_cfg["dim"],
        num_resblocks=gen_cfg["num_resblocks"]
    ).to(device)
    
    discriminator = MultiDomainProjectionDiscriminator(
        in_channels=gen_cfg["in_channels"],
        latent_dim=gen_cfg["latent_dim"],
        dim=gen_cfg["dim"]
    ).to(device)
    
    loss_module = MultiDomainLosses(
        lambda_adv=gen_cfg["lambda_adv"],
        lambda_sty=gen_cfg["lambda_sty"],
        lambda_cyc=gen_cfg["lambda_cyc"],
        lambda_sem=gen_cfg["lambda_sem"],
        lambda_id=gen_cfg["lambda_id"],
        lambda_geo=gen_cfg["lambda_geo"],
        lambda_edge=gen_cfg.get("lambda_edge", 2.0)
    ).to(device)
    
    # Set of angle-domain names — geometric preservation loss is activated for these pairs
    from data.synthetic_generator import SyntheticSatelliteGenerator
    angle_domain_set = set(SyntheticSatelliteGenerator.ANGLE_DEGREES.keys())
    
    opt_g = optim.Adam(generator.parameters(), lr=gen_cfg["lr_g"], betas=(gen_cfg["beta1"], gen_cfg["beta2"]))
    opt_d = optim.Adam(discriminator.parameters(), lr=gen_cfg["lr_d"], betas=(gen_cfg["beta1"], gen_cfg["beta2"]))
    sched_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=gen_cfg["epochs"], eta_min=1e-5)
    sched_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=gen_cfg["epochs"], eta_min=1e-5)
    
    dataset = MultiSensorSatelliteDataset(
        root_dir=data_cfg["data_dir"],
        domains=data_cfg["domains"],
        split="train"
    )
    loader = DataLoader(dataset, batch_size=data_cfg["batch_size"], shuffle=True, drop_last=True)
    
    for epoch in range(1, gen_cfg["epochs"] + 1):
        total_d_loss = 0.0
        total_g_loss = 0.0
        
        for batch in loader:
            x_src = batch["src_img"].to(device)
            x_tgt = batch["tgt_img"].to(device)
            
            with torch.no_grad():
                z_src = encoder(x_src)
                z_tgt = encoder(x_tgt)
                
            # ---------------------
            #  Train Discriminator
            # ---------------------
            opt_d.zero_grad()
            
            # Generate translated fake image (Source -> Target style)
            x_fake = generator(x_src, z_tgt)
            
            d_real = discriminator(x_tgt, z_tgt)
            d_fake = discriminator(x_fake.detach(), z_tgt)
            
            loss_d = loss_module.d_loss(d_real, d_fake)
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            opt_d.step()
            total_d_loss += loss_d.item()
            
            # -----------------
            #  Train Generator
            # -----------------
            opt_g.zero_grad()
            
            # Adversarial loss
            d_fake_for_g = discriminator(x_fake, z_tgt)
            loss_adv = loss_module.g_adv_loss(d_fake_for_g) * gen_cfg["lambda_adv"]
            
            # Style reconstruction loss: Target style must be measurable in x_fake
            z_fake_pred = encoder(x_fake)
            loss_sty = loss_module.style_recon_loss(z_fake_pred, z_tgt) * gen_cfg["lambda_sty"]
            
            # Multi-domain Cycle-consistency loss: x_fake -> z_src -> reconstructed x_src
            x_recon = generator(x_fake, z_src)
            loss_cyc = loss_module.cycle_consistency_loss(x_recon, x_src) * gen_cfg["lambda_cyc"]
            
            # Semantic / Content preservation loss
            content_real = generator.extract_content(x_src).detach()
            content_fake = generator.extract_content(x_fake)
            loss_sem = loss_module.semantic_preservation_loss(content_fake, content_real) * gen_cfg["lambda_sem"]
            
            # Identity loss
            x_ident = generator(x_src, z_src)
            loss_id = loss_module.identity_loss(x_ident, x_src) * gen_cfg["lambda_id"]
            
            # Boundary Edge preservation loss (Sobel gradients)
            loss_edge = loss_module.edge_preservation_loss(x_fake, x_src) * gen_cfg.get("lambda_edge", 2.0)
            
            # Geometric preservation loss (active when either src or tgt is an angle domain)
            src_is_angle = batch["src_domain"][0] in angle_domain_set
            tgt_is_angle = batch["tgt_domain"][0] in angle_domain_set
            if src_is_angle or tgt_is_angle:
                loss_geo = loss_module.geometric_preservation_loss(content_fake, content_real) * gen_cfg["lambda_geo"]
            else:
                loss_geo = torch.tensor(0.0, device=device)
            
            loss_g = loss_adv + loss_sty + loss_cyc + loss_sem + loss_id + loss_edge + loss_geo
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            opt_g.step()
            total_g_loss += loss_g.item()
            
        sched_g.step()
        sched_d.step()
        avg_d = total_d_loss / len(loader)
        avg_g = total_g_loss / len(loader)
        
        print(f"  [Epoch {epoch:02d}/{gen_cfg['epochs']:02d}] D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f} (Adv: {loss_adv.item():.2f}, Cyc: {loss_cyc.item():.2f}, Edge: {loss_edge.item():.2f})", flush=True)
            
    # Save checkpoints
    torch.save(generator.state_dict(), os.path.join(output_dir, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(output_dir, "discriminator.pth"))
    print(f"  --> Saved trained Unified Generator to {os.path.join(output_dir, 'generator.pth')}\n", flush=True)
    return generator
