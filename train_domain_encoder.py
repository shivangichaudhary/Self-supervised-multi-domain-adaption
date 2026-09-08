import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.domain_encoder import ContinuousDomainEncoder, ContrastiveDomainLoss
from data.dataset import MultiSensorSatelliteDataset
from data.transforms import SatelliteTransforms

def train_domain_encoder(config, device="cpu"):
    """
    Stage 1: Self-Supervised Pretraining of the Continuous Domain Encoder.
    Learns continuous style embeddings directly from unlabeled satellite image patches.
    """
    print("\n=======================================================")
    print(" [STAGE 1] Pretraining Self-Supervised Domain Encoder  ")
    print("=======================================================")
    
    data_cfg = config["data"]
    enc_cfg = config["domain_encoder"]
    
    # Transforms
    t1 = SatelliteTransforms(is_train=True)
    t2 = SatelliteTransforms(is_train=True)
    
    dataset = MultiSensorSatelliteDataset(
        root_dir=data_cfg["data_dir"],
        domains=data_cfg["domains"],
        split="train"
    )
    
    loader = DataLoader(dataset, batch_size=data_cfg["batch_size"], shuffle=True, drop_last=True)
    
    encoder = ContinuousDomainEncoder(
        in_channels=enc_cfg["in_channels"],
        latent_dim=enc_cfg["latent_dim"]
    ).to(device)
    
    criterion = ContrastiveDomainLoss(temperature=enc_cfg["temperature"]).to(device)
    optimizer = optim.AdamW(encoder.parameters(), lr=enc_cfg["lr"], weight_decay=enc_cfg["weight_decay"])
    
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    encoder.train()
    for epoch in range(1, enc_cfg["epochs"] + 1):
        total_loss = 0.0
        for batch in loader:
            imgs = batch["src_img"].to(device)
            
            # Apply dual views for contrastive self-supervision
            view1 = t1(imgs.clone())
            view2 = t2(imgs.clone())
            
            z1 = encoder(view1)
            z2 = encoder(view2)
            
            loss = criterion(z1, z2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        if epoch % 2 == 0 or epoch == 1 or epoch == enc_cfg["epochs"]:
            print(f"  [Epoch {epoch:02d}/{enc_cfg['epochs']:02d}] Contrastive InfoNCE Loss: {avg_loss:.4f}", flush=True)
            
    save_path = os.path.join(output_dir, "domain_encoder.pth")
    torch.save(encoder.state_dict(), save_path)
    print(f"  --> Saved trained Domain Encoder to {save_path}\n", flush=True)
    return encoder
