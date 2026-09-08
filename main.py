import argparse
import os
import torch
from configs.config import load_config, apply_mode, apply_dataset
from data.synthetic_generator import SyntheticSatelliteGenerator
from data.convert_real_datasets import process_pastis_hd
from train_domain_encoder import train_domain_encoder
from train_generator import train_generator
from train_downstream import train_downstream_model
from evaluate_generalization import evaluate_framework

def main():
    parser = argparse.ArgumentParser(description="Self-Supervised Multi-Domain Adaptation for Satellite Imagery")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config yaml")
    parser.add_argument("--dataset", type=str, default="pastis", choices=["synthetic", "pastis"],
                        help="Dataset to use: 'pastis' (real PASTIS-HD benchmark) or 'synthetic' (synthetic simulation)")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "data", "encoder", "generator", "downstream", "eval"],
                        help="Execution stage to run")
    parser.add_argument("--device", type=str, default="auto", help="Execution device: auto, cpu, cuda")
    parser.add_argument("--mode", type=str, default="sensor", choices=["sensor", "angle", "both"],
                        help="Domain mode: 'sensor' (multi-sensor, default), 'angle' (multi-angle), or 'both' (combined)")
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_dataset(config, args.dataset)
    if args.dataset == "synthetic":
        config = apply_mode(config, args.mode)
    print(f"[*] Dataset: {args.dataset.upper()} | Active Domains: {config['data']['domains']}")
    
    # Device setup
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[*] Running on Device: {device.upper()}")

    data_dir = config["data"]["data_dir"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(config["output_dir"], exist_ok=True)

    # 0. Data Preparation Check
    if args.dataset == "pastis":
        s2_img_dir = os.path.join(data_dir, "sentinel2", "images")
        if args.stage == "data" or not os.path.exists(s2_img_dir) or len(os.listdir(s2_img_dir)) == 0:
            print("[*] Converting downloaded PASTIS-HD real dataset into aligned multi-sensor format...")
            process_pastis_hd("data/PASTIS-HD", data_dir, max_patches=300)
            if args.stage == "data":
                return
    else:
        if args.stage == "data" or not os.path.exists(os.path.join(data_dir, "sentinel2")):
            print("[*] Generating Benchmark Synthetic Satellite Datasets across domains...")
            gen = SyntheticSatelliteGenerator(
                image_size=config["data"]["image_size"],
                seed=config["seed"]
            )
            gen.generate_dataset(
                output_dir=data_dir,
                num_samples_per_domain=config["data"]["num_samples_per_domain"],
                domains=config["data"]["domains"]
            )
            if args.stage == "data":
                return

    # Stage 1: Self-Supervised Continuous Domain Encoder
    encoder = None
    if args.stage in ["all", "encoder"]:
        encoder = train_domain_encoder(config, device=device)

    # Stage 2: Unified Multi-Domain Generator
    generator = None
    if args.stage in ["all", "generator"]:
        generator = train_generator(config, encoder=encoder, device=device)

    # Stage 3: Downstream Land-Cover Task Model
    downstream_model = None
    if args.stage in ["all", "downstream"]:
        src_domain = "sentinel2" if "sentinel2" in config["data"]["domains"] else config["data"]["domains"][0]
        downstream_model = train_downstream_model(config, source_domain=src_domain, device=device)

    # Stage 4: Comprehensive Evaluation & Generalization Benchmarking
    if args.stage in ["all", "eval"]:
        evaluate_framework(config, encoder=encoder, generator=generator, downstream_model=downstream_model, device=device)

if __name__ == "__main__":
    main()
