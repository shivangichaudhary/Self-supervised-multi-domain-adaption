
# Self-Supervised Multi-Domain Adaptation for Satellite Imagery

### A Continuous Domain-Embedding Approach

> Final Year Project — M.Tech Computer Science (AI/ML)

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Key Idea](#key-idea)
- [Related Work](#related-work)
- [Methodology](#methodology)
  - [Phase 1 — Pairwise Baseline](#phase-1--pairwise-baseline-months-14)
  - [Phase 2 — Multi-Domain Extension](#phase-2--multi-domain-extension-months-49)
- [Datasets](#datasets)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Timeline](#timeline)
- [Expected Outcomes](#expected-outcomes)
- [Target Venues](#target-venues)
- [References](#references)
- [Author](#author)
- [License](#license)

---

## Overview

Satellite imagery models degrade significantly when applied across different sensors, seasons, or regions — a problem known as **domain shift**. Most domain adaptation methods require either labeled target-domain data or a clean, pre-defined notion of what a "domain" is. Neither is easy to obtain in real multi-sensor satellite archives, where a single mosaic can blend several sensors and acquisition dates.

This project extends **SS(DA)²** (Zhang et al., 2023) — a self-supervised, domain-agnostic adaptation framework limited to translating between exactly two domains at a time — into a **self-supervised, continuous multi-domain setting**. A domain encoder learns a continuous embedding of sensor/seasonal characteristics without labels, and a single conditional generator translates content into any target domain's style. Success is measured by whether the model generalizes to sensor-pair combinations it never saw jointly during training, using downstream classification/segmentation accuracy rather than visual realism alone.

The project runs in two phases: a pairwise baseline (guaranteed, complete deliverable) followed by the multi-domain extension (stretch contribution).

## Motivation

Models trained on satellite imagery from one sensor, season, or region routinely underperform on another, even for identical land-cover categories. Fixing this normally requires:

- Collecting labeled data from the target domain — expensive and often impractical at scale, and
- Defining a clean domain boundary, which frequently doesn't exist in large-scale, multi-temporal, multi-sensor mosaics.

A method that adapts across domains without labels and without a fixed domain definition would meaningfully lower the cost of deploying satellite-imagery models across new sensors and geographies.

## Key Idea

No existing method combines all three of the following for satellite imagery:

1. **Label-free, self-supervised domain representation** — no sensor/region metadata required.
2. **Generalization to domain-pair combinations not seen together during training.**
3. **Evaluation via downstream task performance** (classification/segmentation accuracy), not visual-quality metrics alone.

This project's contribution is combining all three: extending SS(DA)²'s label-free philosophy from pairwise to a continuous, many-domain embedding space (in the style of StarGAN v2), and evaluating on downstream accuracy rather than image realism.

## Related Work

| Source | Key Idea | Gap Relative to This Project |
|---|---|---|
| Zhang, Shi & Zhu (2023) — arXiv:2309.11109 | SS(DA)²: self-supervised, domain-agnostic adaptation via a contrastive GAN, no predefined domain labels. | Limited to pairwise translation; doesn't scale to N domains or unseen sensor pairs. |
| Choi et al. (2018) — arXiv:1711.09020 | StarGAN: one generator translates across multiple domains using a discrete domain label. | Needs labeled domain categories; not self-supervised; not applied to remote sensing. |
| Choi et al. (2020) — StarGAN v2 | Continuous style codes (via AdaIN) replace discrete labels for diverse multi-domain synthesis. | Still trained on labeled domain groups; evaluated on visual diversity, not downstream accuracy; not remote-sensing specific. |
| Tasar et al. (2020) — arXiv:2004.06402 | StandardGAN: multi-source domain adaptation for VHR satellite segmentation via data standardization. | Segmentation-specific; doesn't test unseen domain combinations; not label-free. |
| Vinholi et al. (2024) — arXiv:2404.11243 | Diffusion-based multi-sensor optical translation preserving radiometric content. | Struggles with radiometric consistency across large mosaics; diffusion inference is compute-heavy. |
| Kieu et al. (2025) — arXiv:2510.03252 | Universal MDT via Diffusion Routers: generalizes to unseen domain pairs via a central domain. | Needs K−1 paired datasets with a defined central domain; general-purpose, not satellite-specific; diffusion-based. |

## Methodology

### Phase 1 — Pairwise Baseline (Months 1–4)

- Curate source and target domain datasets (e.g., EuroSAT as source; NAIP or Landsat-8 as target).
- Train a ResNet classifier on the source domain only; quantify the accuracy drop on the target domain.
- Implement a pairwise CycleGAN / SS(DA)²-style translation module between source and target.
- Retrain the classifier on real source + translated images; re-evaluate on the real target domain.

### Phase 2 — Multi-Domain Extension (Months 4–9)

- Extend the pipeline to three or more domains (distinct sensors and/or seasons).
- Train the domain encoder via a contrastive self-supervised pretext task, using spatial/temporal proximity of patches as a weak positive-pair signal.
- Build a single generator with a content encoder + AdaIN-style injection, conditioned on the learned domain embedding (StarGAN v2-style, adapted to be label-free).
- Train with adversarial, cycle-consistency/content-preservation, style-reconstruction, and contrastive domain losses.
- Evaluate with a **leave-one-pair-out** protocol: hold out one sensor-pair combination entirely and test translation quality and downstream accuracy on it.

## Datasets

| Dataset | Sensor(s) | Role |
|---|---|---|
| PASTIS-HD | Sentinel-1 (SAR), Sentinel-2 (optical), SPOT 6-7 (VHR optical) | Primary training + evaluation dataset (pixel-level crop-type labels, 18 classes) — used for the generator and downstream classification/segmentation accuracy |
| MSC-France | Sentinel-2, Landsat-8, SPOT-6 | Self-supervised pretraining only (unlabeled) — trains the domain encoder before PASTIS-HD fine-tuning |
| MultiEarth 2022 | Sentinel-1, Sentinel-2, Landsat-5, Landsat-8 | Generalization test dataset — leave-one-pair-out evaluation on unseen sensor combinations |
| SpaceNet 6 | SAR (Capella), high-res optical (WorldView-2) | Held-out zero-shot test only — never used in training; tests generalization to an unseen sensor and unseen geography |

## Repository Structure

```
.
├── data/                   # Raw and processed dataset splits (not committed — see .gitignore)
├── src/
│   ├── datasets/            # Dataset loaders and preprocessing
│   ├── models/               # Domain encoder, generator, classifier/segmentation heads
│   ├── losses/               # Adversarial, cycle-consistency, contrastive, style-reconstruction losses
│   ├── training/             # Training loops for Phase 1 and Phase 2
│   └── evaluation/           # Leave-one-pair-out evaluation, downstream accuracy metrics
├── configs/                  # YAML configs per experiment/phase
├── notebooks/                # Exploratory analysis and result visualization
├── results/                  # Logs, checkpoints, evaluation outputs
├── requirements.txt
└── README.md
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Training and evaluation scripts (`src/training/`, `src/evaluation/`) will be added as each phase is implemented — see the [Timeline](#timeline) below for the current stage.

## Timeline

| Timeline | Activities | Deliverable |
|---|---|---|
| Months 1–2 | Literature survey; finalize datasets; build data pipeline for source + one target domain. | Survey note; curated dataset |
| Months 2–4 | Train baseline classifier on source only; quantify accuracy drop. Implement pairwise CycleGAN/SS(DA)²-style translation; retrain and re-evaluate. | Phase 1 baseline (safety-net deliverable) |
| Months 4–5 | Design self-supervised domain encoder (contrastive pretext task). | Working domain encoder |
| Months 5–7 | Build single conditional generator with AdaIN-style embedding conditioning; extend to 3+ domains; train with combined losses. | Multi-domain translation model |
| Months 7–8 | Leave-one-pair-out evaluation of generalization to unseen domain combinations. | Generalization results |
| Months 8–9 | Downstream classification/segmentation evaluation; ablations; comparison vs. Phase 1 baseline; write-up. | Final report + manuscript draft |

## Expected Outcomes

- A complete, working pairwise domain-adaptation baseline (Phase 1).
- A self-supervised multi-domain translation framework that generalizes to unseen sensor-pair combinations (Phase 2).
- A comparative evaluation of downstream accuracy: baseline vs. multi-domain approach.
- A manuscript draft suitable for a Scopus-indexed venue.

## Target Venues

IGARSS, IEEE Geoscience and Remote Sensing Letters (GRSL), and Earth-observation-focused workshops at major CV venues (e.g., CVPR's CVDP workshop).

## References

1. Zhang, F., Shi, Y., & Zhu, X. X. (2023). *Self-supervised Domain-agnostic Domain Adaptation for Satellite Images.* arXiv:2309.11109.
2. Choi, Y., Choi, M., Kim, M., Ha, J.-W., Kim, S., & Choo, J. (2018). *StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation.* CVPR 2018 / arXiv:1711.09020.
3. Choi, Y., Uh, Y., Yoo, J., & Ha, J.-W. (2020). *StarGAN v2: Diverse Image Synthesis for Multiple Domains.* CVPR 2020.
4. Tasar, O., et al. (2020). *StandardGAN: Multi-source Domain Adaptation for Semantic Segmentation of Very High Resolution Satellite Images by Data Standardization.* arXiv:2004.06402.
5. Vinholi, J. G., Chini, M., Amziane, A., Machado, R., Silva, D., & Matgen, P. (2024). *Multi-Sensor Diffusion-Driven Optical Image Translation for Large-Scale Applications.* arXiv:2404.11243.
6. Kieu, D., Do, K., Hoang, T., Le, T. M., Kieu, T., Nguyen, D., & Nguyen, T. (2025). *Universal Multi-Domain Translation via Diffusion Routers.* arXiv:2510.03252.

## Author

**[Your Name]**
M.Tech Computer Science (AI/ML) · [Institution Name]
Guide: [Guide Name]

## License

_Add your preferred license here (e.g., MIT, Apache 2.0) — see [choosealicense.com](https://choosealicense.com/) for guidance._
