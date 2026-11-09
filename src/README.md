# TF-MoECLIP: Frequency-Topology Routed MoE-CLIP for Pixel-Level Bridge Defect Detection

## 1. Overview & Final Three-Module Architecture (`M0 -> M1 -> M2 -> M3`)
This repository implements **`TF-MoECLIP`**, a parameter-efficient foundation model adaptation framework for pixel-level bridge defect anomaly detection and cross-domain zero-shot inspection.

| Paper Name | Code Variant Alias | Architectural Composition | Trainable Params |
| :--- | :---: | :--- | :---: |
| **`M0`** | `M0` / `A0` | **`MoECLIP` Baseline *(CVPR 2026)***: Frozen `CLIP ViT-L/14@336px` ($518\times 518$, layers $\{6,12,18,24\}$), $K=4$ point-wise LoRA experts ($r=8$, Top-2 routing), **FOFS**, **Simplex ETF Loss**, $\text{CV}^2$ Balance Loss, isotropic **PAA** ($s\in\{1,3,5\}$), and **Depth-wise Image Adapter**. | `4,091,153` |
| **`M1`** | `M1` | **`+ SFR-Graph` *(Module 1 — Semantic Structure)***: Expands binary prompts to **12 structured semantic anchors** (`1 Normal` + `6 Benign Nuisances` + `4 Specific Bridge Defects` + `1 Generic Defect`), adapts anchors via Representation Steering (`RS`), refines logits via a 2-layer Generalized Relational GCN (`GR`) over physical defect co-occurrence edges (`Spalling <-> Corrosion`, `Crack <-> Efflorescence`), and applies stratified anti-collapse background supervision. *(Does not use wavelet descriptors $c_i$)*. | `4,288,755` |
| **`M2`** | `M2` / `A3` | **`+ WT-HetMoE` *(Module 2 — Physical Structure)***: On top of `M1`'s 12-class semantic supervision, uses lossless 2-level 2D Haar DWT descriptors $c_i = [e_{\text{hf}}, e_{\text{hv}}, e_{\text{diag}}, e_{\text{var}}]$ to jointly drive: (i) **`WT-Router`** (`M1_WT`), (ii) **`Het-LoRA`** bottleneck spatial experts (`M1_HET`), and (iii) companion **`Aniso-PAA`** directional strip pooling (`M1_PAA`). | `4,291,448` |
| **`M3`** | `M3` / `A4` | **`+ CC-DWCPR` *(Module 3 — Ours Full)***: $129\times 129$ **Trunk-Decoupled Dual-Regime Conditional Wavelet Refiner**. Uses stop-gradient trunk isolation (`coarse_prob.detach()`, `fork_rng()`) and 12-anchor semantic posteriors to boost fine cracks in high-confidence defect zones while suppressing low-threshold background halos. | `4,297,908` |

---

## 2. Evaluation Protocol & Metric Definitions
1. **Input & Mask Resolution**: Following standard CLIP ViT-L/14 adaptation protocols, input images from **`Bridge512`** ($512\times 512$, `2,861 train / 400 val / 809 test`) and **`Official S2DS`** ($1024\times 1024$ untouched full images in `official_full` mode, `563 train / 87 val / 93 test`) are resized to $518\times 518$ ($37\times 37$ patch grid), and ground-truth masks are resized to $518\times 518$ via nearest-neighbor interpolation.
2. **Threshold-Free vs. Optimal-F1 Metrics**:
   - `Image-AUROC`, `Image-AP`, `Pixel-AUROC`, `Pixel-AP`, and `Pixel-AUPRO` (integrated over $\text{FPR} \le 0.30$) are threshold-free.
   - `Image-F1` and `Pixel-F1` (`Pixel-IoU`) report the optimal F1 score along the precision-recall curve on the evaluation split (`Best-Th` denotes the corresponding binary threshold $\tau^*$).
   - **Per-Category Recall (`Crack`, `Spalling`, `Corrosion`, `Efflorescence`)**: Measures the pixel recall of the binary anomaly prediction $\mathbb{I}(P_{\text{final}} \ge \tau^*)$ on ground-truth pixels of each defect category (category-conditioned anomaly recall), rather than closed-set 4-way classification accuracy.
   - **`Nuisance-FPR`**: Measures the false positive rate of $\mathbb{I}(P_{\text{final}} \ge \tau^*)$ on explicit benign nuisance masks (`Vegetation` and `Control Point`) in `Official S2DS`.

---

## 3. Quick Start Commands

```bash
# Run the complete 4-stage reordered ablation (M0 -> M1 -> M2 -> M3) from scratch (12 epochs, seed=111)
python train_eval.py \
  --codebrim_root /path/to/Bridge512_balance_CODEBRIM \
  --s2ds_root /path/to/S2DS \
  --s2ds_mode official_full \
  --train_on BOTH \
  --variants M0,M1,M2,M3 \
  --epochs 12 \
  --batch_size 16 \
  --lr 5e-4 \
  --seed 111 \
  --use_cache 1 \
  --output_dir ./outputs_final

# Run Module 2 internal sub-component ablations on Bridge512
python train_eval.py \
  --codebrim_root /path/to/Bridge512_balance_CODEBRIM \
  --s2ds_root /path/to/S2DS \
  --train_on CODEBRIM \
  --variants M1_PAA,M1_HET,M1_WT \
  --epochs 12 \
  --seed 111 \
  --use_cache 1 \
  --output_dir ./outputs_final
```
