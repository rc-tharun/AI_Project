# ODD + OOD Detection for Safe Autonomy

Camera-based safety gate for autonomous driving. For every front-camera frame the system decides one of three actions: **TRUST** the perception output, **SLOW DOWN** if the evidence is uncertain, or **ABSTAIN** if the frame looks out-of-distribution (rain, fog, snow, night, etc.).

Course final-project repository covering the full experimental study — six method families compared on BDD100K weather classification — and a runnable demo of the recommended pipeline.

> **Members:** Tharun Reddy Challabotla · HariChandana Srikurmum · Srija

---

## Repository layout

| Path | What it contains |
|---|---|
| `app.py` | Gradio web demo of the safety gate (zero-shot CLIP + trained ResNet-50 fusion). |
| `build_index.py` | Builds the kNN reference set + tunes thresholds from cached CLIP features (used by the original kNN gate variant). |
| `demo_thresholds.json` | Persisted gate config (k, thresholds, validation metrics). |
| `ResNet50/AI_RESNET.ipynb` | ResNet-50 baseline + calibration (MSP, energy, dropout, ensemble entropy, temperature scaling). |
| `Deep_Ensemble/Deep_Ensemble.ipynb` | Deep ensemble experiments. |
| `SVDD/Deep_Ensemble_(2).ipynb` | One-class Deep SVDD baseline. |
| `Vit+knn/Deep_Ensemble_(1).ipynb` | CLIP ViT-B/32 features + FAISS kNN OOD detector — strongest explicit OOD detector in the study. |
| `vit_l_14/Deep_Ensemble_(1).ipynb` | Same approach with CLIP ViT-L/14 backbone. |
| `results of vit/Deep_Ensemble_Mahalanobis.ipynb` | Mahalanobis distance OOD detector on CLIP features. |
| `results of vit/VIT_BACKBONE.ipynb` | Supervised backbone sweep (ResNet-50, EfficientNet-B3, ConvNeXt-Tiny, CLIP ViT-B/16). |
| `results of vit/train_safety_gate.py` + `submit_grace.sh` | Cluster training scripts (Texas A&M HPRC Grace). |
| `outputs/final_project_presentation*/` | Generated narrative and final slide decks. |
| `ODD-OOD-Detection-for-Safe-Autonomy.pptx` | Original proposal deck. |

### Files **not** in the repo (excluded by `.gitignore`)

The trained weights and cached feature tensors are large binaries that live outside Git:

- `ResNet50/best_resnet50.pth` (≈99 MB) — trained safety-gate classifier head.
- `results of vit/best_clip_vitb16.pth` (≈346 MB) — fine-tuned CLIP ViT-B/16 backbone.
- `Vit+knn/train_clip_features.pt`, `val_clip_features.pt` (≈90 MB) — cached CLIP ViT-B/32 features.
- `vit_l_14/train_vit_l_14_features.pt`, `val_vit_l_14_features.pt` (≈140 MB) — cached CLIP ViT-L/14 features.
- `SVDD/train_features.pt` (≈500 MB) — cached ResNet features for SVDD.
- `id_bank.npy` — derived ID-only feature bank for the kNN demo (built by `build_index.py`).

To regenerate them, open the relevant notebook and run the feature-extraction cells. They'll repopulate the `*.pt` files in place; the demo will pick them up automatically.

---

## Final recommended pipeline

After comparing six method families, the recommendation is a **hybrid safety gate**:

1. **Classifier** — ConvNeXt-Tiny (best validation accuracy 0.9368 in the backbone sweep). Used as the in-domain perception head. The trained weights for ResNet-50 are also provided as a strong baseline.
2. **Explicit OOD detector** — CLIP ViT-B/32 + FAISS kNN against a clear-weather reference bank (validation AUROC 0.7412).
3. **Fused output** — temperature-scaled classifier confidence + kNN distance → TRUST / SLOW DOWN / ABSTAIN.

| Method family | Best metric reported | Take-away |
|---|---|---|
| ResNet-50 + MSP/energy/dropout | OOD AUROC 0.5263 | Calibration helps (ECE 0.104 → 0.016 with temperature scaling), but confidence-only OOD separation is weak. |
| Deep SVDD | (low) | Weakest of the explicit OOD detectors tried. |
| Mahalanobis on CLIP | mid | Outperforms SVDD; below CLIP+kNN. |
| **CLIP ViT-B/32 + kNN** | **AUROC 0.7412** | **Strongest explicit OOD detector.** |
| CLIP ViT-L/14 + kNN | AUROC ≈ 0.74 | Comparable to ViT-B/32, slightly better coverage. |
| Backbone sweep | ConvNeXt-Tiny 0.9368 acc | Best classifier; pair with CLIP+kNN for the gate. |

---

## Live demo (`app.py`)

`app.py` is a Gradio web app that runs the safety gate end-to-end. To make it work on **any** uploaded image (not just BDD100K-style dashcam frames), the demo uses a **bank-free** variant of the pipeline:

1. **CLIP ViT-B/32 zero-shot** — cosine similarity between the image and natural-language prompts (`"a photo of a road on a clear sunny day"`, `"a foggy road with poor visibility"`, …) softmaxed into a single P(clear).
2. **Trained ResNet-50** classifier — direct clear-vs-adverse prediction.
3. **Fusion rule:**
   - Both P(clear) ≥ 0.65 → **TRUST**
   - Both P(clear) ≤ 0.35 → **ABSTAIN**
   - Otherwise → **SLOW DOWN** (disagreement / uncertainty)

A single chart shows where each method places the image on the P(clear) axis, with the trust / slow / abstain bands shaded.

### Run it

```bash
# Use Python 3.11 (torch + faiss + open_clip wheels are stable here)
python3.11 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install torch torchvision open_clip_torch faiss-cpu gradio scikit-learn matplotlib pillow numpy

# You also need the trained ResNet-50 head at ResNet50/best_resnet50.pth.
# (Excluded from the repo — copy it in from your training run.)

.venv/bin/python app.py
# -> http://127.0.0.1:7860
```

The demo loads CLIP ViT-B/32 (downloaded once from HuggingFace) and the local `best_resnet50.pth` head. No feature bank is required for the zero-shot variant.

---

## Reproducing the experiments

Each track is self-contained in its notebook directory. Open the corresponding `.ipynb`, run all cells, and the cached `.pt` feature tensors and `.pth` weights will be regenerated locally. The `kagglehub` cell at the top of each notebook fetches the BDD100K weather classification dataset on first run (browser auth on first call, cached afterwards).

```bash
# Example: reproduce the CLIP+kNN OOD detector
jupyter notebook 'Vit+knn/Deep_Ensemble_(1).ipynb'
```

---

## Honesty guardrails

The notebooks use three different evaluation splits — a curated binary split (ResNet baseline, backbone sweep), a clear-only ID bank split (CLIP+kNN, Mahalanobis), and a full-weather binary split (Deep Ensemble, SVDD). Cross-track comparison is directional only; final deployment claims should be revalidated on a single shared evaluation protocol with latency and intervention-quality measurements.
