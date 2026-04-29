"""ODD + OOD Safety Gate Demo - Gradio front-end (zero-shot variant).

This pipeline is **bank-free**, so it works on any driving image regardless of camera/source:
  1. CLIP ViT-B/32 zero-shot weather classifier:
        cosine-similarity between the image embedding and a set of natural-language
        prompts ("a clear sunny day driving scene", "a rainy/foggy/snowy/nighttime
        road scene", ...). Sums to a single P(clear) probability.
  2. Trained ResNet-50 classifier (clear vs adverse) for a second independent vote.
  3. Fuse the two P(clear) probabilities into a 3-way safety decision:
        TRUST     - both methods are confidently "clear"
        SLOW DOWN - the two methods disagree, or both are uncertain
        ABSTAIN   - both methods are confidently "adverse"
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

import open_clip

ROOT = Path(__file__).parent
DEVICE = torch.device('mps' if torch.backends.mps.is_available()
                      else 'cuda' if torch.cuda.is_available()
                      else 'cpu')

# ----------------------------------------------------------------------------
# Decision thresholds for the fused-probability gate (no feature bank).
# ----------------------------------------------------------------------------
TRUST_PCLEAR = 0.65    # both methods >= this -> TRUST
ABSTAIN_PCLEAR = 0.35  # both methods <= this -> ABSTAIN
print(f'[gate] zero-shot fusion: TRUST P(clear)>={TRUST_PCLEAR}, ABSTAIN P(clear)<={ABSTAIN_PCLEAR}')


# ----------------------------------------------------------------------------
# CLIP ViT-B/32 zero-shot weather classifier via natural-language prompts.
# ----------------------------------------------------------------------------
print('[load] CLIP ViT-B/32 (OpenAI weights)...')
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
clip_model = clip_model.to(DEVICE).eval()
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

CLEAR_PROMPTS = [
    'a photo of a road on a clear sunny day',
    'a daytime driving scene with clear weather',
    'a highway under blue sky',
    'a clear urban street in daytime',
]
ADVERSE_PROMPTS = [
    'a photo of a road in heavy rain',
    'a foggy road with poor visibility',
    'a snowy winter road',
    'a road at night with low light',
    'a stormy driving scene',
    'a road with mist and reduced visibility',
]
ALL_PROMPTS = CLEAR_PROMPTS + ADVERSE_PROMPTS
N_CLEAR = len(CLEAR_PROMPTS)


@torch.inference_mode()
def _encode_text_prompts() -> torch.Tensor:
    tokens = clip_tokenizer(ALL_PROMPTS).to(DEVICE)
    feats = clip_model.encode_text(tokens).float()
    feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return feats.cpu()  # (P, 512)


PROMPT_FEATS = _encode_text_prompts()
print(f'[clip] encoded {len(CLEAR_PROMPTS)} clear + {len(ADVERSE_PROMPTS)} adverse prompts')


# ----------------------------------------------------------------------------
# ResNet-50 safety-gate classifier (trained head over a frozen ImageNet backbone).
# Architecture mirrors ResNet50/AI_RESNET.ipynb -> SafetyGateModel.
# ----------------------------------------------------------------------------
class SafetyGateResNet(nn.Module):
    """Architecture inferred from checkpoint keys (frozen_layers / finetune_layers / head with BN)."""

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3):
        super().__init__()
        backbone = models.resnet50(weights=None)
        kids = list(backbone.children())
        # conv1, bn1, relu, maxpool, layer1, layer2  -> frozen
        self.frozen_layers = nn.Sequential(*kids[0:6])
        # layer3, layer4  -> fine-tuned
        self.finetune_layers = nn.Sequential(*kids[6:8])
        self.avgpool = kids[8]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 128),  nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.frozen_layers(x)
        x = self.finetune_layers(x)
        x = self.avgpool(x)
        return self.head(x)


print('[load] ResNet-50 safety-gate weights...')
resnet = SafetyGateResNet(num_classes=2)
state = torch.load(ROOT / 'ResNet50' / 'best_resnet50.pth', map_location='cpu', weights_only=False)
if isinstance(state, dict) and 'state_dict' in state:
    state = state['state_dict']
missing, unexpected = resnet.load_state_dict(state, strict=False)
print(f'[load] resnet missing={len(missing)} unexpected={len(unexpected)}')
if missing or unexpected:
    print(f'[load] missing sample: {missing[:5]}')
    print(f'[load] unexpected sample: {unexpected[:5]}')
resnet = resnet.to(DEVICE).eval()

resnet_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ----------------------------------------------------------------------------
# Inference helpers.
# ----------------------------------------------------------------------------
@torch.inference_mode()
def clip_zero_shot(pil_img: Image.Image) -> tuple[float, dict]:
    """Return (P(clear), per-prompt similarities). Bank-free."""
    x = clip_preprocess(pil_img).unsqueeze(0).to(DEVICE)
    f = clip_model.encode_image(x).float()
    f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    f = f.cpu().squeeze(0)
    sims = (PROMPT_FEATS @ f).numpy()  # (P,)
    # Softmax with CLIP's standard temperature of 100
    logits = sims * 100.0
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    p_clear = float(probs[:N_CLEAR].sum())
    per_prompt = {ALL_PROMPTS[i]: float(probs[i]) for i in range(len(ALL_PROMPTS))}
    return p_clear, per_prompt


@torch.inference_mode()
def classify(pil_img: Image.Image) -> tuple[str, float, dict, float]:
    x = resnet_preprocess(pil_img).unsqueeze(0).to(DEVICE)
    logits = resnet(x).cpu().squeeze(0)
    probs = torch.softmax(logits, dim=0).numpy()
    label = 'clear (in-distribution)' if probs[0] >= probs[1] else 'adverse (rain/fog/snow)'
    return label, float(max(probs)), {
        'clear (in-ODD)': float(probs[0]),
        'adverse (OOD)':  float(probs[1]),
    }, float(probs[0])  # P(clear) from ResNet


def safety_decision(p_clear_clip: float, p_clear_resnet: float) -> tuple[str, str, str]:
    """Three-way fused gate using two independent P(clear) probabilities.

    - TRUST     : both methods agree the scene is clear (>= TRUST_PCLEAR).
    - ABSTAIN   : both methods agree the scene is adverse (<= ABSTAIN_PCLEAR).
    - SLOW DOWN : disagreement, or both in the uncertain middle band.
    """
    avg = 0.5 * (p_clear_clip + p_clear_resnet)
    disagree = abs(p_clear_clip - p_clear_resnet) > 0.30

    if min(p_clear_clip, p_clear_resnet) >= TRUST_PCLEAR:
        return ('TRUST', '#27ae60',
                f'CLIP zero-shot P(clear)={p_clear_clip:.2f} and ResNet-50 P(clear)={p_clear_resnet:.2f} '
                f'both agree the scene is clear. Safe to act on the perception output.')
    if max(p_clear_clip, p_clear_resnet) <= ABSTAIN_PCLEAR:
        return ('ABSTAIN', '#c0392b',
                f'CLIP zero-shot P(clear)={p_clear_clip:.2f} and ResNet-50 P(clear)={p_clear_resnet:.2f} '
                f'both indicate adverse conditions (rain / fog / snow / night). Hand control back to the driver.')
    if disagree:
        return ('SLOW DOWN', '#e67e22',
                f'The two methods disagree (CLIP P(clear)={p_clear_clip:.2f}, ResNet-50 P(clear)={p_clear_resnet:.2f}). '
                f'Reduce speed and increase margin until they agree.')
    return ('SLOW DOWN', '#e67e22',
            f'Both methods are uncertain (CLIP P(clear)={p_clear_clip:.2f}, ResNet-50 P(clear)={p_clear_resnet:.2f}, '
            f'avg={avg:.2f}). Apply a soft caution until the scene resolves.')


def predict(pil_img):
    if pil_img is None:
        return ('<i>Awaiting input.</i>', {}, {}, '<i>Drop in a driving frame to evaluate.</i>', None)

    if not isinstance(pil_img, Image.Image):
        pil_img = Image.fromarray(pil_img)
    pil_img = pil_img.convert('RGB')

    p_clear_clip, per_prompt = clip_zero_shot(pil_img)
    cls_label, cls_conf, cls_probs, p_clear_resnet = classify(pil_img)
    decision, color, rationale = safety_decision(p_clear_clip, p_clear_resnet)

    decision_html = (
        f'<div style="padding:18px;border-radius:10px;background:{color};color:white;'
        f'font-family:Lato,Arial,sans-serif;text-align:center;">'
        f'<div style="font-size:13px;letter-spacing:2px;opacity:.85;">SAFETY GATE</div>'
        f'<div style="font-size:34px;font-weight:700;margin-top:4px;">{decision}</div>'
        f'</div>'
    )

    # Top-3 most likely prompts
    top3 = sorted(per_prompt.items(), key=lambda kv: -kv[1])[:3]
    top3_md = '\n'.join(f'- _"{p}"_  →  **{v*100:.1f}%**' for p, v in top3)

    metrics_md = (
        f"**Method 1 — CLIP zero-shot prompts (no feature bank):** P(clear) = `{p_clear_clip:.3f}`  \n"
        f"**Method 2 — Trained ResNet-50 classifier:** P(clear) = `{p_clear_resnet:.3f}`  ({cls_label})\n\n"
        f"**Top matching CLIP prompts:**\n{top3_md}\n\n"
        f"**Trust band:** P(clear) ≥ `{TRUST_PCLEAR}`  ·  **Abstain band:** P(clear) ≤ `{ABSTAIN_PCLEAR}`\n\n"
        f"---\n_{rationale}_"
    )

    clip_label = {
        'clear (CLIP zero-shot)': p_clear_clip,
        'adverse (CLIP zero-shot)': 1 - p_clear_clip,
    }
    return decision_html, clip_label, cls_probs, metrics_md, _make_pclear_strip(p_clear_clip, p_clear_resnet)


def _make_pclear_strip(p_clear_clip: float, p_clear_resnet: float):
    """Render a horizontal P(clear) axis with both methods' positions and the gate bands."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 1.6))
    ax.axvspan(0.0, ABSTAIN_PCLEAR, color='#c0392b', alpha=0.28, label='ABSTAIN band (both)')
    ax.axvspan(ABSTAIN_PCLEAR, TRUST_PCLEAR, color='#e67e22', alpha=0.28, label='SLOW DOWN band')
    ax.axvspan(TRUST_PCLEAR, 1.0, color='#27ae60', alpha=0.28, label='TRUST band (both)')

    ax.scatter([p_clear_clip], [0.65], s=180, color='#1f4e79', zorder=5, label='CLIP zero-shot')
    ax.scatter([p_clear_resnet], [0.35], s=180, color='#7a1f15', marker='s', zorder=5, label='ResNet-50')
    ax.text(p_clear_clip, 0.85, f'{p_clear_clip:.2f}', ha='center', fontsize=9, fontweight='bold', color='#1f4e79')
    ax.text(p_clear_resnet, 0.15, f'{p_clear_resnet:.2f}', ha='center', fontsize=9, fontweight='bold', color='#7a1f15')

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel('P(clear)  —  higher = more confident the scene is clear-weather')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.45), ncol=5, fontsize=7, framealpha=0.9)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ----------------------------------------------------------------------------
# UI.
# ----------------------------------------------------------------------------
HEADER = f"""
# ODD + OOD Safety Gate — live demo (zero-shot variant)
**Camera-based safety gate for autonomous driving.** Decides per frame whether to **TRUST**, **SLOW DOWN**, or **ABSTAIN** by combining **two independent classifiers** that don't depend on any feature bank:

1. **CLIP ViT-B/32 zero-shot** — cosine similarity between the image and natural-language prompts like _"a photo of a road on a clear sunny day"_ vs _"a foggy road with poor visibility"_. Robust to camera/dataset shift because CLIP was pretrained on web-scale image–text pairs.
2. **ResNet-50** classifier (your trained `best_resnet50.pth`, fine-tuned on BDD100K clear-vs-adverse).

**Fusion rule:** TRUST if both methods agree P(clear) ≥ {TRUST_PCLEAR}; ABSTAIN if both agree P(clear) ≤ {ABSTAIN_PCLEAR}; SLOW DOWN otherwise (disagreement or uncertainty).
"""

with gr.Blocks(title='Safety Gate Demo') as demo:
    gr.Markdown(HEADER)

    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(type='pil', label='Front-camera frame', height=360)
            run_btn = gr.Button('Evaluate frame', variant='primary')
            gr.Markdown(
                'Drop in any driving image (highway, city, dashcam, etc.). '
                'Adverse-weather frames should push the OOD distance up and trigger SLOW DOWN or ABSTAIN.'
            )
        with gr.Column(scale=1):
            decision_out = gr.HTML('<i>Awaiting input.</i>')
            score_strip = gr.Image(label='Where each method places this frame on the P(clear) axis', height=200)
            with gr.Row():
                clip_probs_out = gr.Label(label='CLIP zero-shot P(clear)')
                cls_probs_out = gr.Label(label='ResNet-50 P(clear)')
            metrics_out = gr.Markdown()

    run_btn.click(predict,
                  inputs=img_in,
                  outputs=[decision_out, clip_probs_out, cls_probs_out, metrics_out, score_strip])
    img_in.change(predict,
                  inputs=img_in,
                  outputs=[decision_out, clip_probs_out, cls_probs_out, metrics_out, score_strip])


if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1', server_port=7860, inbrowser=True,
                show_error=True, theme=gr.themes.Soft())
