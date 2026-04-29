"""Run a small batch of test images through the safety gate and save composite
demo-output PNGs into demo_outputs/. These are embedded in the README to show
what the running app produces.

We synthesize a few simple "driving scenes" so this script can run with no
external image assets. The decisions are real (the actual app.py pipeline);
only the inputs are stand-ins.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Headless matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse the live pipeline from app.py without spinning up Gradio.
import gradio
gradio.Blocks.launch = lambda *a, **kw: None  # noqa
import app as gate  # noqa: E402

OUT = Path('demo_outputs')
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic driving-scene generators (640x360, road horizon at ~y=180).
# ---------------------------------------------------------------------------
def _base(width=640, height=360):
    return Image.new('RGB', (width, height))


def _draw_road(img: Image.Image, asphalt=(60, 60, 65)):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    # Trapezoidal road
    draw.polygon([(w * 0.10, h), (w * 0.90, h), (w * 0.55, h * 0.55), (w * 0.45, h * 0.55)],
                 fill=asphalt)
    # Lane stripes
    for i in range(5):
        y = h - i * (h * 0.07) - h * 0.04
        x_off = (h - y) / h * (w * 0.04) + 2
        draw.line([(w * 0.50 - x_off, y), (w * 0.50 + x_off, y - 6)],
                  fill=(220, 215, 180), width=4)


def scene_clear() -> Image.Image:
    img = _base()
    # Blue sky gradient
    arr = np.zeros((360, 640, 3), dtype=np.uint8)
    for y in range(180):
        t = y / 180
        arr[y] = (int(95 + 80 * t), int(155 + 70 * t), int(220 + 30 * t))
    # Greenish horizon
    arr[180:240] = (110, 145, 90)
    # Asphalt placeholder; will be replaced by polygon
    arr[240:] = (140, 145, 130)
    img = Image.fromarray(arr)
    _draw_road(img, asphalt=(70, 70, 75))
    # Sun glare
    glare = Image.new('RGBA', img.size, (0, 0, 0, 0))
    g = ImageDraw.Draw(glare)
    g.ellipse((480, 30, 580, 120), fill=(255, 245, 180, 130))
    img = Image.alpha_composite(img.convert('RGBA'), glare).convert('RGB')
    return img.filter(ImageFilter.GaussianBlur(radius=0.6))


def scene_fog() -> Image.Image:
    img = scene_clear()
    fog = Image.new('RGB', img.size, (210, 215, 218))
    return Image.blend(img, fog, alpha=0.72).filter(ImageFilter.GaussianBlur(radius=2.0))


def scene_snow() -> Image.Image:
    arr = np.full((360, 640, 3), 230, dtype=np.uint8)
    arr[:130] = (200, 205, 215)
    arr[130:200] = (215, 220, 225)
    img = Image.fromarray(arr)
    _draw_road(img, asphalt=(170, 170, 175))
    # Speckle "snowflakes"
    rng = np.random.default_rng(42)
    a = np.array(img)
    mask = rng.random(a.shape[:2]) > 0.985
    a[mask] = (250, 252, 255)
    return Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius=0.8))


def scene_night() -> Image.Image:
    arr = np.zeros((360, 640, 3), dtype=np.uint8)
    arr[:180] = (10, 10, 25)
    arr[180:] = (20, 20, 25)
    img = Image.fromarray(arr)
    _draw_road(img, asphalt=(15, 15, 20))
    # Headlight cones
    over = Image.new('RGBA', img.size, (0, 0, 0, 0))
    g = ImageDraw.Draw(over)
    g.polygon([(280, 360), (360, 360), (340, 220), (300, 220)], fill=(255, 240, 180, 90))
    g.ellipse((310, 210, 330, 230), fill=(255, 255, 220, 200))
    img = Image.alpha_composite(img.convert('RGBA'), over).convert('RGB')
    return img.filter(ImageFilter.GaussianBlur(radius=1.2))


def scene_rain() -> Image.Image:
    img = scene_clear()
    rain_layer = Image.new('RGB', img.size, (90, 95, 110))
    img = Image.blend(img, rain_layer, alpha=0.55)
    rng = np.random.default_rng(7)
    a = np.array(img)
    # Streaks
    for _ in range(800):
        x = rng.integers(0, 640)
        y = rng.integers(0, 320)
        for k in range(rng.integers(6, 14)):
            if 0 <= y + k < 360 and 0 <= x - k // 2 < 640:
                a[y + k, x - k // 2] = (200, 210, 220)
    return Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius=1.0))


SCENES = [
    ('clear',  'Clear sunny day',   scene_clear),
    ('rain',   'Heavy rain',         scene_rain),
    ('fog',    'Foggy / low vis',    scene_fog),
    ('snow',   'Snowy road',         scene_snow),
    ('night',  'Nighttime',          scene_night),
]


# ---------------------------------------------------------------------------
# Composite renderer: input thumbnail + decision banner + P(clear) strip.
# ---------------------------------------------------------------------------
def render_panel(name: str, label: str, scene: Image.Image, decision: str,
                 color: str, p_clip: float, p_resnet: float, rationale: str) -> Image.Image:
    fig = plt.figure(figsize=(13, 4.4), facecolor='white')
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.4], height_ratios=[1.0, 0.9],
                          hspace=0.35, wspace=0.25)

    # Input thumbnail (top-left)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(scene)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    ax_img.set_title(f'Synthetic test input — {label}', fontsize=11, fontweight='bold')
    for s in ax_img.spines.values():
        s.set_color('#bbb')

    # Decision banner (top-right)
    ax_dec = fig.add_subplot(gs[0, 1])
    ax_dec.set_xticks([]); ax_dec.set_yticks([])
    ax_dec.set_facecolor(color)
    for s in ax_dec.spines.values():
        s.set_visible(False)
    ax_dec.text(0.5, 0.72, 'SAFETY GATE', ha='center', va='center',
                fontsize=10, color='white', fontweight='bold', alpha=0.9,
                family='sans-serif')
    ax_dec.text(0.5, 0.32, decision, ha='center', va='center',
                fontsize=24, color='white', fontweight='bold',
                family='sans-serif')

    # P(clear) strip (bottom-right)
    ax_p = fig.add_subplot(gs[1, 1])
    ax_p.axvspan(0.0, gate.ABSTAIN_PCLEAR, color='#c0392b', alpha=0.28)
    ax_p.axvspan(gate.ABSTAIN_PCLEAR, gate.TRUST_PCLEAR, color='#e67e22', alpha=0.28)
    ax_p.axvspan(gate.TRUST_PCLEAR, 1.0, color='#27ae60', alpha=0.28)
    ax_p.scatter([p_clip], [0.65], s=130, color='#1f4e79', zorder=5, label='CLIP zero-shot')
    ax_p.scatter([p_resnet], [0.35], s=130, color='#7a1f15', marker='s', zorder=5, label='ResNet-50')
    ax_p.text(p_clip, 0.85, f'{p_clip:.2f}', ha='center', fontsize=8, fontweight='bold', color='#1f4e79')
    ax_p.text(p_resnet, 0.15, f'{p_resnet:.2f}', ha='center', fontsize=8, fontweight='bold', color='#7a1f15')
    ax_p.set_xlim(0, 1); ax_p.set_ylim(0, 1)
    ax_p.set_yticks([])
    ax_p.set_xlabel('P(clear)  —  higher means the scene looks clear-weather')
    ax_p.legend(loc='upper center', bbox_to_anchor=(0.5, -0.40), ncol=2, fontsize=8, framealpha=0.9)

    out = OUT / f'{name}.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------
print('Running synthetic scenes through the safety gate...')
results = []
for slug, label, fn in SCENES:
    scene = fn().convert('RGB')
    p_clip, _ = gate.clip_zero_shot(scene)
    cls_label, cls_conf, cls_probs, p_resnet = gate.classify(scene)
    decision, color, rationale = gate.safety_decision(p_clip, p_resnet)
    out = render_panel(slug, label, scene, decision, color, p_clip, p_resnet, rationale)
    print(f'  [{slug:<6}] CLIP P(clear)={p_clip:.3f}  ResNet P(clear)={p_resnet:.3f}  -> {decision:<10}  ({out})')
    results.append((slug, label, decision, p_clip, p_resnet))

# Markdown snippet for README embedding
md_lines = [
    '| Scene | CLIP P(clear) | ResNet P(clear) | Gate |',
    '|---|---:|---:|---|',
]
for slug, label, decision, p_clip, p_resnet in results:
    md_lines.append(f'| {label} | {p_clip:.2f} | {p_resnet:.2f} | **{decision}** |')

(OUT / 'results.md').write_text('\n'.join(md_lines) + '\n')
print(f'\nWrote {len(results)} panels + results.md to {OUT}/')
