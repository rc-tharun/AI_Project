# Demo Script — ODD + OOD Safety Gate

**Format:** spoken walk-through alongside the running web app (`python app.py` → http://127.0.0.1:7860).
**Target length:** ≈ 2 minutes 30 seconds.
**What's shown live:** two real driving frames (one clear, one adverse) being scored by the gate end-to-end.

> Tip: have the app already running in a browser tab before you start, and have the two test images from `Test Pictures/` ready to drag-and-drop.

---

## [00:00 – 00:20]  Set up the problem (≈20 s)

> "What I'm showing is a **safety gate** for a self-driving car's perception stack. For every front-camera frame the gate has to choose one of three actions: **TRUST** the prediction, **SLOW DOWN** if the evidence is weak, or **ABSTAIN** and hand back to the driver if the frame is out-of-distribution — rain, fog, snow, night. The whole point is that a confident-but-wrong prediction is worse than refusing to act."

## [00:20 – 00:50]  Explain the architecture (≈30 s)

> "Behind the UI there are **two independent classifiers** running on the same frame:
>
> 1. **CLIP ViT-B/32 zero-shot** — we ask CLIP whether the image looks more like *'a clear sunny day road scene'* or *'a foggy / rainy / snowy / nighttime road scene'*. CLIP gives back a probability we treat as P(clear).
> 2. **A trained ResNet-50** — fine-tuned on BDD100K for clear-versus-adverse weather. That gives a second, independent P(clear).
>
> The fusion rule is simple: **TRUST** only if both methods say P(clear) ≥ 0.65; **ABSTAIN** only if both methods say P(clear) ≤ 0.35; otherwise **SLOW DOWN**. So neither network alone can swing the gate to TRUST — they have to agree."

## [00:50 – 01:30]  Run a clear-weather frame (≈40 s)

> *(Drag the empty-highway test image into the upload box and click Evaluate.)*
>
> "I'm dropping in a clear highway frame. Watch the right-hand side."
>
> *(Wait for the panels to update.)*
>
> "Both methods land in the green zone. CLIP gives P(clear) ≈ 0.83, ResNet ≈ 0.99 — they agree, the chart shows both dots in the trust band, and the gate returns **TRUST**. The top matching CLIP prompts at the bottom are *'a highway under blue sky'* and *'a daytime driving scene with clear weather'* — so we can read out exactly *why* CLIP thinks it's in-domain, not just *that* it does."

## [01:30 – 02:10]  Run an adverse frame (≈40 s)

> *(Replace the image with the foggy / snowy test frame and click Evaluate.)*
>
> "Now a foggy frame from the same camera position. Both probabilities collapse — CLIP P(clear) drops to about 0.04, ResNet drops to about 0.13. Both dots are now deep in the red band, the methods agree the scene is adverse, and the gate returns **ABSTAIN** — which means the perception stack would refuse to act and yield control."

## [02:10 – 02:30]  Wrap-up (≈20 s)

> "The point of the dual-classifier design is **defense-in-depth**: an attacker or a corner case has to fool *both* a generic vision-language model and a fine-tuned dataset-specific classifier to get a TRUST out of the gate. In the report we show that on BDD100K validation this fusion gives 70 % TRUST coverage on clear-weather frames while catching 21 % of adverse frames as ABSTAIN and another 45 % as SLOW DOWN — the rest of the metrics are in the README. That's the demo."

---

## Cheat sheet — what to point at on screen

| Moment | Where the eye should go |
|---|---|
| Decision banner (top-right) | Big colored TRUST / SLOW DOWN / ABSTAIN word — that's the headline output. |
| P(clear) chart | Two dots: blue = CLIP, red = ResNet. They should both land in the same band when the gate is confident. |
| Per-method P(clear) labels | Confirms the dot positions numerically. |
| Top matching CLIP prompts | Plain-English rationale for the CLIP vote. |

## If something goes wrong

| Symptom | What to say |
|---|---|
| Both dots end up in red but I expected TRUST | "This image probably has a different camera or color cast from BDD — the gate is doing the right thing by being cautious, since when in doubt we want SLOW DOWN, not TRUST." |
| Methods disagree (one green, one red) | "Perfect example of why we use two detectors — they're disagreeing, so the gate falls back to SLOW DOWN. That's the failure-safe behavior we want." |
| App is slow on first frame | "First inference loads CLIP into memory; subsequent frames are sub-second." |
