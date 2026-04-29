# Demo Script — ODD + OOD Safety Gate

**Format:** spoken walk-through alongside the running web app (`python app.py` → http://127.0.0.1:7860).
**Target length:** ≈ 3 minutes 0 seconds.
**Demo plan:** four real driving frames from `Test Pictures/`, walking the gate through every decision band — two TRUST cases, one ABSTAIN, and one SLOW DOWN where the two detectors disagree.

> The four screenshots in `demo outputs/` are saved as a fallback; if the live app misbehaves, fall back to walking through the PNGs in the same order.

---

## [00:00 – 00:20]  Set up the problem (≈20 s)

> "What I'm showing is a **safety gate** for a self-driving car's perception stack. For every front-camera frame the gate has to choose one of three actions: **TRUST** the prediction, **SLOW DOWN** if the evidence is weak, or **ABSTAIN** and hand back to the driver if the frame is out-of-distribution — rain, fog, snow, night. The whole point is that a confident-but-wrong prediction is worse than refusing to act."

## [00:20 – 00:45]  Explain the architecture (≈25 s)

> "Behind this UI there are **two independent classifiers** running on the same frame.
>
> The first is **CLIP ViT-B/32 zero-shot** — we ask CLIP whether the image looks more like *'a clear sunny day road scene'* or *'a foggy / rainy / snowy / nighttime road scene'*. CLIP returns a probability we treat as P(clear).
>
> The second is a **trained ResNet-50** fine-tuned on BDD100K weather data — that gives a second, independent P(clear).
>
> The fusion rule is intentionally simple: **TRUST** only if both methods say P(clear) ≥ 0.65; **ABSTAIN** only if both methods say P(clear) ≤ 0.35; otherwise **SLOW DOWN**. So neither network alone can flip the gate to TRUST — they have to agree."

## [00:45 – 01:15]  Case 1 — Empty highway → **TRUST** (≈30 s)

> *(Drag in `Test Pictures/empty-highway-beautiful-sunny-weather-...webp`. Click Evaluate.)*
>
> "Frame one is an empty highway under blue sky — the kind of frame the safety gate should obviously TRUST. Look at the panels: CLIP zero-shot returns P(clear) = **0.99**, ResNet-50 returns **0.90**. Both dots sit deep in the green band, the methods agree, and the gate fires **TRUST**.
>
> Notice the bottom of the panel — the top matching CLIP prompts are *'a photo of a road on a clear sunny day'* at 83 percent and *'a highway under blue sky'* at 16 percent. So we get the decision *and* a plain-English rationale for free."

## [01:15 – 01:45]  Case 2 — Curving country road → **TRUST** (≈30 s)

> *(Replace with `Test Pictures/download (2).jpeg` — the curving country road. Click Evaluate.)*
>
> "Different camera, different angle, but again clear daylight. CLIP P(clear) is **0.99**, ResNet **0.90**. Same TRUST, same green band — and that's important, because it shows the system isn't memorizing one specific highway shot. It generalizes across viewpoints. The top CLIP prompts confirm it: *'a clear sunny day'* at 82 percent, *'a highway under blue sky'* at 14 percent."

## [01:45 – 02:15]  Case 3 — Foggy bridge → **ABSTAIN** (≈30 s)

> *(Replace with `Test Pictures/images.jpeg` — the foggy bridge. Click Evaluate.)*
>
> "Now the same kind of road, but in heavy fog. The probabilities flip completely: CLIP P(clear) collapses to about **0.07**, ResNet drops to **0.23**. Both dots land in the red band, the methods agree the scene is adverse, and the gate fires **ABSTAIN**.
>
> In a real car this is the moment perception yields control to the driver. The gate isn't trying to keep driving in fog — it's correctly refusing."

## [02:15 – 02:45]  Case 4 — Snowy industrial scene → **SLOW DOWN** (≈30 s)

> *(Replace with the snowy / industrial track frame — `download (1).jpeg`. Click Evaluate.)*
>
> "This last one is the most interesting case, and it's why we use **two** detectors instead of one.
>
> CLIP P(clear) is **0.88** — it sees daylight and snow-covered ground and calls it clear. The trained ResNet-50, which actually saw BDD's adverse-weather frames, calls it the opposite — P(clear) = **0.20**.
>
> The two methods disagree by almost 0.7. Either of them on its own would give a confident-but-wrong answer here. But because the gate **requires agreement** for TRUST, it correctly drops to **SLOW DOWN**. That's the defense-in-depth value: any single classifier would have failed, the ensemble caught it."

## [02:45 – 03:00]  Wrap-up (≈15 s)

> "So in three minutes we've seen the gate land in all three bands — TRUST when both methods agree the scene is clear, ABSTAIN when both agree it's adverse, and SLOW DOWN when they disagree. On BDD100K validation this gives 70 percent TRUST coverage on clear-weather frames while flagging 21 percent of adverse frames for ABSTAIN and another 45 percent for SLOW DOWN. Numbers and the full method comparison are in the README. That's the demo."

---

## Timing summary

| Section | Duration | Decision shown |
|---|---:|---|
| Problem framing | 0:20 | — |
| Architecture | 0:25 | — |
| Case 1: Empty highway | 0:30 | **TRUST**  (0.99 / 0.90) |
| Case 2: Country road | 0:30 | **TRUST**  (0.99 / 0.90) |
| Case 3: Foggy bridge | 0:30 | **ABSTAIN**  (0.07 / 0.23) |
| Case 4: Snowy industrial | 0:30 | **SLOW DOWN**  (0.88 / 0.20) |
| Wrap-up | 0:15 | — |
| **Total** | **3:00** | |

## Cheat sheet — what to point at on screen

| Moment | Where the eye should go |
|---|---|
| Decision banner (top-right) | Big colored TRUST / SLOW DOWN / ABSTAIN word — that's the headline output. |
| P(clear) chart | Two dots: blue = CLIP, red = ResNet. They should both land in the same band when the gate is confident. |
| Per-method P(clear) labels | Confirms the dot positions numerically for the audience. |
| Top matching CLIP prompts | Plain-English rationale for the CLIP vote — read at least one out loud per case. |

## If something goes wrong

| Symptom | What to say |
|---|---|
| Both dots land in red but I expected TRUST | "This image probably has a different camera or color cast from BDD — the gate is doing the right thing by being cautious. When in doubt we want SLOW DOWN, not TRUST." |
| Methods disagree (one green, one red) on a frame I expected to be clean | "Perfect example of why we use two detectors — they're disagreeing, so the gate falls back to SLOW DOWN. That's the failure-safe behavior we want." |
| App is slow on the first frame | "First inference loads CLIP into memory; subsequent frames are sub-second." |
| App crashes mid-demo | Switch to the saved screenshots in `demo outputs/` — they show every case verbatim, in the same order. |
