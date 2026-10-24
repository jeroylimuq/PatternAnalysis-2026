# Feasibility Review: 2D CAN for Prostate Segmentation (HipMRI)

**Student:** s5042697 · **Project:** 2.2 Segmentation · **Difficulty:** Normal · **Draft:** 21 Oct 2026

## 1. User Need, Scope and Acceptance Criteria

**User:** a radiation oncology team contouring the prostate on pelvic MRI before external-beam radiotherapy. **Prototype purpose:** propose a draft multi-structure contour (prostate, bladder, rectum, bone, body) on each 2D slice that a specialist reviews and corrects, reducing manual contouring time.

**Open dilemma:** a good global Dice can hide dangerous failures at the prostate base and apex, or boundary leakage into the rectum and bladder. The question is whether the multi-scale context of a Context Aggregation Network (CAN) fixes these boundary failures compared with a plain 2D U-Net, or only improves easy central slices while costing more compute.

**Acceptance criteria (all on the held-out test patients):**

1. **Accuracy:** prostate Dice ≥ 0.75 for the CAN.
2. **Baseline comparison:** per-class Dice and IoU reported for both models on identical patient-level splits; no patient appears in more than one split.
3. **Boundary failure analysis:** prostate Dice reported separately for the base, apex and edge slices versus central slices, with 3–5 failure cases visualised (input, ground truth, prediction, error overlay).
4. **Resources:** training completes within one 8-hour Rangpur GPU job per model; peak GPU memory and per-case inference time reported.

## 2. Model Choice and Course Concepts

- **Data:** HipMRI 2D slices (`keras_slices_data`), 256×128, six labels (background, body, bone, bladder, rectum, prostate). The provided train/validate/test folders are already split by patient (31 / 4 / 3) and are used unchanged.
- **Baseline:** 2D U-Net: encoder-decoder with skip connections, max-pool downsampling, batch norm.
- **Chosen model:** 2D Context Aggregation Network (CAN) (Normal difficulty). Instead of shrinking and re-growing the image like a U-Net, it widens its view while staying at full size:
  1. **Front-end:** two ordinary 3×3 convolutions pick up local features such as edges.
  2. **Dilated convolutions:** 3×3 filters with gaps between the pixels they look at. They still use 9 weights, but cover a much wider area.
  3. **Doubling dilation:** the gaps grow 1, 1, 2, 4, 8, 16, 32, 1 across the context module, so each pixel's prediction draws on about 135 pixels of surroundings (prostate plus bladder, rectum and bones).
  4. **Full resolution throughout:** no pooling or upsampling, so small structures are never shrunk away.
  5. **1×1 classifier:** turns each pixel's features into scores for the six labels.
- **Comparison set-up:** both models use batch norm with ReLU and identical training settings. The CAN uses 64 channels, as in a standard context module, giving 0.33M parameters against 7.76M for the baseline.
- **Course concepts:** convolutional encoder-decoders, dilated convolutions and receptive fields, skip connections, normalisation layers, augmentation, and Dice + cross-entropy loss to handle class imbalance.

## 3. Preliminary Feasibility Evidence

**Data audit:** 11,460 / 660 / 540 slices across train / validate / test with no patient overlap. Severe class imbalance: prostate 0.43% and rectum 0.36% of training pixels. One case (`case_019_week_1`, 60 slices) is 256×144 and is centre-cropped to 256×128.

**Pilot runs** (5 epochs, first 1,500 training slices ≈ 4 patients, full validation and test splits, Apple M2):

| Model | Params | Test prostate Dice | Test bladder Dice | Test rectum Dice | Time / 1,500-slice epoch |
|---|---|---|---|---|---|
| Baseline U-Net | 7.76M | 0.81 | 0.95 | 0.80 | ~90 s |
| CAN | 0.33M | 0.79 | 0.95 | 0.78 | ~170 s |

Both models pass the 0.75 prostate target after a short run, so the target seems achievable. The CAN reaches similar accuracy with 23× fewer parameters, but is about 2× slower per epoch because every layer works on the full-size image. A short single run on about 4 patients cannot separate the two models; full runs are needed.

## 4. Risks, Budget and Fallback

| Risk | Mitigation |
|---|---|
| Only 3 test patients and 4 validation patients make scores noisy | Report per-patient results and avoid over-claiming; select checkpoints on validation prostate Dice only |
| Global Dice hides base/apex failures | Analyse prostate Dice by position (base, apex, edge slices), as in criterion 3 |
| Full-resolution layers make the CAN slow and memory-hungry | Measure peak GPU memory and per-case inference time; reduce batch size if needed |
| CAN does not beat the baseline | A valid finding for the dilemma, report it with evidence |

**Compute budget:** about 30 epochs per model, at most one 8-hour Rangpur GPU job each, plus short test jobs.

**Next experiment:** full 30-epoch training of both models on Rangpur (3 seeds each), then per-slice failure analysis in `predict.py`.

**Fallback:** if the full runs show no clear accuracy gain from the CAN, use the repeat seeds and a wider CAN (128 channels) to confirm the result is real and not caused by its small size, then recommend the cheaper model with the evidence. If prostate Dice falls below 0.75 on the stricter per-slice average, tune the loss weighting and augmentation and extend training. Edge-slice failures are reported as a limitation, with a recommendation for mandatory human review of those slices.
