# 2D Context Aggregation Network for Prostate MRI Segmentation (HipMRI)

**Student:** s5042697 · **Project:** 2.2 Segmentation · **Model:** 2D CAN [Normal difficulty] · **Baseline:** 2D U-Net

## Contents

1. [Problem](#1-problem)
2. [How the CAN works](#2-how-the-can-works)
3. [Feasibility review](#3-feasibility-review)
4. [Data and preprocessing](#4-data-and-preprocessing)
5. [Training set-up](#5-training-set-up)
6. [Results: CAN vs U-Net](#6-results-can-vs-u-net)
7. [Failure case analysis](#7-failure-case-analysis)
8. [Recommendation](#8-recommendation)
9. [Usage](#9-usage)
10. [Dependencies and reproducibility](#10-dependencies-and-reproducibility)
11. [Artificial Intelligence Usage Disclosure](#11-artificial-intelligence-usage-disclosure)
12. [References](#12-references)

## 1. Problem

External-beam radiotherapy for prostate cancer needs the prostate and nearby organs at risk (bladder, rectum, bones) contoured on every MRI slice. Doing this by hand is slow. This project builds a prototype that proposes a draft contour for a specialist to review and correct.

The model segments each 2D pelvic MRI slice from the HipMRI study into six labels: background, body, bone, bladder, rectum and prostate. The prostate covers under 0.5% of the pixels.

**The engineering dilemma.** A high overall Dice score can hide the failures that matter clinically: missed prostate at the base and apex, or contours that leak into the rectum or bladder. This project asks whether the multi-scale context of a Context Aggregation Network (CAN) fixes these boundary failures compared with a plain 2D U-Net, or whether it only does as well on the easy central slices while costing more compute.

## 2. How the CAN works

![U-Net vs CAN architecture](images/architecture.png)

*Top: the U-Net halves the image size four times (box height shows resolution, labels show channels and size), then grows it back, with skip connections (grey) carrying detail across. Bottom: every CAN layer stays at 256×128. The red dilated layers widen the view instead. The small grids show which pixels a 3×3 filter samples at dilations 1, 2 and 4.*

A U-Net shrinks the image to see a wide area, then grows it back, using skip connections to recover detail. The CAN (Yu and Koltun, 2016) never shrinks the image. It widens its view with **dilated convolutions**: 3×3 filters that sample pixels with gaps between them. A filter with dilation *d* still has 9 weights but covers a (2*d*+1)×(2*d*+1) area.

The network in [`modules.py`](modules.py) has three parts:

1. **Front-end:** two ordinary 3×3 convolutions that pick up local features such as edges.
2. **Context module:** eight 3×3 convolutions with dilations 1, 1, 2, 4, 8, 16, 32, 1. Doubling the dilation grows the receptive field exponentially, so each output pixel draws on about 135 px of surroundings: the prostate plus the bladder, rectum and pelvic bones around it. The paper stops at dilation 16; dilation 32 was added so the receptive field spans these 256×128 slices.
3. **Head:** a 1×1 convolution that turns each pixel's 64 features into scores for the six labels.

Every layer runs at the full 256×128 resolution, so small structures are never shrunk away. Both models use batch norm and ReLU, so the comparison isolates "dilated full-resolution context" against "encoder-decoder downsampling".

| Model | Parameters | Receptive field | Resolution inside the network |
|---|---|---|---|
| U-Net (baseline) | 7.76M | whole image (4 poolings) | 256×128 down to 16×8 |
| CAN | 0.33M | ~135 px | 256×128 throughout |
| CAN-128 (wider) | 1.33M | ~135 px | 256×128 throughout |
| ResidualCAN | 0.33M | ~145 px | 256×128 throughout |

**ResidualCAN** is a variant tried after the first results (see [Section 6.3](#63-can-variants)). It replaces the doubling dilations with non-doubling ones (1, 2, 5, 9, 17, 33, 2, 1), following Hybrid Dilated Convolution (Wang et al., 2018), to avoid the "gridding" gaps of doubling dilations on thin structures. Its context layers are residual and start as identities, which plays the role of the identity initialisation recommended by Yu and Koltun.

## 3. Feasibility review

*Presented at the midpoint check-off on 21 Oct 2026. Reproduced as submitted; the final results in Section 6 replace the pilot numbers.*

<!-- TODO: paste feasibility_review.md here (sections 1-4), then delete feasibility_review.md or keep it as the source -->

## 4. Data and preprocessing

**Dataset:** HipMRI 2D slices (`keras_slices_data`, Rangpur `/home/groups/comp3710/HipMRI_Study_open`), stored as Nifti files and loaded with `nibabel`.

| Split | Patients | Slices |
|---|---|---|
| Train | 31 | 11,460 |
| Validate | 4 | 660 |
| Test | 3 | 540 |

**Why these splits.** The provided folders are already split by patient, and they are used unchanged. [`dataset.py`](dataset.py) asserts that no patient ID appears in more than one split, because slices from the same patient are highly correlated, so a patient-level leak would inflate test scores. The weakness is size: 3 test patients and 4 validation patients make the scores noisy, so every model is trained with 3 seeds and results are reported as mean ± standard deviation.

**Preprocessing:**

- **Shape:** slices are 256×128, except `case_019_week_1` (256×144), which is centre-cropped to 256×128. Cropping keeps the pixel spacing, unlike resizing.
- **Intensity:** per-slice z-score normalisation (zero mean, unit variance), applied after augmentation so padded regions stay at the scanner's background value.
- **Labels:** integer masks, converted to one-hot channels inside the loss.
- **Augmentation (training only):** left-right mirror, plus a random affine (±10° rotation, ±5% shift, 0.9–1.1 scale), applied jointly to the image and mask (masks use nearest-neighbour resampling). No anterior-posterior flip, because that is not anatomically plausible.

**Class imbalance:** prostate 0.43% and rectum 0.36% of training pixels. This is handled in the loss (Section 5).

## 5. Training set-up

All models use identical settings ([`train.py`](train.py)):

| Setting | Value |
|---|---|
| Loss | Cross-entropy + soft Dice over the foreground classes ([`utils.py`](utils.py)) |
| Optimiser | Adam, learning rate 5e-4, decayed ×0.985 per epoch, weight decay 1e-5 |
| Batch size / epochs | 16 / 30 |
| Model selection | Checkpoint with the best **validation** prostate Dice |
| Seeds | 42, 1, 2 |
| Hardware | One NVIDIA A100-PCIE-40GB on Rangpur, one Slurm job per run ([`job_train.sh`](job_train.sh)) |

The Dice term counters class imbalance. It averages over foreground classes only, so the large background class does not dominate it.

![Training curves](images/curves.png)

*Validation loss and validation prostate Dice per epoch. Lines are the mean over 3 seeds and the shading is ± 1 standard deviation. The U-Net leads on validation prostate Dice from about epoch 4 onwards. All models flatten out by epoch 20–30, so 30 epochs is enough.*

## 6. Results: CAN vs U-Net

All numbers are on the 3 held-out test patients. Each is the mean ± standard deviation over 3 seeds.

### 6.1 Accuracy

| Class | U-Net Dice | CAN Dice | CAN-128 Dice | U-Net IoU | CAN IoU | CAN-128 IoU |
|---|---|---|---|---|---|---|
| Body | 0.981 ± 0.001 | 0.980 ± 0.000 | 0.980 ± 0.000 | 0.963 ± 0.001 | 0.961 ± 0.001 | 0.962 ± 0.000 |
| Bone | 0.940 ± 0.003 | 0.936 ± 0.000 | 0.937 ± 0.001 | 0.886 ± 0.004 | 0.880 ± 0.001 | 0.882 ± 0.002 |
| Bladder | 0.964 ± 0.002 | 0.963 ± 0.002 | 0.964 ± 0.001 | 0.930 ± 0.003 | 0.929 ± 0.003 | 0.930 ± 0.001 |
| Rectum | **0.876 ± 0.007** | 0.845 ± 0.026 | 0.844 ± 0.015 | **0.779 ± 0.011** | 0.732 ± 0.039 | 0.730 ± 0.022 |
| **Prostate** | **0.879 ± 0.004** | 0.863 ± 0.015 | 0.870 ± 0.002 | **0.784 ± 0.007** | 0.759 ± 0.023 | 0.771 ± 0.003 |

<!-- TODO: add ResidualCAN columns once job 699979 finishes -->

Every model clears the 0.75 prostate Dice target. The U-Net leads on rectum by about 0.03 across all seeds. Its prostate lead (0.01–0.02) is within the noise of a 3-patient test set.

### 6.2 Boundary analysis: where the errors are

Overall Dice is dominated by the central slices. [`predict.py`](predict.py) splits the prostate two ways:

- **Superior-inferior (within each coronal slice):** the gland's vertical extent per scan is cut into thirds. The upper third is the **base** (next to the bladder) and the lower third is the **apex**. Dice is pooled over the pixels in each band.
- **Anterior-posterior (across slices):** the slices containing prostate in each scan are cut into an anterior edge, central and posterior edge third. Dice is averaged per slice.

The figure shows the anterior-posterior split for one test scan. Each bar is one slice, with height equal to the number of true prostate pixels, coloured by its third. The lines show how many pixels each model labels as prostate.

![Slice thirds for one test scan](images/slice_thirds.png)

*In the central third both models track the truth closely. At the posterior end (slices 33–38) both under-predict and then stop early: slice 38 still has 65 true prostate pixels, but both models predict none.*

| Prostate region | U-Net | CAN | CAN-128 |
|---|---|---|---|
| Base band | **0.847 ± 0.009** | 0.822 ± 0.015 | 0.843 ± 0.002 |
| Mid band | **0.922 ± 0.004** | 0.911 ± 0.012 | 0.909 ± 0.002 |
| Apex band | **0.831 ± 0.006** | 0.820 ± 0.023 | 0.823 ± 0.002 |
| Anterior edge slices | **0.731 ± 0.021** | 0.673 ± 0.046 | 0.638 ± 0.012 |
| Central slices | 0.915 ± 0.003 | 0.910 ± 0.006 | 0.916 ± 0.003 |
| Posterior edge slices | 0.685 ± 0.022 | 0.686 ± 0.041 | 0.704 ± 0.025 |
| Worst single slice | 0.000 | 0.000 | 0.000 |
| False prostate on slices with none (of 345) | **1.3 ± 1.2** | 8.7 ± 3.1 | 8.7 ± 4.6 |

**Findings:**

1. **Overall Dice hides edge failures.** Central slices score about 0.91 for every model, while edge slices drop to 0.64–0.73. Every model has slices where the prostate is missed completely (Dice 0). The base and apex bands sit 0.08–0.09 below the mid band.
2. **The CAN's extra context does not fix them.** It is no better than the U-Net at the base and apex, and worse on the anterior edge slices.
3. **The CAN hallucinates prostate about 6× as often**: 8.7 vs 1.3 false-positive slices. In radiotherapy that is over-contouring, which irradiates healthy tissue.

<!-- TODO: add ResidualCAN row; short paragraph on posterior edge, the one place CAN-128 is slightly ahead (within noise) -->

### 6.3 CAN variants

Two variants tested whether the CAN's gap was a fixable design issue. Both were chosen on **validation** prostate Dice, so the test set was not used for model selection.

| Model | Validation prostate Dice (best epoch, 3 seeds) | Hypothesis tested |
|---|---|---|
| U-Net | **0.863** | baseline |
| CAN | 0.841 | |
| CAN-128 | 0.848 | Is the CAN behind only because it is small? (4× parameters) |
| ResidualCAN | 0.844 | Do gridding gaps from doubling dilations hurt thin structures? |

Neither variant closes the gap. Width makes the CAN steadier across seeds (prostate std 0.002 vs 0.015) but not more accurate, and the non-doubling dilations did not improve the rectum. So the gap does not come from model size or from gridding.

<!-- TODO: ResidualCAN test numbers from job 699979 -->

### 6.4 Resource profile

| | U-Net | CAN | CAN-128 | ResidualCAN |
|---|---|---|---|---|
| Parameters | 7.76M | **0.33M** | 1.33M | **0.33M** |
| Peak GPU memory, training (batch 16) | **1.56 GB** | 2.85 GB | 5.68 GB | 2.96 GB |
| Peak GPU memory, inference (batch 1) | TODO | TODO | TODO | TODO |
| Inference time per slice (batch 1) | 2.7 ms | **1.6 ms** | 3.0 ms | TODO |
| Training time per epoch (11,460 slices) | **99 s** | 105 s | 127 s | 114 s |
| Training time, 30 epochs | **~50 min** | ~53 min | ~64 min | ~57 min |

Measured on an A100-PCIE-40GB. Training memory and time are from the seed-42 run.

**The CAN has 23× fewer parameters but needs 1.8× the training memory.** Every layer keeps a full-resolution 64-channel feature map for the backward pass, so memory goes on activations, not weights. Its one practical advantage is inference speed at 64 channels (1.6 vs 2.7 ms per slice), which is irrelevant at this scale: a 60-slice scan takes well under a second with either model.

## 7. Failure case analysis

<!-- TODO: copy compare_worst.png and 3-5 per-case crops into images/, then write each case:
     what the slice shows, what each model predicted, why it fails, clinical consequence -->

The five worst slices (lowest prostate Dice averaged across models) are **the same slices for all nine trained models**. Every one is an edge slice (four anterior, one posterior) where the true prostate covers only a few pixels, and every model misses it completely.

![Worst slices across all models](images/compare_worst.png)

### Case 1: TODO

### Case 2: TODO

### Case 3: TODO

**Likely root cause.** The same slices fail for every architecture, which suggests the failure comes from the slice-by-slice 2D input rather than from the CAN or U-Net design. A single slice gives no evidence that the gland is just starting or ending in the neighbouring slice, so a few-pixel cross-section can look like background. More in-plane context, which is what the CAN adds, would not supply information that lies across slices. This is consistent with neither CAN variant improving the edge slices, but it has not been tested directly. The natural next experiment is a 2.5D input (the neighbouring slices as extra channels).

## 8. Recommendation

<!-- TODO: finalise after ResidualCAN numbers -->

**To the project manager:**

1. **Use the 2D U-Net, not the CAN.** The CAN's multi-scale context gives no measurable clinical gain at the prostate boundaries. It is less accurate at the rectum, the critical organ at risk, produces about 6× more false-positive prostate slices, and needs 1.8× the GPU memory to train. Its 23× smaller size and faster inference do not matter for an offline contouring tool.
2. **Do not automate the edge slices.** With every model, the first and last few prostate-bearing slices of each scan, plus the base and apex regions, need mandatory review by a specialist. The tool is suitable as a draft on central slices (Dice ≈ 0.91–0.92).
3. **Next investment: inter-slice context, not more in-plane context.** The shared failures point to a 2.5D input (neighbouring slices as extra channels) or a 3D model as the next step, not a wider 2D context.
4. **Caveat:** 3 test patients is a small sample. Validate on more patients before any clinical use.

## 9. Usage

All commands run from this folder.

```bash
# Check the data split and one batch
python dataset.py

# Shape check for all models
python modules.py

# Train (outputs go to runs/<run-name>/: best.pt, history.json, curves.png, test_metrics.json)
python train.py --model can --epochs 30                 # CAN
python train.py --model unet --epochs 30                # baseline U-Net
python train.py --model can --base-channels 128         # wider CAN
python train.py --model can_res                         # ResidualCAN
python train.py --model can --epochs 2 --limit 64       # smoke test

# On Rangpur (see job_train.sh for more examples)
MODEL=can SEED=1 RUN_NAME=can_s1 sbatch job_train.sh

# Evaluate, boundary analysis and failure figures
python predict.py --checkpoints runs/unet/best.pt runs/can/best.pt --out-dir runs/predict
```

**Example output of `predict.py`** (U-Net, seed 42):

```
=== unet (runs/unet/best.pt) on cuda ===
  rectum     Dice 0.8790  IoU 0.7841
  prostate   Dice 0.8787  IoU 0.7837
  prostate Dice by superior-inferior band (pooled over test scans):
    base 0.841  mid 0.923  apex 0.838
  prostate by slice position (mean / worst per-slice Dice):
    anterior   68 slices  0.739 / 0.000
    central    67 slices  0.917 / 0.871
    posterior  60 slices  0.680 / 0.000
  false-positive prostate on 2 of 345 slices with no prostate
  latency 2.7 ms per slice (batch 1)
```

<!-- TODO: example input / ground truth / prediction figure -->

## 10. Dependencies and reproducibility

<!-- TODO: requirements.txt pins torch 2.14.1 / numpy 2.5.4, but the Rangpur runs used torch 2.13.0+cu130 / numpy 2.4.6. Align before submitting. -->

| Package | Version (Rangpur runs) |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.13.0 (CUDA 13.0) |
| NumPy | 2.4.6 |
| nibabel | 5.4.2 |
| matplotlib | 3.11.2 |

**Reproducibility:**

- `--seed` sets `torch.manual_seed`, which fixes weight initialisation, data order and augmentation. Results are reported over seeds 42, 1 and 2.
- cuDNN is not forced into deterministic mode, so re-running the same seed can differ slightly. This is one more reason the results are given as mean ± std rather than single numbers.
- Every checkpoint stores its model name, constructor arguments, seed and epoch, so `predict.py` rebuilds the exact model.
- Data and weights are not committed (see `.gitignore`).

## 11. Artificial Intelligence Usage Disclosure

<!-- TODO (student): write this in your own words. The spec requires:
     - which AI tools were used
     - what specific tasks they helped with
     - how you audited, verified and tested the AI-suggested code -->

## 12. References

<!-- TODO: check and complete -->

1. F. Yu and V. Koltun, "Multi-scale context aggregation by dilated convolutions," ICLR, 2016.
2. O. Ronneberger, P. Fischer and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," MICCAI, 2015.
3. P. Wang et al., "Understanding convolution for semantic segmentation," WACV, 2018 (Hybrid Dilated Convolution).
4. F. Yu, V. Koltun and T. Funkhouser, "Dilated residual networks," CVPR, 2017.
5. P. Goyal et al., "Accurate, large minibatch SGD: Training ImageNet in 1 hour," arXiv:1706.02677, 2017 (zero-initialised batch-norm scale in residual blocks).
6. F. Isensee et al., "Brain tumor segmentation and radiomics survival prediction: Contribution to the BRATS 2017 challenge," 2017 (learning-rate schedule).
7. HipMRI study data: <!-- TODO: dataset citation from the course spec appendix -->
