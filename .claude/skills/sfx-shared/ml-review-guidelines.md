# ML review guidelines (Hit_finder)

Used by the `sfx-ml-reviewer` agent and by `/sfx-evaluate validity`. The user's
guidelines are the source and are kept verbatim in the appendix. The sections before the
appendix say how they apply in this repo. Read `invariants.md` first.

## Scope

- A finding must be caused by the diff under review (or, in a validity audit, by the
  experiment under review).
- Pre-existing gaps are listed once under "Pre-existing gaps" and never block. Known
  today: no recall-at-fixed-FPR metric in `src/evaluation/metrics.py`; no
  saturated-pixel handling outside the hitfinder.
- **Conflict rule.** Where a guideline conflicts with the fixed architecture in
  `CLAUDE.md`, raise a **design note** for discussion. Never a finding, never a severity.
  If the same note keeps recurring, the user updates `CLAUDE.md`.

## Severity

- 🔴 **BLOCKING** — calibration leakage (both rules below); pixel mask smearing;
  sensor-split data leakage; evaluation using plain accuracy on rare events. Also: a new
  loss function or mask generator without the unit tests of directive 2; a normalisation
  or feature-scaling change without the citations of directive 3.
- 🟡 **WARNING** — inefficient per-frame loops in dataloaders; unvalidated
  mixed-precision underflow; RGB-oriented augmentations on grayscale sensor data.
- 🟢 **NIT** — naming, type annotations, docstring formatting. At most three per review.

Where the category text in the appendix says "reject" and this list says 🟡 (RGB
augmentations), this list wins.

## Core verification directives

1. For changes under `src/models/`, `src/data/` or `src/preprocessing/`, trace arrays
   from the raw read through the valid-pixel mask and confirm that no invalid pixel
   enters a convolutional neighbourhood.
2. Demand unit tests for any new loss function or mask generator: output shape, and
   masking behaviour that matches the stated formulation.
3. Require line numbers and tensor shapes (with dtype) for any proposed normalisation or
   feature-scaling change, before and after.

## Calibration leakage rules

1. 🔴 **Detector hardware calibration leakage.** Dark frames, flat-field gain
   corrections, and pixel status masks (dead/hot/saturated pixels) must be derived
   strictly from dedicated calibration runs or isolated training runs. Panel geometry
   files and coordinate maps must never be fitted, optimized, or tuned using
   events/frames from the validation or test splits.
2. 🔴 **Evaluation and statistical calibration leakage.** Normalization statistics
   (percentile bounds, median background estimators) must be computed per-frame or
   strictly fitted on the training split. Decision thresholds (e.g. peak-finding SNR
   cutoffs, classification operating points) must be calibrated exclusively on a
   validation fold and evaluated frozen on the hold-out test set.

In this pipeline rule 1 is the more critical and the more easily overlooked. `src/` has
no pedestal, gain or mask-fitting code today, so rule 1 is mostly a provenance question
for `/sfx-evaluate validity`. In a diff it applies to any new mask, calibration table or
geometry derived from frames. Under rule 2: the PF8 hitfinder produces the training
labels, and under leave-one-detector-out every detector is the test set of one fold, so
which frames the PF8 parameters were tuned on is a validity question.

## The five categories, applied

| Check | Status | In this repo |
|---|---|---|
| 1. Calibration order | Adapted | No calibration code in `src/`; frames arrive calibrated. Nothing may rescale or clip between read and GCN. Any future calibration step sits before GCN |
| 1. Bit depth and saturation | Adapted | Flag `uint8` casts and clipping before GCN. The frame cache stores GCN'd frames as `float16` by design; check that its bit-exactness test (`tests/test_frame_cache.py::TestBitExactnessAgainstRealData`) covers the changed path |
| 1. Pixel mask leakage | As written, 🔴 | Matches the no-resize rule (I3) and masked LCN (I5) |
| 2. RGB anti-patterns | As written, 🟡 | Allowed: rot90, flip, peak-aware cutout. The 224 crop is fixed; check that the peak-centred crop keeps its peak and that the cutout margin holds (I6) |
| 2. Receptive-field leakage | Adapted | MAE form: masked patches never reach the encoder; the loss covers masked patches only; invalid pixels are weighted out via `valid_mask`; `peak_aware_masking` keeps its budget |
| 2. Pretext–downstream alignment | As written | Does the reconstruction loss, including `norm_pix_loss`, preserve Bragg peaks |
| 3. Leakage across sensors | As written, 🔴 | Splits by detector and beamtime session (I9) |
| 3. Sensor-agnostic normalisation | Part design note | Hard-coded constants are flagged as written. Requiring median/MAD conflicts with the fixed per-frame GCN: design note |
| 3. Domain adaptation monitoring | Design note | Conflicts with the single detector-agnostic model. Check instead that every result reports per-detector metrics |
| 4. Metric sanity | As written | Accuracy as headline or selection metric is 🔴. ROC-AUC is flagged only as headline or selection metric (`CLAUDE.md` requires logging it). Recall at fixed FPR: pre-existing gap until the metric lands |
| 4. Batch imbalance | As written | Training crops are balanced by `hit_frac`; plain cross-entropy. Check the realised hit fraction against the config; the evaluation-set composition is stated |
| 4. Deterministic inference | As written | `model.eval()`; no batch-norm updates at evaluation; stable tile-score tie-breaking |
| 5. I/O bottlenecks | As written, 🟡 | Exempt: the per-frame HDF5 open in `__getitem__` (I8); no inference-latency findings before Phase 7 (I17) |

## Applies only if introduced

These checks are dormant. They wake up when a diff introduces the thing they are about.

| Check | Wakes up when | Then |
|---|---|---|
| 1. Correlated noise / blind-spot masking | A blind-spot denoiser (Noise2Void, Noise2Self) appears | Apply the appendix text as written |
| 2. Collapse checks | A joint-embedding method (SimCLR, BYOL, Barlow Twins) appears | Apply the appendix text as written |
| 5. Mixed precision | `autocast` or `GradScaler` appears in `src/` | 🟡 unless the diff shows that faint peaks survive (a test or a measured comparison) |

## Output format

```
Verdict: no blocking findings | blocking findings

🔴 BLOCKING
- <file>:<line> — <title>. Rule: <rule or category>. Shape/dtype: <where relevant>.
  Evidence: <what in the diff shows it>. Fix: <proposed change>.

🟡 WARNING
- (same fields)

🟢 NIT (at most 3)
- <file>:<line> — <text>

Design notes
- <guideline> conflicts with <CLAUDE.md rule>: <one sentence>. For discussion.

Pre-existing gaps
- <gap> (not caused by this diff)

Coverage
- Checked: <categories and directives applied>
- Dormant: <checks not applicable to this diff>
```

Empty sections are written as "None."

---

## Appendix — the user's guidelines (verbatim source)

### ML Code Reviewer Guidelines

**Severity Classification**

- 🔴 **BLOCKING**: Calibration leakage, pixel mask smearing, sensor-split data leakage, or evaluation using plain Accuracy on rare events.
- 🟡 **WARNING**: Inefficient per-frame loops in dataloaders, unvalidated mixed-precision underflow, or RGB-oriented augmentations on grayscale sensor data.
- 🟢 **NIT**: Variable naming, typing annotations, or non-critical docstring formatting (limit to max 3 per review).

**Core Verification Directives**

1. When reviewing changes under `models/` or `data/`, trace data arrays from raw sensor inputs through calibration masks to verify no invalid pixels leak into convolutional neighborhoods.
2. Demand unit tests for any new loss function or mask generator ensuring the output shape and blind-spot masking match mathematical formulations.
3. Require authors to cite line numbers and tensor shapes when proposing normalization or feature-scaling transformations.

**1. Sensor Calibration & Preprocessing Integrity**

- Calibration Order of Operations: Ensure baseline subtraction, dark current pedestal removal, and flat-field/gain calibration happen before normalisation, clipping, or tensor conversion.
- Bit Depth & Saturation Invariance: Flag any casting to uint8 or unscaled float32. Ensure raw high-dynamic-range sensor data (14-bit, 16-bit, or signed scientific integers) is preserved, with dedicated handling for saturated pixels and dead-pixel mask propagation.
- Correlated vs. Independent Noise Checks: In self-supervised blind-spot denoising (e.g., Noise2Void / Noise2Self), verify the noise model assumptions. If sensor readout noise exhibits structured patterns (e.g., column/row striping or 1/f noise), reject naive single-pixel blind-spots and require structured/doughnut masking schemes.
- Pixel Mask Leakage: Reject interpolation, bilinear resampling, or convolution operations on unmasked raw frames that smear invalid/bad detector pixels into valid data prior to inference.

**2. Self-Supervised Learning (SSL) Architecture & Invariants**

- Single-Channel Physics vs. RGB Anti-Patterns: Reject standard natural image augmentations (such as ImageNet color jitter, random hue/saturation, or aggressive cropping that truncates sparse signals). Augmentations must respect physical sensor invariances (e.g., Poisson noise injection, background scaling, or rotational symmetries).
- Representation Collapse Checks: For joint-embedding architectures (SimCLR, BYOL, Barlow Twins), ensure explicit protections against collapse: non-learnable momentum teacher branches, stop-gradients, or cross-correlation loss formulations.
- Blind-Spot & Receptive Field Leakage: For autoencoding or self-prediction approaches, inspect kernel sizes and receptive fields to verify the target coordinate does not leak through receptive field overlap in central receptive paths.
- Pretext-Downstream Alignment: Ensure pre-training objectives do not suppress subtle high-frequency peaks or diffuse anomalies as "noise" when they are actually the rare events intended for detection.

**3. Cross-Sensor Robustness & Domain Shift**

- Data Leakage Across Sensors: Verify that train/val/test splits are split strictly by sensor hardware, run, or acquisition session, never by random frame splitting within the same capture session.
- Sensor-Agnostic Normalization: Flag hardcoded min-max normalization or global dataset-wide constants. Require dynamic robust estimators (e.g., median/MAD, percentile clipping, or per-detector calibrated normalization).
- Domain Adaptation Monitoring: In downstream feature layers, check whether feature distributions are explicitly conditioned or regularized across sensor IDs (e.g., via domain adversarial loss, gradient reversal layers, or sensor-specific adapters).

**4. Rare-Event Imbalance & Evaluation Rigor**

- Metric Sanity Checks: Flag any reliance on standard Accuracy, standard ROC-AUC, or balanced accuracy on extreme-imbalance datasets. Require PR-AUC (Average Precision), calibrated thresholding, and recall at fixed false positive rates (e.g., Recall @ 10⁻³ FPR).
- Batch Imbalance Handling: In downstream validation or contrastive batch formation, ensure rare positive occurrences are not washed out in uniform batching. Check for stratified sampling or focal loss implementations.
- Deterministic Inference: Check that calibration offsets, running batch-norm statistics, or test-time augmentations do not drift during streaming high-volume batch evaluation.

**5. Execution Performance & Pipeline Efficiency**

- GPU Memory & I/O Bottlenecks: High-throughput streaming image pipelines frequently choke on CPU-side decoding. Ensure image decompression and calibration array operations leverage vectorized GPU execution (or optimized multithreaded readers) rather than per-frame Python for-loops.
- Gradient Accumulation & Precision: Verify whether mixed-precision (`torch.amp.autocast`) maintains sufficient dynamic range for faint scientific features, ensuring loss scaling does not result in underflow on small rare-event loss signals.

**Calibration leakage — the two kinds (user's explanation, 2026-11-11)**

1. *Detector calibration leakage (hardware and physics level).* Dark/pedestal drift: a pedestal or dark-current matrix averaged over a run or shift that contains both training and test data gives the model a static hardware fingerprint across splits. Bad-pixel/hot-pixel masking: masks computed over the whole experimental block, not on calibration runs or training partitions, make detection thresholds artificially easy. Flat-field and geometry maps: panel shifts, beam centres or detector distance must never be fitted using hit frames from the test fold. Why it matters: if the test fold's sensor artifacts bleed into calibration tables used in training, the network can overfit to panel gain patterns or readout noise signatures instead of learning invariant representations of the rare events.
2. *Statistical / ML metric leakage (evaluation level).* Threshold tuning on test sets; global preprocessing statistics computed over the whole corpus before the split; evaluating the threshold on frames from the same physical run or sensor mount as the training frames.
