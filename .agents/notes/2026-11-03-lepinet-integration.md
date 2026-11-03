# Lepinet recipe in mini_trainer: parity audit, plan and protocol

**Status:** Phase 0 (read-only audit) done 2026-11-03; no source changes yet.
**Branch:** `feature/lepinet-integration` in `.worktrees/lepinet-integration`, base `4477e1a`. Local only:
no push until the owner says so. Own venv: Python 3.14.3, torch 2.12.0+cu130 (MT lock); lepinet runs
torch 2.12.1+cu130, torchvision 0.27.1, fastai 2.8.7.
**Scope:** reproduce lepinet's 20 M EfficientNetV2-S baseline (heads F, BU, I) in MT as opt-in
components. MT defaults and the Nemo recipe (`dev/ucloud/production.py:50-84`) stay unchanged.
Supervised by the lepinet session (`lepinet-e1`).
**References:** MT paths are relative to this repository at `4477e1a`; `L:` is
`/home/au761367/codes/lepinet` at `a007ab2`; `FA:` is fastai 2.8.7 in lepinet's venv
(`L:.venv/lib/python3.14/site-packages/fastai/`).

## Decision and reason

- **Reference implementation:** lepinet's `src/lepinet` package. Its clean port and the older dev/030
  trainer agree within 0.0007 species macro-F1 at every epoch; that is the reproducibility to aim for.
- **Gate (owner, 2026-11-03):** mini_metrics' current convention at threshold 0, with the calibrated
  threshold as a second column. Test = set 0 restricted to the 12,041 training species (629,742
  images). Species / genus / family: F 0.9135 / 0.9606 / 0.9739, BU 0.9135 / 0.9633 / 0.9778,
  I 0.9110 / 0.9587 / 0.9708. The older 0.9148 and 0.9152 used per-rank `--optimal` thresholds.
  Micro-accuracy is not a gate.
- **Setting:** parity runs use lepinet's 50-image floor (12,041 species). Everything after the gate
  moves to MT's test set (no floor, 12,632 species).
- **All three heads exist in MT.** I = `mt_htrain --head independent`. BU = `--head hierarchical`
  (the default): species logits are log-sum-exp'd to parents (`mini_trainer/hierarchical/model.py:118-126`),
  which equals lepinet's `MarginalHead` because cross entropy ignores the per-row log-softmax shift.
  F = `--head hierarchical --loss_weights 1 0 0`: species loss only, and the same class marginalises at
  inference with lepinet's argmax and confidence. Train F this way: `mt_hpredict --head` cannot
  marginalise a flat `Classifier` checkpoint because the stored class wins
  (`mini_trainer/modeling/classifier.py:366-370`).
- **Shape:** opt-in parameters on existing modules and builders, each defaulting to today's
  behaviour; one `HierarchicalBuilder` subclass switches them on; YAML presets select it by import
  path (`mini_trainer/config.py:127`). Margin heads are not ported (owner). Trap augmentation and
  per-row sampling weights follow in phase 2.

## Evidence and limits

### Findings that correct the brief

1. **Lepinet's "frozen" head parameters train.** `vision_learner` calls `freeze()`, `_fit` calls
   `unfreeze()` (`L:src/lepinet/train.py:190`), and fastai's `freeze_to` sets `requires_grad_(True)` on
   every tensor in its optimiser groups (`FA:optimizer.py:25-40`, `FA:learner.py:661-670`), which
   include frozen tensors. Each level's bias and `weight_norm` g therefore train under AdamW with
   wd 0.01. Measured in the dev/030 and the clean-package I checkpoints alike: species g 1.60-2.62
   (mean 1.88), bias -0.92 to 0.17 (mean -0.30). Both are higher for rare species (Spearman against
   log training count: g -0.38, bias -0.34), so the bias is not a log prior. MT excludes frozen
   tensors from optimisation (`mini_trainer/builders.py:243`) and keeps them at 1 and 0.
2. **Label smoothing is on in lepinet.** `label_smoothing=None` resolves to 1/C at species (8.3e-5),
   shrunk per level (`L:src/lepinet/loss.py:94-95,110-111`); no reference config sets it. MT's
   hierarchical default is identical (`mini_trainer/hierarchical/integration.py:216-221`), so this is
   not a difference. fastai's `valid_loss` uses the same smoothed loss.
3. **At 20 M lepinet is plain AdamW.** MT's `MuonAuxAdamW` gives Muon exactly one tensor, the head's
   1280×1280 hidden layer (measured, row 10).
4. **"1:35 per epoch" is fastai's h:mm:ss:** 1 h 35 min on the shared 5090 (5,039,220 train plus
   630,097 validation images, about 990 img/s). Clean package on a B200: 1 h 14 min to 1 h 22 min,
   about 1,200 img/s.
5. **F's species curve is usable.** `f1_speciesKey` in F's log postdates the single-level metric
   fix (lepinet 6021a38 at 18:26; the run is stamped 18:27). Only `F1(macro)` and `F1(micro)` are
   broken there (fixed in 9fb765c).

### Component diff

"Knob" = existing MT option; **new** = opt-in component to add. Line references are at the
revisions above.

| # | Component | lepinet recipe | MT master default | To reproduce in MT; expected effect |
|---|---|---|---|---|
| 1 | Split, classes | Drop set 0, keep species with ≥ 50 images in sets 1-9 (`L:src/lepinet/data.py:127`, floor `:60-62`): 12,041 / 4,333 / 102. Train sets 2-9 (5,039,220), validation set 1 (630,097). Vocabularies are sorted key strings (`L:src/lepinet/train.py:34`). | Same split (`mini_trainer/integrations/parquet.py:57-65`). No floor: every parquet species, 12,632 (`:209-225`). Species of other classes become `None` labels, not dropped (`:174`). | **New** floor counted over non-test sets, applied to the class spec and train/val rows, plus a vocabulary restriction for test rows (never a count floor on the test fold). Defines the class set. Class order differs, which is metric-neutral. |
| 2 | Decode, resize | PIL decode, no EXIF transpose (`FA:vision/core.py:98-103`). `Resize(460)`: square crop of side min(w, h), random position in training, centred in validation, then PIL bilinear with antialias (`FA:vision/augment.py:244-279`, `:129-141`). | `decode_image`, no EXIF, squash to S×S with NEAREST (`mini_trainer/data/io.py:143-171`; `mini_trainer/data/loader.py:172`). Then `preprocess` on GPU, torchvision `ImageClassification`: resize to S·256/224, centre-crop S (`mini_trainer/modeling/architectures/torchvision.py:36-45`; applied at `mini_trainer/trainer.py:120,197`). Measured at `--size 256`: only source columns 16-239 survive, an 87.5 % field of view. S defaults to 384 with no crop, but an explicit `--size 384` crops (438→384). | **New** reader option: aspect-preserving square crop (random/centre by split), antialiased bilinear to 460; **new** preprocess without the crop. Moderate to large: aspect distortion, NEAREST aliasing and the lost border remove signal, and lepinet's random crop is an augmentation MT lacks. |
| 3 | GPU augmentation | `aug_transforms(size=256, flip_vert=True, max_rotate=15, max_zoom=1.1, no warp, no lighting)` (`L:configs/20260724_ucloud_lepinet_5ep_oversample.yaml:33-39`). Per image: dihedral p 0.5 over 8 elements including identity, so 56 % identity; rotation U(±15°) p 0.75; zoom U(1, 1.1) p 0.75 with a random centre (`FA:vision/augment.py:1259-1262,656-670,715-727,760-783`). Composed into one affine and sampled once to 256 with bilinear `grid_sample`, reflection padding, `align_corners=True` (`:491-506,379-393`). The area pre-filter needs > 2× (`:354-366`), so 460→256 is plain, aliased bilinear. Validation: identity affine, same resample. | torchvision v2 `Compose` of salt-and-pepper (0.001-0.05, p 0.75), h-flip, v-flip, rotation 15, affine (translate 0.1, scale 0.9-1.1) on the uint8 batch (`mini_trainer/builders.py:176-194`). One draw per call, i.e. per batch: measured, the whole batch flips and rotates together; salt-and-pepper draws one Bernoulli per batch (`mini_trainer/data/augmentation.py:66-67`). | **New** per-image affine (dihedral ∘ rotate ∘ zoom, one `grid_sample` to 256) and a preprocess that resamples 460→256 only when needed (validation), then normalises. Large together with row 2: light per-image augmentation was part of a +4.7 pt bundle in lepinet (`L:docs/design-decisions.md:16-17`). MT's batch-level draws add less diversity per step, and it adds noise and zoom-out that lepinet lacks. |
| 4 | Normalisation | ImageNet mean/std on float /255 (`L:src/lepinet/data.py:207-208`; `FA:data/transforms.py:348-378`). | Same statistics via the torchvision transform. | Nothing. None. |
| 5 | Sampler | Row weight n_c^-0.5 on the training split (`L:src/lepinet/data.py:143-157`), drawn with replacement, epoch length kept (`FA:callback/data.py:21-31`). Measured: the rarest 10 % of classes get 4.0 % of draws (1.2 % naturally), and each epoch shows about 58 % of distinct images. | `RandomSampler`, drop_last (`mini_trainer/data/loader.py:101-118`). `resample` raises (`:184-185`). `Reindexed*` (`mini_trainer/data/io.py:264-354`) is an unused deterministic oversampler, not lepinet's. | **New** weighted sampler: per-row weights with a class-power helper, with replacement, length = training set, distributed variant (phase 2 reuses it for self-training). Large: +1.9 pt species in-distribution (no oversampling 0.8949 vs √ 0.9135, zero convention). |
| 6 | Batch size | 64 (`L:src/lepinet/config.py:54`). | 16 (`mini_trainer/train.py:67`). | Knob. Large if left at 16. |
| 7 | Backbone | torchvision `efficientnet_v2_s(pretrained=True)`, i.e. IMAGENET1K_V1 (no fastai `model_meta` entry, `FA:vision/learner.py:127,165-176`), cut to `features` (`:81-87`), pooled in the head (`L:src/lepinet/heads.py:282-288`). BN in train mode; torchvision's stochastic depth 0.2. | Same weights (`mini_trainer/modeling/architectures/torchvision.py:28-47`); torchvision's `classifier` replaced by the head, so the same graph. | Nothing on one GPU (MT converts BN to SyncBN under DDP, `mini_trainer/trainer.py:281`). None. |
| 8 | Head | Dropout 0.1 → Linear 1280 → LeakyReLU → L2; per level `weight_norm` cosine → z-score + bias (`L:src/lepinet/heads.py:131-210`). Built with bias 0 and g 1 frozen (`:181-197`) but trained in practice (finding 1). Whole head in fp32 with autocast off (`:285-288`), no TF32. Spherical init step `lr` (`:127`). | Same structure and math (`mini_trainer/modeling/classifier.py:24-234`; `mini_trainer/hierarchical/model.py:153-226`), plus an unused BatchNorm1d. Bias and g frozen and never optimised. **No log prior**: a prior applies only when passed explicitly (`mini_trainer/modeling/classifier.py:135-138`; `mini_trainer/hierarchical/model.py:33-39`), and the train-label prior path is unreachable from `mt_train` (`mini_trainer/modeling/classifier.py:301-304`). **Not fp32**: the hidden and prototype matmuls run in fp16 under autocast (no guard, `:214-234`; `acos` autocasts to fp32), and TF32 is on (`mini_trainer/utils/_core/device.py:55`). Spherical step `lr/num_classes` since d1a9cc9 (`mini_trainer/modeling/classifier.py:45`). Measured: eigenvalues of WᵀW/(N/E) 0.48-1.67, max\|cos\| 0.146, practically random init; lepinet's step gives 0.02-2.52 and 0.219. | **New** `Classifier` options, stored in metadata: trainable scale and bias, fp32 head (autocast off; TF32 off for the head), legacy spherical step. Learned g and bias: unknown, possibly large; they widen the logit range and sharpen rare classes. fp32: small for accuracy; it removes fp16 cosine rounding (about 0.02 in logits) and overflow risk once g > 1. Init: small, first epoch only. |
| 9 | Loss | Per-level CE summed, smoothing 1/C (finding 2), no regulariser. BU = `MarginalHead` (`L:dev/050_hierarchical_heads.py:85-120`). | Same CE and smoothing for `mt_htrain` (`mini_trainer/hierarchical/loss.py:27`). Prototype regulariser 0.1 (`mini_trainer/train.py:77`; `mini_trainer/builders.py:410-421`) on every level's prototypes for I (`mini_trainer/hierarchical/model.py:154-156`). EMLA off (on for Nemo). | Knob `--regularization_strength 0`. Effect of 0.1 unmeasured; 0.001 was a wash in lepinet. |
| 10 | Optimiser | `MuonAuxAdamW` over fastai's body and head groups; the body has no 2-D tensors and the head is forced to AdamW (`L:src/lepinet/optim.py:236-262,275-292`). Net effect: one AdamW (β 0.9/0.999, eps 1e-8), one lr for everything, wd 0.01 on every trainable tensor including BN, biases, g and prototype directions; `OptimWrapper` ignores fastai's `do_wd` (`FA:learner.py:186-200`; `FA:optimizer.py:458-497`). Σ lr·wd over the run = 1.97 (×0.14 decoupled decay). | Measured routing: Muon gets `classifier.hidden.weight` (1.64 M). AdamW: backbone convolutions (19.99 M) at lr/3, wd 0.01; BN and biases at lr/3, wd 0; hidden bias and prototype directions at lr, wd 0. g and bias are frozen (`mini_trainer/builders.py:197-310`; lr/3 at `:287`). | Knobs `optimizer_cls: torch.optim.AdamW` and `backbone_lr: 0.001`; **new** grouping option that decays norms, biases and parametrisations. Moderate: backbone lr/3 and Muon on the hidden layer change update scales; decay on BN, g and directions adds regularisation. |
| 11 | Schedule | Per step, pos = i/N: linear lr/100 → lr over 0.5 epoch (10 %), then cosine to 0; same curve for every group; backbone from step 0 (`L:src/lepinet/schedules.py:19-35`). Advances on AMP-skipped steps. | Head warms up from 1e-4·lr over 2 epochs (`mini_trainer/train.py:78`); backbone at lr 0 during warm-up, then its own cosine; both end at 1e-3· (`mini_trainer/builders.py:424-465`). Advances only on successful steps (`mini_trainer/trainer.py:142-145`). | Knobs `warmup_epochs 0.5, start_factor 0.01, min_factor 0, pretrained_backbone false`: measured equal to lepinet's curve within 1.3e-8 over all 393,685 steps. MT's default is large: over 5 epochs its backbone trains 3 epochs at lr/3 (schedule shape alone was +1.2 pt in lepinet). |
| 12 | AMP | fp16 autocast over forward and loss; `GradScaler('cuda')` defaults 2^16, growth every 2000 (`FA:callback/fp16.py:25-58`; `L:configs/20260724_ucloud_lepinet_5ep_oversample.yaml:26-27`). | fp16 autocast over forward, loss and regulariser (`mini_trainer/trainer.py:119`); scaler 2^14, growth every 100 (`mini_trainer/builders.py:313-316`). | Knob `scaler_builder_kwargs {init_scale: 65536, growth_interval: 2000}`. Small: growing every 100 steps skips about 1 % of steps at equilibrium, against about 0.05 %. |
| 13 | Gradient clipping | 5.0 on unscaled gradients of all parameters (`FA:callback/training.py:38-42`, ordered after `MixedPrecision`). | Same (`mini_trainer/trainer.py:79,137-139`). | Nothing. None. |
| 14 | Epochs | 5. | 15 (`mini_trainer/train.py:48`). | Knob. |
| 15 | Evaluation | Each epoch on set 1: per-level accuracy and argmax macro-F1 over classes present in the truth (`L:src/lepinet/metrics.py:54-94`). Final-epoch weights. Test: set 0, rows restricted to the model's species, missing files skipped (`L:src/lepinet/test.py:347-351,370-376`); marginals by summed probabilities (`:192-203`). | Each epoch on set 1: per-level top-1 (equals lepinet's `acc_*`) and loss, no macro-F1 (`mini_trainer/logging/core.py:564,795-825`). `best.pt` chosen by species acc1, plus `last.pt` (`mini_trainer/trainer.py:343-344`; `mini_trainer/train.py:311-318`). `mt_hpredict` keeps every set-0 row, out-of-vocabulary ones included (`mini_trainer/data/metadata.py:301-303`). | **New** opt-in per-level macro-F1 statistic. Evaluate `last.pt`. Restrict test rows to the vocabulary. Score both codes with one scorer: mini_metrics `61db373` with lepinet's rescoring settings. No effect on the model; needed to compare numbers. |
| 16 | Throughput | fastai loader, `min(16, cpus)` = 16 workers locally (`FA:data/core.py:76`), 256 on UCloud. PIL decode, crop and resize in workers; no pin_memory; GPU transforms in the main process. No channels_last, compile or cache; cudnn.benchmark off (`FA:torch_core.py:30-39` is never called). Workers forced to fork (`L:src/lepinet/data.py:30-44`). | ≤ 16 workers (`mini_trainer/data/_workers.py:89-91`), libjpeg-turbo decode, pin_memory, persistent workers, optional CUDA prefetch, `--compile` (Nemo uses it), RAM/VRAM cache, DDP. Python 3.14 starts workers with forkserver. | The parity reader (crop + antialiased bilinear) will dominate CPU cost. compile and channels_last are speed levers to measure after parity. A full-scale cache is not viable (5.04 M × 460²×3 B ≈ 3.2 TB). No accuracy effect if numerics are unchanged. |

### MT features lepinet lacks (inputs to the recipe choice, goal 3)

- **EMLA** (`mini_trainer/training/loss.py:18-84`; hierarchical `weighted: true`), used by Nemo. In lepinet
  every prior-correcting loss lost under shift; EMLA is untested there.
- **Prototype regulariser:** 0.1 in `train.main` and Nemo; the builder default is 1e-3
  (`mini_trainer/builders.py:410`). Asger's documents disagree on the value.
- **`torch.compile`** (`mini_trainer/trainer.py:287-293`): throughput only.
- **Caching** (`mini_trainer/data/io.py:445-543`): only for subsets that fit in RAM. Do not use it on the
  shared 125 GB workstation (systemd-oomd).
- **DDP:** `DistributedSampler` plus SyncBatchNorm (`mini_trainer/trainer.py:276-285`). The weighted sampler
  needs a distributed variant, and SyncBN changes BN statistics, so parity runs use one GPU at bs 64.
- **EMA:** unsupported (`README.md`, `docs/roadmap.md` §2); keep it off. Lepinet has none.
- **Label smoothing:** 1/C in both (finding 2).

### Verified and not verified

- Verified by reading the code at the stated revisions. Measured on CPU only: MT optimiser routing on
  built models, MT preprocessing crop, batch-level torchvision draws, schedule parity, spherical-init
  geometry, head parameters in two lepinet checkpoints, parquet counts and oversampling statistics.
- Not verified: anything on GPU; fastai's `pretrained=True` path (read, not executed); whether learned
  g and bias matter (no single-factor run exists); pixel equality of PIL and torchvision decoding.

## Remaining work

### Implementation plan (phase 1, this branch)

Every change is an opt-in parameter whose default reproduces today's behaviour, with no default change
to `BaseBuilder`, `HierarchicalBuilder`, `train.main`, the CLI or checkpoints. Each component gets one
focused, normally prefixed commit with its tests.

| Step | Component (location) | Opt-in | Test failure contract |
|---|---|---|---|
| 1 | Class floor and vocabulary filter (`mini_trainer/integrations/parquet.py`, class-spec builders) | `min_class_count`, counted over non-test sets | A species with 49 images in sets 1-9 is excluded however many it has in set 0, and one with 50 is kept; excluded species never reach train/val rows; test rows are restricted only by vocabulary; without the option the class spec is identical. |
| 2 | Weighted sampler (`mini_trainer/data/loader.py`) | per-row `sample_weights` or `class_sampling_power` | Without it the loader is unchanged. With power 0.5 the epoch length equals the training set and per-class draw shares match n_c^0.5 within binomial tolerance. Same seed, same indices. DDP ranks draw disjoint streams with the global length kept. |
| 3 | Head options (`mini_trainer/modeling/classifier.py`) | trainable scale and bias, fp32 head, legacy spherical step | By default g and bias stay frozen and outside every optimiser group; with the option they get gradients and move after one step; state-dict keys and reload round-trip are unchanged. The fp32 head returns float32 logits under CUDA fp16 autocast (CUDA-marked test). The legacy step reproduces the pre-d1a9cc9 init for a fixed seed. |
| 4 | Decay-all grouping (`mini_trainer/builders.py`, `parameter_groups`) | `decay_all` | Every trainable tensor lands in a decayed group; the default grouping is unchanged. |
| 5 | Crop reader (`mini_trainer/data/io.py`) | `resize_method="crop"` | On a coordinate-coded image, training crops are squares of side min(w, h) inside the image at varying offsets; validation crops are centred; the result matches PIL `crop` + `resize(BILINEAR)` within 1 LSB. |
| 6 | Per-image affine and resample-normalise preprocess (`mini_trainer/data/augmentation.py`, `mini_trainer/modeling/architectures`) | new augmentation and preprocess, recorded in classifier metadata for `mt_predict` and export | Images in one batch get independent transforms; p = 0 equals the validation path; rotation ≤ 15°, zoom ≥ 1, reflection (no zero borders); output 256, float, in [0, 1]. |
| 7 | Validation macro-F1 statistic (`mini_trainer/logging/core.py`) | statistic name | On a fixed confusion example it equals lepinet's `LevelMacroF1`: a never-predicted class scores 0, absent classes are excluded. |
| 8 | Recipe builder and presets I, BU, F (builder in the package; YAML in `dev/lepinet/`) | `builder:` import path | CPU end-to-end on a tiny synthetic parquet for each head; checkpoint reload; F predicts genus and family as marginals. |

Presets also set existing knobs: `--batch_size 64 --epochs 5 --size 256 --dtype float16`;
`optimizer_cls: torch.optim.AdamW`, `lr: 0.001`, `weight_decay: 0.01`, `backbone_lr: 0.001`; the
schedule knobs of row 11; the scaler knobs of row 12; regulariser 0; `weighted: false`; F adds
`weights: [1, 0, 0]`. Validation: `bash dev/check.sh static` and `bash dev/check.sh all` (the change spans
training, loading and checkpointing). No new dependency: fastai stays out of MT.

Order: the GPU-free equivalence checks (protocol a0) first; then steps 1-4 (small); then 5-6 (the
data path, largest); then 7-8 and the full checks; then protocol (a) and (b).

### Parity protocol

Paired-comparison rules follow [docs/training-feature-validation.md](../../docs/training-feature-validation.md).

**(a0) GPU-free equivalence, CPU, minutes.**
- Head: load the clean-package I checkpoint
  (`L:data/local_models/5ep_baseline/lepinet-heads-global-independent-muon-5ep-oversample-effnetv2s-ucloud.pt`)
  into MT's `IndependentClassifier` by key remap (`0.0.*` → `features.*`; `1.head.*` → `classifier.*`;
  layer 0 is `linear`). Require identical fp32 logits on one batch, then the same check through
  `HierarchicalClassifier` for the marginals.
- Data: pass the same 64 validation images through lepinet's `dls.valid` and MT's reader and
  preprocess; require ≤ 1 LSB before normalisation. For training transforms compare the drawn
  parameters (flip, angle, zoom, centre), not pixels.
- Optimiser: identical per-tensor group membership, lr, wd and betas for 1,000 steps.
- Scorer: re-score lepinet's I `predictions.csv` and reproduce 0.9110 / 0.9587 / 0.9708 before scoring MT.

**(a) Subset side-by-side on the local RTX 5090** (accuracy only; timings are biased by the other user).
- Subset: apply the 50-image floor, then MT's frozen family × abundance selector
  (`publication/experiments/training_ablations/data.py:40-54`), 512 species, seed 20261003. Both codes read
  the same parquet rows; about 214 k training images, to be measured.
- Recipe as in the full run, 5 epochs, two seeds per code (three if the spread is wide). Compare per-epoch
  validation accuracy and macro-F1 per level, and the subset's test fold. A shared seed value does not
  make trajectories comparable across codes; compare against the within-code spread.
- Pass: per-epoch code means within max(2 × seed spread, 0.005). On failure, change one component at a
  time towards lepinet, in order of expected impact: head scale and bias, sampler, reader and augmentation,
  grouping, precision.
- Host memory: at most 8 workers per code, never `--cache`, one job at a time, peak RSS logged.
- A match shows no implementation difference visible at 512 species; it does not prove the full-scale match.

**(b) Full scale, the acceptance gate (UCloud).**
- One B200 per run, single GPU, bs 64, fp16, 5 epochs, lepinet's floor; MT presets I, BU and F, in parallel.
- Test: species at threshold 0 within ±0.003 of the references (retraining noise on the test fold is
  about 0.000, so 0.003 is a real gap); genus, family and the calibrated column reported against the same
  tolerance.
- Curve: per-epoch validation macro-F1 against the clean-package logs in
  `L:data/ucloud_models_logs/` (target ≤ 0.001, investigate > 0.002):
  - I (`20260724-181230-…`): species 0.7848, 0.8350, 0.8697, 0.8985, 0.9099.
  - BU (`20260730-074913-…`): species 0.7856, 0.8384, 0.8691, 0.8968, 0.9133; genus 0.8758 → 0.9614;
    family 0.9237 → 0.9794.
  - F (`20260729-182718-…`): species 0.7978, 0.8375, 0.8717, 0.9002, 0.9128; `valid_loss` 0.6995 →
    0.2834 (smoothed CE, as MT logs it).
- Throughput: one dedicated B200 job running both codes back to back on the same data, with matched worker
  counts and each code's best count, cold and warm reads. Report train img/s, epoch time including
  validation, and GPU utilisation.
- Process: run `L:ucloud/budget_check.py <spec>` before every submission; check `ucloud q tick; echo $?`;
  read job logs, not job status; send `lepinet-e1` every spec path and job name. Code reaches UCloud by
  ucloud-api `[sync]` or a worktree tarball (`L:ucloud/README.md`). Do not touch lepinet's running V0F
  (12444405) and V0BU (12444406); they also show whether parallel `/work` readers starve each other
  (`L:ucloud/stage.py` if they do).

### Later phases

- **Phase 2:** self-training on trap crops (per-row weights; mount `flemming`); classifier re-training on a
  frozen backbone (MT's `fine_tune` casts the backbone to bf16 and leaves BN statistics updating,
  `mini_trainer/builders.py:68-76`); trap augmentation (`L:src/lepinet/augment.py:33-111`); open-set and
  back-off evaluation; trap calibration grouped by (trap, night). F needs no new code (see above).
- **Phase 3 onwards:** single-factor runs on MT's test set (EMLA vs CE, oversampling vs none, schedule,
  augmentation, image size, epochs, learned vs frozen head scale and bias); paper reruns; Nemo vs this
  recipe on the Flemming trap crops; the production model.

### Open questions

1. Learned g and bias are part of lepinet's recipe as run, so parity reproduces them. Should the recipe
   choice also test frozen vs learned as a single factor?
2. Lepinet's preprocessing (460 crop, then aliased 460→256) becomes part of the model interface. Keep it
   for production, or decide after the gate? MT must record it for `mt_predict` and ONNX export either way.
3. Where should the recipe builder and presets live (package module vs `dev/lepinet/`)? This is Asger's call.
4. Throughput target: equal img/s, or better img/s at equal accuracy?
5. Gate tolerances above (curve ≤ 0.001 target / 0.002 alarm; test ±0.003): agreed?
