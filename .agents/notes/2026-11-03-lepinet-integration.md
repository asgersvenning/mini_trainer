# Lepinet recipe in mini_trainer: parity audit, plan and protocol

**Status:** 2026-11-03: Phase 1 implemented (`a3826c3`..`491b7a5`, presets and usage in
[dev/lepinet/README.md](../../dev/lepinet/README.md)); GPU-free checks (a0) passed; protocol (a) accepted
by lepinet-e1 (lepinet RESULTS V1). 2026-11-04: gate (b) and V2T ran from `5099cc0` (jobs 12445754-6,
12445759). 2026-11-05: **the gate passed** for I, BU and F (below, "Gate result"); the paper campaign has
started (section "Campaign").
**Branch:** `feature/lepinet-integration` in `.worktrees/lepinet-integration`, base `4477e1a`. Pushed to
`origin` (asgersvenning/mini_trainer) on 2026-11-05 at the owner's request, without a PR; ask before
later pushes and before opening the merge PR (trim words first). Own venv: Python 3.14.3, torch 2.12.0+cu130 (MT lock); lepinet runs
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
- **Tolerances (agreed 2026-11-03):** per-epoch validation macro-F1 within 0.001 of lepinet's
  clean-package logs (investigate above 0.002) at every rank with a curve; test at threshold 0
  within ±0.003 at species, genus and family, calibrated column reported. Throughput gate: same or
  better img/s at identical settings (bs 64, 256 px, one B200, same staged data, same job); compile,
  channels_last and batch size are later single factors with an accuracy check.
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

### GPU-free equivalence (a0), 2026-11-03

Scripts and outputs: ignored `.agents/local/lepinet-a0/` (lepinet's venv for references).

- Backbone initial weights: all 780 tensors of fastai's `create_vision_model(pretrained=True)` equal MT's.
- Head: the clean I checkpoint loaded into MT's `IndependentClassifier` gives identical fp32 logits
  (max Δ 0) at all ranks; MT's `HierarchicalClassifier` marginals match lepinet's (max Δp 1.4e-5,
  from lepinet's +1e-12 in the log; argmax 100 %).
- Data: PIL and `decode_image` agree on 3,064 images; MT's crop reader is 0 LSB from fastai's crops;
  `ResampleNormalize` and path-based `Preprocess` equal fastai's validation batch exactly. Given fastai's
  drawn matrices, MT's training resample equals fastai's batch exactly; MT's draw distribution matches
  fastai's (6 matrix entries, KS ≤ 0.010 against a 0.0195 critical value).
- Optimiser: all 461 tensors of lepinet's real learner match MT's in trainability, shape, lr, decay,
  betas and eps (MT adds only the head's unused BatchNorm).
- Scorer: lepinet's `dev/093` `score()` reproduces I at threshold 0 (0.9110 / 0.9587 / 0.9708).
- Images: median short side 384 px, maximum long side 512; 86 % are below 460, so `Resize(460)` mostly
  upsamples before the aliased 460→256 resample. The learned g acts as a per-class temperature
  (z ≈ √(D−2)·g·cos); no logits sit at the arccos clamp.
- lepinet's clean package keeps each batch alive through a reference cycle without its dormant
  `GCCallback`: +0.23 GiB allocated per batch on the 5090 (OOM within ~100 batches next to another
  6 GiB process); with it, a flat 6.8 GiB peak, as MT. Local lepinet runs add `GCCallback` through
  `train_from_config(..., extra_cbs=...)`; it does not change numerics.

### Verified and not verified

- Verified by reading the code at the stated revisions. Measured on CPU only: MT optimiser routing on
  built models, MT preprocessing crop, batch-level torchvision draws, schedule parity, spherical-init
  geometry, head parameters in two lepinet checkpoints, parquet counts and oversampling statistics.
- Not verified: anything on GPU; fastai's `pretrained=True` path (read, not executed); whether learned
  g and bias matter (no single-factor run exists); pixel equality of PIL and torchvision decoding.

## Remaining work

### Implementation (phase 1, done)

Opt-in components with their test contracts, in commit order: square-crop reader and weighted epoch
sampler (`a3826c3`), per-image affine augmentation (`d967200`), head options and square-crop
preprocessing (`b806134`), class floor, unknown-row filter and `decay_all` (`ebfdbb9`), validation
macro-F1 (`b38766e`), `LepinetBuilder` with the I/BU/F presets and `dev/lepinet/score.py`
(`491b7a5`). The full suite passed on that tree (909 passed, 247 skipped, 1 strict EMA xfail).
The recipe and its commands are maintained in [dev/lepinet/README.md](../../dev/lepinet/README.md).

### Parity protocol

**(a) Subset side-by-side, local RTX 5090**, done 2026-11-03 and accepted by lepinet-e1 without a third
seed pair: 34 of 35 validation cells and every set-0 rank within the rule (accuracy only; another user's
process shares the GPU, so timings are biased). Epochs are counted 1-5 here; lepinet's CSVs and MT's
`summary.csv` count from 0.
- Subset: the 50-image floor, then MT's frozen family x abundance selector
  (`publication/experiments/training_ablations/data.py:40-54`), 512 species, seed 20261003: 448
  genera, 40 families, 218,756 training images, 27,354 in set 1 and 27,349 in set 0.
- I head, 5 epochs, alternating lepinet (with `GCCallback`) and MT runs, two per code; 8 workers;
  set 0 of the subset scored by the shared `score()` (zero, 5 calibration seeds).
- Pass: per-epoch code means within max(2 x seed spread, 0.005), the spread being the larger of the two
  codes' seed ranges at that epoch; on failure change one component at a time towards lepinet, in order
  of expected impact.
- Validation, final epoch, MT minus lepinet (code means): macro-F1 species -0.0008, genus -0.0020,
  family +0.0068 (seed ranges 0.0013, 0.0019, 0.0050); accuracy -0.0007, -0.0001, +0.0001; loss -0.0017.
  The one cell outside the rule is family macro-F1 in epoch 2 of 5: +0.0111 against 0.0089, MT higher,
  over 40 families whose lepinet seeds differ by up to 0.0079 in epoch 3. In epoch 2 MT leads on every
  F1 and its loss is 0.035 lower, both MT values below both lepinet values: with two seeds per code that
  is the strongest result possible, one-sided permutation p = 1/6, and after epoch 1 lepinet's own seeds
  differ by 0.128 in loss and 0.018 in species F1. In epochs 3-4 MT trails on species and genus F1 by
  0.0014-0.0041, within the rule. (b) decides: at full scale two lepinet I runs agree within 0.0007 per
  epoch, so a real lead of a few thousandths will show there.
- Ruled out as causes of the epoch-2 lead: head initialisation (lepinet passes `init=None` to
  `vision_learner`, `L:src/lepinet/model.py:155`, so the hidden layer keeps PyTorch's default in both;
  species spherical step 0.5, genus and family default, as in MT), the update rule (lepinet's AdamW path
  is `torch.optim.AdamW`, `L:src/lepinet/optim.py:29,190`) and, checked by lepinet-e1, the schedule (MT's
  logged per-epoch mean lr 7.468e-4, 8.755e-4, 5.855e-4, 2.556e-4 against lepinet's analytic 7.474e-4,
  8.753e-4, 5.851e-4, 2.551e-4 for epochs 1-4; lepinet's ParamScheduler drives lr only, not momentum).
- Set 0, identical 27,349 images in all four prediction files, macro-F1 at threshold 0: lepinet
  0.9648 / 0.9604 species, MT 0.9631 / 0.9619. MT minus lepinet: species -0.0001, genus +0.0007,
  family +0.0004; calibrated +0.0016, -0.0012, +0.0004. Within the gate's ±0.003 at every rank.
- Throughput (biased, see above): training phase lepinet 5:20-5:22 per epoch (681 img/s), MT
  4:54-4:56 (about 743 img/s, from its progress clock; first epoch 5:10); epoch with validation 5:37 against
  4:56-4:58. MT's logged `item/s` (about 900) starts its timer after the batch arrives, so it excludes
  data wait: report wall-clock img/s in (b). Peak allocated GPU memory MT 6.56-6.64 GiB, lepinet with
  `GCCallback` 6.8 GiB (probe). MT exits about 80 s after saving `last.pt` in both runs, outside the
  epoch times.
- Exit delay (separate, low priority): building the preset's real subset loaders on CPU and deleting them
  takes 20.2, 40.3 and 80.3 s at 2, 4 and 8 workers, i.e. 5 s per persistent worker per loader, the
  DataLoader's join timeout (`MP_STATUS_CHECK_INTERVAL`) expiring for every worker; the synthetic data of
  the integration test exits in under 1 s at the same counts. Why the workers miss the timeout is open.
  Expect about 5.3 min (2 x 32 x 5 s) between "Final weights saved" and the predict step in each gate
  job; much longer would be a hang. Probes: session scratchpad `exit_delay/probe_real.py`.
- Working files: ignored `.agents/local/lepinet-a/` (`run_a.sh`, `compare_a.py`, logs and predictions
  under `runs/`).

**(b) Full scale, the acceptance gate (UCloud B200).**
- `/work` is not a 1,000 img/s ceiling (corrected by lepinet-e1): V0's slow first epoch most likely
  overlapped a 206 GB download from the same drive; its epochs 2-4 run at about 1,000 img/s per job,
  2,000 together, and `stage.py` moved 5,669,317 files (450 GB) in 12.6 min with 512 readers
  (`L:journal/engineering/2026-07-18-ucloud-throughput.md:37`). Staging is still right for three gate
  jobs next to V0's evals and costs about 15-25 min per job from submission to verified staging.
- Order (lepinet-e1, revised 2026-11-03): once the owner approves, submit mt-gate-I, mt-gate-BU, mt-gate-F
  and mt-v2t together; each stages with `--readers 256` (not 512) so four stagers share `/work` with V0's
  evaluations (30-50 min each). Each job's sync uploads the worktree as it is at launch (`.gitignore`
  respected, deletions not propagated), so tracked code must not change until all four have launched;
  the jobs also import it from the drive until predict ends (see "Submission").
- Submission (2026-11-04, after the owner's approval in this session): `budget_check.py` ok for all four
  (GPU product, not CPU-gated); the first submit synced 495 files from a clean `5099cc0` to
  `/12347837/repos/mini_trainer-lepinet`; all four RUNNING by 13:51 and `ucloud q tick` exits 0. Jobs:
  mt-gate-I 12445754, mt-gate-BU 12445755, mt-gate-F 12445756, mt-v2t 12445759; lepinet-e1 has the list.
  - The user crontab runs `ucloud q tick` every 5 min (`~/.ucloud-tick.log`, which only logs events),
    so auto-extend works without a daemon. Queue records: `~/.local/share/ucloud-api/queue/<name>.json`.
  - `uv sync` installs mini_trainer editable (a `.pth` to the project root), so each gate's predict,
    about 8 h after launch, imports the code then on the drive. Nothing may sync to that remote until
    the three predicts finish.
  - First log check (13:52): all three gates passed preflight (torch 2.12.0+cu130 on a B200) and
    `stage.py` listed 5,669,317 images, as for lepinet; V2T was copying lepinet's source.
  - Staging (14:14): every job staged 5,669,317/5,669,317 files (450.3 GB, 0 errors) in 12.7-15.1 min;
    each gate then staged set 0 (632,913/632,913) in 1.5-1.7 min. Training started 14:06:39 (BU),
    14:08:31 (F) and 14:09:24 (I); the first-minutes ETA was 5.7-6.3 h for 5 epochs with validation.
    V2T: cgroup 48 CPUs, so MT_WORKERS=44; L1 started 14:12:08. "Free on /tmp" fell from job to job
    (953, 802, 701, 455 GB), so the four jobs probably share one node's disk: V2T then measures under
    the gates' load. lepinet-e1 was told staging is done.
  - Counts from the parquet (2026-11-04): 12,041 floor species, 5,039,220 training rows (sets 2-9) and
    630,097 validation rows (set 1); set 0 has 632,913 rows of 12,632 species, 629,742 of them in the
    floor vocabulary (the reference count). MT drops the last partial training batch and draws one
    sample per row, so a gate epoch is 78,737 training and 9,846 validation steps. V2T's sample,
    reproduced locally to the job's printed counts, keeps 3,448 species after the floor: 377,968
    training and 47,487 validation rows, 5,905 + 742 steps.
  - Gate epoch times are not a speed comparison: lepinet's three reference runs read every image from
    `/work` with 256 workers (`img_dir` and `num_workers` in each run's `config.yaml` under
    `L:data/ucloud_models_logs/`), while the gates read the staged copy with 32 workers on a shared node.
    Only V2T compares throughput. The images are small (300 sampled: median 512×384, 78 kB; short side
    p10-p90 324-499), so the 460 crop upscales most of them, decoding is cheap, and both parity readers
    do the same PIL work (MT's `SquareCropReader`); PIL draft decoding would not help.
  - V2T result (job 12445759, done 14:42, all four runs rc 0; `v2t_b.py` on the downloaded folder):
    steady training img/s from batch 500 to the end of training, lepinet L1 1,201 and L2 1,205, MT M1
    1,422 and M2 1,428: MT/lepinet 1.185, each code's runs within 0.4 %, no drift from L1 to L2.
    Peak allocated GPU memory lepinet 10.4-10.6 GiB (nvidia-smi 12.5-12.9 GB), MT 6.6 GiB (8.0 GB);
    GPU busy lepinet 72 %, MT 85 %, so neither saturates the B200 at bs 64. Node: 376 CPUs, 2.27 TB,
    cgroup 48 CPUs and 288 GB, the three gates training alongside. Validation of 47,487 images: MT
    18 s, lepinet 29 s. First training batch: MT 15 s (44 workers), lepinet 25 s (256 workers).
  - V2T wall time: lepinet's process 386-390 s (fastai epoch 6:12); MT's 508-552 s, about 190 s of it
    DataLoader worker start-up (below), plus the dendrogram (46 s in M1, 3 s cached in M2). MT exited
    1.5 s after saving `last.pt`: the workstation's exit delay did not occur on UCloud.
  - Cause of that overhead (2026-11-04; workstation profile of one F epoch on V2T's 3,448 species capped
    at 16 + 4 rows, with faulthandler stack dumps every 0.5 s, then timed launches): Python 3.14 starts
    DataLoader workers with forkserver. Each launch pickles the worker's arguments, dataset included
    (11.1 MB for 55,168 training rows, 2.75 MB for 13,682 validation rows), and blocks in
    `popen_forkserver.py:58` writing them into the pipe until the child has re-imported
    `mini_trainer.train`: about 1.6 s per worker on the workstation and 2.1 s on UCloud, one worker at
    a time. With persistent workers that happens once per loader per run, at the first training and
    the first validation batch, not every epoch. Earlier readings of "93 s before and 96 s after
    validation" were wrong about the second gap: the progress clock starts after the training loader's
    workers are up, so its launch hid in the offset. Preloading `mini_trainer.train` into the forkserver
    (`multiprocessing.set_forkserver_preload`) cut a launch to 0.11 s and `fork` (lepinet's choice) to
    0.02 s; the logged accuracies were identical in all three. The 5 s-per-worker exit delay is a
    separate, workstation-only effect (81 s in all three). Scripts: session scratchpad `prof/`.
  - Gate epoch 1 (validation set 1, MT minus clean log): I species / genus / family macro-F1 +0.0013 /
    +0.0013 / +0.0009, loss equal (1.3354 vs 1.3355); BU +0.0027 / +0.0043 / -0.0035, loss -0.0325;
    F species -0.0047, loss +0.0041. lepinet's own full-scale repeats (lepinet-e1; lepinet seeds
    nothing, so a repeat is an independent draw; logs under
    `/data/au761367/lepinet/ucloud_mirror/data/ucloud_models/`): F repeat A5 `20260731-080034` minus
    reference, species F1 by epoch -0.0060, +0.0024, -0.0007, -0.0006, -0.0007, loss +0.0184 at epoch
    1; BU repeat M2 `20260801-134647`, species +0.0018, -0.0028, -0.0006, -0.0013, -0.0007, genus
    +0.0031, -0.0018, +0.0008, 0.0000, +0.0005, family +0.0024, +0.0005, +0.0050, +0.0020, +0.0014,
    loss -0.0260 at epoch 1. So MT's epoch 1 lies inside lepinet's run-to-run spread, which is 2-3×
    the 0.002 threshold at epoch 1 and about 0.0007 on species by epoch 5: judge later epochs against
    these spreads (`compare_b.py` prints them as Δrep); the test fold carries the gate.
  - Per-epoch figures at 12,041 species: confusion matrices 10-16 s, the species dendrogram 135-157 s
    (85-100 s resolving names, apparently GBIF lookups from the job). After the gates, cache the names
    or switch the per-epoch figures off in the lepinet presets (lepinet-e1): about 12 min per 5-epoch
    run and a network call that can fail mid-training. Gate I's epoch 1 took 70.6 min with validation.
  - GPU-option job `mt-gpuopt` (job 12446034, submitted 15:36 at the owner's request; spec
    `.agents/local/lepinet-b/mt-gpuopt.toml`, setup `setup-gpuopt.sh` built by `gpuopt/build-setup.sh`
    from `gpuopt/{setup-head.sh,driver.py,orchestrate.py}`). Synced from `108e43f` (MT code identical
    to the gates' `5099cc0`) to its own folder `/12347837/repos/mini_trainer-gpuopt`. F preset, one
    epoch per run on V2T's sample capped at 40 + 5 rows per species (137,920 + 16,975 rows, 2,155 + 266
    steps), 44 workers, forkserver preload from the driver. Phases: base; A cudnn.benchmark ×
    channels_last × {foreach, fused AdamW}; base; B four compile modes × channels_last on A's winner;
    base; C compiled optimizer (± CUDA graphs) on the winner; winner, base, winner. Workstation smoke
    (shared 5090, 8 workers): baseline 783 img/s; benchmark + channels_last + fused 1,005; + compile
    default 1,072 (peak 4.5 GiB); reduce-overhead 1,061 (4.3 GiB); compiled optimizer 759. The smoke
    caught two bugs, both fixed: the driver split its arguments at the first "--" inside "--benchmark",
    and run tags must use MT's renaming (`base-1` becomes `base_1`).
  - Gate training rate from the step counter at 35-39 min (total 5 × (78,737 + 9,846) steps):
    BU about 1,420 img/s, F about 1,400, I about 1,290 (three heads), 32 workers, so as fast as V2T's
    MT with 44: the loaders are not the limit.
  - Analysis tools (ignored, in `.agents/local/lepinet-b/`): `compare_b.py --gate H <run folder>` prints
    each gate's validation curve against the clean log (flags beyond 0.001 and 0.002) and epoch times
    and training img/s from the sibling `<run folder>-train.log`; `v2t_b.py <v2t folder>` gives
    steady-state img/s from batch 500, peak memory and GPU use per V2T run. Dry runs on the local
    end-to-end and protocol (a) logs parse correctly (protocol (a): L1/L2 684-687 img/s, M0 748).
- Run guard (`setup-mt.sh` writes `/tmp/gate-run.sh`; each gate's run line is `bash /tmp/gate-run.sh I|BU|F`).
  - Training gets 14 h. Once "Final weights saved at: <path>" is logged and that file exists (per-epoch
    saves are `checkpoint_*.pth` and `best.pt`), teardown gets 15 min before the process is killed.
    Predict then runs whenever training finished, under a 2 h timeout.
  - The path is read from the log line. MT replaces characters outside `[A-Za-z0-9_]` in run names
    and appends `_1`, `_2`, ... when the folder already exists (`mini_trainer/utils/_core/fs.py:43-55`).
    The first guard built `gate-I/weights/last.pt`, but MT writes `gate_I/...`, so all three gates
    would have skipped predict. lepinet-e1's request for a real local predict run found this on
    2026-11-04; the stub interpreter used earlier had written to the guard's own path.
  - Runs are now named `gate_I|BU|F` (training under `tmp-runs/`, predictions under `tmp-runs/eval/`).
    A log left by an earlier attempt is moved aside, so a stale marker cannot start the teardown clock.
  - Tests:
    - A stub that renames like MT: normal exit, teardown hang ignoring SIGTERM, training error, hang
      during training, `last.pt` without the log line, a name collision (`gate_I_1`), and a stale log
      with a hanging run. All seven behave as intended.
    - The real trainer and predict through the extracted guard for I, BU and F, plus a second I run
      that collided. Each run had 5 epochs on 592 rows of 8 species and predicted 200 set-0 images.
      All exited 0.
- Out-of-vocabulary set-0 rows: the gate models have 12,041 species, but set 0 has 632,913 images, so
  predict sees 3,171 rows of 591 other species. The parquet adapter maps their class to None, while
  predict carries labels as strings. Checked on 2026-11-04 with CUDA and fp16, using a 200-row set-0
  sample of the full parquet:
  - Protocol-(a) checkpoint M0 (I, 512 species; 170 of the 200 sample rows outside it): 600 rows with
    no nulls, labels equal to the parquet keys at every rank, `known_label` equal to vocabulary
    membership. `score.py` drops the 170 and scores 30.
  - The end-to-end I, BU and F checkpoints (8 species, all 200 rows unknown at species level): the same
    checks pass.
  - `tests/integration/test_recipe_lepinet.py` covers the I head on CPU.
- Gate scoring (lepinet-e1, 2026-11-04): score on exactly lepinet's 629,742 rows, matched by
  `<speciesKey>/<filename>` to the floor runs' predictions. Before scoring, assert 629,742 rows and
  12,041 species. The extra 3,171 rows are outside the reference fold. In mini_metrics' convention
  (`known_only=False`) they would add 591 classes the model can never predict, about -0.04 on species
  macro-F1. Report the 632,913-row number only as a labelled side figure.
  - References (read only; from lepinet-e1), under `/data/au761367/lepinet/ucloud_preds/`, each
    `<dir>/eval/predictions.csv`:
    - I: `base-coarse-eval/lepinet-heads-global-independent-muon-5ep-oversample-effnetv2s-ucloud`,
      sha256 `330d81aa6f9d617f…`.
    - F: `singlehead/lepinet-singlehead-species-effnetv2s-ucloud`, sha256 `72b8b81b9bdf3496…`.
    - BU: `marginal/lepinet-marginal-effnetv2s-ucloud`, sha256 `a21d050468cdb1f9…`.
    - Not the reference: `baseline-cleanfold/` holds the same I model name on a 217,856-image subset
      (0.8954).
  - Checked 2026-11-04 on copies:
    - The hashes match.
    - Each file has 629,742 images per level and 12,041 species, all `known_label` True.
    - Filenames are `<speciesKey>/<file>.jpg`, and all three files cover the same image set.
    - `dev/lepinet/score.py` reproduces all nine gate numbers from them to four digits: I 0.910975 /
      0.958745 / 0.970785, F 0.913471 / 0.960627 / 0.973940, BU 0.913532 / 0.963272 / 0.977842.
    - `score.py` writes `predictions.csv` and `scores.csv` next to its input, so never run it on a
      reference file in place.
  - `.agents/local/lepinet-b/match_ref.py` (ignored):
    - It matches by `<speciesKey>/<file>` and asserts the reference counts (629,742 images, 12,041
      species, all known).
    - It asserts that MT's vocabulary is those 12,041 species, that every reference image has an MT
      prediction, and that the labels agree at every rank.
    - It reports MT-only images, writes `restricted/mini_metric.csv`, and with `--lepinet` scores the
      whole-set-0 side figure.
    - Dry run on protocol (a), L1 as reference and M0 as MT: the restricted score equals M0's set-0
      scores exactly. With 170 out-of-vocabulary images appended, they are reported and excluded,
      and the restricted file is unchanged.
  - Per gate, after downloading `tmp-runs/eval/gate_H/` and `tmp-runs/gate_H/class_spec.json`, each
    to its own LOCAL (`MM=L:data/tools/mini_metrics-61db373`, lepinet's venv):
    `PYTHONPATH=$MM python -I .agents/local/lepinet-b/match_ref.py --mt <eval>/mini_metric.csv
    --reference <ref> --class-spec <class_spec.json> --out <dir> --lepinet L:`, then
    `PYTHONPATH=$MM python dev/lepinet/score.py --lepinet L: --predictions <dir>/restricted/mini_metric.csv
    --seeds 5 --expect-instances 629742`.
- Predict reads set 0 from local disk (2026-11-04). On a parquet input, predict reads every set-0 row
  (632,913 images) from the `images/` folder next to its `--input` parquet
  (`mini_trainer/integrations/parquet.py:57-70`). Predict uses at most 32 workers by default. /work serves
  80-94 img/s to 32 readers (lepinet-e1, V0 evals), so reading set 0 from /work takes 1.9-2.2 h and
  overruns the 2 h limit. `setup-mt.sh` therefore also stages set 0 to NVMe with 256 threads, after
  `stage.py`, and predict reads the `/tmp` parquet copy. The file list comes from MT's own parquet
  adapter, and any missing image aborts setup before training. Tested locally on a 1,500-row sample:
  staged files = set-0 rows = predict's list (161, byte-identical); a missing image and a misplaced
  parquet both abort. Prediction paths start with `/tmp/global_lepi/images/`, so check (b) matches on
  `<speciesKey>/<filename>`.
- Downloads: give each eval folder its own LOCAL. `ucloud files download REMOTE LOCAL` copies REMOTE's
  contents into LOCAL, so same-named model folders merge and their `predictions.csv` files overwrite
  each other (lepinet-e1).
- Checks requested by lepinet-e1: the gap between saving `last.pt` and the predict step (see the exit
  delay above); MT's set-0 predictions cover exactly the images of lepinet's I-port prediction file
  (629,742) before scoring; per-epoch validation accuracy, macro-F1 and loss against the clean logs,
  epochs counted from 1, plus wall-clock training img/s and epoch time from each gate log.
- One B200 per run, single GPU, bs 64, fp16, 5 epochs, lepinet's floor; presets I, BU and F.
- Curves against the clean-package logs in `L:data/ucloud_models_logs/` (validation, argmax macro-F1):
  I `20260724-181230` species 0.7848, 0.8350, 0.8697, 0.8985, 0.9099; BU `20260730-074913` species
  0.7856, 0.8384, 0.8691, 0.8968, 0.9133, genus 0.8758 → 0.9614, family 0.9237 → 0.9794; F
  `20260729-182718` species 0.7978, 0.8375, 0.8717, 0.9002, 0.9128 (only its `F1(macro)` and
  `F1(micro)` columns are broken). The dev/030 I run agrees with the clean one within 0.0007 per epoch.
- Throughput, job V2T (`.agents/local/lepinet-b/mt-v2t.toml`, `setup-v2t.sh`), design by lepinet-e1:
  stage the 50-floor set, draw a 10 % random sample of its rows (seed 20261103), one epoch per run in the
  order L, M, M, L, steady-state training img/s timed after the first ~500 batches. lepinet runs as on
  UCloud: its `ucloud/setup-lepinet.sh` (on a `/tmp` copy, since setuptools writes `lepinet.egg-info`
  into the read-only mount), the singlehead config `20260729_ucloud_singlehead_species_effnetv2s.yaml`
  with only data file, image directory, output directory and epochs changed, 256 workers, no
  `GCCallback`; MT runs the F preset without compile, with CPUs minus 4 workers (MT's own rule without
  its cap of 16). Both configs re-apply the 50 floor to the sample (species with at least 50 sampled
  images). Peak memory: `max_memory_allocated` for both, plus an `nvidia-smi` trace per run.
- Process: `L:ucloud/budget_check.py <spec>` before every submission; `ucloud q tick; echo $?`; read job
  logs, not status; send lepinet-e1 each spec path and job name; code reaches UCloud by ucloud-api
  `[sync]` or a worktree tarball (the branch is local only).
- Gate result (scored by lepinet-e1 with `match_ref.py` and `score.py`; lepinet journal V2, `L:` 513a6a7;
  copies in `L:data/ucloud_preds/V2{I,BU,F}-*`): on the 629,742 reference images, macro-F1 at threshold 0,
  species / genus / family, MT minus lepinet: I 0.9107 / 0.9588 / 0.9726 (-0.0002 / +0.0001 / +0.0018),
  BU 0.9131 / 0.9629 / 0.9803 (-0.0004 / -0.0004 / +0.0025), F 0.9144 / 0.9608 / 0.9759 (+0.0009 /
  +0.0002 / +0.0020); calibrated species within 0.0009. lepinet's F repeat moves family by 0.0023, so
  family noise is about 0.002. Validation gaps peak at 0.005 (BU species, epoch 2) and close by epoch 5.

### Campaign (paper, 2026-11-05 on)

- Owner's decisions (via lepinet-e1, submissions approved in this session): pinned recipe = cross-entropy,
  no oversampling, one-cycle, crop + per-image affine, regulariser 0, learned scale and bias, fp32 head,
  5 epochs at 256 px, no floor (12,632 species), no per-epoch figures, `channels_last`; one seed (42) for
  now. Multi-GPU jobs, one run per GPU, only `last.pt` kept, test and trap predictions per run.
- Runner: `.agents/local/campaign/` (ignored): `build.py PLAN.json` writes `<job>.toml` and `setup-<job>.sh`
  from `setup-template.sh`; plans in `plans/`. The job stages once (no-floor rows, set 0, 47,905 trap crops),
  installs MT non-editable (later syncs of the shared folder cannot change a running job) and syncs to
  `/12347837/repos/mini_trainer-campaign`. Tested locally end to end on one run.
- Submitted 2026-11-05 10:13 from `dbedd68`: `c-v2s` (job 12447133, 6 B200: v2s_bu_s42 = cross-entropy arm,
  teacher and grid cell; v2s_bu_emla_s42; v2s_f/i/td_s42; `ctl_gate_bu_cl` = the gate BU preset as run at
  `5099cc0` plus `channels_last`, seed 0, 32 workers, to compare with V2BU within 0.003 at every rank) and
  `c-b0` (job 12447137, 4 B200: b0_f/bu/i/td_s42). `c-vitl` is held (below).
- GPU options (job 12446034, F preset, one epoch on V2T's sample capped at 40 + 5 rows per species):
  baseline 1,335 img/s (4 runs); `channels_last` 1,570-1,723; compile reduce-overhead about 2,080 (with or
  without `channels_last`); max-autotune no better and slow to compile; winner `channels_last` + compile
  reduce-overhead + compiled optimizer with CUDA graphs 2,242 (3 runs, 1.68x), same one-epoch accuracy,
  peak 4.3 vs 6.6 GiB. Owner: smoke-test compile on every planned head and backbone and adopt it for later
  jobs only if training gets significantly faster.
- Readiness (one epoch, local 5090): TD and EfficientNet-B0 train normally. AR does not learn (acc1 1.9 %):
  a subagent ruled out the recipe's head options and found two AR defects, a teacher-forcing leak
  (`mini_trainer/hierarchical/model.py:251`, `:254-256` read each level at its own teacher-forced token
  under a mask that includes the diagonal) and N(0,1) start/positional embeddings that drown the class
  tokens; fixing both did not help (candidate patch, tests, probes and logs: `.agents/local/ar-fix/`,
  ignored; branch `fix/ar-lepinet-recipe` holds no extra commits). ViT-L/16
  needs its native 224 px and does not learn at the recipe's lr 1e-3 (acc1 1.2 %, loss near chance).
- Tools on the branch: `dev/lepinet/flemming.py` (trap predict and scoring, `4e30941`), `pseudo.py`
  (dev/065 gate, dev/066 doses, dev/089 folds, `0b4f671`, by a subagent; reproduces lepinet's summaries; use
  `--min-count 1` for no-floor runs). Options: `logger_builder_kwargs.diagnostics` (`a47be22`),
  `channels_last` (`dbedd68`).
- 2026-11-05 owner decisions: AR dropped (four heads: F, BU, I, TD); seeds 43/44 wait for seed 42; ViT-L at
  224 px, read 403, lr 1e-4 for head and backbone (lr 1e-3 does not learn in a one-epoch probe: acc1 1.2 %
  vs 15.5 %); compile only after a smoke test shows a real gain; reproducing lepinet first (shift block,
  DINOv3 pair, open-set on C3b, ViT-L grid), Pl@ntNet when GPUs are free; T1rep (dev/067 round-robin
  labels) plus a matched arm (lepinet's 12,230 gated crops with true labels), both at 11x.
- Submitted 2026-11-05: `mt-compile-smoke` 12447240, `c-openset` 12447244 (C3b, openset.py summaries),
  `c-dinov3` 12447249 (failed: MT could not attach its head to timm's nested `head.fc`; fixed in `0b08370`,
  resubmitted as `c-dinov3-r2`), `c-vitl` 12447251, `c-t1rep` 12447346. Queued after `c-v2s`, whose
  v2s_bu_s42 trap predictions or weights they use: `c-shift-a/b/c` (18 runs: doses 0.24/2.6/12.3 %, X1, X2),
  `c-dinov3-st`, `c-crt-t2`. `c-b0` finished 14:40: all four runs trained and predicted set 0 and the traps
  in about 4 h.
- Runner pitfall, fixed for later jobs: `python -m` imports from the working directory first, so the jobs
  started from the synced folder (c-v2s, c-b0, c-openset, c-vitl, c-t1rep) import mini_trainer from
  `/work/mini_trainer-campaign`, not from their non-editable install; `run-one.sh` now starts from
  `/tmp/campaign`. Until those jobs finish, sync only behaviour-preserving code to that folder.
- DINOv3 pair (`c-dinov3-r2` 12447408, then `c-dinov3-st`): B8's backbone settings (convnext_large.dinov3_lvd1689m,
  320 px from 384, batch 96, lr 8e-4, warm-up 0.5, Muon on backbone matrices, `head_muon: false` as lepinet
  routes it) with the pinned recipe otherwise. Differences from B8: no ArcFace (the recipe's cosine head), no
  trap augmentation, no floor, fp16 (B8 bf16), 5 epochs (B8 6), the global parquet for the baseline (B8 trained
  on the 2 % merge), and timm's pretrained head LayerNorm before mini_trainer's head (lepinet cut at
  `forward_features` and pooled in its own head); the same in both arms. A compressed local probe (warm-up
  done after 431 steps, batch 16) did not learn at 8e-4 (eval loss 18.4, chance about 19.8) and did at 8e-5
  (11.8): read the full run's training loss near the end of warm-up (about 2.5 M images). Any fallback lr is
  a deviation from B8.
- T1rep arms (`c-t1rep` 12447346): `merge` drops label rows whose species has no non-test row in the
  quality-filtered global parquet, so the label set stays the baseline's. That is 25 of dev/067's 12,230 rows
  (Acronicta cuspis 12, Parastichtis ypsillon 8, Eupithecia denotata 2, Xestia rhomboidea 2, Dichonia aprilina
  1; keys 1775152, 1983694, 5112362, 7825213, 9710182) and 6 of the gated crops (A. cuspis 5, D. aprilina 1).
  Four of the five species are in the unfiltered parquet (removed by the quality filter), 5112362 in neither;
  lepinet's arms kept all 12,230 rows (134,530 at 11x). The gated crops' true labels cover 306 species against
  346 pseudo-label species: the 1.85 % wrong pseudo-labels spread over about 40 extra species.
- Pl@ntNet (`pp-*`): `spec_model_dataloader_kwargs.levels: 5` (species, genus, family, order, class: 987 /
  325 / 111 / 46 / 5 classes), loss weights equal across ranks as in the three-level recipe ([1,1,1,1,1]
  for BU, I, TD; [1,0,0,0,0] for F), 10 epochs; V2-S 384 px (read 512), B0 224 (read 403, as Asger's B0 ran
  at its weights' size), ViT-L 224 (read 403, lr 1e-4). A local one-epoch BU run on a capped copy trained and
  predicted all five ranks. Submitted 11:32-11:48: `pp-v2s` 12447481, `pp-vitl` 12447493, `pp-b0-r2`
  12447540 (the first `pp-b0` upload hit repeated 502s; run names stay `pp_b0_*`). All 12 runs trained and
  predicted, 24-54 min each.
- Lepinet's `--freeze-body` did not freeze conv backbones: `_fit` calls `learn.unfreeze()`, and
  `vision_learner`'s optimiser holds every parameter. All 450 of L4's backbone tensors differ from its
  starting weights, and its epochs took as long as full training. lepinet-e1 found the same for T2, F2, F3,
  R4, R5b, L6, G1, G2, G3 and G3b, but not the BioCLIP P-runs. Draft claim C2 rests on L4 (cRT) and T2
  (adaptation).
- `c-crt-t2` (4 B200, queued after `c-v2s`) re-tests both. Every run starts from v2s_bu_s42's `last.pt`
  (`model_builder_kwargs.weights`, with a fresh optimiser and schedule) and trains 2 epochs: lr 1e-3,
  one-cycle, warm-up 0.5, BU head. Each predicts set 0 and the traps.
  - `crt_full`: sqrt oversampling, whole network. This is what L4 did: deferred re-sampling, not cRT.
  - `crt_frozen`: the same with `freeze_backbone`.
  - `t2_full` and `t2_frozen`: the dose-0.02 merge selected from v2s_bu_s42's trap predictions, no
    oversampling.
  - The full variants re-warm lr 1e-3 on the whole network, as lepinet's runs did. The frozen runs test
    the draft's claims.
- `freeze_backbone` (`e55b65f`, opt-in) trains only the head. Every module off the head path stays in eval
  mode, which keeps BN statistics fixed and turns off dropout and stochastic depth. Precision is unchanged.
  MT's `fine_tune` fails under the recipe ("Expected weight to have type Float but got BFloat16") and would
  cast the trunk to bf16. Checked on V2-S BU from a checkpoint: 0 of 670 backbone tensors changed, at 2,786
  img/s against about 950 for full training.
- Compile smoke (`mt-compile-smoke` 12447240: one epoch per entry, campaign recipe, one B200). The flags
  `--compile --compile-mode reduce-overhead --compile-optimizer --optimizer-cudagraphs` make steady-state
  training 1.21-1.70x faster:

  | Backbone | F | BU | I | TD |
  |---|---|---|---|---|
  | V2-S | 1.33 | 1.55 | 1.34 | 1.50 |
  | B0 | 1.70 | 1.32 | 1.21 | 1.22 |
  | ViT-L | 1.50 | 1.63 | 1.62 | 1.67 |

  - V2-S BU at 384 px gains 1.21x.
  - Compiling costs 5-108 s. Peak memory drops (V2-S 6.7 -> 4.3 GiB).
  - Accuracy is unchanged: with the same seed, compiled minus plain acc1 ranges from -0.42 to +0.31, mean
    +0.04 over 13 pairs.
  - DINOv3 ran on code from before `0b08370` and failed; it needs a re-test.
  - MT compiles the training step, not the module, so `last.pt` keys don't change.
  - A local 3-epoch run (5090, synthetic data) checked train -> eval -> train under compile, and predict from
    its weights.
- Owner, 2026-11-05 15:20: compile the shift block and add a compiled control.
  - `c-shift-a/b` were rebuilt in place, since queued jobs read their setup scripts at launch.
  - `c-shift-c` was resubmitted on 3 B200 with `v2s_bu_compiled_s42`: v2s_bu_s42 plus compile.
  - `c-crt-t2` and `c-dinov3-st` stay uncompiled, because their configs aren't smoke-tested.
  - Runner: `"compile": true` per run or per plan. The flags go on the command line, because store_true
    flags override YAML, and are kept in `tmp-runs/configs/<run>.args`.
  - Caveat (lepinet-e1): c-t1rep's real-label arms (T1rep, gatedtrue) are uncompiled, but the self-training
    arm they're compared with is compiled. That comparison stands only if v2s_bu_compiled_s42 against
    v2s_bu_s42 is inside the noise (test fold about 0.000 at species, probe about 0.004). Otherwise, say so.
- TD gap (lepinet-e1): b0_td's validation species F1 is 0.787, against 0.874 for b0_f. Asger's B0
  conditional trailed his flat head by 4.3, so the recipe gains F, BU and I 3-4 points but TD nothing. The
  gap grows towards species: TD's family is 0.939 against F's 0.964, genus 0.885 against 0.938.
  - Flattening by decay is ruled out. `decay_all: true` decays the head's per-class scales g and biases, but
    every scale in the B0 checkpoints grew from 1. Medians: F 1.72; BU 1.54; I species/genus/family
    1.92/1.87/1.69; TD 1.51/1.99/2.12. TD's species scales spread from 0.99 to 2.25 (p10-p90).
  - Ceiling ties are ruled out too. The embedding is unit-norm and `cosine_to_zscore` clamps g·cos at
    ±1, so every class with g·cos >= 1 gets the same maximum z (about 56 at D = 1280) and no gradient.
    Counted on 10,000 random set-0 images (`.agents/local/td-saturation/saturation.py`, fp32 head math):
    - F, BU, I and lepinet's V2-S F: no class at the ceiling on any image.
    - TD: one class at the ceiling on 9.9 % (species layer) and 9.3 % (genus) of images. Ties of two or
      more on 0.07 % (species only), always including the true class; a wrong tied pick on 0.04 %.
  - Next step: if v2s_td_s42 shows the gap, run TD and F each with the recipe head and with MT's default
    (frozen unit scale, zero bias), one epoch on the smoke sample: 4 runs, about 15 min on one B200.
    Needs the owner's approval. MT can't run TD without cosine logits: HierarchicalClassifier rejects
    `normalized: false`.
  - Context on the same sample: top-1 at species F 0.916, BU 0.912, I 0.907, TD 0.875, lepinet V2-S F
    0.932. TD's family layer is as accurate as I's (0.983). TD's layers work at positive g·cos; F and BU
    at the negative end (median per-image maximum -0.86 and -0.88).
  - lepinet's V2-S F median scale g is 1.74. Cite that, not lepinet PLAN's 1.88, which is a mean
    (2026-08-06).
- Limitation of the recipe's head, the same function in lepinet and MT: the clamp's floor. With learned
  scales above 1, F and BU push most classes to g·cos <= -1: on average 94 % (F) and 97 % (BU) of
  species logits per image, 57 % for lepinet's V2-S F, 15 % for I's species layer, about 0 for TD.
  - The true class sits at the floor on 0.33 % of B0 F's images, 0.31 % of BU's and 0.09 % of lepinet
    V2-S F's. Such an image is always misclassified, and in training it sends no gradient through its
    own class. That's about 4 % of B0 F's top-1 errors.
  - A fix (scale on z, or after the clamp) changes every head, so it waits until after the paper unless
    the owner asks for it to be measured.
- Owner, 2026-11-05 (via lepinet-e1; submissions approved here):
  - DINOv3 ConvNeXt-L replaces ViT-L/16 in both head grids. `c-vitl` finishes as the appendix row for the
    original ViT choice.
  - Queued after `c-v2s`, created after the reproduction jobs so they launch behind them in the same tick:
    - `c-dinov3-grid`: Lepidoptera F, I and TD on 3 B200, about 15 h each.
    - `pp-dinov3`: Pl@ntNet F, BU, I and TD on 4 B200, 10 epochs, five ranks.
    - Both are uncompiled, with exactly dinov3_bu_s42's settings.
  - I and TD were checked locally on the same architecture (random weights, 1 epoch, MuonAuxAdamW,
    `head_muon: false`): both train and predict. Every other timm ConvNeXt size is on MT's blacklist.
  - EfficientNetV2-B0 is shelved: no GPU hours.
  - MT's timm wrapper exposes no mean or std, so the crop reader uses ImageNet statistics for every timm
    backbone. That's documented in `dev/lepinet/README.md` (`f4c8114`).
- Launch after `c-v2s` (2026-11-05 16:21-16:36), with job IDs:
  - Running: `c-dinov3-st` 12448200, `c-crt-t2` 12448202, `c-shift-c` 12448203, `c-dinov3-grid` 12448204,
    `pp-dinov3` 12448205. All passed preflight.
  - UCloud starts 3- and 4-GPU jobs at once, but an 8-GPU job waits for a whole free node: `c-shift-b`
    (12448199) sat IN_QUEUE while later, smaller jobs started. Prefer jobs of 4 GPUs or fewer.
  - `c-shift-a` failed twice on UCloud's transient 502 from `/api/files/upload`. The queue doesn't retry,
    so a failed record needs a new submission.
  - A cron tick submitted the DINOv3 jobs before they could be held back, so they run ahead of the shift
    block.
  - Owner's decision: repackage the two 8-GPU shift jobs as four 4-GPU jobs. `c-shift-b` (12448199) and
    `c-shift-a-r2` (12448212) were cancelled under their record locks while IN_QUEUE; UCloud lists them as
    SUCCESS.
  - `c-shift-a1` 12448224 and `c-shift-a2` 12448233 hold the same 16 run entries, checked
    programmatically. Both wait IN_QUEUE: the cluster is full.
  - `c-shift-b1` and `c-shift-b2` failed on the 502 for over an hour (16:36-17:53). They went through at
    18:01 as 12448347 and 12448348. `mt-td-head-check` went through as 12448351.
  - The 502 wasn't transient. A submission's sync requests an upload session for every worktree file (501),
    64 per POST to `/api/files/upload`. ucloud-api notes a fixed ~5 s gateway timeout on that call that
    grows with batch size.
    - At 17:58, 16 files took 1.4 s and passed, while 64 overran the limit and returned 502 on every retry.
      The backend had slowed, plausibly from our own jobs each copying 5.7 M images out of /work.
    - Workaround: submit through a wrapper that sets `ucloud_api.transfer._CONTROL_BATCH = 16` for that
      process, then runs `q submit`. The tool's source is unchanged.
    - Cron ticks still use 64, so a QUEUED record can keep failing until the backend recovers.
  - `pp-dinov3` finished 17:56: all four runs trained and predicted in 85-87 min.
  - Queue safety: a tick re-reads each record under an exclusive file lock and submits only if it's still
    QUEUED, so overlapping ticks can't submit a record twice. `q rm` takes no lock, so remove a QUEUED
    record under `Queue.lock`.
- Scores by lepinet-e1 (lepinet RESULTS VS, VC, VT; `L:` 04e4fe1):
  - TD's gap persists at V2-S. Test species F 0.8890, BU 0.8877, I 0.8812, TD 0.8413: TD -4.8 against F,
    where Svenning et al.'s V2-S had -1.4.
  - Owner approved the head check: `mt-td-head-check` (`.agents/local/td-head-check/`). TD and F each run
    with the recipe head and with MT's default head (`trainable_scale`, `trainable_bias`, `float32_head`
    and `legacy_spherical_init` false). One plain V2-S epoch each on the compile smoke's capped sample,
    seed 0, one B200.
  - Result (12448351, 2026-11-05 18:15): MT's default head doesn't close the gap, so the recipe's head
    options aren't the cause.
    - acc1 and species / genus / family F1: TD recipe 60.25, 0.5847 / 0.6699 / 0.7547; TD default 59.58,
      0.5780 / 0.6651 / 0.7578; F recipe 65.38, 0.6360 / 0.7288 / 0.7911; F default 65.32, 0.6363 /
      0.7256 / 0.8024.
    - TD - F at species F1: -0.0514 with the recipe head, -0.0583 with the default head.
    - Noise: F reproduces the compile smoke's plain run bit for bit. TD drifts about 0.1 acc1 between
      identical runs (scatter-based normalization isn't deterministic on GPU).
  - Remaining candidates, in `ConditionalClassifier.forward` (`mini_trainer/hierarchical/model.py:204-218`):
    - The sibling normalizer is detached, so each layer gets the TD-marginal gradient (p_TD - y). A
      sibling's push-down is scaled by its parent's probability.
    - With loss weights [1,1,1], the species and genus losses send coarser layers exactly their own loss's
      gradient again: family three times, genus twice.
    - Checks: TD with weights [1,0,0] (config only), and an opt-in non-detached normalizer (core change,
      feature branch). Asked lepinet-e1 how Svenning et al.'s TD (-1.4 at V2-S) differs. Neither runs
      without the owner.
  - Cross-entropy against EMLA: EMLA is +0.59 at species in-distribution but -0.41 genus, -0.76 family,
    and -3.9 / -5.5 / -4.9 on trap / probe / probe-HO. Cross-entropy stays.
  - The `channels_last` control passes on lepinet's 629,742 rows: 0.9135 / 0.9631 / 0.9790 against V2BU's
    0.9131 / 0.9629 / 0.9803, within 0.0013 at every rank.
  - T1rep reproduces: probe 0.722 and probe-HO 0.305, against lepinet's 0.723 / 0.312 in the same grouped
    convention. gatedtrue: probe 0.682, probe-HO 0.343. That's +3.7 on held-out species over no adaptation
    (0.306), which the round-robin draw doesn't give.
  - Traps, species calibrated by (trap, night), trap / probe / probe-HO: I 0.381 / 0.405 / 0.307, F
    0.348 / 0.366 / 0.298, BU 0.339 / 0.371 / 0.306, TD 0.315 / 0.347 / 0.271.
- Predict with the nested-head change (`0b08370`), which the running jobs import from the synced folder: on
  200 images, checkpoints of V2-S, B0 and ViT-L/16 give identical prediction files before and after it
  (confidences within 2e-7).
- For Asger: YAML values of `store_true` flags are silently overridden by the CLI default `False`
  (`compile: true` in a config has no effect; `mini_trainer/config.py` `merge_dicts` after
  `restructure_cli_args`); `--channels-last` uses `default=None` to avoid it. The AR defects above.
- Weekend queue (owner, 2026-11-06 ~09:35, approved in this session; requested by lepinet-ac):
  - 56 single-GPU jobs, one run each, about 304 GPU-h. On UCloud, 2-8-GPU jobs start much later than 1-GPU
    ones, and staging per job is accepted.
  - Plans are in `.agents/local/campaign/plans/weekend1/`, with order and estimates. They were submitted with
    `.agents/local/campaign/submit-all.sh` (16-file wrapper and retries); log `weekend1/submit.log`.
  - Queue names are `w-` plus the run name with dashes. Everything is uncompiled, with seed 42's settings
    except the named factor:
    1. Tests: `mt-td-emla-check` (12448887; TD and F with `criterion weighted: true`, one epoch on the smoke
       sample, against the CE gap of -0.051) and pp_dinov3_f_lr2e4_s42 (lr and backbone_lr 2e-4).
    2. Seeds 43/44, Lepidoptera V2-S and B0, F/BU/I/TD.
    3. The same on Pl@ntNet.
    4. Pl@ntNet V2-S, four heads, with power 0.5 (`_sqrt_`) and with EMLA (`_emla_`).
    5. Pl@ntNet DINOv3 BU/I/TD at lr 2e-4.
    6. DINOv3 Lepidoptera seeds 43/44.
    7. V2-S F and TD at 10 epochs, 384 px read 512 (`_384x10_`).
    8. V2-S BU at 30 epochs, 384/512 (`_384x30_`). Its `train_hours` is 40; `build.py` gained a per-plan
       `train_hours` (default 24) for it.
  - Evidence from lepinet-ac: TD's validation gap to F shrinks every epoch (V2-S -0.104 → -0.047 over 5
    epochs; B0 -0.113 → -0.087).

### Later phases

- **Phase 2:** self-training on trap crops (per-row weights through `sampling_weights`; mount
  `flemming`); classifier re-training on a frozen backbone (`freeze_backbone`, `e55b65f`; MT's
  `fine_tune` casts the backbone to bf16 and leaves BN statistics updating, `mini_trainer/builders.py:68-76`); trap augmentation
  (`L:src/lepinet/augment.py:33-111`); open-set and back-off evaluation; trap calibration grouped by
  (trap, night). F needs no new code.
- **Phase 3 onwards** (single factors on MT's test set): EMLA vs CE, oversampling vs none, schedule,
  augmentation, image size, epochs; frozen vs learned head scale and bias, scored on the test fold and
  the trap sets (it changes MSP, energy and calibration); the deployment preprocessing contract (lepinet's
  app used a plain resize; one eval-only comparison settles it); paper reruns; Nemo vs this recipe on the
  Flemming trap crops; the production model. MT's per-batch augmentation draws and its 87.5 % centre
  crop also apply to Nemo's recipe; lepinet-e1 raised them with Asger as phase-3 factors. MT defaults
  stay unchanged.
- **Phase 3 anchors**, lepinet V0 (`L:` commit `743fc26`, journal section "V0"), no floor. These are
  scored on the whole test fold (632,913 images, 12,632 species), so they are not comparable with the
  gate references, which use the 12,041-species restriction.
  - Macro-F1 at threshold 0, species/genus/family: V0F 0.9110/0.9600/0.9760, V0BU 0.9107/0.9625/0.9782.
  - The 591 species that the floor excludes score about 0.875. Their presence costs the other 12,041
    species 0.0009.
  - Trap crops, species macro-F1 calibrated by (trap, night), trap/probe/probe-HO: V0F
    0.279/0.300/0.235, V0BU 0.330/0.351/0.288, Nemo 0.305/0.338/0.293. The floor costs nothing on trap
    crops, and BU is 5 points above F on every trap set.

### Open questions

1. Where the recipe builder and presets finally live is Asger's call; they are easy to move at review.
