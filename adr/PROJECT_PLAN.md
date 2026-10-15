# Project Plan: Supervised Non-Neural Artifact Detection and Reconstruction for Continuous Speech-Listening EEG

Owner: Joshua Ighalo (PhD student, McMaster University)
Target journal: Journal of Neural Engineering (methods paper)
Status: implementation started (see `PROGRESS.md`). Revision 2 (2026-10-15):
TUAR replaced by PhysioMotion (detector training) and the phantom EEG
dataset (templates and ground truth).

This document is the single source of truth for the project. Read it fully
before writing code. When something here is marked **VERIFY**, check it
against the source before relying on it. When something is ambiguous, ask
instead of guessing.

---

## 1. Goal and claims

Build and validate a pipeline that:

1. **Detects** artifacts in continuous EEG with supervised, non-neural classifiers.
2. **Removes** them and **reconstructs** the underlying EEG with supervised,
   non-neural methods, routed by artifact type.

The paper must show two things:

- **Section A (correctness):** on semi-synthetic data with a known ground
  truth, the pipeline recovers the true neural response (TRF) better than
  baselines.
- **Section B (real-world benefit):** on real speech-listening EEG, the
  pipeline improves forward-model prediction, backward-model reconstruction,
  attention decoding and TRF reliability, without exploiting stimulus-locked
  eye artifacts.

Main claim: routing by artifact type (detector class decides which
reconstruction method fixes each artifact) beats a single MWF and standard
automated pipelines. Supporting claims:

- A detector trained on one public lab dataset (PhysioMotion, Neuracle
  system) transfers to BioSemi 64-channel speech-listening EEG; the domain
  gap is quantified.
- Detection quality limits reconstruction quality (shown by comparing
  detector flags, oracle flags, heuristic flags and random flags at matched
  flag rates).

If routing does not beat the automated-MWF baseline, the paper is reframed as
a benchmark study. Report results honestly either way.

## 2. Hard constraints

- **Proposed method:** supervised, non-neural ML only.
  - Detection: gradient-boosted trees (LightGBM or XGBoost), SVM (RBF), LDA
    (baseline), optional random forest and stacked ensemble.
  - Reconstruction: multi-channel Wiener filter (MWF), ridge regression,
    learned spatial interpolation (ridge).
- **The proposed method must NOT use** ICA, wavelets (including DWT features)
  or neural networks, anywhere.
- ICA + ICLabel is the comparison method. ICA and ICLabel may appear **only
  as the baseline**, never inside the proposed method.
- Features: time-domain or FFT/Welch only.
- Continuous EEG only (no event-locked epoching inside the method).

## 3. Datasets

| Dataset | Role | Access |
|---|---|---|
| Das et al. Dataset I v2.0 (KU Leuven AAD dataset) | Section A base data (semi-synthetic) | https://zenodo.org/records/4004271 (DOI 10.5281/zenodo.4004271), open, CC BY-NC-SA 4.0 |
| PhysioMotion Artifact (OpenNeuro ds006386) | Detector training (labelled artifacts plus clean rest) | https://openneuro.org/datasets/ds006386 (DOI 10.18112/openneuro.ds006386.v1.0.1), open, no registration |
| Phantom EEG (OpenNeuro ds004784) | Artifact templates with known ground truth; external reconstruction test | https://openneuro.org/datasets/ds004784 (mirror: https://nemar.org/dataset/on004784), open, CC0 |
| Joshua's own dataset | Section B real-data validation | Local (`../dataset/cocktail`, see `configs/paths.yaml`) |
| Klados and Bamidis 2016 (optional) | Extra ocular ground-truth check | https://data.mendeley.com/datasets/wb6yvr725d |

Do **not** use the old Dataset I version (https://zenodo.org/records/3377911).
It was the version used in Das et al. 2020 but has been withdrawn and is
restricted.

### 3.1 Dataset I v2.0 details

- 16 normal-hearing subjects, BioSemi ActiveTwo, 64 channels, recorded at 8192 Hz.
- Public files: `S1.mat` ... `S16.mat`, `stimuli.zip`, `preprocess_data.m`, `README.txt.txt`.
- Each trial was already: high-pass filtered at 0.5 Hz, downsampled to
  128 Hz, and artifact-cleaned with the MWF method of Somers et al. 2018.
  This MWF-cleaned data is our "clean" base.
- 20 trials per subject: 4 from experiment 1, 4 from experiment 2,
  12 repetitions in experiment 3.
- Trial fields: `RawData.Channels` (1 to 64), `RawData.EegData` (samples x
  channels), `FileHeader.SampleRate`, `TrialID`, `attended_ear` ('L'/'R'),
  `stimuli` (left-ear and right-ear wav names), `condition` ('HRTF' or
  'dry'), `experiment`, `part`, `attended_track`, `repetition`, `subject`.
- Stimuli: `part{p}_track{t}_{condition}.wav`. Envelopes are extracted from
  the **dry** versions.
- Truncate audio to the EEG length per trial (stories differ in length).
- Known biases (from the authors): trial fingerprints (never split one trial
  between train and test) and eye-gaze bias.
- Dataset citation requested by authors: Biesmans W, Das N, Francart T,
  Bertrand A (2016). Auditory-inspired speech envelope extraction methods for
  improved EEG-based auditory attention detection in a cocktail party
  scenario. IEEE TNSRE 25(5):402-412.
- **VERIFIED (2026-10-15, `scripts/02_verify_dataset1.py`):** channel labels are
  stored in every trial's `FileHeader.Channels` and match the standard BioSemi
  64 order (A01..B32, MNE `biosemi64`) in all 16 subjects; correlation
  falls off with electrode distance under these labels (Spearman rho -0.42 to
  -0.90 vs shuffled-label null >= -0.18). **Cz = column 48 (0-based 47)**,
  matching `EegData(:,48)` in the authors' `preprocess_data.m`; C5 = column 14
  (0-based 13). `condition` is stored lowercase (`'hrtf'`/`'dry'`), and
  `stimuli` holds the hrtf wav names (swap the suffix for `_dry.wav`).
- The MWF cleaning is stated on the Zenodo record page, not in the
  `README.txt.txt` shipped with the download.

### 3.2 PhysioMotion Artifact details (detector training)

- Data descriptor: Scientific Data 13:807 (2026),
  doi:10.1038/s41597-026-07120-7. Code:
  https://github.com/JiangweiYu221/PhysioMotion_Artifact (MIT).
- 30 healthy adults, Neuracle NeuSen W system, about 120 min of continuous
  EEG per participant across six blocks.
- Two versions shared in BIDS: raw (59 EEG channels, 1000 Hz, no filtering)
  and preprocessed (34 bipolar channels, 50 Hz notch, 0.5 to 150 Hz). **Use
  the raw version** so channels can be matched to BioSemi positions.
- Tasks: eyes-closed and eyes-open rest (2 min each per session); vertical
  eye movements, horizontal eye movements, continuous blinking; head rotation
  and nodding; tongue movement, chewing, swallowing, eyebrow raising; five
  combined tasks (e.g., blinking plus eyebrow raising).
- Annotations: point-wise and per channel, CSV columns `channel, start_time,
  stop_time, label`, in a `Manual_Annotations` folder. Two blind annotators,
  disagreements resolved jointly. No agreement statistic reported.
- No dedicated EOG or EMG channels.
- **VERIFIED (2026-10-15, `scripts/04_verify_openneuro.py`):**
  - Raw: 59 EEG channels, 10-10 names, 1000 Hz, 30 subjects x 6 runs
    (~1250 s each). 57 names shared with BioSemi 64 (PhysioMotion-only:
    PO5, PO6; BioSemi-only: P1, P2, P9, P10, Iz, CPz, AFz).
  - Licence: CC0 (`dataset_description.json`). BIDS authors: Jiangwei Yu,
    Aonan He (paper author list still to check).
  - **Annotations are on the 34 bipolar pairs of the preprocessed version**
    (19 10-20 electrodes, e.g. `Fp1-F7`), plus a channel `ALL`. They must be
    mapped from pairs to electrodes (§6). Times in seconds.
  - Labels (180 CSVs): `blink`, `hor_eyem`, `ver_eyem`, `hor_headm`,
    `ver_headm`, `tongue`, `chew`, `swallow`, `eyebrow`; combinations
    `blink_hor_headm`, `blink_ver_headm`, `blink_eyebrow`, `tongue_eyebrow`,
    `swallow_eyebrow`; rest blocks `close_base`, `open_base` (on `ALL`).
  - Rest blocks are labelled (536 blocks), and **no artifact annotation
    falls inside any of them**, so they can serve as `clean`.
  - BIDS `events.tsv` trial codes are undocumented; use the annotation CSVs.
- Plan a small re-annotation subset to estimate label noise (report Cohen's kappa).

### 3.3 Phantom EEG details (templates and ground truth)

- Downey RJ and Ferris DP (2023), Sensors 23(19):8214. Electrically
  conductive gelatin phantom head recorded with a dual-layer BioSemi
  ActiveTwo system (128 scalp channels plus 128 outer noise-layer channels).
  Sampling rate 512 Hz (verified).
- Brain sources: 10 dipolar antennas broadcasting neural-mass-model signals
  (about 20 uV at the scalp), time-aligned to the EEG with triggers.
- Artifact sources: eye (saccade and blink antennas, about 150 uV),
  facial/jaw muscle (bilateral masseter and temporalis antennas, recorded
  chewing, swallowing, drinking EMG, about 300 uV), neck muscle (trapezius
  and sternocleidomastoid antennas, walking EMG, about 150 uV), motion
  (robotic platform reproducing a 1.5 m/s walking head trajectory).
- Conditions (separate recordings): brain only, brain + eyes, brain + facial
  muscle, brain + neck muscle, brain + motion, brain + all. Trials have start
  and stop trigger markers; artifact sources are switched on per condition.
- Extras: 8 neck EMG reference electrodes, noise-layer electrodes; iCanClean
  plugin and example scripts.
- **VERIFIED (2026-10-15, `scripts/04_verify_openneuro.py`):**
  - Each condition (`task-Brain`, `Eyes`, `Facial`, `Neck`, `Walking`,
    `All`) is **one 5-minute trial**, bounded by two `65471` triggers
    (153596 samples apart); other codes (e.g. 65343) are to be ignored.
    EEGLAB .set/.fdt; 128 scalp EEG (A1..D32) + 128 noise layer (`N-A1`..,
    MISC in channels.tsv) + 8 EMG (EXG1-8); reference CMS (unreferenced).
    The EEGLAB reader types all 256 electrodes as EEG, so select scalp
    channels by name.
  - Broadcast waveforms are included: `stimuli/GTdata_croppedToRisingEdge.mat`,
    `GTdata` 153600 x 21 at 512 Hz (ch 1-10 brain, 11-12 eyes, 13-16 neck,
    17-20 facial, 21 trigger), cropped to the sync rising edges. No motion
    channel (motion is mechanical).
  - Alignment: the EEG trial is 4 samples shorter than the ground truth
    (clock skew), so stretch the GT onto the EEG clock. After that and
    average referencing over the 128 scalp channels, the 10 brain sources
    explain a median R^2 of 0.64 of the 1-40 Hz scalp EEG in the brain-only
    condition (peak at lag 0/-1); with artifacts on: eyes 0.24, facial 0.31,
    neck 0.50, walking 0.43, all 0.10. (Corrected 2026-10-15: an earlier
    run mixed in the noise-layer channels.)
    **Use regression on the GT (not brain-only subtraction) to isolate
    artifact fields**, as the plan allows.
  - Electrode positions: `electrodes.tsv` (264 rows, mm, CTF/EEGLAB); A1 at
    (0, 0, 56) suggests idealized positions - check against the phantom
    geometry before spline mapping.
- Artifacts here propagate through a physical volume conductor onto real
  BioSemi electrode positions, so templates are realistic in topography
  without the 19-to-64 spline projection.

### 3.4 Electrode pops and shivering (no public labelled source)

- Electrode pops: simulate with the standard model (abrupt step on one
  channel followed by exponential decay back to baseline; randomize
  amplitude, decay constant and polarity). Document parameters and cite the
  model in the methods.
- Shivering: not included as a separate class (rare even in clinical
  corpora). State this as a limitation; shivering-like activity falls under
  the muscle class.

### 3.5 Joshua's dataset (Section B)

- BioSemi 64-channel EEG plus dedicated EOG channels, recorded with
  ActiView; stimuli presented with TDT, triggers mark stimulus onset.
- n = 13 so far (more subjects coming).
- Stimulus: A Child's History of England (Dickens), LibriVox, male and female
  narrators, 24 segments of about 4 min (12 per narrator), long pauses cropped.
- Spatialization: SLAB KEMAR HRTFs, azimuths around +45 and -45 degrees.
- Four conditions: single speaker (control), diotic (both speakers mixed,
  same to both ears), dichotic (one speaker per ear), binaural
  (HRTF-spatialized). About 6 segments (about 24 min) per condition per subject.
- Attended speaker (male/female) and attended side (left/right) alternate
  across conditions; attended speaker starts slightly earlier; two
  counterbalanced stimulus lists.
- Comprehension questions after each segment.
- Existing analysis code: this repo (`Oiggy/cocktail-cortical`, eelbrain
  boosting TRFs; `predictors/gammatone_predictors.py`,
  `analysis/cortical_analysis*.ipynb`). Reuse its predictor and TRF
  conventions for Section B.
- Details extracted from the parent repo (EOG channels, montage, triggers,
  segment design, subjects, predictors, TRF settings) are in PROGRESS.md
  "Section B dataset", with open discrepancies under "Still open" (notably:
  the code has 12 segments per subject, not 24). Raw data: BDF at
  `../dataset/cocktail/eeg/sN/sN_cocktail.bdf` (4096 Hz), BIDS copy at
  `../dataset/cocktail/bids`.

## 4. Environment

- Python 3.11+.
- Core: `numpy`, `scipy`, `pandas`, `mne`, `mne-bids`, `scikit-learn`,
  `lightgbm` (and/or `xgboost`), `eelbrain`, `matplotlib`.
- Data download: `openneuro-py` (or DataLad) for ds006386 and ds004784;
  direct Zenodo download for Dataset I.
- Comparison method: `mne-icalabel` (ICA + ICLabel).
- TRF/decoding: eelbrain boosting for Section B (matches existing pipeline).
  For Section A, use ridge-regression TRFs to match Das et al. (mTRF-style,
  lambda = 0.2 for short-term TRFs). A Python mTRF implementation (e.g.,
  `mtrf` package) or a small custom ridge is fine; **VERIFY** equivalence on
  a toy example.
- Ridge TRFs: `adr.trf.ridge` matches mTRFpy 2.1.2 (mTRF-Toolbox v2
  convention) to machine precision on a toy example: bias column,
  trial-averaged X'X, penalty lambda * fs, weights x fs (verified
  2026-10-15).
- MWF: implement the GEVD-based MWF of Somers et al. 2018 in Python (the
  original is MATLAB). Validate against a simple synthetic test (known rank-1
  artifact) before use.
- Fix all random seeds; log package versions.

## 5. Repository layout

```
adr/
  PROJECT_PLAN.md
  configs/           # yaml configs: paths, splits, hyperparameters, SNR/SAR grids
  data/              # not committed; raw downloads and caches
  src/adr/
    io/              # loaders: dataset1.py, physiomotion.py, phantom.py, own_dataset.py
    preprocessing/   # filters, resampling, referencing, channel alignment
    envelope/        # gammatone envelope (Das spec) and project predictors
    synth/           # Dataset III synthesis, artifact templates, injection
    detection/       # features.py, models.py, evaluate.py
    reconstruction/  # mwf.py, ridge_frontal.py, spatial_interp.py, router.py
    baselines/       # ica_iclabel.py, mwf_oracle.py, mwf_auto.py (optional)
    trf/             # forward/backward models, metrics (Eq. 14-15), AAD
    stats/           # Wilcoxon + Holm, mixed models, bootstrap CIs
  scripts/           # numbered entry points: 01_download.py ... 09_figures.py
  tests/             # unit tests for every module
  results/           # figures, tables, logs (not committed if large)
  splits/            # frozen split files (committed)
```

## 6. Shared preprocessing and alignment

- **Detector channel set:** the channels whose names exist in both
  PhysioMotion raw (10-10 names expected) and BioSemi 64. **Frozen
  2026-10-15: 57 channels** (`detector_channels` in `configs/project.yaml`);
  features use all 57. Fallback if names do not match:
  the 19 standard 10-20 channels (Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz,
  C4, T8, P7, P3, Pz, P4, P8, O1, O2), renaming old labels (T3->T7, T4->T8,
  T5->P7, T6->P8) if they occur.
- **Phantom to BioSemi 64:** the phantom uses the BioSemi 128 layout (A1 to
  D32 labels, not 10-20 names). Map to BioSemi 64 positions with
  spherical-spline interpolation from the 128-channel positions (dense to
  sparser, so little smoothing).
- **Label taxonomy (used for routing):**
  - `ocular`: blinks, vertical and horizontal eye movements (PhysioMotion);
    phantom eye condition
  - `muscle`: chewing, swallowing, tongue, eyebrow raising (PhysioMotion);
    phantom facial and neck muscle conditions
  - `motion`: head rotation and nodding (PhysioMotion); phantom motion condition
  - `pop`: simulated electrode pops
  - `clean`: PhysioMotion rest blocks without annotations; phantom
    brain-only condition. **Refined 2026-10-15:** the eyes-open rest blocks
    contain many unannotated blinks (Fp1/Fp2 peak-to-peak > 100 uV in
    39-80% of 1 s windows in 8 of 13 runs checked), so rest windows with
    frontal (mean Fp1/Fp2, 0.5-10 Hz) peak-to-peak > 100 uV are dropped
    from training and from the false-alarm denominator
    (`clean_max_frontal_ptp_uv` in configs/detection.yaml).
  - Combined PhysioMotion labels are multi-label (a window can carry more
    than one class).
- **Window labels:** a window is assigned a class if any annotated channel in
  the detector channel set carries that class during the window (OR).
  PhysioMotion annotations exist only on the bipolar montage over 19 10-20
  electrodes (`label_electrodes`) plus `ALL`, so window labels come from
  those pairs (decided 2026-10-15). Also
  report a stricter rule (at least 2 channels), since OR inflates positives.
- **Sampling rates:** PhysioMotion 1000 Hz, phantom 512 Hz. **Two separate
  detectors (decided 2026-10-15):** a 128 Hz detector for Section A and a
  250 Hz detector for Section B, each trained on PhysioMotion resampled to
  that rate. Feature definitions are identical in both; only the Welch band
  powers differ (below 64 Hz in the 128 Hz model, up to about 120 Hz in the
  250 Hz model). Why: one shared model would have to drop everything above
  64 Hz, discarding most of the muscle information Section B can use.
- **Reference for detector features:** average of the detector channel set,
  identical across datasets.

## 7. Detection (supervised, non-neural; trained on PhysioMotion)

- Windows: 0.5 to 1 s, overlapping (e.g., 50 percent).
- Features per window (time-domain or FFT only):
  - variance, peak-to-peak, kurtosis, line length
  - Hjorth activity, mobility, complexity
  - Welch band powers (delta, theta, alpha, beta, low gamma; capped below
    Nyquist: < 64 Hz in the 128 Hz detector, up to ~120 Hz in the 250 Hz
    detector)
  - frontal vs. posterior amplitude gradient
  - correlation with frontal channels (Fp1, Fp2)
- Models: LightGBM/XGBoost (main), SVM RBF, LDA (baseline), optional random
  forest and stacked ensemble.
- Targets: binary (artifact vs clean) and per-class (ocular, muscle, motion,
  pop, clean; multi-label). Routing needs the per-class output.
- Pop class: trained on simulated pops injected into PhysioMotion clean rest
  segments.
- Class imbalance: class weights or balanced sampling.
- **Two feature variants (decided 2026-10-15):** `raw` features, and `norm`
  features robust-z-scored within each recording (median, IQR/1.349).
  Both are trained on PhysioMotion; the variant and the decision threshold
  are chosen on Section A Set A (milestone 9), then frozen.
- Evaluation on PhysioMotion: grouped k-fold by participant. Metrics: PR-AUC,
  ROC-AUC, F1, sensitivity, specificity, event-level detection and false
  alarms per hour. Report cross-validated results separately for the 128 Hz
  and the 250 Hz detector, and freeze both before any test data are touched.
- Transfer evaluation:
  - Phantom: score against condition and trigger labels (artifact source on
    vs off). This tests transfer to a different system and to physically
    propagated artifacts.
  - Section A: score against the injected-artifact log.
  - Section B: score ocular detection against labels derived from Joshua's
    EOG channels (threshold-based blink and saccade labels, spot-checked by hand).
- Label noise: re-annotate a small PhysioMotion subset (and a Section B
  subset), double-annotate part of it, report Cohen's kappa.
- Caveat to report: PhysioMotion artifacts are task-induced and may be more
  stereotyped than spontaneous ones; Section B measures how well detection
  holds on spontaneous artifacts.

## 8. Reconstruction (supervised, non-neural; routed by artifact type)

Only flagged samples are corrected. Clean samples are left untouched.

| Artifact class | Method |
|---|---|
| ocular | MWF fitted per recording using detector flags as labels (Somers et al. 2018), plus frontal-reference ridge regression (Fp1, Fp2, AF7, AF8 as EOG stand-in) fitted within flagged windows |
| motion | MWF fitted per recording on motion flags (spatially broad, low-frequency); compare against spatial interpolation as an ablation |
| muscle, pop | Learned spatial interpolation: ridge regression trained on clean data to predict each channel from its neighbours; replace contaminated samples with the prediction |

- Boundary handling: taper the transition between corrected and uncorrected
  samples (test with and without as an ablation).
- Over-cleaning cap: limit the fraction of data modified; report the fraction
  for every pipeline.
- Tuning: all hyperparameters (MWF lags, rank, regularization; ridge
  penalties; neighbour count; detector threshold) tuned **only** on Section A
  Set A, then frozen.

## 9. Section A: semi-synthetic validation (Dataset I v2.0)

### 9.1 Clean data and envelope (follow Das et al. 2020, NeuroImage 204:116211)

EEG preprocessing for TRF analysis (applied after artifact handling):

1. Start from v2.0 data (already 0.5 Hz high-pass, 128 Hz, MWF-cleaned).
2. Keep only **HRTF** condition trials, **including** the experiment-3
   repetition trials (`repetition == 1`). This gives 38.4-38.7 min per
   subject, matching the paper's "38 min per subject" (Das et al. 2020 §2
   uses all HRTF data and builds noise from "the concatenated 38 min EEG" of
   one subject). Excluding repetitions would leave only 26.0-26.3 min.
   (Decided by Joshua 2026-10-15; earlier draft excluded repetitions.)
   - Consequence: the first 2 min of each experiment-1 story part appear up
     to 4 times per subject (once in the original trial, plus repetitions).
3. Band-pass 1 to 9 Hz.
4. Downsample to 32 Hz.
5. Re-reference to Cz (63 channels remain).
6. z-score each channel.

- The paper does not state the exact order of filtering and downsampling.
  Choose filter-then-downsample, document it. (Implemented as in the
  authors' `preprocess_data.m`: Cz reference, equiripple 1-9 Hz FIR -
  181 taps at 128 Hz meeting the MATLAB spec - applied forward-backward,
  then keep every 4th sample; `adr.preprocessing.das`.)

Envelope:

1. Gammatone filterbank, 15 filters, applied to the **dry** stimulus.
2. Absolute value of each subband.
3. Power-law compression, exponent 0.6.
4. Sum the 15 subbands.
5. Downsample to 32 Hz.
6. Band-pass 1 to 9 Hz.
7. z-score (per lag).

- Implemented in `adr.envelope.das` following `preprocess_data.m`: audio
  resampled to 8 kHz; 15 real 4th-order gammatone FIR filters at
  `erbspacebw(150, 4000, 1.5)` (167-3588 Hz, 1.5-ERB spacing, one at
  1 kHz), bandwidth 1.5 x ERB, unit gain at each centre; |.|^0.6 per band;
  resampled to 128 Hz, band-passed 1-9 Hz (same FIR as the EEG), decimated
  to 32 Hz, then summed. This band-passes at 128 Hz before decimating,
  i.e. steps 5 and 6 above in the authors' order.

### 9.2 Synthetic response (Dataset III recipe)

1. Fit long-term TRFs from the attended envelope to the EEG across all
   subjects (lags 0 to 400 ms = 14 samples at 32 Hz), separately for
   attend-left (W_L) and attend-right (W_R).
2. Find the reference lag where W_L and W_R jointly have maximum norm.
3. Take the C5 TRF as the base shape (one for left, one for right).
4. For every channel, scale that shape by the channel's TRF amplitude at the
   reference lag. This gives W_baseL and W_baseR.
5. Convolve with the attended envelope to get X_L and X_R (38 min per condition).

- Implemented in `adr.synth.dataset3` (milestone 4). Interpretation of
  "38 min per condition": per noise subject, that subject's HRTF attended
  envelopes (~38.5 min) are convolved trial by trial with W_baseL (X_L)
  and with W_baseR (X_R); the same time-reversed noise is added to each,
  giving two recordings per noise subject and 19 x 120 s trials per
  condition (38 total, as in §9.4). The C5 shape is normalised by its value
  at the reference lag, so W_base equals the long-term TRF at C5 and at the
  reference lag. X is multiplied per channel by the noise sd (analysis
  space) because the base TRFs come from z-scored EEG; X is added with
  Cz = 0 (it lives in the Cz-referenced space). For each SNR the signal is
  scaled and the noise keeps its real amplitude.

### 9.3 Noise, SNR and artifacts (our extension)

1. Upsample X to 128 Hz (X is band-limited to 1 to 9 Hz, so this is lossless).
2. Noise: concatenated 128 Hz v2.0 EEG of one subject, **time-reversed**.
3. Scale noise to the target SNR. SNR = 20 x log10(rms(X) / rms(noise)),
   averaged across channels, with both measured **after 1 to 9 Hz
   filtering** so the definition matches the paper.
4. **VERIFIED:** SNR range is 0 dB to -25 dB (Das et al. 2020 §3.2 text,
   PMC7355237: "For all the SNRs analyzed (0 dB to −25 dB)").
5. **Repeat the synthesis once per noise subject** so subject becomes the
   statistical unit.
   - Set A noise subjects (8): tuning.
   - Set B noise subjects (8): testing.
   - Cross-fit: swap A and B and report both folds.
   - Save the split to `splits/` and never change it after tuning starts.
     Frozen 2026-10-15 in `splits/dataset1_noise_subjects.yaml`:
     Set A = S4, S5, S6, S7, S8, S10, S11, S14; Set B = S1, S2, S3, S9,
     S12, S13, S15, S16.
6. Artifact injection at 128 Hz:
   - **Primary templates from the phantom** (ocular, muscle, motion):
     - Isolate the artifact field by subtracting the trigger-aligned
       brain-only signal from each brain + artifact trial. Brain waveforms
       are broadcast identically and time-aligned, so the difference is
       artifact plus sensor noise (**VERIFY** alignment quality; if the
       broadcast brain waveforms are provided, regress them out instead).
     - Keep templates broadband (no 20 Hz split).
     - Map from BioSemi 128 to BioSemi 64 positions (spherical spline), then
       average-reference across 64.
     - Split phantom trials into disjoint pools: Set A templates, Set B
       templates, and a held-out pool for the phantom check (§9.5). Each
       condition is a single 5-min trial, so the pools are **three contiguous
       100 s blocks** (0-100 s, 100-200 s, 200-300 s from the first trigger),
       the same for every condition (`splits/phantom.yaml`, decided 2026-10-15).
   - **Secondary templates from PhysioMotion** (an ablation, to test
     templates recorded on real heads):
     - Use only participants not used to train the detector (participants
       split into three disjoint groups: detector training, Set A templates,
       Set B templates).
     - Isolate artifacts from the underlying EEG: muscle and chewing:
       high-pass about 20 Hz (plus a broadband version); eye movement:
       low-pass frontal channels; motion: low-pass.
     - Map matched channels directly; spline-interpolate the remaining
       BioSemi 64 positions.
   - Electrode pops: simulated (§3.4), injected into one randomly chosen
     electrode, **never Cz** (Cz re-referencing would spread it everywhere).
   - Signal-to-artifact ratio (SAR): -10 to +10 dB, same 20 x log10 RMS definition.
   - Log every injected artifact (type, onset, offset, channels, SAR). This
     log is the ground truth for detection scoring.
7. Processing order: inject artifacts -> run pipeline (detect + reconstruct,
   or baseline) -> 1 to 9 Hz, 32 Hz, Cz reference, z-score -> TRF analysis.

### 9.4 Section A outcomes

- **TRF weights:**
  - Short-term TRFs within each 120 s trial (19 trials per condition, 38
    total), ridge lambda = 0.2.
  - Figure: one trial, channel C5, every pipeline overlaid on the true base
    TRF (style of Das et al. Fig. 3).
  - Topoplots at the reference lag per pipeline.
  - Relative TRF MSE (Das et al. Eq. 14-15): concatenate all channels' TRFs,
    scale the estimate to the base TRF by least squares, then relative MSE.
- **Forward prediction:** cross-validated across the 120 s trials, **grouped
  by stimulus** (decided 2026-10-15): individual 120 s trials are not
  independent. The group label is the original presentation (attended_track
  + part); every 120 s trial cut from one presentation is held out together,
  and any trials that share a stimulus are merged into the same group. Use
  GroupKFold or leave-one-group-out on that label (separate left and right
  TRFs, predict the held-out group). Report correlation with the EEG and with
  the noise-free X. Why: adjacent 120 s trials from one presentation leak
  information between train and test (the trial-fingerprint problem the
  Dataset I authors warn about), and a shared stimulus would let the model
  predict from the stimulus itself instead of from a generalizable TRF.
  - Implemented as `adr.io.dataset1.presentation_groups`. Consequence: in
    Dataset I every trial plays both tracks of one story part, experiment 2
    replays experiment 1's pairs with attention switched, and the
    repetitions are the openings of those parts, so the groups collapse to
    the 4 story parts; each subject's HRTF trials span only 2 of them
    (~19.3 min each). See PROGRESS.md "Still open".
- **Signal level:** RRMSE (time and spectral) and correlation between
  reconstructed and clean synthetic EEG.
- **Detection:** sensitivity, specificity, F1, event-level metrics against
  the injection log.
- **Fraction of data modified.**
- Report over the grid: SNR x SAR x artifact type x pipeline.
- **Ground-truth sensitivity check:** rebuild a non-MWF clean reference
  (e.g., ICA+ICLabel cleaned, or visually clean segments) and confirm
  pipeline rankings hold.
- **Readout check:** repeat key outcomes with a 1 to 30 Hz band, because 1
  to 9 Hz hides residual muscle artifact.

### 9.5 Phantom ground-truth check (real recorded artifacts, known brain signal)

- Use the held-out phantom pool only; nothing here is tuned.
- Run every pipeline (§11) on the brain + artifact recordings (each artifact
  condition and the all-artifacts condition), with frozen settings.
- Ground truth: the brain-only recording (trigger-aligned) or the broadcast
  brain waveforms if provided.
- Metrics: RRMSE (time and spectral) and correlation against ground truth,
  per channel and per artifact condition; detection metrics against the
  condition and trigger labels; fraction of data modified.
- Run at the phantom's native layout (128 channels) and after mapping to
  BioSemi 64, and report both.
- Purpose: answers the reviewer concern that injected artifacts are
  unrealistic, because these artifacts propagated physically and were
  recorded, not added in software.
- Caveat to state: a phantom has no real brain, skin or electrode-skin
  interface, and only one "head".

## 10. Section B: real-data validation (Joshua's dataset)

- No synthetic data. Real artifacts, no ground truth.
- Apply every pipeline (Section 11) with frozen settings, then identical
  downstream analysis.
- Downstream analysis uses eelbrain boosting, matching
  `Oiggy/cocktail-cortical` conventions. Cross-validation is grouped by
  audiobook segment (same rule as §9.4: a segment is never split between
  training and testing, and segments sharing a stimulus share a group).

Outcomes:

- **Forward model (envelope -> EEG):** prediction accuracy per channel and
  per condition. Two-talker conditions include attended and ignored envelopes.
- **TRF plots:** waveforms at a fronto-central channel group with all
  pipelines overlaid; topoplots at P1, N1, P2; peak amplitudes and latencies.
- **Backward model (EEG -> envelope):** reconstruction accuracy per
  condition; auditory attention decoding (AAD) accuracy in diotic, dichotic
  and binaural conditions.
- **Ground-truth substitutes:**
  - Split-half TRF reliability (fit on two halves of the segments, correlate).
  - EOG TRF: fit envelope -> EOG channels to measure stimulus-locked eye
    activity; report how much raw-EEG prediction accuracy it could explain.
  - Backward model without frontal channels: if raw accuracy drops much more
    than cleaned accuracy, raw decoding relied on eye signals.
  - Shifted-envelope null distribution for significance.
  - Detector scored against EOG-derived ocular labels.
  - Fraction of data modified.

Note: this dataset is also used for Joshua's Brodbeck et al. 2020 follow-up
study. Freeze the cleaning pipeline before that analysis. Reporting split
between papers to be agreed with the supervisor.

## 11. Pipelines compared (both sections)

Core comparison:

1. Raw (no artifact handling; only the shared filtering and referencing)
2. Proposed method (detector + routed reconstruction)
3. **ICA + ICLabel (the comparison method)**
4. Clean synthetic data, no artifacts (Section A only; ground-truth
   reference, not a competitor)
5. MWF with oracle flags (Section A only; upper bound used in the
   detection-quality ablation, not a competitor)

Optional extra baseline (add only if time allows):

- MWF with automatic marking (RELAX-style)

### 11.1 ICA + ICLabel specification

- Implementation: `mne` ICA + `mne-icalabel` (ICLabel classifier).
- ICLabel expects average-referenced data band-passed 1 to 100 Hz and an
  extended-infomax ICA. Use
  `mne.preprocessing.ICA(method="infomax", fit_params=dict(extended=True))`
  (or `picard` with `ortho=False, extended=True`, which mne-icalabel also
  accepts). **VERIFY** current mne-icalabel requirements.
- Fit ICA on a copy of the data filtered 1 to 100 Hz (or 1 Hz to just below
  Nyquist when the sampling rate is lower), average-referenced. Apply the
  unmixing to the pipeline's own data.
- Number of components: rank of the data (63 or 64 after referencing; reduce
  if channels were interpolated).
- Component removal rule (fixed before any test data are touched): remove a
  component if its top ICLabel class is eye blink, muscle artifact, heart
  beat, line noise or channel noise with probability >= 0.7. Report results
  for 0.5 and 0.9 as a sensitivity check.
- Section A caveat: the data are at 128 Hz (Nyquist 64 Hz), below ICLabel's
  intended 1 to 100 Hz input. Run it at 1 to 63 Hz, and state this as a
  limitation of the baseline.
- Section B: run ICLabel at 250 Hz or higher so the full 1 to 100 Hz band is
  available.
- Report the number and classes of removed components per subject.
- Note: ICLabel is itself a neural-network classifier. It is allowed here
  because it is a baseline, not part of the proposed method.

## 12. Ablations

- Detector flags vs. oracle vs. heuristic vs. random flags (matched flag rate)
- Routing vs. a single MWF for all artifact types
- Correct flagged samples only vs. filter the whole recording
- Boundary tapering on/off
- MWF lag count and rank
- Detector window length
- Phantom-derived vs. PhysioMotion-derived templates
- Simulated pop parameters (amplitude, decay constant)
- High-passed vs. broadband muscle templates
- SAR definition: all-channel RMS vs. affected-channel RMS
- Over-cleaning cap values

## 13. Statistics

- Unit of inference: subject (noise subject in Section A, participant in Section B).
- Paired Wilcoxon signed-rank tests with Holm correction.
- Linear mixed-effects models:
  - Section A: pipeline x SNR x SAR x artifact type, random effect of subject.
  - Section B: pipeline x condition, random effect of subject.
- Effect sizes with bootstrap 95 percent CIs.
- Report per-subject results, not only means.
- Freeze the analysis plan before Section B (pre-register on OSF if possible).

## 14. Leakage rules (non-negotiable)

- PhysioMotion split by participant into three disjoint groups (detector
  training, Set A templates, Set B templates).
- Phantom trials split into three disjoint pools (Set A templates, Set B
  templates, §9.5 check): contiguous 100 s blocks of each 5-min trial
  (`splits/phantom.yaml`).
- Dataset I split by noise subject (Set A tuning, Set B testing), cross-fitted.
- Never split one trial or segment between training and testing.
- Cross-validation groups are stimulus-based: Section A by original
  presentation, merged across shared stimuli (`presentation_groups`);
  Section B by audiobook segment.
- All tuning on Set A only; settings frozen before Set B and Section B.
- Section B is evaluation only; nothing is tuned on it.

## 15. Reproducibility

- Release code, configs, split files, template-generation scripts and
  trained detectors.
- Respect the Dataset I CC BY-NC-SA 4.0 licence and the PhysioMotion and
  phantom dataset licences (do not redistribute raw data; share code, splits
  and derived summaries).
- Every script reads paths and parameters from `configs/`, writes logs, and
  is deterministic given a seed.
- Unit tests for: envelope extraction, Dataset III synthesis (recover a known
  TRF at high SNR), SNR/SAR scaling, MWF on a synthetic rank-1 artifact,
  Eq. 14-15 metric, split integrity, simulated pop generator, phantom
  template subtraction (brain-only residual near zero on a brain-only vs
  brain-only check), PhysioMotion annotation-to-window labelling.

## 16. Limitations to state in the paper

- Section A background noise was cleaned with MWF (partial circularity with
  MWF-based methods).
- 128 Hz limits muscle-artifact realism in Section A.
- Spline projection smooths focal artifacts.
- Domain shift from the PhysioMotion system (Neuracle) to BioSemi recordings.
- PhysioMotion artifacts are task-induced; no inter-annotator agreement is published.
- Phantom templates come from one physical head with no real tissue or skin interface.
- Electrode pops are simulated; shivering is not modelled as a separate class.
- Eye-gaze bias in Dataset I.
- Small subject numbers.

## 17. Milestones (in order)

1. Download PhysioMotion (ds006386), the phantom dataset (ds004784) and
   Dataset I v2.0 (all open, no registration). Verify the items marked VERIFY.
2. Loaders and channel alignment for all three datasets, with tests.
3. Envelope extraction and Das preprocessing; reproduce a sensible long-term
   TRF on Dataset I.
4. Dataset III synthesis; unit test that a known TRF is recovered at high SNR.
5. PhysioMotion feature extraction (plus simulated pops) and detector
   training; cross-validated results; phantom transfer test.
6. Artifact template extraction and injection; injection log.
7. Reconstruction methods (MWF, frontal ridge, spatial interpolation,
   router), with tests.
8. ICA + ICLabel comparison pipeline (Section 11.1), then the optional
   automatic-MWF baseline if time allows.
9. Section A tuning on Set A, freeze, then Set B and cross-fit.
10. Section B analysis on Joshua's dataset.
11. Ablations and statistics.
12. Figures and tables for the paper.

## 18. Key references

- Das N, Vanthornhout J, Francart T, Bertrand A (2020). Stimulus-aware
  spatial filtering for single-trial neural response and temporal response
  function estimation in high-density EEG with applications in auditory
  research. NeuroImage 204:116211. doi:10.1016/j.neuroimage.2019.116211
- Somers B, Francart T, Bertrand A (2018). A generic EEG artifact removal
  algorithm based on the multi-channel Wiener filter. J Neural Eng 15(3):036007.
- Biesmans W, Das N, Francart T, Bertrand A (2016). Auditory-inspired speech
  envelope extraction methods for improved EEG-based auditory attention
  detection in a cocktail party scenario. IEEE TNSRE 25(5):402-412.
- PhysioMotion Artifact: a task-driven EEG dataset with point-wise motion
  artifact annotations (2026). Sci Data 13:807.
  doi:10.1038/s41597-026-07120-7 (**VERIFY** author list).
- Downey RJ, Ferris DP (2023). iCanClean removes motion, muscle, eye, and
  line-noise artifacts from phantom EEG. Sensors 23(19):8214.
  doi:10.3390/s23198214 (**VERIFY** exact title).
- Ingolfsson TM et al. (2024). Minimizing artifact-induced false-alarms for
  seizure detection in wearable EEG devices with gradient-boosted tree
  classifiers. Sci Rep 14:2980.
- Ronca V et al. (2024). o-CLEAN. J Neural Eng 21:056023.
- Bailey NW et al. (2023). RELAX, Part 1. Clin Neurophysiol 149:178-201.
- Rotaru I et al. (2024). What are we really decoding? J Neural Eng 21:016017.
- Brodbeck C, Jiao A, Hong LE, Simon JZ (2020). Neural speech restoration at
  the cocktail party: Auditory cortex recovers masked speech of both attended
  and ignored speakers. PLoS Biology 18(10): e3000883.
  doi:10.1371/journal.pbio.3000883
