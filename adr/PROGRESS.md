# Progress

Milestones follow PROJECT_PLAN.md §17.

## Done

- **Step 0 - scaffold:** folder layout, plan, environment, configs, config
  loader (paths, seeding, package-version logging), tests.
- **Milestone 1a - Dataset I v2.0:** `scripts/01_download_dataset1.py`
  (Zenodo API, md5-checked, resumable); `adr.io.dataset1` loader;
  `scripts/02_verify_dataset1.py`. Fresh full download (2026-10-15) at
  `~/Github Dataset/aad_dataset_kuleuven_v2` (set in
  `configs/paths.local.yaml`): all 19 files md5-verified, `stimuli.zip`
  extracted to `stimuli/` (32 wavs). Results on all 16 subjects:
  - Channel order = standard BioSemi 64, labels in the files; Cz = col 48
    (0-based 47); C5 = 0-based 13; identical in every subject. Confirmed
    empirically (distance rho -0.42 to -0.90, null >= -0.18).
  - Per subject: 20 trials at 128 Hz, 77.1 min total; 8 x ~6.5 min
    (exp 1-2) + 12 x 124 s (exp 3 repetitions).
  - 38 min = all HRTF trials *including* repetitions (26 min without).
    **Decision (2026-10-15): keep repetitions** to match the paper; plan
    §9.1 step 2 updated.
  - SNR range 0 to -25 dB (paper text).
  - Every dry stimulus is at least as long as its EEG trial.
  - `preprocess_data.m` re-references to Cz *before* band-passing, then
    filters (equiripple FIR, filtfilt) and decimates; for the envelope it
    downsamples to 128 Hz, band-passes, then decimates to 32 Hz. Gammatone
    centre frequencies: `erbspacebw(150, 4000, 1.5)`. Relevant for milestone 3.

- **Milestone 1b - PhysioMotion + phantom:** `scripts/03_download_openneuro.py`
  (OpenNeuro API, pinned snapshots, size-checked) and
  `scripts/04_verify_openneuro.py`. Local copies (in `paths.local.yaml`):
  `~/Github Dataset/physiomotion_ds006386` (raw sub-* + Manual_Annotations,
  24.7 GB; preprocessed_BIDS skipped) and `~/Github Dataset/phantom_ds004784`
  (1.5 GB; the authors' Data/Imported and Data/Preprocessed copies skipped).
  All §3.2/§3.3 VERIFY items resolved - details in PROJECT_PLAN.md. Key facts:
  PhysioMotion labels live on the 34 bipolar pairs (19 electrodes), rest
  blocks are clean; phantom has one 5-min trial per condition, ground-truth
  waveforms included, alignment median R^2 0.64 (scalp channels;
  corrected from 0.68, which wrongly included the noise layer) after
  clock-skew correction.
  **Decisions (2026-10-15):** phantom pools = three contiguous 100 s blocks
  per condition (`splits/phantom.yaml`); detector features on all 57 shared
  channels (frozen in `configs/project.yaml`), window labels from the
  19-electrode bipolar annotations via OR.

- **Answers to open questions (2026-10-15)** - applied in PROJECT_PLAN.md:
  1. **CV with repeated stimuli:** hold shared stimuli out together; 120 s
     trials are not independent. Section A group = original presentation
     (attended_track + part), merged across any shared stimulus, used with
     GroupKFold / leave-one-group-out (§9.4, §14). Section B group =
     audiobook segment (§10). Implemented for Dataset I as
     `adr.io.dataset1.presentation_groups` (tested). It shows that the
     groups collapse to the 4 story parts (see Still open).
  2. **Detector sampling rate:** two detectors, 128 Hz (Section A) and
     250 Hz (Section B), each trained on PhysioMotion resampled to that
     rate; identical features except Welch bands (< 64 Hz vs up to ~120 Hz);
     both frozen before test data, cross-validated results reported for
     each (§6, §7).
  3. **Section B dataset details:** extracted from the parent repo, below.
  4. **Citation:** Brodbeck, Jiao, Hong & Simon (2020), PLoS Biol
     18(10):e3000883 - added to PROJECT_PLAN.md §18.

- **Milestone 2 - loaders and channel alignment (2026-10-15):**
  - `adr.io.physiomotion`: raw EDF (unfiltered, volts, `standard_1005`
    montage), annotations, label -> class mapping (`blink`/`hor_eyem`/
    `ver_eyem` -> ocular; `hor_headm`/`ver_headm` -> motion; `tongue`/
    `chew`/`swallow`/`eyebrow` -> muscle; combinations are multi-label),
    `window_labels` (OR rule or >= N pairs; `ALL` counts as all 34 pairs;
    rest-block windows are `clean`; unannotated task windows are
    `unlabelled`, not assumed clean), `bad_channels` (flat / noisy on a
    0.5-40 Hz copy - the raw data are DC-dominated). Found AF4 flat in
    sub-1 runs 1-2; full bad-channel survey comes with milestone 5.
  - `adr.io.phantom`: one trial per condition, scalp / noise layer / EMG
    split by channel name (the EEGLAB reader types all 256 electrodes as
    EEG), electrode positions converted from CTF/EEGLAB axes to MNE head
    axes (checked: A1 vertex, A row posterior, B right, C anterior, D
    left), ground truth stretched onto the EEG clock, pool slices from
    `splits/phantom.yaml`.
  - `adr.preprocessing.channels`: pick/reorder by name, average reference,
    spherical-spline interpolation of bad channels, polyphase resampling
    (1000 -> 128/250, 512 -> 128/250).
  - Dataset I loader unchanged apart from `presentation_groups`.
  - 22 tests pass (data tests skip when a dataset is absent).
  - Correction: the phantom alignment R^2 was re-run on scalp channels only
    (0.64 brain-only; see above).

- **Milestone 3 - envelope, Das preprocessing, long-term TRF (2026-10-15):**
  - `adr.envelope.das`, `adr.preprocessing.das`, `adr.preprocessing.section_a`,
    `adr.trf.ridge`, `configs/section_a.yaml`, `scripts/05_long_term_trf.py`.
  - Ridge TRF verified against mTRFpy 2.1.2 (installed only in the session
    scratchpad, not the environment): identical weights and bias once the
    mTRF conventions are applied (bias column, trial-averaged covariance,
    lambda * fs, weights x fs). `tests/test_trf_ridge.py` repeats the check
    when `mtrf` is installed.
  - Data: 160 HRTF trials, 16 subjects, 38.5 min/subject.
  - With lambda = 0.2 the long-term TRF was essentially unregularised and
    alternated sign lag to lag. Switched the long-term fit to Das et al.'s
    own rule (penalty = max |X'X|, mTRF lambda ~231): smooth TRF, same
    held-out prediction (r 0.0178 vs 0.0177).
  - Result (`results/section_a/long_term_trf.png`): C5 peaks at ~156 ms,
    trough ~250 ms, second peak ~312 ms; reference lag (max joint norm of
    W_L, W_R) = 250 ms, with fronto-central positive / posterior negative
    topography; W_L and W_R topographies correlate 0.96 there.
    Story-part-held-out prediction r = 0.018 (null -0.000; 88% of trials
    beat their null; best channel P9, r = 0.039).
  - 32 tests pass (1 skipped: mtrf comparison).

- **Milestone 4 - Dataset III synthesis (2026-10-15):**
  - `adr.synth.dataset3` (base TRFs, X_L/X_R, upsampling, Cz handling,
    noise from the time-reversed HRTF EEG, SNR measurement and scaling,
    120 s trial cutting), `adr.trf.metrics.relative_mse` (Eq. 14-15),
    `scripts/06_synthesize_dataset3.py`, SNR grid in `configs/section_a.yaml`.
  - Noise-subject split frozen: Set A = S4, S5, S6, S7, S8, S10, S11, S14;
    Set B = S1, S2, S3, S9, S12, S13, S15, S16
    (`splits/dataset1_noise_subjects.yaml`, project seed).
  - Unit test (plan §15): a known TRF is recovered through synthesis ->
    Das preprocessing -> ridge at +20 dB (relative MSE < 0.02) and recovery
    degrades at -20 dB. Eq. 14-15 metric tested (scale-free; 1 for
    orthogonal or zero estimates).
  - Real-data check on Set A subject S4 (no artifacts yet): every target SNR
    is hit exactly; 19 x 120 s trials per condition. Long-term (38 min)
    recovery relative MSE 0.10-0.16 across 0 to -25 dB (best at -10 dB). Short-term 120 s
    TRFs: median relative MSE 0.29-0.35 at 0 dB rising to 0.94 at -25 dB
    with lambda 0.2, versus 0.15 rising to 0.47-0.48 with the Das rule
    (`results/section_a/dataset3_check.png`) - relevant to open item 11.
  - Per-channel z-scoring slightly flattens the topography when the signal
    is strong (0 dB), so recovery there is a little worse than at -10 dB;
    this is inherent to the Das design and is reported, not corrected.
  - 37 tests pass (1 skipped: mtrf comparison).

- **Notebooks and runnable steps (2026-10-15):** script logic moved into the
  package (`adr.verify`, `adr.trf.long_term`, `adr.synth.check`,
  `adr.detection.train`, `adr.detection.transfer`); scripts are thin
  wrappers. Notebooks `notebooks/01_data_checks`, `02_long_term_trf`,
  `03_dataset3_synthesis`, `04_detectors` (executed, outputs saved); run
  instructions in README.md. New `adr` conda env (LightGBM 4.7, XGBoost 3.4,
  Python 3.14) from `environment.yml`.
- **Bug fixed:** Dataset I .mat files store microvolts; the loader now returns
  volts like the other datasets. TRF/SNR results were unaffected
  (scale-free); detectors and pops would not have been.
- **Milestone 5 - detectors (2026-10-15):**
  - PhysioMotion split frozen: 20 detector-training / 5 Set A templates /
    5 Set B templates (`splits/physiomotion_participants.yaml`).
  - Features (`adr.detection.features`, `configs/detection.yaml`): 1 s
    windows, 0.5 s step; per channel log variance, log peak-to-peak,
    kurtosis, log line length, Hjorth mobility/complexity, log Welch band
    powers (1-4, 4-8, 8-13, 13-30, 30-45 Hz; + 65-95, 102-118 Hz at
    250 Hz - bands avoid both mains families), correlation with mean
    Fp1/Fp2; plus frontal/posterior RMS ratio. 685 features at 128 Hz,
    799 at 250 Hz. Detector preprocessing (`adr.detection.prep`):
    resample -> (pops) -> 0.5 Hz high-pass -> 57 channels -> average ref.
  - Simulated pops (`adr.synth.pops`): step + exponential decay, 50-500 uV,
    tau 0.05-0.5 s, random sign, never Cz; one per ~5 s of rest.
  - Bad channels: robust rule (median over 1 s windows of the per-window sd);
    the first version flagged healthy frontal channels from blinks. Only
    AF4 in sub-1 runs 1-2 is bad.
  - **Clean class refined (decided):** eyes-open rest blocks contain many
    unannotated blinks, so rest windows with frontal peak-to-peak
    > 100 uV are dropped (`clean_max_frontal_ptp_uv`).
  - CV (participant-grouped 5-fold; 92k labelled windows incl. 6.8k pop
    examples), PR-AUC:

    | target | 128 LDA | 128 LGBM | 128 SVM | 250 LDA | 250 LGBM | 250 SVM |
    |---|---|---|---|---|---|---|
    | artifact | 0.945 | 0.970 | 0.964 | 0.938 | 0.969 | 0.964 |
    | ocular | 0.879 | 0.923 | 0.917 | 0.861 | 0.917 | 0.905 |
    | muscle | 0.801 | 0.871 | 0.848 | 0.846 | 0.909 | 0.896 |
    | motion | 0.764 | 0.887 | 0.874 | 0.735 | 0.882 | 0.874 |
    | pop | 0.258 | 0.866 | 0.483 | 0.248 | 0.852 | 0.435 |

    LightGBM ROC-AUC 0.94-0.97; event sensitivity 0.87-0.99. At the default
    0.5 threshold, specificity for artifact-vs-clean is only 0.73-0.78 and
    false alarms are 89-618 per hour of clean rest (threshold is tuned on
    Section A Set A at milestone 9). The 250 Hz detector is clearly better
    for muscle (PR-AUC 0.909 vs 0.871; FA/h 89 vs 122), as intended.
    RBF SVM trained on a 20k stratified subsample per fold.
  - Frozen models: `results/detectors/<rate>Hz/<model>_<target>.joblib`.
  - **Phantom transfer** (held-out 200-300 s pool, spline-mapped; leave-one-
    out spline accuracy r = 0.976): LDA/SVM rank artifact conditions well
    (AUC vs brain-only 0.91-1.00 for eyes/facial/neck/all; motion weak,
    0.23-0.79), but LightGBM does not (motion 0.23-0.68, neck muscle at
    128 Hz 0.45) and flags most brain-only windows: the phantom's spectrum
    is flatter (Hjorth mobility 1.26 vs 0.60 for PhysioMotion), so tree
    thresholds learned on real EEG do not carry over.
  - **Domain gap on clean Dataset I (Set A, 128 Hz, artifact-free HRTF EEG):**
    at the default threshold the detectors flag a median 40-66 % of windows
    as artifact (LightGBM 66 %, by subject 19-84 %; ocular 37 %, muscle
    16 %, motion 11 %, pop 1 %). Amplitudes match PhysioMotion after the
    unit fix, so this is system/subject shift (plus possibly residual
    activity MWF does not remove). See "Still open" item 12.
  - 46 tests pass (1 skipped: mtrf comparison).

- **Detector variants (item 12 decided 2026-10-15: option c):** a `norm`
  variant robust-z-scores every feature within its recording (median,
  IQR/1.349; `features.normalize`, `--variant norm`), trained alongside the
  raw one (LightGBM + LDA; the slow RBF SVM was not retrained). The choice
  between variants, and the threshold, is made on Section A Set A at
  milestone 9. Models: `results/detectors/<rate>Hz/norm/`.
  - Participant-grouped CV favours `norm` everywhere (LightGBM PR-AUC
    128 Hz raw -> norm: artifact 0.970 -> 0.980, ocular 0.923 -> 0.939,
    muscle 0.871 -> 0.955, motion 0.887 -> 0.928, pop 0.866 -> 0.906;
    false alarms per hour down 25-60 %, e.g. muscle 122 -> 48;
    artifact-vs-clean specificity 0.78 -> 0.90). 250 Hz similar.
  - Clean Dataset I (Set A, 128 Hz, default threshold): LightGBM flags
    68 % of windows (raw 66 %), now uniform across subjects (65-73 % vs
    19-84 %). Likely cause: PhysioMotion runs are mostly artifact-laden task
    time, so a run's median window is artifact-heavy and a clean
    recording's median looks like "typical training run".
  - Phantom: the condition-level test compares *between* recordings, which
    the `norm` variant cancels by construction (AUCs ~0.5), so it cannot
    evaluate `norm`. A within-recording phantom test needs window labels
    from the broadcast artifact-source waveforms (GT channels 11-20).
  - Notebook 04 shows the comparison.

## Section B dataset (from the parent repo, read 2026-10-15)

Sources: `analysis/biosemi64mod.txt`, `analysis/cortical_analysis_1.ipynb`
(cells cited by number), `analysis/experiment.py`, `data/bids_extraction.py`,
`diagnostics/check_montage.py`, `predictors/*.py`. Items marked *(data)*
were read from the local BIDS metadata in `../dataset/cocktail/bids` because
the code does not state them.

**a. EOG and external channels**
- EXG1-4 are typed `eog` (`data/bids_extraction.py:55-59`) and renamed
  EXG1 -> `EOG-LS`, EXG2 -> `EOG-LI`, EXG3 -> `EOG-L`, EXG4 -> `EOG-R`
  (`data/bids_extraction.py:93`). Placement: Ex1 above left eye, Ex2 below
  left eye, Ex3 next to left eye, Ex4 next to right eye
  (`diagnostics/check_montage.py:61-64`).
- EXG5/EXG6 are the left/right mastoids (`diagnostics/check_montage.py:65-66`),
  renamed `A1`/`A2` (`data/bids_extraction.py:94`); not given a type, so
  they stay EEG (*(data)* `EEGChannelCount` 66 = 64 + 2 mastoids). Used as
  the reference (b).
- EXG7, EXG8, GSR1, GSR2, Erg1, Erg2, Resp, Plet, Temp typed `misc`
  (`data/bids_extraction.py:60-68`; *(data)* `MiscChannelCount` 9). None is
  dropped explicitly; they are simply not EEG.

**b. Montage and reference**
- BioSemi A1-A32/B1-B32 renamed to 10-10 names before writing BIDS
  (`data/bids_extraction.py:76-90`) - same standard BioSemi 64 order as
  Dataset I. `diagnostics/check_montage.py:72-77` does the same via
  biosemi128 -> biosemi64 names.
- Positions from the custom layout `analysis/biosemi64mod.txt` (theta/phi,
  64 electrodes + Nz, LPA, RPA, A1, A2; `biosemi64mod.txt:1-70`), loaded at
  `analysis/experiment.py:155`.
- Recorded CMS/DRL (*(data)* `EEGReference` n/a). Analysis reference: linked
  mastoids `['A1', 'A2']` (`analysis/experiment.py:236`, `:261`).

**c. Trigger codes**
- `1` or `2`: segment (story) onset; the value is the stimulus list
  (`analysis/experiment.py:797-811`; `data/bids_extraction.py:101-105`).
  *(data)* every subject has exactly 12 onsets, all with one value.
- `65536`: spurious amplifier value, removed (`data/bids_extraction.py:100`,
  `:120`).
- `8`: in `EVENT_ID` (`data/bids_extraction.py:104`) but never used by the
  analysis. *(data)* it falls within 0.1 s of the attended (fg) stimulus's
  end, i.e. it marks fg offset, but is present for only 6-11 of the 12
  segments per subject. See Still open.

**d. Segment design** (`analysis/experiment.py:163-186`, `:797-825`)
- 12 segments per subject, fixed order; condition by position:
  clean, binaural, dichotic, diotic, binaural, dichotic, repeated
  (`SPATIAL`, `:175-177`). Per subject: clean 2, binaural 4, dichotic 4,
  diotic 2.
- Foreground (attended, `fg`) speaker: list 1 male for segments 1-6, female
  for 7-12; list 2 the reverse (`SPEAKER`, `:165-168`). Background
  (ignored, `bg`) = the other speaker, same index; none in clean segments 1
  and 7 (`:819-820`). Mixture = `List_<list>_stim_<i>` (`:821`).
- `side` (`SIDE`, `:180-186`), per the code comment "which ear the
  background story came from" (`:179`):

  | seg | condition | list 1 fg / bg / side | list 2 fg / bg / side |
  |---|---|---|---|
  | 1 | clean | male_1 / - / - | female_1 / - / - |
  | 2 | binaural | male_2 / female_2 / right | female_2 / male_2 / left |
  | 3 | dichotic | male_3 / female_3 / left | female_3 / male_3 / right |
  | 4 | diotic | male_4 / female_4 / - | female_4 / male_4 / - |
  | 5 | binaural | male_5 / female_5 / left | female_5 / male_5 / right |
  | 6 | dichotic | male_6 / female_6 / right | female_6 / male_6 / left |
  | 7 | clean | female_7 / - / - | male_7 / - / - |
  | 8 | binaural | female_8 / male_8 / left | male_8 / female_8 / right |
  | 9 | dichotic | female_9 / male_9 / right | male_9 / female_9 / left |
  | 10 | diotic | female_10 / male_10 / - | male_10 / female_10 / - |
  | 11 | binaural | female_11 / male_11 / right | male_11 / female_11 / left |
  | 12 | dichotic | female_12 / male_12 / left | male_12 / female_12 / right |

- *(data)* list 1: subjects 01, 03, 05, 07, 09, 10, 12; list 2: 02, 04, 06,
  08, 11, 13.
- Epochs: whole segment from -0.2 s to the fg duration
  (`analysis/experiment.py:831`); two-talker conditions start at 3 s
  (`:833-837`), consistent with the attended speaker starting earlier.
  `dichotic-left`/`-right` split by `side` (`:835-836`).

**e. Subjects**
- All `s<N>` folders are converted (`data/bids_extraction.py:44-47`) and
  `'all'` means every BIDS subject (`analysis/experiment.py:294`). The
  notebook runs subjects 01-13 (cell 0 says 13 subjects; cell 3/4 outputs
  list 01-13; every test reports n=13). No exclusion list anywhere.

**f. Raw data**
- `../dataset/cocktail/eeg/s<N>/s<N>_cocktail.bdf`
  (`data/bids_extraction.py:36-37`, `:110`), converted to
  `../dataset/cocktail/bids/sub-XX/eeg/sub-XX_task-cocktail_eeg.bdf`
  (re-exported BDF with renamed channels; `:122-151`, `:155-159`).
- Sampling rate 4096 Hz (*(data)* `eeg.json`; also the comment at
  `analysis/experiment.py:740-743`). Recordings 3289-4066 s.
- Stimuli in `../dataset/cocktail/stimuli` (`male_1..12`, `female_1..12`,
  `List_1/2_stim_1..12`; `predictors/gammatone_predictors.py:50-53`);
  predictors in `bids/derivatives/predictors`
  (`predictors/gammatone_predictors.py:38`).

**g. Predictors and envelope**
- eelbrain `gammatone_bank(wav, 80, 8000, 128, tstep=0.001)`: 128 bands,
  80-8000 Hz, 1 ms steps; stereo files done per ear and averaged
  (`predictors/gammatone_predictors.py:64-71`).
- Log compression `log(x + 1)` (`:89`); onsets via
  `edge_detector(gt_log, c=30)` (`:93`).
- 1-band = sum over frequency (`:97-98`); 8-band = `bin(nbins=8,
  func='sum')` (`:103-106`). No filtering of the predictors in code;
  `UTSPredictor(resample='bin')` bins them to the analysis rate
  (`analysis/experiment.py:854-856`).
- Attended = `fg` (no prefix), ignored = `bg~`, mixture = `mix~`
  (notebook cell 7). Overt/masked onsets per Brodbeck 2020
  (`predictors/gammatone_predictors.py:108-200`, validated by
  `predictors/validate_masked_overt_onsets.py`); mixture onset loudness
  tertiles (`predictors/gammatone_predictors.py:202-281`).
- Duration table `SEGMENT_DURATION` (`analysis/experiment.py:114-139`),
  produced by `predictors/duration.py`.

**h. Current TRF settings**
- `Boosting(basis=0.050, error='l1', partitions=-4, selective_stopping=1)`
  (`analysis/experiment.py:275-277`).
- `PARAMETERS`: `raw='ica'`, `samplingrate=128`, `data='eeg'`,
  `tstart=-0.100`, `tstop=0.600`, `group='all'` (notebook cell 11);
  metric `r` (cell 9); ROI of 24 fronto-central sensors (cell 11).
- Preprocessing before the TRF (`analysis/experiment.py:198-266`): bad
  channels (automatic neighbour correlation r < 0.3, from scratch; cell 3)
  -> 0.5-20 Hz filter -> linked-mastoid reference -> ICA (fit with
  decim=16) with components removed automatically by ICLabel at >= 0.75
  for eye blink, muscle, heart, line noise, channel noise (cell 4,
  `analysis/experiment.py:161`, `:686-795`). Note that the existing
  analysis already uses ICA, so Section B pipelines must start from
  `raw='raw'`, not `raw='ica'`.

## Still open (please answer)

1. **Number of segments.** PROJECT_PLAN.md §3.5 says 24 segments of ~4 min,
   ~6 per condition (~24 min). The code and the data say 12 segments per
   subject (12 onsets in every recording): clean 2, binaural 4, dichotic 4,
   diotic 2. The 24 wavs are 12 male + 12 female, one of each per segment.
   Which is right?
2. **Binaural.** The plan says HRTF-spatialized (KEMAR, about +/-45 deg).
   `analysis/experiment.py:172` says "both stories play in both ears (no
   side difference)", yet `SIDE` (`:180-186`) gives a left/right for every
   binaural segment. Which is it?
3. **What `side` means.** The comment (`analysis/experiment.py:179`) says
   it is the ear the *background* story came from; the plan talks about the
   *attended* side. Which ear does `side` give? (It decides what
   `dichotic-left`/`-right` mean.)
4. **Trigger 8.** It matches the attended stimulus's offset but is missing
   for 1-6 segments per subject. Is it the offset trigger, and why is it
   sometimes missing? Is there a trigger for the comprehension questions?
5. **TRF sampling rate.** The epoch is defined at 64 Hz
   (`analysis/experiment.py:831`) but `PARAMETERS` asks for 128 Hz
   (notebook cell 11). Which is intended?
6. **Section B cross-validation.** The current code uses boosting's
   `partitions=-4` for early stopping. Under your answer 1, Section B is
   grouped by segment. Should the forward/backward models use
   leave-one-segment-out (12 folds; or per condition, 2-4 folds), with
   `partitions` only for early stopping inside the training segments?
7. **EOG derivation.** Should VEOG = `EOG-LS` - `EOG-LI` and HEOG = `EOG-L` -
   `EOG-R` for the EOG-based ocular labels? What are EXG7 and EXG8?
8. **Subjects.** All 13 are included and nothing is excluded in code. Should
   any be excluded? (sub-08 has the longest recording, 4066 s.)
9. **Section A groups.** With your grouping rule, Dataset I has only 4
   groups (one per story part), and each subject's HRTF trials cover just
   2 of them (~19.3 min each). Within-subject leave-one-group-out is
   therefore 2-fold, replacing the plan's 38-fold leave-one-trial-out. Is
   2-fold acceptable, or should the synthetic forward model pool subjects
   (still leaving whole story parts out)?
10. **`../dataset/cocktail/predictors`.** This folder exists next to
    `bids/derivatives/predictors`, but no code reads or writes it. Is it
    stale?
11. **Short-term TRF lambda (Section A, needed at milestone 9).** The plan
    sets lambda = 0.2 for the 120 s short-term TRFs. In the mTRF convention
    that is essentially no regularisation (the long-term TRF zig-zagged
    with it). Keep 0.2 as written, or use Das et al.'s per-trial rule
    (penalty = max |X'X|, as now used for the long-term TRF)? Milestone 4
    preview on Set A subject S4 without artifacts: median relative MSE of
    120 s TRFs 0.29-0.94 with lambda 0.2 vs 0.15-0.48 with the Das rule
    across 0 to -25 dB.
12. **Detector domain gap - DECIDED (c), see Done.** On clean Dataset I
    EEG the frozen detectors flag 40-66 % of windows at the default
    threshold, and LightGBM does not transfer to the phantom. Options:
    (a) tune only the threshold on Section A Set A (plan as written);
    (b) add per-recording feature normalisation (robust z-score of each
    feature within a recording) and retrain; (c) train both variants and
    choose between them on Section A Set A (allowed: Set A is for tuning).

## Next

Order agreed 2026-10-15: finish everything for Section A first - detection,
reconstruction, the ICA baseline, semi-synthetic validation on Dataset I
v2.0 and the phantom check (§9.5) - then the own-dataset loader and
Section B (real-data validation). Still-open items 1-8 and 10 wait until
Section B; items 9 and 11 are needed at Section A evaluation (milestone 9).

- Milestone 6: artifact template extraction (phantom primary, PhysioMotion
  secondary) and injection with the injection log. Suggested addition:
  window-level phantom labels from the broadcast artifact-source waveforms,
  for a within-recording transfer test that can evaluate both variants.
- Milestones 7-9 (Section A), then 10 (Section B), 11 (ablations and
  statistics), 12 (figures).
