# Project Plan: Supervised Non-Neural Artifact Detection and Reconstruction for Continuous Speech-Listening EEG

Owner: Joshua Ighalo (PhD student, McMaster University)
Target journal: Journal of Neural Engineering (methods paper)
Status: planning complete, implementation started (see `PROGRESS.md`)

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

- A detector trained on clinical EEG (TUAR) transfers to research-grade
  64-channel EEG; the domain gap is quantified.
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
| TUAR v3.0.1 (TUH EEG Artifact Corpus) | Detector training; artifact templates | https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ (free registration, data-use agreement) |
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

### 3.2 TUAR v3.0.1 details

- Clinical scalp EEG, EDF files under `edf/01_tcp_ar/` (signals in
  average-reference configuration), annotations in matching `.csv` files
  using TCP bipolar channel pairs.
- Artifact labels: `eyem` (eye movement), `chew`, `shiv` (shivering), `musc`
  (muscle), `elec` (electrode pop, electrostatic, lead), plus combinations
  (e.g., `eyem_chew`).
- Annotations are per bipolar channel (about 22 TCP pairs).
- Mixed sampling rates (250, 256, 400, 512, 1000 Hz; most at 250 Hz).
- **VERIFY** from the v3.0.1 README after download: file/patient counts,
  exact label strings, whether every artifact is annotated (sources
  disagree), channel naming.
- No published TUAR-specific interrater agreement. Plan a re-annotation
  subset to estimate label noise.

### 3.3 Joshua's dataset (Section B)

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
- **ASK JOSHUA:** EOG channel names, trigger codes, segment-to-condition
  mapping, which subjects are included. (Raw data location and format are
  known: BDF at `../dataset/cocktail/eeg/sX/sX_cocktail.bdf`, BIDS copy at
  `../dataset/cocktail/bids`.)

## 4. Environment

- Python 3.11+.
- Core: `numpy`, `scipy`, `pandas`, `mne`, `scikit-learn`, `lightgbm`
  (and/or `xgboost`), `eelbrain`, `matplotlib`.
- Comparison method: `mne-icalabel` (ICA + ICLabel).
- TRF/decoding: eelbrain boosting for Section B (matches existing pipeline).
  For Section A, use ridge-regression TRFs to match Das et al. (mTRF-style,
  lambda = 0.2 for short-term TRFs). A Python mTRF implementation (e.g.,
  `mtrf` package) or a small custom ridge is fine; **VERIFY** equivalence on
  a toy example.
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
    io/              # loaders: dataset1.py, tuar.py, own_dataset.py
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

- **Channel alignment (TUAR to BioSemi 64):** use the 19 shared 10-20
  channels: Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4,
  P8, O1, O2. Rename TUAR T3->T7, T4->T8, T5->P7, T6->P8. Drop A1, A2.
- **TUAR labels:** map each bipolar pair label to its two electrodes. Primary
  rule: a window is artifact if any pair in it is labelled (OR). Also report
  alternative mappings, since OR inflates positives.
- **Sampling rates:**
  - Detector features: compute at a common rate. Section A runs at 128 Hz
    (Dataset I v2.0), so Welch band powers must stay below 64 Hz there.
    Section B can run at 250 Hz (keeps more muscle content). Train the
    detector so it can operate at both rates, or train two versions;
    document the choice.
  - Document how each TUAR sampling rate is resampled.
- **Reference for detector features:** average of the 19 shared channels,
  identical across datasets.

## 7. Detection (supervised, non-neural; trained on TUAR)

- Windows: 0.5 to 1 s, overlapping (e.g., 50 percent).
- Features per window (time-domain or FFT only):
  - variance, peak-to-peak, kurtosis, line length
  - Hjorth activity, mobility, complexity
  - Welch band powers (delta, theta, alpha, beta, low gamma; capped below Nyquist)
  - frontal vs. posterior amplitude gradient
  - correlation with frontal channels (Fp1, Fp2)
- Models: LightGBM/XGBoost (main), SVM RBF, LDA (baseline), optional random
  forest and stacked ensemble.
- Targets: binary (artifact vs clean) and multiclass (eyem, chew, shiv, musc,
  elec, clean). Routing needs the multiclass output.
- Class imbalance: class weights or balanced sampling.
- Evaluation on TUAR: grouped k-fold by patient. Metrics: PR-AUC, ROC-AUC,
  F1, sensitivity, specificity, event-level detection and false alarms per hour.
- Transfer evaluation:
  - Section A: score against the injected-artifact log.
  - Section B: score ocular detection against labels derived from Joshua's
    EOG channels (threshold-based blink and saccade labels, spot-checked by hand).
- Label noise: re-annotate a small TUAR subset (and a Section B subset),
  double-annotate part of it, report Cohen's kappa.

## 8. Reconstruction (supervised, non-neural; routed by artifact type)

Only flagged samples are corrected. Clean samples are left untouched.

| Artifact class | Method |
|---|---|
| eyem | MWF fitted per recording using detector flags as labels (Somers et al. 2018), plus frontal-reference ridge regression (Fp1, Fp2, AF7, AF8 as EOG stand-in) fitted within flagged windows |
| musc, chew, shiv, elec | Learned spatial interpolation: ridge regression trained on clean data to predict each channel from its neighbours; replace contaminated samples with the prediction |

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
  Choose filter-then-downsample, document it.

Envelope:

1. Gammatone filterbank, 15 filters, applied to the **dry** stimulus.
2. Absolute value of each subband.
3. Power-law compression, exponent 0.6.
4. Sum the 15 subbands.
5. Downsample to 32 Hz.
6. Band-pass 1 to 9 Hz.
7. z-score (per lag).

### 9.2 Synthetic response (Dataset III recipe)

1. Fit long-term TRFs from the attended envelope to the EEG across all
   subjects (lags 0 to 400 ms = 14 samples at 32 Hz), separately for
   attend-left (W_L) and attend-right (W_R).
2. Find the reference lag where W_L and W_R jointly have maximum norm.
3. Take the C5 TRF as the base shape (one for left, one for right).
4. For every channel, scale that shape by the channel's TRF amplitude at the
   reference lag. This gives W_baseL and W_baseR.
5. Convolve with the attended envelope to get X_L and X_R (38 min per condition).

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
6. Artifact injection at 128 Hz:
   - Templates from TUAR. TUAR patients are split into three non-overlapping
     groups: detector training, Set A templates, Set B templates.
   - Isolate artifacts from the patient EEG: muscle, chewing, shivering:
     high-pass about 20 Hz (also test a broadband version as an ablation);
     eye movement: low-pass frontal channels; electrode pops: subtract a
     smoothed baseline.
   - Spread-out artifacts: project from 19 to 64 channels with
     spherical-spline interpolation, then average-reference across 64.
   - Electrode pops: inject into one randomly chosen electrode, **never Cz**
     (Cz re-referencing would spread it everywhere).
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
- **Forward prediction:** leave-one-trial-out across the 38 trials (train on
  37, separate left and right TRFs, predict the held-out trial). Report
  correlation with the EEG and with the noise-free X.
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

## 10. Section B: real-data validation (Joshua's dataset)

- No synthetic data. Real artifacts, no ground truth.
- Apply every pipeline (Section 11) with frozen settings, then identical
  downstream analysis.
- Downstream analysis uses eelbrain boosting, leave-one-segment-out
  cross-validation, matching `Oiggy/cocktail-cortical` conventions.

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
- Spline-projected vs. native-topography templates
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

- TUAR split by patient into three disjoint groups (detector training, Set A
  templates, Set B templates).
- Dataset I split by noise subject (Set A tuning, Set B testing), cross-fitted.
- Never split one trial or segment between training and testing.
- All tuning on Set A only; settings frozen before Set B and Section B.
- Section B is evaluation only; nothing is tuned on it.

## 15. Reproducibility

- Release code, configs, split files, template-generation scripts and
  trained detectors.
- Respect the Dataset I CC BY-NC-SA 4.0 licence and the TUAR data-use
  agreement (do not redistribute raw data).
- Every script reads paths and parameters from `configs/`, writes logs, and
  is deterministic given a seed.
- Unit tests for: envelope extraction, Dataset III synthesis (recover a known
  TRF at high SNR), SNR/SAR scaling, MWF on a synthetic rank-1 artifact,
  Eq. 14-15 metric, split integrity.

## 16. Limitations to state in the paper

- Section A background noise was cleaned with MWF (partial circularity with
  MWF-based methods).
- 128 Hz limits muscle-artifact realism in Section A.
- Spline projection smooths focal artifacts.
- Domain shift from clinical TUAR to research EEG.
- TUAR label reliability is not published.
- Eye-gaze bias in Dataset I.
- Small subject numbers.

## 17. Milestones (in order)

1. Register for TUAR; download TUAR v3.0.1 and Dataset I v2.0. Verify the
   items marked VERIFY.
2. Loaders and channel alignment for all three datasets, with tests.
3. Envelope extraction and Das preprocessing; reproduce a sensible long-term
   TRF on Dataset I.
4. Dataset III synthesis; unit test that a known TRF is recovered at high SNR.
5. TUAR feature extraction and detector training; TUAR cross-validated results.
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
- Hamid A et al. (2020). The Temple University Artifact Corpus: an annotated
  corpus of EEG artifacts. IEEE SPMB.
- Ingolfsson TM et al. (2024). Minimizing artifact-induced false-alarms for
  seizure detection in wearable EEG devices with gradient-boosted tree
  classifiers. Sci Rep 14:2980.
- Ronca V et al. (2024). o-CLEAN. J Neural Eng 21:056023.
- Bailey NW et al. (2023). RELAX, Part 1. Clin Neurophysiol 149:178-201.
- Rotaru I et al. (2024). What are we really decoding? J Neural Eng 21:016017.
- Brodbeck C et al. (2020). (Follow-up study basis; full citation from Joshua.)
