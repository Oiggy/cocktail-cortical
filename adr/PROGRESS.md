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
  waveforms included, alignment median R^2 0.68 after clock-skew correction.
  **Decisions (2026-10-15):** phantom pools = three contiguous 100 s blocks
  per condition (`splits/phantom.yaml`); detector features on all 57 shared
  channels (frozen in `configs/project.yaml`), window labels from the
  19-electrode bipolar annotations via OR.

## Next

- **Plan revision 2 (2026-10-15):** TUAR dropped. Detector training now uses
  PhysioMotion (OpenNeuro ds006386); templates and the §9.5 ground-truth
  check use the phantom EEG dataset (ds004784). Both open, no registration.
- Milestone 2: loaders and channel alignment for Dataset I, PhysioMotion and
  the phantom, with tests (after the open questions below are settled).

## Open questions for Joshua

- Section A forward prediction (§9.4, leave-one-trial-out): should trials
  that share a stimulus (an exp-1 part and its repetitions) be held out
  together, so the test envelope never appears in training?
- Own dataset: EOG channel names, trigger codes, segment-to-condition
  mapping, which subjects are included.
- Brodbeck et al. 2020 full citation.
- Detector sampling rate: one model at both 128 and 250 Hz, or two models?
