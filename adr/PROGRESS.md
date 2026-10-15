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

## Next

- **Milestone 1b - TUAR v3.0.1:** needs Joshua's registration + data-use
  agreement; then verify the §3.2 items.

## Open questions for Joshua

- Section A forward prediction (§9.4, leave-one-trial-out): should trials
  that share a stimulus (an exp-1 part and its repetitions) be held out
  together, so the test envelope never appears in training?
- Own dataset: EOG channel names, trigger codes, segment-to-condition
  mapping, which subjects are included.
- Brodbeck et al. 2020 full citation.
- Detector sampling rate: one model at both 128 and 250 Hz, or two models?
