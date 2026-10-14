# Artifact cleaning for continuous speech-listening EEG

Supervised, non-neural detection and reconstruction of **ocular** (blinks,
saccades) and **muscle (EMG)** artifacts in the Binaural Cocktail EEG,
validated on semi-simulated and real data. Separate from the cortical TRF
analysis in `analysis/`: nothing here changes or imports that pipeline
(it only reuses the bad channels `analysis/experiment.py` already marked).

## Plan and status

The work is done in batches. Each batch is complete, tested and reviewed
before the next starts.

| Batch | Plan section | Contents | Status |
|---|---|---|---|
| 1 | 1-2: scope, data, labels | Memory-safe preprocessing, segment/condition table, ocular auto-labels, EMG weak labels, hand spot-check tooling | **done** |
| 2 | 3: features | Sliding windows; variance, peak-to-peak, kurtosis, line length, Hjorth, Welch band powers, frontal-posterior gradient, frontal correlation | next |
| 3 | 3: models | LightGBM/XGBoost, RBF-SVM, LDA, random forest, stacking; leave-one-subject-out CV; sample- and event-level metrics | |
| 4 | 4: reconstruction | Multi-channel Wiener filter, local EOG ridge regression, learned spatial interpolation; cap on the proportion modified | |
| 5 | 5: semi-simulation | Das et al. (2020) Dataset III-style data with injected real artifacts, SNR sweep, relative TRF MSE, RRMSE, correlation, SNR gain | |
| 6 | 5: real data, public data | TRF accuracy before/after, condition effects, EOG leakage, raw and ICA baselines; TUAR and Das et al. (2020) Dataset I | |

## Layout

```
artifacts/
  01_labels.ipynb        batch 1: preprocess, auto-label, review, label accuracy
  README.md
  config.py              paths, channel groups, preprocessing defaults
  paths.py               BIDS-derivatives file names
  preprocessing/
    loading.py           subjects, lazy BIDS raw, eelbrain's bad channels
    segments.py          the 12 speech segments and their conditions
    preprocess.py        per-segment downsampling, filtering, VEOG/HEOG, re-reference
  labels/
    ocular.py            blink and saccade auto-labels from VEOG/HEOG
    emg.py               weak EMG labels from temporal high-frequency power
    review.py            review sheets, event plots, precision / miss rate
    store.py             saving/loading labels, per-condition summaries
    pipeline.py          the steps 01_labels.ipynb runs, over a list of subjects
  tests/                 synthetic-data tests with known injected artifacts
```

**Notebooks vs. modules.** Each pipeline step whose results you look at
(tables, plots, example events) is a numbered notebook directly in
`artifacts/` - never inside a subfolder. The notebooks only choose
settings, run steps and show results; all the actual work lives in the
modules under the subfolders, where it is tested. Later batches add
`02_...ipynb`, `03_...ipynb` and so on, in pipeline order.

Everything is written to `<dataset>/cocktail/bids/derivatives/artifacts/`,
BIDS-derivatives style (`sub-XX/eeg/sub-XX_task-cocktail_desc-<what>_<suffix>.<ext>`,
group-level tables under `group/`, review sheets under `review/`).

## Setup

Uses the repo's existing `cocktail-cortical` environment. Batch 1 needs
nothing beyond what eelbrain already pulls in, plus mne-bids and pandas
(already used by `data/bids_extraction.py`). From a terminal:

```
conda activate cocktail-cortical
conda install -c conda-forge mne-bids pandas pytest pybv
```

(`pybv` is only needed by the end-to-end test, which writes a small
synthetic BIDS dataset; that test skips itself without it.)

Later batches add `scikit-learn lightgbm xgboost` (batch 3); the batch
that needs them will say so.

The dataset location defaults to the sibling `dataset/cocktail/bids`
folder the rest of the repo uses; set the `COCKTAIL_BIDS_ROOT`
environment variable to point elsewhere without editing any code.

## Batch 1: preprocessing and labels

Open `01_labels.ipynb` with the `cocktail-cortical` environment active
(select it as the kernel in VS Code, or from a terminal):

```
conda activate cocktail-cortical
jupyter lab artifacts/01_labels.ipynb
```

and run its cells in order. Settings (which subjects, how many events to
review, whether to redo preprocessing) are in the cell under **Settings**.

### Step 1: preprocess and auto-label

Per subject this writes the preprocessed recording (`desc-preproc_eeg.fif`,
~150-250 MB each at float32), its segment table and a JSON of every
preprocessing choice; ocular and EMG label tables; and
`group/label_summary.tsv`. The notebook shows a table of the
preprocessing choices per subject, the label summary, bar plots of event
rates and % of time labelled per condition, and one example event of each
type. Preprocessing is cached: re-running only re-labels unless
`REBUILD = True`.

### Step 2: spot-check a subset by hand

For each review subject and label kind, this draws `N_PER_TYPE` random
events per type plus `N_UNLABELLED` random **unlabelled** 1 s windows, and
writes one PNG each plus a `review.csv`. Open the PNGs and fill in
`verdict`:

- labelled events: `ok`, `wrong` or `unsure` (optionally `corrected_onset`
  / `corrected_duration`, in seconds)
- unlabelled windows (`trial_type` = `none`): `clean`, or `missed:<type>`
  (e.g. `missed:blink`)

Sheets that already contain verdicts are never overwritten (unless
`overwrite=True`).

### Step 3: label accuracy

After filling in the sheets, the last cells report **precision** per type
(from the labelled sample) and a **miss rate** (from the unlabelled
sample), as tables and a bar plot - the two numbers to report for the
auto-labeller. Reviewed sheets are applied automatically whenever labels
are loaded for training (`labels/store.py: load_labels`): events marked
`wrong` are dropped and corrections applied.

EMG labels are weak by design and **must** be reviewed before batch 3
trains on them; ocular labels should be reviewed on at least a few
subjects to report their accuracy.

## Tests

From the repo root (the folder containing `artifacts/`):

```
conda activate cocktail-cortical
python -m pytest artifacts
```

No real data is needed: the tests inject blinks, saccades, EMG bursts,
mains noise and a CMS-like common-mode signal at known times into
synthetic recordings and check they are found / removed. One test writes
a small synthetic BIDS dataset laid out like the real one (channel names,
trigger codes, eelbrain's bad-channel file) and runs the whole
`01_labels.ipynb` pipeline on it. Real-data behaviour still has to be
checked with the review sheets.

## Batch 1 design decisions

**Memory-safe loading.** The recordings are ~1 h at ~4096 Hz, and loading
one whole already crashed a kernel (see `analysis/experiment.py`). Each
speech segment is read separately, a few channels at a time, and
downsampled to 256 Hz with a polyphase anti-aliasing filter
(`scipy.signal.resample_poly`) before anything else.

**Speech segments only.** The 12 story segments (with their condition,
mirroring `label_events()` in `analysis/experiment.py`) are kept; breaks
between stories are dropped, since the target is continuous speech
listening. Each segment is filtered with 10 s padding that is trimmed
afterwards, so filter edge transients never reach the kept data. The
segments are concatenated into one file; everything downstream works per
segment and never across a boundary (see the segment table's
`start`/`stop` samples). `tests/test_segments.py` fails if the copied
condition tables drift from `analysis/experiment.py`.

**256 Hz, 0.5 Hz high-pass, no low-pass.** 256 Hz keeps muscle power up
to ~100 Hz for EMG; 0.5 Hz removes drift without distorting blinks.
Mains noise and its harmonics are notched: the frequency is
`config.LINE_FREQ` if set, else whichever of 50/60 Hz shows a clear
spectral peak (3x its neighbours), else the BIDS `PowerLineFrequency`,
else nothing is notched. Which one applied is recorded per subject
(`line_freq_source` in the preprocessing JSON) - worth checking once,
since `data/bids_extraction.py` doesn't set `PowerLineFrequency`.

**Re-referencing.** BioSemi records against CMS/DRL, not a fixed
reference, so scalp channels are re-referenced (to the mastoids by default,
matching `analysis/experiment.py`; `config.REFERENCE = 'average'` is the
alternative) before any feature is computed. If one mastoid is bad the
other is used; if both are, average reference.

**Bipolar EOG.** VEOG = above minus below the left eye (EOG-LS - EOG-LI),
HEOG = left minus right outer canthus (EOG-L - EOG-R). Bipolar derivations
cancel the reference, so they need no re-referencing.

**Bad channels.** Reused from `analysis/experiment.py`'s
`mark_bad_channels()` (its BIDS `channels.tsv`), then interpolated with
spherical splines so every subject has the same 64 channels for the
detection features. The interpolated channels are recorded in each
subject's preprocessing JSON. A bad EOG electrode can't be interpolated;
it is flagged instead, since ocular labels from it are unreliable.

**Ocular labels** (`labels/ocular.py`). Blinks: VEOG band-passed 1-10 Hz,
sign set from the distribution's skew so blinks point up, peaks above
5 x a robust (MAD) noise estimate and at least 40 uV, half-height width
40-800 ms; the window covers the deflection down to 10% of the peak, plus
50 ms. Saccades: EOG low-passed at 30 Hz and differentiated; runs above
6 x the median-based velocity SD (Engbert & Kliegl, 2003), kept only if the
EOG level shifts by at least 20 uV across them; HEOG gives horizontal
saccades, VEOG gives vertical ones outside blink windows. Windows extend
50 ms before and 150 ms after (the spike potential and the step settling).
Thresholds are data-driven per segment except the 20 uV step and 40 uV
blink floors - check both against the review sheets.

**EMG weak labels** (`labels/emg.py`). Log Welch power in 20-100 Hz (mains
harmonics excluded) on FT7/T7/TP7/FT8/T8/TP8, in 0.5 s windows every
0.25 s, z-scored against the subject's own median/MAD; windows whose
highest channel z exceeds 4 are flagged. Known blind spot: tension that
is constant throughout a recording raises the subject's baseline and isn't
flagged. Known false positives: loose electrodes and saccadic spike
potentials on frontotemporal sites. Hence weak, and hence the review step.

## Plan pitfalls, and where each is handled

- **Data leakage** - all splits by subject (batch 3); labels and features
  never pool windows across subjects.
- **Stimulus-locked eye movements** - checked in batch 6 by comparing TRFs
  before/after cleaning per condition, especially dichotic and binaural.
- **Over-cleaning** - reconstruction is capped and the modified proportion
  reported (batch 4); batch 1's `label_summary.tsv` already gives the
  labelled proportion per condition.
- **Contaminated simulation noise** - batch 5 draws background EEG only from
  windows with no ocular or (reviewed) EMG label.

## References

- Das, N., Vanthornhout, J., Francart, T., & Bertrand, A. (2020).
  Stimulus-aware spatial filtering for single-trial neural response and
  temporal response function estimation in high-density EEG with
  applications in auditory research. *NeuroImage*, 204, 116211.
- Engbert, R., & Kliegl, R. (2003). Microsaccades uncover the orientation
  of covert attention. *Vision Research*, 43(9), 1035-1045.
- Ingolfsson et al. (2024), *Scientific Reports* - gradient-boosted trees for
  EEG artifact detection on TUAR (full citation to be added with batch 3,
  where it's used).
- Lawhern, V., Hairston, W. D., McDowell, K., Westerfield, M., & Robbins, K.
  (2012). Detection and classification of subject-generated artifacts in EEG
  signals using autoregressive models. *Journal of Neuroscience Methods*,
  208(2), 181-189.
- Somers, B., Francart, T., & Bertrand, A. (2018). A generic EEG artifact
  removal algorithm based on the multi-channel Wiener filter. *Journal of
  Neural Engineering*, 15(3), 036007.
