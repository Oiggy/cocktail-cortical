"""
Build the cortical speech predictors: gammatone spectrograms.

Why: to test how well the brain's response to sound can be predicted from
the sound itself, eelbrain needs a numeric description of each audio
stimulus that lines up in time with the EEG. A gammatone spectrogram is a
model of how the cochlea (inner ear) breaks sound into frequency bands -
it's a standard stand-in for "what the ear sends to the brain".

From each spectrogram we derive two predictors used in the cortical TRF
analysis (see analysis/experiment.py and analysis/cortical_analysis.py):
  - "gammatone": the sound envelope (how loud each frequency band is
    over time).
  - "gammatone-on": an onset/edge version (highlights sudden increases
    in loudness), because the brain responds strongly to acoustic onsets.

Each is saved in two resolutions: a single broadband channel ("-1") and
an 8-frequency-band version ("-8"), so the analysis scripts can pick
whichever resolution they need.

Run this after data/bids_extraction.py and before analysis/experiment.py.
"""
from pathlib import Path

from eelbrain import *

# The dataset lives outside this repo, in a sibling "dataset/cocktail"
# folder next to it. Working this out from this file's own location
# means the script runs the same way no matter whose computer, or which
# folder, the repo is cloned into.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT.parent / "dataset" / "cocktail"
STIMULUS_DIR = DATA_ROOT / "stimuli"
# UTSPredictor (see analysis/experiment.py) looks for predictors under
# <BIDS root>/derivatives/predictors, not <BIDS root>/predictors - this
# has to match that exactly.
PREDICTOR_DIR = DATA_ROOT / "bids" / "derivatives" / "predictors"

# The 24 audio stimuli used in the experiment: 12 recordings from a male
# speaker, 12 from a female speaker.
STIMULI = [f"{speaker}_{i}" for speaker in ["male", "female"] for i in range(1, 13)]

# --- Step 1: turn each sound file into a gammatone spectrogram ---
for stimulus in STIMULI:
    dst = STIMULUS_DIR / f"{stimulus}-gammatone.pickle"
    if dst.exists():
        # Already built, skip it - this loop is safe to re-run.
        continue

    wav = load.wav(STIMULUS_DIR / f"{stimulus}.wav")

    if wav.ndim == 2:
        # Stereo file: build the spectrogram for each ear separately,
        # then average them into one "what both ears heard" version.
        gt_left = gammatone_bank(wav.sub(channel=0), 80, 8000, 128, location="left", tstep=0.001)
        gt_right = gammatone_bank(wav.sub(channel=1), 80, 8000, 128, location="left", tstep=0.001)
        gt = (gt_left + gt_right) / 2
    else:
        gt = gammatone_bank(wav, 80, 8000, 128, location="left", tstep=0.001)

    save.pickle(gt, dst)

# --- Step 2: derive the envelope and onset predictors from the spectrogram ---
PREDICTOR_DIR.mkdir(exist_ok=True)

for stimulus in STIMULI:
    dst_envelope_1band = PREDICTOR_DIR / f"{stimulus}~gammatone-1.pickle"
    if dst_envelope_1band.exists():
        continue
    print(stimulus, end=", ")

    gt = load.unpickle(STIMULUS_DIR / f"{stimulus}-gammatone.pickle")

    # Log-compress the spectrogram: this mimics how the ear compresses
    # loud sounds, and is a standard step before using a spectrogram as
    # a brain-response predictor.
    gt_log = (gt + 1).log()
    # Detect sudden increases in loudness ("acoustic onsets") - the
    # brain reacts strongly to these, so they make a useful separate
    # predictor.
    gt_onset = edge_detector(gt_log, c=30)

    # Broadband (1-band) versions: sum across all frequency bands to get
    # a single "overall loudness over time" signal.
    save.pickle(gt_log.sum("frequency"), dst_envelope_1band)
    save.pickle(gt_onset.sum("frequency"), PREDICTOR_DIR / f"{stimulus}~gammatone-on-1.pickle")

    # 8-band versions: group the frequency bands into 8 wider bands
    # instead of collapsing them entirely, for analyses that care about
    # which frequencies drove the response.
    gt_log_8band = gt_log.bin(nbins=8, func="sum", dim="frequency")
    save.pickle(gt_log_8band, PREDICTOR_DIR / f"{stimulus}~gammatone-8.pickle")
    gt_onset_8band = gt_onset.bin(nbins=8, func="sum", dim="frequency")
    save.pickle(gt_onset_8band, PREDICTOR_DIR / f"{stimulus}~gammatone-on-8.pickle")

# --- Step 3: sanity check - make sure nothing produced missing (NaN) values ---
import numpy

for stimulus in STIMULI:
    gt = load.unpickle(PREDICTOR_DIR / f"{stimulus}~gammatone-1.pickle")
    print(stimulus, "contains NaN:", numpy.isnan(gt.x).any())
