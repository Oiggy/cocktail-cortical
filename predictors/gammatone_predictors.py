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

import numpy
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

# The audio stimuli used in the experiment:
#   - 24 single-speaker recordings: 12 from a male speaker, 12 from a
#     female speaker (fg/bg terms in experiment.py's label_events()
#     use these).
#   - 24 pre-mixed recordings, "List_<list_id>_stim_<i>" for list_id in
#     1-2 and i in 1-12 - the combined foreground+background audio for
#     each segment (the mix term label_events() defines; the two
#     "clean", no-background segments per list still get a predictor
#     built here even though label_events() never references them as
#     'mix', for consistency - it's just unused, not wrong).
STIMULI = (
    [f"{speaker}_{i}" for speaker in ["male", "female"] for i in range(1, 13)]
    + [f"List_{list_id}_stim_{i}" for list_id in (1, 2) for i in range(1, 13)]
)

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

# --- Block 1: overt and masked onsets (attended vs. ignored speaker, 8-band) ---
#
# Why: not every onset in the ignored speaker actually reaches the ear the
# same way. Some are loud enough to poke through the attended speaker's
# voice ("overt" - still present in the mixture the ear receives), while
# others get buried under a louder attended-speaker onset at that exact
# instant ("masked" - present in the original recording, but effectively
# gone from the mixture). Brodbeck et al. 2020 found the brain treats
# these two kinds of onsets differently, so this splits the plain onset
# predictor into an overt half and a masked half - once using the
# attended speaker as the "source", once using the ignored speaker.
#
#   masked onset = max(source onset - mixture onset, 0)   - how much
#     stronger the source's onset is than what the mixture shows; 0 if
#     the mixture's onset is already as big (nothing got masked)
#   overt onset  = min(mixture onset, source onset)       - the smaller
#     of the two; this is what's left of the source's onset once the
#     mixture is allowed to cap it
#
# Only two-talker segments have a competing speaker and a mixture to
# compare against - the "clean" single-speaker segments (i == 1 or 7,
# see label_events() in experiment.py) are skipped.
#
# Which single-speaker recording is attended (fg) and which is ignored
# (bg) for a given segment depends on the stimulus list - this mirrors
# label_events()'s own SPEAKER lookup in experiment.py exactly, but is
# duplicated here rather than imported, since this script is meant to
# run before experiment.py exists to import (no BIDS dataset needed yet).
SPEAKER = [
    ['male'] * 6 + ['female'] * 6,
    ['female'] * 6 + ['male'] * 6,
]

# eelbrain only recognizes "gammatone" as a predictor key (the one key
# registered in experiment.py's `predictors` dict) - it reads everything
# after that first "-"-separated part just to tell versions of the same
# predictor apart. So the new predictor codes below have to keep starting
# with "gammatone-", exactly like the existing "-on-8" does, for
# `e.load_trfs()`/`e.load_model_test()` model strings to resolve them.


def _match_duration(a, b):
    "Trim two onset NDVars (same tmin/tstep) to their shorter shared duration"
    tstop = min(a.time.tstop, b.time.tstop)
    if a.time.tstop != b.time.tstop:
        print(f"[trimming {a.time.tstop:.3f}s/{b.time.tstop:.3f}s to {tstop:.3f}s]", end=" ")
    window = (a.time.tmin, tstop)
    return a.sub(time=window), b.sub(time=window)


for list_id in (1, 2):
    fg_speakers = SPEAKER[list_id - 1]
    bg_speakers = SPEAKER[-list_id + 2]
    for i in range(1, 13):
        if i in (1, 7):
            continue  # clean segment: no competing speaker or mixture

        fg_stim = f"{fg_speakers[i - 1]}_{i}"
        bg_stim = f"{bg_speakers[i - 1]}_{i}"
        mix_stim = f"List_{list_id}_stim_{i}"

        dst_attend_masked = PREDICTOR_DIR / f"{fg_stim}~gammatone-on-masked-8.pickle"
        dst_attend_overt = PREDICTOR_DIR / f"{fg_stim}~gammatone-on-overt-8.pickle"
        dst_ignore_masked = PREDICTOR_DIR / f"{bg_stim}~gammatone-on-masked-8.pickle"
        dst_ignore_overt = PREDICTOR_DIR / f"{bg_stim}~gammatone-on-overt-8.pickle"
        if all(p.exists() for p in (dst_attend_masked, dst_attend_overt, dst_ignore_masked, dst_ignore_overt)):
            # Already built. Every speaker file gets visited twice across
            # the two lists (once attended, once ignored), but list 1 and
            # list 2's mixtures for the same segment are the same acoustic
            # content - only the attention instruction differs - so the
            # second visit always recomputes the same thing; skipping it
            # is safe, not just faster.
            continue
        print(f"list {list_id} seg {i} ({fg_stim} vs {bg_stim})", end=", ")

        fg_onset = load.unpickle(PREDICTOR_DIR / f"{fg_stim}~gammatone-on-8.pickle")
        bg_onset = load.unpickle(PREDICTOR_DIR / f"{bg_stim}~gammatone-on-8.pickle")
        mix_onset = load.unpickle(PREDICTOR_DIR / f"{mix_stim}~gammatone-on-8.pickle")

        if not (dst_attend_masked.exists() and dst_attend_overt.exists()):
            fg_aligned, mix_for_fg = _match_duration(fg_onset, mix_onset)
            if not dst_attend_masked.exists():
                save.pickle((fg_aligned - mix_for_fg).clip(0, None), dst_attend_masked)
            if not dst_attend_overt.exists():
                # min(mixture, source) == mixture - max(mixture - source, 0)
                save.pickle(mix_for_fg - (mix_for_fg - fg_aligned).clip(0, None), dst_attend_overt)

        if not (dst_ignore_masked.exists() and dst_ignore_overt.exists()):
            bg_aligned, mix_for_bg = _match_duration(bg_onset, mix_onset)
            if not dst_ignore_masked.exists():
                save.pickle((bg_aligned - mix_for_bg).clip(0, None), dst_ignore_masked)
            if not dst_ignore_overt.exists():
                save.pickle(mix_for_bg - (mix_for_bg - bg_aligned).clip(0, None), dst_ignore_overt)

# --- Block 2: three intensity-level mixture onsets (confound control) ---
#
# Why: the overt/masked split above could, in principle, just be a
# roundabout way of measuring loudness - maybe the brain simply responds
# more to bigger onsets, and "masked" onsets just happen to be smaller
# ones. This predictor lets that possibility be tested directly: it
# splits the mixture's own onsets into three loudness tiers (low/mid/high)
# using *only* the mixture - it has no idea which speaker is attended or
# ignored, or even that there are two speakers at all. A model built from
# this can be compared against the overt/masked model (analysis 6) to
# check whether loudness alone explains that earlier result.
#
# Steps, worked out separately for each of the 8 frequency bands, using
# only the mixture's own onset predictor (gammatone-on-8):
#   1. A "run" is one onset event: a stretch of consecutive nonzero
#      timepoints.
#   2. A run's intensity is the sum of its values.
#   3. Tertile cutoffs (33rd/66th percentile of intensity) are computed
#      per band, pooling every run from every mixture stimulus together -
#      so "low"/"mid"/"high" mean the same thing everywhere in the
#      dataset, not just within one recording.
#   4/5. Each stimulus's onset predictor is then split into three new
#      versions per band: everything zeroed out except the runs that fall
#      in that band's low/mid/high tier.
#
# Naming: for the same reason explained in block 1 above (eelbrain only
# recognizes "gammatone" as a registered predictor key), these are named
# "gammatone-on-low-8" / "-mid-8" / "-high-8", not "mixture-onset-...":
# the leading "gammatone-" is what lets e.load_trfs()/e.load_model_test()
# resolve them at all. Used with the "mix~" prefix (e.g.
# "mix~gammatone-on-low-8"), same as the existing mixture-envelope
# predictor.
MIX_STIMULI = [f"List_{list_id}_stim_{i}" for list_id in (1, 2) for i in range(1, 13)]
N_BANDS = 8
TIERS = ('low', 'mid', 'high')


def _onset_runs(band_x):
    "(start, stop) index pairs for each contiguous nonzero stretch in a 1D array"
    is_on = (band_x != 0).astype(int)
    padded = numpy.concatenate(([0], is_on, [0]))
    diffs = numpy.diff(padded)
    starts = numpy.flatnonzero(diffs == 1)
    stops = numpy.flatnonzero(diffs == -1)
    return list(zip(starts, stops))


all_tier_paths = [PREDICTOR_DIR / f"{stim}~gammatone-on-{tier}-8.pickle" for stim in MIX_STIMULI for tier in TIERS]
if not all(p.exists() for p in all_tier_paths):
    # Pass 1: collect every run's intensity per band, across every
    # mixture stimulus, so the tertile cutoffs reflect the whole dataset.
    onset_arrays = {}  # stimulus -> (frequency, time) array
    intensities_by_band = [[] for _ in range(N_BANDS)]
    for stim in MIX_STIMULI:
        onset = load.unpickle(PREDICTOR_DIR / f"{stim}~gammatone-on-8.pickle")
        x = onset.get_data(('frequency', 'time'))
        onset_arrays[stim] = (onset, x)
        for band in range(N_BANDS):
            for start, stop in _onset_runs(x[band]):
                intensities_by_band[band].append(x[band, start:stop].sum())

    band_cutoffs = [numpy.percentile(intensities_by_band[band], [33, 66]) for band in range(N_BANDS)]

    # Pass 2: use those cutoffs to build the three tier-only versions of
    # each stimulus's onsets, one band at a time.
    for stim in MIX_STIMULI:
        print(stim, end=", ")
        onset, x = onset_arrays[stim]
        tier_x = {tier: numpy.zeros_like(x) for tier in TIERS}
        for band in range(N_BANDS):
            low_cut, high_cut = band_cutoffs[band]
            for start, stop in _onset_runs(x[band]):
                intensity = x[band, start:stop].sum()
                tier = 'low' if intensity <= low_cut else 'mid' if intensity <= high_cut else 'high'
                tier_x[tier][band, start:stop] = x[band, start:stop]

        dims = onset.get_dims(('frequency', 'time'))
        for tier in TIERS:
            tier_ndvar = NDVar(tier_x[tier], dims, onset.name, onset.info)
            save.pickle(tier_ndvar, PREDICTOR_DIR / f"{stim}~gammatone-on-{tier}-8.pickle")

# --- Step 3: sanity check - make sure nothing produced missing (NaN) values ---
for stimulus in STIMULI:
    gt = load.unpickle(PREDICTOR_DIR / f"{stimulus}~gammatone-1.pickle")
    print(stimulus, "contains NaN:", numpy.isnan(gt.x).any())

# Same check for the two new predictor blocks above.
for list_id in (1, 2):
    fg_speakers = SPEAKER[list_id - 1]
    bg_speakers = SPEAKER[-list_id + 2]
    for i in range(1, 13):
        if i in (1, 7):
            continue
        for stim in (f"{fg_speakers[i - 1]}_{i}", f"{bg_speakers[i - 1]}_{i}"):
            for suffix in ('masked-8', 'overt-8'):
                p = PREDICTOR_DIR / f"{stim}~gammatone-on-{suffix}.pickle"
                x = load.unpickle(p)
                print(p.name, "contains NaN:", numpy.isnan(x.x).any())

for stim in MIX_STIMULI:
    for tier in TIERS:
        p = PREDICTOR_DIR / f"{stim}~gammatone-on-{tier}-8.pickle"
        x = load.unpickle(p)
        print(p.name, "contains NaN:", numpy.isnan(x.x).any())
