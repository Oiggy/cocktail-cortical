"""
Overt and masked onsets with each speaker at the level it has in the mixture.

Why: gammatone_predictors.py (block 1) splits each speaker's onsets into
overt and masked by comparing them with the mixture's onsets,

    masked = max(source onset - mixture onset, 0)
    overt  = min(source onset, mixture onset)

but it takes the source onsets from the speaker's own .wav file, which is
4x (12 dB) louder than that speaker inside the mixture that was played
(predictors/check_alignment.py: mixture = 0.25 x attend + 0.25 x ignore in
the diotic and dichotic segments). The spectrograms are log-compressed and
the onset detector saturates, so a louder copy has bigger onsets even where
nothing masks it, and part of every onset was counted as "masked" just
because of that level difference (46% "masked" in dichotic, where the two
speakers are in different ears and cannot mask each other).

Here each speaker is compared at its level in the mixture, ear by ear:

1. The level of each speaker in each ear of the mixture: a least-squares
   fit of the mixture's waveform to the two speakers' waveforms. Where the
   two speakers explain the mixture exactly (R^2 >= 0.99: diotic,
   dichotic) these gains are used as they are. Where they do not (binaural:
   the speakers were filtered through head-related transfer functions,
   which the plain files cannot reproduce), a gain per frequency band
   instead, fitted to the mixture's band power (non-negative least
   squares; the two talkers' powers add up on average).
2. The speaker's spectrogram at that level: the gammatone spectrogram is
   linear in the sound's amplitude, so the stored one is just multiplied
   by the gain (no new filtering).
3. Onsets of that spectrogram and of the same ear of the mixture, exactly
   as in gammatone_predictors.py (log(x + 1), edge detector c=30, 8 bands),
   then overt/masked by the equations above, per ear, and averaged over
   the ears (the same averaging the mixture's own predictor uses).

Saves, per speaker in a two-talker segment:
    <stim>~gammatone-on-inmix-8.pickle         the speaker's onsets at its level in the mixture
    <stim>~gammatone-on-overt-inmix-8.pickle   overt part   } overt + masked == inmix
    <stim>~gammatone-on-masked-inmix-8.pickle  masked part  }
The older gammatone-on-overt-8 / -masked-8 files are left as they are
(cortical_analysis.ipynb uses them).

Each speaker plays in one segment, once per list; the two lists' mixtures
for a segment are the same sound within 0.7% (check_alignment.py), so the
list 1 mixture is used, as gammatone_predictors.py does.

Runs on the cluster (cluster/run_analysis.sbatch runs it before any
overt/masked notebook). Skips speakers already done; safe to re-run.
"""
import os
from pathlib import Path
import sys

import numpy
from scipy.io import wavfile
from scipy.optimize import nnls
from eelbrain import NDVar, UTS, edge_detector, gammatone_bank, load, save

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT.parent / "dataset" / "cocktail"
STIMULUS_DIR = DATA_ROOT / "stimuli"
PREDICTOR_DIR = DATA_ROOT / "bids" / "derivatives" / "predictors"

# The design, as in analysis/experiment.py
SPEAKER = [
    ['male'] * 6 + ['female'] * 6,
    ['female'] * 6 + ['male'] * 6,
]
SPATIAL = ['clean', 'binaural', 'dichotic', 'diotic', 'binaural', 'dichotic'] * 2
LIST_ID = 1

CODES = {'inmix': 'gammatone-on-inmix-8', 'overt': 'gammatone-on-overt-inmix-8', 'masked': 'gammatone-on-masked-inmix-8'}
EXACT_R2 = 0.99          # the speakers explain the mixture's waveform exactly: use plain gains
FIT_WINDOW = (30.0, 90.0)  # s of audio used for the waveform fit


def read_wav(name):
    "(rate, samples x channels as float)"
    rate, x = wavfile.read(STIMULUS_DIR / f"{name}.wav")
    x = x.astype(float) / numpy.iinfo(x.dtype).max if x.dtype.kind in 'iu' else x.astype(float)
    return rate, x[:, None] if x.ndim == 1 else x


def save_atomic(ndvar, path):
    "Write to a temporary name first, so a crash or a parallel job never leaves half a file under the real name"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    save.pickle(ndvar, tmp)
    os.replace(tmp, path)


def source_gt(stim):
    "The speaker's stored gammatone spectrogram (128 bands, 1 ms), as gammatone_predictors.py made it"
    return load.unpickle(STIMULUS_DIR / f"{stim}-gammatone.pickle")


def mixture_ear_gts(mix, n_channels):
    "The mixture's gammatone spectrogram for each ear, computed as gammatone_predictors.py does (cached)"
    if n_channels == 1:
        return [load.unpickle(STIMULUS_DIR / f"{mix}-gammatone.pickle")]
    gts = []
    wav = None
    for c in range(n_channels):
        path = STIMULUS_DIR / f"{mix}-gammatone-ear{c}.pickle"
        if not path.exists():
            if wav is None:
                wav = load.wav(STIMULUS_DIR / f"{mix}.wav")
            print(f"  gammatone of {mix}, ear {c} ...", flush=True)
            save_atomic(gammatone_bank(wav.sub(channel=c), 80, 8000, 128, location="left", tstep=0.001), path)
        gts.append(load.unpickle(path))
    return gts


def onsets8(x, dims):
    "Onsets exactly as gammatone_predictors.py makes them, from a (128 bands x time) spectrogram"
    gt = NDVar(x, dims, 'gammatone')
    return edge_detector((gt + 1).log(), c=30).bin(nbins=8, func="sum", dim="frequency")


def waveform_gains(mix_c, fg, bg, rate):
    "Gains a, b and R^2 of mixture channel ~ a * attend + b * ignore"
    end = min(len(mix_c), len(fg), len(bg))
    start, stop = (min(int(t * rate), end) for t in FIT_WINDOW)
    if stop - start < 10 * rate:  # short audio: use all of it
        start, stop = 0, end
    design = numpy.stack([fg[start:stop], bg[start:stop]], axis=1)
    target = mix_c[start:stop]
    coef, *_ = numpy.linalg.lstsq(design, target, rcond=None)
    r2 = 1 - (target - design @ coef).var() / (target.var() or 1)
    return numpy.clip(coef, 0, None), float(r2)


def band_gains(mix_x, fg_x, bg_x):
    "Per-band amplitude gains (2 x bands) of mixture power ~ a^2 * attend power + b^2 * ignore power"
    gains = numpy.zeros((2, len(mix_x)))
    for k in range(len(mix_x)):
        design = numpy.stack([fg_x[k] ** 2, bg_x[k] ** 2], axis=1)
        coef, _ = nnls(design, mix_x[k] ** 2)
        gains[:, k] = numpy.sqrt(coef)
    return gains


shares = {}
for i in range(1, 13):
    if i in (1, 7):
        continue  # clean segment: nothing to mask
    condition = SPATIAL[i - 1]
    fg = f"{SPEAKER[LIST_ID - 1][i - 1]}_{i}"
    bg = f"{SPEAKER[-LIST_ID + 2][i - 1]}_{i}"
    mix = f"List_{LIST_ID}_stim_{i}"
    paths = {stim: {kind: PREDICTOR_DIR / f"{stim}~{code}.pickle" for kind, code in CODES.items()} for stim in (fg, bg)}
    if all(p.exists() for stim in paths.values() for p in stim.values()):
        continue
    print(f"segment {i} ({condition}): {fg} + {bg} in {mix}", flush=True)

    rate, mix_wav = read_wav(mix)
    _, fg_wav = read_wav(fg)
    _, bg_wav = read_wav(bg)
    fg_mono, bg_mono = fg_wav.mean(1), bg_wav.mean(1)
    gts = {fg: source_gt(fg), bg: source_gt(bg)}
    ears = mixture_ear_gts(mix, mix_wav.shape[1])
    n = min(gt.time.nsamples for gt in [*gts.values(), *ears])
    x = {stim: gt.get_data(('frequency', 'time'))[:, :n] for stim, gt in gts.items()}
    dims = (gts[fg].get_dim('frequency'), UTS(gts[fg].time.tmin, gts[fg].time.tstep, n))

    parts = {stim: {kind: [] for kind in CODES} for stim in (fg, bg)}
    for c, ear in enumerate(ears):
        mix_x = ear.get_data(('frequency', 'time'))[:, :n]
        (a, b), r2 = waveform_gains(mix_wav[:, c], fg_mono, bg_mono, rate)
        if r2 >= EXACT_R2:
            gains = {fg: numpy.full(len(mix_x), a), bg: numpy.full(len(mix_x), b)}
            how = f"waveform gains {a:.3f} / {b:.3f} (R^2 {r2:.3f})"
        else:
            g = band_gains(mix_x, x[fg], x[bg])
            gains = {fg: g[0], bg: g[1]}
            how = (f"waveform R^2 only {r2:.3f}, so per-band gains: attend {g[0].mean():.3f} "
                   f"({g[0].min():.3f}-{g[0].max():.3f}), ignore {g[1].mean():.3f} ({g[1].min():.3f}-{g[1].max():.3f})")
        print(f"  ear {c}: {how}", flush=True)
        mix_on = onsets8(mix_x, dims).get_data(('frequency', 'time'))
        for stim in (fg, bg):
            src_on = onsets8(x[stim] * gains[stim][:, None], dims).get_data(('frequency', 'time'))
            parts[stim]['inmix'].append(src_on)
            parts[stim]['overt'].append(numpy.minimum(src_on, mix_on))
            parts[stim]['masked'].append(numpy.clip(src_on - mix_on, 0, None))

    template = onsets8(x[fg], dims)  # for the 8-band frequency axis
    for stim, role in ((fg, 'attend'), (bg, 'ignore')):
        out = {kind: numpy.mean(arrays, axis=0) for kind, arrays in parts[stim].items()}
        for kind, path in paths[stim].items():
            save_atomic(NDVar(out[kind], template.dims, 'gammatone'), path)
        share = out['masked'].sum() / (out['inmix'].sum() or 1)
        shares.setdefault(condition, []).append(share)
        print(f"  {role} {stim}: masked share {share:.0%}", flush=True)

if shares:
    print("\nMasked share of onset magnitude (Brodbeck et al. 2020, diotic: ~33%): " + ', '.join(
        f"{condition} {numpy.mean(v):.0%}" for condition, v in shares.items()))
print("In-mixture overt/masked onsets ready.")
