"""
Predictors for the overt/masked notebooks, made as in Brodbeck et al. 2020
(PLOS Biology 18(10): e3000883, "Predictor variables"):

- gammatone spectrogram, 256 bands from 20 to 5,000 Hz (ERB spaced), 1 ms
- scaled with exponent 0.6 (not log(x + 1), which gammatone_predictors.py uses)
- onsets: auditory edge detector (saturation c = 30), negative values set to 0
- overt/masked split element-wise on the full 256-band onset spectrograms:
      masked = max(source onset - mixture onset, 0)
      overt  = min(source onset, mixture onset)       (overt + masked == source onset)
  and only then binned into 8 bands, equally spaced in ERB, leaving out
  frequencies below 100 Hz. (gammatone_predictors.py binned first and split
  after, which cancels masking in one narrow band against extra mixture energy
  in a neighbouring one.)
- intensity tiers (the level-aware control): the mixture's 8-band onsets split
  into low/mid/high by the tertiles of each onset's summed intensity, per band.

Two things this dataset needs that the paper's did not:
- Each speaker is compared at the level it has in the mixture. Brodbeck's
  mixtures were the two speakers added together; here the speaker files are
  4x louder than inside the played mixture (check_alignment.py), and the
  saturating edge detector gives a louder copy bigger onsets. The level of
  each speaker in each ear: least-squares gains of the mixture waveform
  (diotic, dichotic: R^2 = 1) or, where head filtering makes that inexact
  (binaural), a gain per frequency band fitted to the mixture's band power.
  The gammatone spectrogram is linear in amplitude, so scaling it is exact.
- Stereo mixtures are handled ear by ear and averaged over the two ears.

Saves (8 bands, 1 ms), for the two-talker segments only:
    <speaker>~gammatone-b20-8.pickle            envelope
    <speaker>~gammatone-b20-on-8.pickle         onsets at the speaker's level in the mixture
    <speaker>~gammatone-b20-on-overt-8.pickle   } overt + masked == onsets
    <speaker>~gammatone-b20-on-masked-8.pickle  }
    List_<l>_stim_<i>~gammatone-b20-8.pickle    mixture envelope
    List_<l>_stim_<i>~gammatone-b20-on-8.pickle mixture onsets
    List_<l>_stim_<i>~gammatone-b20-on-{low,mid,high}-8.pickle

Each speaker plays in one segment, once per list; the two lists' mixtures of a
segment are the same sound within 0.7%, so a speaker's overt/masked split uses
the list 1 mixture. The 256-band spectrograms are only kept while this runs
(node-local scratch in a cluster job). Skips whatever is already saved; safe to
re-run. cluster/run_analysis.sbatch runs it before any overt/masked notebook.
"""
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy
from scipy.io import wavfile
from scipy.optimize import nnls

from eelbrain import NDVar, Scalar, UTS, edge_detector, gammatone_bank, load, save

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT.parent / "dataset" / "cocktail"
STIMULUS_DIR = DATA_ROOT / "stimuli"
PREDICTOR_DIR = DATA_ROOT / "bids" / "derivatives" / "predictors"

F_MIN, F_MAX, N_BANDS = 20, 5000, 256
EXPONENT = 0.6
EDGE_C = 30
BIN_F_MIN, N_BINS = 100, 8
CODE = 'gammatone-b20'
TIERS = ('low', 'mid', 'high')
EXACT_R2 = 0.99            # speakers explain the mixture waveform exactly: plain gains
FIT_WINDOW = (30.0, 90.0)  # s of audio for the waveform fit

# The design, as in analysis/experiment.py
SPEAKER = [
    ['male'] * 6 + ['female'] * 6,
    ['female'] * 6 + ['male'] * 6,
]
SPATIAL = ['clean', 'binaural', 'dichotic', 'diotic', 'binaural', 'dichotic'] * 2
SEGMENTS = [i for i in range(1, 13) if i not in (1, 7)]


def speakers(list_id, i):
    return f"{SPEAKER[list_id - 1][i - 1]}_{i}", f"{SPEAKER[-list_id + 2][i - 1]}_{i}"


def out(name, kind=''):
    return PREDICTOR_DIR / f"{name}~{CODE}{'-' + kind if kind else ''}-8.pickle"


def save_atomic(ndvar, path):
    "Write under a temporary name first, so a crash or a parallel job never leaves half a file"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    save.pickle(ndvar, tmp)
    os.replace(tmp, path)


def n_channels(name):
    _, x = wavfile.read(STIMULUS_DIR / f"{name}.wav", mmap=True)
    return 1 if x.ndim == 1 else x.shape[1]


def read_wav(name):
    "(rate, samples x channels as float)"
    rate, x = wavfile.read(STIMULUS_DIR / f"{name}.wav")
    x = x.astype(float) / numpy.iinfo(x.dtype).max if x.dtype.kind in 'iu' else x.astype(float)
    return rate, x[:, None] if x.ndim == 1 else x


# 256-band spectrograms, computed in parallel and kept only while this runs

def _gammatone_task(args):
    name, channel, path = args
    if path.exists():
        return path
    wav = load.wav(STIMULUS_DIR / f"{name}.wav")
    if channel is not None:
        wav = wav.sub(channel=channel)
    gt = gammatone_bank(wav, F_MIN, F_MAX, N_BANDS, location='left', tstep=0.001)
    gt = NDVar(gt.x.astype(numpy.float32), gt.dims, gt.name)
    save_atomic(gt, path)
    return path


def spectrogram_path(scratch, name, channel):
    return scratch / f"{name}-gt{N_BANDS}{'' if channel is None else f'-ear{channel}'}.pickle"


def make_spectrograms(scratch, names, workers):
    tasks = []
    for name in names:
        n = n_channels(name)
        for channel in ([None] if n == 1 else range(n)):
            tasks.append((name, channel, spectrogram_path(scratch, name, channel)))
    todo = [t for t in tasks if not t[2].exists()]
    print(f"gammatone spectrograms: {len(todo)} to compute ({workers} at a time)", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for path in pool.map(_gammatone_task, todo):
            print(f"  {path.name}", flush=True)


def load_ears(scratch, name):
    n = n_channels(name)
    return [load.unpickle(spectrogram_path(scratch, name, c)) for c in ([None] if n == 1 else range(n))]


# Paper steps

def onsets(x, dims):
    "Edge-detector onsets (negatives set to 0) of an exponent-scaled (bands x time) spectrogram"
    on = edge_detector(NDVar(x, dims, 'gammatone'), c=EDGE_C)
    return numpy.clip(on.get_data(('frequency', 'time')), 0, None)


def binner(freq):
    "Function: (256 bands x time) array -> (8 bands x time) NDVar data, summed over ERB-equal bins from 100 Hz"
    keep = numpy.flatnonzero(freq.values >= BIN_F_MIN)
    edges = numpy.linspace(0, len(keep), N_BINS + 1).round().astype(int)
    groups = [keep[a:b] for a, b in zip(edges[:-1], edges[1:])]
    centers = [float(numpy.exp(numpy.log(freq.values[g]).mean())) for g in groups]
    dim = Scalar('frequency', centers, 'Hz')

    def apply(x):
        return numpy.stack([x[g].sum(0) for g in groups])
    return apply, dim


def save_8(x8, freq8, n, path):
    save_atomic(NDVar(x8, (freq8, UTS(0, 0.001, n)), 'gammatone'), path)


def waveform_gains(mix_c, fg, bg, rate):
    "Gains (clipped at 0) and R^2 of mixture channel ~ a * attend + b * ignore"
    end = min(len(mix_c), len(fg), len(bg))
    start, stop = (min(int(t * rate), end) for t in FIT_WINDOW)
    if stop - start < 10 * rate:
        start, stop = 0, end
    design = numpy.stack([fg[start:stop], bg[start:stop]], axis=1)
    target = mix_c[start:stop]
    coef, *_ = numpy.linalg.lstsq(design, target, rcond=None)
    r2 = 1 - (target - design @ coef).var() / (target.var() or 1)
    return numpy.clip(coef, 0, None), float(r2)


def band_gains(mix_x, fg_x, bg_x):
    "Per-band amplitude gains (2 x bands): mixture power ~ a^2 attend power + b^2 ignore power"
    gains = numpy.zeros((2, len(mix_x)))
    for k in range(len(mix_x)):
        coef, _ = nnls(numpy.stack([fg_x[k] ** 2, bg_x[k] ** 2], axis=1), mix_x[k] ** 2)
        gains[:, k] = numpy.sqrt(coef)
    return gains


def main():
    mixtures = [(list_id, i, f"List_{list_id}_stim_{i}") for list_id in (1, 2) for i in SEGMENTS]
    speaker_names = sorted({s for i in SEGMENTS for s in speakers(1, i)})
    mixture_kinds = ['', 'on', *(f'on-{t}' for t in TIERS)]
    speaker_kinds = ['', 'on', 'on-overt', 'on-masked']
    if all(out(m, k).exists() for _, _, m in mixtures for k in mixture_kinds) and \
            all(out(s, k).exists() for s in speaker_names for k in speaker_kinds):
        print("Brodbeck et al. 2020 predictors already made.")
        return

    scratch_root = os.environ.get('SLURM_TMPDIR') or os.environ.get('SCRATCH') or tempfile.gettempdir()
    scratch = Path(tempfile.mkdtemp(prefix='b20-', dir=scratch_root))
    workers = max(1, min(16, int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1)) // 2))
    try:
        make_spectrograms(scratch, [m for _, _, m in mixtures] + speaker_names, workers)
        freq = load_ears(scratch, speaker_names[0])[0].get_dim('frequency')
        to8, freq8 = binner(freq)
        mixture_onsets8 = {}  # for the tiers, which pool every mixture

        for list_id, i, mix in mixtures:
            ears = load_ears(scratch, mix)
            n = min(e.time.nsamples for e in ears)
            dims = (freq, UTS(0, 0.001, n))
            ear_x = [e.get_data(('frequency', 'time'))[:, :n].astype(float) for e in ears]
            ear_on = [onsets(x ** EXPONENT, dims) for x in ear_x]
            env8 = numpy.mean([to8(x ** EXPONENT) for x in ear_x], 0)
            on8 = numpy.mean([to8(o) for o in ear_on], 0)
            mixture_onsets8[mix] = on8
            if not out(mix).exists():
                save_8(env8, freq8, n, out(mix))
            if not out(mix, 'on').exists():
                save_8(on8, freq8, n, out(mix, 'on'))
            print(f"list {list_id} segment {i} ({SPATIAL[i - 1]}): mixture {mix} done", flush=True)

            if list_id != 1:
                continue
            fg, bg = speakers(1, i)
            if all(out(s, k).exists() for s in (fg, bg) for k in speaker_kinds):
                continue
            rate, mix_wav = read_wav(mix)
            fg_mono, bg_mono = read_wav(fg)[1].mean(1), read_wav(bg)[1].mean(1)
            src = {s: load_ears(scratch, s)[0] for s in (fg, bg)}
            m = min(n, *(g.time.nsamples for g in src.values()))
            src_x = {s: g.get_data(('frequency', 'time'))[:, :m].astype(float) for s, g in src.items()}
            dims_m = (freq, UTS(0, 0.001, m))
            parts = {s: {k: [] for k in ('on', 'on-overt', 'on-masked')} for s in (fg, bg)}
            for c, (x_mix, o_mix) in enumerate(zip(ear_x, ear_on)):
                x_mix, o_mix = x_mix[:, :m], o_mix[:, :m]
                (a, b), r2 = waveform_gains(mix_wav[:, c], fg_mono, bg_mono, rate)
                if r2 >= EXACT_R2:
                    gains = {fg: numpy.full(len(freq), a), bg: numpy.full(len(freq), b)}
                    how = f"waveform gains {a:.3f} / {b:.3f} (R^2 {r2:.3f})"
                else:
                    g = band_gains(x_mix, src_x[fg], src_x[bg])
                    gains = {fg: g[0], bg: g[1]}
                    how = f"waveform R^2 {r2:.3f}: per-band gains, mean {g[0].mean():.3f} / {g[1].mean():.3f}"
                print(f"  ear {c}: {how}", flush=True)
                for s in (fg, bg):
                    o_src = onsets((src_x[s] * gains[s][:, None]) ** EXPONENT, dims_m)
                    parts[s]['on'].append(to8(o_src))
                    parts[s]['on-overt'].append(to8(numpy.minimum(o_src, o_mix)))
                    parts[s]['on-masked'].append(to8(numpy.clip(o_src - o_mix, 0, None)))
            for s, role in ((fg, 'attend'), (bg, 'ignore')):
                if not out(s).exists():
                    save_8(to8(src_x[s] ** EXPONENT), freq8, m, out(s))
                result = {k: numpy.mean(v, 0) for k, v in parts[s].items()}
                for k, x8 in result.items():
                    save_8(x8, freq8, m, out(s, k))
                share = result['on-masked'].sum() / (result['on'].sum() or 1)
                print(f"  {role} {s}: masked share {share:.0%}", flush=True)

        # Intensity tiers of the mixture onsets, cutoffs pooled over every mixture, per band
        if not all(out(m, f'on-{t}').exists() for _, _, m in mixtures for t in TIERS):
            intensities = [[] for _ in range(N_BINS)]
            runs = {}
            for mix, x in mixture_onsets8.items():
                runs[mix] = []
                for band in range(N_BINS):
                    on = numpy.concatenate(([0], (x[band] != 0).astype(int), [0]))
                    starts, stops = numpy.flatnonzero(numpy.diff(on) == 1), numpy.flatnonzero(numpy.diff(on) == -1)
                    for a, b in zip(starts, stops):
                        intensities[band].append(x[band, a:b].sum())
                        runs[mix].append((band, a, b))
            cuts = [numpy.percentile(v, [100 / 3, 200 / 3]) for v in intensities]
            for mix, x in mixture_onsets8.items():
                tier_x = {t: numpy.zeros_like(x) for t in TIERS}
                for band, a, b in runs[mix]:
                    total = x[band, a:b].sum()
                    t = 'low' if total <= cuts[band][0] else 'mid' if total <= cuts[band][1] else 'high'
                    tier_x[t][band, a:b] = x[band, a:b]
                for t in TIERS:
                    save_8(tier_x[t], freq8, x.shape[1], out(mix, f'on-{t}'))
            print("intensity tiers done", flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print("Brodbeck et al. 2020 predictors ready.")


if __name__ == '__main__':
    main()
