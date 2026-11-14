"""
Check that the overt/masked onset predictors line up with the sound the
listeners actually heard.

The overt/masked predictors (gammatone_predictors.py, block 1) come from
three separate audio files per two-talker segment: the attended speaker
(male_i or female_i), the ignored speaker, and the pre-mixed
List_<list>_stim_<i> that was played. The equations

    masked = max(source onset - mixture onset, 0)
    overt  = min(source onset, mixture onset)

only mean something if those files start at the same moment and the
mixture really is the two speakers added together. The predictors also
reuse one list's mixture for the other list (each speaker's overt/masked
file is made once), which assumes List_1_stim_i and List_2_stim_i are the
same sound. validate_masked_overt_onsets.py re-checks the equations only;
this checks the assumptions underneath them:

1. Audio: how far each speaker's file is shifted against the mixture
   (envelope cross-correlation at 1 ms, then the waveform at single
   samples), how loud each speaker is inside the mixture, and how much of
   the mixture the two speakers explain (R^2), per ear.
2. List 1 vs list 2 mixture for the same segment: same sound or not.
3. Predictors: the time axes of the source, mixture and overt/masked
   files, the shift between the mixture and source onset predictors, and
   whether each saved overt/masked file matches the equations computed
   against *this list's* mixture.

Runs on the cluster, where the data are: cluster/check_alignment.sbatch,
or check_alignment() from any notebook. Prints a report and saves it as
alignment_check.txt in the results folder. Optional argument: the
dataset folder (default: the sibling dataset/cocktail folder).
"""
from pathlib import Path
import sys
import time

import numpy
from scipy.io import wavfile
from scipy import signal
from eelbrain import load

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT.parent / "dataset" / "cocktail"
STIMULUS_DIR = DATA_ROOT / "stimuli"
PREDICTOR_DIR = DATA_ROOT / "bids" / "derivatives" / "predictors"
REPORT = DATA_ROOT / "bids" / "derivatives" / "results" / "alignment_check.txt"

# The design, as in analysis/experiment.py (copied, like the other predictor
# scripts do, so this runs without building the pipeline).
SPEAKER = [
    ['male'] * 6 + ['female'] * 6,
    ['female'] * 6 + ['male'] * 6,
]
SPATIAL = ['clean', 'binaural', 'dichotic', 'diotic', 'binaural', 'dichotic'] * 2
SIDE = ((
    '', 'right', 'left', '', 'left', 'right',
    '', 'left', 'right', '', 'right', 'left',
), (
    '', 'left', 'right', '', 'right', 'left',
    '', 'right', 'left', '', 'left', 'right',
))
SEGMENTS = [i for i in range(1, 13) if i not in (1, 7)]  # two-talker segments only

ENV_RATE = 1000         # Hz: audio envelopes at 1 ms
MAX_LAG = 3.0           # s: largest shift searched between files
EXCERPT = (30.0, 60.0)  # s: stretch used for the sample-level and loudness fit
LAG_OK = 0.001          # s: shifts up to 1 ms count as aligned
MATCH_OK = 1e-4         # relative difference: saved overt/masked == recomputed

lines = []
problems = []


def say(text=''):
    print(text, flush=True)
    lines.append(text)


def problem(text):
    problems.append(text)


def read_wav(name):
    "(sampling rate, samples x channels as float) of a stimulus .wav"
    rate, x = wavfile.read(STIMULUS_DIR / f"{name}.wav")
    if x.dtype.kind in 'iu':
        x = x.astype(float) / numpy.iinfo(x.dtype).max
    else:
        x = x.astype(float)
    if x.ndim == 1:
        x = x[:, None]
    return rate, x


def envelope(x, rate):
    "RMS envelope of each channel in 1 ms bins (bins x channels)"
    edges = numpy.round(numpy.arange(0, len(x) / rate, 1 / ENV_RATE) * rate).astype(int)
    edges = numpy.append(edges[edges < len(x)], len(x))
    power = numpy.concatenate([numpy.zeros((1, x.shape[1])), numpy.cumsum(x ** 2, axis=0)])
    width = numpy.diff(edges)[:, None]
    return numpy.sqrt((power[edges[1:]] - power[edges[:-1]]) / numpy.maximum(width, 1))


def best_lag(a, b, max_lag):
    """Shift (in samples) at which b best matches a, and the correlation there.
    Positive: b's content comes that much later in a than in b (b starts early)."""
    a = (a - a.mean()) / (a.std() or 1)
    b = (b - b.mean()) / (b.std() or 1)
    corr = signal.correlate(a, b, mode='full', method='fft')
    lags = signal.correlation_lags(len(a), len(b), mode='full')
    keep = numpy.abs(lags) <= max_lag
    corr, lags = corr[keep], lags[keep]
    i = int(numpy.argmax(corr))
    lag = int(lags[i])
    return lag, _overlap_r(a, b, lag)


def _overlap_r(a, b, lag):
    "Pearson r of a[t] and b[t - lag] where both exist"
    if lag >= 0:
        x, y = a[lag:], b[:len(a) - lag]
    else:
        x, y = a[:len(a) + lag], b[-lag:]
    n = min(len(x), len(y))
    if n < 2:
        return float('nan')
    return float(numpy.corrcoef(x[:n], y[:n])[0, 1])


def shifted(x, lag, start, stop):
    "x[t - lag] for t in start..stop (zeros outside x)"
    out = numpy.zeros(stop - start)
    lo, hi = start - lag, stop - lag
    src_lo, src_hi = max(lo, 0), min(hi, len(x))
    if src_hi > src_lo:
        out[src_lo - lo:src_hi - lo] = x[src_lo:src_hi]
    return out


def refine(mix, src, lag, rate, start, stop):
    "Sample-level shift of src within mix, searched +-2 ms around `lag` (samples)"
    window = int(round(0.002 * rate))
    target = mix[start:stop]
    best, best_c = lag, -numpy.inf
    for candidate in range(lag - window, lag + window + 1):
        c = float(numpy.dot(target, shifted(src, candidate, start, stop)))
        if c > best_c:
            best, best_c = candidate, c
    return best


def loudness_fit(mix, fg, bg, lag_fg, lag_bg, start, stop):
    "Gains a, b and R^2 of mix ~ a * fg + b * bg (each at its own shift) over start..stop"
    target = mix[start:stop]
    design = numpy.stack([shifted(fg, lag_fg, start, stop), shifted(bg, lag_bg, start, stop)], axis=1)
    coef, *_ = numpy.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coef
    r2 = 1 - residual.var() / (target.var() or 1)
    return float(coef[0]), float(coef[1]), float(r2)


def predictor(name, code):
    path = PREDICTOR_DIR / f"{name}~{code}.pickle"
    return load.unpickle(path) if path.exists() else None


def axis(x):
    return f"tmin {x.time.tmin:.3f} s, step {x.time.tstep * 1e3:.1f} ms, {x.time.nsamples} samples"


def ms(seconds):
    return f"{seconds * 1e3:+.1f} ms"


say(f"Overt/masked onset alignment check  ({time.strftime('%Y-%m-%d %H:%M')})")
say(f"stimuli:    {STIMULUS_DIR}")
say(f"predictors: {PREDICTOR_DIR}")
say("Shift = how much later the speaker's sound is in the mixture than in its own file (0 = aligned).")

# 1. Audio ------------------------------------------------------------------
say("\n=== 1. Audio: each speaker against the mixture that was played")
audio = {}
envelopes = {}
for list_id in (1, 2):
    for i in SEGMENTS:
        fg = f"{SPEAKER[list_id - 1][i - 1]}_{i}"
        bg = f"{SPEAKER[-list_id + 2][i - 1]}_{i}"
        mix = f"List_{list_id}_stim_{i}"
        side = SIDE[list_id - 1][i - 1]
        condition = SPATIAL[i - 1] + (f" (ignored on the {side})" if side else '')
        say(f"\nlist {list_id} segment {i:>2}  {condition}:  attend {fg}, ignore {bg}, mixture {mix}")
        try:
            rates, x = {}, {}
            for role, name in (('fg', fg), ('bg', bg), ('mix', mix)):
                rates[role], x[role] = read_wav(name)
        except (OSError, ValueError) as error:
            say(f"  could not read the audio: {error}")
            problem(f"list {list_id} seg {i}: audio not readable ({error})")
            continue
        if len(set(rates.values())) > 1:
            say(f"  sampling rates differ: {rates}")
            problem(f"list {list_id} seg {i}: sampling rates differ {rates}")
            continue
        rate = rates['mix']
        durations = {role: len(v) / rate for role, v in x.items()}
        say(f"  {rate} Hz; channels attend {x['fg'].shape[1]}, ignore {x['bg'].shape[1]}, mixture {x['mix'].shape[1]}; "
            f"duration attend {durations['fg']:.3f} s, ignore {durations['bg']:.3f} s, mixture {durations['mix']:.3f} s")
        if abs(durations['mix'] - durations['fg']) > 0.05:
            say(f"  ! the mixture is {durations['mix'] - durations['fg']:+.3f} s longer than the attended file")

        fg_mono, bg_mono = x['fg'].mean(1), x['bg'].mean(1)
        env = {role: envelope(v, rate) for role, v in x.items()}
        envelopes[mix] = (rate, x['mix'], env['mix'])
        start, stop = (int(t * rate) for t in EXCERPT)
        stop = min(stop, len(x['mix']))
        ear_names = ['left', 'right'] if x['mix'].shape[1] == 2 else [f'channel {c}' for c in range(x['mix'].shape[1])]
        for c, ear in enumerate(ear_names):
            mix_c = x['mix'][:, c]
            found = {}
            for role, mono in (('attend', fg_mono), ('ignore', bg_mono)):
                src_env = env['fg' if role == 'attend' else 'bg'].mean(1)
                lag_env, r_env = best_lag(env['mix'][:, c], src_env, int(MAX_LAG * ENV_RATE))
                found[role] = (refine(mix_c, mono, int(round(lag_env / ENV_RATE * rate)), rate, start, stop), r_env)
            a, b, r2 = loudness_fit(mix_c, fg_mono, bg_mono, found['attend'][0], found['ignore'][0], start, stop)
            # A speaker that is not in this ear (dichotic) has no shift to speak of
            present = {role: abs(gain) >= 0.1 * max(abs(a), abs(b), 1e-12) for role, gain in (('attend', a), ('ignore', b))}
            row = []
            for role, (lag, r_env) in found.items():
                if not present[role]:
                    row.append(f"{role} not in this ear")
                    continue
                row.append(f"{role} shift {ms(lag / rate)} (envelope r {r_env:.2f})")
                if abs(lag / rate) > LAG_OK:
                    problem(f"list {list_id} seg {i} ({condition}), {ear}: {role} speaker shifted {ms(lag / rate)} "
                            f"against the mixture")
            say(f"  {ear:>6}: {'; '.join(row)}")
            say(f"          mixture = {a:.3f} x attend + {b:.3f} x ignore, R^2 {r2:.3f}")
            audio[(list_id, i, ear)] = dict(a=a if present['attend'] else None, b=b if present['ignore'] else None,
                                            r2=r2, condition=SPATIAL[i - 1])
            if SPATIAL[i - 1] == 'diotic' and r2 < 0.95:
                problem(f"list {list_id} seg {i} (diotic), {ear}: the two speakers explain only R^2 {r2:.3f} of the "
                        f"mixture (expected ~1 for a plain sum)")

# 2. List 1 vs list 2 ------------------------------------------------------------
say("\n=== 2. List 1 vs list 2 mixture for the same segment (the predictors reuse one for the other)")
for i in SEGMENTS:
    m1, m2 = f"List_1_stim_{i}", f"List_2_stim_{i}"
    if m1 not in envelopes or m2 not in envelopes:
        say(f"segment {i:>2}: not compared (audio missing)")
        continue
    rate, x1, e1 = envelopes[m1]
    _, x2, e2 = envelopes[m2]
    same_samples = x1.shape == x2.shape and numpy.array_equal(x1, x2)
    parts = []
    for c in range(min(x1.shape[1], x2.shape[1])):
        lag, r = best_lag(e1[:, c], e2[:, c], int(MAX_LAG * ENV_RATE))
        n = min(len(x1), len(x2))
        rms1, rms2 = numpy.sqrt((x1[:n, c] ** 2).mean()), numpy.sqrt((x2[:n, c] ** 2).mean())
        parts.append(f"ch {c}: shift {ms(lag / ENV_RATE)}, envelope r {r:.3f}, level ratio {rms2 / rms1:.3f}")
        if abs(lag / ENV_RATE) > LAG_OK or r < 0.99 or abs(rms2 / rms1 - 1) > 0.02:
            problem(f"segment {i} ({SPATIAL[i - 1]}): List_1 and List_2 mixtures differ (ch {c}: shift "
                    f"{ms(lag / ENV_RATE)}, r {r:.3f}, level ratio {rms2 / rms1:.3f}), but the overt/masked files "
                    f"of one list were made with the other list's mixture")
    say(f"segment {i:>2} ({SPATIAL[i - 1]}): {'identical samples' if same_samples else 'not identical'}; "
        + '; '.join(parts))

# 3. Predictors ------------------------------------------------------------------
say("\n=== 3. Predictor files (what the models use)")
masked_total = {}
for list_id in (1, 2):
    for i in SEGMENTS:
        fg = f"{SPEAKER[list_id - 1][i - 1]}_{i}"
        bg = f"{SPEAKER[-list_id + 2][i - 1]}_{i}"
        mix = f"List_{list_id}_stim_{i}"
        mix_on = predictor(mix, 'gammatone-on-8')
        mix_env = predictor(mix, 'gammatone-1')
        say(f"\nlist {list_id} segment {i:>2} ({SPATIAL[i - 1]}):")
        if mix_on is None or mix_env is None:
            say("  mixture predictors missing")
            problem(f"list {list_id} seg {i}: mixture predictor files missing")
            continue
        say(f"  mixture onsets:  {axis(mix_on)}")
        for role, stim in (('attend', fg), ('ignore', bg)):
            src_on = predictor(stim, 'gammatone-on-8')
            src_env = predictor(stim, 'gammatone-1')
            overt = predictor(stim, 'gammatone-on-overt-8')
            masked = predictor(stim, 'gammatone-on-masked-8')
            if any(v is None for v in (src_on, src_env, overt, masked)):
                say(f"  {role} {stim}: predictor files missing")
                problem(f"list {list_id} seg {i}: {role} {stim} predictor files missing")
                continue
            step = mix_on.time.tstep
            if src_on.time.tmin != mix_on.time.tmin or src_on.time.tstep != step:
                problem(f"list {list_id} seg {i}: {role} {stim} onset time axis differs from the mixture's")
            lag_env, r_env = best_lag(mix_env.x, src_env.x, int(MAX_LAG / src_env.time.tstep))
            lag_on, r_on = best_lag(mix_on.sum('frequency').x, src_on.sum('frequency').x, int(MAX_LAG / step))
            # The equations against this list's mixture, over the shared stretch
            n = min(src_on.time.nsamples, mix_on.time.nsamples, overt.time.nsamples, masked.time.nsamples)
            s = src_on.get_data(('frequency', 'time'))[:, :n]
            m = mix_on.get_data(('frequency', 'time'))[:, :n]
            saved_masked = masked.get_data(('frequency', 'time'))[:, :n]
            saved_overt = overt.get_data(('frequency', 'time'))[:, :n]
            expect_masked = numpy.clip(s - m, 0, None)
            expect_overt = numpy.minimum(s, m)
            scale = numpy.abs(s).sum() or 1
            diff_masked = numpy.abs(saved_masked - expect_masked).sum() / scale
            diff_overt = numpy.abs(saved_overt - expect_overt).sum() / scale
            share = saved_masked.sum() / ((saved_masked.sum() + saved_overt.sum()) or 1)
            masked_total.setdefault(SPATIAL[i - 1], []).append(share)
            say(f"  {role} {stim}: onsets {axis(src_on)}; overt/masked {overt.time.nsamples}/{masked.time.nsamples} samples")
            say(f"      shift vs mixture: envelope {ms(lag_env * src_env.time.tstep)} (r {r_env:.2f}), "
                f"onsets {ms(lag_on * step)} (r {r_on:.2f})")
            say(f"      saved vs this list's mixture: masked differs by {diff_masked:.2%}, overt by {diff_overt:.2%}; "
                f"masked share {share:.0%}")
            if abs(lag_env * src_env.time.tstep) > LAG_OK or abs(lag_on * step) > LAG_OK:
                problem(f"list {list_id} seg {i} ({SPATIAL[i - 1]}): {role} {stim} predictors shifted against the "
                        f"mixture's (envelope {ms(lag_env * src_env.time.tstep)}, onsets {ms(lag_on * step)})")
            if diff_masked > MATCH_OK or diff_overt > MATCH_OK:
                problem(f"list {list_id} seg {i} ({SPATIAL[i - 1]}): {role} {stim} overt/masked files do not match this "
                        f"list's mixture (masked {diff_masked:.2%}, overt {diff_overt:.2%} off); they were made with "
                        f"the other list's mixture")

# Summary ------------------------------------------------------------------------
say("\n=== Summary")
if masked_total:
    say("Masked share of onset magnitude (Brodbeck et al. 2020: ~33%): " + ', '.join(
        f"{condition} {numpy.mean(v):.0%}" for condition, v in masked_total.items()))
gains = {}
for (list_id, i, ear), fit in audio.items():
    gains.setdefault(fit['condition'], []).append(fit)
for condition, fits in gains.items():
    a = [f['a'] for f in fits if f['a'] is not None]
    b = [f['b'] for f in fits if f['b'] is not None]
    say(f"Mixture loudness, {condition} (ears where the speaker plays): attend gain {numpy.mean(a):.3f} "
        f"(range {min(a):.3f}-{max(a):.3f}), ignore gain {numpy.mean(b):.3f} (range {min(b):.3f}-{max(b):.3f}), "
        f"R^2 {numpy.mean([f['r2'] for f in fits]):.3f} (a plain sum is gain 1.000 each)")
if problems:
    say(f"\n{len(problems)} problem(s):")
    for text in problems:
        say(f"  - {text}")
else:
    say("\nNo problems: every speaker file and predictor lines up with its mixture (within 1 ms), and every "
        "overt/masked file matches its own list's mixture.")

try:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(lines) + '\n')
    say(f"\nSaved as {REPORT}")
except OSError as error:
    say(f"\nCould not save the report ({error})")
