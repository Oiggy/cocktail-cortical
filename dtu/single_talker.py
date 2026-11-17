"""What the DTU single-talker (clean speech) trials played, and whether each sound is speech: its length and how
much of its envelope's modulation lies at the syllable rate (3-8 Hz), compared with a known story. Read-only.
Run by dtu_single_talker() in analysis/dtu.ipynb."""
import pickle
import sys
import wave
from collections import Counter
from pathlib import Path

import numpy

DTU = Path(__file__).resolve().parents[2] / 'dataset' / 'dtu'
subjects = DTU / 'derivatives' / 'raw' / 'subjects'


def describe(name):
    path = DTU / f'{name}.wav'
    if not path.exists():
        return f"{name}: no file {path.name}"
    with wave.open(str(path)) as w:
        rate, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        x = numpy.frombuffer(w.readframes(n), dtype=numpy.int16).astype(float).reshape(-1, ch).mean(1)
    step = rate // 100  # 100 Hz envelope
    env = numpy.abs(x[:len(x) // step * step]).reshape(-1, step).mean(1)
    env = env - env.mean()
    spectrum = numpy.abs(numpy.fft.rfft(env)) ** 2
    freqs = numpy.fft.rfftfreq(len(env), 0.01)
    band = spectrum[(freqs >= 3) & (freqs <= 8)].sum() / spectrum[(freqs >= 0.5) & (freqs <= 32)].sum()
    silent = numpy.mean(numpy.abs(x) < 0.01 * numpy.abs(x).max())
    return (f"{name}: {n / rate:.1f} s, {ch} channel(s), {rate} Hz; share of envelope modulation at 3-8 Hz: {band:.2f}; "
            f"near-silent samples: {silent:.0%}")


played = Counter()
for path in sorted(subjects.glob('S*.pickle')):
    with open(path, 'rb') as file:
        ds = pickle.load(file)
    single = [a for a, n in zip(ds['attended'], ds['n_speakers'].x) if n == 1]
    played.update(single)
    print(f"{path.stem}: {len(single)} single-talker trials: {single}")
print()
others = sorted(n for n in played if not n.startswith(('aske_', 'marianne_')))
print("Single-talker sounds that are not one of the two talkers' stories:", others or 'none')
for name in others:
    print("  " + describe(name))
reference = next((n for n in played if n.startswith(('aske_', 'marianne_'))), None) or \
    next(p.stem for p in sorted(DTU.glob('aske_*.wav')))
print("  for comparison, a story: " + describe(reference))
