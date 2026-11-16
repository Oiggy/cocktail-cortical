"""Report how the downloaded DTU dataset (dataset/dtu, dtu/download.sh) is laid out: folders, the
structure of one subject's .mat files (fields, shapes, sampling rates, channel names, events, the
expinfo trial table) and the audio files. The forward-model analysis is written against this report.

Run on the cluster (cluster/dtu_inspect.sbatch, or dtu_inspect() in a notebook). Writes only
dataset/dtu/inspect_report.txt and prints the same.
"""
import io
import sys
import wave
from pathlib import Path

import numpy

DTU = Path(__file__).resolve().parents[2] / 'dataset' / 'dtu'
out = io.StringIO()


def say(*args):
    print(*args, file=out)


def describe(x, name, depth=0, max_depth=8):
    pad = '  ' * depth
    if depth > max_depth:
        say(f"{pad}{name}: ...")
        return
    if isinstance(x, dict):
        say(f"{pad}{name}: dict with {len(x)} keys")
        for k, v in x.items():
            if not str(k).startswith('__'):
                describe(v, k, depth + 1, max_depth)
    elif isinstance(x, numpy.ndarray) and x.dtype.names:
        say(f"{pad}{name}: struct array {x.shape}, fields {list(x.dtype.names)}")
        if x.size:
            first = x.flat[0]
            for field in x.dtype.names:
                describe(first[field], field, depth + 1, max_depth)
    elif isinstance(x, numpy.ndarray) and x.dtype == object and x.size and all(isinstance(i, str) for i in x.flat):
        say(f"{pad}{name}: {x.size} strings = {list(x.flat)[:80]}")
    elif isinstance(x, numpy.ndarray) and x.dtype == object:
        say(f"{pad}{name}: cell/object array {x.shape}")
        for i, item in enumerate(x.flat):
            if i >= 3:
                say(f"{pad}  ... ({x.size} items)")
                break
            describe(item, f"[{i}]", depth + 1, max_depth)
    elif isinstance(x, numpy.ndarray):
        text = f"{pad}{name}: array {x.shape} {x.dtype}"
        if x.dtype.kind in 'USa':
            text += f" = {x.ravel()[:8].tolist()}"
        elif x.size <= 8:
            text += f" = {x.ravel().tolist()}"
        elif x.dtype.kind in 'fiu' and x.size:
            text += f" (min {numpy.nanmin(x):.4g}, max {numpy.nanmax(x):.4g})"
        say(text)
    elif hasattr(x, '_fieldnames'):  # scipy mat_struct
        say(f"{pad}{name}: struct, fields {x._fieldnames}")
        for field in x._fieldnames:
            describe(getattr(x, field), field, depth + 1, max_depth)
    else:
        say(f"{pad}{name}: {type(x).__name__} = {str(x)[:200]}")


def load_mat(path):
    try:
        from scipy.io import loadmat
        return 'scipy.io.loadmat', loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:  # MATLAB v7.3 = HDF5
        import h5py
        f = h5py.File(path, 'r')
        tree = {}
        f.visititems(lambda k, v: tree.__setitem__(k, getattr(v, 'shape', None)))
        return 'h5py (v7.3)', tree


say(f"DTU dataset folder: {DTU}")
if not DTU.exists():
    say("Not there yet: run dtu_download() first.")
else:
    say("\n== Folders and file counts")
    for folder in sorted({p.parent for p in DTU.rglob('*') if p.is_file()}):
        files = sorted(p for p in folder.iterdir() if p.is_file())
        size = sum(p.stat().st_size for p in files) / 1e9
        say(f"  {folder.relative_to(DTU)}: {len(files)} files, {size:.2f} GB; e.g. {[p.name for p in files[:4]]}")

    for kind in ('EEG', 'DATA_preproc', 'preproc'):
        mats = sorted(DTU.rglob(f'*{kind}*/**/*.mat')) or sorted(p for p in DTU.rglob('*.mat') if kind.lower() in str(p).lower())
        if not mats:
            continue
        path = mats[0]
        say(f"\n== {kind}: {len(mats)} .mat files; structure of {path.relative_to(DTU)}")
        try:
            how, mat = load_mat(path)
            say(f"  read with {how}")
            describe(mat, path.name, 1)
        except Exception as error:
            say(f"  could not read: {type(error).__name__}: {error}")
        # The trial table (expinfo) is often a MATLAB table object, which scipy cannot read
        try:
            from matio import load_from_mat
            data = load_from_mat(str(path))
            for key, value in data.items():
                if 'exp' in key.lower() or 'info' in key.lower():
                    say(f"  {key} via mat-io:\n{str(value)[:3000]}")
        except ImportError:
            say("  (mat-io not installed: a MATLAB table such as expinfo shows as opaque above)")
        except Exception as error:
            say(f"  mat-io could not read it: {type(error).__name__}: {error}")

    wavs = sorted(DTU.rglob('*.wav'))
    say(f"\n== Audio: {len(wavs)} .wav files")
    for path in wavs[:6]:
        try:
            with wave.open(str(path)) as w:
                say(f"  {path.relative_to(DTU)}: {w.getnchannels()} ch, {w.getframerate()} Hz, "
                    f"{w.getnframes() / w.getframerate():.1f} s, {8 * w.getsampwidth()} bit")
        except Exception as error:
            say(f"  {path.relative_to(DTU)}: {type(error).__name__}: {error}")
    others = sorted(p for p in DTU.rglob('*') if p.is_file() and p.suffix in ('.m', '.txt', '.csv', '.md'))
    for path in others[:5]:
        n = 400 if path.suffix == '.m' else 60
        say(f"\n== {path.relative_to(DTU)} (first {n} lines)")
        say(''.join(path.read_text(errors='replace').splitlines(keepends=True)[:n]))

report = out.getvalue()
print(report)
if DTU.exists():
    (DTU / 'inspect_report.txt').write_text(report)
sys.exit(0)
