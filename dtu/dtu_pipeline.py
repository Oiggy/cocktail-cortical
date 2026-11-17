"""Forward-model (TRF) analysis of the DTU dataset (Fuglsang, Dau & Hjortkjaer 2017, NeuroImage 156:435-444;
Zenodo 1199011), the same way the cocktail notebooks analyse this project's data. Runs on the cluster only
(cluster/dtu_analysis.sbatch, started by dtu_run() in analysis/dtu.ipynb); everything it writes goes under
dataset/dtu/derivatives.

Stages (python dtu/dtu_pipeline.py <stage> [subject]):
  prepare - predictors and EEG, once:
      * every story .wav -> 8-band gammatone envelope and onset predictors at 64 Hz, exactly as
        predictors/gammatone_predictors.py makes this project's (128 bands 80-8,000 Hz, log(x + 1),
        edge_detector(c=30), summed into 8 bands)
      * which two stories each EEG trial played: the attended one (wavA) and the ignored one (wavB) are
        found by matching the authors' audio envelopes to every story, and checked against the trial
        table (expinfo, which also gives the room: anechoic / mild / high reverberation)
      * the mixture predictor of every pair: from the two stories summed, as the dry recordings (the
        rooms were simulated at presentation, so the exact mixture at the ears is not available)
      * EEG: the authors' preprocessed data (S*_data_preproc.mat: 64 Hz, eye artefacts removed),
        re-referenced to the mastoids (EXG1, EXG2) and filtered 0.5-20 Hz, as this project's EEG
  fit S - one subject: boosting TRFs as in this project's pipeline (8-band predictors, -100 to 400 ms,
      50 ms basis, l1 error, selective stopping, cross-validated over 5 partitions) for the full model
      (attended, ignored and mixture envelopes and onsets) and the full model without each predictor and
      without each whole stream; for all trials and each room. Saved model by model, so it resumes.
  stats - group tests (as this project's notebooks: one-tailed TFCE in the ROI the clean-speech notebook
      found, roi.txt, and the ROI-averaged test) and the full model's TRFs, one file per epoch, read by
      analysis/dtu.ipynb.
"""
import json
import pickle
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy

import os
REPO = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get('DTU_WORK', REPO.parent))  # DTU_WORK: only for testing elsewhere
DTU = WORK / 'dataset' / 'dtu'
DERIV = DTU / 'derivatives'
PRED_DIR = DERIV / 'predictors'
SUBJ_DIR = DERIV / 'subjects'
FIT_DIR = DERIV / 'fits'
RESULT_DIR = DERIV / 'results'
COCKTAIL_ROI = WORK / 'dataset' / 'cocktail' / 'bids' / 'derivatives' / 'results' / 'roi.txt'

SUBJECTS = [int(s) for s in os.environ['DTU_SUBJECTS'].split()] if 'DTU_SUBJECTS' in os.environ else list(range(1, 19))
FS = 64
N_TIMES = 3200            # 50 s trials at 64 Hz
TSTART, TSTOP = -0.100, 0.400
BOOSTING = dict(basis=0.050, error='l1', partitions=5, test=1, selective_stopping=1)
FILTER = (0.5, 20)
ROOMS = {1: 'anechoic', 2: 'mild', 3: 'high'}
EPOCHS = ('all', 'anechoic', 'mild', 'high')
# Predictor columns, named as in this project's TRF results ('' attended, 'bg_' ignored, 'mix_' mixture)
COLUMNS = {
    'attend': ('gammatone_8', 'gammatone_on_8'),
    'ignore': ('bg_gammatone_8', 'bg_gammatone_on_8'),
    'mix': ('mix_gammatone_8', 'mix_gammatone_on_8'),
}
ALL_COLUMNS = [c for cs in COLUMNS.values() for c in cs]
MODELS = {'full': ALL_COLUMNS}
MODELS.update({f'no_{c}': [x for x in ALL_COLUMNS if x != c] for c in ALL_COLUMNS})
MODELS.update({f'no_{s}_stream': [x for x in ALL_COLUMNS if x not in COLUMNS[s]] for s in COLUMNS})
if 'DTU_TEST_MODELS' in os.environ:  # only for testing the mechanics quickly elsewhere
    MODELS = {m: MODELS[m] for m in os.environ['DTU_TEST_MODELS'].split()}
# The comparisons the notebook reports: label -> reduced model (each tested against the full model)
COMPARISONS = {
    'Onset attend': 'no_gammatone_on_8', 'Onset ignore': 'no_bg_gammatone_on_8', 'Onset mix': 'no_mix_gammatone_on_8',
    'Envelope attend': 'no_gammatone_8', 'Envelope ignore': 'no_bg_gammatone_8', 'Envelope mix': 'no_mix_gammatone_8',
    'Attend stream': 'no_attend_stream', 'Ignore stream': 'no_ignore_stream', 'Mix stream': 'no_mix_stream',
}
COMPARISONS = {label: model for label, model in COMPARISONS.items() if model in MODELS}


def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)


def save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    with open(part, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    part.replace(path)


def load(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def cpus():
    return int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))


def audio_workers():
    "Parallel gammatone filterbanks: each holds ~2 GB for a 50 s story at 44.1 kHz"
    return max(1, min(cpus(), 16))


# ---------------------------------------------------------------------------------------------- predictors

def _wav_array(name):
    from scipy.io import wavfile
    rate, x = wavfile.read(DTU / f'{name}.wav')
    x = x.astype(numpy.float64)
    return rate, (x.mean(1) if x.ndim == 2 else x)


def _predictors_from_audio(rate, x):
    "8-band envelope and onset predictors at 64 Hz, N_TIMES long, as predictors/gammatone_predictors.py"
    from eelbrain import NDVar, UTS, gammatone_bank, edge_detector
    wav = NDVar(x, (UTS(0, 1 / rate, len(x)),), name='wav')
    gt = gammatone_bank(wav, 80, 8000, 128, location='left', tstep=0.001)
    gt_log = (gt + 1).log()
    onset = edge_detector(gt_log, c=30)
    out = {}
    for kind, y in (('env', gt_log), ('on', onset)):
        y8 = y.bin(nbins=8, func='sum', dim='frequency')
        y8 = y8.bin(1 / FS, dim='time', func='mean', label='start')
        data = y8.get_data(('frequency', 'time'))[:, :N_TIMES]
        if data.shape[1] < N_TIMES:
            data = numpy.pad(data, ((0, 0), (0, N_TIMES - data.shape[1])))
        out[kind] = NDVar(data, (y8.get_dim('frequency'), UTS(0, 1 / FS, N_TIMES)))
    return out


def _make_story(name):
    path = PRED_DIR / f'{name}.pickle'
    if not path.exists():
        save(_predictors_from_audio(*_wav_array(name)), path)
    return name


def _make_mix(pair):
    a, b = pair
    path = PRED_DIR / f'mix~{a}~{b}.pickle'
    if not path.exists():
        rate_a, xa = _wav_array(a)
        rate_b, xb = _wav_array(b)
        assert rate_a == rate_b, (a, b, rate_a, rate_b)
        n = max(len(xa), len(xb))
        mix = numpy.pad(xa, (0, n - len(xa))) + numpy.pad(xb, (0, n - len(xb)))
        save(_predictors_from_audio(rate_a, mix), path)
    return pair


def _story_envelope(name):
    "Broadband envelope at 64 Hz, to recognise which story a trial's wavA/wavB envelope came from"
    p = load(PRED_DIR / f'{name}.pickle')['env']
    return p.sum('frequency').get_data('time')


# ---------------------------------------------------------------------------------------------- trial table

def _expinfo(subject, why=None):
    """The trial table of S<subject>.mat as a list of dicts, or None. It is a MATLAB table object, which only
    mat-io reads (installed by dtu/download.sh). `why`: a list that collects the reason it could not be read."""
    why = why if why is not None else []
    try:
        from matio import load_from_mat
    except ImportError:
        why.append("mat-io is not installed in the cocktail-cortical environment")
        return None
    path = DTU / f'S{subject}.mat'
    try:
        mat = load_from_mat(str(path), variable_names=['expinfo'])
    except TypeError:
        mat = load_from_mat(str(path))
    except Exception as error:
        why.append(f"mat-io could not read S{subject}.mat ({type(error).__name__}: {error})")
        log(f"  S{subject}: expinfo not readable ({type(error).__name__}: {error})")
        return None
    table = mat.get('expinfo') if isinstance(mat, dict) else None
    if table is None:
        keys = sorted(k for k in mat if not k.startswith('__')) if isinstance(mat, dict) else type(mat).__name__
        why.append(f"no variable 'expinfo' in S{subject}.mat (variables: {keys})")
        return None
    try:
        import pandas
        if not isinstance(table, pandas.DataFrame):
            table = pandas.DataFrame(table)
        return table.to_dict('records')
    except Exception as error:
        why.append(f"expinfo is a {type(table).__name__}, not a table ({type(error).__name__}: {error}): {str(table)[:300]}")
        log(f"  S{subject}: expinfo in an unexpected form ({type(error).__name__}: {error})")
        return None


def _text(value):
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, numpy.ndarray):
        value = value.ravel()[0] if value.size else ''
    return re.sub(r'\.wav$', '', str(value).strip()) if value is not None else ''


def _number(value):
    try:
        return int(numpy.asarray(value).ravel()[0])
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------- prepare

def prepare():
    from scipy.io import loadmat
    from eelbrain import Dataset, Factor, NDVar, Sensor, UTS, Var, combine, filter_data

    stories = sorted(p.stem for p in DTU.glob('*.wav'))
    log(f"{len(stories)} stories; making their predictors ({audio_workers()} at a time)")
    try:
        import matio  # noqa: F401
        log("mat-io is installed: the trial table (expinfo) can be read")
    except ImportError:
        log("mat-io is not installed: no trial table, so no rooms (run dtu_download() again to install it)")
    with Pool(audio_workers()) as pool:
        for _ in pool.imap_unordered(_make_story, stories):
            pass
    envelopes = {name: _story_envelope(name) for name in stories}
    names = list(envelopes)
    matrix = numpy.array([(e - e.mean()) / e.std() for e in envelopes.values()])

    def match(envelope):
        z = (envelope - envelope.mean()) / envelope.std()
        n = min(len(z), matrix.shape[1])
        r = matrix[:, :n] @ z[:n] / n
        order = numpy.argsort(r)[::-1]
        return names[order[0]], float(r[order[0]]), float(r[order[0]] - r[order[1]])

    montage = Sensor.from_montage('biosemi64')
    report, pairs = [], set()
    for subject in SUBJECTS:
        mat = loadmat(DTU / f'S{subject}_data_preproc.mat', squeeze_me=True, struct_as_record=False)['data']
        n_trials = len(mat.eeg)
        channels = [str(c) for c in numpy.atleast_1d(mat.dim.chan.eeg[0])]
        info = _expinfo(subject)
        rows = None
        if info:
            rows = [row for row in info if _number(row.get('n_speakers')) == 2] or \
                   [row for row in info if _text(row.get('wavfile_female')) and _text(row.get('wavfile_male'))]
            if len(rows) != n_trials:
                log(f"  S{subject}: {len(rows)} two-talker rows in expinfo but {n_trials} trials; rooms not used")
                rows = None

        ds_rows, agree, weak = [], 0, 0
        for i in range(n_trials):
            attended, r_a, margin_a = match(numpy.asarray(mat.wavA[i], float))
            ignored, r_b, margin_b = match(numpy.asarray(mat.wavB[i], float))
            if min(r_a, r_b) < 0.5 or min(margin_a, margin_b) < 0.1:
                weak += 1
            room = side = None
            if rows:
                row = rows[i]
                sex = _number(row.get('attend_mf'))
                expected = _text(row.get('wavfile_male' if sex == 1 else 'wavfile_female'))
                agree += expected == attended
                room = ROOMS.get(_number(row.get('acoustic_condition')))
                side = {1: 'left', 2: 'right'}.get(_number(row.get('attend_lr')))
            pairs.add((attended, ignored))
            ds_rows.append((i + 1, attended, ignored, room or 'unknown', side or 'unknown', r_a, r_b))

        # EEG: mastoid reference (EXG1, EXG2), 0.5-20 Hz, the 64 scalp channels
        scalp = [c for c in channels if not c.startswith('EXG')]
        idx = [channels.index(c) for c in scalp]
        mastoids = [channels.index(c) for c in ('EXG1', 'EXG2') if c in channels]
        missing = [c for c in scalp if c not in montage.names]
        assert not missing, f"channels not in the biosemi64 montage: {missing}"
        sensor = montage[[montage.names.index(c) for c in scalp]]
        eeg = []
        for i in range(n_trials):
            x = numpy.asarray(mat.eeg[i], float)[:N_TIMES]
            if mastoids:
                x = x - x[:, mastoids].mean(1, keepdims=True)
            y = NDVar(x[:, idx].T, (sensor, UTS(0, 1 / FS, len(x))), name='eeg')
            eeg.append(filter_data(y, *FILTER))
        ds = Dataset({
            'trial': Var([r[0] for r in ds_rows]),
            'attended': Factor([r[1] for r in ds_rows]),
            'ignored': Factor([r[2] for r in ds_rows]),
            'room': Factor([r[3] for r in ds_rows]),
            'side': Factor([r[4] for r in ds_rows]),
        })
        ds['eeg'] = combine(eeg)
        ds.info['reference'] = 'mastoids (EXG1, EXG2)' if len(mastoids) == 2 else 'average (as preprocessed)'
        save(ds, SUBJ_DIR / f'S{subject}.pickle')
        line = (f"S{subject}: {n_trials} trials; stories matched by envelope (lowest r {min(min(r[5], r[6]) for r in ds_rows):.2f}"
                f"{f', {weak} weak' if weak else ''}); "
                + (f"trial table agrees on {agree}/{n_trials} attended stories; rooms "
                    f"{ {str(k): int(v) for k, v in zip(*numpy.unique(list(ds['room']), return_counts=True))} }" if rows else "no trial table: rooms unknown")
                + f"; reference: {ds.info['reference']}")
        log(line)
        report.append(line)

    log(f"{len(pairs)} story pairs; making the mixture predictors")
    with Pool(audio_workers()) as pool:
        for _ in pool.imap_unordered(_make_mix, sorted(pairs)):
            pass
    (DERIV / 'prepare_report.txt').write_text('\n'.join(report) + '\n')
    log("prepare done")


# ---------------------------------------------------------------------------------------------- rooms

def _two_talker_rows(info, n_trials, why):
    rows = [row for row in info if _number(row.get('n_speakers')) == 2] or \
           [row for row in info if _text(row.get('wavfile_female')) and _text(row.get('wavfile_male'))]
    if len(rows) != n_trials:
        why.append(f"{len(rows)} two-talker rows in expinfo but {n_trials} trials")
        return None
    return rows


def rooms():
    """Add the rooms (and attended side) from the trial table to subjects prepared without them, without
    remaking their EEG or predictors. Subjects that get rooms lose their DONE marker, so the next fit adds the
    room fits (the all-rooms fits are kept). Writes rooms_report.txt; says exactly why if the table is unreadable."""
    from eelbrain import Factor
    report = []
    try:
        import matio
        report.append(f"mat-io {getattr(matio, '__version__', '')} is installed")
    except ImportError:
        report.append("mat-io is NOT installed")
    try:
        from scipy.io import whosmat
        report.append(f"S1.mat variables: {[(n, s, c) for n, s, c in whosmat(DTU / 'S1.mat')]}")
    except Exception as error:
        report.append(f"S1.mat variables: not listable ({type(error).__name__}: {error})")
    for subject in SUBJECTS:
        path = SUBJ_DIR / f'S{subject}.pickle'
        if not path.exists():
            report.append(f"S{subject}: not prepared yet")
            continue
        ds = load(path)
        why = []
        info = _expinfo(subject, why)
        rows = _two_talker_rows(info, ds.n_cases, why) if info else None
        if rows is None:
            report.append(f"S{subject}: no rooms: {'; '.join(why)}")
            continue
        agree = sum(_text(row.get('wavfile_male' if _number(row.get('attend_mf')) == 1 else 'wavfile_female')) == attended
                    for row, attended in zip(rows, ds['attended']))
        if agree < 0.9 * ds.n_cases:
            report.append(f"S{subject}: trial table agrees on only {agree}/{ds.n_cases} attended stories; rooms not used")
            continue
        ds['room'] = Factor([ROOMS.get(_number(row.get('acoustic_condition')), 'unknown') for row in rows])
        ds['side'] = Factor([{1: 'left', 2: 'right'}.get(_number(row.get('attend_lr')), 'unknown') for row in rows])
        save(ds, path)
        (FIT_DIR / f'S{subject}' / 'DONE').unlink(missing_ok=True)
        counts = {str(k): int(v) for k, v in zip(*numpy.unique(list(ds['room']), return_counts=True))}
        report.append(f"S{subject}: trial table agrees on {agree}/{ds.n_cases} attended stories; rooms {counts}")
    text = '\n'.join(report) + '\n'
    (DERIV / 'rooms_report.txt').write_text(text)
    print(text)
    log("rooms done")


# ---------------------------------------------------------------------------------------------- fit

def _trial_predictors(ds):
    cache, out = {}, {c: [] for c in ALL_COLUMNS}

    def get(key):
        if key not in cache:
            cache[key] = load(PRED_DIR / f'{key}.pickle')
        return cache[key]

    for attended, ignored in zip(ds['attended'], ds['ignored']):
        for prefix, p in (('', get(attended)), ('bg_', get(ignored)), ('mix_', get(f'mix~{attended}~{ignored}'))):
            out[f'{prefix}gammatone_8'].append(p['env'].copy(f'{prefix}gammatone_8'))
            out[f'{prefix}gammatone_on_8'].append(p['on'].copy(f'{prefix}gammatone_on_8'))
    return out


def fit(subject):
    from eelbrain import boosting, configure
    configure(n_workers=cpus())
    ds = load(SUBJ_DIR / f'S{subject}.pickle')
    predictors = _trial_predictors(ds)
    for epoch in EPOCHS:
        index = numpy.ones(ds.n_cases, bool) if epoch == 'all' else numpy.asarray(ds['room'] == epoch)
        if index.sum() < 5:
            continue
        path = FIT_DIR / f'S{subject}' / f'{epoch}.pickle'
        done = load(path) if path.exists() else {'r': {}, 'h': None, 'n_trials': int(index.sum())}
        y = [ds['eeg'][i] for i in numpy.flatnonzero(index)]
        for model, columns in MODELS.items():
            if model in done['r']:
                continue
            x = [[p for p, keep in zip(predictors[c], index) if keep] for c in columns]
            started = time.time()
            res = boosting(y, x, TSTART, TSTOP, **BOOSTING)
            done['r'][model] = res.r
            if model == 'full':
                done['h'] = {h.name: h for h in res.h}
            save(done, path)
            log(f"S{subject} {epoch} {model}: r = {float(res.r.mean()):.4f} ({time.time() - started:.0f} s)")
    # Marker read by dtu_run() in the notebook: this subject needs no more fitting
    (FIT_DIR / f'S{subject}' / 'DONE').write_text(time.strftime('%F %T') + '\n')
    log(f"S{subject} done")


# ---------------------------------------------------------------------------------------------- stats

def _roi(sensor_names):
    names = [line.strip() for line in COCKTAIL_ROI.read_text().splitlines() if line.strip() and not line.startswith('#')]
    roi = [n for n in names if n in sensor_names]
    return roi, names


def stats(samples=10000):
    from eelbrain import Dataset, Factor, combine, testnd
    fits = {s: {e: FIT_DIR / f'S{s}' / f'{e}.pickle' for e in EPOCHS} for s in SUBJECTS}
    summary = {}
    for epoch in EPOCHS:
        subjects = [s for s in SUBJECTS if fits[s][epoch].exists() and len(load(fits[s][epoch])['r']) == len(MODELS)]
        if len(subjects) < 3:
            log(f"{epoch}: only {len(subjects)} subjects finished; skipped")
            continue
        loaded = {s: load(fits[s][epoch]) for s in subjects}
        sensor_names = list(loaded[subjects[0]]['r']['full'].sensor.names)
        roi, cocktail_roi = _roi(sensor_names)
        tests = {}
        for label, reduced in COMPARISONS.items():
            rows = [(f'S{s}', model, loaded[s]['r'][key]) for s in subjects
                    for model, key in (('test', 'full'), ('baseline', reduced))]
            data = Dataset({'subject': Factor([r[0] for r in rows], random=True),
                            'model': Factor([r[1] for r in rows])})
            data['r'] = combine([r[2] for r in rows])
            in_roi = data.copy()
            in_roi['r'] = data['r'].sub(sensor=roi)
            result = testnd.TTestRelated('r', 'model', 'test', 'baseline', 'subject', data=in_roi, tail=1,
                                         tfce=True, samples=samples)
            tests[label] = (data, result)
            log(f"{epoch} {label}: t_max = {float(numpy.nanmax(result.t.x)):.2f}, p = {float(numpy.nanmin(result.p.x)):.3f}")
        trfs = Dataset({'subject': Factor([f'S{s}' for s in subjects], random=True)})
        for column in ALL_COLUMNS:
            trfs[column] = combine([loaded[s]['h'][column] for s in subjects])
        save({'tests': tests, 'trfs': trfs, 'roi': roi, 'n_subjects': len(subjects),
              'n_trials': {f'S{s}': loaded[s]['n_trials'] for s in subjects}}, RESULT_DIR / f'{epoch}.pickle')
        summary[epoch] = {'subjects': len(subjects), 'roi': roi}
    (RESULT_DIR / 'summary.json').write_text(json.dumps(summary, indent=1))
    (RESULT_DIR / 'DONE').write_text(time.strftime('%F %T') + '\n')  # read by dtu_run()
    log("stats done")


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'prepare':
        prepare()
    elif stage == 'fit':
        fit(int(sys.argv[2]))
    elif stage == 'rooms':
        rooms()
    elif stage == 'stats':
        stats(int(os.environ.get('DTU_SAMPLES', 10000)))
    else:
        raise SystemExit(f"Unknown stage {stage!r}: prepare, rooms, fit <subject> or stats")
