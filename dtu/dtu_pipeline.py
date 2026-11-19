"""Forward-model (TRF) analysis of the DTU dataset (Fuglsang, Dau & Hjortkjaer 2017, NeuroImage 156:435-444;
Zenodo 1199011), done the way this project analyses its own data, so the two can be compared as converging
evidence. Runs on the cluster only (cluster/dtu_analysis.sbatch, started by dtu_run() in analysis/dtu.ipynb);
everything it writes goes under dataset/dtu/derivatives/raw.

Aligned with this project, step for step:
  EEG     the raw recordings (S*.mat, 512 Hz BioSemi), preprocessed as this project's pipeline does
          (analysis/experiment.py, raw pipeline): bad channels by neighbour correlation (< 0.2, worst first),
          0.5-20 Hz filter, mastoid reference (EXG1, EXG2), extended-infomax ICA (random_state 0, fitted on
          every 16th sample) with ICLabel rejecting eye-blink and heart-beat components at >= 0.9, bad channels
          interpolated; trials cut at the trigger events as the authors' preproc_data.m does, at 64 Hz
  Trials  from the trial table (expinfo): which story was played, attended talker, one or two talkers, room
  Stimuli 8-band gammatone envelope and onsets, made exactly as predictors/gammatone_predictors.py; the
          mixture from the two stories summed (the rooms were simulated at presentation, so the exact mixture at
          the ears is not available)
  TRFs    the same boosting settings (-100 to 400 ms, 50 ms basis, l1, selective stopping, cross-validated)
  Tests   all in this project's clean-speech ROI (roi.txt), found from the cocktail clean-speech whole-scalp
          tests: clean speech tested over the whole scalp, as this project's; two talkers tested within that
          ROI (one-tailed TFCE, and the ROI-averaged t-test), as this project's two-talker notebooks

Stages (python dtu/dtu_pipeline.py <stage> [subject]):
  prepare      the story predictors, once
  rawcheck     a quick check on subject 1 that the raw file, the trial table and the triggers can be read and
               agree; the subject jobs only start if it passes
  fit S        one subject: preprocessing (once), then the clean-speech fits on the single-talker trials and
               the two-talker fits (full model, without each predictor, without each stream; all rooms and
               each room). Saved model by model, so it resumes.
  stats        the group tests (clean speech over the whole scalp, two talkers within this project's ROI) and
               the TRFs, read by the dtu_*.ipynb
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
OWN = DERIV / 'raw'                 # this pipeline's EEG, fits and results (from the raw recordings)
SUBJ_DIR = OWN / 'subjects'
FIT_DIR = OWN / 'fits'
RESULT_DIR = OWN / 'results'
COCKTAIL_ROI = WORK / 'dataset' / 'cocktail' / 'bids' / 'derivatives' / 'results' / 'roi.txt'
AUTHORS_SUBJ_DIR = DERIV / 'subjects'  # the first run, from the authors' preprocessed data: a cross-check

SUBJECTS = [int(s) for s in os.environ['DTU_SUBJECTS'].split()] if 'DTU_SUBJECTS' in os.environ else list(range(1, 19))
FS = 64
TRIAL_SECONDS = 50
N_TIMES = TRIAL_SECONDS * FS  # 3200
TSTART, TSTOP = -0.100, 0.400
BOOSTING = dict(basis=0.050, error='l1', partitions=5, test=1, selective_stopping=1)
FILTER = (0.5, 20)
MASTOIDS = ('EXG1', 'EXG2')
BAD_CHANNEL_R = 0.2                 # AUTO_BAD_CHANNELS_R in analysis/parameters.py
ICA_REJECT = ('eye blink', 'heart beat')  # AUTO_ICA_REJECT_LABELS
ICA_CONFIDENCE = 0.9                # AUTO_ICA_CONFIDENCE
ALPHA = 0.05
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
# Clean speech (single-talker trials), as this project's FULL_CLEAN: attended envelope + onsets
CLEAN_MODELS = {'full': ['gammatone_8', 'gammatone_on_8'], 'no_gammatone_on_8': ['gammatone_8'],
                'no_gammatone_8': ['gammatone_on_8']}
CLEAN_COMPARISONS = {'Onset': 'no_gammatone_on_8', 'Envelope': 'no_gammatone_8'}
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
    part = path.with_name(f'{path.name}.{os.getpid()}.part')  # several jobs may write the same predictor
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

def _table_parts(obj, depth=0):
    "The {'data', 'varnames'} parts of a MATLAB table somewhere inside what mat-io returns unconverted, or None"
    if depth > 6 or obj is None:
        return None
    props = getattr(obj, 'properties', None)
    if props is not None:
        found = _table_parts(props, depth + 1)
        if found:
            return found
    if isinstance(obj, numpy.ndarray) and obj.dtype.names:
        if {'data', 'varnames'} <= set(obj.dtype.names):
            return {n: obj[n].flat[0] for n in obj.dtype.names}
        return _table_parts(obj.flat[0] if obj.size else None, depth + 1)
    if hasattr(obj, 'keys'):
        if 'data' in obj and 'varnames' in obj:
            return obj
        for value in obj.values():
            found = _table_parts(value, depth + 1)
            if found:
                return found
    elif isinstance(obj, (list, tuple)) or (isinstance(obj, numpy.ndarray) and obj.dtype == object):
        for value in (obj if isinstance(obj, (list, tuple)) else obj.flat):
            found = _table_parts(value, depth + 1)
            if found:
                return found
    return None


def _scalar(v):
    "A MATLAB cell entry as a plain value: one-element arrays unwrapped, empty ones as ''"
    while isinstance(v, (numpy.ndarray, list, tuple)):
        flat = numpy.ravel(numpy.asarray(v, dtype=object)) if not isinstance(v, numpy.ndarray) else v.ravel()
        if flat.size == 0:
            return ''
        if flat.size > 1:
            return ''.join(map(str, flat)) if getattr(v, 'dtype', None) is not None and v.dtype.kind in 'US' else v
        v = flat[0]
    return v.item() if isinstance(v, numpy.generic) else v


def _mat_strings(props):
    "A MATLAB string array, stored unconverted: version, ndims, dims, one char count per string, then UTF-16 text"
    data = numpy.asarray(props['any'], numpy.uint64).ravel()
    ndims = int(data[1])
    n = int(numpy.prod(data[2:2 + ndims]))
    counts = data[2 + ndims:2 + ndims + n]
    text = data[2 + ndims + n:].tobytes()
    out, pos = [], 0
    for count in counts:
        if count >= 2 ** 63:  # missing string
            out.append('')
            continue
        out.append(text[pos:pos + 2 * int(count)].decode('utf-16-le'))
        pos += 2 * int(count)
    return out


def _column(col):
    "One MATLAB table column as a list: cells and numbers as values, char matrices and strings as one string per row"
    if getattr(col, 'classname', None) == 'string':
        return _mat_strings(col.properties)
    if isinstance(col, numpy.ndarray) and col.dtype == object:
        return [_scalar(v) for v in col.ravel(order='F')]
    if isinstance(col, numpy.ndarray) and col.dtype.kind in 'US' and col.ndim == 2 and col.shape[1] > 1:
        return [''.join(map(str, row)).strip() for row in col]
    return list(numpy.ravel(col, order='F'))


def _table_from_parts(parts):
    import pandas
    names = [str(_scalar(v)) for v in numpy.ravel(parts['varnames'], order='F')]
    data = list(numpy.ravel(parts['data'], order='F')) if isinstance(parts['data'], numpy.ndarray) else list(parts['data'])
    return pandas.DataFrame({name: _column(col) for name, col in zip(names, data)})


def _expinfo(subject, why=None):
    """The trial table of S<subject>.mat as a list of dicts, or None. It is a MATLAB table object, which only
    mat-io reads (installed by dtu/download.sh). `why`: a list that collects the reason it could not be read."""
    why = why if why is not None else []
    try:
        import matio
        from matio import load_from_mat
    except ImportError:
        why.append("mat-io is not installed in the cocktail-cortical environment")
        return None
    path = DTU / f'S{subject}.mat'
    try:
        mat = load_from_mat(str(path), variable_names=['expinfo'])
    except Exception as first:
        # Tables saved before MATLAB R2018a (as the DTU ones) lack fields mat-io's converter expects: read the
        # table unconverted and build it from its own columns and variable names
        try:
            raw = load_from_mat(str(path), variable_names=['expinfo'], raw_data=True)
            parts = _table_parts(raw.get('expinfo') if hasattr(raw, 'get') else raw)
            if parts is None:
                raise ValueError(f"no table columns found in: {repr(raw)[:600]}")
            mat = {'expinfo': _table_from_parts(parts)}
        except Exception as error:
            why.append(f"mat-io {getattr(matio, '__version__', '?')} could not read the trial table of S{subject}.mat "
                       f"({type(first).__name__}: {first}; unconverted: {type(error).__name__}: {error})")
            log(f"  S{subject}: expinfo not readable: {why[-1]}")
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
    "The predictors of every story (the mixtures are made when a subject's fit first needs them)"
    stories = sorted(p.stem for p in DTU.glob('*.wav'))
    log(f"{len(stories)} stories; making their predictors ({audio_workers()} at a time)")
    with Pool(audio_workers()) as pool:
        for _ in pool.imap_unordered(_make_story, stories):
            pass
    (DERIV / 'prepare_report.txt').write_text(f"{len(stories)} story predictors made\n")
    log("prepare done")


# ---------------------------------------------------------------------------------------------- raw EEG

def _h5(f, x):
    "A MATLAB v7.3 (HDF5) value as Python: cells as lists, char arrays as str, numbers as squeezed arrays"
    import h5py
    if isinstance(x, h5py.Group):
        return {k: _h5(f, x[k]) for k in x}
    if isinstance(x, h5py.Reference):
        return _h5(f, f[x])
    data = x[()]
    if x.dtype == h5py.ref_dtype or (hasattr(data, 'dtype') and data.dtype == object):
        return [_h5(f, f[ref]) for ref in numpy.ravel(data, order='F')]
    if x.attrs.get('MATLAB_class', b'') in (b'char', 'char'):
        return ''.join(chr(c) for c in numpy.ravel(data, order='F'))
    return numpy.squeeze(data)


def _flat(x):
    "Unwrap MATLAB cells of one element"
    while isinstance(x, (list, tuple)) and len(x) == 1 or (isinstance(x, numpy.ndarray) and x.dtype == object and x.size == 1):
        x = x[0] if isinstance(x, (list, tuple)) else x.flat[0]
    return x


def _numbers(x):
    if isinstance(x, (list, tuple)) or (isinstance(x, numpy.ndarray) and x.dtype == object):
        return numpy.array([float(numpy.ravel(_flat(v))[0]) for v in (x if isinstance(x, (list, tuple)) else x.flat)])
    return numpy.atleast_1d(numpy.asarray(x, float)).ravel()


def _load_raw(subject):
    """Subject's raw recording: (data samples x channels, sampling rate, channel names, event samples (0-based),
    event values). The COCOHA layout of preproc_data.m: data.eeg, data.fsample.eeg, data.dim.chan.eeg, data.event.eeg"""
    path = DTU / f'S{subject}.mat'
    try:
        from scipy.io import loadmat
        d = loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=['data'])['data']
        eeg, fs, chans = _flat(d.eeg), float(_flat(d.fsample.eeg)), _flat(d.dim.chan.eeg)
        sample, value = d.event.eeg.sample, d.event.eeg.value
    except NotImplementedError:  # MATLAB v7.3
        import h5py
        with h5py.File(path, 'r') as f:
            d = _h5(f, f['data'])
        eeg, fs, chans = _flat(d['eeg']), float(_flat(d['fsample']['eeg'])), _flat(d['dim']['chan']['eeg'])
        sample, value = d['event']['eeg']['sample'], d['event']['eeg']['value']
    eeg = numpy.asarray(eeg, float)
    chans = [str(_flat(c)) for c in (chans if isinstance(chans, (list, tuple)) else numpy.ravel(chans))]
    if eeg.shape[0] == len(chans) and eeg.shape[1] != len(chans):
        eeg = eeg.T
    sample, value = _numbers(sample).astype(int) - 1, _numbers(value)
    return eeg, fs, chans, sample, value


def _trials(subject, why):
    "The two-talker and single-talker trials from the trial table, with where each starts in the recording"
    info = _expinfo(subject, why)
    if not info:
        return None
    eeg, fs, chans, sample, value = _load_raw(subject)
    triggers = numpy.array([_number(row.get('trigger')) for row in info], float)
    # preproc_data.m: the trial starts are every other event, and their values are the table's triggers
    starts = sample[0::2]
    if len(starts) != len(info) or not numpy.array_equal(value[0::2], triggers):
        hits = [i for i, v in enumerate(value) if v in set(triggers)]
        if len(hits) == len(info) and numpy.array_equal(value[hits], triggers):
            starts = sample[hits]
        else:
            why.append(f"the {len(value)} trigger events do not match the {len(info)} trials of the trial table "
                       f"(first event values {value[:8].tolist()}, first table triggers {triggers[:8].tolist()})")
            return None
    trials = []
    for row, start in zip(info, starts):
        n = _number(row.get('n_speakers'))
        male, female = _text(row.get('wavfile_male')), _text(row.get('wavfile_female'))
        sex = _number(row.get('attend_mf'))
        attended, ignored = (male, female) if sex == 1 else (female, male)
        if n is None:
            n = 2 if male and female else 1
        trials.append(dict(start=int(start), n_speakers=n, attended=attended, ignored=ignored if n == 2 else '',
                           room=ROOMS.get(_number(row.get('acoustic_condition')), 'unknown'),
                           side={1: 'left', 2: 'right'}.get(_number(row.get('attend_lr')), 'unknown')))
    return eeg, fs, chans, trials


def _neighbour_bads(raw, threshold=BAD_CHANNEL_R):
    "As this project's automatic bad channels: drop the channel least correlated with its neighbours until all are >= threshold"
    import mne
    adjacency, names = mne.channels.find_ch_adjacency(raw.info, 'eeg')
    adjacency = adjacency.toarray().astype(bool)
    corr = numpy.corrcoef(raw.get_data(picks=names))
    bads = []
    while True:
        good = [i for i, n in enumerate(names) if n not in bads]
        scores = {}
        for i in good:
            neighbours = [j for j in good if j != i and adjacency[i, j]]
            if neighbours:
                scores[i] = corr[i, neighbours].mean()
        worst = min(scores, key=scores.get)
        if scores[worst] >= threshold:
            return bads
        bads.append(names[worst])


def preprocess(subject):
    """Subject's raw EEG preprocessed as this project's pipeline, cut into trials at 64 Hz; saved with the trial
    table's information and a report"""
    import mne
    from mne.preprocessing import ICA
    from mne_icalabel import label_components
    from eelbrain import Dataset, Factor, NDVar, Sensor, UTS, Var, combine
    mne.set_log_level('WARNING')
    why = []
    loaded = _trials(subject, why)
    if loaded is None:
        raise SystemExit(f"S{subject}: cannot use the raw recording: {'; '.join(why)}")
    eeg, fs, chans, trials = loaded
    montage = mne.channels.make_standard_montage('biosemi64')
    scalp = [c for c in montage.ch_names if c in chans]
    assert len(scalp) == 64, f"S{subject}: only {len(scalp)} of the 64 BioSemi channels found in {chans[:70]}"
    keep = scalp + [c for c in MASTOIDS if c in chans]
    x = eeg[:, [chans.index(c) for c in keep]].T
    if numpy.abs(x).max() > 1:  # BioSemi data in microvolts; MNE wants volts
        x = x * 1e-6
    raw_orig = mne.io.RawArray(x, mne.create_info(keep, fs, 'eeg'))
    raw_orig.set_montage(montage, on_missing='ignore')

    raw = raw_orig.copy().filter(*FILTER)
    if all(c in keep for c in MASTOIDS):
        raw.set_eeg_reference(list(MASTOIDS))
        reference = 'mastoids (EXG1, EXG2)'
    else:
        raw.set_eeg_reference('average')
        reference = 'average (no mastoid channels)'
    raw.pick(scalp)
    raw.info['bads'] = _neighbour_bads(raw)
    ica = ICA(method='infomax', fit_params=dict(extended=True), random_state=0)
    ica.fit(raw, decim=16)
    # ICLabel on a copy filtered 1-100 Hz and averaged-referenced, as it expects (this project does the same)
    probe = raw_orig.copy().crop(0, min(300, raw_orig.times[-1])).pick(ica.ch_names)
    probe.filter(1., min(100., fs / 2 - 1)).set_eeg_reference('average')
    labels = label_components(probe, ica, method='iclabel')
    ica.exclude = [i for i, (label, p) in enumerate(zip(labels['labels'], labels['y_pred_proba']))
                   if label in ICA_REJECT and p >= ICA_CONFIDENCE]
    rejected = [labels['labels'][i] for i in ica.exclude]
    ica.apply(raw)
    bads = list(raw.info['bads'])
    if bads:
        raw.interpolate_bads(reset_bads=True)

    sensor = Sensor.from_montage('biosemi64')
    sensor = sensor[[sensor.names.index(c) for c in scalp]]
    data = raw.get_data(picks=scalp)
    step = int(round(fs * TRIAL_SECONDS))
    epochs = []
    for trial in trials:
        segment = data[:, trial['start']:trial['start'] + step]
        if segment.shape[1] < step:
            segment = numpy.pad(segment, ((0, 0), (0, step - segment.shape[1])))
        segment = mne.filter.resample(segment, down=fs / FS, npad='auto')[:, :N_TIMES]
        epochs.append(NDVar(segment, (sensor, UTS(0, 1 / FS, N_TIMES)), name='eeg'))
    ds = Dataset({
        'trial': Var(list(range(1, len(trials) + 1))),
        'n_speakers': Var([t['n_speakers'] for t in trials]),
        'attended': Factor([t['attended'] for t in trials]),
        'ignored': Factor([t['ignored'] for t in trials]),
        'room': Factor([t['room'] for t in trials]),
        'side': Factor([t['side'] for t in trials]),
    })
    ds['eeg'] = combine(epochs)
    two = [t for t in trials if t['n_speakers'] == 2]
    rooms = {r: sum(t['room'] == r for t in two) for r in sorted({t['room'] for t in two})}
    check = ''
    authors = AUTHORS_SUBJ_DIR / f'S{subject}.pickle'
    if authors.exists():  # the stories found earlier by envelope matching in the authors' preprocessed data
        matched = list(load(authors)['attended'])
        agree = sum(a == t['attended'] for a, t in zip(matched, two))
        check = f"; attended stories agree with the envelope matching on {agree}/{len(two)} two-talker trials"
    report = (f"S{subject}: {len(two)} two-talker + {len(trials) - len(two)} single-talker trials; rooms {rooms}; "
              f"reference {reference}; bad channels {bads or 'none'}; ICA rejected {len(ica.exclude)}/{ica.n_components_} "
              f"components ({', '.join(rejected) or 'none'}){check}")
    ds.info['report'] = report
    save(ds, SUBJ_DIR / f'S{subject}.pickle')
    (SUBJ_DIR / f'S{subject}_report.txt').write_text(report + '\n')
    log(report)
    return ds


def rawcheck(subject=1):
    "Checks on one subject that the raw recording, the trial table and the triggers can be read and agree"
    why = []
    try:
        import matio  # noqa: F401
    except ImportError:
        raise SystemExit("FAILED: mat-io is not installed in the cocktail-cortical environment (run dtu_run() "
                         "from the Mac: it installs it on the login node)")
    try:
        eeg, fs, chans, sample, value = _load_raw(subject)
    except Exception as error:
        raise SystemExit(f"FAILED: S{subject}.mat could not be read as the COCOHA layout "
                         f"({type(error).__name__}: {error})")
    log(f"S{subject}.mat: {eeg.shape[0]} samples x {eeg.shape[1]} channels at {fs:g} Hz "
        f"({eeg.shape[0] / fs / 60:.1f} min); channels {chans[:3]} ... {chans[-9:]}; {len(value)} events, "
        f"first values {value[:6].tolist()}")
    loaded = _trials(subject, why)
    if loaded is None:
        raise SystemExit(f"FAILED: {'; '.join(why)}")
    trials = loaded[3]
    two = [t for t in trials if t['n_speakers'] == 2]
    log(f"OK: trial table read; {len(two)} two-talker and {len(trials) - len(two)} single-talker trials; rooms "
        f"{ {r: sum(t['room'] == r for t in two) for r in sorted({t['room'] for t in two})} }; first trial "
        f"{trials[0]}")


# ---------------------------------------------------------------------------------------------- fit

def _trial_predictors(ds, index, columns):
    cache = {}

    def get(key):
        if key not in cache:
            path = PRED_DIR / f'{key}.pickle'
            if not path.exists() and key.startswith('mix~'):
                _make_mix(tuple(key.split('~')[1:]))
            cache[key] = load(path)
        return cache[key]

    out = {c: [] for c in columns}
    for i in numpy.flatnonzero(index):
        attended, ignored = ds[i, 'attended'], ds[i, 'ignored']
        sources = {'': attended, 'bg_': ignored, 'mix_': f'mix~{attended}~{ignored}'}
        for c in columns:
            prefix = re.match(r'^(bg_|mix_)?', c).group(0)
            p = get(sources[prefix])
            out[c].append((p['on'] if '_on_' in c else p['env']).copy(c))
    return out


def _fit_models(subject, epoch, ds, index, models):
    from eelbrain import boosting, concatenate
    path = FIT_DIR / f'S{subject}' / f'{epoch}.pickle'
    done = load(path) if path.exists() else {'r': {}, 'h': None, 'n_trials': int(index.sum())}
    y = [ds['eeg'][i] for i in numpy.flatnonzero(index)]
    columns = sorted({c for cs in models.values() for c in cs})
    predictors = _trial_predictors(ds, index, columns)
    joined = len(y) < BOOSTING['partitions']  # too few trials to spread over the partitions: one continuous segment
    if joined:
        y = concatenate(y, 'time')
        predictors = {c: concatenate(p, 'time') for c, p in predictors.items()}
    for model, cols in models.items():
        if model in done['r']:
            continue
        x = [predictors[c] for c in cols]
        started = time.time()
        res = boosting(y, x, TSTART, TSTOP, **BOOSTING)
        done['r'][model] = res.r
        if model == 'full':
            done['h'] = {h.name: h for h in res.h}
        save(done, path)
        log(f"S{subject} {epoch} {model}: r = {float(res.r.mean()):.4f} ({time.time() - started:.0f} s)")


def fit(subject):
    from eelbrain import configure
    configure(n_workers=cpus())
    path = SUBJ_DIR / f'S{subject}.pickle'
    ds = load(path) if path.exists() else preprocess(subject)
    single = numpy.asarray(ds['n_speakers'].x == 1)
    two = numpy.asarray(ds['n_speakers'].x == 2)
    if single.sum() >= 1:
        _fit_models(subject, 'clean', ds, single, CLEAN_MODELS)
    for epoch in EPOCHS:
        index = two if epoch == 'all' else two & numpy.asarray(ds['room'] == epoch)
        if index.sum() >= 5:
            _fit_models(subject, epoch, ds, index, MODELS)
    # Marker read by dtu_run() in the notebook: this subject needs no more fitting
    (FIT_DIR / f'S{subject}' / 'DONE').write_text(time.strftime('%F %T') + '\n')
    log(f"S{subject} done")


# ---------------------------------------------------------------------------------------------- stats

def _names(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith('#')]


def _complete(epoch, models):
    out = {}
    for s in SUBJECTS:
        path = FIT_DIR / f'S{s}' / f'{epoch}.pickle'
        if path.exists():
            fit_ = load(path)
            if set(models) <= set(fit_['r']):
                out[s] = fit_
    return out


def _test(loaded, full, reduced, sensors, samples):
    from eelbrain import Dataset, Factor, combine, testnd
    rows = [(f'S{s}', model, f['r'][key]) for s, f in loaded.items() for model, key in (('test', full), ('baseline', reduced))]
    data = Dataset({'subject': Factor([r[0] for r in rows], random=True), 'model': Factor([r[1] for r in rows])})
    data['r'] = combine([r[2] for r in rows])
    tested = data.copy()
    if sensors is not None:
        tested['r'] = data['r'].sub(sensor=sensors)
    result = testnd.TTestRelated('r', 'model', 'test', 'baseline', 'subject', data=tested, tail=1, tfce=True,
                                 samples=samples)
    return data, result


def _trfs(loaded, columns):
    from eelbrain import Dataset, Factor, combine
    trfs = Dataset({'subject': Factor([f'S{s}' for s in loaded], random=True)})
    for column in columns:
        trfs[column] = combine([f['h'][column] for f in loaded.values()])
    return trfs


def stats(samples=10000):
    """Group results, all in this project's clean-speech ROI (roi.txt), found from the cocktail clean-speech
    whole-scalp tests:
      clean       TFCE over the whole scalp (as this project's clean speech); ROI means and TRFs in that ROI
      two talkers TFCE within that ROI, as this project's two-talker conditions
    """
    summary = {}
    cocktail_names = _names(COCKTAIL_ROI) if COCKTAIL_ROI.exists() else []
    if not cocktail_names:
        raise SystemExit(f"This project's ROI ({COCKTAIL_ROI}) is missing: run the clean notebook first")

    # 1. Clean speech (single-talker trials): whole-scalp tests, as this project's clean speech
    clean = _complete('clean', CLEAN_MODELS)
    if len(clean) >= 3:
        tests = {}
        for label, reduced in CLEAN_COMPARISONS.items():
            data, result = _test(clean, 'full', reduced, None, samples)
            tests[label] = (data, result)
            n = int(sum(p <= ALPHA for p in result.p.x))
            log(f"clean {label}: t_max = {float(numpy.nanmax(result.t.x)):.2f}, "
                f"p = {float(numpy.nanmin(result.p.x)):.3f}, {n} significant sensors")
        names = list(result.p.sensor.names)
        rois = {'cocktail': [n for n in cocktail_names if n in names]}
        save({'tests': {'scalp': tests}, 'trfs': _trfs(clean, CLEAN_MODELS['full']), 'rois': rois,
              'n_subjects': len(clean), 'n_trials': {f'S{s}': f['n_trials'] for s, f in clean.items()}},
             RESULT_DIR / 'clean.pickle')
        summary['clean'] = {'subjects': len(clean)}
    else:
        log(f"clean: only {len(clean)} subjects have clean-speech fits; skipped")

    # 2. The two-talker tests, within this project's ROI
    for epoch in EPOCHS:
        loaded = _complete(epoch, MODELS)
        if len(loaded) < 3:
            log(f"{epoch}: only {len(loaded)} subjects; skipped")
            continue
        names = list(next(iter(loaded.values()))['r']['full'].sensor.names)
        rois = {'cocktail': [n for n in cocktail_names if n in names]}
        tests = {'cocktail': {}}
        for label, reduced in COMPARISONS.items():
            tests['cocktail'][label] = _test(loaded, 'full', reduced, rois['cocktail'], samples)
            result = tests['cocktail'][label][1]
            log(f"{epoch} {label} (this project's ROI): t_max = {float(numpy.nanmax(result.t.x)):.2f}, "
                f"p = {float(numpy.nanmin(result.p.x)):.3f}")
        save({'tests': tests, 'trfs': _trfs(loaded, ALL_COLUMNS), 'rois': rois,
              'n_subjects': len(loaded), 'n_trials': {f'S{s}': f['n_trials'] for s, f in loaded.items()}},
             RESULT_DIR / f'{epoch}.pickle')
        summary[epoch] = {'subjects': len(loaded)}
    (RESULT_DIR / 'summary.json').write_text(json.dumps(summary, indent=1))
    (RESULT_DIR / 'DONE').write_text(time.strftime('%F %T') + '\n')  # read by dtu_run()
    log("stats done")


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'prepare':
        prepare()
    elif stage == 'rawcheck':
        rawcheck()
    elif stage == 'fit':
        fit(int(sys.argv[2]))
    elif stage == 'stats':
        stats(int(os.environ.get('DTU_SAMPLES', 10000)))
    else:
        raise SystemExit(f"Unknown stage {stage!r}: prepare, rawcheck, fit <subject> or stats")
