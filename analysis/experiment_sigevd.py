"""
Everything in experiment.py, plus the machinery to apply a Stimulus-
Informed Generalized Eigenvalue Decomposition (SI-GEVD) spatial denoising
filter as a real raw-pipeline stage, between ICA artifact removal and
epoching - the same position 'ica' itself sits in
cortical_analysis_1.ipynb's/cortical_analysis.ipynb's pipeline, just one
step further along.

Kept in a separate file, not merged into experiment.py, so nothing here
can ever affect cortical_analysis.ipynb / cortical_analysis_1.ipynb, which
both import from experiment.py directly and never see this file.

How it works
------------
- `RawSIGEVD` is a new raw-pipeline stage (`CachedRawPipe` subclass,
  structurally the same kind of stage as `RawFilter`/`RawReReference` - a
  fixed linear transform applied to the continuous EEG), registered here
  as 'ica-sigevd', built on top of the existing 'ica' stage.
- The transform itself is one SI-GEVD spatial filter matrix per subject,
  fit once - pooling every condition with real two-talker/single-talker
  EEG (clean, diotic, binaural, dichotic), so one filter denoises whatever
  condition later asks for 'ica-sigevd' data - and cached to disk under
  derivatives/sigevd/. `RawSIGEVD._make()` loads that cached filter
  (fitting and caching it first if it doesn't exist yet) and applies it to
  the continuous EEG channels.
- Because this is a real raw-pipeline stage - not an ad-hoc, per-epoch
  transform bolted on after the fact - every e.load_trfs()/
  e.load_model_test() call already written throughout
  cortical_analysis_2.ipynb picks up SI-GEVD-denoised EEG automatically
  the moment PARAMETERS['raw'] changes from 'ica' to 'ica-sigevd', with no
  change to any of those calls themselves.

SIGEVD_NO_OF_COMPS/SIGEVD_LAM below are the paper's own example values,
not tuned for this dataset yet - see sigevd.cross_validate_sigevd for the
leave-one-trial-out procedure to pick them properly later. Changing either
one here automatically invalidates every subject's cached 'ica-sigevd' raw
file (both are part of RawSIGEVD's fingerprint) - but NOT the per-subject
filter .npz cache below, which this module manages itself outside
eelbrain's cache tracking; delete derivatives/sigevd/ by hand to force a
refit (e.g. after changing NO_OF_COMPS/LAM, or after redoing a subject's
bad channels/ICA upstream).
"""
from pathlib import Path

import mne
import numpy
from eelbrain import Datalist, NDVar, load
from eelbrain._experiment.preprocessing.config import CachedRawPipe

from experiment import BinauralCocktail, DATA_ROOT
from sigevd import find_sigevd_filter, apply_sigevd_filter

PREDICTOR_DIR = Path(DATA_ROOT) / 'derivatives' / 'predictors'
SIGEVD_CACHE_DIR = Path(DATA_ROOT) / 'derivatives' / 'sigevd'

# Matches cortical_analysis_2.ipynb's own PARAMETERS - the filter has to be
# fit at the same sample rate and lag window it will actually be used with.
SIGEVD_SAMPLINGRATE = 128
SIGEVD_TSTART = -0.100  # seconds
SIGEVD_TSTOP = 0.600  # seconds
SIGEVD_NO_OF_COMPS = 2  # the paper's own example value - not tuned for this dataset yet
SIGEVD_LAM = 0.2  # the paper's own example value - not tuned for this dataset yet


def _load_onset(stim, samplingrate):
    "Broadband onset predictor for one stimulus, resampled to match the EEG (mirrors UTSPredictor(resample='bin'))"
    x = load.unpickle(PREDICTOR_DIR / f'{stim}~gammatone-on-1.pickle')
    return x.bin(step=1 / samplingrate, label='start')


def _eeg_ndvar(ds):
    "The EEG column in a load_epochs() Dataset - a Datalist of per-trial NDVars when trials have different durations, or a single case-dimensioned NDVar otherwise - whatever it's named"
    keys = [k for k, v in ds.items() if isinstance(v, (NDVar, Datalist))]
    assert len(keys) == 1, f"expected exactly one EEG column, found {keys}"
    return ds[keys[0]]


def _trial_arrays(ds, samplingrate, stim_column):
    "Per-trial (time, sensor) EEG and matching (time,) onset predictor arrays for one stim_column ('fg' or 'bg')"
    eeg = _eeg_ndvar(ds)
    resp_trials, stim_trials = [], []
    for i in range(len(eeg)):
        resp_i = eeg[i].get_data(('time', 'sensor'))
        stim_i = _load_onset(ds[i, stim_column], samplingrate).x
        n = min(len(resp_i), len(stim_i))
        resp_trials.append(resp_i[:n])
        stim_trials.append(stim_i[:n])
    return numpy.concatenate(stim_trials), numpy.vstack(resp_trials)


def _condition_pairs_single(ds, samplingrate):
    "Clean-style: one stimulus, one TRF"
    stim, resp = _trial_arrays(ds, samplingrate, 'fg')
    return [(stim, resp)], resp


def _condition_pairs_shared(ds, samplingrate):
    "Diotic/binaural-style: same EEG, two concurrent streams (attend + ignore)"
    attend_stim, resp = _trial_arrays(ds, samplingrate, 'fg')
    ignore_stim, _ = _trial_arrays(ds, samplingrate, 'bg')
    return [(attend_stim, resp), (ignore_stim, resp)], resp


def _condition_pairs_disjoint(ds_left, ds_right, samplingrate):
    "Dichotic-style: disjoint EEG (left vs right side), each with its own attended stimulus"
    stim_l, resp_l = _trial_arrays(ds_left, samplingrate, 'fg')
    stim_r, resp_r = _trial_arrays(ds_right, samplingrate, 'fg')
    full_resp = numpy.vstack([resp_l, resp_r])
    return [(stim_l, resp_l), (stim_r, resp_r)], full_resp


def _sigevd_cache_path(subject, no_of_comps, lam):
    return SIGEVD_CACHE_DIR / f'sub-{subject}_comps-{no_of_comps}_lam-{lam}_sigevd-filter.npz'


def _fit_sigevd_filter(subject, no_of_comps, lam):
    """Fit one subject's SI-GEVD filter, pooling every condition with real
    two-talker/single-talker EEG (clean, diotic, binaural, dichotic) into
    one shared filter.

    Loads epochs at raw='ica' explicitly, not the ambient raw state (which
    may currently be resolving 'ica-sigevd' itself, if this is running
    inside RawSIGEVD._make()) - fitting always reads the *undenoised*
    cleaned EEG; there would be nothing left to fit a filter against
    otherwise.
    """
    kwargs = dict(subject=subject, raw='ica', samplingrate=SIGEVD_SAMPLINGRATE, ndvar='eeg', baseline=False)
    ds_clean = e.load_epochs(epoch='clean', **kwargs)
    ds_diotic = e.load_epochs(epoch='diotic', **kwargs)
    ds_binaural = e.load_epochs(epoch='binaural', **kwargs)
    ds_dichotic_left = e.load_epochs(epoch='dichotic-left', **kwargs)
    ds_dichotic_right = e.load_epochs(epoch='dichotic-right', **kwargs)

    pairs_clean, resp_clean = _condition_pairs_single(ds_clean, SIGEVD_SAMPLINGRATE)
    pairs_diotic, resp_diotic = _condition_pairs_shared(ds_diotic, SIGEVD_SAMPLINGRATE)
    pairs_binaural, resp_binaural = _condition_pairs_shared(ds_binaural, SIGEVD_SAMPLINGRATE)
    pairs_dichotic, resp_dichotic = _condition_pairs_disjoint(ds_dichotic_left, ds_dichotic_right, SIGEVD_SAMPLINGRATE)

    condition_pairs = pairs_clean + pairs_diotic + pairs_binaural + pairs_dichotic
    full_resp = numpy.vstack([resp_clean, resp_diotic, resp_binaural, resp_dichotic])

    ch_names = list(_eeg_ndvar(ds_clean)[0].get_dim('sensor').names)
    _, w, _ = find_sigevd_filter(
        condition_pairs, full_resp, SIGEVD_SAMPLINGRATE,
        SIGEVD_TSTART * 1e3, SIGEVD_TSTOP * 1e3, no_of_comps, lam,
    )
    return w, ch_names


def _load_or_fit_sigevd_filter(subject, no_of_comps, lam):
    "Cached per-subject SI-GEVD filter - fit once, reused after that. Delete the cache file to force a refit."
    path = _sigevd_cache_path(subject, no_of_comps, lam)
    if path.exists():
        cached = numpy.load(path, allow_pickle=False)
        return cached['w'], list(cached['ch_names'])
    w, ch_names = _fit_sigevd_filter(subject, no_of_comps, lam)
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy.savez(path, w=w, ch_names=numpy.array(ch_names))
    return w, ch_names


class RawSIGEVD(CachedRawPipe):
    """Applies a pre-fit, per-subject SI-GEVD spatial denoising filter to
    the continuous EEG - structurally the same kind of stage as
    RawFilter/RawReReference (a fixed linear transform of the continuous
    recording), except the transform itself is fit from this subject's own
    stimulus-response relationship (see _fit_sigevd_filter) rather than a
    universal formula.
    """
    DICT_ATTRS = CachedRawPipe.DICT_ATTRS + ('no_of_comps', 'lam')

    def __init__(self, source, no_of_comps, lam, cache=True):
        CachedRawPipe.__init__(self, source, cache)
        self.no_of_comps = no_of_comps
        self.lam = lam

    def _make(self, raw, *, path, noise=False, raw_name=None, log=None, source_pipe=None):
        w, ch_names = _load_or_fit_sigevd_filter(path.subject, self.no_of_comps, self.lam)
        picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude=())
        raw_ch_names = [raw.ch_names[i] for i in picks]
        assert raw_ch_names == ch_names, (
            f"SI-GEVD filter channel order does not match this raw file's EEG "
            f"channels for subject {path.subject!r} - delete its cache file "
            f"under {SIGEVD_CACHE_DIR} and refit."
        )
        raw._data[picks] = apply_sigevd_filter(raw._data[picks].T, w).T
        return raw

    def _make_info(self, info, *, path, noise=False, raw_name=None, log=None):
        return None


class BinauralCocktailSIGEVD(BinauralCocktail):
    raw = {**BinauralCocktail.raw, 'ica-sigevd': RawSIGEVD('ica', SIGEVD_NO_OF_COMPS, SIGEVD_LAM, cache=True)}


e = BinauralCocktailSIGEVD(DATA_ROOT)
