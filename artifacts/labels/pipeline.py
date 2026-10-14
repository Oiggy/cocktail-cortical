"""The labelling pipeline, step by step, as called from 01_labels.ipynb.

Each function does one notebook step over a list of subjects, writes its
files, and returns a table for the notebook to show.
"""
import json
from dataclasses import asdict

import pandas as pd

from .. import config
from ..paths import deriv_path, group_path
from ..preprocessing.preprocess import load_preprocessed
from . import emg, ocular
from .review import make_review_sheet, summarize_reviews
from .store import KINDS, label_summary, load_labels, review_dir, save_labels

INFO_COLUMNS = ['line_freq', 'line_freq_source', 'reference', 'interpolated', 'eog_bads', 'truncated_segments']


def label_subject(subject, rebuild=False, ocular_params=None, emg_params=None):
    """Preprocess (if not cached) and auto-label one subject. Writes the
    ocular and EMG event tables plus a JSON of the parameters used, and
    returns (summary per condition, preprocessing info)."""
    raw, segments, info = load_preprocessed(subject, rebuild=rebuild)
    ocular_params = ocular_params or ocular.OcularParams()
    emg_params = emg_params or emg.EMGParams()
    emg_params = emg.EMGParams(**{**asdict(emg_params), 'line_freq': info['line_freq']})

    ocular_events, meta = ocular.label_recording(raw, segments, ocular_params)
    save_labels(subject, 'ocular', ocular_events)
    deriv_path(subject, 'ocular', 'events', 'json').write_text(json.dumps(
        {'params': asdict(ocular_params), **meta, 'eog_bads': info['eog_bads']}, indent=2))

    emg_events = emg.label_recording(raw, segments, emg_params)
    save_labels(subject, 'emg', emg_events)
    deriv_path(subject, 'emg', 'events', 'json').write_text(json.dumps({'params': asdict(emg_params)}, indent=2))

    events = pd.concat([ocular_events, emg_events], ignore_index=True)
    summary = label_summary(events, segments, raw.info['sfreq'])
    summary.insert(0, 'subject', subject)
    return summary, info


def label_subjects(subjects, rebuild=False, ocular_params=None, emg_params=None, verbose=True):
    """label_subject() for each subject. Returns (label summary, one row of
    preprocessing choices per subject); the summary is also written to
    group/label_summary.tsv."""
    summaries, infos = [], []
    for subject in subjects:
        if verbose:
            print(f'sub-{subject} ...', flush=True)
        summary, info = label_subject(subject, rebuild, ocular_params, emg_params)
        summaries.append(summary)
        infos.append({'subject': subject, **{k: info.get(k) for k in INFO_COLUMNS}})
        if info['eog_bads'] and verbose:
            print(f'  WARNING: bad EOG electrodes {info["eog_bads"]} - ocular labels from them are unreliable')
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(group_path('label_summary.tsv'), sep='\t', index=False)
    return summary, pd.DataFrame(infos)


def write_review_sheets(subjects, kinds=KINDS, n_per_type=20, n_unlabelled=20, seed=0, overwrite=False):
    """Review sheets for each subject and label kind. A sheet that already
    has verdicts is skipped unless overwrite=True - that's hand-made work.
    Returns one row per sheet: where it is and what happened."""
    rows = []
    for subject in subjects:
        raw, segments, _ = load_preprocessed(subject)
        for kind in kinds:
            out_dir = review_dir(kind, subject)
            sheet_path = out_dir / 'review.csv'
            if sheet_path.exists() and not overwrite:
                existing = pd.read_csv(sheet_path, keep_default_na=False)
                if (existing['verdict'].astype(str).str.strip() != '').any():
                    rows.append({'subject': subject, 'kind': kind, 'items': len(existing),
                                 'status': 'kept (has verdicts)', 'sheet': str(sheet_path)})
                    continue
            events = load_labels(subject, kind, reviewed=False)
            sheet = make_review_sheet(raw, events, segments, subject, out_dir, n_per_type, n_unlabelled, seed=seed)
            rows.append({'subject': subject, 'kind': kind, 'items': len(sheet), 'status': 'written',
                         'sheet': str(sheet_path)})
    return pd.DataFrame(rows)


def review_summaries(kinds=KINDS):
    """Precision per type and miss rate, from every filled-in sheet of each
    kind. Returns {kind: table}; each is also written to
    group/review_summary_<kind>.tsv. Kinds without verdicts yet are left out."""
    out = {}
    for kind in kinds:
        paths = sorted((config.DERIV_ROOT / 'review' / kind).glob('sub-*/review.csv'))
        sheets = [pd.read_csv(p, keep_default_na=False) for p in paths]
        if not sheets:
            continue
        summary = summarize_reviews(sheets)
        if summary.empty:
            continue
        summary.insert(0, 'subjects', len(sheets))
        summary.to_csv(group_path(f'review_summary_{kind}.tsv'), sep='\t', index=False)
        out[kind] = summary
    return out
