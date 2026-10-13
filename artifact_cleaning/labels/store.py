"""Saving/loading labels, and summarizing them per condition.

Labels are BIDS-style events TSVs (onset, duration, trial_type, ...),
onsets in the preprocessed recording's time, one file per kind:
  sub-XX_task-cocktail_desc-ocular_events.tsv
  sub-XX_task-cocktail_desc-emg_events.tsv
Review sheets live under <DERIV_ROOT>/review/<kind>/sub-XX/review.csv;
load_labels() applies a subject's sheet automatically when one exists.
"""
import numpy as np
import pandas as pd

from .. import config
from ..paths import deriv_path
from .review import apply_review

KINDS = ('ocular', 'emg')


def review_dir(kind, subject, root=None):
    return (root or config.DERIV_ROOT) / 'review' / kind / f'sub-{subject}'


def save_labels(subject, kind, events):
    path = deriv_path(subject, kind, 'events', 'tsv')
    events.to_csv(path, sep='\t', index=False, na_rep='n/a')
    return path


def load_labels(subject, kind, reviewed=True):
    events = pd.read_csv(deriv_path(subject, kind, 'events', 'tsv', mkdir=False), sep='\t', na_values='n/a',
                         keep_default_na=False)
    sheet_path = review_dir(kind, subject) / 'review.csv'
    if reviewed and sheet_path.exists():
        events = apply_review(events, pd.read_csv(sheet_path, keep_default_na=False), subject)
    return events


def coverage_mask(events, n_times, sfreq):
    """Boolean per-sample mask of the time any event covers."""
    mask = np.zeros(n_times, bool)
    for onset, duration in zip(events['onset'], events['duration']):
        mask[int(onset * sfreq):int(np.ceil((onset + duration) * sfreq))] = True
    return mask


def label_summary(events, segments_df, sfreq):
    """Per condition: minutes of data, events per minute and % of time
    covered for each trial_type, and % covered by any label."""
    rows = []
    for condition, segs in segments_df.groupby('condition'):
        minutes = segs['duration'].sum() / 60
        row = {'condition': condition, 'minutes': round(minutes, 2)}
        n_total = int(segs['stop'].max())
        any_mask = np.zeros(n_total, bool)
        in_condition = np.zeros(n_total, bool)
        for seg in segs.itertuples():
            in_condition[seg.start:seg.stop] = True
        for trial_type, group in events[events['condition'] == condition].groupby('trial_type'):
            mask = coverage_mask(group, n_total, sfreq) & in_condition
            any_mask |= mask
            row[f'{trial_type}_per_min'] = round(len(group) / minutes, 2)
            row[f'{trial_type}_pct_time'] = round(100 * mask.sum() / in_condition.sum(), 2)
        row['any_pct_time'] = round(100 * any_mask.sum() / in_condition.sum(), 2)
        rows.append(row)
    table = pd.DataFrame(rows)
    rates = [c for c in table.columns if c.endswith(('_per_min', '_pct_time'))]
    table[rates] = table[rates].fillna(0.)
    return table
