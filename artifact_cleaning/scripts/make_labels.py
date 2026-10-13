"""Preprocess (if not cached yet) and auto-label ocular and EMG artifacts.

    python -m artifact_cleaning.scripts.make_labels --subjects all
    python -m artifact_cleaning.scripts.make_labels --subjects 01 02 --rebuild

Writes per subject (under <bids>/derivatives/artifact_cleaning/sub-XX/eeg/):
  ..._desc-preproc_eeg.fif / _segments.tsv / _eeg.json   (preprocessed data)
  ..._desc-ocular_events.tsv / .json                      (blinks, saccades)
  ..._desc-emg_events.tsv / .json                         (weak EMG labels)
and group/label_summary.tsv: events per minute and % of time labelled,
per subject and condition.
"""
import argparse
import json
from dataclasses import asdict

import pandas as pd

from ..labels import emg, ocular
from ..labels.store import label_summary, save_labels
from ..paths import deriv_path, group_path
from ..preprocessing.loading import resolve_subjects
from ..preprocessing.preprocess import load_preprocessed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--subjects', nargs='+', default=['all'])
    parser.add_argument('--rebuild', action='store_true', help='redo preprocessing even if cached')
    parser.add_argument('--emg-z', type=float, default=emg.EMGParams.z_threshold)
    args = parser.parse_args(argv)

    summaries = []
    for subject in resolve_subjects(args.subjects):
        print(f'sub-{subject}: preprocessing')
        raw, segments, info = load_preprocessed(subject, rebuild=args.rebuild)
        if info['eog_bads']:
            print(f'  WARNING: bad EOG electrodes {info["eog_bads"]} - ocular labels from them are unreliable')

        ocular_params = ocular.OcularParams()
        ocular_events, meta = ocular.label_recording(raw, segments, ocular_params)
        save_labels(subject, 'ocular', ocular_events)
        deriv_path(subject, 'ocular', 'events', 'json').write_text(json.dumps(
            {'params': asdict(ocular_params), **meta, 'eog_bads': info['eog_bads']}, indent=2))

        emg_params = emg.EMGParams(z_threshold=args.emg_z, line_freq=info['line_freq'])
        emg_events = emg.label_recording(raw, segments, emg_params)
        save_labels(subject, 'emg', emg_events)
        deriv_path(subject, 'emg', 'events', 'json').write_text(json.dumps({'params': asdict(emg_params)}, indent=2))

        summary = label_summary(pd.concat([ocular_events, emg_events], ignore_index=True), segments, raw.info['sfreq'])
        summary.insert(0, 'subject', subject)
        summaries.append(summary)
        counts = pd.concat([ocular_events, emg_events])['trial_type'].value_counts().to_dict()
        print(f'  {counts}; line noise {info["line_freq"]} Hz; reference {info["reference"]}; '
              f'interpolated {info["interpolated"] or "none"}')

    if summaries:
        table = pd.concat(summaries, ignore_index=True)
        out = group_path('label_summary.tsv')
        table.to_csv(out, sep='\t', index=False)
        print(f'\nsummary written to {out}')
        print(table.to_string(index=False))


if __name__ == '__main__':
    main()
