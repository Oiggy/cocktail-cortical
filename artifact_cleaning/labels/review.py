"""Hand spot-checking of automatic labels.

make_review_sheet() draws a random sample of labelled events per type, plus
random *unlabelled* windows, and writes one PNG per item and a CSV with an
empty `verdict` column to fill in:
  - labelled events: 'ok', 'wrong' or 'unsure' (optionally corrected
    onset/duration in s, preprocessed-recording time)
  - unlabelled windows (trial_type 'none'): 'clean', or 'missed:<type>'
    (e.g. 'missed:blink') if an artifact was there but not labelled
summarize_reviews() turns filled-in sheets into precision per type (from
the labelled sample) and a miss rate (from the unlabelled sample) - the
two numbers to report for the auto-labeller. apply_review() drops events
marked wrong and applies corrections, so reviewed labels feed training.
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PLOT_CHANNELS = ('VEOG', 'HEOG', 'Fp1', 'Fpz', 'Fp2', 'F7', 'F8', 'T7', 'T8', 'TP7', 'TP8')
REVIEW_COLUMNS = ['id', 'subject', 'trial_type', 'onset', 'duration', 'segment', 'condition', 'channel',
                  'amplitude', 'source', 'png', 'verdict', 'corrected_onset', 'corrected_duration', 'notes']


def event_ids(subject, events):
    """Stable ids: identical labelling parameters give identical ids, so a
    filled-in sheet still matches after re-running the labeller."""
    return [f'{subject}-{t}-{int(round(o * 1000))}' for t, o in zip(events['trial_type'], events['onset'])]


def _segment_of(time, segments_df):
    inside = segments_df[(segments_df['onset'] <= time) & (time < segments_df['onset'] + segments_df['duration'])]
    return None if inside.empty else inside.iloc[0]


def _unlabelled_windows(events, segments_df, n, duration, rng, margin=1.):
    """Random windows, wholly inside one segment, at least `margin` s from any event."""
    spans = np.column_stack([events['onset'] - margin, events['onset'] + events['duration'] + margin]) if len(events) else np.empty((0, 2))
    picked, tries = [], 0
    weights = segments_df['duration'] / segments_df['duration'].sum()
    while len(picked) < n and tries < 100 * n:
        tries += 1
        seg = segments_df.iloc[rng.choice(len(segments_df), p=weights)]
        if seg['duration'] <= duration:
            continue
        start = seg['onset'] + rng.uniform(0, seg['duration'] - duration)
        stop = start + duration
        if np.any((spans[:, 0] < stop) & (spans[:, 1] > start)):
            continue
        picked.append((start, seg))
    return picked


def _plot_item(raw, row, segment, out_png, context):
    sfreq = raw.info['sfreq']
    t0 = max(segment['onset'], row['onset'] - context)
    t1 = min(segment['onset'] + segment['duration'], row['onset'] + row['duration'] + context)
    channels = [ch for ch in PLOT_CHANNELS if ch in raw.ch_names]
    if isinstance(row.get('channel'), str) and row['channel'] in raw.ch_names and row['channel'] not in channels:
        channels.append(row['channel'])
    start, stop = int(round(t0 * sfreq)), int(round(t1 * sfreq))
    data = raw.get_data(picks=channels, start=start, stop=stop) * 1e6
    times = np.arange(start, start + data.shape[1]) / sfreq

    spacing = max(np.percentile(np.ptp(data, axis=1), 75), 1.)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(channels) + 1.2))
    for i, (ch, trace) in enumerate(zip(channels, data)):
        ax.plot(times, trace - trace.mean() - i * spacing, lw=0.7, color='crimson' if 'EOG' in ch else 'k')
    ax.set_yticks([-i * spacing for i in range(len(channels))])
    ax.set_yticklabels(channels)
    ax.axvspan(row['onset'], row['onset'] + row['duration'], color='gold', alpha=0.35)
    ax.set_xlabel('time (s, preprocessed recording)')
    amp = row.get('amplitude')
    if amp is None or pd.isna(amp):
        amp_text = ''
    elif row['trial_type'] == 'emg':
        amp_text = f'  z={amp:.1f}'  # EMG 'amplitude' is the high-frequency power z-score
    else:
        amp_text = f'  amplitude={amp * 1e6:.0f} uV'
    ax.set_title(f"{row['id']}  [{row['trial_type']}, segment {segment['index']} {segment['condition']}]{amp_text}"
                 f"   (scale bar: {spacing:.0f} uV between traces)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def make_review_sheet(raw, events, segments_df, subject, out_dir, n_per_type=20, n_unlabelled=20,
                      unlabelled_duration=1., context=2., seed=0):
    """Write review PNGs and `review.csv` to out_dir; return the sheet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    events = events.copy()
    events['id'] = event_ids(subject, events)

    items = []
    for trial_type, group in events.groupby('trial_type'):
        pick = rng.choice(len(group), size=min(n_per_type, len(group)), replace=False)
        items.extend(group.iloc[sorted(pick)].to_dict('records'))
    for start, seg in _unlabelled_windows(events, segments_df, n_unlabelled, unlabelled_duration, rng):
        items.append({'id': f'{subject}-none-{int(round(start * 1000))}', 'trial_type': 'none', 'onset': start,
                      'duration': unlabelled_duration, 'segment': seg['index'], 'condition': seg['condition'],
                      'channel': '', 'amplitude': np.nan, 'source': 'random'})

    rows = []
    for item in items:
        segment = _segment_of(item['onset'], segments_df)
        png = out_dir / f"{item['id']}.png"
        _plot_item(raw, item, segment, png, context)
        rows.append({**{c: item.get(c, '') for c in REVIEW_COLUMNS}, 'subject': subject, 'png': png.name,
                     'verdict': '', 'corrected_onset': '', 'corrected_duration': '', 'notes': ''})
    sheet = pd.DataFrame(rows, columns=REVIEW_COLUMNS)
    sheet.to_csv(out_dir / 'review.csv', index=False)
    return sheet


def summarize_reviews(sheets):
    """Precision per labelled type and the miss rate from unlabelled windows,
    over one or more filled-in sheets (DataFrames)."""
    sheet = pd.concat(sheets, ignore_index=True)
    sheet['verdict'] = sheet['verdict'].fillna('').astype(str).str.strip().str.lower()
    done = sheet[sheet['verdict'] != '']
    rows = []
    for trial_type, group in done[done['trial_type'] != 'none'].groupby('trial_type'):
        ok, wrong = (group['verdict'] == 'ok').sum(), (group['verdict'] == 'wrong').sum()
        rows.append({'trial_type': trial_type, 'reviewed': len(group), 'ok': ok, 'wrong': wrong,
                     'unsure': len(group) - ok - wrong,
                     'precision': ok / (ok + wrong) if ok + wrong else np.nan, 'miss_rate': np.nan})
    unlabelled = done[done['trial_type'] == 'none']
    if len(unlabelled):
        missed = unlabelled['verdict'].str.startswith('missed')
        # For unlabelled windows, 'ok' counts clean windows and 'wrong'
        # counts windows hiding a missed artifact.
        rows.append({'trial_type': 'none', 'reviewed': len(unlabelled), 'ok': int((~missed).sum()),
                     'wrong': int(missed.sum()), 'unsure': 0, 'precision': np.nan,
                     'miss_rate': float(missed.mean())})
    return pd.DataFrame(rows)


def apply_review(events, sheet, subject):
    """Events with review verdicts applied: 'wrong' dropped, corrections
    applied, and a `reviewed` column ('' where not reviewed)."""
    events = events.copy()
    events['id'] = event_ids(subject, events)
    sheet = sheet[sheet['trial_type'] != 'none'].set_index('id')
    verdict = sheet['verdict'].fillna('').astype(str).str.strip().str.lower()
    events['reviewed'] = events['id'].map(verdict).fillna('')
    for column, target in (('corrected_onset', 'onset'), ('corrected_duration', 'duration')):
        corrected = pd.to_numeric(events['id'].map(sheet[column]), errors='coerce')
        events[target] = corrected.where(corrected.notna(), events[target])
    return events[events['reviewed'] != 'wrong'].reset_index(drop=True)
