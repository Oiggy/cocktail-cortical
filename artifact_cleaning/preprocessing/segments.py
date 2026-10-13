"""The 12 speech segments in each recording, and which condition each is.

Mirrors label_events() in analysis/experiment.py. The constants are copied
rather than imported, because importing experiment.py pulls in eelbrain's
in-development Pipeline, which this project doesn't otherwise need.
tests/test_segments.py reads experiment.py's own copies and fails if the
two ever drift apart.
"""
from dataclasses import dataclass

import pandas as pd

# --- Copied from analysis/experiment.py (kept in sync by test_segments.py) ---
SEGMENT_DURATION = {
    'male_1': 247.521, 'male_2': 246.358, 'male_3': 245.919, 'male_4': 235.938,
    'male_5': 240.617, 'male_6': 243.213, 'male_7': 242.723, 'male_8': 242.894,
    'male_9': 248.464, 'male_10': 241.766, 'male_11': 251.065, 'male_12': 247.342,
    'female_1': 257.150, 'female_2': 251.807, 'female_3': 240.025, 'female_4': 247.964,
    'female_5': 250.498, 'female_6': 245.643, 'female_7': 246.122, 'female_8': 242.718,
    'female_9': 252.099, 'female_10': 239.235, 'female_11': 251.402, 'female_12': 245.898,
}
SPEAKER = [
    ['male'] * 6 + ['female'] * 6,
    ['female'] * 6 + ['male'] * 6,
]
SPATIAL = [
    'clean', 'binaural', 'dichotic', 'diotic', 'binaural', 'dichotic',
] * 2
SIDE = ((
    '', 'right', 'left', '', 'left', 'right',
    '', 'left', 'right', '', 'right', 'left',
), (
    '', 'left', 'right', '', 'right', 'left',
    '', 'right', 'left', '', 'left', 'right',
))
# ---------------------------------------------------------------------------

ONSET_CODES = (1, 2)  # trigger values marking a story onset; the value is the stimulus list
N_SEGMENTS = len(SPATIAL)


@dataclass(frozen=True)
class Segment:
    index: int        # 1-12, story order in the session
    onset: float      # s, in the original recording's time
    duration: float   # s
    condition: str    # clean / binaural / dichotic / diotic
    side: str         # dichotic only: ear the background story came from
    fg: str           # attended (foreground) stimulus
    bg: str           # ignored (background) stimulus, '' for clean

    @property
    def offset(self):
        return self.onset + self.duration


def segments_from_events(onsets, values):
    """Build the 12 Segments from trigger onsets (s) and trigger values.

    Only values in ONSET_CODES count, exactly as label_events() filters
    with value.isin((1, 2)); the first one's value is the stimulus list.
    """
    story = [(t, v) for t, v in zip(onsets, values) if v in ONSET_CODES]
    if len(story) != N_SEGMENTS:
        raise ValueError(f"expected {N_SEGMENTS} story onsets (trigger values {ONSET_CODES}), found {len(story)}")
    list_id = story[0][1]
    fg_speakers = SPEAKER[list_id - 1]
    bg_speakers = SPEAKER[-list_id + 2]
    segments = []
    for i, (onset, _) in enumerate(story, start=1):
        fg = f'{fg_speakers[i - 1]}_{i}'
        bg = '' if i in (1, 7) else f'{bg_speakers[i - 1]}_{i}'
        segments.append(Segment(
            index=i, onset=float(onset), duration=SEGMENT_DURATION[fg],
            condition=SPATIAL[i - 1], side=SIDE[list_id - 1][i - 1], fg=fg, bg=bg,
        ))
    return segments


def segments_from_annotations(annotations):
    """Segments from an MNE Raw's annotations, as written by mne-bids from
    data/bids_extraction.py's EVENT_ID ('code_1', 'code_2', 'code_8')."""
    onsets, values = [], []
    for onset, description in zip(annotations.onset, annotations.description):
        if description.startswith('code_'):
            onsets.append(onset)
            values.append(int(description.split('_', 1)[1]))
    return segments_from_events(onsets, values)


def segments_table(segments):
    return pd.DataFrame([{**s.__dict__, 'offset': s.offset} for s in segments])
