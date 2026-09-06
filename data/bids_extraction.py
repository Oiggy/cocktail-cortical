"""
Convert the raw BioSemi recordings (.bdf files) into the BIDS folder format.

Why: eelbrain's pipeline (see analysis/experiment.py) expects the EEG
recordings to sit in a predictable, standardized folder layout. BIDS is
that standard layout. Running this script once, before anything else,
turns the "eeg/s1/s1_cocktail.bdf, eeg/s2/s2_cocktail.bdf, ..." files
into a "bids/sub-01/eeg/..., bids/sub-02/eeg/..." structure that the
rest of the pipeline (and any other BIDS-aware tool) can read.

Run this first, before predictors/gammatone_predictors.py or analysis/experiment.py.

This only converts the raw recordings. Bad-channel marking and ICA
fitting happen afterwards, interactively, through eelbrain itself (see
the note at the bottom of this file) - they are not part of this script.

Channels are renamed here, before writing BIDS (see CH_MAP below), not
later in analysis/experiment.py. eelbrain's interactive bad-channel
GUI cross-checks the recording's channel names against the
`channels.tsv` file BIDS writes here, and that check breaks if the two
don't already match - so the names written to `channels.tsv` have to
be the final, renamed ones from the start.
"""
import re
from pathlib import Path

import mne
from mne_bids import BIDSPath, write_raw_bids

# The dataset lives outside this repo, in a sibling "dataset/cocktail"
# folder next to it (not inside it, since data shouldn't go into git).
# Working this out from this file's own location means the scripts run
# the same way no matter whose computer, or which folder, the repo is
# cloned into.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT.parent / "dataset" / "cocktail"
RAW_DATA_DIR = DATA_ROOT / "eeg"
BIDS_ROOT = DATA_ROOT / "bids"

TASK_NAME = "cocktail"

# Find every subject folder (s1, s2, s3, ...) instead of hard-coding a
# subject count, so this keeps working if subjects are added or removed.
SUBJECT_FOLDERS = sorted(
    (p for p in RAW_DATA_DIR.iterdir() if p.is_dir() and re.fullmatch(r"s\d+", p.name)),
    key=lambda p: int(p.name[1:]),
)

# The BioSemi cap has extra channels beyond the 64 EEG electrodes: four
# electrodes around the eyes (used to detect blinks/eye movements) and a
# handful of "misc" channels (skin conductance, respiration, etc.) that
# aren't brain signal at all. Telling MNE what each channel actually is
# means later steps (like ICA for blink removal) know which channels to
# use for what.
CHANNEL_TYPES = {
    "EXG1": "eog",
    "EXG2": "eog",
    "EXG3": "eog",
    "EXG4": "eog",
    "EXG7": "misc",
    "EXG8": "misc",
    "GSR1": "misc",
    "GSR2": "misc",
    "Erg1": "misc",
    "Erg2": "misc",
    "Resp": "misc",
    "Plet": "misc",
    "Temp": "misc",
}

# The BioSemi cap numbers electrodes A1-A32, B1-B32. This maps those
# numbers to standard 10-20 electrode names, so plots and montages use
# names other EEG software recognizes. Must match analysis/experiment.py's
# copy of this same mapping (kept in both files rather than shared,
# since each script here is meant to be read on its own).
CH_MAP = {
    'A1': 'Fp1', 'A2': 'AF7', 'A3': 'AF3', 'A4': 'F1', 'A5': 'F3', 'A6': 'F5',
    'A7': 'F7', 'A8': 'FT7', 'A9': 'FC5', 'A10': 'FC3', 'A11': 'FC1',
    'A12': 'C1', 'A13': 'C3', 'A14': 'C5', 'A15': 'T7', 'A16': 'TP7',
    'A17': 'CP5', 'A18': 'CP3', 'A19': 'CP1', 'A20': 'P1', 'A21': 'P3',
    'A22': 'P5', 'A23': 'P7', 'A24': 'P9', 'A25': 'PO7', 'A26': 'PO3',
    'A27': 'O1', 'A28': 'Iz', 'A29': 'Oz', 'A30': 'POz', 'A31': 'Pz',
    'A32': 'CPz',
    'B1': 'Fpz', 'B2': 'Fp2', 'B3': 'AF8', 'B4': 'AF4', 'B5': 'AFz',
    'B6': 'Fz', 'B7': 'F2', 'B8': 'F4', 'B9': 'F6', 'B10': 'F8',
    'B11': 'FT8', 'B12': 'FC6', 'B13': 'FC4', 'B14': 'FC2', 'B15': 'FCz',
    'B16': 'Cz', 'B17': 'C2', 'B18': 'C4', 'B19': 'C6', 'B20': 'T8',
    'B21': 'TP8', 'B22': 'CP6', 'B23': 'CP4', 'B24': 'CP2', 'B25': 'P2',
    'B26': 'P4', 'B27': 'P6', 'B28': 'P8', 'B29': 'P10', 'B30': 'PO8',
    'B31': 'PO4', 'B32': 'O2',
    # Extra channels: four around the eyes (for detecting blinks), two
    # mastoid channels used as the reference.
    'EXG1': 'EOG-LS', 'EXG2': 'EOG-LI', 'EXG3': 'EOG-L', 'EXG4': 'EOG-R',
    'EXG5': 'A1', 'EXG6': 'A2',
}

# The trigger channel marks the moment each audio stimulus started, using
# small integer codes. 65536 is a spurious value the BioSemi amplifier
# sometimes emits and isn't a real trigger, so it gets filtered out.
SPURIOUS_TRIGGER_VALUE = 65536
EVENT_ID = {
    "code_1": 1,
    "code_2": 2,
    "code_8": 8,
}

for subject_dir in SUBJECT_FOLDERS:
    subject_name = subject_dir.name  # e.g. "s13"
    subject_num = int(subject_name[1:])
    raw_file = subject_dir / f"{subject_name}_cocktail.bdf"

    # preload=True: needed below so write_raw_bids can actually re-export
    # the recording (with the renamed channels) rather than just copying
    # the original file's bytes unchanged.
    raw = mne.io.read_raw_bdf(raw_file, preload=True)
    raw.set_channel_types(CHANNEL_TYPES, on_unit_change="ignore")
    raw.rename_channels(CH_MAP)

    events = mne.find_events(raw, stim_channel="Status")
    events = events[events[:, 2] != SPURIOUS_TRIGGER_VALUE]

    bids_path = BIDSPath(
        subject=f"{subject_num:02d}",
        task=TASK_NAME,
        datatype="eeg",
        suffix="eeg",
        root=BIDS_ROOT,
    )

    write_raw_bids(
        raw,
        bids_path,
        events=events,
        event_id=EVENT_ID,
        overwrite=True,
        # format="BDF" (rather than the default "auto") forces mne-bids
        # to actually re-export the recording using this renamed `raw`
        # object. Without it, since the input is already .bdf and BIDS
        # allows keeping that format, mne-bids takes a shortcut: it
        # copies the *original* .bdf file's bytes as-is and only writes
        # the renamed names into channels.tsv - leaving the real
        # recording still carrying the original A1/A2/.../EXG1/...
        # channel names, silently disagreeing with channels.tsv.
        format="BDF",
    )

    print(f"Wrote {subject_name} to {bids_path.directory}")

# Note: the output stays in .bdf format here (format="BDF" above just
# forces mne-bids to re-export it - see the comment there - not switch
# to a different file type), so the output is
# BIDS_ROOT/sub-XX/eeg/sub-XX_task-cocktail_eeg.bdf. This matches the
# pattern used in analysis/experiment.py's RawSource.
#
# Bad channels and ICA: after this conversion, those `.bdf` files have
# no bad-channel or ICA information yet (those are per-subject files
# eelbrain creates the first time you ask it to preprocess a subject -
# see the "Preprocessing" section of the README for the exact commands).
