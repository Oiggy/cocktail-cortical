"""
Convert the raw BioSemi recordings (.bdf files) into the BIDS folder format.

Why: eelbrain's pipeline (see analysis/experiment.py) expects the EEG
recordings to sit in a predictable, standardized folder layout. BIDS is
that standard layout. Running this script once, before anything else,
turns the "eeg/s1/s1_cocktail.bdf, eeg/s2/s2_cocktail.bdf, ..." files
into a "bids/sub-01/eeg/..., bids/sub-02/eeg/..." structure that the
rest of the pipeline (and any other BIDS-aware tool) can read.

Run this first, before predictors/gammatone.py or analysis/experiment.py.

This only converts the raw recordings. Bad-channel marking and ICA
fitting happen afterwards, interactively, through eelbrain itself (see
the note at the bottom of this file) - they are not part of this script.
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

    raw = mne.io.read_raw_bdf(raw_file, preload=False)
    raw.set_channel_types(CHANNEL_TYPES, on_unit_change="ignore")

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
    )

    print(f"Wrote {subject_name} to {bids_path.directory}")

# Note: BDF is not one of the formats mne-bids can write as-is, so it
# converts the recording to BrainVision format (.vhdr/.eeg/.vmrk) during
# the write above. After running this once, check the file extension
# actually created under BIDS_ROOT/sub-XX/eeg/ and make sure it matches
# the pattern used in analysis/experiment.py's RawSource (it expects
# ``.vhdr``). If your mne-bids version converts to something else,
# update that pattern to match.
#
# Bad channels and ICA: after this conversion, `experiment.py`'s raw
# `sub-XX_..._eeg.vhdr` files have no bad-channel or ICA information yet
# (those are per-subject files eelbrain creates the first time you ask
# it to preprocess a subject - see the "Preprocessing" section of the
# README for the exact commands).
