"""
Convert the raw BioSemi recordings (.bdf files) into the BIDS folder format.

Why: eelbrain's pipeline (see analysis/experiment.py) expects the EEG
recordings to sit in a predictable, standardized folder layout. BIDS is
that standard layout. Running this script once, before anything else,
turns the messy "raw_data/S3.bdf, S4.bdf, ..." files into a clean
"bids/sub-03/eeg/..., bids/sub-04/eeg/..." structure that the rest of the
pipeline (and any other BIDS-aware tool) can read.

Run this first, before predictors/gammatone.py or analysis/experiment.py.
"""
from pathlib import Path

import mne
from mne_bids import BIDSPath, write_raw_bids

# Where the original .bdf files live, and where the BIDS copy should go.
RAW_DATA_DIR = Path("~/Data/BinauralCocktail/raw_data").expanduser()
BIDS_ROOT = Path("~/Data/BinauralCocktail/bids").expanduser()

TASK_NAME = "cocktail"

# Subjects 1 and 2 were pilot recordings and are excluded, matching the
# rest of the scripts in this repo (they only ever refer to subjects 3-14).
SUBJECT_NUMBERS = range(3, 15)

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

for subject_num in SUBJECT_NUMBERS:
    raw_file = RAW_DATA_DIR / f"S{subject_num}.bdf"

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

    print(f"Wrote subject {subject_num} to {bids_path.directory}")

# Note: BDF is not one of the formats mne-bids can write as-is, so it
# converts the recording to BrainVision format (.vhdr/.eeg/.vmrk) during
# the write above. After running this once, check the file extension
# actually created under BIDS_ROOT/sub-XX/eeg/ and make sure it matches
# the pattern used in analysis/experiment.py's RawSource (it expects
# ``.vhdr``). If your mne-bids version converts to something else,
# update that pattern to match.
