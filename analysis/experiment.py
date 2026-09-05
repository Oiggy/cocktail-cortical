"""
What this file is for
----------------------
- This file does two things: it defines the setup/config for the whole
  project (how to find and clean the data), and it's also the one
  place you run to do the interactive part of cleaning it.
- It tells eelbrain everything it needs to know about this dataset once,
  in one place: where the recordings are, how to clean them up, what
  the trigger codes mean, which stimulus goes with which EEG segment,
  and so on.
- Every analysis script then just says "from experiment import e" and
  asks that object for what it needs (e.g. "give me the cleaned EEG
  for the diotic condition"), instead of repeating all of this setup
  itself. Importing this file never launches anything interactive.
- Running this file directly (`python analysis/experiment.py`) is how
  you do the one interactive step this project needs: marking bad
  channels and picking ICA artifact components, once per subject (see
  `preprocess_all_subjects()` below). Run this after
  data/bids_extraction.py and predictors/gammatone_predictors.py.

What runs, top to bottom, the moment this file is imported
------------------------------------------------------------
- Work out DATA_ROOT: the one folder everything else below is found in.
- Build SEGMENT_DURATION: a lookup table of how long (in seconds) each
  audio stimulus is, so eelbrain knows how much EEG to cut out for each
  one.
- Build CH_MAP: renames the EEG cap's numbered electrodes (A1, A2, ...)
  to their standard names (Fp1, AF7, ...) that other software recognizes.
- Load MONTAGE from biosemi64mod.txt: the physical 3D position of each
  electrode, needed to draw scalp maps.
- Build SPEAKER / SPATIAL / SIDE: lookup tables describing the
  experiment design itself, i.e. which speaker, which listening
  condition, and which ear applied to each of the 12 segments a
  subject heard.
- Define the BinauralCocktail class. Defining a class doesn't run
  anything by itself, it just registers the rules eelbrain will follow
  later, once a subject is actually requested. Those rules are:
    - raw: the cleanup steps applied to the EEG, in order (filtering,
      re-referencing, removing eye-blink artifacts with ICA). See the
      numbered comments on this dict for exactly where the two
      interactive steps below plug in.
    - preprocess_all_subjects(): loops over every subject and runs the
      two interactive steps `raw` depends on (mark bad channels, pick
      ICA artifact components). This is a method, not a script, so it
      only runs when you ask for it (see the bottom of this file).
    - groups: a named shortcut for referring to a subset of subjects.
    - fix_events / label_events: how to figure out, from the trigger
      codes in the recording, which condition and which stimulus each
      segment was.
    - epochs: which time windows of EEG to extract, and how to split
      them up by condition (clean / diotic / binaural / dichotic).
    - predictors: which generated predictor files (from
      predictors/gammatone_predictors.py) are available to use in the analysis.
- "e = BinauralCocktail(DATA_ROOT)" builds the one ready-to-use pipeline
  object, `e`, that every analysis script imports.
- Last line: `if __name__ == '__main__': e.preprocess_all_subjects()`.
  This only fires when this file is run directly, never on import - see
  "What this file is for" above.
"""
from eelbrain import Factor, Var
from eelbrain.pipeline import *
from trftools.pipeline import *
import mne
from pathlib import Path

# The dataset lives outside this repo, in a sibling "dataset/cocktail"
# folder next to it. Working this out from this file's own location
# means the pipeline works the same way no matter whose computer, or
# which folder, the repo is cloned into. It should contain:
#   bids/        the BIDS-formatted EEG recordings (data/bids_extraction.py)
#   stimuli/     the stimulus .wav files
#   predictors/  the generated predictor files (predictors/gammatone_predictors.py)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = str(REPO_ROOT.parent / "dataset" / "cocktail")

# Each audio stimulus has a different length. This tells the pipeline how
# long to make the EEG segment ("epoch") for each one.
SEGMENT_DURATION = {
    'male_1': 247.521,
    'male_2': 246.358,
    'male_3': 245.919,
    'male_4': 235.938,
    'male_5': 240.617,
    'male_6': 243.213,
    'male_7': 242.723,
    'male_8': 242.894,
    'male_9': 248.464,
    'male_10': 241.766,
    'male_11': 251.065,
    'male_12': 247.342,
    'female_1': 257.150,
    'female_2': 251.807,
    'female_3': 240.025,
    'female_4': 247.964,
    'female_5': 250.498,
    'female_6': 245.643,
    'female_7': 246.122,
    'female_8': 242.718,
    'female_9': 252.099,
    'female_10': 239.235,
    'female_11': 251.402,
    'female_12': 245.898,
}

# The BioSemi cap numbers electrodes A1-A32, B1-B32. This maps those
# numbers to standard 10-20 electrode names, so plots and montages use
# names other EEG software recognizes.
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

MONTAGE = mne.channels.read_custom_montage('biosemi64mod.txt')

# Which speaker (male/female) was the foreground story in each of the 12
# segments, for each of the two stimulus lists used in the experiment.
SPEAKER = [
    ['male'] * 6 + ['female'] * 6,
    ['female'] * 6 + ['male'] * 6,
]

# The spatial listening condition for each of the 12 segments:
#   clean    - only the foreground story plays, nothing to ignore
#   binaural - both stories play in both ears (no side difference)
#   dichotic - each story plays in one ear only
#   diotic   - both stories play identically in both ears
SPATIAL = [
    'clean', 'binaural', 'dichotic', 'diotic', 'binaural', 'dichotic',
] * 2

# For dichotic segments, which ear the background story came from.
SIDE = ((
    '', 'right', 'left', '', 'left', 'right',
    '', 'left', 'right', '', 'right', 'left',
), (
    '', 'left', 'right', '', 'right', 'left',
    '', 'right', 'left', '', 'left', 'right',
))


class BinauralCocktail(TRFExperiment):

    auto_delete_cache = 'ask'

    # Recordings live under <DATA_ROOT>/bids, in BIDS format.
    data_dir = 'bids'
    # BIDS subject folders are named "sub-03", "sub-04", etc.
    subject_re = r'sub-\d+'
    # BIDS task name used when the recordings were extracted
    # (see data/bids_extraction.py's TASK_NAME).
    sessions = ['cocktail']

    # The preprocessing steps applied to the raw EEG, each one building on
    # the previous. This dict only describes *how* to compute each stage -
    # two of the stages below (1 and 4) also need a human to look at a
    # plot and click, which happens by calling preprocess_all_subjects()
    # below, not by anything written in this dict itself. For full
    # details on the raw-processing pipeline see
    # https://eelbrain.readthedocs.io/en/stable/experiment.html
    raw = {
        # STEP 1 - load the recording. mne-bids keeps BioSemi recordings
        # in their original .bdf format rather than converting them, so
        # this reads the same way the original .bdf files did.
        #
        # Bad channels are excluded right here, automatically, the
        # moment this stage loads - not because this dict says so, but
        # because RawSource always checks for a per-subject
        # bad-channels file first. That file is created by running
        # `e.make_bad_channels()`, which the preprocess_all_subjects()
        # method below calls for you. Every stage below inherits
        # whatever gets excluded here.
        'raw': RawSource(
            'eeg/{subject}_task-{recording}_eeg.bdf',
            reader=mne.io.read_raw_bdf,
            eog=['EXG1', 'EXG2', 'EXG3', 'EXG4'],
            misc=['EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp'],
            adjacency='auto',
            rename_channels=CH_MAP,
            montage=MONTAGE,
        ),
        # STEP 2 - band-pass filter for the cortical analysis: cortical
        # responses to speech are slow, so a 0.5-20 Hz filter keeps the
        # signal of interest while removing slow drift and
        # high-frequency noise.
        '0.5-20': RawFilter('raw', 0.5, 20, cache=False),
        # STEP 3 - re-reference to the two mastoid electrodes (a
        # standard EEG reference choice).
        '0.5-20-mast': RawReReference('0.5-20', ['A1', 'A2']),
        # STEP 4 - fit ICA on step 3's output, and remove whichever
        # components get marked as artifacts (mainly eye blinks).
        #
        # The fitting and the human component selection both happen by
        # running `e.make_ica_selection()`, which
        # preprocess_all_subjects() below also calls for you - the
        # first time it runs for a subject, this line's RawICA(...) is
        # what actually gets fit; the selection you make is then
        # cached and reused automatically by every later request for
        # the 'ica' stage.
        'ica': RawICA('0.5-20-mast', 'cocktail', cache=True, fit_kwargs=dict(decim=16)),
        # STEP 5 - a wider filter band, for analyses that need more
        # than 0.5-20 Hz.
        '1-40': RawFilter('raw', 1, 40, cache=False),
        # STEP 6 - re-reference this wider band the same way as step 3.
        '1-40-mast': RawReReference('1-40', ['A1', 'A2']),
        # STEP 7 - apply the same ICA solution from step 4 to this
        # wider band, reusing the same component selection without
        # redoing it.
        '1-40-ica': RawApplyICA('1-40-mast', 'ica', cache=True),
    }

    groups = {
        # Subject 13 is analyzed separately in some notebooks, to check
        # the predictive power of the averaged "mixture" predictor.
        'mix-av': SubGroup('all', 'sub-13'),
    }

    def preprocess_all_subjects(self, skip=()):
        """Run the interactive part of steps 1 and 4 of `raw`, for every subject.

        `raw` above only describes *how* to compute each stage; steps 1
        and 4 also need a human to look at a plot and click (marking
        bad channels, then picking which ICA components are artifacts).
        This method is what actually walks through every subject and
        asks for that input, one subject at a time.

        Both make_bad_channels() and make_ica_selection() open a plot
        and pause until you close it, so this loop advances to the
        next subject automatically the moment you're done with the
        current one - there's nothing else to run or edit by hand.

        Results are cached per subject (see steps 1 and 4 above), so a
        subject that's already done is never reprocessed. Pass
        `skip=['sub-01', ...]` to explicitly resume partway through.
        """
        subjects = [f'sub-{i:02d}' for i in range(1, 14)]  # sub-01 .. sub-13

        for subject in subjects:
            if subject in skip:
                continue

            print(f"\n=== {subject} ===")
            self.set(subject=subject)
            self.make_bad_channels()
            self.make_ica_selection(epoch='clean', decim=16)

        print("\nDone. Every subject has bad channels marked and ICA fit.")

    def fix_events(self, ds):
        # Trigger codes 1 and 2 mark the start of a story; anything else
        # in the trigger channel isn't a stimulus onset and is dropped.
        return ds.sub("value.isin((1, 2))")

    def label_events(self, ds):
        # The trigger value (1 or 2) tells us which of the two stimulus
        # lists was used for this subject/recording.
        list_id = ds[0, 'value']
        ds[:, 'list_id'] = list_id
        ds['condition'] = Factor(SPATIAL)

        speakers = SPEAKER[list_id - 1]
        ds['fg'] = Factor([f'{speaker}_{i}' for i, speaker in zip(range(1, 13), speakers)])

        # The background speaker is the other list's speaker order; there
        # is no background during the two "clean" segments (1 and 7).
        speakers = SPEAKER[-list_id + 2]
        ds['bg'] = Factor(['' if i in (1, 7) else f'{speaker}_{i}' for i, speaker in zip(range(1, 13), speakers)])
        ds['mix'] = Factor(['' if i in (1, 7) else f'List_{list_id}_stim_{i}' for i in range(1, 13)])

        ds['side'] = Factor(SIDE[list_id - 1])
        ds['duration'] = Var([SEGMENT_DURATION[stimulus] for stimulus in ds['fg']])
        return ds

    # The data segment ("epoch") to extract around each stimulus onset.
    # "all" covers the whole story; the rest are subsets by condition.
    # For details see https://eelbrain.readthedocs.io/en/stable/experiment.html
    epochs = {
        'all': PrimaryEpoch('cocktail', tmin=-0.200, tmax='duration', samplingrate=64),
        'clean': SecondaryEpoch('all', "condition == 'clean'"),
        'binaural': SecondaryEpoch('all', "condition == 'binaural'", tmin=3),
        'dichotic': SecondaryEpoch('all', "condition == 'dichotic'", tmin=3),
        'dichotic-left': SecondaryEpoch('dichotic', "side == 'left'"),
        'dichotic-right': SecondaryEpoch('dichotic', "side == 'right'"),
        'diotic': SecondaryEpoch('all', "condition == 'diotic'", tmin=3),
    }

    # Which event variable identifies the stimulus for each segment. This
    # is used to match each EEG segment to its predictor file, e.g. the
    # "gammatone-8" predictor for stimulus "male_3" is expected at
    # <DATA_ROOT>/predictors/male_3~gammatone-8.pickle.
    stim_var = 'fg'

    # The predictors available to the TRF models, generated by
    # predictors/gammatone_predictors.py.
    predictors = {
        'gammatone': FilePredictor(resample='bin'),
    }


# Creating the pipeline instance here means other scripts can just do
# `from experiment import e` instead of repeating this setup.
e = BinauralCocktail(DATA_ROOT)

# This only runs when this file is executed directly
# (`python analysis/experiment.py`), never when it's imported
# (`from experiment import e`, what analysis/cortical_analysis.ipynb
# does) - so importing this file is always safe and never launches the
# interactive preprocessing by accident.
if __name__ == '__main__':
    e.preprocess_all_subjects()
