"""
What this file is for
----------------------
- This file does not analyze anything by itself. It's the setup/config
  file for the whole project.
- It tells eelbrain everything it needs to know about this dataset once,
  in one place: where the recordings are, how to clean them up, what
  the trigger codes mean, which stimulus goes with which EEG segment,
  and so on.
- Every analysis script then just says "from experiment import e" and
  asks that object for what it needs (e.g. "give me the cleaned EEG
  for the diotic condition"), instead of repeating all of this setup
  itself. Importing this file never launches anything interactive.
- The one interactive step this project needs (marking bad channels
  and picking ICA artifact components, once per subject) is
  `preprocess_all_subjects()` below. It's called automatically as the
  second cell of analysis/cortical_analysis.ipynb - you don't need to
  call it yourself, but you can (`e.preprocess_all_subjects()`, from
  any Python session) if you ever want to do that step on its own.

This uses eelbrain's own `Pipeline` class directly (its in-development
BIDS support), not the `trftools` package. `trftools`'s published code
hasn't been updated to match this eelbrain rewrite yet, so it can't
currently be imported against it at all - and it turns out it isn't
needed anyway, this eelbrain version already has TRF fitting,
UTSPredictor, and everything else this project uses built in natively.
Verified against the original author's own reference implementation
for this dataset: https://github.com/christianbrodbeck/binaural-cocktail/tree/eelbrain-0.43

What runs, top to bottom, the moment this file is imported
------------------------------------------------------------
- Work out DATA_ROOT: the BIDS folder Pipeline reads the EEG recordings
  from (predictors and stimuli live one level up from this, in the
  outer dataset folder - see the comment above DATA_ROOT below).
- Build SEGMENT_DURATION: a lookup table of how long (in seconds) each
  audio stimulus is, so eelbrain knows how much EEG to cut out for each
  one.
- Load MONTAGE from biosemi64mod.txt: the physical 3D position of each
  electrode, needed to draw scalp maps. (Channels were already renamed
  from the cap's numbered electrodes, A1, A2, ..., to these standard
  names, back in data/bids_extraction.py, before the recordings were
  even written to BIDS - not here.)
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
      only runs when something calls it - see "What this file is for"
      above for where that happens.
    - label_events: how to figure out, from the trigger codes in the
      recording, which condition and which stimulus each segment was.
    - epochs: which time windows of EEG to extract, and how to split
      them up by condition (clean / diotic / binaural / dichotic).
    - predictors: which generated predictor files (from
      predictors/gammatone_predictors.py) are available to use in the analysis.
- Last line: "e = BinauralCocktail(DATA_ROOT)" builds the one
  ready-to-use pipeline object, `e`, that every analysis script imports.
  This is the only line that produces something other scripts use.
"""
from eelbrain import Factor, Var, gui
from eelbrain.pipeline import *
import mne
from pathlib import Path

# The dataset lives outside this repo, in a sibling "dataset/cocktail"
# folder next to it. Working this out from this file's own location
# means the pipeline works the same way no matter whose computer, or
# which folder, the repo is cloned into. That outer folder contains:
#   bids/                          the BIDS-formatted EEG recordings
#                                  (data/bids_extraction.py) - this is
#                                  what Pipeline needs as its root, not
#                                  the outer folder itself
#   stimuli/                       the stimulus .wav files
#   bids/derivatives/predictors/   the generated predictor files
#                                  (predictors/gammatone_predictors.py) -
#                                  UTSPredictor looks for these here,
#                                  not directly under bids/
#
# Pipeline() must point directly at the BIDS root itself (the folder
# containing dataset_description.json, sub-01/, sub-02/, ...), not its
# parent - so unlike the other scripts here, this file's DATA_ROOT
# includes the "bids" folder.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = str(REPO_ROOT.parent / "dataset" / "cocktail" / "bids")

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

# Channel names here are already the renamed ones (Fp1, AF7, ..., plus
# the two mastoid channels as "A1"/"A2") - see data/bids_extraction.py's
# CH_MAP for that renaming. It has to happen there, before the
# recordings are written to BIDS, not here: eelbrain's interactive
# bad-channel selection GUI checks each channel's name against BIDS's
# own channels.tsv file, and that file is written once, at BIDS-extraction
# time, with whatever names the recording had then. Renaming again here
# (after loading from BIDS) would make live channel names disagree with
# channels.tsv, and the GUI would end up matching almost no channels -
# this was a real bug, and exactly the failure that first exposed it:
#   QhullError: QH6214 qhull input error: not enough points(2) to
#   construct initial simplex (need 4)
# (only the 2 mastoid channels coincidentally matched by name).

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


class BinauralCocktail(Pipeline):

    # The preprocessing steps applied to the raw EEG, each one building on
    # the previous. This dict only describes *how* to compute each stage -
    # two of the stages below (1 and 4) also need a human to look at a
    # plot and click, which happens by calling preprocess_all_subjects()
    # below, not by anything written in this dict itself. For full
    # details on the raw-processing pipeline see
    # https://eelbrain.readthedocs.io/en/stable/experiment.html
    raw = {
        # STEP 1 - load the recording. Pipeline finds each subject's raw
        # file automatically from the BIDS dataset (whatever format it's
        # in), so unlike a plain filename, nothing needs to be specified
        # here beyond how to prepare the channels. Channels arrive
        # already renamed by data/bids_extraction.py (see the comment
        # above MONTAGE near the top of this file for why) - only the
        # montage (electrode positions) needs applying here.
        #
        # Bad channels are excluded right here, automatically, the
        # moment this stage loads - not because this dict says so, but
        # because RawSource always checks for a per-subject
        # bad-channels file first. That file is created by running
        # `e.make_bad_channels_selection()`, which the
        # preprocess_all_subjects() method below calls for you. Every
        # stage below inherits whatever gets excluded here.
        'raw': RawSource(
            adjacency='auto',
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
        'ica': RawICA('0.5-20-mast', fit_kwargs=dict(decim=16)),
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

    # TRF-fitting settings for the boosting algorithm. These used to be
    # passed directly as loose keyword arguments to load_trfs()/
    # load_model_test() (see analysis/cortical_analysis.ipynb's
    # PARAMETERS); they now belong here instead. `basis` and `error`
    # match Boosting's own defaults - listed explicitly so they're easy
    # to find and change - while `partitions` and `selective_stopping`
    # override the defaults.
    estimators = {
        'boosting': Boosting(basis=0.050, error='l1', partitions=-4, selective_stopping=1),
    }

    def preprocess_all_subjects(self, skip=()):
        """Run the interactive part of steps 1 and 4 of `raw`, for every subject.

        `raw` above only describes *how* to compute each stage; steps 1
        and 4 also need a human to look at a plot and click (marking
        bad channels, then picking which ICA components are artifacts).
        This method is what actually walks through every subject and
        asks for that input, one subject at a time.

        Both make_bad_channels_selection() and make_ica_selection() open
        a plot; the gui.run() call right after each one is what actually
        pauses execution until you close that window, so this loop
        advances to the next subject automatically the moment you're
        done with the current one - there's nothing else to run or edit
        by hand.

        Results are cached per subject (see steps 1 and 4 above), so a
        subject that's already done is never reprocessed. Pass
        `skip=['01', ...]` to explicitly resume partway through.
        """
        # Pipeline's own subject values, as found in the BIDS dataset -
        # plain "01", "02", ... (no "sub-" prefix; that prefix is only
        # part of the folder/file names on disk, not the subject field
        # value itself).
        subjects = self.get_field_values('subject')

        for subject in subjects:
            if subject in skip:
                continue

            print(f"\n=== {subject} ===")
            self.set(subject=subject)
            self.make_bad_channels_selection()
            # gui.run() hands control to the open window and waits for
            # it to close before continuing. In a plain terminal,
            # eelbrain can do this on its own the moment the window
            # opens; in Jupyter it can't (its message says so - "Use
            # eelbrain.gui.run() to start GUI interaction"), so without
            # this line the loop would race ahead to the next subject
            # before you've even looked at the window.
            gui.run()
            # raw='ica' points this at the 'ica' stage in `raw` above
            # (the one built with RawICA) - without it, this defaults
            # to whatever the current `raw` state happens to be, which
            # may be a stage upstream of ICA and would fail.
            self.make_ica_selection(raw='ica', epoch='clean', decim=16)
            gui.run()

        print("\nDone. Every subject has bad channels marked and ICA fit.")

    def label_events(self, ds):
        # Trigger codes 1 and 2 mark the start of a story; anything else
        # in the trigger channel isn't a stimulus onset and is dropped.
        # This has to happen here, as the first step of label_events,
        # rather than in a separate fix_events method - Pipeline (unlike
        # the older MneExperiment API) doesn't call a fix_events hook,
        # confirmed against the original author's own reference
        # implementation for this dataset (see the Setup section of
        # the README).
        ds = ds.sub("value.isin((1, 2))")

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
    # <DATA_ROOT>/derivatives/predictors/male_3~gammatone-8.pickle
    # (DATA_ROOT here is the bids/ folder itself - see the comment above
    # DATA_ROOT near the top of this file).
    stim_var = 'fg'

    # The predictors available to the TRF models, generated by
    # predictors/gammatone_predictors.py. UTSPredictor is eelbrain's
    # built-in predictor type for a continuous time-series file per
    # stimulus (envelope, onsets, etc.); resample='bin' matches how
    # gammatone_predictors.py saved these (at their own native rate,
    # to be averaged down to whatever rate an analysis asks for).
    predictors = {
        'gammatone': UTSPredictor(resample='bin'),
    }


# Creating the pipeline instance here means other scripts can just do
# `from experiment import e` instead of repeating this setup.
e = BinauralCocktail(DATA_ROOT)
