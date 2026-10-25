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
- The two interactive steps this project needs, once per subject, are
  `mark_bad_channels()` and `select_ica_artifacts()` below. Both are
  called automatically near the start of
  analysis/cortical_analysis.ipynb - you don't need to call them
  yourself, but you can (`e.mark_bad_channels('all')` /
  `e.select_ica_artifacts('all')`, from any Python session) if you
  ever want to do either step on its own.

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
    - mark_bad_channels() / select_ica_artifacts(): each loops over one
      or every subject and runs one of the two interactive steps `raw`
      depends on (mark bad channels; pick ICA artifact components).
      These are methods, not scripts, so they only run when something
      calls them - see "What this file is for" above for where that
      happens.
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
from eelbrain import Factor, NDVar, Var, configure, gui, plot, resample, table
from eelbrain.pipeline import *
# Private import: this is the exception eelbrain itself raises when a saved
# ICA file no longer matches the bad channels (or other raw settings) it was
# fit against - see _ica_is_stale() below. Not part of eelbrain's public API,
# so this could break on a future eelbrain version; if the import ever fails,
# the fix is to find wherever ProtectedArtifactError lives in that version.
from eelbrain._experiment.derivative_cache.base import ProtectedArtifactError
from matplotlib.ticker import PercentFormatter
import mne
import numpy
import os
from pathlib import Path
import re

# On a computing cluster (SLURM), eelbrain would otherwise start one
# worker per core of the whole node, not just the cores this job was
# given - so match it to the job's allocation instead.
if 'SLURM_CPUS_PER_TASK' in os.environ:
    configure(n_workers=int(os.environ['SLURM_CPUS_PER_TASK']))


def _ask(prompt):
    """input(), unless COCKTAIL_ANSWER is set (unattended runs, e.g.
    cluster/run_notebook.sbatch): then that answer is used without
    waiting for anyone, and printed so the log shows what was chosen."""
    answer = os.environ.get('COCKTAIL_ANSWER')
    if answer is None:
        return input(prompt)
    print(f"{prompt}{answer}  (COCKTAIL_ANSWER)")
    return answer


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

# Where every figure this project saves lives: BIDS-Derivatives
# convention for a group-level figure (not any one subject's data) -
# no `sub-` entity, directly under derivatives/<pipeline>/figures/.
# Exported so the notebook can save its own plots (ROI map, TRFs) here
# too, instead of somewhere outside the dataset (e.g. the Desktop).
FIGURES_DIR = Path(DATA_ROOT) / 'derivatives' / 'eelbrain' / 'figures'

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

# mne-icalabel's ICLabel classifier assigns each ICA component one of
# these categories. "brain" and "other" are always kept; every other
# category is a candidate for automatic rejection in
# select_ica_artifacts() below (see auto_ica_confidence there).
ICA_ARTIFACT_LABELS = ('eye blink', 'muscle artifact', 'heart beat', 'line noise', 'channel noise')

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


# Boosting settings shared by the default pipeline below and by
# set_partitions(). partitions=-4: 4-fold cross-validation on the
# concatenated data, the same 4 folds Brodbeck et al. 2020 used.
BOOSTING_SETTINGS = dict(basis=0.050, error='l1', partitions=-4, selective_stopping=1)


class BinauralCocktail(Pipeline):

    # The preprocessing steps applied to the raw EEG, each one building on
    # the previous. This dict only describes *how* to compute each stage -
    # two of the stages below (1 and 4) also need a human to look at a
    # plot and click, which happens by calling mark_bad_channels() /
    # select_ica_artifacts() below, not by anything written in this dict
    # itself. For full details on the raw-processing pipeline see
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
        # bad-channels file first. That file is created by
        # mark_bad_channels() below, either interactively
        # (make_bad_channels_selection()) or automatically
        # (make_bad_channels_neighbor_correlation()) depending on how
        # it's called. Every stage below inherits whatever gets
        # excluded here.
        'raw': RawSource(
            adjacency='auto',
            montage=MONTAGE,
        ),
        # STEP 2 - band-pass filter for the cortical analysis: cortical
        # responses to speech are slow, so a 0.5-20 Hz filter keeps the
        # signal of interest while removing slow drift and
        # high-frequency noise.
        #
        # cache=True: this stage's output (the whole filtered
        # recording, before re-referencing or ICA) is reused across
        # every epoch/condition and every predictor model analyzed
        # against raw='ica' below - none of that depends on this step,
        # only on which subject it is. Without caching, this exact
        # filtering is redone from the raw file every single time
        # anything asks for 'ica' data, even for a condition or
        # predictor already analyzed moments earlier.
        '0.5-20': RawFilter('raw', 0.5, 20, cache=True),
        # STEP 3 - re-reference to the two mastoid electrodes (a
        # standard EEG reference choice).
        '0.5-20-mast': RawReReference('0.5-20', ['A1', 'A2']),
        # STEP 4 - fit ICA on step 3's output, and remove whichever
        # components get marked as artifacts (mainly eye blinks).
        #
        # The fitting and the human component selection both happen by
        # running `e.make_ica_selection()`, which
        # select_ica_artifacts() below also calls for you - the
        # first time it runs for a subject, this line's RawICA(...) is
        # what actually gets fit; the selection you make is then
        # cached and reused automatically by every later request for
        # the 'ica' stage.
        #
        # cache=True: same reasoning as step 2 - this stage's output
        # (the fully cleaned continuous recording: filtered,
        # re-referenced, ICA artifacts removed) is what every
        # load_trfs(..., raw='ica', ...) call actually uses, regardless
        # of which condition or predictor it's for. Caching it means
        # only the very first request per subject does this work; every
        # later one - a different epoch, a different predictor, a
        # different day's session - just loads the saved result.
        'ica': RawICA('0.5-20-mast', fit_kwargs=dict(decim=16), cache=True),
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
    # override the defaults. The notebook picks its own `partitions`
    # through set_partitions() below, so the value here is only the
    # default for anything that imports `e` without choosing one.
    estimators = {
        'boosting': Boosting(**BOOSTING_SETTINGS),
    }

    def _resolve_subjects(self, subjects, skip=()):
        """Turn a `mark_bad_channels()`/`select_ica_artifacts()` `subjects`
        argument into the list of subject names to loop over.

        subjects
            'all' to go through every subject in the dataset (Pipeline's
            own subject values, as found in the BIDS dataset - plain
            "01", "02", ... , no "sub-" prefix), same as `load_trfs()`.
            Otherwise a single subject, e.g. 1 or '01' - normalized to
            that same zero-padded, two-digit form.
        skip
            Only relevant when subjects='all' - subject names to leave
            out of the loop (to resume partway through).
        """
        if subjects == 'all':
            return [s for s in self.get_field_values('subject') if s not in skip]
        return [f'{int(subjects):02d}']

    def _subject_deriv_dir(self, subject):
        """Where mark_bad_channels()/select_ica_artifacts() write this
        subject's files - `derivatives/mne/sub-XX/eeg/`, same folder
        _auto_select_ica() below writes its .tsv/.png to."""
        return Path(DATA_ROOT) / 'derivatives' / 'mne' / f'sub-{subject}' / 'eeg'

    def _ica_is_stale(self, subject):
        """True if `subject`'s saved ICA file no longer matches their
        current bad channels (or whatever else the 'ica' raw stage
        depends on) - e.g. because mark_bad_channels() changed what's
        excluded after ICA was already fit for them.

        This is exactly the check eelbrain itself makes before handing
        out a saved ICA file (see ProtectedArtifactError): loading it
        without accept_stale raises if the two disagree. Used by
        _filter_already_done() below so select_ica_artifacts() can
        force a redo automatically for a stale subject, instead of
        this surfacing later as a cryptic error the first time
        something else (e.g. load_trfs()) needs that subject's cleaned
        data.
        """
        try:
            self.load_ica(raw='ica', subject=subject)
        except ProtectedArtifactError:
            return True
        return False

    def _make_ica_unattended_safe(self):
        """make_ica(raw='ica'), without eelbrain's stale-ICA prompt in unattended runs.

        Copying the dataset to a cluster rounds file times to whole seconds,
        which eelbrain reports as a changed input and answers with an
        interactive prompt (a crash without a terminal). When COCKTAIL_ANSWER
        is set and every recorded change is such a rounded file time, this
        gives eelbrain's own 'incorporate' answer: keep the ICA, update its
        manifest. Any other change raises with eelbrain's reason.
        """
        if os.environ.get('COCKTAIL_ANSWER') is None:
            return self.make_ica(raw='ica')
        try:
            self.load_ica(raw='ica')
            return self.make_ica(raw='ica')
        except ProtectedArtifactError as error:
            reason = str(error)
        changes = re.findall(r'([^\s(,]+): (\S+?) -> (\S+?)[,)]', reason)
        only_rounded_mtimes = (
            changes and len(changes) == reason.count(' -> ')
            and all(key.endswith('mtime') and int(float(old)) == int(float(new)) for key, old, new in changes)
        )
        if not only_rounded_mtimes:
            raise RuntimeError(f"Stale ICA for subject {self.get('subject')} needs a decision by hand:\n{reason}")
        from eelbrain._experiment import pipeline as eelbrain_pipeline
        original_ask = eelbrain_pipeline.ask
        eelbrain_pipeline.ask = lambda *args, **kwargs: 'incorporate'
        try:
            print("  ICA only differs in rounded file times - keeping it (incorporate).")
            return self.make_ica(raw='ica')
        finally:
            eelbrain_pipeline.ask = original_ask

    def _filter_already_done(self, subjects_arg, resolved_subjects, glob_pattern, description, is_stale=None):
        """Drop subjects that already have saved output, after asking.

        Used by mark_bad_channels() and select_ica_artifacts() when
        `confirm_overwrite=True` (the default), right before their main
        loop, to decide which of `resolved_subjects` to actually run.

        Only subjects with a matching file under
        `derivatives/mne/sub-XX/eeg/` (glob_pattern) are ever in
        question - a subject with nothing saved yet is always run, no
        prompt needed.

        is_stale
            Optional `subject -> bool` check (select_ica_artifacts()
            passes _ica_is_stale). A subject whose existing file is
            stale is always redone - its old file can't be reused
            regardless of what the caller would otherwise choose - so
            it's never part of the skip/redo question below, and never
            silently kept via a "skip" answer either.

        The scope of the skip/redo question (for whichever already-done
        subjects are left after removing stale ones) matches the scope
        of the original call, via `subjects_arg` (the raw `subjects`
        value the caller passed in, before _resolve_subjects() expanded
        it):
          - `subjects_arg == 'all'`: one combined question covering
            every already-done, non-stale subject in the group at once
            - "skip all of them" or "redo all of them", not one prompt
            per subject.
          - a specific subject: a single, subject-specific question
            (there is only ever one subject to ask about in that case).
        Typing anything other than 'r' - including just pressing enter
        - skips, since skipping (keeping existing results) is the safer
        default than accidentally overwriting a manually-reviewed
        selection. If nothing is left to ask about (every already-done
        subject turned out stale, or there simply are none), no prompt
        is shown at all.
        """
        already_done = {
            s: existing[0]
            for s in resolved_subjects
            if (existing := list(self._subject_deriv_dir(s).glob(glob_pattern)))
        }
        if not already_done:
            return resolved_subjects

        stale = set()
        if is_stale is not None:
            for subject in already_done:
                if is_stale(subject):
                    stale.add(subject)
                    print(f"  {subject}: {description} is stale (no longer matches current settings) - redoing.")

        askable = {s: p for s, p in already_done.items() if s not in stale}
        if not askable:
            return resolved_subjects

        if subjects_arg == 'all':
            names = ', '.join(askable)
            answer = _ask(
                f"  {description} already exists for: {names}. "
                f"Skip all of them (default) or redo and overwrite all of them? [s/r]: "
            ).strip().lower()
            if answer == 'r':
                return resolved_subjects
            print(f"  skipping (already done): {names}")
            return [s for s in resolved_subjects if s not in askable]
        else:
            subject, existing_path = next(iter(askable.items()))
            answer = _ask(
                f"  {subject}: {description} already exists ({existing_path.name}). "
                f"Skip (default) or redo and overwrite? [s/r]: "
            ).strip().lower()
            if answer == 'r':
                return resolved_subjects
            print(f"  skipping {subject} (already done)")
            return [s for s in resolved_subjects if s not in askable]

    def mark_bad_channels(
            self, subjects='all', skip=(),
            auto_bad_channels_r=False, manual_bad_channels=True, redo_bad_channels=False,
            confirm_overwrite=True,
    ):
        """Run the interactive part of step 1 of `raw` (mark bad channels).

        `raw` above only describes *how* to compute each stage; step 1
        also needs a human to look at a plot and click, marking bad
        channels. This method is what actually walks through one
        subject, or every subject, and asks for that input.

        make_bad_channels_selection() opens a plot; the gui.run() call
        right after it is what actually pauses execution until you
        close that window, so a `subjects='all'` loop advances to the
        next subject automatically the moment you're done with the
        current one - there's nothing else to run or edit by hand.

        Before running anyone, this checks which of the requested
        subjects already have a channels.tsv under
        `derivatives/mne/sub-XX/eeg/` (i.e. this step has already been
        run for them at least once). If any do, and `confirm_overwrite`
        is True (the default), it asks - via a plain text prompt, not a
        GUI - whether to skip or redo them. The question's scope
        matches `subjects`: with `subjects='all'`, one combined question
        covers every already-done subject at once ("skip all of them"
        or "redo all of them"), not a prompt per subject; with a
        specific subject, the question is just about that one. See
        confirm_overwrite below. Pass `skip=['01', ...]` to instead
        unconditionally skip specific subjects without being asked.

        subjects
            'all' (default) to go through every subject, or a single
            subject (e.g. 1 or '01') to do just that one - same
            convention as `load_trfs()`'s first argument.
        confirm_overwrite
            True (default): if any requested subject already has a
            channels.tsv file, ask before redoing them (typing anything
            other than 'r', including just pressing enter, skips -
            skipping is the safer default). False: never ask, always
            run every requested subject regardless of what's already
            saved (the old, pre-confirmation behavior) - overwriting
            automatic results, or re-opening the GUI pre-loaded with
            whatever was marked before, for manual ones.
        manual_bad_channels
            True (default): mark bad channels by hand, in the GUI.
        auto_bad_channels_r
            Correlation threshold (e.g. 0.3) for finding bad channels
            automatically instead - only used when `manual_bad_channels`
            is False. Uses _safe_bad_channels_neighbor_correlation()
            below (the same algorithm as eelbrain's own
            make_bad_channels_neighbor_correlation(), with one of its
            bugs worked around - see that method) to compute each
            channel's correlation with its neighbors - the exact same
            computation behind the GUI's "Neighbor corr" scalp maps -
            and mark any channel below this threshold as bad, with no
            window to close.

            This check is relative, not absolute: it repeatedly removes
            whichever channel currently correlates worst with its
            neighbors, then re-checks what's left, until nothing remains
            below the threshold. That means the result depends on the
            starting point - if some channels are already marked bad
            from an earlier run, this starts from that smaller set
            instead of the full one, and excluding channels shifts
            everyone else's neighbor correlation. Run repeatedly without
            resetting first, this can keep finding "new" bad channels
            indefinitely, compounding rather than converging. See
            redo_bad_channels below to avoid that.
        redo_bad_channels
            Only relevant when manual_bad_channels is False. False
            (default): add any newly-found bad channels to whatever this
            subject already has marked (unchanged from before). True:
            clear this subject's previously auto-found bad channels
            first, so the check starts fresh from the full channel set
            every time - the same starting point gives the same result
            every time, instead of drifting further with each re-run.

        Examples:
            e.mark_bad_channels('all')                                                 # by hand (default)
            e.mark_bad_channels('all', manual_bad_channels=False, auto_bad_channels_r=0.3)  # automatic
            e.mark_bad_channels('all', manual_bad_channels=False, auto_bad_channels_r=0.3,
                                 redo_bad_channels=True)                                # automatic, from a clean slate
            e.mark_bad_channels(1)                                                     # just one subject, by hand
            e.mark_bad_channels('all', confirm_overwrite=False)                        # rerun everyone, no prompts
        """
        if not manual_bad_channels and auto_bad_channels_r is False:
            raise ValueError(
                "manual_bad_channels=False needs a real auto_bad_channels_r "
                "threshold (e.g. 0.3), not the default False."
            )

        subjects_to_run = self._resolve_subjects(subjects, skip)
        if confirm_overwrite:
            subjects_to_run = self._filter_already_done(
                subjects, subjects_to_run, '*_channels.tsv', "bad-channels file",
            )

        for subject in subjects_to_run:
            print(f"\n=== {subject} ===")
            self.set(subject=subject)
            if manual_bad_channels:
                self.make_bad_channels_selection()
                # gui.run() hands control to the open window and waits
                # for it to close before continuing. In a plain
                # terminal, eelbrain can do this on its own the moment
                # the window opens; in Jupyter it can't (its message
                # says so - "Use eelbrain.gui.run() to start GUI
                # interaction"), so without this line the loop would
                # race ahead to the next subject before you've even
                # looked at the window.
                gui.run()
            else:
                if redo_bad_channels:
                    # bad_chs=(), redo=True: reset this subject to no bad
                    # channels first, so the check below starts from the
                    # full channel set instead of building on whatever an
                    # earlier run already excluded.
                    self.make_bad_channels([], redo=True)
                _, bad_channels = self._safe_bad_channels_neighbor_correlation(
                    auto_bad_channels_r, epoch='clean',
                )
                print(f"  bad channels (r < {auto_bad_channels_r}): {bad_channels or 'none'}")

        print("\nDone. Bad channels marked.")

    def select_ica_artifacts(
            self, subjects='all', skip=(),
            auto_ica_confidence=False, manual_ica=True, auto_ica_reject_labels=ICA_ARTIFACT_LABELS,
            confirm_overwrite=True,
    ):
        """Run the interactive part of step 4 of `raw` (fit ICA, pick artifacts).

        `raw` above only describes *how* to compute each stage; step 4
        also needs a human to look at a plot and click, picking which
        ICA components are artifacts. This method is what actually
        walks through one subject, or every subject, and asks for that
        input - fitting ICA along the way if it hasn't been fit yet.

        make_ica_selection() opens a plot; the gui.run() call right
        after it is what actually pauses execution until you close that
        window, so a `subjects='all'` loop advances to the next subject
        automatically the moment you're done with the current one -
        there's nothing else to run or edit by hand.

        Before running anyone, this checks which of the requested
        subjects already have an ICA file under
        `derivatives/mne/sub-XX/eeg/` (i.e. ICA has already been fit,
        and - in normal use, since nothing else in this project touches
        the 'ica' raw stage first - already reviewed for artifacts at
        least once).

        Any of those whose ICA file is stale - no longer matching that
        subject's *current* bad channels, e.g. because
        mark_bad_channels() changed what's excluded for them after ICA
        was already fit - are always redone, regardless of what you'd
        otherwise choose below; eelbrain refuses to reuse an ICA fit on
        a different set of channels than the ones actually in use now
        (see _ica_is_stale()), so keeping a stale one isn't really an
        option, only recomputing it or reverting the bad-channels
        change is. Left as-is, a stale ICA doesn't fail here - it fails
        later, as a confusing ProtectedArtifactError, the first time
        something else (e.g. load_trfs()) needs that subject's data.

        For whichever already-done subjects are left (not stale), if
        `confirm_overwrite` is True (the default), it asks - via a
        plain text prompt, not a GUI - whether to skip or redo them.
        The question's scope matches `subjects`: with `subjects='all'`,
        one combined question covers all of them at once ("skip all of
        them" or "redo all of them"), not a prompt per subject; with a
        specific subject, the question is just about that one. See
        confirm_overwrite below. Pass `skip=['01', ...]` to instead
        unconditionally skip specific subjects without being asked.

        Run this after mark_bad_channels() - ICA is fit on top of
        whichever channels that step already excluded.

        subjects
            'all' (default) to go through every subject, or a single
            subject (e.g. 1 or '01') to do just that one - same
            convention as `load_trfs()`'s first argument.
        confirm_overwrite
            True (default): if a subject already has an ICA file, ask
            before redoing that subject (typing anything other than
            'r', including just pressing enter, skips it - skipping is
            the safer default, since re-fitting and reselecting
            replaces a possibly manually-reviewed result). False: never
            ask, always run every requested subject regardless of
            what's already saved (the old, pre-confirmation behavior).
        manual_ica
            True (default): pick ICA artifact components by hand, in
            the GUI.
        auto_ica_confidence
            Confidence threshold (e.g. 0.75) for finding artifact
            components automatically instead - only used when
            `manual_ica` is False. Runs mne-icalabel's ICLabel
            classifier on each component and marks it excluded if its
            predicted category is in `auto_ica_reject_labels` with at
            least this much confidence. See _auto_select_ica() for
            exactly what this computes and writes.
        auto_ica_reject_labels
            Which ICLabel categories count as "reject" for
            auto_ica_confidence above. Defaults to every artifact
            category ICA_ARTIFACT_LABELS defines near the top of this
            file (eye blink, muscle artifact, heart beat, line noise,
            channel noise) - "brain" and "other" are never
            auto-rejected.

        Examples:
            e.select_ica_artifacts('all')                                              # by hand (default)
            e.select_ica_artifacts('all', manual_ica=False, auto_ica_confidence=0.75)   # automatic
            e.select_ica_artifacts(1)                                                   # just one subject, by hand
            e.select_ica_artifacts('all', confirm_overwrite=False)                      # rerun everyone, no prompts
        """
        if not manual_ica and auto_ica_confidence is False:
            raise ValueError(
                "manual_ica=False needs a real auto_ica_confidence "
                "threshold (e.g. 0.75), not the default False."
            )

        subjects_to_run = self._resolve_subjects(subjects, skip)
        if confirm_overwrite:
            subjects_to_run = self._filter_already_done(
                subjects, subjects_to_run, '*_ica.fif', "ICA file",
                is_stale=self._ica_is_stale,
            )

        for subject in subjects_to_run:
            print(f"\n=== {subject} ===")
            self.set(subject=subject)
            if manual_ica:
                # raw='ica' points this at the 'ica' stage in `raw`
                # above (the one built with RawICA) - without it, this
                # defaults to whatever the current `raw` state happens
                # to be, which may be a stage upstream of ICA and would
                # fail.
                self.make_ica_selection(raw='ica', epoch='clean', decim=16)
                gui.run()
            else:
                self._auto_select_ica(auto_ica_confidence, auto_ica_reject_labels)

        print("\nDone. ICA fit and artifact components selected.")

    def _safe_bad_channels_neighbor_correlation(self, r, epoch=None):
        """Same algorithm as eelbrain's own make_bad_channels_neighbor_correlation(),
        with one real bug in that method worked around.

        eelbrain's version repeatedly removes whichever channel currently
        correlates worst with its neighbors, then recomputes correlations
        on what's left - but if a channel's *only* neighbors happen to be
        among the ones already removed, it has nothing left to correlate
        against, and eelbrain's neighbor_correlation() crashes outright
        (ValueError: Some elements do not have any neighbors) instead of
        handling that case. This hit a real subject's data (one channel,
        somewhere in the middle of the process, lost all its neighbors as
        other channels around it kept getting excluded first).

        This copies eelbrain's own algorithm exactly, except that when
        that specific crash happens, it stops the loop right there instead
        of aborting the whole subject - keeping whatever bad channels were
        already found, and leaving the now-neighborless channel and
        everything else untested rather than crashing.
        """
        from eelbrain import neighbor_correlation

        data, full_nc = self.load_neighbor_correlation(1, epoch, return_data=True)
        bad_chs = []
        nc = full_nc
        while nc.min() < r:
            sensor = nc.argmin()
            bad_chs.append(sensor)
            new_index = nc.sensor.index(exclude=sensor)
            data = data.sub(sensor=new_index)
            try:
                nc = neighbor_correlation(data)
            except ValueError:
                break
            full_index = full_nc.sensor.index(exclude=bad_chs)
            full_nc[full_index] = nc

        if bad_chs:
            self.make_bad_channels(bad_chs)
        return full_nc, bad_chs

    def _auto_select_ica(self, confidence, reject_labels):
        """Classify this subject's ICA components with mne-icalabel and mark
        artifacts excluded automatically, instead of picking them by hand in
        the GUI (see the auto_ica_confidence parameter of
        select_ica_artifacts() above, which calls this).

        Writes the same things a human reviewer needs to sanity-check the
        result, saved next to this subject's other preprocessing files
        under derivatives/mne/sub-XX/eeg/:
          - the ICA file itself, with `.exclude` set to the rejected
            component indices (the same file make_ica_selection() would
            have updated by hand)
          - sub-XX_task-cocktail_desc-iclabel_components.tsv: one row per
            component, with its predicted label, confidence, and whether
            it was rejected
          - sub-XX_task-cocktail_desc-iclabel_components.png: a grid of
            every component's scalp topography, numbered and labeled the
            same way as the .tsv, so the two line up directly and a
            component's number in this project always means the same
            index in eelbrain, the .tsv, and this image.

        Note: components labeled "channel noise" here can sometimes mean
        a channel should have been marked bad but wasn't - worth a manual
        look rather than trusting the automatic rejection blindly for
        those specifically.
        """
        from mne_icalabel import label_components
        import matplotlib.pyplot as plt

        path = self._make_ica_unattended_safe()
        ica = mne.preprocessing.read_ica(path, verbose=False)

        # mne-icalabel's ICLabel classifier expects data filtered 1-100 Hz
        # and referenced to a common average (see
        # https://mne.tools/mne-icalabel) - this project's own raw{} uses
        # a narrower filter and a mastoid reference instead, since that's
        # what the actual analysis needs, not what the classifier needs.
        # So this reprocesses a throwaway *copy* of the data just for
        # classification; nothing about the real preprocessing changes.
        #
        # Source from 'raw' (before any of this project's own filtering),
        # not '0.5-20-mast': that stage is already low-pass filtered to
        # 20 Hz, and a filter can never restore content above a cutoff
        # that's already been removed - asking for 1-100 Hz on top of
        # already-0.5-20-Hz data leaves the real upper edge stuck at 20 Hz,
        # which is exactly the mismatch mne-icalabel's own warning caught:
        # "Raw instance is not filtered between 1 and 100 Hz."
        #
        # preload=False + tstart/tstop crops the *lazy* file window before
        # anything is actually read into memory - load_raw() only calls
        # load_data() itself when samplingrate or preload is requested, so
        # with preload=False here, cropping happens first; load_data()
        # below then only ever reads that already-cropped window.
        # preload=True here instead would load the full continuous
        # recording - at its native ~4096 Hz - in full before cropping,
        # which is large enough to exhaust memory outright (this crashed
        # the Jupyter kernel entirely). A few minutes is already enough
        # data for stable topography/spectrum features, so there's no
        # need for more.
        raw = self.load_raw(preload=False, raw='raw', tstart=0, tstop=300)
        raw_for_iclabel = raw.copy()
        raw_for_iclabel.pick(ica.ch_names)
        # filter() needs the data loaded first - unlike some other MNE
        # methods, it does not preload on its own and raises instead.
        raw_for_iclabel.load_data(verbose=False)
        raw_for_iclabel.filter(1., 100., verbose=False)
        raw_for_iclabel.set_eeg_reference('average', verbose=False)

        result = label_components(raw_for_iclabel, ica, method='iclabel')
        labels, probs = result['labels'], result['y_pred_proba']

        rejected = [
            i for i, (label, prob) in enumerate(zip(labels, probs))
            if label in reject_labels and prob >= confidence
        ]
        ica.exclude = rejected
        ica.save(path, overwrite=True)

        subject = self.get('subject')
        out_dir = Path(DATA_ROOT) / 'derivatives' / 'mne' / f'sub-{subject}' / 'eeg'
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f'sub-{subject}_task-cocktail_desc-iclabel_components'

        with open(out_dir / f'{stem}.tsv', 'w') as f:
            f.write('component\tlabel\tprobability\trejected\n')
            for i, (label, prob) in enumerate(zip(labels, probs)):
                f.write(f'{i}\t{label}\t{prob:.4f}\t{i in rejected}\n')

        n = ica.n_components_
        ncols = 8
        nrows = -(-n // ncols)  # ceiling division
        fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2.4 * nrows), squeeze=False)
        components = ica.get_components()
        for i, ax in enumerate(axes.flatten()):
            if i >= n:
                ax.axis('off')
                continue
            mne.viz.plot_topomap(components[:, i], ica.info, axes=ax, show=False)
            is_bad = i in rejected
            ax.set_title(f'IC{i}: {labels[i]}\np={probs[i]:.2f}', fontsize=7, color='crimson' if is_bad else 'black')
            for spine in ax.spines.values():
                spine.set_edgecolor('crimson' if is_bad else 'lightgray')
                spine.set_linewidth(2 if is_bad else 1)
        fig.suptitle(f'sub-{subject}: ICLabel components (red = auto-rejected)')
        fig.tight_layout()
        fig.savefig(out_dir / f'{stem}.png', dpi=150)
        plt.close(fig)

        print(f"  ICLabel rejected {len(rejected)}/{n} components: {rejected}")
        print(f"  see {out_dir / (stem + '.tsv')} and the matching .png")

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


def set_partitions(partitions):
    """Rebuild `e` with a different number of boosting partitions.

    The boosting settings are a class attribute (`estimators`), read
    when the pipeline is built, so changing `partitions` means building
    a new pipeline object: a subclass of BinauralCocktail with only that
    one setting changed, everything else identical. This module's `e`
    (used by the analysis helpers further down) is replaced too, and the
    new object is returned so the notebook can do
    `e = set_partitions(PARTITIONS)`.

    partitions
        Number of cross-validation folds. Negative (e.g. -10) means the
        data are concatenated first and then split into that many
        contiguous folds, the same convention as the -4 default.
        More folds: each fit trains on a larger share of the data, but
        fitting takes proportionally longer.
    """
    global e
    estimators = {'boosting': Boosting(**{**BOOSTING_SETTINGS, 'partitions': partitions})}
    cls = type(BinauralCocktail.__name__, (BinauralCocktail,), {'estimators': estimators, '__module__': __name__})
    e = cls(DATA_ROOT)
    return e


def _bids_value(s):
    """Turn an arbitrary string into a BIDS-legal entity value.

    BIDS entity values may only contain letters and digits - no spaces,
    hyphens, underscores, '+', '~', etc. (all of which show up in
    predictor formulas like "gammatone-1 + bg~gammatone-1"). Used by
    _topomap_figure_path() below to fold a `condition`/`predictor`
    string into a filename safely.
    """
    return re.sub(r'[^A-Za-z0-9]', '', s)


def _topomap_figure_path(label, scope, condition, predictor):
    """Where topomap_with_colorbar()/topomap_by_subject_with_colorbar()
    save their figure.

    Follows BIDS-Derivatives convention: a figure that isn't any one
    subject's data (a group average, or - as here - a single image
    already covering every subject) has no `sub-` entity and lives
    directly under derivatives/<pipeline>/figures/. `condition` and
    `predictor` aren't official BIDS entities (BIDS has none for this),
    so they're added as project-specific `epoch-`/`pred-` entities,
    which BIDS permits; `scope` (e.g. "GroupAvg" vs. "AllSubjects")
    only distinguishes the two plot functions from each other, folded
    into `desc-` alongside the metric label.
    """
    out_dir = FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = ['task-cocktail']
    if condition is not None:
        parts.append(f'epoch-{_bids_value(condition)}')
    if predictor is not None:
        parts.append(f'pred-{_bids_value(predictor)}')
    parts.append(f'desc-{_bids_value(label)}{scope}')
    return out_dir / ('_'.join(parts) + '_topomap.png')


def topomap_with_colorbar(y, data, label=None, pct=True, condition=None, predictor=None, save=True,
                           colorbar_side='right', colorbar_label=True, **topo_args):
    """One topomap image (group average across subjects), with its
    colorbar embedded in the same figure - not a separate plot - and
    the plotted value, averaged across subjects and sensors, shown in
    the title.

    y
        Column in `data` to plot (e.g. 'ev', 'r'). Ignored (pass `None`)
        when `data` is already a bare NDVar - see `data` below.
    data
        The usual case: a Dataset with one row per subject, `y` naming
        the column to average across subjects and plot. Alternatively,
        a single NDVar that already *is* the map to plot, with no
        further per-subject averaging needed - e.g.
        `result.masked_parameter_map()`, a significance-masked group
        statistic. `y` is unused in that case.
    label
        Display name for `y`, used in the title, colorbar, and saved
        filename (defaults to `y` itself) - e.g. pass label='exp' for
        the 'ev' proportion-explained column, which is conventionally
        shortened to "exp".
    pct
        True (default): the value is a fraction (like 'ev', proportion
        of variance explained) - show it, and the colorbar, as a
        percentage. False: the value already has no unit (like 'r', a
        correlation) - show it as a plain number instead.
    condition
        Name of whatever this plot's condition/epoch is (e.g.
        'diotic'). Prepended to the title, and folded into the saved
        filename as an `epoch-` entity - so when a loop produces one of
        these images per condition, each one is still identifiable by
        its title and filename alone, without needing to track print
        order.
    predictor
        The predictor/model formula this data was fit with (e.g.
        'gammatone-1', the same string passed to load_trfs()). Only
        affects the saved filename, as a `pred-` entity - not shown in
        the title, which already gets long enough with the value.
    save
        True (default): save this figure - see _topomap_figure_path()
        for exactly where. False: just display it, save nothing.
    colorbar_side
        'right' (default) or 'left' - which side of the topomap the
        colorbar is drawn on (ticks move to that same side).
    colorbar_label
        True (default): label the colorbar's own axis with `label` (or
        `label (%)` for a percentage), in addition to the tick numbers.
        False: tick numbers only, no separate axis label.
    """
    label = (y if y is not None else 'value') if label is None else label
    if isinstance(data, NDVar):
        # Already a group-level map (e.g. a significance-masked
        # statistic) - nothing left to average across subjects, only
        # across whatever the map itself still varies over (sensors,
        # and - for a masked map - only the unmasked/significant ones).
        mean_value = float(data.mean())
    else:
        mean_value = float(data[y].mean(('case', 'sensor')))
    value_str = f"{mean_value:.2%}" if pct else f"{mean_value:.3f}"
    title = f"Mean {label}: {value_str}"
    if condition is not None:
        title = f"{condition}: {title}"
    if isinstance(data, NDVar):
        p = plot.Topomap(data, title=title, **topo_args)
    else:
        p = plot.Topomap(y, data=data, title=title, **topo_args)
    mappable = p.plots[0].plots[0].im
    cb = p.figure.colorbar(mappable, ax=p.axes[0], shrink=0.7, location=colorbar_side,
                            label=(f'{label} (%)' if pct else label) if colorbar_label else None)
    vmin, vmax = mappable.get_clim()
    cb.set_ticks([vmin, 0, vmax])
    if pct:
        cb.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    if save:
        p.save(_topomap_figure_path(label, 'GroupAvg', condition, predictor))
    return p


def topomap_by_subject_with_colorbar(y, data, label=None, pct=True, condition=None, predictor=None, save=True, **topo_args):
    """One topomap image per subject, all in a single figure with one
    shared colorbar embedded in it - not a separate plot - and each
    subject's value, averaged across sensors, shown in their own title.

    y, label, pct, predictor, save
        See topomap_with_colorbar() above.
    condition
        Name of whatever this plot's condition/epoch is (e.g.
        'diotic'). Shown as the whole figure's own title, above the
        per-subject titles, and folded into the saved filename as an
        `epoch-` entity - so when a loop produces one of these images
        per condition, each one is still identifiable by its title and
        filename alone, without needing to track print order.

    Assumes Topomap lays out subjects in the same order they appear in
    `data` (true for 'all' subjects - the order load_trfs() already
    used to build `data`).
    """
    label = y if label is None else label
    values = data[y].mean('sensor').x
    if pct:
        titles = [f"{subject}\n{value:.2%}" for subject, value in zip(data['subject'], values)]
    else:
        titles = [f"{subject}\n{value:.3f}" for subject, value in zip(data['subject'], values)]
    # Not passing condition as `title=` here: eelbrain draws that title
    # at a height based on the figure's *fraction*, which doesn't
    # account for the two-line axtitle (subject + value) sitting right
    # under it - and this figure (one short row of many subjects) is
    # physically short, so a fraction of its height is a tiny amount of
    # real space, nowhere near enough to clear that title. A fixed
    # fractional offset (tried first) failed for the same reason.
    #
    # Fix: grow the actual canvas by a fixed number of real inches, and
    # hand that whole new strip to the condition title. Growing the
    # figure without touching subplotpars.top would just stretch
    # everything proportionally (mostly enlarging the topomaps, barely
    # the margin) - shrinking `top` afterwards, by exactly the fraction
    # the new height represents, keeps the existing content (topomaps +
    # their titles) at the same absolute size it already had, so the
    # added inches become entirely new, guaranteed-empty space at the
    # top, regardless of how tall or short this particular figure is.
    p = plot.Topomap(y, 'subject', rows=1, data=data, axtitle=titles, **topo_args)
    if condition is not None:
        fig = p.figure
        width, old_height = fig.get_size_inches()
        old_top = fig.subplotpars.top
        extra_inches = 0.8
        new_height = old_height + extra_inches
        fig.set_size_inches(width, new_height)
        new_top = old_height * old_top / new_height
        fig.subplots_adjust(top=new_top)
        fig.suptitle(condition, y=new_top + (1 - new_top) * 0.85, fontsize=13, fontweight='bold')
    mappable = p.plots[0].plots[0].im
    cb = p.figure.colorbar(
        mappable, ax=p.axes, orientation='horizontal', fraction=0.05, pad=0.15,
        label=f'{label} (%)' if pct else label,
    )
    vmin, vmax = mappable.get_clim()
    cb.set_ticks([vmin, 0, vmax])
    if pct:
        cb.ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    if save:
        p.save(_topomap_figure_path(label, 'AllSubjects', condition, predictor))
    return p


# ---------------------------------------------------------------------
# Analysis helpers for analysis/cortical_analysis.ipynb
# ---------------------------------------------------------------------
# The notebook keeps only settings, calls and results; the functions it
# calls live here. Settings the helpers need (fit PARAMETERS, ROI,
# metric, ...) stay in the notebook, where they are easy to see and
# change, and are handed over once with configure_analysis(). Each
# helper reads them at call time, so re-running configure_analysis()
# after changing a setting is enough - no re-import needed.

# Stimulus-role prefixes for predictor names: which event column
# supplies the stimulus for a term. '' = attended speaker (`fg`, the
# default stim_var), 'bg~' = ignored speaker, 'mix~' = the mixture.
ATTEND, IGNORE, MIX = '', 'bg~', 'mix~'

ANALYSIS = {
    'parameters': None,             # keyword arguments for load_trfs()/load_model_test()
    'roi': None,                    # sensors averaged over for ROI summaries
    'metric': 'r',                  # model-fit metric: 'r' or 'ev'
    'pmin': 0.05,                   # cluster-forming threshold for model tests
    'topo_args': {},                # extra arguments for topomap_with_colorbar()
    'taper': (0.003, 0.033),        # display taper: floor up to taper[0], full strength from taper[1]
    'taper_floor': 0.1,             # display taper: share of the TRF kept before taper[0]
    'end_taper': 0.050,             # display taper: fade-out at the end of the fit window
    'plot_window': (-0.050, 0.550), # time range TRF plots show
}


def configure_analysis(**settings):
    """Set the notebook's analysis settings (keys of ANALYSIS) for the
    helpers below. Unknown keys raise, so a typo can't be silently
    ignored."""
    unknown = set(settings) - set(ANALYSIS)
    if unknown:
        raise KeyError(f"Unknown analysis settings: {sorted(unknown)}")
    ANALYSIS.update(settings)


def _setting(name, value=None):
    "An explicitly passed value, else the configured setting."
    if value is not None:
        return value
    if ANALYSIS[name] is None:
        raise RuntimeError(f"Analysis setting {name!r} is not configured - call configure_analysis({name}=...) first")
    return ANALYSIS[name]


# Predictor names. Every predictor code has to start with `gammatone-`
# to resolve (it is the only registered predictor key, see `predictors`
# above).

def envelope(role=ATTEND):
    return f"{role}gammatone-8"


def onset(role=ATTEND):
    return f"{role}gammatone-on-8"


def overt_onset(role=ATTEND):
    return f"{role}gammatone-on-overt-8"


def masked_onset(role=ATTEND):
    return f"{role}gammatone-on-masked-8"


def level_onset(tier, role=MIX):
    assert tier in ('low', 'mid', 'high')
    return f"{role}gammatone-on-{tier}-8"


# Model comparisons

def _model_test(comparison, epoch, roi, metric):
    roi = _setting('roi', roi)
    metric = _setting('metric', metric)
    data, result = e.load_model_test(
        comparison, **_setting('parameters'), pmin=ANALYSIS['pmin'], metric=metric, epoch=epoch, return_data=True)
    diff = table.difference(metric, 'model', 'test', 'baseline', 'subject', data=data)
    diff[f'roi_{metric}'] = diff[metric].mean(sensor=roi)
    return data, diff, result


def compute_unique_contribution(full_model, predictor, epoch, roi=None, metric=None):
    "Omission comparison: does `predictor` uniquely improve on `full_model` with it removed?"
    return _model_test(f"{full_model} @ {predictor}", epoch, roi, metric)


def compute_direct_comparison(model_a, model_b, epoch, roi=None, metric=None):
    "Direct, non-nested comparison: does model_a fit better than model_b?"
    return _model_test(f"{model_a} > {model_b}", epoch, roi, metric)


def show_topomap_trio(data, diff, result, label, epoch, predictor, metric=None):
    "Group-averaged topomaps: the test model's own fit, the baseline model's own fit, and their significance-masked difference"
    metric = _setting('metric', metric)
    pmin = ANALYSIS['pmin']
    pct = metric == 'ev'
    topo_kwargs = dict(pct=pct, condition=epoch, predictor=predictor, colorbar_side='left', colorbar_label=False, **ANALYSIS['topo_args'])
    topomap_with_colorbar(metric, data.sub("model == 'test'"), label=f'{label}: test', **topo_kwargs)
    topomap_with_colorbar(metric, data.sub("model == 'baseline'"), label=f'{label}: baseline', **topo_kwargs)
    # The third panel shows the mean r/ev difference (the actual unique
    # contribution), masked by significance - NOT result's own
    # masked_parameter_map() directly, which holds the test's underlying
    # t-values, not the r/ev difference reported everywhere else. So:
    # take the real mean difference, and only borrow the *mask* - which
    # sensors were significant - from the test result.
    mean_diff = diff[metric].mean('case')
    significant = numpy.ma.getmaskarray(result.masked_parameter_map(pmin=pmin).x)
    masked_diff = NDVar(numpy.ma.MaskedArray(mean_diff.x, mask=significant), mean_diff.dims, mean_diff.name, mean_diff.info)
    topomap_with_colorbar(None, masked_diff, label=f'{label}: test - baseline', **topo_kwargs)


def show_subject_swarm(diff, label, epoch, metric=None):
    "One dot per subject: their own ROI-averaged test-minus-baseline score (Brodbeck Fig 3B/4B style)"
    import matplotlib.pyplot as plt
    import seaborn as sns

    metric = _setting('metric', metric)
    fig, ax = plt.subplots(figsize=(2, 2.5))
    sns.swarmplot(y=diff[f'roi_{metric}'].x, ax=ax, size=5, color='.25')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_ylabel(metric)
    ax.set_xticks([])
    ax.set_title(f'{label} ({epoch})', fontsize=9)
    fig.tight_layout()
    return fig


# Reading TRFs out of an already-loaded load_trfs() result, and peaks

def taper_prestim(trf, crop=True):
    """For display only: fade the TRF in across 0 ms and out at the end of
    the fit window, then crop to the configured plot window.

    The TRF's first lags absorb activity from just outside the -100 ms
    fit edge, and the 50 ms basis smears responses back towards 0 ms. Up
    to taper[0] the TRF is scaled down to `taper_floor` (keeping its own
    small fluctuations), then a half-Hann ramp brings it to full strength
    at taper[1]; the matching fade-out removes the dip at the other edge. The fitted model
    is untouched, and peaks measured from 40 ms on are unaffected.
    """
    times = trf.time.times
    ramp = numpy.ones(len(times))
    start, stop = ANALYSIS['taper']
    floor = ANALYSIS['taper_floor']
    ramp[times <= start] = floor
    rise = (times > start) & (times < stop)
    ramp[rise] = floor + (1 - floor) * 0.5 * (1 - numpy.cos(numpy.pi * (times[rise] - start) / (stop - start)))
    end_taper = ANALYSIS['end_taper']
    fall_start = times[-1] - end_taper
    fall = times > fall_start
    ramp[fall] *= 0.5 * (1 + numpy.cos(numpy.pi * (times[fall] - fall_start) / end_taper))
    out = trf * NDVar(ramp, (trf.time,))
    return out.sub(time=ANALYSIS['plot_window']) if crop else out


def extract_strf(data_full, column, roi=None):
    "Group-average (frequency, time) STRF for one predictor column, read from an already-loaded load_trfs() result"
    roi = _setting('roi', roi)
    return taper_prestim(data_full[column].mean(sensor=roi).mean('case'))


def extract_trf(data_full, column, roi=None):
    "Per-subject, ROI-averaged, frequency-summed, smoothed TRF for one predictor column, read from an already-loaded load_trfs() result"
    roi = _setting('roi', roi)
    trf = taper_prestim(data_full[column].sum('frequency').mean(sensor=roi), crop=False)
    ds = data_full['subject',]
    # Crop only after smoothing, so resample() never sees a cut edge
    ds['trf_sm'] = resample(trf, 512).sub(time=ANALYSIS['plot_window'])
    return ds


def extract_attention_effect(data, attend_column, ignore_column, roi=None):
    "Per-subject attend - ignore TRF difference, ROI-averaged and frequency-summed, with the same display taper as the other TRF plots"
    roi = _setting('roi', roi)
    attend_trf = data[attend_column].sum('frequency').mean(sensor=roi)
    ignore_trf = data[ignore_column].sum('frequency').mean(sensor=roi)
    return taper_prestim(attend_trf - ignore_trf)


def add_peak(ds, name, window, kind='max'):
    "Add `{name}_time`/`{name}_amp`: the location/value of ds['trf_sm']'s peak within `window`"
    segment = ds['trf_sm'].sub(time=window)
    if kind == 'max':
        ds[f'{name}_time'] = segment.argmax('time')
        ds[f'{name}_amp'] = segment.max('time')
    else:
        ds[f'{name}_time'] = segment.argmin('time')
        ds[f'{name}_amp'] = segment.min('time')
    return ds


add_peak_1 = add_peak


def add_peak_2(ds, name, window, kind='max', halfwidth=0.025, by=None, suffix='_2'):
    """Like add_peak(), but each subject's peak is searched only near the group-average peak.

    The group-average of ds['trf_sm'] is searched for its largest (kind='max')
    or smallest (kind='min') value inside `window`: the anchor. Each subject's
    peak is then the largest/smallest value within `halfwidth` seconds of the
    anchor (and inside `window`), so a subject's peak cannot jump to an
    unrelated bump far from where the average waveform peaks.

    Adds `{name}_time{suffix}`, `{name}_amp{suffix}` and `{name}_edge{suffix}`
    (1 where the subject's peak sits on the edge of its search range, i.e. the
    waveform had no local peak there) and stores the anchor time in seconds in
    ds.info['{name}_anchor{suffix}'] (one entry per level of `by`).

    by=None pools every case in `ds` into one anchor, so conditions share a
    search range: no condition can pull its own range towards its own peak,
    but a condition peaking more than `halfwidth` from the shared anchor is cut
    off. Pass a factor name (e.g. by='role') to give each level its own anchor
    when latencies differ a lot between levels.
    """
    import warnings
    trf = ds['trf_sm']
    x = trf.get_data(('case', 'time'))
    times = trf.time.times
    sign = 1 if kind == 'max' else -1
    in_window = (times >= window[0]) & (times <= window[1])
    if not in_window.any():
        raise ValueError(f"window {window} contains no time points of the TRF")
    first_in, last_in = in_window.argmax(), len(times) - 1 - in_window[::-1].argmax()
    if by is None:
        groups = [('all', numpy.ones(len(x), bool))]
    else:
        groups = [(str(cell), numpy.asarray(ds[by] == cell)) for cell in ds[by].cells]
    anchors = numpy.empty(len(x))
    anchor_info = {}
    for label, index in groups:
        grand = numpy.where(in_window, sign * x[index].mean(axis=0), -numpy.inf)
        i = grand.argmax()
        anchors[index] = times[i]
        anchor_info[label] = float(times[i])
        if i in (first_in, last_in):
            warnings.warn(f"{name} anchor ({label}) at {times[i] * 1e3:.0f} ms is on the edge of the search window {window}")
    allowed = in_window[None, :] & (numpy.abs(times[None, :] - anchors[:, None]) <= halfwidth)
    idx = numpy.where(allowed, sign * x, -numpy.inf).argmax(axis=1)
    rows = numpy.arange(len(x))
    first = allowed.argmax(axis=1)
    last = allowed.shape[1] - 1 - allowed[:, ::-1].argmax(axis=1)
    ds[f'{name}_time{suffix}'] = Var(times[idx])
    ds[f'{name}_amp{suffix}'] = Var(x[rows, idx])
    ds[f'{name}_edge{suffix}'] = Var(((idx == first) | (idx == last)).astype(int))
    ds.info[f'{name}_anchor{suffix}'] = anchor_info
    return ds


def compare_peak_methods(data, names=('positive', 'negative'), group=None, suffix='_2'):
    """Side by side: add_peak_1() columns (no suffix) vs add_peak_2() columns (`suffix`) already in `data`.

    Prints each method's mean/SD of peak time and mean amplitude, how many
    subjects' method-2 peaks sit on the edge of their search range, and a
    paired t-test between the methods. With `group` (e.g. 'role'), also runs
    the usual pairwise tests on peak time and amplitude, once per method.
    """
    from IPython.display import display
    from eelbrain import test
    from scipy import stats
    methods = (('1', ''), ('2', suffix))
    print(f"{'peak':<10}{'method':<8}{'time (ms)':>11}{'SD':>7}{'amp':>12}{'edge':>6}")
    for name in names:
        values = {}
        for method, sfx in methods:
            t = data[f'{name}_time{sfx}'].x * 1e3
            a = data[f'{name}_amp{sfx}'].x
            edge = int(data[f'{name}_edge{sfx}'].x.sum()) if sfx else '-'
            values[method] = (t, a)
            print(f"{name:<10}{method:<8}{t.mean():>11.1f}{t.std(ddof=1):>7.1f}{a.mean():>12.4g}{edge:>6}")
        (t1, a1), (t2, a2) = values['1'], values['2']
        if numpy.allclose(t1, t2) and numpy.allclose(a1, a2):
            print(f"{'':<10}method 1 vs 2: identical for every subject")
            continue
        rt, ra = stats.ttest_rel(t1, t2), stats.ttest_rel(a1, a2)
        print(f"{'':<10}method 1 vs 2 (paired): time t={rt.statistic:.2f} p={rt.pvalue:.3f}; amp t={ra.statistic:.2f} p={ra.pvalue:.3f}")
    if group:
        for name in names:
            for measure in ('time', 'amp'):
                for method, sfx in methods:
                    display(test.pairwise(f'{name}_{measure}{sfx}', group, match='subject', data=data, corr=False,
                                          title=f'{name} peak {measure} by {group}, method {method}'))


def print_peak_table(data, kind, group):
    "Per-subject table of one peak's time (ms) and amplitude, broken down by `group` (e.g. 'role'/'masking')"
    print(f"\n{kind} peak")
    print(f"{'subject':<10}{group:<10}{'time (ms)':>12}{'amp':>14}")
    for subject, g, t, amp in zip(data['subject'], data[group], data[f'{kind}_time'], data[f'{kind}_amp']):
        print(f"{subject:<10}{g:<10}{t * 1e3:>12.0f}{amp:>14.4g}")
