"""
Preprocessing: bad channels and ICA, one subject after another,
automatically.

Why this can't be fully automatic: two things need a human to look at
plots and click, since they can't be known ahead of time from code
alone - which electrodes were noisy for this particular subject, and
which of the automatically-found ICA components are actually eye
blinks/movements for this particular subject. eelbrain's automated
cleanup steps (filtering, re-referencing) are already defined in
`experiment.py` and don't need any of this.

What this script automates: moving between subjects. Both
make_bad_channels() and make_ica_selection() open a plot and pause the
script until you close it, so this loop advances to the next subject
by itself the moment you're done with the current one, there is
nothing to edit or re-run by hand between subjects.

Both steps save their result to that subject's files under
`bids/sub-XX/eeg/` automatically. Once saved, a subject is never
reprocessed, running this again on an already-done subject is a no-op
(comment out SUBJECTS_TO_SKIP below if you want to force a redo).

Run this after data/bids_extraction.py, before analysis/cortical_analysis.ipynb.
"""
from experiment import e

SUBJECTS = [f'sub-{i:02d}' for i in range(1, 14)]  # sub-01 .. sub-13

# List any subjects you've already finished, to pick up where you left
# off instead of starting over from sub-01.
SUBJECTS_TO_SKIP = []

for subject in SUBJECTS:
    if subject in SUBJECTS_TO_SKIP:
        continue

    print(f"\n=== {subject} ===")
    e.set(subject=subject)

    # Opens a plot of this subject's raw EEG. Click any electrode that
    # looks consistently noisy/flat/disconnected to mark it bad, then
    # close the plot to continue to the next step.
    e.make_bad_channels()

    # Fits ICA for this subject (if not already cached) and opens a GUI
    # showing each component. Select the ones that are clearly eye
    # blinks or eye movements, then close the plot to move on to the
    # next subject.
    e.make_ica_selection(epoch='clean', decim=16)

print("\nDone. Every subject has bad channels marked and ICA fit.")
