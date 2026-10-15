import pytest

from adr.io.dataset1 import available_subjects, load_subject

subjects = available_subjects()
pytestmark = pytest.mark.skipif(not subjects, reason="Dataset I not available locally")


def test_first_subject_structure():
    trials = load_subject(subjects[0])
    assert len(trials) == 20
    t = trials[0]
    assert t.sfreq == 128
    assert t.eeg.shape[1] == 64
    assert t.ch_names[47] == "Cz" and t.ch_names[13] == "C5"
    assert {tr.condition for tr in trials} == {"hrtf", "dry"}
    assert sum(tr.repetition for tr in trials) == 12
    assert all(name.endswith("_dry.wav") for tr in trials for name in tr.dry_stimuli)


def test_stimulus_key():
    from adr.io.dataset1 import stimulus_key
    assert stimulus_key("part3_track2_hrtf.wav") == (3, 2)
    assert stimulus_key("rep_part3_track2_dry.wav") == (3, 2)
    with pytest.raises(ValueError):
        stimulus_key("male_1.wav")


def test_presentation_groups_merge_shared_stimuli():
    from adr.io.dataset1 import presentation_groups
    trials = load_subject(subjects[0])
    groups = presentation_groups(trials)
    by_part = {}
    for trial, group in zip(trials, groups):
        parts = {int(s.split("part")[1].split("_")[0]) for s in trial.stimuli}
        assert len(parts) == 1  # each trial plays both tracks of one part
        by_part.setdefault(parts.pop(), set()).add(group)
    # Experiment 1, experiment 2 and the repetitions of one part share
    # audio, so each story part is exactly one group, and groups differ
    # between parts.
    assert by_part == {part: {part} for part in (1, 2, 3, 4)}
