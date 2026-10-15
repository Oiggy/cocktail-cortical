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
