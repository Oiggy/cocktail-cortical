import pandas as pd
import pytest

from adr.io import physiomotion as pm


def test_label_classes():
    assert pm.label_classes("blink") == {"ocular"}
    assert pm.label_classes("blink_hor_headm") == {"ocular", "motion"}
    assert pm.label_classes("swallow_eyebrow") == {"muscle"}
    assert pm.label_classes("open_base") == frozenset()
    with pytest.raises(ValueError):
        pm.label_classes("sneeze")


def test_window_labels_or_vs_strict():
    ann = pd.DataFrame([
        ("ALL", 0.0, 10.0, "close_base"),
        ("Fp1-F7", 12.0, 13.0, "blink"),     # one pair
        ("Fp1-F7", 20.0, 21.0, "chew"),      # two pairs
        ("F7-T7", 20.0, 21.0, "chew"),
        ("ALL", 30.0, 31.0, "hor_headm"),    # ALL = every pair
    ], columns=["channel", "start", "stop", "label"])
    win = pm.window_labels(ann, duration=40, window=1.0, step=0.5)
    at = lambda t: win.loc[win.start == t].iloc[0]
    assert at(2.0)["rest"] and not at(2.0)["unlabelled"]
    assert at(12.0)["ocular"] and at(20.0)["muscle"] and at(30.0)["motion"]
    assert at(15.0)["unlabelled"]          # task period, not annotated
    assert not at(9.5)["rest"]             # straddles the end of the rest block
    strict = pm.window_labels(ann, duration=40, window=1.0, step=0.5, min_pairs=2)
    s_at = lambda t: strict.loc[strict.start == t].iloc[0]
    assert not s_at(12.0)["ocular"]        # single pair fails the >= 2 rule
    assert s_at(20.0)["muscle"] and s_at(30.0)["motion"]


try:
    HAVE_DATA = pm.eeg_path(1, 1).exists()
except Exception:
    HAVE_DATA = False


@pytest.mark.skipif(not HAVE_DATA, reason="PhysioMotion not available locally")
def test_real_run():
    from adr.preprocessing.channels import detector_channels
    raw = pm.load_raw(1, 1, preload=False)
    assert raw.info["sfreq"] == 1000 and len(raw.ch_names) == 59
    assert set(detector_channels()) <= set(raw.ch_names)
    ann = pm.load_annotations(1, 1)
    for label in ann["label"].unique():
        pm.label_classes(label)            # every label parses
    rest = raw.copy().crop(10, 130).load_data()
    assert "AF4" in pm.bad_channels(rest)  # flat in sub-1 run 1
