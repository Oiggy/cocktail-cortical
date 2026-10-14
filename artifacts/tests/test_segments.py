import ast

import mne
import pytest

from artifacts import config
from artifacts.preprocessing import segments as seg


def _experiment_constants():
    """SEGMENT_DURATION, SPEAKER, SPATIAL, SIDE as defined in
    analysis/experiment.py, read without importing it (that would need
    eelbrain's in-development Pipeline)."""
    tree = ast.parse((config.REPO_ROOT / 'analysis' / 'experiment.py').read_text())
    wanted = {'SEGMENT_DURATION', 'SPEAKER', 'SPATIAL', 'SIDE'}
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], 'id', None) in wanted:
            found[node.targets[0].id] = eval(compile(ast.Expression(node.value), 'experiment.py', 'eval'), {})
    return found


def test_constants_match_experiment_py():
    constants = _experiment_constants()
    assert constants.keys() == {'SEGMENT_DURATION', 'SPEAKER', 'SPATIAL', 'SIDE'}
    for name, value in constants.items():
        assert getattr(seg, name) == value, f'{name} drifted from analysis/experiment.py'


@pytest.mark.parametrize('list_id', [1, 2])
def test_segments_from_events(list_id):
    onsets = [10. + 300 * i for i in range(12)]
    # a code-8 trigger in between must be ignored, like label_events() does
    segments = seg.segments_from_events(onsets[:3] + [999.] + onsets[3:], [list_id] * 3 + [8] + [list_id] * 9)
    assert len(segments) == 12
    assert [s.condition for s in segments] == seg.SPATIAL
    first, second = segments[0], segments[1]
    fg_speaker, bg_speaker = ('male', 'female') if list_id == 1 else ('female', 'male')
    assert first.fg == f'{fg_speaker}_1' and first.bg == '' and first.condition == 'clean'
    assert second.fg == f'{fg_speaker}_2' and second.bg == f'{bg_speaker}_2'
    assert second.side == ('right' if list_id == 1 else 'left')
    assert first.duration == seg.SEGMENT_DURATION[first.fg]
    assert segments[6].bg == ''  # segment 7 is the second clean one


def test_wrong_number_of_onsets():
    with pytest.raises(ValueError, match='expected 12'):
        seg.segments_from_events([1., 2.], [1, 1])


def test_from_annotations():
    descriptions = ['code_2'] * 12 + ['code_8']
    annotations = mne.Annotations(onset=[5. + 260 * i for i in range(13)], duration=[0.] * 13, description=descriptions)
    segments = seg.segments_from_annotations(annotations)
    assert len(segments) == 12 and segments[0].fg == 'female_1'
    table = seg.segments_table(segments)
    assert list(table['offset']) == [s.onset + s.duration for s in segments]
