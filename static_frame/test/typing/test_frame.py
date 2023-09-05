import typing as tp

import static_frame as sf
from static_frame.core.util import TLabel

def test_frame_from_dict() -> None:

    d: tp.Dict[int, tp.Tuple[bool, ...]] = {10: (False, True,), 20: (True, False)}
    f = sf.Frame.from_dict(d)
    assert f.shape == (2, 2)

def test_frame_from_dict_fields() -> None:

    d1 = {'a': 1, 'b':10, 'c': 5}
    d2 = {'b': 10, 'c': 5, 'a': 1}

    f: sf.FrameGO = sf.FrameGO.from_dict_fields((d1, d2))
    assert f.shape == (3, 2)

def test_frame_from_dict_records() -> None:

    d1 = {'a': 1, 'b':10, 'c': 5}
    d2 = {'b': 10, 'c': 5, 'a': 1}

    f: sf.FrameGO = sf.FrameGO.from_dict_records((d1, d2))
    assert f.shape == (2, 3)