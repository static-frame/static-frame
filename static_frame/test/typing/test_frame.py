import typing as tp

import static_frame as sf
from static_frame.core.util import TLabel

def test_frame_from_dict() -> None:

    d: tp.Dict[int, tp.Tuple[bool, ...]] = {10: (False, True,), 20: (True, False)}
    f = sf.Frame.from_dict(d)
    assert f.shape == (2, 2)


