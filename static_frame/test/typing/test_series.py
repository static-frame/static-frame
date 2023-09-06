import typing as tp

import static_frame as sf


def test_series_from_dict() -> None:
    d: tp.Dict[str, int] = {'a': 1, 'b': 20, 'c': 300}
    s = sf.Series.from_dict(d)
    assert len(s) == 3


