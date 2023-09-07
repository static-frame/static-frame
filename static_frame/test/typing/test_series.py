import typing as tp

import static_frame as sf


def test_series_from_dict() -> None:
    d: tp.Dict[str, int] = {'a': 1, 'b': 20, 'c': 300}
    s = sf.Series.from_dict(d)
    assert len(s) == 3


def test_series_getitem() -> None:
    s = sf.Series((10, 20, 30), index=('a', 'b', 'c'))

    v1: int = s['b']
    v2: sf.Series = s['b':]

    def proc(x: sf.Series) -> sf.Series:
        return x.dropna()

    y = proc(s['b':]) # this checks that x.dropna() is on a Series



def test_series_drop() -> None:
    s1 = sf.Series((10, 20, 30), index=('a', 'b', 'c'))
    s2: sf.Series = s1.drop['b'] # dropping always returns a series

    def proc(x: sf.Series) -> sf.Series:
        return x.dropna()

    y = proc(s1.drop['b'])