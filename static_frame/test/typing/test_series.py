import numpy as np
import typing_extensions as tp

import static_frame as sf


def test_series_from_dict() -> None:
    d: tp.Dict[str, int] = {'a': 1, 'b': 20, 'c': 300}
    s = sf.Series[sf.Index[np.str_], np.int64].from_dict(d)
    assert len(s) == 3


def test_series_getitem_a() -> None:
    s = sf.Series[sf.Index[np.str_], np.int64]((10, 20, 30), index=('a', 'b', 'c'))

    v1: np.int64 = s['b']
    v2: sf.Series[sf.Index[np.str_], np.int64] = s['b':]

    def proc(x: sf.Series[sf.Index[np.str_], np.int64]) -> sf.Series[sf.Index[np.str_], np.int64]:
        return x.dropna()

    y = proc(s['b':]) # this checks that x.dropna() is on a Series


def test_series_getitem_b() -> None:
    s = sf.SeriesHE[sf.Index[np.str_], np.int64]((10, 20, 30), index=('a', 'b', 'c'))

    v1: np.int64 = s['b']
    v2: sf.SeriesHE[sf.Index[np.str_], np.int64] = s['b':]

    def proc(x: sf.SeriesHE[sf.Index[np.str_], np.int64]) -> sf.SeriesHE[sf.Index[np.str_], np.int64]:
        return x.dropna()

    y = proc(s['b':]) # this checks that x.dropna() is on a Series


def test_series_iloc_a() -> None:
    s = sf.Series[sf.Index[np.str_], np.int64]((10, 20, 30), index=('a', 'b', 'c'))

    v1: np.int64 = s.iloc[0]
    v2: sf.Series[sf.Index[np.str_], np.int64] = s.iloc[[0, 2]]
    assert len(v2) == 2

    v3 = s.iloc[1:]
    assert len(v3) == 2

def test_series_he_iloc_a() -> None:
    s = sf.SeriesHE[sf.Index[np.str_], np.int64]((10, 20, 30), index=('a', 'b', 'c'))

    v1: np.int64 = s.iloc[0]
    v2: sf.SeriesHE[sf.Index[np.str_], np.int64] = s.iloc[[0, 2]]
    assert len(v2) == 2

    v3: sf.SeriesHE[sf.Index[np.str_], np.int64] = s.iloc[1:]
    assert len(v3) == 2


def test_series_drop() -> None:
    TSeries = sf.Series[sf.Index[np.str_], np.int64]
    s1: TSeries = sf.Series((10, 20, 30), index=('a', 'b', 'c'))
    s2: TSeries = s1.drop['b'] # dropping always returns a series

    def proc1(x: TSeries) -> TSeries:
        return x.dropna()

    y1 = proc1(s1.drop['b'])


    def proc2(x: sf.Series[sf.Index[np.str_], np.str_]) -> sf.Series[sf.Index[np.str_], np.str_]:
        return x.dropna()

    # y2 = proc2(s1)  #error: Argument 1 to "proc2" has incompatible type "Series[Index[str_], signedinteger[_64Bit]]"; expected "Series[Index[str_], str_]"  [arg-type]



def test_series_ih1() -> None:
    TSeries1 = sf.Series[sf.IndexDate, np.float64]
    TSeries2 = sf.Series[sf.Index[np.int64], np.float64]

    s1: TSeries1 = sf.Series((10, 20, np.nan), index=sf.IndexDate(('2022-01-01', '2022-01-02', '2022-01-03')))


    def proc1(x: TSeries1) -> TSeries1:
        return x.dropna()

    y1 = proc1(s1)

    def proc2(x: TSeries2) -> TSeries2:
        return x.dropna()

    #y1 = proc2(s1) # error: Argument of type "Series[IndexDate, float64]" cannot be assigned to parameter "x" of type "Series[Index[int64], float64]" in function "proc2"


# NOTE: tested and shown that neither beartype or typeguard work for these nested containers
# def test_alt_checkers() -> None:

#     from beartype.door import die_if_unbearable
#     from typeguard import check_type

#     TSeries1 = sf.Series[sf.Index[np.str_], np.int64]
#     TSeries2 = sf.Series[sf.Index[np.str_], np.str_]

#     s1: TSeries1 = sf.Series((10, 20, 30), index=('a', 'b', 'c'))

#     check_type(s1, TSeries1)
#     check_type(s1, TSeries2)

#     die_if_unbearable(s1, TSeries1)
#     die_if_unbearable(s1, TSeries2)

