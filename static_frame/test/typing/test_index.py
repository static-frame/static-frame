import numpy as np
import typing_extensions as tp

import static_frame as sf
from static_frame.core.index_base import IndexBase

TFrameAny = sf.Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]]


def test_index_len_a() -> None:
    idx = sf.Index[np.str_](('a', 'b', 'c'))
    l: int = len(idx)
    assert l == 3


def test_index_len_b() -> None:
    idx: TFrameAny = sf.Frame.from_records([('a', 'b'), ('c', 'd')])
    l: int = len(idx.columns)
    assert l == 2


def test_index_c() -> None:
    idx = sf.Index[np.str_](('a', 'b', 'c'))
    assert len(idx) == 3

    x1: str = idx[1]

    x2: sf.Index[np.str_] = idx[1:]
    x3: sf.Index[np.str_] = idx[[0, 1]]
    x4: sf.Index[np.str_] = idx[idx.values == 'b']


def test_index_d() -> None:
    idx = sf.IndexDate(('2021-01-01', '2022-02-03', '2023-05-03'))
    assert len(idx) == 3
    x1: np.datetime64 = idx[1]

    x2: sf.IndexDate = idx[1:]
    x3: sf.IndexDate = idx[[0, 1]]
    x4: sf.IndexDate = idx[idx.values == 'b']
    dt: np.dtype[np.datetime64] = idx.dtype


def test_index_e() -> None:
    idx = sf.IndexHierarchy.from_product(('a', 'b'), (10, 20))
    assert len(idx) == 4

    x1: tp.Tuple[str, int] = idx[1]

    x2: sf.IndexHierarchy = idx[1:]
    x3: sf.IndexHierarchy = idx[[0, 1]]
    x4: sf.IndexHierarchy = idx[idx.values_at_depth(1) == 20]


def test_index_f() -> None:
    idx = sf.Index[np.int64](np.array([4, 1, 0], np.int64))
    x1: np.int64 = idx[1]


def test_index_g() -> None:
    def proc(ib: IndexBase) -> int:
        return len(tuple(ib.iter_label()))

    idx1 = sf.Index[np.int64](np.array([1, 2, 3], np.int64))
    idx2 = sf.IndexDate(('2022-01-01', '2023-06-15', '2024-12-31'))

    x1: int = proc(idx1)
    x2: int = proc(idx2)


def test_index_go_c() -> None:
    idx = sf.IndexGO[np.str_](('a', 'b', 'c'))
    assert len(idx) == 3

    x1: str = idx[1]

    x2: sf.IndexGO[np.str_] = idx[1:]
    x3: sf.IndexGO[np.str_] = idx[[0, 1]]
    x4: sf.IndexGO[np.str_] = idx[idx.values == 'b']
