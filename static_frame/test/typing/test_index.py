import typing as tp

import numpy as np

import static_frame as sf


def test_index_len_a() -> None:

    idx = sf.Index[np.unicode_](('a', 'b', 'c'))
    l: int = len(idx)
    assert l == 3

def test_index_len_b() -> None:

    idx = sf.Frame.from_records([('a', 'b'), ('c', 'd')])
    l: int = len(idx.columns)
    assert l == 2


def test_index_c() -> None:

    idx = sf.Index[np.unicode_](('a', 'b', 'c'))
    assert len(idx) == 3

    x1: str = idx[1]

    x2: sf.Index[np.unicode_] = idx[1:]
    x3: sf.Index[np.unicode_] = idx[[0, 1]]
    x4: sf.Index[np.unicode_] = idx[idx.values == 'b']


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


def test_index_go_c() -> None:

    idx = sf.IndexGO[np.unicode_](('a', 'b', 'c'))
    assert len(idx) == 3

    x1: str = idx[1]

    x2: sf.IndexGO[np.unicode_] = idx[1:]
    x3: sf.IndexGO[np.unicode_] = idx[[0, 1]]
    x4: sf.IndexGO[np.unicode_] = idx[idx.values == 'b']





