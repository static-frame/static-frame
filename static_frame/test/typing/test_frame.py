import numpy as np
import typing_extensions as tp

import static_frame as sf


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


def test_frame_from_records_items() -> None:

    d1 = {'a': (1, 2, 3), 'b': (10, 20, 30), 'c': (5, 5, 5)}

    f: sf.FrameGO = sf.FrameGO.from_records_items(d1.items())
    assert f.shape == (3, 3)


def test_frame_getitem_a() -> None:
    records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            )
    f1 = sf.Frame.from_records(records)
    s1: sf.Series[sf.Index[np.int64], np.str_] = f1[2]
    f2: sf.Frame = f1[[0, 2]]
    f3: sf.Frame = f1[f1.columns.values == 2]
    f4: sf.Frame = f1[1:]


def test_frame_getitem_b() -> None:
    records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            )
    f1 = sf.FrameGO.from_records(records)
    f2: sf.FrameGO = f1[2:]
    assert isinstance(f2, sf.FrameGO)

    f3: sf.FrameGO = f1[f1.columns.values % 2 == 0]
    assert isinstance(f3, sf.FrameGO)

def test_frame_getitem_c() -> None:
    records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            )
    f1 = sf.FrameHE.from_records(records)
    f2: sf.FrameHE = f1[2:]
    assert isinstance(f2, sf.FrameHE)

    f3: sf.FrameHE = f1[f1.columns.values % 2 == 0]
    assert isinstance(f3, sf.FrameHE)

def test_frame_getitem_d() -> None:
    records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            )
    f1 = sf.Frame.from_records(records, columns=('a', 'b', 'c', 'd'))
    s1: sf.Series[sf.Index[np.int64], np.str_] = f1['c']
    f2: sf.Frame = f1['c':]
    f3: sf.Frame = f1[['b', 'd']]


def test_frame_iloc_a() -> None:
    records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            )
    f1 = sf.Frame.from_records(records)
    s1: sf.Series[sf.Index[np.int64], np.object_] = f1.iloc[2]
    s2: sf.Series[sf.Index[np.int64], np.int64] = f1.iloc[:, 1]
    s3: sf.Series[sf.Index[np.int64], np.object_] = f1.iloc[0, :]
    s4: sf.Series[sf.Index[np.int64], np.int64] = f1.iloc[[0, 1], 1]
    s5: sf.Series[sf.Index[np.int64], np.object_] = f1.iloc[0, [1, 2]]
    s6: sf.Series[sf.Index[np.int64], np.int64] = f1.iloc[1:, 1]
    s7: sf.Series[sf.Index[np.int64], np.object_] = f1.iloc[0, 1:]

    f2: sf.Frame = f1.iloc[[0,2], [0, 1]]
    f3: sf.Frame = f1.iloc[0:1, 1:]

    f4: sf.Frame = f1.iloc[f1.index.values == 2]


def test_frame_loc_a() -> None:
    records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            )
    f1 = sf.Frame.from_records(records)
    s1: sf.Series[sf.Index[np.int64], np.object_] = f1.loc[2]
    s2: sf.Series[sf.Index[np.int64], np.int64] = f1.loc[:, 1]
    s3: sf.Series[sf.Index[np.int64], np.object_] = f1.loc[0, :]
    s4: sf.Series[sf.Index[np.int64], np.int64] = f1.loc[[0, 1], 1]
    s5: sf.Series[sf.Index[np.int64], np.object_] = f1.loc[0, [1, 2]]

    f2: sf.Frame = f1.loc[[0, 2], [0, 1]]
    f3: sf.Frame = f1.loc[0:1, 1:]

    f4: sf.Frame = f1.loc[f1.index.values == 2]



def test_frame_go_loc_a() -> None:
    records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            )
    f1 = sf.FrameGO.from_records(records)
    f2: sf.FrameGO = f1.loc[1:, 1:]
    assert isinstance(f2, sf.FrameGO)


def test_frame_he_loc_a() -> None:
    records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            )
    f1 = sf.FrameHE.from_records(records)
    f2: sf.FrameHE = f1.loc[1:, 1:]
    assert isinstance(f2, sf.FrameHE)


def test_frame_astype_a() -> None:
    records = (
            (1, 2, False),
            (30, 34,True),
            (54, 95, False),
            )
    f1 = sf.Frame.from_records(records)
    f2 = f1.astype(int)












