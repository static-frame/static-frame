import numpy as np
import typing_extensions as tp

import static_frame as sf
from static_frame.core.validate import check_interface

TFrameAny = sf.Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]] # type: ignore[type-arg]
TFrameGOAny = sf.FrameGO[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]] # type: ignore[type-arg]
TFrameHEAny = sf.FrameHE[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]] # type: ignore[type-arg]

def test_frame_from_dict() -> None:

    d: tp.Dict[int, tp.Tuple[bool, ...]] = {10: (False, True,), 20: (True, False)}
    f: TFrameAny = sf.Frame.from_dict(d)
    assert f.shape == (2, 2)

def test_frame_from_dict_fields() -> None:

    d1 = {'a': 1, 'b':10, 'c': 5}
    d2 = {'b': 10, 'c': 5, 'a': 1}

    f: TFrameGOAny= sf.FrameGO.from_dict_fields((d1, d2))
    assert f.shape == (3, 2)

def test_frame_from_dict_records() -> None:

    d1 = {'a': 1, 'b':10, 'c': 5}
    d2 = {'b': 10, 'c': 5, 'a': 1}

    f: TFrameGOAny= sf.FrameGO.from_dict_records((d1, d2))
    assert f.shape == (2, 3)


def test_frame_from_records_items() -> None:

    d1 = {'a': (1, 2, 3), 'b': (10, 20, 30), 'c': (5, 5, 5)}

    f: TFrameGOAny= sf.FrameGO.from_records_items(d1.items())
    assert f.shape == (3, 3)


def test_frame_getitem_a() -> None:
    records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            )
    f1: TFrameAny = sf.Frame.from_records(records)
    s1: sf.Series[sf.Index[np.int64], np.str_] = f1[2]
    f2: TFrameAny = f1[[0, 2]]
    f3: TFrameAny = f1[f1.columns.values == 2]
    f4: TFrameAny = f1[1:]


def test_frame_getitem_b() -> None:
    records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            )
    f1: TFrameGOAny = sf.FrameGO.from_records(records)
    f2: TFrameGOAny = f1[2:]
    assert isinstance(f2, sf.FrameGO)

    f3: TFrameGOAny = f1[f1.columns.values % 2 == 0]
    assert isinstance(f3, sf.FrameGO)

def test_frame_getitem_c() -> None:
    records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            )
    f1: TFrameHEAny = sf.FrameHE.from_records(records)
    f2: TFrameHEAny = f1[2:]
    assert isinstance(f2, sf.FrameHE)

    f3: TFrameHEAny = f1[f1.columns.values % 2 == 0]
    assert isinstance(f3, sf.FrameHE)

def test_frame_getitem_d() -> None:
    records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            )
    f1: TFrameAny = sf.Frame.from_records(records, columns=('a', 'b', 'c', 'd'))
    s1: sf.Series[sf.Index[np.int64], np.str_] = f1['c']
    f2: TFrameAny = f1['c':]
    f3: TFrameAny = f1[['b', 'd']]


def test_frame_iloc_a() -> None:
    records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            )
    f1: TFrameAny = sf.Frame.from_records(records)
    s1: sf.Series[sf.Index[np.int64], np.object_] = f1.iloc[2]
    s2: sf.Series[sf.Index[np.int64], np.int64] = f1.iloc[:, 1]
    s3: sf.Series[sf.Index[np.int64], np.object_] = f1.iloc[0, :]
    s4: sf.Series[sf.Index[np.int64], np.int64] = f1.iloc[[0, 1], 1]
    s5: sf.Series[sf.Index[np.int64], np.object_] = f1.iloc[0, [1, 2]]
    s6: sf.Series[sf.Index[np.int64], np.int64] = f1.iloc[1:, 1]
    s7: sf.Series[sf.Index[np.int64], np.object_] = f1.iloc[0, 1:]

    f2: TFrameAny = f1.iloc[[0,2], [0, 1]]
    f3: TFrameAny = f1.iloc[0:1, 1:]

    f4: TFrameAny = f1.iloc[f1.index.values == 2]


def test_frame_loc_a() -> None:
    records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            )
    f1: TFrameAny = sf.Frame.from_records(records)
    s1: sf.Series[sf.Index[np.int64], np.object_] = f1.loc[2]
    s2: sf.Series[sf.Index[np.int64], np.int64] = f1.loc[:, 1]
    s3: sf.Series[sf.Index[np.int64], np.object_] = f1.loc[0, :]
    s4: sf.Series[sf.Index[np.int64], np.int64] = f1.loc[[0, 1], 1]
    s5: sf.Series[sf.Index[np.int64], np.object_] = f1.loc[0, [1, 2]]

    f2: TFrameAny = f1.loc[[0, 2], [0, 1]]
    f3: TFrameAny = f1.loc[0:1, 1:]

    f4: TFrameAny = f1.loc[f1.index.values == 2]



def test_frame_go_loc_a() -> None:
    records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            )
    f1: TFrameGOAny = sf.FrameGO.from_records(records)
    f2: TFrameGOAny = f1.loc[1:, 1:]
    assert isinstance(f2, sf.FrameGO)


def test_frame_he_loc_a() -> None:
    records = (
            (1, 2, 'a', False),
            (30, 34, 'b', True),
            (54, 95, 'c', False),
            )
    f1: TFrameHEAny = sf.FrameHE.from_records(records)
    f2: TFrameHEAny = f1.loc[1:, 1:]
    assert isinstance(f2, sf.FrameHE)


def test_frame_astype_a() -> None:
    records = (
            (1, 2, False),
            (30, 34,True),
            (54, 95, False),
            )
    f1: TFrameAny = sf.Frame.from_records(records)
    f2: TFrameAny = f1.astype(int)


#-------------------------------------------------------------------------------
def test_frame_interface_a() -> None:
    records = (
            (1, 2, False),
            (30, 34,True),
            (54, 95, False),
            )
    h1 = sf.Frame[sf.Index[np.int64], sf.Index[np.str_], np.int64, np.int64, np.bool_] # type: ignore[type-arg]
    f: h1 = sf.Frame.from_records(records,
        columns=('a', 'b', 'c'),
        index=sf.Index((10, 20, 30), dtype=np.int64),
        dtypes=(np.int64, np.int64, np.bool_),
        )
    @check_interface
    def proc1(f: h1) -> sf.Series[sf.Index[np.int64], np.int64]:
        return f['a']

    s1 = proc1(f)

    # h2 = sf.Frame[sf.Index[np.int64], sf.Index[np.str_], np.int64, np.bool_] # type: ignore[type-arg]
    # @check_interface
    # def proc2(f: h2) -> sf.Series[sf.Index[np.int_], np.str_]:
    #     return f['a']

    # s2 = proc2(f) # pyright: Type parameter "TVDtypes@Frame" is invariant, but "*tuple[int_, int_, bool_]" is not the same as "*tuple[int_, bool_]" (reportGeneralTypeIssues)


def test_frame_interface_b() -> None:

    records = (
            (1, 2, True),
            (30, 34, False),
            (54, 95, True),
            )
    hf1 = sf.Frame[sf.IndexDate, sf.Index[np.str_], np.int_, np.int_, np.bool_] # type: ignore[type-arg]
    hs = sf.Series[sf.IndexDate, np.int_]

    f1: hf1 = sf.Frame.from_records(records, columns=('a', 'b', 'c'), index=sf.IndexDate(('2022-01-03', '2022-02-05', '2018-04-02')))

    def proc(f: hf1) -> hs:
        return f.loc[f['c'], 'b']

    s = proc(f1)  # passes


    # if we define a Frame with a different index type, we can statically check it

    hf2 = sf.Frame[sf.Index[np.int_], sf.Index[np.str_], np.int_, np.int_, np.bool_] # type: ignore[type-arg]
    f2: hf2 = sf.Frame.from_records(records, columns=('a', 'b', 'c'))

    # s = proc(f2)  # pyright: error: Argument of type "Frame[Index[int_], Index[str_], int_, int_, bool_]" cannot be assigned to parameter "f" of type "Frame[IndexDate, Index[str_], int_, int_, bool_]" in function "proc"
    # "Frame[Index[int_], Index[str_], int_, int_, bool_]" is incompatible with "Frame[IndexDate, Index[str_], int_, int_, bool_]"
    #   Type parameter "TVIndex@Frame" is invariant, but "Index[int_]" is not the same as "IndexDate" (reportGeneralTypeIssues)

    # if we define a Frame with different column typing, we can statically check it

    hf3 = sf.Frame[sf.IndexDate, sf.Index[np.str_], np.int_, np.bool_] # type: ignore[type-arg]
    f3: hf3 = sf.Frame.from_records((r[1:] for r in records), columns=('b', 'c'), index=sf.IndexDate(('2022-01-03', '2022-02-05', '2018-04-02')))

    # s = proc(f3) #pyright: error: Argument of type "Frame[IndexDate, Index[str_], int_, bool_]" cannot be assigned to parameter "f" of type "Frame[IndexDate, Index[str_], int_, int_, bool_]" in function "proc"
    # "Frame[IndexDate, Index[str_], int_, bool_]" is incompatible with "Frame[IndexDate, Index[str_], int_, int_, bool_]"
    #   Type parameter "TVDtypes@Frame" is invariant, but "*tuple[int_, bool_]" is not the same as "*tuple[int_, int_, bool_]" (reportGeneralTypeIssues)


def test_frame_interface_c() -> None:


    records1 = (
            (1, True),
            (30, False),
            (54, True),
            )
    records2 = (
            (1, 3, 20, True),
            (30, 4, 100, False),
            (54, 3, 8, True),
            )
    records3 = (
            (True, 3),
            (False, 20),
            (True, 3),
            )

    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            np.int_,
            np.bool_]

    h2 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            np.int_,
            np.int_,
            np.int_,
            np.bool_]
    h3 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            np.bool_,
            np.int_,
            ]

    index = sf.IndexDate(('2022-01-03', '2022-02-05', '2018-04-02'))
    f1: h1 = sf.Frame.from_records(records1, columns=('a', 'b'), index=index)
    f2: h2 = sf.Frame.from_records(records2, columns=('a', 'b', 'c', 'd'), index=index)
    f3: h3 = sf.Frame.from_records(records3, columns=('a', 'd'), index=index)

    hflex = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[np.int_, ...]],
            np.bool_]

    fflex1: hflex = f1
    fflex2: hflex = f2

    # fflex3: hflex = f3 # error: Expression of type "Frame[IndexDate, Index[str_], bool_, int_]" cannot be assigned to declared type "Frame[IndexDate, Index[str_], *tuple[int_, ...], bool_]"
    # "Frame[IndexDate, Index[str_], bool_, int_]" is incompatible with "Frame[IndexDate, Index[str_], *tuple[int_, ...], bool_]"
    #   Type parameter "TVDtypes@Frame" is invariant, but "*tuple[bool_, int_]" is not the same as "*tuple[*tuple[int_, ...], bool_]" (reportGeneralTypeIssues)




