from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import typing_extensions as tp

import static_frame as sf
# from static_frame.core.validate import Validator
# from static_frame.core.validate import validate_pair
from static_frame.core.validate import CheckResult
from static_frame.core.validate import Labels
from static_frame.core.validate import Len
from static_frame.core.validate import Name
from static_frame.core.validate import TValidation
from static_frame.core.validate import TypeClinic
from static_frame.core.validate import Validator
from static_frame.core.validate import check_interface
from static_frame.core.validate import check_type
from static_frame.core.validate import is_union
from static_frame.core.validate import is_unpack
from static_frame.test.test_case import skip_pyle310
from static_frame.test.test_case import skip_win


def test_check_type_a():

    check_type(sf.IndexDate(('2022-01-01',)), sf.IndexDate)
    check_type(sf.IndexDate(('2022-01-01',)), tp.Any)

    with pytest.raises(TypeError):
        check_type(sf.IndexDate(('2022-01-01',)), sf.IndexSecond)


def test_check_type_b():

    check_type(3, int)
    check_type('foo', str)
    check_type(False, bool)

    with pytest.raises(TypeError):
        check_type(3, str)

    with pytest.raises(TypeError):
        check_type(True, int)

#-------------------------------------------------------------------------------

def test_is_unpack_a():
    assert is_unpack(tp.Unpack)
    assert not is_unpack(None)

def test_is_union_a():
    assert is_union(tp.Union[int, str])
    assert not is_union(tp.Tuple[str, str])
    assert not is_union(str)

#-------------------------------------------------------------------------------

def test_check_result_a():
    assert CheckResult([]).validated

def test_check_result_b():
    try:
        post = TypeClinic((3, 'x')).check(tp.Tuple[int, str, ...]).to_str()
        assert scrub_str(post) == 'In Tuple[int, str, ...] Invalid ellipses usage'
    except TypeError:
        pass

def test_check_result_c():
    post = TypeClinic(sf.Index(('a', 'b'))).check(tp.Annotated[sf.Index[np.str_], Len(1)]).to_str()
    assert scrub_str(post) == 'In Annotated[Index[str_], Len(1)] Len(1) Expected length 1, provided length 2'


#-------------------------------------------------------------------------------

def test_check_type_union_a():

    check_type(3, tp.Union[int, str])

    with pytest.raises(TypeError):
        check_type('x', tp.Union[int, float])

    check_type('x', tp.Union[str, bytes])
    check_type('x', tp.Union[int, str])


#-------------------------------------------------------------------------------

def test_check_type_type_a():

    check_type(sf.Series, tp.Type[sf.Series])

    with pytest.raises(TypeError):
        check_type(sf.Series, tp.Type[sf.Index])

@skip_pyle310
def test_check_type_type_b():
    try:
        check_type(3, tp.Type[sf.Series])
    except TypeError as e:
        assert str(e).replace('\n', '') == 'Expected Type[Series], provided int invalid'

#-------------------------------------------------------------------------------

@skip_pyle310
def test_validate_numpy_a():
    v = np.array([False, True, False])
    h1 = np.ndarray[tp.Any, np.dtype[np.bool_]]
    h2 = np.ndarray[tp.Any, np.dtype[np.str_]]

    check_type(v, h1)
    with pytest.raises(TypeError):
        check_type(v, h2)


#-------------------------------------------------------------------------------

def test_check_type_containers_a():
    v = sf.Series(('a', 'b'), index=sf.IndexDate(('2021-04-05', '2022-05-03')))
    h1 = sf.SeriesHE[sf.IndexDate, np.str_]
    h2 = sf.Index[np.str_]
    h3 = sf.Series[sf.IndexDate, np.str_]

    with pytest.raises(TypeError):
        check_type(v, h2)

    with pytest.raises(TypeError):
        check_type(v, h1)

def test_check_type_containers_b():
    v = sf.Series(('a', 'b'), index=sf.IndexDate(('2021-04-05', '2022-05-03')))
    h1 = sf.Series[sf.IndexDate, np.str_]
    h2 = sf.Series[sf.IndexDate, np.int64]

    check_type(v, h1)
    with pytest.raises(TypeError):
        check_type(v, h2)


def test_check_type_containers_c():
    v = sf.Series(('a', 'b'), index=sf.Index((10, 20), dtype=np.int64))
    h1 = sf.Series[sf.Index[np.str_], np.str_]
    h2 = sf.Series[sf.IndexDate, np.int64]
    h3 = sf.Series[sf.Index[np.str_], np.int64]
    h4 = sf.Series[sf.Index[np.int64], np.str_]

    with pytest.raises(TypeError):
        check_type(v, h1)
    with pytest.raises(TypeError):
        check_type(v, h2)
    with pytest.raises(TypeError):
        check_type(v, h3)

    check_type(v, h4)


def test_check_type_containers_d():
    v1 = sf.Series(('a', 'b'), index=sf.Index((10, 20), dtype=np.int64))
    v2 = sf.Series(('a', 'b'), index=sf.Index((10, 20), dtype=np.str_))
    v3 = sf.Series(('a', 'b'), index=sf.Index((1, 0), dtype=np.bool_))

    h1 = sf.Series[sf.Index[tp.Union[np.int64, np.str_]], np.str_]

    check_type(v1, h1)
    check_type(v2, h1)
    with pytest.raises(TypeError):
        check_type(v3, h1)


def test_check_type_containers_e():
    v1 = sf.Series(('a', 'b'), index=sf.IndexDate(('2021-04-05', '2022-05-03')))
    v2 = sf.Series(('a', 'b'), index=sf.IndexSecond(('2021-04-05', '2022-05-03')))
    v3 = sf.Series(('a', 'b'), index=sf.Index(('x', 'y')))

    h1 = sf.Series[tp.Union[sf.IndexDate, sf.IndexSecond], np.str_]

    check_type(v1, h1)
    check_type(v2, h1)

    with pytest.raises(TypeError):
        check_type(v3, h1)


#-------------------------------------------------------------------------------

def scrub_str(s: str) -> str:
    s = s.replace('\n', ' '
            ).replace(CheckResult._LINE, ''
            ).replace(CheckResult._CORNER, ''
            ).replace('tuple[', 'Tuple[') # normalize tuple presentation
    while '  ' in s:
        s = s.replace('  ', ' ')
    return s.strip()

@skip_pyle310
def test_check_type_fail_fast_a():
    v = sf.Series(('a', 'b'), index=sf.Index(('x', 'y'), dtype=np.str_))
    h = sf.Series[sf.Index[np.int64], np.int64]


    with pytest.raises(TypeError):
        check_type(v, h, fail_fast=True)
    try:
        check_type(v, h, fail_fast=True)
    except TypeError as e:
        assert scrub_str(str(e)) == 'In Series[Index[int64], int64] Expected int64, provided str_ invalid'

    with pytest.raises(TypeError):
        check_type(v, h, fail_fast=False)
    try:
        check_type(v, h, fail_fast=False)
    except TypeError as e:
        assert scrub_str(str(e)) == 'In Series[Index[int64], int64] Expected int64, provided str_ invalid In Series[Index[int64], int64] Index[int64] Expected int64, provided str_ invalid'

#-------------------------------------------------------------------------------

def test_check_type_sequence_a():
    check_type([3, 4], tp.List[int])

    with pytest.raises(TypeError):
        check_type([3, 4, 'a'], tp.List[int])


    check_type([3, 4, 'a'], tp.List[tp.Union[int, str]])

    check_type(['c', 'b', 'a'], tp.Union[tp.List[int], tp.List[str]])

    with pytest.raises(TypeError):
        check_type([3, 4, 'a', True], tp.List[tp.Union[int, str]])

def test_check_type_sequence_b():

    check_type([3, 4], tp.Sequence[int])

    with pytest.raises(TypeError):
        check_type([3, 4, 'a'], tp.Sequence[int])


    check_type([3, 4, 'a'], tp.Sequence[tp.Union[int, str]])

    check_type(['c', 'b', 'a'], tp.Union[tp.Sequence[int], tp.Sequence[str]])

    with pytest.raises(TypeError):
        check_type([3, 4, 'a', True], tp.Sequence[tp.Union[int, str]])


#-------------------------------------------------------------------------------

def test_check_type_tuple_a():

    with pytest.raises(TypeError):
        check_type([3, 4], tp.Tuple[int, bool])

    with pytest.raises(TypeError):
        check_type((3, False, 'foo'), tp.Tuple[int, ...])

    check_type((3, 4, 5), tp.Tuple[int, ...])
    check_type((3, 4, 5, 3, 20), tp.Tuple[int, ...])
    check_type((3,), tp.Tuple[int, ...])

def test_check_type_tuple_b():

    check_type((3, 4, False), tp.Tuple[int, int, bool])
    check_type((3, 4.1, False), tp.Tuple[int, float, bool])

@skip_pyle310
def test_check_type_tuple_c():

    cr = TypeClinic((3, 4)).check(tp.Tuple[int, int, int])
    assert [r[1] for r in cr] == ['Expected tuple length of 3, provided tuple length of 2']

@skip_pyle310
def test_check_type_tuple_d():

    cr = TypeClinic((3, 4, 5)).check(tp.Tuple[..., int, ...])
    assert [r[1] for r in cr] == ['Invalid ellipses usage']


#-------------------------------------------------------------------------------

@skip_pyle310
def test_check_type_literal_a():
    check_type(42, tp.Literal[42])
    check_type(42, tp.Literal[-1, 42])

    cr = TypeClinic(42).check(tp.Literal['a', 'b'])
    assert list(cr) == [(42, 'a', (tp.Literal['a', 'b'],)),
                        (42, 'b', (tp.Literal['a', 'b'],))]

#-------------------------------------------------------------------------------

def test_check_type_dict_a():
    check_type({'a': 3}, tp.Dict[str, int])
    check_type({'b': 20}, tp.Dict[str, int])

    with pytest.raises(TypeError):
        check_type({'a': 20, 'b': 18, 'c': False}, tp.Dict[str, int])

    with pytest.raises(TypeError):
        check_type({'a': 20, 'b': 18, 20: 3}, tp.Dict[str, int])


#-------------------------------------------------------------------------------
def test_check_interface_a():

    @check_interface(fail_fast=False)
    def proc1(a: int, b: int) -> int:
        return a * b

    assert proc1(2, 3) == 6

def test_check_interface_b():

    @check_interface(fail_fast=False)
    def proc1(a: int, b: int) -> bool:
        return a * b
    try:
        assert proc1(2, 3) == 6
    except TypeError as e:
        assert scrub_str(str(e)) == 'In return of (a: int, b: int) -> bool Expected bool, provided int invalid'

    try:
        assert proc1(2, 'foo') == 6
    except TypeError as e:
        assert scrub_str(str(e)) == 'In args of (a: int, b: int) -> bool Expected int, provided str invalid'

def test_check_interface_c():

    @check_interface(fail_fast=False)
    def proc1(a: int, b) -> int:
        return a * b

    assert proc1(2, False) == 0
    assert proc1(2, 1) == 2

    with pytest.raises(TypeError):
        assert proc1('foo', 1) == 2


def test_check_interface_d():

    @check_interface
    def proc1(a: int, b: int) -> int:
        return a * b

    assert proc1(2, 0) == 0
    assert proc1(2, 1) == 2



def test_check_interface_e():

    @check_interface
    def proc1(a: tp.Annotated[int, 'foo'], b: tp.Annotated[int, 'bar']) -> int:
        return a * b

    assert proc1(2, 0) == 0
    assert proc1(2, 1) == 2


def test_check_interface_f():

    @check_interface

    def proc1(idx: tp.Annotated[sf.Index[np.str_], Len(3), Name('foo')]) -> int:
        return len(idx)

    idx1 = sf.Index(('a', 'b', 'c'), name='foo')
    assert proc1(idx1) == 3

    idx2 = sf.Index(('a', 'b', 'c'), name='fab')
    with pytest.raises(TypeError):
        _ = proc1(idx2)

    idx3 = sf.Index(('a', 'c'), name='fab')
    with pytest.raises(TypeError):
        _ = proc1(idx3)

#-------------------------------------------------------------------------------

def test_check_annotated_a():

    check_type(3, tp.Annotated[int, 'foo'])

def test_check_annotated_b():

    v1 = sf.Series(('a', 'b'), index=sf.Index((10, 20)))
    h1 = tp.Annotated[
            sf.Series[sf.Index[np.int_], np.str_],
            Name('foo'),
    ]
    with pytest.raises(TypeError):
        check_type(v1, h1)

    v1 = sf.Series(('a', 'b'), index=sf.Index((10, 20)), name='foo')
    check_type(v1, h1)

def test_check_annotated_c():

    v1 = sf.Series(('a', 'b'), index=sf.Index((10, 20)))
    h1 = tp.Annotated[
            sf.Series[sf.Index[np.int_], np.str_],
            Len(1),
            ]
    h2 = tp.Annotated[
            sf.Series[sf.Index[np.int_], np.str_],
            Len(2),
            ]

    with pytest.raises(TypeError):
        check_type(v1, h1)

    check_type(v1, h2)

#-------------------------------------------------------------------------------

def test_check_index_hierarchy_a():

    v1 = sf.IndexHierarchy.from_product(('a', 'b'), (1, 2))
    h1 = tp.Annotated[
            sf.IndexHierarchy[sf.Index[np.str_], sf.Index[np.integer]],
            Len(4),
            ]
    check_type(v1, h1)

    h1 = sf.IndexHierarchy[sf.Index[np.str_], sf.Index[np.integer], sf.IndexDate]
    with pytest.raises(TypeError):
        check_type(v1, h1)

def test_check_index_hierarchy_b():

    v1 = sf.IndexHierarchy.from_labels([(1, 100), (1, 200), (2, 100)])
    v2 = sf.IndexHierarchy.from_labels([(1, 100, 3), (1, 200, 3), (2, 100, 3)])

    h1 = sf.IndexHierarchy[tp.Unpack[tp.Tuple[sf.Index[np.int_], ...]]]
    h2 = sf.IndexHierarchy[tp.Unpack[tp.Tuple[sf.Index[np.str_], ...]]]

    check_type(v1, h1)
    assert TypeClinic(v1).check(h1).validated

    check_type(v2, h1)
    assert not TypeClinic(v1).check(h2).validated

def test_check_index_hierarchy_c():

    v1 = sf.IndexHierarchy.from_labels([(1, 'a', False), (1, 'b', False), (2, 'c', True)])

    h1 = sf.IndexHierarchy[sf.Index[np.int_], sf.Index[np.str_], sf.Index[np.bool_]]
    h2 = sf.IndexHierarchy[tp.Unpack[tp.Tuple[sf.Index[np.int_], sf.Index[np.str_], sf.Index[np.bool_]]]]

    h3 = sf.IndexHierarchy[sf.Index[np.int_], sf.Index[np.bool_], sf.Index[np.str_]]

    check_type(v1, h1)
    check_type(v1, h2)

def test_check_index_hierarchy_d1():

    v1 = sf.IndexHierarchy.from_labels([(1, 'a', False), (1, 'b', False), (2, 'c', True)])
    h1 = sf.IndexHierarchy[
            sf.Index[np.int_],
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[sf.Index[np.bool_], ...]],
            ]
    assert v1.via_type_clinic.check(h1).validated

    v2 = sf.IndexHierarchy.from_labels([(1, 'a',), (1, 'b',), (2, 'c',)])
    assert v2.via_type_clinic.check(h1).validated

    v3 = sf.IndexHierarchy.from_labels([
            (1, 'a', False, True),
            (1, 'b', False, True),
            (2, 'c', True, False),
            ])
    assert v3.via_type_clinic.check(h1).validated

def test_check_index_hierarchy_d2():

    v1 = sf.IndexHierarchy.from_labels(
            [(1,  False), (3,  False), (2,  True)],
            index_constructors=(partial(sf.Index, dtype=np.int64), sf.Index),
            )
    h1 = sf.IndexHierarchy[
            sf.Index[np.int64],
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[sf.Index[np.bool_], ...]],
            ]
    assert not v1.via_type_clinic.check(h1).validated
    assert scrub_str(v1.via_type_clinic.check(h1).to_str()) == 'In IndexHierarchy[Index[int64], Index[str_], Unpack[Tuple[Index[bool_], ...]]] Depth 1 Index[str_] Expected str_, provided bool_ invalid'

def test_check_index_hierarchy_e1():

    v1 = sf.IndexHierarchy.from_labels([(1,  3), (3,  2), (2,  3)])
    h1 = sf.IndexHierarchy[
            tp.Unpack[tp.Tuple[sf.Index[np.integer], ...]],
            ]
    assert v1.via_type_clinic.check(h1).validated

    v2 = sf.IndexHierarchy.from_labels([(1,  3, 5), (3,  2, 2), (2,  3, 7)])
    assert v2.via_type_clinic.check(h1).validated

def test_check_index_hierarchy_e2():

    v1 = sf.IndexHierarchy.from_labels([(1,  'a'), (3,  'b'), (2,  'c')])
    h1 = sf.IndexHierarchy[
            tp.Unpack[tp.Tuple[sf.Index[np.integer], ...]],
            ]
    assert not v1.via_type_clinic.check(h1).validated
    assert scrub_str(v1.via_type_clinic.check(h1).to_str()) == 'In IndexHierarchy[Unpack[Tuple[Index[integer], ...]]] Tuple[Index[integer], ...] Index[integer] Expected integer, provided str_ invalid'


def test_check_index_hierarchy_f():

    v1 = sf.IndexHierarchy.from_labels([(1,  'a'), (3,  'b'), (2,  'c')])
    h1 = sf.IndexHierarchy[sf.Index[np.integer], sf.IndexDate, sf.IndexDate]

    assert not v1.via_type_clinic.check(h1).validated
    assert scrub_str(v1.via_type_clinic.check(h1).to_str()) == 'In IndexHierarchy[Index[integer], IndexDate, IndexDate] Expected IndexHierarchy has 3 dtype, provided IndexHierarchy has 2 depth'


def test_check_index_hierarchy_g():

    v1 = sf.IndexHierarchy.from_labels([
            (1, 'a', False, 'a',),
            (3, 'b', True, 'c',),
            (2, 'c', False, 'd',),
            ],
            index_constructors=(
            partial(sf.Index, dtype=np.int64),
            sf.Index,
            sf.Index,
            sf.Index,
            ))
    h1 = sf.IndexHierarchy[
            sf.Index[np.int_],
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[sf.Index[np.bool_], ...]],
            sf.Index[np.str_],
            sf.Index[np.int_],
            sf.Index[np.int_],
            ]

    assert not v1.via_type_clinic.check(h1).validated

    assert scrub_str(v1.via_type_clinic.check(h1).to_str()) == 'In IndexHierarchy[Index[int64], Index[str_], Unpack[Tuple[Index[bool_], ...]], Index[str_], Index[int64], Index[int64]] Expected IndexHierarchy has 5 depth (excluding Unpack), provided IndexHierarchy has 4 depth'


def test_check_index_hierarchy_h1():

    v1 = sf.IndexHierarchy.from_labels([
            (1, 'a', False, 'a', 3, 2),
            (3, 'b', True, 'c', 10, 12),
            (2, 'c', False, 'd', 20, 3),
            ])
    h1 = sf.IndexHierarchy[
            sf.Index[np.int_],
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[sf.Index[np.bool_], ...]],
            sf.Index[np.str_],
            sf.Index[np.int_],
            sf.Index[np.int_],
            ]

    assert v1.via_type_clinic.check(h1).validated

def test_check_index_hierarchy_h2():

    v1 = sf.IndexHierarchy.from_labels([
            (1, 'a', False, True, 'a', 3, 2),
            (3, 'b', True, False, 'c', 10, 12),
            (2, 'c', False, True, 'd', 20, 3),
            ])
    h1 = sf.IndexHierarchy[
            sf.Index[np.int_],
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[sf.Index[np.bool_], ...]],
            sf.Index[np.str_],
            sf.Index[np.int_],
            sf.Index[np.int_],
            ]

    assert v1.via_type_clinic.check(h1).validated


def test_check_index_hierarchy_h3():

    v1 = sf.IndexHierarchy.from_labels([
            (1, 'a', False, True, 'a', 3, 2),
            (3, 'b', True, False, 'c', 10, 12),
            (2, 'c', False, True, 'd', 20, 3),
            ])
    h1 = sf.IndexHierarchy[
            sf.Index[np.int_],
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[sf.Index[np.bool_], ...]],
            sf.Index[np.str_],
            sf.Index[np.int_],
            sf.Index[np.int_],
            sf.Index[np.int_],
            ]

    assert not v1.via_type_clinic.check(h1).validated

#-------------------------------------------------------------------------------

def test_check_frame_a():
    records = ((1, 3, True), (3, 8, True),)
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],]
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))
    f: h1 = sf.Frame.from_records(records,
            columns=('a', 'b', 'c'),
            index=index,
            )
    cr = TypeClinic(f).check(h1)
    # NOTE: langauge support for defaults in TypeVarTuple might changes this
    assert get_hints(cr) == ('Expected Frame has 0 dtype, provided Frame has 3 dtype',)

def test_check_frame_b():
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[tp.Any, ...]],
            ]

    records = ((1, 3, True), (3, 8, True),)
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))
    f1: h1 = sf.Frame.from_records(records,
            columns=('a', 'b', 'c'),
            index=index,
            )
    assert TypeClinic(f1).check(h1).validated

    records = ((1, 3, True, False), (3, 8, True, True),)
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))
    f2: h1 = sf.Frame.from_records(records,
            columns=('a', 'b', 'c', 'd'),
            index=index,
            )
    assert TypeClinic(f2).check(h1).validated

def test_check_frame_c():
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[np.float64, ...]],
            ]
    records = ((1.8, 3.1), (3.2, 8.1),)
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))
    f1: h1 = sf.Frame.from_records(records,
            columns=('a', 'b'),
            index=index,
            )
    assert TypeClinic(f1).check(h1).validated

    records = ((1.8, 3.1, 5.4), (3.2, 8.1, 4.7),)
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))
    f2: h1 = sf.Frame.from_records(records,
            columns=('a', 'b', 'c'),
            index=index,
            )
    assert TypeClinic(f2).check(h1).validated


    records = ((1.8, 3.1, False), (3.2, 8.1, True),)
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))
    f3: h1 = sf.Frame.from_records(records,
            columns=('a', 'b', 'c'),
            index=index,
            )
    assert scrub_str(TypeClinic(f3).check(h1).to_str()) == 'In Frame[IndexDate, Index[str_], Unpack[Tuple[float64, ...]]] Tuple[float64, ...] Expected float64, provided bool_ invalid'

def test_check_frame_d():
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            np.bool_,
            tp.Unpack[tp.Tuple[np.float64, ...]],
            np.str_,
            np.str_
            ]
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))

    records1 = ((True, 1.8, 3.1, 'x', 'y'), (False, 3.2, 8.1, 'a', 'b'),)
    f1 = sf.Frame.from_records(records1,
            columns=('a', 'b', 'c', 'd', 'e'),
            index=index,
            )
    assert TypeClinic(f1).check(h1).validated

    records2 = ((True, 1.8, 3.1, 1.2, 'x', 'y'), (False, 3.2, 8.1, 3.5, 'a', 'b'),)
    f2 = sf.Frame.from_records(records2,
            columns=('a', 'b', 'c', 'd', 'e', 'f'),
            index=index,
            )
    assert TypeClinic(f2).check(h1).validated

    records3 = ((1.8, 3.1, 1.2, 'x', 'y'), (3.2, 8.1, 3.5, 'a', 'b'),)
    f3 = sf.Frame.from_records(records3,
            columns=('a', 'b', 'c', 'd', 'e'),
            index=index,
            )
    assert scrub_str(TypeClinic(f3).check(h1).to_str()) == 'In Frame[IndexDate, Index[str_], bool_, Unpack[Tuple[float64, ...]], str_, str_] Field 0 Expected bool_, provided float64 invalid'
    assert not TypeClinic(f3).check(h1).validated

    records4 = ((True, 1.8, 'x'), (False, 3.2, 'a'),)
    f4 = sf.Frame.from_records(records4,
            columns=('a', 'b', 'c'),
            index=index,
            )
    assert not TypeClinic(f4).check(h1).validated
    assert scrub_str(TypeClinic(f4).check(h1).to_str()) == 'In Frame[IndexDate, Index[str_], bool_, Unpack[Tuple[float64, ...]], str_, str_] Field 1 Expected str_, provided float64 invalid'


def test_check_frame_e1():
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[np.float64, ...]],
            np.str_,
            np.str_
            ]
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))

    records1 = ((3.1, 'x', 'y'), (8.1, 'a', 'b'),)
    f1 = sf.Frame.from_records(records1,
            columns=('a', 'b', 'c'),
            index=index,
            )
    assert TypeClinic(f1).check(h1).validated

    records2 = ((3.1, 3.2, 5.2, 'x', 'y'), (8.1, 1.5, 5.2, 'a', 'b'),)
    f2 = sf.Frame.from_records(records2,
            columns=('a', 'b', 'c', 'd', 'e'),
            index=index,
            )
    assert TypeClinic(f2).check(h1).validated

    records3 = ((3.1, False, 5.2, 'x', 'y'), (8.1, True, 5.2, 'a', 'b'),)
    f3 = sf.Frame.from_records(records3,
            columns=('a', 'b', 'c', 'd', 'e'),
            index=index,
            )
    assert not TypeClinic(f3).check(h1).validated
    assert scrub_str(TypeClinic(f3).check(h1).to_str()) == 'In Frame[IndexDate, Index[str_], Unpack[Tuple[float64, ...]], str_, str_] Fields 0 to 2 Tuple[float64, ...] Expected float64, provided bool_ invalid'


def test_check_frame_e2():
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[np.float64, ...]],
            np.str_,
            np.str_
            ]
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))

    records1 = ((3.1, 'x'), (8.1, 'a'),)
    f1 = sf.Frame.from_records(records1,
            columns=('a', 'b'),
            index=index,
            )

    assert not f1.via_type_clinic.check(h1).validated
    assert scrub_str(f1.via_type_clinic.check(h1).to_str()) == 'In Frame[IndexDate, Index[str_], Unpack[Tuple[float64, ...]], str_, str_] Field 0 Expected str_, provided float64 invalid'

    records2 = ((3.1, 'x', 'p'), (8.1, 'a', 'q'),)
    f2 = sf.Frame.from_records(records2,
            columns=('a', 'b', 'c'),
            index=index,
            )

    assert f2.via_type_clinic.check(h1).validated


def test_check_frame_e3():
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[np.float64, ...]],
            np.str_,
            np.str_
            ]
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))

    records1 = (('a', 'x'), ('b', 'a'),)
    f1 = sf.Frame.from_records(records1,
            columns=('a', 'b'),
            index=index,
            )

    assert f1.via_type_clinic.check(h1).validated

    records2 = ((1.2, 'a', 'x'), (3.4, 'b', 'a'),)
    f2 = sf.Frame.from_records(records2,
            columns=('a', 'b', 'c'),
            index=index,
            )
    assert f2.via_type_clinic.check(h1).validated


def test_check_frame_f1():
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            np.str_,
            np.str_,
            tp.Unpack[tp.Tuple[np.float64, ...]],
            ]
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))

    records1 = (('a', 'x'), ('b', 'a'),)
    f1 = sf.Frame.from_records(records1,
            columns=('a', 'b'),
            index=index,
            )

    assert f1.via_type_clinic.check(h1).validated

    records2 = (('a', 'x', 1.2), ('b', 'a', 5.4),)
    f2 = sf.Frame.from_records(records2,
            columns=('a', 'b', 'c'),
            index=index,
            )

    assert f2.via_type_clinic.check(h1).validated

    records3 = (('a', 'x', 1.2, 5.3, 5.4), ('b', 'a', 5.4, 1.2, 1.4),)
    f3 = sf.Frame.from_records(records3,
            columns=('a', 'b', 'c', 'd', 'e'),
            index=index,
            )

    assert f3.via_type_clinic.check(h1).validated


def test_check_frame_f2():
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            np.str_,
            np.str_,
            tp.Unpack[tp.Tuple[np.float64, ...]],
            ]
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))

    records1 = (('a', 'x', 1.3, 'q'), ('b', 'a', 1.5, 'x'),)
    f1 = sf.Frame.from_records(records1,
            columns=('a', 'b', 'c', 'd'),
            index=index,
            )
    assert not f1.via_type_clinic.check(h1).validated
    assert scrub_str(f1.via_type_clinic.check(h1).to_str()) == 'In Frame[IndexDate, Index[str_], str_, str_, Unpack[Tuple[float64, ...]]] Fields 2 to 3 Tuple[float64, ...] Expected float64, provided str_ invalid'

def test_check_frame_g():
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            np.str_,
            np.str_,
            np.str_,
            tp.Unpack[tp.Tuple[np.float64, ...]],
            ]
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))

    records1 = (('a', 'x'), ('b', 'a'),)
    f1 = sf.Frame.from_records(records1,
            columns=('a', 'b'),
            index=index,
            )
    assert not f1.via_type_clinic.check(h1).validated
    assert scrub_str(f1.via_type_clinic.check(h1).to_str()) == 'In Frame[IndexDate, Index[str_], str_, str_, str_, Unpack[Tuple[float64, ...]]] Expected Frame has 3 dtype (excluding Unpack), provided Frame has 2 dtype'



#-------------------------------------------------------------------------------

def get_hints(records: tp.Iterable[TValidation] | CheckResult) -> tp.Tuple[str]:
    return tuple(r[1] for r in records)

def test_validate_labels_a1():
    idx1 = sf.Index(('a', 'b', 'c'))
    v = Labels('a', 'b', 'c')
    assert not get_hints(v.iter_error_log(idx1, None, (None,)))

def test_validate_labels_a2():
    idx1 = sf.Index(('a', 'x', 'c'))
    v = Labels('a', 'b', 'c')
    assert get_hints(v.iter_error_log(idx1, None, (None,))) == ("Expected 'b', provided 'x'",)

def test_validate_labels_a3():
    idx1 = sf.Index(('a', 'x', 'z'))
    v = Labels('a', 'b', 'c')
    assert get_hints(v.iter_error_log(idx1, None, (None,))) == (
            "Expected 'b', provided 'x'",
            "Expected 'c', provided 'z'")

def test_validate_labels_b():
    idx1 = sf.Index(('a', 'b', 'c', 'd'))
    v = Labels('a', ..., 'd')
    assert not get_hints(v.iter_error_log(idx1, None, (None,)))

def test_validate_labels_c():
    idx1 = sf.Index(('a', 'b', 'c', 'd'))
    v = Labels(..., 'd')
    assert not get_hints(v.iter_error_log(idx1, None, (None,)))

def test_validate_labels_d1():
    idx1 = sf.Index(('a', 'b', 'c', 'd'))
    v = Labels('a', 'b', ...)
    assert not get_hints(v.iter_error_log(idx1, None, (None,)))

def test_validate_labels_d2():
    idx1 = sf.Index(('a', 'b', 'c', 'd'))
    v = Labels('a', 'b', ..., 'e')
    assert get_hints(v.iter_error_log(idx1, None, (None,))) == ("Expected has unmatched labels 'e'",)

def test_validate_labels_e1():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e'))
    v = Labels('a', ..., 'c', ..., 'd')
    assert get_hints(v.iter_error_log(idx1, None, (None,))) == ("Expected labels exhausted at provided 'e'",)

def test_validate_labels_e2():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e'))
    v = Labels('a', ..., 'c', ..., 'e')
    assert not get_hints(v.iter_error_log(idx1, None, (None,)))

def test_validate_labels_e3():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e'))
    v = Labels('a', ..., 'c', ...)
    assert not get_hints(v.iter_error_log(idx1, None, (None,)))

def test_validate_labels_e4():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e'))
    v = Labels(..., 'c', ...)
    assert not get_hints(v.iter_error_log(idx1, None, (None,)))

def test_validate_labels_e5():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e'))
    v = Labels(..., 'b', 'c', ...)
    assert not get_hints(v.iter_error_log(idx1, None, (None,)))

def test_validate_labels_e6():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e'))
    v = Labels(..., 'b', ..., 'd', 'e')
    assert not get_hints(v.iter_error_log(idx1, None, (None,)))

def test_validate_labels_e7():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e'))
    v = Labels('a', 'b', ..., 'd', 'e')
    assert not get_hints(v.iter_error_log(idx1, None, (None,)))

def test_validate_labels_e8():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e'))
    v = Labels('a', 'b', ..., 'f', ...)
    assert get_hints(v.iter_error_log(idx1, None, (None,))) == ("Expected has unmatched labels 'f', ...",)

def test_validate_labels_e9():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e'))
    v = Labels(..., 'x', ..., 'y', ...)
    assert get_hints(v.iter_error_log(idx1, None, (None,))) == ("Expected has unmatched labels 'x', ..., 'y', ...",)

def test_validate_labels_e10():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e'))
    v = Labels(..., 'a', ..., ...)
    assert get_hints(v.iter_error_log(idx1, None, (None,))) == ("Expected cannot be defined with adjacent ellipses",)

def test_validate_labels_e11():
    idx1 = sf.Series(('a',))
    v = Labels('a', ...)
    assert get_hints(v.iter_error_log(idx1, None, (None,))) == ("Expected Labels('a', ...) to be used on Index or IndexHierarchy, not provided Series",)



def test_validate_validator_a():
    idx1 = sf.Index(('a', 'b', 'c'))
    v1 = Validator(lambda i: 'b' in i)
    assert not get_hints(v1.iter_error_log(idx1, None, (None,)))

    v2 = Validator(lambda i: 'q' in i)
    assert get_hints(v2.iter_error_log(idx1, None, (None,))) == ("Index failed validation with <lambda>",)



#-------------------------------------------------------------------------------
@skip_win
def test_check_error_display_a():

    records = (
            (1, 3, True),
            (4, 100, False),
            (3, 8, True),
            )
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            np.int_,
            np.int_,
            np.bool_]

    index = sf.IndexDate(('2022-01-03', '2022-02-05', '2018-04-02'))
    f: h1 = sf.Frame.from_records(records,
            columns=('a', 'b', 'c'),
            index=index,
            )

    h2 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.int_],
            np.int_,
            np.int_,
            np.str_]

    with pytest.raises(TypeError):
        check_type(f, h2)
    try:
        check_type(f, h2)
    except TypeError as e:
        assert scrub_str(str(e)) == 'In Frame[IndexDate, Index[int64], int64, int64, str_] Expected str_, provided bool_ invalid In Frame[IndexDate, Index[int64], int64, int64, str_] Index[int64] Expected int64, provided str_ invalid'


#-------------------------------------------------------------------------------
def test_type_clinic_a():
    records = (
            (1, True, 20, True),
            (30, False, 100, False),
            (54, False, 8, True),
            )
    index = sf.IndexDate(('2022-01-03', '2022-02-05', '2018-04-02'))
    columns = sf.IndexHierarchy.from_product(('a', 'b'), (True, False))
    f = sf.Frame.from_records(records, columns=columns, index=index, dtypes=(np.int64, np.bool_, np.int64, np.bool_))

    h = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.IndexHierarchy[sf.Index[np.str_], sf.Index[np.int64]],
            np.int64,
            np.bool_,
            np.int64,
            np.int64,
            ]

    assert str(TypeClinic(f).check(h)) == '<CheckResult: 2 errors>'
    post = TypeClinic(f).check(h).to_str()
    assert post == '\nIn Frame[IndexDate, IndexHierarchy[Index[str_], Index[int64]], int64, bool_, int64, int64]\n└── Expected int64, provided bool_ invalid\nIn Frame[IndexDate, IndexHierarchy[Index[str_], Index[int64]], int64, bool_, int64, int64]\n└── IndexHierarchy[Index[str_], Index[int64]]\n    └── Index[int64]\n        └── Expected int64, provided bool_ invalid'


def test_type_clinic_to_hint_a():
    s = sf.Series((3, 2), index=sf.Index(('a', 'b')), dtype=np.int64)
    assert TypeClinic(s).to_hint() == sf.Series[sf.Index[np.str_], np.int64]

def test_type_clinic_to_hint_b():
    s = sf.Index(('a', 'b'))
    assert TypeClinic(s).to_hint() == sf.Index[np.str_]

def test_type_clinic_to_hint_c():
    s = sf.IndexHierarchy.from_product(('a', 'b'), (True, False))
    assert TypeClinic(s).to_hint() == sf.IndexHierarchy[sf.Index[np.str_], sf.Index[np.bool_]]


def test_type_clinic_to_hint_d():
    records = ((1, 3, True), (3, 8, True),)
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))
    f = sf.Frame.from_records(records,
            columns=('a', 'b', 'c'),
            index=index,
            dtypes=(np.int64, np.int64, np.bool_)
            )
    h = TypeClinic(f).to_hint()
    assert h == sf.Frame[sf.IndexDate, sf.Index[np.str_], np.int64, np.int64, np.bool_]

def test_type_clinic_to_hint_e():
    assert TypeClinic(3).to_hint() == int
    assert TypeClinic('foo').to_hint() == str
    assert TypeClinic(str).to_hint() == tp.Type[str]
    assert TypeClinic(sf.Frame).to_hint() == tp.Type[sf.Frame]

@skip_pyle310
def test_type_clinic_to_hint_f():
    assert TypeClinic(np.dtype(np.float64)).to_hint() == np.dtype[np.float64]
    assert TypeClinic(np.array([False, True])).to_hint() == np.ndarray[np.dtype[np.bool_]]


#-------------------------------------------------------------------------------
def test_via_type_clinic_a():
    s = sf.Series(('a', 'b'), index=(('x', 'y')))
    assert str(s.via_type_clinic) == 'Series[Index[str_], str_]'
    assert s.via_type_clinic.check(s.via_type_clinic.to_hint()).validated

def test_via_type_clinic_b():
    s = sf.Series(('a', 'b'), index=(('x', 'y')))

    with pytest.raises(TypeError):
        s.via_type_clinic(sf.Series[sf.IndexDate, np.str_])
