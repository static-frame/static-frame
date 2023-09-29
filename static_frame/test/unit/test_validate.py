from __future__ import annotations

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
from static_frame.test.test_case import skip_nple119
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
        assert str(e).replace('\n', '') == 'Expected Type[Series], provided int invalid.'

#-------------------------------------------------------------------------------

@skip_nple119
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
    s = s.replace('\n', '').replace(CheckResult._LINE, '').replace(CheckResult._CORNER, '')
    while '  ' in s:
        s = s.replace('  ', ' ')
    return s

@skip_pyle310
def test_check_type_fail_fast_a():
    v = sf.Series(('a', 'b'), index=sf.Index(('x', 'y'), dtype=np.str_))
    h = sf.Series[sf.Index[np.int64], np.int64]


    with pytest.raises(TypeError):
        check_type(v, h, fail_fast=True)
    try:
        check_type(v, h, fail_fast=True)
    except TypeError as e:
        assert scrub_str(str(e)) == 'In Series[Index[int64], int64] Expected int64, provided str_ invalid.'

    with pytest.raises(TypeError):
        check_type(v, h, fail_fast=False)
    try:
        check_type(v, h, fail_fast=False)
    except TypeError as e:
        assert scrub_str(str(e)) == 'In Series[Index[int64], int64] Expected int64, provided str_ invalid.In Series[Index[int64], int64] Index[int64] Expected int64, provided str_ invalid.'

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
        assert scrub_str(str(e)) == 'In return of (a: int, b: int) -> bool Expected bool, provided int invalid.'

    try:
        assert proc1(2, 'foo') == 6
    except TypeError as e:
        assert scrub_str(str(e)) == 'In args of (a: int, b: int) -> bool Expected int, provided str invalid.'

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

@skip_pyle310
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

@skip_pyle310
def test_check_index_hierarchy_b():

    v1 = sf.IndexHierarchy.from_labels([(1, 100), (1, 200), (2, 100)])
    v2 = sf.IndexHierarchy.from_labels([(1, 100, 3), (1, 200, 3), (2, 100, 3)])

    h1 = sf.IndexHierarchy[tp.Unpack[tp.Tuple[sf.Index[np.int_], ...]]]
    h2 = sf.IndexHierarchy[tp.Unpack[tp.Tuple[sf.Index[np.str_], ...]]]

    check_type(v1, h1)
    check_type(v2, h1)

    assert not TypeClinic(v1).check(h2).validated

@skip_pyle310
def test_check_index_hierarchy_c():

    v1 = sf.IndexHierarchy.from_labels([(1, 'a', False), (1, 'b', False), (2, 'c', True)])

    h1 = sf.IndexHierarchy[sf.Index[np.int_], sf.Index[np.str_], sf.Index[np.bool_]]
    h2 = sf.IndexHierarchy[tp.Unpack[tp.Tuple[sf.Index[np.int_], sf.Index[np.str_], sf.Index[np.bool_]]]]

    h3 = sf.IndexHierarchy[sf.Index[np.int_], sf.Index[np.bool_], sf.Index[np.str_]]

    check_type(v1, h1)
    check_type(v1, h2)

    # try:
    #     check_type(v1, h3)

#-------------------------------------------------------------------------------

def get_hints(records: tp.Iterable[TValidation]) -> tp.Tuple[str]:
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
        assert scrub_str(str(e)) == 'In Frame[IndexDate, Index[int64], int64, int64, str_] Expected str_, provided bool_ invalid.In Frame[IndexDate, Index[int64], int64, int64, str_] Index[int64] Expected int64, provided str_ invalid.'


#-------------------------------------------------------------------------------
def test_generic_factory_a():
    records = (
            (1, True, 20, True),
            (30, False, 100, False),
            (54, False, 8, True),
            )
    index = sf.IndexDate(('2022-01-03', '2022-02-05', '2018-04-02'))
    columns = sf.IndexHierarchy.from_product(('a', 'b'), (True, False))
    f = sf.Frame.from_records(records, columns=columns, index=index)

    h = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.IndexHierarchy[sf.Index[np.str_], sf.Index[np.int_]],
            np.int_,
            np.bool_,
            np.int_,
            np.int_,
            ]

    post = str(TypeClinic(f).check(h))
    assert post == '\nIn Frame[IndexDate, IndexHierarchy[Index[str_], Index[int64]], int64, bool_, int64, int64]\n└── Expected int64, provided bool_ invalid.\nIn Frame[IndexDate, IndexHierarchy[Index[str_], Index[int64]], int64, bool_, int64, int64]\n└── IndexHierarchy[Index[str_], Index[int64]]\n    └── Index[int64]\n        └── Expected int64, provided bool_ invalid.'
