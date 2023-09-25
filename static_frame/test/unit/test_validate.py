import numpy as np
import pytest
import typing_extensions as tp

import static_frame as sf
# from static_frame.core.validate import Validator
# from static_frame.core.validate import validate_pair
# from static_frame.core.validate import Labels
from static_frame.core.validate import Len
from static_frame.core.validate import Name
from static_frame.core.validate import check_interface
from static_frame.core.validate import check_type
from static_frame.test.test_case import skip_nple119
from static_frame.test.test_case import skip_pyle310


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
        assert str(e) == 'Expected Type[Series], provided int invalid.'

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

@skip_pyle310
def test_check_type_fail_fast_a():
    v = sf.Series(('a', 'b'), index=sf.Index(('x', 'y'), dtype=np.str_))
    h = sf.Series[sf.Index[np.int64], np.int64]

    with pytest.raises(TypeError):
        check_type(v, h, fail_fast=True)
    try:
        check_type(v, h, fail_fast=True)
    except TypeError as e:
        assert str(e) == 'In Series[Index[int64], int64]: expected int64, provided str_ invalid.'


    with pytest.raises(TypeError):
        check_type(v, h, fail_fast=False)
    try:
        check_type(v, h, fail_fast=False)
    except TypeError as e:
        assert str(e) == 'In Series[Index[int64], int64]: expected int64, provided str_ invalid. In Series[Index[int64], int64], Index[int64]: expected int64, provided str_ invalid.'

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

def test_check_type_tuple_c():

    with pytest.raises(TypeError):
        check_type((3, 4), tp.Tuple[int, int, int])
    try:
        check_type((3, 4), tp.Tuple[int, int, int])
    except TypeError as e:
        assert str(e) == 'Failed check in Tuple[int, int, int]: expected tuple length of 3, provided tuple length of 2.'


def test_check_type_tuple_d():

    with pytest.raises(TypeError) as e:
        check_type((3, 4, 5), tp.Tuple[..., int, ...])
    try:
        check_type((3, 4, 5), tp.Tuple[..., int, ...])
    except TypeError as e:
        assert str(e) == 'Failed check in Tuple[..., int, ...]: invalid ellipses usage.'

#-------------------------------------------------------------------------------

def test_check_type_literal_a():
    check_type(42, tp.Literal[42])
    check_type(42, tp.Literal[-1, 42])

    with pytest.raises(TypeError):
        check_type(42, tp.Literal['a', 'b'])
    try:
        check_type(42, tp.Literal['a', 'b'])
    except TypeError as e:
        assert str(e) == 'In Literal[a, b]: expected a, provided int invalid. In Literal[a, b]: expected b, provided int invalid.'

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
        assert str(e) == 'In return of (a: int, b: int) -> bool: expected bool, provided int invalid.'

    try:
        assert proc1(2, 'foo') == 6
    except TypeError as e:
        assert str(e) == 'In args of (a: int, b: int) -> bool: expected int, provided str invalid.'

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

#-------------------------------------------------------------------------------

def test_check_annotated_a():

    check_type(3, tp.Annotated[int, 'foo'])

def test_check_annotated_b():

    v1 = sf.Series(('a', 'b'), index=sf.Index((10, 20)))
    h1 = tp.Annotated[
            sf.Series[sf.Index[np.int64], np.str_],
            Name('foo'),
    ]
    with pytest.raises(TypeError):
        check_type(v1, h1)

    v1 = sf.Series(('a', 'b'), index=sf.Index((10, 20)), name='foo')
    check_type(v1, h1)

def test_check_annotated_c():

    v1 = sf.Series(('a', 'b'), index=sf.Index((10, 20)))
    h1 = tp.Annotated[
            sf.Series[sf.Index[np.int64], np.str_],
            Len(1),
            ]
    h2 = tp.Annotated[
            sf.Series[sf.Index[np.int64], np.str_],
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
    try:
        check_type(v1, h1)
    except TypeError as e:
        assert str(e) == 'Failed check in IndexHierarchy[Index[str_], Index[integer], IndexDate]: expected IndexHierarchy depth of 3, provided depth of 2.'



def test_check_index_hierarchy_b():

    v1 = sf.IndexHierarchy.from_labels([(1, 100), (1, 200), (2, 100)])
    v2 = sf.IndexHierarchy.from_labels([(1, 100, 3), (1, 200, 3), (2, 100, 3)])

    h1 = sf.IndexHierarchy[*tp.Tuple[sf.Index[np.int_], ...]]
    h2 = sf.IndexHierarchy[*tp.Tuple[sf.Index[np.str_], ...]]

    check_type(v1, h1)
    check_type(v2, h1)

    try:
        check_type(v1, h2)
    except TypeError as e:
        assert str(e) == 'In IndexHierarchy[Unpack[Tuple[Index[str_], ...]]], Tuple[Index[str_], ...], Index[str_]: expected str_, provided int64 invalid. In IndexHierarchy[Unpack[Tuple[Index[str_], ...]]], Tuple[Index[str_], ...], Index[str_]: expected str_, provided int64 invalid.'
    else: # did not raise
        raise TypeError('expected failure did not raise')

def test_check_index_hierarchy_c():

    v1 = sf.IndexHierarchy.from_labels([(1, 'a', False), (1, 'b', False), (2, 'c', True)])

    h1 = sf.IndexHierarchy[sf.Index[np.int_], sf.Index[np.str_], sf.Index[np.bool_]]
    h2 = sf.IndexHierarchy[sf.Index[np.int_], sf.Index[np.bool_], sf.Index[np.str_]]

    try:
        check_type(v1, h2)
    except TypeError as e:
        assert str(e) == 'In IndexHierarchy[Index[int64], Index[bool_], Index[str_]], Index[bool_]: expected bool_, provided str_ invalid. In IndexHierarchy[Index[int64], Index[bool_], Index[str_]], Index[str_]: expected str_, provided bool_ invalid.'
    else: # did not raise
        raise TypeError('expected failure did not raise')



    # h2 = sf.IndexHierarchy[*tp.Tuple[sf.Index[np.str_], ...]]