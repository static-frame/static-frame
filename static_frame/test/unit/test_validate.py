import numpy as np
import pytest
import typing_extensions as tp

import static_frame as sf
# from static_frame.core.validate import validate_pair
from static_frame.core.validate import check_type
from static_frame.test.test_case import skip_nple119


def test_validate_pair_a():

    check_type(sf.IndexDate(('2022-01-01',)), sf.IndexDate)
    check_type(sf.IndexDate(('2022-01-01',)), tp.Any)

    with pytest.raises(TypeError):
        check_type(sf.IndexDate(('2022-01-01',)), sf.IndexSecond)


def test_validate_pair_b():

    check_type(3, int)
    check_type('foo', str)
    check_type(False, bool)

    with pytest.raises(TypeError):
        check_type(3, str)

    with pytest.raises(TypeError):
        check_type(True, int)

#-------------------------------------------------------------------------------

def test_validate_pair_union_a():

    check_type(3, tp.Union[int, str])

    with pytest.raises(TypeError):
        check_type('x', tp.Union[int, float])

    check_type('x', tp.Union[str, bytes])
    check_type('x', tp.Union[int, str])


#-------------------------------------------------------------------------------

def test_validate_pair_type_a():

    check_type(sf.Series, tp.Type[sf.Series])

    with pytest.raises(TypeError):
        check_type(sf.Series, tp.Type[sf.Index])


def test_validate_pair_type_b():
    try:
        check_type(3, tp.Type[sf.Series])
    except TypeError as e:
        assert str(e) == 'Provided int invalid for typing.Type[static_frame.core.series.Series].'

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

def test_validate_pair_containers_a():
    v = sf.Series(('a', 'b'), index=sf.IndexDate(('2021-04-05', '2022-05-03')))
    h1 = sf.SeriesHE[sf.IndexDate, np.str_]
    h2 = sf.Index[np.str_]
    h3 = sf.Series[sf.IndexDate, np.str_]

    with pytest.raises(TypeError):
        check_type(v, h2)

    with pytest.raises(TypeError):
        check_type(v, h1)

def test_validate_pair_containers_b():
    v = sf.Series(('a', 'b'), index=sf.IndexDate(('2021-04-05', '2022-05-03')))
    h1 = sf.Series[sf.IndexDate, np.str_]
    h2 = sf.Series[sf.IndexDate, np.int64]

    check_type(v, h1)
    with pytest.raises(TypeError):
        check_type(v, h2)


def test_validate_pair_containers_c():
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


def test_validate_pair_containers_d():
    v1 = sf.Series(('a', 'b'), index=sf.Index((10, 20), dtype=np.int64))
    v2 = sf.Series(('a', 'b'), index=sf.Index((10, 20), dtype=np.str_))
    v3 = sf.Series(('a', 'b'), index=sf.Index((1, 0), dtype=np.bool_))

    h1 = sf.Series[sf.Index[tp.Union[np.int64, np.str_]], np.str_]

    check_type(v1, h1)
    check_type(v2, h1)
    with pytest.raises(TypeError):
        check_type(v3, h1)


def test_validate_pair_containers_e():
    v1 = sf.Series(('a', 'b'), index=sf.IndexDate(('2021-04-05', '2022-05-03')))
    v2 = sf.Series(('a', 'b'), index=sf.IndexSecond(('2021-04-05', '2022-05-03')))
    v3 = sf.Series(('a', 'b'), index=sf.Index(('x', 'y')))

    h1 = sf.Series[tp.Union[sf.IndexDate, sf.IndexSecond], np.str_]

    check_type(v1, h1)
    check_type(v2, h1)

    with pytest.raises(TypeError):
        check_type(v3, h1)


#-------------------------------------------------------------------------------

def test_validate_pair_fail_fast_a():
    v = sf.Series(('a', 'b'), index=sf.Index(('x', 'y'), dtype=np.str_))
    h = sf.Series[sf.Index[np.int64], np.int64]

    try:
        check_type(v, h, fail_fast=True)
    except TypeError as e:
        assert str(e) == 'In static_frame.core.series.Series[static_frame.core.index.Index[numpy.int64], numpy.int64], provided str_ invalid for int64.'


    try:
        check_type(v, h, fail_fast=False)
    except TypeError as e:
        assert str(e) == 'In static_frame.core.series.Series[static_frame.core.index.Index[numpy.int64], numpy.int64], provided str_ invalid for int64. In static_frame.core.series.Series[static_frame.core.index.Index[numpy.int64], numpy.int64], static_frame.core.index.Index[numpy.int64], provided str_ invalid for int64.'


