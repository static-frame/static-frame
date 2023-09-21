import numpy as np
import pytest
import typing_extensions as tp

import static_frame as sf
# from static_frame.core.validate import validate_pair
from static_frame.core.validate import validate_pair_raises
from static_frame.test.test_case import skip_nple119


def test_validate_pair_a():

    validate_pair_raises(sf.IndexDate(('2022-01-01',)), sf.IndexDate)
    validate_pair_raises(sf.IndexDate(('2022-01-01',)), tp.Any)

    with pytest.raises(TypeError):
        validate_pair_raises(sf.IndexDate(('2022-01-01',)), sf.IndexSecond)


def test_validate_pair_b():

    validate_pair_raises(3, int)
    validate_pair_raises('foo', str)
    validate_pair_raises(False, bool)

    with pytest.raises(TypeError):
        validate_pair_raises(3, str)

    with pytest.raises(TypeError):
        validate_pair_raises(True, int)

#-------------------------------------------------------------------------------

def test_validate_pair_union_a():

    validate_pair_raises(3, tp.Union[int, str])

    with pytest.raises(TypeError):
        validate_pair_raises('x', tp.Union[int, float])

    validate_pair_raises('x', tp.Union[str, bytes])
    validate_pair_raises('x', tp.Union[int, str])


#-------------------------------------------------------------------------------

def test_validate_pair_type_a():

    validate_pair_raises(sf.Series, tp.Type[sf.Series])

    with pytest.raises(TypeError):
        validate_pair_raises(sf.Series, tp.Type[sf.Index])

#-------------------------------------------------------------------------------

@skip_nple119
def test_validate_numpy_a():
    v = np.array([False, True, False])
    h1 = np.ndarray[tp.Any, np.dtype[np.bool_]]
    h2 = np.ndarray[tp.Any, np.dtype[np.str_]]

    validate_pair_raises(v, h1)
    with pytest.raises(TypeError):
        validate_pair_raises(v, h2)


#-------------------------------------------------------------------------------

def test_validate_pair_containers_a():
    v = sf.Series(('a', 'b'), index=sf.IndexDate(('2021-04-05', '2022-05-03')))
    h1 = sf.SeriesHE[sf.IndexDate, np.str_]
    h2 = sf.Index[np.str_]
    h3 = sf.Series[sf.IndexDate, np.str_]

    with pytest.raises(TypeError):
        validate_pair_raises(v, h2)

    with pytest.raises(TypeError):
        validate_pair_raises(v, h1)

def test_validate_pair_containers_b():
    v = sf.Series(('a', 'b'), index=sf.IndexDate(('2021-04-05', '2022-05-03')))
    h1 = sf.Series[sf.IndexDate, np.str_]
    h2 = sf.Series[sf.IndexDate, np.int64]

    validate_pair_raises(v, h1)
    with pytest.raises(TypeError):
        validate_pair_raises(v, h2)


def test_validate_pair_containers_c():
    v = sf.Series(('a', 'b'), index=sf.Index((10, 20), dtype=np.int64))
    h1 = sf.Series[sf.Index[np.str_], np.str_]
    h2 = sf.Series[sf.IndexDate, np.int64]
    h3 = sf.Series[sf.Index[np.str_], np.int64]
    h4 = sf.Series[sf.Index[np.int64], np.str_]

    with pytest.raises(TypeError):
        validate_pair_raises(v, h1)
    with pytest.raises(TypeError):
        validate_pair_raises(v, h2)
    with pytest.raises(TypeError):
        validate_pair_raises(v, h3)

    validate_pair_raises(v, h4)


def test_validate_pair_containers_d():
    v1 = sf.Series(('a', 'b'), index=sf.Index((10, 20), dtype=np.int64))
    v2 = sf.Series(('a', 'b'), index=sf.Index((10, 20), dtype=np.str_))
    v3 = sf.Series(('a', 'b'), index=sf.Index((1, 0), dtype=np.bool_))

    h1 = sf.Series[sf.Index[tp.Union[np.int64, np.str_]], np.str_]

    validate_pair_raises(v1, h1)
    validate_pair_raises(v2, h1)
    with pytest.raises(TypeError):
        validate_pair_raises(v3, h1)


def test_validate_pair_containers_e():
    v1 = sf.Series(('a', 'b'), index=sf.IndexDate(('2021-04-05', '2022-05-03')))
    v2 = sf.Series(('a', 'b'), index=sf.IndexSecond(('2021-04-05', '2022-05-03')))
    v3 = sf.Series(('a', 'b'), index=sf.Index(('x', 'y')))

    h1 = sf.Series[tp.Union[sf.IndexDate, sf.IndexSecond], np.str_]

    validate_pair_raises(v1, h1)
    validate_pair_raises(v2, h1)

    with pytest.raises(TypeError):
        validate_pair_raises(v3, h1)
