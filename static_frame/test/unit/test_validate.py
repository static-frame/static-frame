import numpy as np
import pytest
import typing_extensions as tp

import static_frame as sf
from static_frame.core.validate import validate_pair
from static_frame.core.validate import validate_pair_raises


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

    # validate_pair_raises(v, h1)
    # validate_pair_raises(v, h2)
    with pytest.raises(TypeError):
        validate_pair_raises(v, h3)