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
    v = sf.Series(('a', 'b'), index=sf.IndexDate(('2021-04-05', '2022-05-03')))
    h = sf.Series[sf.IndexDate, np.str_]

    validate_pair_raises(v, h)


def test_validate_pair_c():

    validate_pair_raises(3, int)
    validate_pair_raises('foo', str)
    validate_pair_raises(False, bool)

    with pytest.raises(TypeError):
        validate_pair_raises(3, str)

    with pytest.raises(TypeError):
        validate_pair_raises(True, int)


def test_validate_pair_union_a():

    validate_pair_raises(3, tp.Union[int, str])

    with pytest.raises(TypeError):
        validate_pair_raises('x', tp.Union[int, float])

    validate_pair_raises('x', tp.Union[str, bytes])
    validate_pair_raises('x', tp.Union[int, str])