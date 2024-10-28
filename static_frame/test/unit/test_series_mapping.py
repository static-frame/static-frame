from collections.abc import Mapping

import numpy as np
import pytest

from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.series import Series


def test_series_mapping_a():
    s = Series((10, 20, 30), index=('x', 'y', 'z'))
    sm = s.via_mapping
    assert len(sm) == 3
    assert sm['y'] == 20
    assert isinstance(sm, Mapping)

def test_series_mapping_b():
    s = Series((10, 20, 30, 40), index=IndexHierarchy.from_product(('x', 'y'), (True, False)))
    sm = s.via_mapping
    assert len(sm) == 4

    with pytest.raises(KeyError):
        assert sm['y'] == 20
    assert sm[('y', True)] == 30
    assert list(sm.keys()) == [('x', True), ('x', False), ('y', True), ('y', False)]
    assert tuple(sm.keys()) == (('x', True), ('x', False), ('y', True), ('y', False))
    assert ('x', False) in sm.keys()
    assert ('q', False) not in sm.keys()



def test_series_mapping_c():
    s = Series((10, 20, 30), index=('x', 'y', 'z'))
    assert str(s.via_mapping) == "SeriesMapping({x: 10, y: 20, z: 30})"


def test_series_mapping_d():
    s = Series((10, 20, 30), index=('x', 'y', 'z'))
    with pytest.raises(KeyError):
        _ = s.via_mapping['y':]

    with pytest.raises(KeyError):
        _ = s.via_mapping[s == 20]


#-------------------------------------------------------------------------------

def test_series_mapping_keys_a():
    s = Series((10, 20, 30), index=('x', 'y', 'z'))
    k = s.via_mapping.keys()
    assert list(k) == ['x', 'y', 'z']
    assert 'z' in k
    assert 'a' not in k

def test_series_mapping_keys_b():
    s = Series((10, 20, 30))
    k = s.via_mapping.keys()
    assert list(k) == [0, 1, 2]
    assert 1 in k
    assert 10 not in k

#-------------------------------------------------------------------------------

def test_series_mapping_values_a():
    s = Series((10, 20, 30), index=('x', 'y', 'z'))
    v = s.via_mapping.values()
    assert list(v) == [10, 20, 30]
    assert tuple(v) == (10, 20, 30)
    assert 30 in v

def test_series_mapping_values_b():
    s = Series((10, 20, 30), index=('x', 'y', 'z'))
    assert list(s.via_mapping.items()) == [('x', 10), ('y', 20), ('z', 30)]
    assert len(s.via_mapping.items()) == 3

def test_series_mapping_values_c():
    s = Series(('2022-01-01', '1954-01-01', '1864-05-23'), index=('x', 'y', 'z'), dtype=np.datetime64)
    v = s.via_mapping.values()
    assert list(v) == [np.datetime64('2022-01-01'), np.datetime64('1954-01-01'), np.datetime64('1864-05-23')]
    assert tuple(v) == (np.datetime64('2022-01-01'), np.datetime64('1954-01-01'), np.datetime64('1864-05-23'))
    assert np.datetime64('2022-01-01') in v
    # no conversion to python datetime
    assert np.datetime64('2022-01-02') not in v

def test_series_mapping_values_d():
    s = Series(('2022-01-01', '1864-05-23'), index=('x', 'z'), dtype='datetime64[ns]')
    v = s.via_mapping.values()

    assert list(v) == [np.datetime64('2022-01-01T00:00:00.000000000'), np.datetime64('1864-05-23T00:00:00.000000000')]
    assert np.datetime64('2022-01-01T00:00:00.000000000') in v
    # no conversion to python datetime
    assert np.datetime64('2022-01-01T00:00:00.000000001') not in v

#-------------------------------------------------------------------------------

def test_series_mapping_iter_a():
    s = Series((10, 20, 30), index=('x', 'y', 'z'))
    k = list(iter(s.via_mapping))
    assert k == ['x', 'y', 'z']

#-------------------------------------------------------------------------------

def test_series_mapping_contains_a():
    s = Series((10, 20, 30), index=('x', 'y', 'z'))
    assert 'z' in s.via_mapping
    assert 'q' not in s.via_mapping
