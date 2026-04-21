from __future__ import annotations

from collections.abc import Mapping

import pytest

from static_frame.core.bus import Bus
from static_frame.core.frame import Frame
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.mfc_mapping import BusMapping, MFCMapping, YarnMapping
from static_frame.core.series import Series
from static_frame.core.store_config import StoreConfigParquet
from static_frame.core.yarn import Yarn
from static_frame.test.test_case import temp_file


def _make_bus() -> Bus:
    f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
    f2 = Frame.from_dict(dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2')
    f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')
    return Bus.from_frames((f1, f2, f3))


def test_mfc_mapping_a() -> None:
    b = _make_bus()
    bm = b.via_mapping
    assert len(bm) == 3
    assert bm['f1'].name == 'f1'
    assert isinstance(bm, Mapping)
    assert isinstance(bm, MFCMapping)
    assert isinstance(bm, BusMapping)


def test_mfc_mapping_b() -> None:
    b = _make_bus()
    bm = b.via_mapping
    assert bm['f1'].equals(
        Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
    )
    assert bm['f3'].shape == (2, 2)


def test_mfc_mapping_c() -> None:
    b = _make_bus()
    bm = b.via_mapping
    assert str(bm) == 'BusMapping({f1: Frame, f2: Frame, f3: Frame})'


def test_mfc_mapping_d() -> None:
    b = _make_bus()
    bm = b.via_mapping

    with pytest.raises(KeyError):
        _ = bm['f1':]  # type: ignore[misc]

    with pytest.raises(KeyError):
        _ = bm[['f1', 'f2']]  # type: ignore[index]


def test_mfc_mapping_e() -> None:
    # Test with IndexHierarchy
    f1 = Frame.from_dict(dict(a=(1, 2)), index=('x', 'y'), name='f1')
    f2 = Frame.from_dict(dict(b=(3, 4)), index=('x', 'y'), name='f2')
    f3 = Frame.from_dict(dict(c=(5, 6)), index=('x', 'y'), name='f3')
    f4 = Frame.from_dict(dict(d=(7, 8)), index=('x', 'y'), name='f4')

    idx = IndexHierarchy.from_product(('a', 'b'), (1, 2))
    b = Bus((f1, f2, f3, f4), index=idx)
    bm = b.via_mapping
    assert len(bm) == 4

    with pytest.raises(KeyError):
        _ = bm['a']

    assert bm[('a', 1)].name == 'f1'
    assert bm[('b', 2)].name == 'f4'

    assert list(bm.keys()) == [('a', 1), ('a', 2), ('b', 1), ('b', 2)]


# -------------------------------------------------------------------------------


def test_mfc_mapping_keys_a() -> None:
    b = _make_bus()
    k = b.via_mapping.keys()
    assert list(k) == ['f1', 'f2', 'f3']
    assert 'f2' in k
    assert 'f99' not in k


def test_mfc_mapping_keys_b() -> None:
    b = _make_bus()
    k = b.via_mapping.keys()
    assert tuple(k) == ('f1', 'f2', 'f3')
    assert len(k) == 3


# -------------------------------------------------------------------------------


def test_mfc_mapping_values_a() -> None:
    b = _make_bus()
    v = b.via_mapping.values()
    frames = list(v)
    assert len(frames) == 3
    assert frames[0].name == 'f1'
    assert frames[1].name == 'f2'
    assert frames[2].name == 'f3'


def test_mfc_mapping_values_b() -> None:
    b = _make_bus()
    v = b.via_mapping.values()
    # containment by equality (frame.equals with name, dtype, class comparison)
    f1 = b['f1']
    assert f1 in v
    # equal frame is also contained
    f_equal = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
    assert f_equal in v
    # different name is not contained
    f_diff_name = Frame.from_dict(
        dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='other'
    )
    assert f_diff_name not in v


def test_mfc_mapping_values_c() -> None:
    b = _make_bus()
    v = b.via_mapping.values()
    assert len(v) == 3


def test_mfc_mapping_views_length_hint_a() -> None:
    b = _make_bus()
    bm = b.via_mapping
    assert bm.keys().__length_hint__() == 3
    assert bm.items().__length_hint__() == 3
    assert bm.values().__length_hint__() == 3


# -------------------------------------------------------------------------------


def test_mfc_mapping_items_a() -> None:
    b = _make_bus()
    pairs = list(b.via_mapping.items())
    assert len(pairs) == 3
    assert pairs[0][0] == 'f1'
    assert pairs[0][1].name == 'f1'
    assert pairs[1][0] == 'f2'
    assert pairs[2][0] == 'f3'


def test_mfc_mapping_items_b1() -> None:
    b = _make_bus()
    im = b.via_mapping.items()
    assert len(im) == 3
    # containment by equality (frame.equals with name, dtype, class comparison)
    f2 = b['f2']
    assert ('f2', f2) in im
    # equal frame is also contained
    f2_equal = Frame.from_dict(
        dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
    )
    assert ('f2', f2_equal) in im
    # frame with different name is not contained
    f_diff_name = Frame.from_dict(
        dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='other'
    )
    assert ('f2', f_diff_name) not in im


def test_mfc_mapping_items_b2() -> None:
    b = _make_bus()
    im = b.via_mapping.items()
    f2 = b['f2']
    assert ('f999', f2) not in im


def test_mfc_mapping_items_c() -> None:
    b = _make_bus()
    im = b.via_mapping.items()
    # non-tuple is not contained
    assert 'f1' not in im  # type: ignore[operator]
    # wrong length tuple is not contained
    assert ('f1',) not in im  # type: ignore[operator]


# -------------------------------------------------------------------------------


def test_mfc_mapping_iter_a() -> None:
    b = _make_bus()
    labels = list(iter(b.via_mapping))
    assert labels == ['f1', 'f2', 'f3']


# -------------------------------------------------------------------------------


def test_mfc_mapping_contains_a() -> None:
    b = _make_bus()
    bm = b.via_mapping
    assert 'f1' in bm
    assert 'f3' in bm
    assert 'f99' not in bm


# -------------------------------------------------------------------------------


def test_mfc_mapping_lazy_a() -> None:
    """MFCMapping preserves the lazy loading paradigm: __getitem__ loads only the
    requested Frame, not all Frames."""
    b = _make_bus()
    with temp_file('.zip') as fp:
        b.to_zip_pickle(fp)
        b2 = Bus.from_zip_pickle(fp)
        assert not b2._loaded.any()

        bm = b2.via_mapping
        # accessing a single frame loads only that frame
        _ = bm['f1']
        assert b2._loaded[0]
        assert not b2._loaded[1]
        assert not b2._loaded[2]

        _ = bm['f3']
        assert b2._loaded[0]
        assert not b2._loaded[1]
        assert b2._loaded[2]


def test_mfc_mapping_lazy_b() -> None:
    """MFCMapping with max_persist respects the max_persist constraint."""
    config = StoreConfigParquet(
        index_depth=1, columns_depth=1, include_columns=True, include_index=True
    )
    b = _make_bus()
    with temp_file('.zip') as fp:
        b.to_zip_parquet(fp, config=config)
        b2 = Bus.from_zip_parquet(fp, config=config, max_persist=1)
        bm = b2.via_mapping

        _ = bm['f1']
        assert b2._loaded.sum() == 1

        _ = bm['f2']
        # with max_persist=1, only f2 should remain loaded
        assert b2._loaded.sum() == 1
        assert b2._loaded[1]
        assert not b2._loaded[0]


# -------------------------------------------------------------------------------


def test_mfc_mapping_reversed_a() -> None:
    b = _make_bus()
    assert list(reversed(b.via_mapping)) == ['f3', 'f2', 'f1']


def test_mfc_mapping_keys_reversed_a() -> None:
    b = _make_bus()
    assert list(reversed(b.via_mapping.keys())) == ['f3', 'f2', 'f1']


def test_mfc_mapping_values_reversed_a() -> None:
    b = _make_bus()
    frames = list(reversed(b.via_mapping.values()))
    assert len(frames) == 3
    assert frames[0].name == 'f3'
    assert frames[1].name == 'f2'
    assert frames[2].name == 'f1'


def test_mfc_mapping_items_reversed_a() -> None:
    b = _make_bus()
    pairs = list(reversed(b.via_mapping.items()))
    assert len(pairs) == 3
    assert pairs[0][0] == 'f3'
    assert pairs[0][1].name == 'f3'
    assert pairs[1][0] == 'f2'
    assert pairs[2][0] == 'f1'


# -------------------------------------------------------------------------------


def _make_yarn() -> Yarn:
    f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
    f2 = Frame.from_dict(dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2')
    f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')
    b = Bus.from_frames((f1, f2, f3))
    return Yarn.from_buses((b,), retain_labels=False)


def test_yarn_mapping_a() -> None:
    y = _make_yarn()
    ym = y.via_mapping
    assert len(ym) == 3
    assert ym['f1'].name == 'f1'
    assert isinstance(ym, Mapping)
    assert isinstance(ym, YarnMapping)


def test_yarn_mapping_b() -> None:
    y = _make_yarn()
    ym = y.via_mapping
    assert ym['f1'].equals(
        Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
    )
    assert ym['f3'].shape == (2, 2)


def test_yarn_mapping_c() -> None:
    y = _make_yarn()
    ym = y.via_mapping
    assert str(ym) == 'YarnMapping({f1: Frame, f2: Frame, f3: Frame})'


def test_yarn_mapping_d() -> None:
    y = _make_yarn()
    ym = y.via_mapping
    with pytest.raises(KeyError):
        _ = ym['f1':]  # type: ignore[misc]

    with pytest.raises(KeyError):
        _ = ym[['f1', 'f2']]  # type: ignore[index]


def test_yarn_mapping_keys_a() -> None:
    y = _make_yarn()
    k = y.via_mapping.keys()
    assert list(k) == ['f1', 'f2', 'f3']
    assert list(reversed(k)) == ['f3', 'f2', 'f1']


def test_yarn_mapping_values_a() -> None:
    y = _make_yarn()
    frames = list(y.via_mapping.values())
    assert len(frames) == 3
    assert frames[0].name == 'f1'
    rev_frames = list(reversed(y.via_mapping.values()))
    assert rev_frames[0].name == 'f3'


def test_yarn_mapping_items_a() -> None:
    y = _make_yarn()
    pairs = list(y.via_mapping.items())
    assert pairs[0][0] == 'f1'
    rev_pairs = list(reversed(y.via_mapping.items()))
    assert rev_pairs[0][0] == 'f3'


def test_yarn_mapping_reversed_a() -> None:
    y = _make_yarn()
    assert list(reversed(y.via_mapping)) == ['f3', 'f2', 'f1']


def test_yarn_mapping_contains_a() -> None:
    y = _make_yarn()
    ym = y.via_mapping
    assert 'f1' in ym
    assert 'f99' not in ym


def test_yarn_mapping_iter_a() -> None:
    y = _make_yarn()
    labels = list(iter(y.via_mapping))
    assert labels == ['f1', 'f2', 'f3']
