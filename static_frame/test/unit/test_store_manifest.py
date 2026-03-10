from __future__ import annotations

import gc
import os
import time
from tempfile import TemporaryDirectory

import frame_fixtures as ff
import numpy as np
import pytest
import typing_extensions as tp

from static_frame.core.exception import StoreFileMutation
from static_frame.core.store import StoreManifest


def _frame_equal(a: tp.Any, b: tp.Any) -> bool:
    return a.equals(b, compare_class=True, compare_dtype=True, compare_name=True)


def test_store_manifest_npz_a() -> None:
    with TemporaryDirectory() as fp:
        f1 = ff.parse('s(4,4)|i(ID,dtD)|v(int)').rename('a')
        f2 = ff.parse('s(4,4)|i(ID,dtD)|v(int)').rename('b')

        label_to_fp = {}
        for f in (f1, f2):
            fpf = os.path.join(fp, (f.name + '.npz'))
            f.to_npz(fpf)
            label_to_fp[f.name] = fpf

        sm1 = StoreManifest(label_to_fp)
        assert list(sm1.labels()) == ['a', 'b']

        assert _frame_equal(sm1.read('a'), f1)
        assert _frame_equal(sm1.read('b'), f2)

        post = list(sm1.read_many(('b', 'a')))
        assert post[0].name == 'b'
        assert post[1].name == 'a'


def test_store_manifest_pickle_a() -> None:
    with TemporaryDirectory() as fp:
        f1 = ff.parse('s(4,4)|i(ID,dtD)|v(int)').rename('x')
        f2 = ff.parse('s(3,2)|v(float)').rename('y')

        label_to_fp = {}
        for f in (f1, f2):
            fpf = os.path.join(fp, (f.name + '.pickle'))
            f.to_pickle(fpf)
            label_to_fp[f.name] = fpf

        sm = StoreManifest(label_to_fp)
        assert list(sm.labels()) == ['x', 'y']
        assert _frame_equal(sm.read('x'), f1)
        assert _frame_equal(sm.read('y'), f2)


def test_store_manifest_npy_a() -> None:
    with TemporaryDirectory() as fp:
        f1 = ff.parse('s(4,4)|i(ID,dtD)|v(int)').rename('n1')
        f2 = ff.parse('s(3,2)|v(float)').rename('n2')

        label_to_fp = {}
        for f in (f1, f2):
            fpf = os.path.join(fp, f.name)
            f.to_npy(fpf)
            label_to_fp[f.name] = fpf

        sm = StoreManifest(label_to_fp)
        assert list(sm.labels()) == ['n1', 'n2']
        assert _frame_equal(sm.read('n1'), f1)
        assert _frame_equal(sm.read('n2'), f2)


def test_store_manifest_npy_read_many() -> None:
    with TemporaryDirectory() as fp:
        f1 = ff.parse('s(2,3)|v(int)').rename('a')
        f2 = ff.parse('s(4,2)|v(float)').rename('b')

        label_to_fp = {}
        for f in (f1, f2):
            fpf = os.path.join(fp, f.name)
            f.to_npy(fpf)
            label_to_fp[f.name] = fpf

        sm = StoreManifest(label_to_fp)
        post = list(sm.read_many(('b', 'a')))
        assert post[0].name == 'b'
        assert post[1].name == 'a'


def test_store_manifest_all_three_formats() -> None:
    with TemporaryDirectory() as fp:
        f1 = ff.parse('s(3,3)|v(int)').rename('npy_f')
        f2 = ff.parse('s(2,2)|v(float)').rename('npz_f')
        f3 = ff.parse('s(4,4)|v(int)').rename('pkl_f')

        fp1 = os.path.join(fp, 'npy_f')
        f1.to_npy(fp1)
        fp2 = os.path.join(fp, 'npz_f.npz')
        f2.to_npz(fp2)
        fp3 = os.path.join(fp, 'pkl_f.pickle')
        f3.to_pickle(fp3)

        sm = StoreManifest({'npy_f': fp1, 'npz_f': fp2, 'pkl_f': fp3})
        assert _frame_equal(sm.read('npy_f'), f1)
        assert _frame_equal(sm.read('npz_f'), f2)
        assert _frame_equal(sm.read('pkl_f'), f3)

        post = list(sm.read_many(('pkl_f', 'npy_f', 'npz_f')))
        assert post[0].name == 'pkl_f'
        assert post[1].name == 'npy_f'
        assert post[2].name == 'npz_f'


def test_store_manifest_int_labels() -> None:
    with TemporaryDirectory() as fp:
        f1 = ff.parse('s(3,3)|v(int)').rename('a')
        f2 = ff.parse('s(2,2)|v(float)').rename('b')

        fp1 = os.path.join(fp, 'a.npz')
        f1.to_npz(fp1)
        fp2 = os.path.join(fp, 'b.npz')
        f2.to_npz(fp2)

        sm = StoreManifest({0: fp1, 1: fp2})
        assert list(sm.labels()) == [0, 1]
        assert _frame_equal(sm.read(0), f1)
        assert _frame_equal(sm.read(1), f2)


def test_store_manifest_tuple_labels() -> None:
    with TemporaryDirectory() as fp:
        f1 = ff.parse('s(3,3)|v(int)').rename('a')
        f2 = ff.parse('s(2,2)|v(float)').rename('b')

        fp1 = os.path.join(fp, 'a.pickle')
        f1.to_pickle(fp1)
        fp2 = os.path.join(fp, 'b.pickle')
        f2.to_pickle(fp2)

        sm = StoreManifest({('x', 1): fp1, ('y', 2): fp2})
        assert list(sm.labels()) == [('x', 1), ('y', 2)]
        assert _frame_equal(sm.read(('x', 1)), f1)
        assert _frame_equal(sm.read(('y', 2)), f2)


def test_store_manifest_date_labels() -> None:
    with TemporaryDirectory() as fp:
        f1 = ff.parse('s(3,3)|v(int)').rename('a')
        f2 = ff.parse('s(2,2)|v(float)').rename('b')

        fp1 = os.path.join(fp, 'a.npz')
        f1.to_npz(fp1)
        fp2 = os.path.join(fp, 'b.npz')
        f2.to_npz(fp2)

        import datetime

        d1 = datetime.date(2020, 1, 1)
        d2 = datetime.date(2021, 6, 15)
        sm = StoreManifest({d1: fp1, d2: fp2})
        assert list(sm.labels()) == [d1, d2]
        assert _frame_equal(sm.read(d1), f1)
        assert _frame_equal(sm.read(d2), f2)


def test_store_manifest_labels_strip_ext_non_string() -> None:
    """Non-string labels should not be stripped of extensions."""
    with TemporaryDirectory() as fp:
        f1 = ff.parse('s(3,3)|v(int)').rename('a')
        fp1 = os.path.join(fp, 'a.npz')
        f1.to_npz(fp1)

        sm = StoreManifest({42: fp1})
        # strip_ext=True should not affect non-string labels
        assert list(sm.labels(strip_ext=True)) == [42]
        assert list(sm.labels(strip_ext=False)) == [42]


def test_store_manifest_labels_strip_ext_string() -> None:
    """String labels with file extensions should be stripped when strip_ext=True."""
    with TemporaryDirectory() as fp:
        f1 = ff.parse('s(3,3)|v(int)').rename('a')
        fp1 = os.path.join(fp, 'a.npz')
        f1.to_npz(fp1)

        sm = StoreManifest({'a.npz': fp1})
        assert list(sm.labels(strip_ext=True)) == ['a']
        assert list(sm.labels(strip_ext=False)) == ['a.npz']


def test_store_manifest_mtime_update_sets_times() -> None:
    with TemporaryDirectory() as fp:
        f1 = ff.parse('s(3,3)|v(int)').rename('a')
        fp1 = os.path.join(fp, 'a.npz')
        f1.to_npz(fp1)

        sm = StoreManifest({'a': fp1})
        # after init, _label_to_last_modified should have the mtime
        assert not np.isnan(sm._label_to_last_modified['a'])
        assert sm._label_to_last_modified['a'] == os.path.getmtime(fp1)


def test_store_manifest_mtime_coherent_ok() -> None:
    """No error when files have not changed."""
    with TemporaryDirectory() as fp:
        f1 = ff.parse('s(3,3)|v(int)').rename('a')
        fp1 = os.path.join(fp, 'a.npz')
        f1.to_npz(fp1)

        sm = StoreManifest({'a': fp1})
        # should not raise
        sm._mtime_coherent()


def test_store_manifest_mtime_coherent_file_changed() -> None:
    """Raise StoreFileMutation when file is modified after init."""
    with TemporaryDirectory() as fp:
        f1 = ff.parse('s(3,3)|v(int)').rename('a')
        fp1 = os.path.join(fp, 'a.npz')
        f1.to_npz(fp1)

        sm = StoreManifest({'a': fp1})
        # modify the file
        time.sleep(0.05)
        f1.to_npz(fp1)

        with pytest.raises(StoreFileMutation):
            sm._mtime_coherent()


def test_store_manifest_mtime_coherent_file_deleted() -> None:
    """Raise StoreFileMutation when a tracked file is deleted."""
    with TemporaryDirectory() as fp:
        f1 = ff.parse('s(3,3)|v(int)').rename('a')
        fp1 = os.path.join(fp, 'a.npz')
        f1.to_npz(fp1)

        sm = StoreManifest({'a': fp1})
        os.remove(fp1)

        with pytest.raises(StoreFileMutation):
            sm._mtime_coherent()


def test_store_manifest_read_raises_after_mutation() -> None:
    """read() uses store_coherent_non_write, so it should raise on mutation."""
    with TemporaryDirectory() as fp:
        f1 = ff.parse('s(3,3)|v(int)').rename('a')
        fp1 = os.path.join(fp, 'a.npz')
        f1.to_npz(fp1)

        sm = StoreManifest({'a': fp1})
        time.sleep(0.05)
        f1.to_npz(fp1)

        with pytest.raises(StoreFileMutation):
            sm.read('a')


def test_store_manifest_labels_raises_after_mutation() -> None:
    """labels() uses store_coherent_non_write, so it should raise on mutation."""
    with TemporaryDirectory() as fp:
        f1 = ff.parse('s(3,3)|v(int)').rename('a')
        fp1 = os.path.join(fp, 'a.npz')
        f1.to_npz(fp1)

        sm = StoreManifest({'a': fp1})
        time.sleep(0.05)
        f1.to_npz(fp1)

        with pytest.raises(StoreFileMutation):
            list(sm.labels())


def test_store_manifest_weak_cache_returns_same_object() -> None:
    """Reading the same label twice should return the same cached object."""
    with TemporaryDirectory() as fp:
        f1 = ff.parse('s(3,3)|v(int)').rename('a')
        fp1 = os.path.join(fp, 'a.npz')
        f1.to_npz(fp1)

        sm = StoreManifest({'a': fp1})
        r1 = sm.read('a')
        r2 = sm.read('a')
        assert r1 is r2


def test_store_manifest_weak_cache_evicted_on_no_reference() -> None:
    """When all external references to a frame are dropped, the weak cache should evict it."""
    with TemporaryDirectory() as fp:
        f1 = ff.parse('s(3,3)|v(int)').rename('a')
        fp1 = os.path.join(fp, 'a.npz')
        f1.to_npz(fp1)

        sm = StoreManifest({'a': fp1})
        result = sm.read('a')
        assert len(sm._weak_cache) == 1
        del result
        gc.collect()
        assert len(sm._weak_cache) == 0


def test_store_manifest_weak_cache_multiple_labels() -> None:
    """Cache should hold entries for each label read."""
    with TemporaryDirectory() as fp:
        f1 = ff.parse('s(3,3)|v(int)').rename('a')
        f2 = ff.parse('s(2,2)|v(float)').rename('b')
        fp1 = os.path.join(fp, 'a.npz')
        fp2 = os.path.join(fp, 'b.pickle')
        f1.to_npz(fp1)
        f2.to_pickle(fp2)

        sm = StoreManifest({'a': fp1, 'b': fp2})
        r1 = sm.read('a')
        r2 = sm.read('b')
        assert len(sm._weak_cache) == 2

        del r1
        gc.collect()
        assert len(sm._weak_cache) == 1
        assert 'b' in sm._weak_cache


def test_store_manifest_weak_cache_npy() -> None:
    """NPY frames should also be cached."""
    with TemporaryDirectory() as fp:
        f1 = ff.parse('s(3,3)|v(int)').rename('a')
        fp1 = os.path.join(fp, 'a')
        f1.to_npy(fp1)

        sm = StoreManifest({'a': fp1})
        assert len(sm._weak_cache) == 0
        result = sm.read('a')
        assert len(sm._weak_cache) == 1
        assert sm._weak_cache['a'] is result
