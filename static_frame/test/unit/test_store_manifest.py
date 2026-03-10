from __future__ import annotations

import os

# from static_frame.test.test_case import temp_file
from tempfile import TemporaryDirectory

import frame_fixtures as ff
import typing_extensions as tp

from static_frame.core.store import StoreManifest


def test_store_manifest_a() -> None:
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

        assert sm1.read('a').equals(
            f1, compare_class=True, compare_dtype=True, compare_name=True
        )

        assert sm1.read('b').equals(
            f2, compare_class=True, compare_dtype=True, compare_name=True
        )

        post = list(sm1.read_many(('b', 'a')))
        assert post[0].name == 'b'
        assert post[1].name == 'a'
