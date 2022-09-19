import unittest
from sys import getsizeof

import typing as tp
import numpy as np

import frame_fixtures as ff

from static_frame import Index
from static_frame import IndexHierarchy
from static_frame import Frame
from static_frame import Series
from static_frame import Bus
from static_frame import StoreConfig
from static_frame.core.util import getsizeof_total
from static_frame.test.test_case import TestCase
from static_frame.test.test_case import temp_file


class TestUnit(TestCase):
    def test_simple_index_hierarchy(self) -> None:
        idxa = Index(('a', 'b', 'c'))
        idxb = Index((1, 2, 3))
        idx = IndexHierarchy.from_product(idxa, idxb)
        seen = set()
        self.assertEqual(getsizeof_total(idx), sum(getsizeof_total(e, seen=seen) for e in (
            idx._indices,
            idx._indexers,
            idx._name,
            idx._blocks,
            idx._recache,
            idx._values,
            idx._map,
            idx._index_types,
            idx._pending_extensions,
        )) + getsizeof(idx))

    def test_simple_bus(self) -> None:
        f1 = ff.parse('s(3,6)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        b = Bus.from_frames((f1, f2))
        seen = set()
        self.assertEqual(getsizeof_total(b), sum(getsizeof_total(e, seen=seen) for e in (
            b._loaded,
            b._loaded_all,
            b._values_mutable,
            b._index,
            b._name,
            b._store,
            b._config,
            # b._last_accessed, # not initialized, not a "max_persist" bus
            b._max_persist,
        )) + getsizeof(b))

    def test_maxpersist_bus(self) -> None:
        def items() -> tp.Iterator[tp.Tuple[str, Frame]]:
            for i in range(20):
                yield str(i), Frame(np.arange(i, i+10).reshape(2, 5))

        s = Series.from_items(items(), dtype=object)
        b1 = Bus.from_series(s)

        config = StoreConfig(
                index_depth=1,
                columns_depth=1,
                include_columns=True,
                include_index=True
                )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)

            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=3)

            seen = set()
            self.assertEqual(getsizeof_total(b2), sum(getsizeof_total(e, seen=seen) for e in (
                b2._loaded,
                b2._loaded_all,
                b2._values_mutable,
                b2._index,
                b2._name,
                b2._store,
                b2._config,
                b2._last_accessed,
                b2._max_persist,
            )) + getsizeof(b2))


if __name__ == '__main__':
    unittest.main()
