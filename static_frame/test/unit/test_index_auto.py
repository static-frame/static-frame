from __future__ import annotations

import datetime

import frame_fixtures as ff
import numpy as np

from static_frame.core.index import Index
from static_frame.core.index import IndexGO
from static_frame.core.index_auto import IndexAutoConstructorFactory
from static_frame.core.index_auto import IndexAutoFactory
from static_frame.core.index_auto import IndexDefaultConstructorFactory
from static_frame.core.index_datetime import IndexDate
from static_frame.core.index_datetime import IndexYearMonth
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    def test_index_auto_factory_a(self) -> None:

        idx1 = IndexAutoFactory.from_optional_constructor(4,
                default_constructor=Index)
        self.assertEqual(idx1._map is None, True) #type: ignore
        self.assertEqual(len(idx1), 4)
        self.assertEqual(idx1.STATIC, True)

    def test_index_auto_factory_b(self) -> None:

        idx1 = IndexAutoFactory.from_optional_constructor(8,
                default_constructor=IndexGO)
        self.assertEqual(idx1._map is None, True) #type: ignore
        self.assertEqual(len(idx1), 8)
        self.assertEqual(idx1.STATIC, False)

        # go funcitonality
        assert isinstance(idx1, IndexGO)
        idx1.append(8)
        self.assertEqual(idx1._map is None, True)
        self.assertEqual(len(idx1), 9)

    def test_index_auto_factory_c(self) -> None:

        idx1 = IndexAutoFactory.from_optional_constructor(5,
                default_constructor=IndexGO,
                explicit_constructor=Index)
        self.assertTrue(idx1._map is None) #type: ignore
        self.assertEqual(len(idx1), 5)
        self.assertEqual(idx1.STATIC, True)

    def test_index_auto_factory_from_optional_constructor(self) -> None:
        initializer = 3
        explicit_constructor = IndexDefaultConstructorFactory(name='foo')
        default_constructor = IndexDate
        post = IndexAutoFactory.from_optional_constructor(
                initializer=initializer,
                explicit_constructor=explicit_constructor,
                default_constructor=default_constructor,
                )
        self.assertEqual(post.name, 'foo')
        self.assertEqual(post.values.tolist(),
            [datetime.date(1970, 1, 1), datetime.date(1970, 1, 2), datetime.date(1970, 1, 3)])

    def test_index_auto_constructor_a(self) -> None:
        a1 = np.array(('2021-05',), dtype=np.datetime64)
        self.assertEqual(
                IndexAutoConstructorFactory.to_index(
                        a1, default_constructor=Index).__class__,
                IndexYearMonth,
                )

    def test_index_auto_constructor_b(self) -> None:
        a1 = np.array(('2021-05',), dtype=np.datetime64)
        idx = IndexAutoConstructorFactory('foo')(a1,
                default_constructor=Index)
        self.assertEqual(idx.name, 'foo')
        self.assertEqual(idx.__class__, IndexYearMonth)


    def test_index_auto_constructor_c(self) -> None:
        f = ff.parse('v(dtD)|s(3,1)').set_index(0,
                index_constructor=IndexAutoConstructorFactory,
                )
        self.assertEqual(f.index.name, 0)
        self.assertEqual(f.index.__class__, IndexDate)

    def test_index_auto_constructor_d(self) -> None:
        f = ff.parse('v(dtD)|s(3,1)').set_index(0,
                index_constructor=IndexAutoConstructorFactory('foo'),
                )
        self.assertEqual(f.index.name, 'foo')
        self.assertEqual(f.index.__class__, IndexDate)

    #---------------------------------------------------------------------------
    def test_index_auto_factory_equals_a(self) -> None:

        idx1 = IndexAutoFactory.from_optional_constructor(10_000,
                default_constructor=Index)
        idx2 = IndexAutoFactory.from_optional_constructor(10_000,
                default_constructor=Index)
        self.assertTrue(idx1.equals(idx2))


if __name__ == '__main__':
    import unittest
    unittest.main()
