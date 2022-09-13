import unittest
from sys import getsizeof

import frame_fixtures as ff

from static_frame.test.test_case import TestCase


class TestUnit(TestCase):
    def test_simple(self) -> None:
        f = ff.parse('s(3,4)')
        self.assertEqual(getsizeof(f), sum(getsizeof(e) for e in (
            f._blocks,
            f._columns,
            f._index,
            f._name,
            None # for Frame instance garbage collector overhead
        )))

    def test_string_index(self) -> None:
        f = ff.parse('s(3,4)|i(I,str)|c(I,str)')
        self.assertEqual(getsizeof(f), sum(getsizeof(e) for e in (
            f._blocks,
            f._columns,
            f._index,
            f._name,
            None # for Frame instance garbage collector overhead
        )))

    def test_object_index(self) -> None:
        f = ff.parse('s(8,12)|i(I,object)|c(I,str)')
        self.assertEqual(getsizeof(f), sum(getsizeof(e) for e in (
            f._blocks,
            f._columns,
            f._index,
            f._name,
            None # for Frame instance garbage collector overhead
        )))

    def test_multiple_value_types(self) -> None:
        f = ff.parse('s(8,4)|i(I,object)|c(I,str)|v(object,int,bool,str)')
        self.assertEqual(getsizeof(f), sum(getsizeof(e) for e in (
            f._blocks,
            f._columns,
            f._index,
            f._name,
            None # for Frame instance garbage collector overhead
        )))

    def test_frame_he_before_hash(self) -> None:
        f = ff.parse('s(3,4)').to_frame_he()
        self.assertEqual(getsizeof(f), sum(getsizeof(e) for e in (
            f._blocks,
            f._columns,
            f._index,
            f._name,
            # f._hash, # has not been initialized yet
            None # for FrameHE instance garbage collector overhead
        )))

    def test_frame_he_after_hash(self) -> None:
        f = ff.parse('s(3,4)').to_frame_he()
        hash(f) # to initialize _hash
        self.assertEqual(getsizeof(f), sum(getsizeof(e) for e in (
            f._blocks,
            f._columns,
            f._index,
            f._name,
            f._hash,
            None # for FrameHE instance garbage collector overhead
        )))

if __name__ == '__main__':
    unittest.main()