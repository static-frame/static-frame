

import unittest
import frame_fixtures as ff


from static_frame import Series
from static_frame import SeriesHE
from static_frame.test.test_case import TestCase
from static_frame import ILoc

class TestUnit(TestCase):

    def test_frame_he_slotted_a(self) -> None:

        f1 = SeriesHE.from_element(1, index=(1,2))

        with self.assertRaises(AttributeError):
            f1.g = 30 #type: ignore #pylint: disable=E0237
        with self.assertRaises(AttributeError):
            f1.__dict__ #pylint: disable=W0104


    #---------------------------------------------------------------------------
    def test_series_he_hash_a(self) -> None:

        s1 = SeriesHE(('a', 'b', 'c'))
        s2 = SeriesHE(('a', 'b', 'c'))
        s3 = SeriesHE(('a', 'b', 'c'), index=tuple('xyz'))
        s4 = SeriesHE(('p', 'q', 'r'))
        s5 = Series(('a', 'b', 'c'))

        self.assertTrue(s1 == s2)
        # we do not compare container class, so
        self.assertTrue(s1 == s5)
        self.assertTrue(isinstance(s5 == s5, Series))

        self.assertTrue(isinstance(hash(s1), int))
        self.assertEqual(hash(s1), hash(s2))

        # has is based on index labels
        self.assertEqual(hash(s1), hash(s4))
        self.assertNotEqual(hash(s1), hash(s3))

        q = {s1, s2, s3}
        self.assertEqual(len(q), 2)
        self.assertTrue(s1 in q)
        self.assertTrue(s2 in q)
        self.assertTrue(s4 not in q)
        with self.assertRaises(TypeError):
            _ = s5 in q

        d = {s1: 'foo', s3: 'bar'}
        self.assertTrue(s2 in d)
        self.assertEqual(d[s1], 'foo')

        with self.assertRaises(TypeError):
            _ = s5 in d

        s6 = s1.to_series()

        with self.assertRaises(TypeError):
            _ = s6 in d


    def test_series_he_hash_b(self) -> None:

        s1 = SeriesHE.from_dict(dict(a=10, b=42))
        s2 = SeriesHE.from_dict(dict(x='foo', y='bar'), name=s1)

        self.assertEqual(s2.name['b'], 42)

        s3 = Series.from_dict({s1: 40, s2: 1000})

        self.assertEqual(s3[s2], 1000)
        self.assertFalse(s1 == s2) # name attr is different


    def test_series_he_hash_c(self) -> None:

        s1 = ff.parse('s(10,1)|i(I,str)')[ILoc[0]].to_series_he()
        self.assertFalse(hasattr(s1, '_hash'))
        self.assertEqual(hash(s1), s1._hash)


if __name__ == '__main__':
    unittest.main()

