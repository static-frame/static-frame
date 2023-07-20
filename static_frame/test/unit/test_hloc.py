from __future__ import annotations

import unittest

from static_frame import HLoc
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    def test_hloc_getitem(self) -> None:
        self.assertEqual(HLoc[1].key, (1,))
        self.assertEqual(HLoc[1,].key, (1,))
        self.assertEqual(HLoc[1, 2].key, (1, 2))
        self.assertEqual(HLoc[:, 1].key, (slice(None), 1))
        self.assertEqual(HLoc[:, 1, 2].key, (slice(None), 1, 2))
        self.assertEqual(HLoc[1:2, :].key, (slice(1, 2), slice(None)))
        self.assertEqual(HLoc[1:2, :2].key, (slice(1, 2), slice(None, 2)))
        self.assertEqual(HLoc[1:2, :2, :3].key, (slice(1, 2), slice(None, 2), slice(None, 3)))
        self.assertEqual(HLoc[::1].key, (slice(None, None, 1),))
        self.assertEqual(HLoc[1:, 1:2, 1:2:3, :2, :2:3, ::3].key, (slice(1, None), slice(1, 2), slice(1, 2, 3), slice(None, 2), slice(None, 2, 3), slice(None, None, 3)))
        self.assertEqual(HLoc[()].key, ((),))
        self.assertEqual(HLoc[(1,)].key, (1,))
        self.assertEqual(HLoc[(),].key, ((),))
        self.assertEqual(HLoc[:].key, (slice(None),))
        self.assertEqual(HLoc[:, :].key, (slice(None), slice(None)))
        self.assertEqual(HLoc[:, :, 4].key, (slice(None), slice(None), 4))

    def test_hloc_repr(self) -> None:
        self.assertEqual(repr(HLoc[1]), '<HLoc[1]>')
        self.assertEqual(repr(HLoc[1,]), '<HLoc[1]>')
        self.assertEqual(repr(HLoc[1, 2]), '<HLoc[1,2]>')
        self.assertEqual(repr(HLoc[:, 1]), '<HLoc[:,1]>')
        self.assertEqual(repr(HLoc[:, 1, 2]), '<HLoc[:,1,2]>')
        self.assertEqual(repr(HLoc[1:2, :]), '<HLoc[1:2,:]>')
        self.assertEqual(repr(HLoc[1:2, :2]), '<HLoc[1:2,:2]>')
        self.assertEqual(repr(HLoc[1:2, :2, :3]), '<HLoc[1:2,:2,:3]>')
        self.assertEqual(repr(HLoc[::1]), '<HLoc[:]>')
        self.assertEqual(repr(HLoc[1:, 1:2, 1:2:3, :2, :2:3, ::3]), '<HLoc[1:,1:2,1:2:3,:2,:2:3,::3]>')
        self.assertEqual(repr(HLoc[()]), '<HLoc[()]>')
        self.assertEqual(repr(HLoc[(1,),]), '<HLoc[(1,)]>')
        self.assertEqual(repr(HLoc[(),]), '<HLoc[()]>')
        self.assertEqual(repr(HLoc[:]), '<HLoc[:]>')
        self.assertEqual(repr(HLoc[:, :]), '<HLoc[:,:]>')
        self.assertEqual(repr(HLoc[:, :, 4]), '<HLoc[:,:,4]>')


if __name__ == '__main__':
    unittest.main()
