# pylint: disable=comparison-overlap
import unittest

from static_frame import HLoc
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    def test_hloc_getitem(self) -> None:
        assert HLoc[1].key == (1,)
        assert HLoc[1,].key == (1,)
        assert HLoc[1, 2].key == (1, 2)
        assert HLoc[:,1].key == (slice(None), 1)
        assert HLoc[:,1,2].key == (slice(None), 1, 2)
        assert HLoc[1:2,:].key == (slice(1, 2), slice(None))
        assert HLoc[1:2,:2].key == (slice(1, 2), slice(None, 2))
        assert HLoc[1:2,:2,:3].key == (slice(1, 2), slice(None, 2), slice(None, 3))
        assert HLoc[::1].key == (slice(None, None, 1),)
        assert HLoc[1:, 1:2, 1:2:3, :2, :2:3, ::3].key == (slice(1, None), slice(1, 2), slice(1, 2, 3), slice(None, 2), slice(None, 2, 3), slice(None, None, 3))
        assert HLoc[()].key == ((),)
        assert HLoc[(1,)].key == (1,)
        assert HLoc[(),].key == ((),)
        assert HLoc[:].key == (slice(None),)
        assert HLoc[:,:].key == (slice(None), slice(None))
        assert HLoc[:, :, 4].key == (slice(None), slice(None), 4)

    def test_hloc_repr(self) -> None:
        assert repr(HLoc[1]) == 'HLoc[1]'
        assert repr(HLoc[1,]) == 'HLoc[1]'
        assert repr(HLoc[1, 2]) == 'HLoc[1, 2]'
        assert repr(HLoc[:,1]) == 'HLoc[:,1]'
        assert repr(HLoc[:,1,2]) == 'HLoc[:,1,2]'
        assert repr(HLoc[1:2,:]) == 'HLoc[1:2,:]'
        assert repr(HLoc[1:2,:2]) == 'HLoc[1:2,:2]'
        assert repr(HLoc[1:2,:2,:3]) == 'HLoc[1:2,:2,:3]'
        assert repr(HLoc[::1]) == 'HLoc[:]'
        assert repr(HLoc[1:,1:2,1:2:3,:2,:2:3,::3]) == 'HLoc[1:,1:2,1:2:3,:2,:2:3,::3]'
        assert repr(HLoc[()]) == 'HLoc[()]'
        assert repr(HLoc[(1,),]) == 'HLoc[(1,)]'
        assert repr(HLoc[(),]) == 'HLoc[()]'
        assert repr(HLoc[:]) == 'HLoc[:]'
        assert repr(HLoc[:,:]) == 'HLoc[:,:]'
        assert repr(HLoc[:, :, 4]) == 'HLoc[:,:,4]'


if __name__ == '__main__':
    unittest.main()
