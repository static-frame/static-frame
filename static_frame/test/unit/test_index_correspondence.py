




import unittest


import numpy as np


from static_frame.core.index_correspondence import IndexCorrespondence
from static_frame.core.index import Index

from static_frame.test.test_case import TestCase


class TestUnit(TestCase):


    def test_index_correspondence_a(self) -> None:
        idx0 = Index([0, 1, 2, 3, 4], loc_is_iloc=True)
        idx1 = Index([0, 1, 2, 3, 4, '100185', '100828', '101376', '100312', '101092'], dtype=object)
        ic = IndexCorrespondence.from_correspondence(idx0, idx1)
        self.assertFalse(ic.is_subset)
        self.assertTrue(ic.has_common)
        # this is an array, due to loc_is_iloc being True
        assert isinstance(ic.iloc_src, np.ndarray)
        self.assertEqual(ic.iloc_src.tolist(),
                [0, 1, 2, 3, 4]
                )
        self.assertEqual(ic.iloc_dst,
                [0, 1, 2, 3, 4]
                )


    def test_index_correspondence_b(self) -> None:
        # issue found with a hypothesis test

        idx = Index([False], loc_is_iloc=False)
        ic = IndexCorrespondence.from_correspondence(idx, idx)
        self.assertTrue(ic.is_subset)
        self.assertTrue(ic.has_common)
        self.assertEqual(ic.size, 1)
        self.assertEqual(ic.iloc_src, [0]) # this is as list in this use case
        self.assertEqual(ic.iloc_dst.tolist(), [0]) # type: ignore



if __name__ == '__main__':
    unittest.main()
