'''
Just a scratch file for my notes etc
'''

import unittest
import pickle

import numpy as np


# assuming located in the same directory
from static_frame import Index
from static_frame import IndexGO
from static_frame import Series
from static_frame import Frame
from static_frame import FrameGO
from static_frame import TypeBlocks
from static_frame import Display
from static_frame import mloc
from static_frame import DisplayConfig


from static_frame.core.util import immutable_filter
from static_frame.core.util import NULL_SLICE

from static_frame.core.index_correspondence import IndexCorrespondence
from static_frame.test.test_case import TestCase
from static_frame.test.test_case import skip_win

from static_frame.core.exception import ErrorInitTypeBlocks
from static_frame.core.exception import AxisInvalid
from static_frame import Index



from static_frame.core.index_correspondence import IndexCorrespondence


nan = np.nan

from static_frame import Frame

f1 = Frame.from_element(1, index=(1,2), columns=(3,4,5))
f2 = FrameGO(index=(1,2))
f3 = Frame.from_element(None, index=(1,2), columns=(3,4,5))
f4 = Frame.from_records([[1,2], [3,4], [5,6]])


f5 = f4.relabel(index=('orange', 'apple', 'banana'), columns=('own', 'want'))
f5.reindex(index=('grape','banana'), columns=('want','own','cost'))
# index_a = Index.from_labels(list('abcde'))
# index_b = Index.from_labels(list('aedf'))
# ic = IndexCorrespondence.from_correspondence(index_a, index_b)

index_4 = Index.from_labels(('orange', 'apple', 'banana'))
index_5 = Index.from_labels(('grape', 'banana'))
ic = IndexCorrespondence.from_correspondence(index_4, index_5)

a1 = np.array([1, 2, 3])
a2 = np.array([False, True, False])
a3 = np.array(['b', 'c', 'd'])
tb1 = TypeBlocks.from_blocks((a1, a2, a3))

a1 = np.array([1, 2, 3])
a2 = np.array([10,50,30])
a3 = np.array([1345,2234,3345])
a4 = np.array([False, True, False])
a5 = np.array([False, False, False])
a6 = np.array(['g', 'd', 'e'])
tb2 = TypeBlocks.from_blocks((a1, a2, a3, a4, a5, a6))

a1 = np.array([[1, 2, 3],[10,50,30],[1345,2234,3345]])
a4 = np.array([False, True, False])
a5 = np.array([False, False, False])
a6 = np.array(['g', 'd', 'e'])
tb21 = TypeBlocks.from_blocks((a1, a4, a5, a6))

# can show that with tb2, a6 remains unchanged

a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
tb3 = TypeBlocks.from_blocks((a1, a2, a3))
tbfruit = TypeBlocks.from_blocks(np.random.random(6).reshape(3,2))


def main():
    print('hello')
    print(f1._blocks)
    print(tb2)
    print(tb21)



if __name__ == "__main__":
    main()
