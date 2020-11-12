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


def main():
    pass
