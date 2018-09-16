from itertools import zip_longest
from itertools import combinations
import unittest
from collections import OrderedDict
from io import StringIO
import string
import hashlib

import numpy as np

from static_frame.test.test_case import TestCase

import static_frame as sf
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


nan = np.nan



class TestUnit(TestCase):

    def test_demo(self):
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        value1 = Series((100, 200, 300), index=('s', 'u', 't'))

        value2 = Frame.from_records(((20, 21, 22), (23, 24, 25)),
                index=('x', 'y'),
                columns=('s', 't', 'w'))

        f2 = FrameGO([
                [np.nan, 2, 3, 0],
                [3, 4, np.nan, 1],
                [0, 1, 2, 3]],
                columns=list('ABCD'))

        f3 = FrameGO([
                [np.nan, 2, 3, 0],
                [3, 4, np.nan, 1],
                [0, 1, 2, 3]],
                columns=list('EFGH'))


