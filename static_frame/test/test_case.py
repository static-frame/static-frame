
import unittest
import os
import sys
from itertools import zip_longest
import typing as tp
import itertools as it
import string
import cmath
import sqlite3
import contextlib
import tempfile
from pathlib import Path

import numpy as np
import pytest


from static_frame import TypeBlocks
from static_frame.core.container import ContainerOperand
from static_frame.core.index_base import IndexBase
from static_frame.core.frame import Frame
from static_frame.core.util import PathSpecifier


# for running with coverage
# pytest -s --color no --disable-pytest-warnings --cov=static_frame --cov-report html static_frame/test
# for running with native traveback
# pytest -s --color no --disable-pytest-warnings --tb=native


skip_win = pytest.mark.skipif(
        sys.platform == 'win32',
        reason='Windows default dtypes'
        )

@contextlib.contextmanager
def temp_file(suffix: tp.Optional[str] = None,
        path: bool = False
        ) -> tp.Iterator[PathSpecifier]:
    try:
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_name = f.name
        f.close()
        if path:
            yield Path(tmp_name)
        else:
            yield tmp_name
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


class TestCase(unittest.TestCase):
    '''
    TestCase specialized for usage with StaticFrame
    '''

    @staticmethod
    def get_arrays_a() -> tp.Generator[np.ndarray, None , None]:
        '''
        Return sample array suitable for TypeBlock creation, testing. Unique values required.
        '''

        a1 = np.array([[1, 2, 3], [10, 20, 30], [100, 200, 300]])
        a1.flags.writeable = False

        a2 = np.array([[4], [5], [6]])
        a2.flags.writeable = False

        a3 = np.array([[None, 'a', None], ['q', 'x', 'c'], ['f', 'y', 'e']])
        a3.flags.writeable = False

        a4 = np.array([1.2, np.nan, 30.5])
        a4.flags.writeable = False

        for arrays in it.permutations((a1, a2, a3, a4)):
            yield arrays


    @staticmethod
    def get_arrays_b() -> tp.Generator[np.ndarray, None , None]:
        '''
        Return sample array suitable for TypeBlock creation, testing. Many NaNs.
        '''

        a1 = np.array([[1, 2, 3], [10, 20, 30], [100, 200, 300]])
        a1.flags.writeable = False

        a2 = np.array([[4], [5], [6]])
        a2.flags.writeable = False

        a3 = np.array([[None, 'a', None], [None, None, 'c'], ['f', None, 'e']])
        a3.flags.writeable = False

        a4 = np.array([np.nan, np.nan, np.nan])
        a4.flags.writeable = False

        for arrays in it.permutations((a1, a2, a3, a4)):
            yield arrays


    @staticmethod
    def get_letters(*slice_args: tp.Optional[int]) -> tp.Iterator[str]:
        for letter in string.ascii_lowercase[slice(*slice_args)]:
            yield letter

    @staticmethod
    def get_test_input(file_name: str) -> str:
        # input dir should be a sibling of this module
        fp_module = os.path.join(os.getcwd(), __file__)
        fp = os.path.join(os.path.dirname(fp_module), 'input', file_name)
        if not os.path.isfile(fp):
            raise RuntimeError('file not found', fp)
        return fp


    @staticmethod
    def get_containers() -> tp.Iterator[tp.Type[ContainerOperand]]:

        def yield_sub(cls: tp.Type[ContainerOperand]) -> tp.Iterator[tp.Type[ContainerOperand]]:
            for cls in cls.__subclasses__():
                if cls is not IndexBase:
                    yield cls
                if issubclass(cls, ContainerOperand):
                    yield from yield_sub(cls)

        yield from yield_sub(ContainerOperand)

    @staticmethod
    def get_test_db_a() -> sqlite3.Connection:
        conn = sqlite3.connect(':memory:')
        c = conn.cursor()
        c.execute('''CREATE TABLE events
             (date text, identifier text, value real, count int)''')
        for identifier in ('a1', 'b2'):
            for date in ('2006-01-01', '2006-01-02'):
                c.execute(f"INSERT INTO events VALUES ('{date}','{identifier}',12.5,8)")
        conn.commit()
        return conn


    #---------------------------------------------------------------------------

    def assertEqualWithNaN(self,
            v1: object,
            v2: object,
            ) -> None:
        if ((isinstance(v1, float) or isinstance(v1, np.floating))
                and np.isnan(v1)
                and (isinstance(v2, float) or isinstance(v2, np.floating))
                and np.isnan(v2)
                ):
            return

        if ((isinstance(v1, complex) or isinstance(v1, np.complexfloating))
                and cmath.isnan(v1)
                and (isinstance(v2, complex) or isinstance(v1, np.complexfloating))
                and cmath.isnan(v2)  # type: ignore
                ):
            return
        return self.assertEqual(v1, v2)


    def assertAlmostEqualArray(self, a1: np.ndarray, a2: np.ndarray) -> None:
        # NaNs are treated as equal
        np.testing.assert_allclose(a1, a2)
        # np.testing.assert_array_almost_equal(a1, a2, decimal=5)

    def assertTypeBlocksArrayEqual(self,
            tb: TypeBlocks, match: tp.Iterable[object],
            match_dtype: tp.Optional[tp.Union[type, np.dtype, str]] = None) -> None:
        '''
        Args:
            tb: a TypeBlocks instance
            match: can be anything that can be used to create an array.
        '''
        # could use np.testing
        if not isinstance(match, np.ndarray):
            match = np.array(match, dtype=match_dtype)
        self.assertTrue((tb.values == match).all())


    def assertAlmostEqualValues(self,
            values1: tp.Iterable[object], values2: tp.Iterable[object]) -> None:

        for v1, v2 in zip_longest(values1, values2):
            self.assertEqualWithNaN(v1, v2)

    def assertAlmostEqualItems(self,
            pairs1: tp.Iterable[tp.Tuple[tp.Hashable, object]],
            pairs2: tp.Iterable[tp.Tuple[tp.Hashable, object]]) -> None:

        for (k1, v1), (k2, v2) in zip_longest(pairs1, pairs2):
            self.assertEqual(k1, k2)

            if isinstance(v1, float) and np.isnan(v1) and isinstance(v2, float) and np.isnan(v2):
                continue

            self.assertEqual(v1, v2)


    def assertAlmostEqualFramePairs(self,
            pairs1: tp.Iterable[tp.Tuple[tp.Hashable, tp.Iterable[object]]],
            pairs2: tp.Iterable[tp.Tuple[tp.Hashable, tp.Iterable[object]]]) -> None:
        '''
        For comparing nested tuples returned by Frame.to_pairs()
        '''
        for (k1, v1), (k2, v2) in zip_longest(pairs1, pairs2):
            self.assertEqual(k1, k2)
            self.assertAlmostEqualItems(v1, v2)


    def assertEqualFrames(self,
            f1: Frame,
            f2: Frame,
            check_dtypes: bool = True
            ) -> None:
        self.assertEqual(f1.shape, f2.shape)
        self.assertEqual(f1.__class__, f2.__class__)
        self.assertEqual(f1.name, f2.name)

        self.assertEqual(f1.index.depth, f2.index.depth)
        self.assertEqual(f1.index.__class__, f2.index.__class__)

        self.assertEqual(f1.columns.depth, f2.columns.depth)
        self.assertEqual(f1.columns.__class__, f2.columns.__class__)

        # NOTE: this will not handle types
        self.assertAlmostEqualFramePairs(f1.to_pairs(0), f2.to_pairs(0))

        if check_dtypes:
            self.assertTrue((f1.dtypes.values == f2.dtypes.values).all())


    def assertEqualLines(self, lines1: str, lines2: str) -> None:
        '''After splitting and stripping, compare non-empty lines.
        '''
        def clean_lines(lines: str) -> tp.Iterator[str]:
            for line in lines.split('\n'):
                line = line.strip()
                if line:
                    yield line
        for l1, l2 in zip(clean_lines(lines1), clean_lines(lines2)):
            self.assertEqual(l1, l2)


