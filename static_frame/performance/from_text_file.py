import pathlib
import tempfile
import typing as tp
import getpass
import shutil
import string
import itertools
import random
import sys

import numpy as np
import static_frame as sf

from static_frame.performance.perf_test import PerfTest


class SampleData:
    '''An instance masquerading as a class!'''
    _store: tp.Dict[str, tp.Any] = {}
    _td = pathlib.Path(tempfile.gettempdir()) / f'{__name__}-of-{getpass.getuser()}'

    @staticmethod
    def _get_random_strings(count: int, min_len=1, max_len=20):
        s = []
        for i in range(count):
            s.append(''.join(random.choice(string.printable) for i in range(random.randint(min_len, max_len))))
        return s

    @classmethod
    def _string_series(cls, n_elements: int, min_len=1, max_len=20):
        return sf.Series(cls._get_random_strings(n_elements, min_len, max_len))

    @classmethod
    def _int_series(cls, n_elements: int, min_=sys.maxsize*-1, max_=sys.maxsize):
        a = np.random.randint(low=min_, high=max_, size=n_elements)
        a.flags.writeable = False
        return sf.Series(a)

    @classmethod
    def _float_series(cls, n_elements: int, min_=sys.maxsize*-1, max_=sys.maxsize):
        a = cls._int_series(n_elements, min_, max_)
        return sf.Series(a*np.random.rand(*a.shape))

    @classmethod
    def random_frame(cls, n_rows: int, n_cols: int):
        col_constructors = (cls._string_series, cls._int_series, cls._float_series)
        series = [random.choice(col_constructors)(n_rows) for i in range(n_cols)]
        return sf.Frame.from_concat(series, columns=cls._get_random_strings(n_cols), axis=1)

    @classmethod
    def create(cls) -> None:
        '''Aka __init__'''
        shutil.rmtree(cls._td, ignore_errors=True)
        cls._td.mkdir()

        cls.r1000c5_no_i = cls._td / 'r1000c5_no_i.tsv'
        f = cls.random_frame(1000, 5)
        f.to_tsv(cls.r1000c5_no_i, include_index=False)

        # cls.r100000c50_no_i = cls._td / 'r100000c50_no_i.tsv'
        # f = cls.random_frame(100000, 50)
        # f.to_tsv(cls.r100000c50_no_i, include_index=False)

        # cls.r1000c50000_no_i = cls._td / 'r1000c50000_no_i.tsv'
        # f = cls.random_frame(100000, 50)
        # f.to_tsv(cls.r1000c50000_no_i, include_index=False)


