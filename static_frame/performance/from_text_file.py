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
import pandas as pd

from static_frame.performance.perf_test import PerfTest


class SampleData:
    '''An instance masquerading as a class!'''
    _store: tp.Dict[str, tp.Any] = {}
    _td = pathlib.Path(tempfile.gettempdir()) / f'{__name__}-of-{getpass.getuser()}'

    @staticmethod
    def _get_random_strings(count: int, min_len=1, max_len=20, unique=False):
        s = []
        seen = set()
        for i in range(count):
            new_value = ''.join(random.choice(string.ascii_letters) for i in range(random.randint(min_len, max_len)))
            while new_value in seen:
                new_value = ''.join(random.choice(string.ascii_letters) for i in range(random.randint(min_len, max_len)))

            s.append(new_value)
            seen.add(new_value)
        return s

    @classmethod
    def _string_series(cls, n_elements: int, min_len=1, max_len=20, unique=False):
        return sf.Series(cls._get_random_strings(n_elements, min_len, max_len, unique))

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
        return sf.Frame.from_concat(series, columns=cls._get_random_strings(n_cols, unique=True), axis=1)

    @classmethod
    def create(cls) -> None:
        '''Aka __init__'''
        shutil.rmtree(cls._td, ignore_errors=True)
        cls._td.mkdir()
        print(cls._td)

        cls.frame_r1000c5_no_i = cls.random_frame(1000, 5)
        cls.path_r1000c5_tsv = cls._td / 'r1000c5.tsv'
        cls.path_r1000c5_csv = cls._td / 'r1000c5.csv'
        cls.frame_r1000c5_no_i.to_tsv(cls.path_r1000c5_tsv, include_index=False)
        cls.frame_r1000c5_no_i.to_csv(cls.path_r1000c5_csv, include_index=False)

        f = cls.random_frame(10000, 50)
        cls.path_r10000c50_tsv = cls._td / 'r10000c50_no_i.tsv'
        cls.path_r10000c50_csv = cls._td / 'r10000c50_no_i.csv'
        f.to_tsv(cls.path_r10000c50_tsv, include_index=False)
        f.to_csv(cls.path_r10000c50_csv, include_index=False)

        # f = cls.random_frame(1000, 50000)
        # cls.r1000c50000_tsv = cls._td / 'r1000c50000_no_i.tsv'
        # cls.r1000c50000_csv = cls._td / 'r1000c50000_no_i.csv'
        # f.to_tsv(cls.r1000c50000_tsv, include_index=False)
        # f.to_csv(cls.r1000c50000_csv, include_index=False)


class ReadTsv_r1000c5(PerfTest):

    @classmethod
    def pd(cls):
        return pd.read_csv(SampleData.path_r1000c5_tsv, sep='/t')

    @classmethod
    def sf(cls):
        return sf.Frame.from_tsv(SampleData.path_r1000c5_tsv)


class ReadCsv_r1000c5(PerfTest):

    @classmethod
    def pd(cls):
        return pd.read_csv(SampleData.path_r1000c5_csv)

    @classmethod
    def sf(cls):
        return sf.Frame.from_csv(SampleData.path_r1000c5_csv)


class ReadTxt_Csv_r1000c5(PerfTest):

    @classmethod
    def pd(cls):
        return pd.read_csv(SampleData.path_r1000c5_csv)

    @classmethod
    def sf(cls):
        with open(SampleData.path_r1000c5_csv) as f:
            return sf.Frame.from_txt(f, delimiter=',')


class ReadTxt_Tsv_r1000c5(PerfTest):

    @classmethod
    def pd(cls):
        return pd.read_csv(SampleData.path_r1000c5_tsv, sep='\t')

    @classmethod
    def sf(cls):
        with open(SampleData.path_r1000c5_tsv) as f:
            return sf.Frame.from_txt(f, delimiter='\t')


class ReadTsv_r1000c5(PerfTest):

    @classmethod
    def pd(cls):
        return pd.read_csv(SampleData.path_r1000c5_tsv, sep='/t')

    @classmethod
    def sf(cls):
        return sf.Frame.from_tsv(SampleData.path_r1000c5_tsv)

########## r10000c50

class ReadCsv_r10000c50(PerfTest):

    @classmethod
    def pd(cls):
        return pd.read_csv(SampleData.path_r10000c50_csv)

    @classmethod
    def sf(cls):
        return sf.Frame.from_csv(SampleData.path_r10000c50_csv)


class ReadTxt_Csv_r10000c50(PerfTest):

    @classmethod
    def pd(cls):
        return pd.read_csv(SampleData.path_r10000c50_csv)

    @classmethod
    def sf(cls):
        with open(SampleData.path_r10000c50_csv) as f:
            return sf.Frame.from_txt(f, delimiter=',')


class ReadTxt_Tsv_r10000c50(PerfTest):

    @classmethod
    def pd(cls):
        return pd.read_csv(SampleData.path_r10000c50_tsv, sep='\t')

    @classmethod
    def sf(cls):
        with open(SampleData.path_r10000c50_tsv) as f:
            return sf.Frame.from_txt(f, delimiter='\t')
