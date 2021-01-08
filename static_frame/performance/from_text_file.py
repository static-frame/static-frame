import pathlib
import tempfile
import typing as tp
import getpass
import shutil
import string
import itertools
import random
import sys
import functools

import numpy as np
import static_frame as sf
import pandas as pd

from static_frame.performance.perf_test import PerfTest


class SampleData:
    '''An instance masquerading as a class!'''
    _store: tp.Dict[str, tp.Any] = {}
    _td = pathlib.Path(tempfile.gettempdir()) / f'{__name__}-of-{getpass.getuser()}'

    constructor_to_suffix = {
            sf.Frame.to_tsv: '.tsv',
            sf.Frame.to_csv: '.csv',
    }

    @classmethod
    def create(cls) -> None:
        '''Aka __init__'''
        shutil.rmtree(cls._td, ignore_errors=True)
        cls._td.mkdir()
        print(cls._td)

        cls.r1000c5 = cls.create_frames_if_not_exists(cls._td / 'r1000c5', 1000, 5)
        cls.r10000c50 = cls.create_frames_if_not_exists(cls._td / 'r10000c50', 10000, 50)
        cls.r10000c500 = cls.create_frames_if_not_exists(cls._td / 'r10000c500', 10000, 500)

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
    def create_frames_if_not_exists(cls, base_target: pathlib.Path, n_rows: int, n_cols: int):
        f = cls.random_frame(n_rows, n_cols)
        constructor_to_target = {k: base_target.with_suffix(v) for k, v in cls.constructor_to_suffix.items()}

        if not all(t.exists() for t in constructor_to_target.values()):
            for c, t in constructor_to_target.items():
                c(f, t, include_index=False)
        return {p.suffix: p for p in constructor_to_target.values()}


def run_method(func, data_name, suffix):
    path = getattr(SampleData, data_name)[suffix]
    with open(path) as f:
        return func(f)


# Build performance classes for all frames
sf_method_names = ('guess', 'no_guess')
suffix_to_sf_read_methods = {
    '.tsv': (sf.Frame.from_tsv, functools.partial(sf.Frame.from_delimited_no_guess, delimiter='\t')),
    '.csv': (sf.Frame.from_csv, functools.partial(sf.Frame.from_delimited_no_guess, delimiter=',')),
}
suffix_to_pd_read_method = {
    '.tsv': functools.partial(pd.read_csv, delimiter='\t'),
    '.csv': functools.partial(pd.read_csv, delimiter=','),
}
sizes = (
    'r1000c5',
    'r10000c50',
    # 'r10000c500',
)
for size in sizes:
    for suffix in SampleData.constructor_to_suffix.values():
        for sf_name, sf_method in zip(sf_method_names, suffix_to_sf_read_methods[suffix]):
            nodot = suffix[1:]

            name = f'{size}_{nodot}_{sf_name}'
            partial_kwargs = dict(

                data_name=size,
                suffix=suffix,
            )

            pandas = functools.partial(run_method, func=suffix_to_pd_read_method[suffix], **partial_kwargs)
            frame = functools.partial(run_method, func=sf_method, **partial_kwargs)

            methods = {
                'pd': staticmethod(pandas),
                'sf': staticmethod(frame),
                'NUMBER': 5,
            }

            class_ = type(name, (PerfTest, ), methods)
            globals()[name] = class_

            # Remove the class_ variable so it doesn't show up in module namespace later.
            del class_


