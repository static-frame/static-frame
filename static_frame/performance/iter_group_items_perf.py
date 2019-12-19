import typing as tp
from time import time
from datetime import datetime

import numpy as np
import static_frame as sf
import pandas as pd

from static_frame.performance.perf_test import PerfTest

HMS = '%H:%M:%S'
GROUPBY_COL = 'groupby'


class _PerfTest(PerfTest):
    NUMBER = 3


#-------------------------------------------------------------------------------
# performance tests

class SampleData:

    _store: tp.Dict[str, tp.Any] = {}

    @classmethod
    def create(cls) -> None:
        print(f'({datetime.now().strftime(HMS) }) Building cache.')
        rows = 20_000_000
        cols = 9
        num_groups = 100_000
        columns = tuple('abcdefghi') + (GROUPBY_COL,)

        arr = np.random.random(rows * cols).reshape(rows, cols)
        groups = np.array([i % num_groups for i in np.random.permutation(rows)]).reshape(rows, 1)

        int_arr = np.hstack((arr, groups))
        df_int = pd.DataFrame(int_arr, columns=columns)
        frame_int = sf.Frame(int_arr, columns=columns)

        obj_arr = np.hstack((arr, groups)).astype(object)
        df_obj = pd.DataFrame(obj_arr, columns=columns).astype({GROUPBY_COL: int})
        frame_obj = sf.Frame(obj_arr, columns=columns).astype[GROUPBY_COL](int)
        print(f'({datetime.now().strftime(HMS) }) Finished building cache.')

        cls._store['pdf_20mil_int'] = df_int
        cls._store['sff_20mil_int'] = frame_int
        cls._store['pdf_20mil_obj'] = df_obj
        cls._store['sff_20mil_obj'] = frame_obj

        print(f'({datetime.now().strftime(HMS) }) Priming generators.')
        df_int_iterable_primed = iter(df_int.groupby(GROUPBY_COL, sort=False))
        next(df_int_iterable_primed)
        frame_int_iterable_primed = iter(frame_int.iter_group_items(GROUPBY_COL))
        next(frame_int_iterable_primed)
        df_obj_iterable_primed = iter(df_obj.groupby(GROUPBY_COL, sort=False))
        next(df_obj_iterable_primed)
        frame_obj_iterable_primed = iter(frame_obj.iter_group_items(GROUPBY_COL))
        next(frame_obj_iterable_primed)
        print(f'({datetime.now().strftime(HMS) }) Finisehd priming generators.')

        cls._store['pdf_20mil_int_iterable_primed'] = df_int_iterable_primed
        cls._store['sff_20mil_int_iterable_primed'] = frame_int_iterable_primed
        cls._store['pdf_20mil_obj_iterable_primed'] = df_obj_iterable_primed
        cls._store['sff_20mil_obj_iterable_primed'] = frame_obj_iterable_primed


    @classmethod
    def get(cls, key: str) -> tp.Any:
        return cls._store[key]

#-------------------------------------------------------------------------------


class FrameInt_setup(_PerfTest):
    @classmethod
    def pd(cls) -> None:
        pd_frame = SampleData.get('pdf_20mil_int')
        for _ in pd_frame.groupby(GROUPBY_COL, sort=False):
            break

    @classmethod
    def sf(cls) -> None:
        sf_frame = SampleData.get('sff_20mil_int')
        for _ in sf_frame.iter_group_items(GROUPBY_COL):
            break


class FrameObj_setup(_PerfTest):
    @classmethod
    def pd(cls) -> None:
        pd_frame = SampleData.get('pdf_20mil_obj')
        for _ in pd_frame.groupby(GROUPBY_COL, sort=False):
            break

    @classmethod
    def sf(cls) -> None:
        sf_frame = SampleData.get('sff_20mil_obj')
        for _ in sf_frame.iter_group_items(GROUPBY_COL):
            break


class FrameInt_iterate(_PerfTest):
    @classmethod
    def pd(cls) -> None:
        iterator = SampleData.get('pdf_20mil_int_iterable_primed')
        for _ in iterator:
            pass

    @classmethod
    def sf(cls) -> None:
        iterator = SampleData.get('sff_20mil_int_iterable_primed')
        for _ in iterator:
            pass

class FrameObj_iterate(_PerfTest):
    @classmethod
    def pd(cls) -> None:
        iterator = SampleData.get('pdf_20mil_obj_iterable_primed')
        for _ in iterator:
            pass

    @classmethod
    def sf(cls) -> None:
        iterator = SampleData.get('sff_20mil_obj_iterable_primed')
        for _ in iterator:
            pass
