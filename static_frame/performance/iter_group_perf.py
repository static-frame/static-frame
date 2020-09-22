import typing as tp
import timeit
import random
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd

import static_frame as sf
from static_frame.core.frame import Frame
from static_frame.performance.perf_test import PerfTest


HMS = '%H:%M:%S'
GROUPBY_COL = 'groupby'


frame_func_t = tp.Callable[[Frame], tp.Any]



class _PerfTest(PerfTest):
    NUMBER = 3



class BuildTestFrames:
    __slots__ = (
            'dims',
            'nan_chance',
            'none_chance'
    )

    _DTYPES = ['int', 'float', 'bool', 'str', 'object', 'mixed']
    _NUMBER = 3
    _REPEAT = 10

    def __init__(self,
            dims: tp.Tuple[int, ...] = (5, 20, 100, 1000),
            nan_chance: float = 0.33,
            none_chance: float = 0.33):
        self.dims = dims
        self.nan_chance = nan_chance
        self.none_chance = none_chance

    class Test_Object:
        def __init__(self, x: int):
            self.x = x

        def __str__(self) -> str:
            return f'Test_Object({self.x})'

        def __repr__(self) -> str:
            return str(self)

    def _make_float(self, val: int) -> float:
        if np.random.random() <= self.nan_chance:
            return np.nan # type: ignore
        else:
            return val * 0.1

    def _make_bool(self, val: int) -> bool:
        return val % 2 == 0

    def _make_str(self, val: int) -> str:
        return ''.join(chr((val + i) % 26 + 65) for i in range(3))

    def _make_object(self, val: int) -> tp.Optional['Test_Object']:
        if np.random.random() <= self.none_chance:
            return None
        else:
            return self.Test_Object(val)

    def _make_mixed(self, val: int): # type: ignore
        r: int = np.random.randint(len(BuildTestFrames._DTYPES))
        if r == 0:
            return val
        if r == 1:
            return self._make_float(val)
        if r == 2:
            return self._make_bool(val)
        if r == 3:
            return self._make_str(val)
        if r == 4:
            return self._make_object(val)

    def _build_col(self, rows: int, dtype: str) -> np.ndarray:
        if dtype == 'int':
            return np.arange(rows)

        if dtype == 'float':
            return np.vectorize(self._make_float)(np.arange(rows))

        if dtype == 'bool':
            return np.vectorize(self._make_bool)(np.arange(rows))

        if dtype == 'str':
            return np.vectorize(self._make_str)(np.arange(rows))

        if dtype == 'object':
            return np.vectorize(self._make_object)(np.arange(rows))

        if dtype == 'mixed':
            # Cannot vectorize this call :(
            return np.array([self._make_mixed(val) for val in range(rows)])

    @staticmethod
    def _build_groups(num_of_groups: int, num_of_rows: int) -> np.ndarray:
        assert num_of_groups > 0
        i: int = 0
        build: tp.List[int] = []
        while i < num_of_rows:
            build.append(i % num_of_groups)
            i += 1
        return np.array(build)

    def _next_frame_dims(self,
            mixed_data_options: tp.Iterable[bool]
            ) -> tp.Iterator[tp.Tuple[int, int, int, bool]]:
        for rows in self.dims:
            for cols in self.dims:
                for groups in self.dims:
                    if groups <= rows:
                        for mixed_data in mixed_data_options:
                            yield rows, cols - 1, groups, mixed_data

    @staticmethod
    def _shuffle(frame: Frame) -> Frame:
        random.seed(0)
        return frame.loc[random.sample(frame.index.values.tolist(), len(frame))] # type: ignore

    def build_frame(self, rows: int, cols: int, groups: int, mixed_data: bool) -> Frame:
        group_col: np.ndarray = self._build_groups(groups, rows)

        if mixed_data:
            built_cols: tp.List[tp.Tuple[str, np.ndarray]] = []
            for col in range(cols):
                dtype = BuildTestFrames._DTYPES[col % len(BuildTestFrames._DTYPES)]
                built_cols.append((str(col), self._build_col(rows, dtype)))

            built_cols.append((GROUPBY_COL, group_col))
            f = Frame.from_items(built_cols)
        else:
            arr = np.arange(rows*cols).reshape(rows, cols)
            arr = np.hstack((arr, group_col.reshape(rows, 1)))

            columns = [str(i) for i in range(cols)] + [GROUPBY_COL]
            f = Frame(arr, columns=columns)

        return BuildTestFrames._shuffle(f)

    def next_frame(self,
            mixed_data_options: tp.Iterable[bool] = (True, False)
            ) -> tp.Iterator[Frame]:
        for rows, cols, groups, mixed_data in self._next_frame_dims(mixed_data_options):
            yield self.build_frame(rows, cols, groups, mixed_data)

    @staticmethod
    def get_perf(
            frame: Frame,
            func: frame_func_t,
            repeat: int = _REPEAT,
            number: int = _NUMBER
            ) -> float:
        timer = timeit.Timer(partial(func, frame))
        return round(np.mean(timer.repeat(repeat=repeat, number=number)), 4) # type: ignore

    @staticmethod
    def test_frames(
            frames: tp.List[Frame],
            funcs: tp.List[frame_func_t],
            repeat: int = _REPEAT,
            number: int = _NUMBER
            ) -> tp.List[tp.Dict[str, float]]:
        results: tp.List[tp.Dict[str, float]] = []

        for frame in frames:
            # Frame metadata shared across tests
            rows, cols = frame.shape
            groups: int = len(frame[GROUPBY_COL].unique())

            result: tp.Dict[str, float] = {}
            for func in funcs:
                key: str = func.__name__ + f': {rows}, {groups}, {cols}'
                result[key] = BuildTestFrames.get_perf(frame, func, repeat, number)
            results.append(result)

        return results


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


class FrameInt_iter_group_items_setup(_PerfTest):
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


class FrameObj_iter_group_items_setup(_PerfTest):
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


class FrameInt_iter_group_items_iterate(_PerfTest):
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

class FrameObj_iter_group_items_iterate(_PerfTest):
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
