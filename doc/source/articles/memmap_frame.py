
import zipfile
import os
import timeit
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context as get_mp_context
import shutil
import typing as tp
from functools import partial

import numpy as np
import frame_fixtures as ff
import static_frame as sf
from static_frame.core.display_color import HexColor


# COUNT_ARRAY = 100
CHUNK_SIZE = 20


FF_100kx1k = 's(100_000,1_000)|v(int,int,bool,float,float)|i(I,int)|c(I,str)'
FF_10kx1k = 's(10_000,1_000)|v(int,int,bool,float,float)|i(I,int)|c(I,str)'
FF_100kx1k_auto = 's(100_000,1_000)|v(int,int,bool,float,float)'
FF_10kx1k_auto = 's(10_000,1_000)|v(int,int,bool,float,float)'

# FF_wide_col = 's(100,1_000)|v(int,bool,float)|i(I,int)|c(I,str)'
# FF_wide_ext = 's(1000,10_000)|v(int,int,bool,float,float)|i(I,int)|c(I,str)'

# FF_tall = 's(10_000,10)|v(int,int,bool,float,float)|i(I,int)|c(I,str)'
# FF_tall_col = 's(10_000,10)|v(int,bool,float)|i(I,int)|c(I,str)'
# FF_tall_ext = 's(10_000,1000)|v(int,int,bool,float,float)|i(I,int)|c(I,str)'

# FF_square = 's(1_000,1_000)|v(float)|i(I,int)|c(I,str)'

def work(frame: sf.Frame) -> int:
    count = 0
    for row in frame.iter_series(axis=1):
        count += 1
    return count


class MMapTest:
    NUMBER = 1

    def __init__(self, fixture):
        self.fp_dir = '/tmp/npy'
        self.fixture = ff.parse(fixture)
        self.fixture.to_npy(self.fp_dir)

        partitions = 100
        ref = np.arange(len(self.fixture.columns))
        self.selections = [(ref % partitions) == i for i in range(partitions)]
        # import ipdb; ipdb.set_trace()
        print(self, 'init complete')

    def clear(self) -> None:
        if os.path.exists(self.fp_dir):
            shutil.rmtree(self.fp_dir)
        print(self, 'del complete')



# class CopyThread(MMapTest):

#     @staticmethod
#     def func(args):
#         frame, sel = args
#         return work(frame[sel])

#     def __call__(self):
#         with ThreadPoolExecutor() as executor:
#             args = ((self.fixture, sel) for sel in self.selections)
#             post = tuple(executor.map(self.func, args, chunksize=CHUNK_SIZE))
#             assert self.fixture.shape[0] == post[0]
#             assert len(post) == len(self.selections)

class CopyFork(MMapTest):

    @staticmethod
    def func(args):
        frame, sel = args
        return work(frame[sel])

    def __call__(self):
        with ProcessPoolExecutor(mp_context=get_mp_context('fork')) as executor:
            args = ((self.fixture, sel) for sel in self.selections)
            post = tuple(executor.map(self.func, args, chunksize=CHUNK_SIZE))
            assert self.fixture.shape[0] == post[0]
            assert len(post) == len(self.selections)

class CopySpawn(MMapTest):

    @staticmethod
    def func(args):
        frame, sel = args
        return work(frame[sel])

    def __call__(self):
        with ProcessPoolExecutor(mp_context=get_mp_context('spawn')) as executor:
            args = ((self.fixture, sel) for sel in self.selections)
            post = tuple(executor.map(self.func, args, chunksize=CHUNK_SIZE))
            assert self.fixture.shape[0] == post[0]
            assert len(post) == len(self.selections)



# class MMapThread(MMapTest):

#     @staticmethod
#     def func(fp: str, cols=np.ndarray):
#         return work(sf.Frame.from_npy(fp)[cols])

#     def __call__(self):
#         fp = self.fp_dir
#         func = partial(self.func, fp)

#         with ThreadPoolExecutor() as executor:
#             post = tuple(executor.map(func, self.selections, chunksize=CHUNK_SIZE))
#             assert self.fixture.shape[0] == post[0]
#             assert len(post) == len(self.selections)

class MMapFork(MMapTest):

    @staticmethod
    def func(fp: str, cols=np.ndarray):
        return work(sf.Frame.from_npy(fp)[cols])

    def __call__(self):
        fp = self.fp_dir
        func = partial(self.func, fp)

        with ProcessPoolExecutor(mp_context=get_mp_context('fork')) as executor:
            post = tuple(executor.map(func, self.selections, chunksize=CHUNK_SIZE))
            assert self.fixture.shape[0] == post[0]
            assert len(post) == len(self.selections)

class MMapSpawn(MMapTest):

    @staticmethod
    def func(fp: str, cols=np.ndarray):
        return work(sf.Frame.from_npy(fp)[cols])

    def __call__(self):
        fp = self.fp_dir
        func = partial(self.func, fp)

        with ProcessPoolExecutor(mp_context=get_mp_context('spawn')) as executor:
            post = tuple(executor.map(func, self.selections, chunksize=CHUNK_SIZE))
            assert self.fixture.shape[0] == post[0]
            assert len(post) == len(self.selections)





#-------------------------------------------------------------------------------
def get_format():

    name_root_last = None
    name_root_count = 0

    def format(key: tp.Tuple[tp.Any, str], v: object) -> str:
        nonlocal name_root_last
        nonlocal name_root_count

        if isinstance(v, float):
            if np.isnan(v):
                return ''
            return str(round(v, 4))
        if isinstance(v, (bool, np.bool_)):
            if v:
                return HexColor.format_terminal('green', str(v))
            return HexColor.format_terminal('orange', str(v))

        return str(v)

    return format


def run_test():
    records = []
    for label, fixture in (
            ('100kx1k', FF_100kx1k),
            ('10kx1k', FF_10kx1k),
            ('100kx1k_auto', FF_100kx1k_auto),
            ('10kx1k_auto', FF_10kx1k_auto),            ):
        for cls in (
                # CopyThread,
                CopyFork,
                # CopySpawn,
                # MMapThread,
                MMapFork,
                # MMapSpawn,
                ):
            runner = cls(fixture)
            record = [cls.__name__, cls.NUMBER, label]
            result = timeit.timeit(
                    f'runner()',
                    globals=locals(),
                    number=cls.NUMBER)
            record.append(result)
            records.append(record)
            runner.clear()

    f = sf.FrameGO.from_records(records,
            columns=('name', 'number', 'fixture', 'time')
            )

    display = f.iter_element_items().apply(get_format())

    config = sf.DisplayConfig(
            cell_max_width_leftmost=np.inf,
            cell_max_width=np.inf,
            type_show=False,
            display_rows=200,
            include_index=False,
            )
    print(display.display(config))
    # import ipdb; ipdb.set_trace()



if __name__ == '__main__':
    run_test()

