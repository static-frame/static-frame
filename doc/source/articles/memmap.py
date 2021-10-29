
import zipfile
import os
import timeit
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context as get_mp_context
import shutil
import typing as tp
import gc

import numpy as np
import frame_fixtures as ff
import static_frame as sf
from static_frame.core.display_color import HexColor


COUNT_ARRAY = 100
CHUNK_SIZE = 20

def work(array: np.ndarray):
    # post = []
    # for x in array:
    #     if x > 0:
    #         post.append(x)
    v = array ** 2
    return (v / v.sum()) ** 0.5

class MMapTest:
    NUMBER = 1

    def __init__(self, fixture):
        self.fp_npz = '/tmp/memmap.npz'
        self.fp_dir = '/tmp/memmap'

        self.arrays = {}
        for i in range(COUNT_ARRAY):
            self.arrays[str(i)] = np.arange(1_000_000)

        np.savez(self.fp_npz, **self.arrays)

        # cannot read from zip in mmmap mode, so must extract all
        # https://stackoverflow.com/questions/29080556/how-does-numpy-handle-mmaps-over-npz-files

        with zipfile.ZipFile(self.fp_npz) as zf:
            zf.extractall(self.fp_dir)

        print(self, 'init complete')

    def clear(self) -> None:
        if os.path.exists(self.fp_npz):
            os.unlink(self.fp_npz)
        if os.path.exists(self.fp_dir):
            shutil.rmtree(self.fp_dir)

        print(self, 'del complete')

class MemorySum(MMapTest):

    def __call__(self):
        for a in self.arrays.values():
            work(a)


class MemoryThreadSum(MMapTest):
    def __call__(self):
        with ThreadPoolExecutor() as executor:
            post = tuple(executor.map(work, self.arrays.values(), chunksize=CHUNK_SIZE))
            assert len(post) == COUNT_ARRAY

class MemoryForkSum(MMapTest):

    def __call__(self):
        with ProcessPoolExecutor(mp_context=get_mp_context('fork'), ) as executor:
            post = tuple(executor.map(work, self.arrays.values(), chunksize=CHUNK_SIZE))
            assert len(post) == COUNT_ARRAY

class MemorySpawnSum(MMapTest):

    def __call__(self):
        with ProcessPoolExecutor(mp_context=get_mp_context('spawn')) as executor:
            post = tuple(executor.map(work, self.arrays.values(), chunksize=CHUNK_SIZE))
            assert len(post) == COUNT_ARRAY



class MMapThreadSum(MMapTest):

    @staticmethod
    def func(fp: str):
        a = np.load(fp, mmap_mode='r')
        return work(a)

    def __call__(self):
        fps = (os.path.join(self.fp_dir, f'{fn}.npy') for fn in self.arrays.keys())
        with ThreadPoolExecutor() as executor:
            post = tuple(executor.map(self.func, fps, chunksize=CHUNK_SIZE))
            assert len(post) == COUNT_ARRAY

class MMapForkSum(MMapTest):

    @staticmethod
    def func(fp: str):
        a = np.load(fp, mmap_mode='r')
        return work(a)

    def __call__(self):
        fps = (os.path.join(self.fp_dir, f'{fn}.npy') for fn in self.arrays.keys())
        with ProcessPoolExecutor(mp_context=get_mp_context('fork')) as executor:
            post = tuple(executor.map(self.func, fps, chunksize=CHUNK_SIZE))
            assert len(post) == COUNT_ARRAY

class MMapSpawnSum(MMapTest):

    @staticmethod
    def func(fp: str):
        a = np.load(fp, mmap_mode='r')
        return work(a)

    def __call__(self):
        fps = (os.path.join(self.fp_dir, f'{fn}.npy') for fn in self.arrays.keys())
        with ProcessPoolExecutor(mp_context=get_mp_context('spawn')) as executor:
            post = tuple(executor.map(self.func, fps, chunksize=CHUNK_SIZE))
            assert len(post) == COUNT_ARRAY





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
            ('None', None),
            ):
        for cls in (
                MemorySum,
                MemoryThreadSum,
                MemoryForkSum,
                MemorySpawnSum,
                MMapThreadSum,
                MMapForkSum,
                MMapSpawnSum,
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

