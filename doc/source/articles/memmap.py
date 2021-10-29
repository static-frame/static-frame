
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


class MMapTest:
    NUMBER = 1

    def __init__(self, fixture):
        self.fp_npz = '/tmp/memmap.npz'
        self.fp_dir = '/tmp/memmap'

        self.arrays = {}
        for i in range(100):
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

class MMapMemorySum(MMapTest):

    def __call__(self):
        for a in self.arrays.values():
            a.sum()

class MMapMemoryMPForkSum(MMapTest):

    @staticmethod
    def func(a: np.ndarray):
        return a.sum()

    def __call__(self):
        with ProcessPoolExecutor(mp_context=get_mp_context('fork')) as executor:
            post = tuple(executor.map(self.func, self.arrays.values()))


class MMapMemoryMPSpawnSum(MMapTest):

    @staticmethod
    def func(a: np.ndarray):
        return a.sum()

    def __call__(self):
        with ProcessPoolExecutor(mp_context=get_mp_context('spawn')) as executor:
            post = tuple(executor.map(self.func, self.arrays.values()))


class MMapMMapMPForkSum(MMapTest):

    @staticmethod
    def func(fp: str):
        a = np.load(fp, mmap_mode='r')
        return a.sum()

    def __call__(self):
        fps = (os.path.join(self.fp_dir, f'{fn}.npy') for fn in self.arrays.keys())
        with ProcessPoolExecutor(mp_context=get_mp_context('fork')) as executor:
            post = tuple(executor.map(self.func, fps))






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
                MMapMemorySum,
                MMapMemoryMPForkSum,
                MMapMemoryMPSpawnSum,
                MMapMMapMPForkSum,
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

