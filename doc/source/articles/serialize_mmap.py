








import argparse
from pathlib import Path
import shutil
import time

import psutil


import numpy as np
import static_frame as sf

FP_C = Path('/tmp/npy-c')
FP_UC = Path('/tmp/npy-uc')
SHAPE = (10_000_000, 10)


def memory_usage(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        m_data = process.memory_info().data
        m_vms = process.memory_info().vms
        post = func(*args, **kwargs)
        time.sleep(1)
        print('memory: data', process.memory_info().data - m_data)
        print('memory: vms ', process.memory_info().vms - m_vms)
        return post
    return wrapper

@memory_usage
def _get_frame() -> sf.Frame:
    f = sf.FrameGO(index=np.arange(SHAPE[0]))
    # >>> columns = tuple('abcdef')
    for i in range(SHAPE[1]):
        f[i] = np.arange(SHAPE[0])
    return f


def build_fixture_consolidated(fp: Path = FP_C):
    shutil.rmtree(fp)
    _get_frame().to_npy(fp, consolidate_blocks=True)
    print('\n'.join(str(p) for p in fp.iterdir()))

def build_fixture_unconsolidated(fp: Path = FP_UC):
    shutil.rmtree(fp)
    _get_frame().to_npy(fp, consolidate_blocks=False)
    print('\n'.join(str(p) for p in fp.iterdir()))

@memory_usage
def read_consolidated_mmap() -> None:
    f, _ = sf.Frame.from_npy_mmap(FP_C)

@memory_usage
def read_consolidated() -> None:
    f = sf.Frame.from_npy(FP_C)


@memory_usage
def read_unconsolidated_mmap() -> None:
    f, _ = sf.Frame.from_npy_mmap(FP_UC)

@memory_usage
def read_unconsolidated() -> None:
    f = sf.Frame.from_npy(FP_UC)


@memory_usage
def read_unconsolidated_mmap_use() -> None:
    f, _ = sf.Frame.from_npy_mmap(FP_UC)
    @memory_usage
    def use():
        for i in f.columns:
            s = f[i].values[:30]
    use()


@memory_usage
def read_consolidated_mmap_use() -> None:
    f, _ = sf.Frame.from_npy_mmap(FP_C)
    @memory_usage
    def use():
        for x in f.iter_element():
            pass
    use()


def get_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
            # description='Performance testing and profiling',
            # formatter_class=argparse.RawDescriptionHelpFormatter,
            )
    p.add_argument('--build_fixture_consolidated',
            action='store_true',
            )
    p.add_argument('--build_fixture_unconsolidated',
            action='store_true',
            )
    p.add_argument('--read_consolidated',
            action='store_true',
            )
    p.add_argument('--read_consolidated_mmap',
            action='store_true',
            )
    p.add_argument('--read_unconsolidated',
            action='store_true',
            )
    p.add_argument('--read_unconsolidated_mmap',
            action='store_true',
            )
    p.add_argument('--read_unconsolidated_mmap_use',
            action='store_true',
            )
    p.add_argument('--read_consolidated_mmap_use',
            action='store_true',
            )
    return p

if __name__ == '__main__':
    options = get_arg_parser().parse_args()
    for opt, value in vars(options).items():
        if value:
            locals()[opt]()
