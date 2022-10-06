import argparse
import enum
import shutil
import time
import typing as tp
from pathlib import Path

import numpy as np
import psutil

import static_frame as sf

FP_C = Path('/tmp/npy-c')
FP_UC = Path('/tmp/npy-uc')
SHAPE = (10_000_000, 10)

class Format(enum.Enum):
    CONSOLIDATED = 0
    UNCONSOLIDATED = 1

    def get_path(self) -> Path:
        return FP_UC if self.value else FP_C


def memory_info_attrs(proc: psutil.Process):
    minfo = proc.memory_info()
    return {k: getattr(minfo, k) for k in dir(minfo) if (not k.startswith('_') and not callable(getattr(minfo, k)))}

def memory_info_diff(previous: tp.Dict[str, int], current: tp.Dict[str, int]):
    for k in previous.keys():
        print(f'psutil: {k}: {current[k] - previous[k]: 2e}')

def memory_usage(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        m_init = memory_info_attrs(process)
        post = func(*args, **kwargs)
        time.sleep(1)
        # print('memory: data', process.memory_info().data - m_data)
        memory_info_diff(m_init, memory_info_attrs(process))
        return post
    return wrapper

@memory_usage
def _get_frame() -> sf.Frame:
    f = sf.FrameGO(index=np.arange(SHAPE[0]))
    # >>> columns = tuple('abcdef')
    for i in range(SHAPE[1]):
        f[i] = np.arange(SHAPE[0])
    return f

#-------------------------------------------------------------------------------
def write_fixture(format: Format):
    fp = format.get_path()
    if fp.exists():
        shutil.rmtree(fp)
    _get_frame().to_npy(fp, consolidate_blocks=(format is Format.CONSOLIDATED))
    print('\n'.join(str(p) for p in fp.iterdir()))

@memory_usage
def read_mmap(format: Format) -> None:
    fp = format.get_path()
    f, _ = sf.Frame.from_npy_mmap(fp)

@memory_usage
def read_mmap_use(format: Format) -> None:
    fp = format.get_path()
    f, _ = sf.Frame.from_npy_mmap(fp)

    @memory_usage
    def use():
        post = []
        for i in f.columns:
            post.append(f[i].values[:30])
        return post
    use()

@memory_usage
def read(format: Format) -> None:
    fp = format.get_path()
    f = sf.Frame.from_npy(fp)

@memory_usage
def read_use(format: Format) -> None:
    fp = format.get_path()
    f = sf.Frame.from_npy(fp)

    @memory_usage
    def use():
        post = []
        for i in f.columns:
            post.append(f[i].values[:30])
        return post
    use()


#-------------------------------------------------------------------------------

module_locals = locals()

def _get_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(f'--unc', action='store_true')
    for name, value in module_locals.items():
        if callable(value) and (name.startswith('read') or name.startswith('write')):
            p.add_argument(f'--{name}', action='store_true')
    return p

if __name__ == '__main__':
    options = _get_arg_parser().parse_args()
    format = Format.UNCONSOLIDATED if options.unc else Format.CONSOLIDATED
    for opt, value in vars(options).items():
        if opt == 'unc':
            continue
        if value:
            module_locals[opt](format)




