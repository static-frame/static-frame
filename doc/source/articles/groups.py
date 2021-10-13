import timeit
import typing as tp
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor


import numpy as np

import static_frame as sf




class PerfTest:
    NUMBER = 50

    def __init__(self, size: int, groups: int):
        self.fixture = np.arange(0, size) % groups
        self.groups = range(groups)

    def run(self):
        raise NotImplementedError()


class Serial(PerfTest):
    def run(self):
        selections = []
        for i in self.groups:
            v = self.fixture == i
            selections.append(v)

class SerialNoCopy(PerfTest):
    def run(self):
        v = np.empty(len(self.fixture), dtype=bool)
        for i in self.groups:
            np.equal(self.fixture, i, out=v)

def func(args):
    return np.equal(*args)

class ThreadPool(PerfTest):
    def run(self):
        futures = []
        with ThreadPoolExecutor(max_workers=100) as executor:
            for i in self.groups:
                futures.append(executor.submit(func, (self.fixture, i)))
            results = [f.result() for f in futures]

# class ProcessPool(PerfTest):
#     def run(self):
#         futures = []
#         with ProcessPoolExecutor() as executor:
#             for i in self.groups:
#                 futures.append(executor.submit(func, (self.fixture, i)))
#             results = [f.result() for f in futures]


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

    for size, groups in (
        (1_000, 2),
        (1_000, 100),
        (100_000, 2),
        (100_000, 100),
        (100_000, 1_000),
        ):
        for cls in (
                Serial,
                SerialNoCopy,
                ThreadPool,
                # ProcessPool,
                ):
            runner = cls(size, groups)
            record = [cls.__name__, cls.NUMBER, size, groups]
            result = timeit.timeit(
                    f'runner.run()',
                    globals=locals(),
                    number=cls.NUMBER)
            record.append(result)
            records.append(record)

    f = sf.FrameGO.from_records(records,
            columns=('name', 'number', 'size', 'groups', 'time')
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


