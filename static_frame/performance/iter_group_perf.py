from collections import namedtuple
from functools import partial
import pickle
import timeit

import numpy as np

from static_frame.core.frame import Frame
from static_frame.core.frame import Series


np.random.seed(0)


_DIMS = [5, 20, 100, 1000]
_DTYPES = ['int', 'float', 'bool', 'str', 'object', 'mixed']
_NAN_CHANCE = 0.33
_NONE_CHANCE = 0.33


class Object:
    def __init__(self, x):
        self.x = x
    def __str__(self):
        return f'Object({self.x})'
    def __repr__(self):
        return str(self)


def make_float(val):
    if np.random.random() <= _NAN_CHANCE:
        return np.nan
    else:
        return val * 0.1


def make_bool(val):
    return val % 2 == 0


def make_str(val):
    return str(val)


def make_object(val):
    if np.random.random() <= _NONE_CHANCE:
        return None
    else:
        return Object(val)


def make_mixed(val):
    r = np.random.randint(len(_DTYPES))
    if r == 0:
        return val
    if r == 1:
        return make_float(val)
    if r == 2:
        return make_bool(val)
    if r == 3:
        return make_str(val)
    if r == 4:
        return make_object(val)


def build_col(rows, dtype):
    if dtype == 'int':
        return np.arange(rows)

    if dtype == 'float':
        return np.vectorize(make_float)(np.arange(rows))

    if dtype == 'bool':
        return np.vectorize(make_bool)(np.arange(rows))

    if dtype == 'str':
        return np.vectorize(make_str)(np.arange(rows))

    if dtype == 'object':
        return np.vectorize(make_object)(np.arange(rows))

    if dtype == 'mixed':
        # Cannot vectorize this call :(
        return np.array([make_mixed(val) for val in range(rows)])


def build_groups(num_of_groups, num_of_rows):
    assert num_of_groups > 0
    i = 0
    build = []
    while i < num_of_rows:
        build.append(i % num_of_groups)
        i += 1
    return np.array(build)


def next_frame_dims(mixed_data_options):
    for rows in _DIMS:
        for cols in _DIMS:
            for groups in _DIMS:
                if groups <= rows: # This was cols on my first set of benchmarks :(
                    for mixed_data in mixed_data_options:
                        yield rows, cols - 1, groups, mixed_data


def shuffle(frame):
    import random
    random.seed(0)
    return frame.loc[random.sample(frame.index.values.tolist(), len(frame))]


def build_frame(rows, cols, groups, mixed_data):
    group_col = build_groups(groups, rows)

    if mixed_data:
        built_cols = []
        for col in range(cols):
            dtype = _DTYPES[col % len(_DTYPES)]
            built_cols.append((str(col), build_col(rows, dtype)))

        built_cols.append(('groupby', group_col))
        f = Frame.from_items(built_cols)
    else:
        arr = np.arange(rows*cols).reshape(rows, cols)
        arr = np.hstack((arr, group_col.reshape(rows, 1)))

        columns = [str(i) for i in range(cols)] + ['groupby']
        f = Frame(arr, columns=columns)

    return shuffle(f)


def next_frame(mixed_data_options = (True, False)):
    for rows, cols, groups, mixed_data in next_frame_dims(mixed_data_options):
        yield build_frame(rows, cols, groups, mixed_data)


REPEAT = 7
NUMBER = 100


def groupby(frame):
    for group, sub in frame.groupby('groupby'):
        pass

def iter_group_items(frame):
    for group, sub in frame.iter_group_items('groupby'):
        pass



TestResult = namedtuple('TestResult', ['groupby', 'iter_group_items'])



def get_perf(frame, func, repeat=REPEAT, number=NUMBER):
    timer = timeit.Timer(partial(func, frame))
    return round(np.mean(timer.repeat(repeat=repeat, number=number)), 4)


def test_frames(frames, groupby_only=False):
    rows = []
    cols = []
    groups = []

    groupby_results = []
    iter_group_items_results = []

    for frame in frames:
        # Frame metadata shared across tests
        rows.append(frame.shape[0])
        cols.append(frame.shape[1])
        groups.append(len(frame['groupby'].unique()))

        # Test groupby impl
        groupby_results.append(get_perf(frame, groupby))

        # Test iter_group_by impl
        if not groupby_only:
            iter_group_items_results.append(get_perf(frame, iter_group_items))

    if not groupby_only:
        iter_group_items_test_result = np.array([rows, cols, groups, iter_group_items_results]).T
    else:
        iter_group_items_test_result = None

    return TestResult(
            np.array([rows, cols, groups, groupby_results]).T,
            iter_group_items_test_result,
    )


def pickle_frame(frame):
    print(f'Caching to {frame.name}')
    with open(f'/home/burkland/.cached_objects/{frame.name}_obj.pkl', 'wb') as f:
        pickle.dump(frame, f)


if __name__ == '__main__':
    groupby_only = False

    for is_mixed in (True, False):
        prefix = '' if is_mixed else 'un'
        frames_unmixed = [frame for frame in next_frame(mixed_data_options=(is_mixed,))]

        results = test_frames(frames_unmixed, groupby_only)

        frame = Frame(
                results.groupby,
                columns=['row', 'col', 'groups', 'time (s)'],
                name=f'{prefix}mixed_groupby_results',
        ).astype['row'](int).astype['col'](int).astype['groups'](int)

        frame = frame.sort_values('time (s)', ascending=False)
        pickle_frame(frame)
        #print(frame)


        if not groupby_only:
            frame = Frame(
                    results.iter_group_items,
                    columns=['row', 'col', 'groups', 'time (s)'],
                    name=f'{prefix}mixed_iter_group_items_results',
            ).astype['row'](int).astype['col'](int).astype['groups'](int)

            frame = frame.sort_values('time (s)', ascending=False)
            pickle_frame(frame)
            #print(frame)

        #print()
