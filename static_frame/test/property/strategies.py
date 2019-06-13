import typing as tp

from functools import partial
from hypothesis import strategies as st
from hypothesis.extra import numpy as hypo_np

import numpy as np


def get_dtype_pairs() -> tp.Tuple[np.dtype]:
    return st.tuples(hypo_np.scalar_dtypes(), hypo_np.scalar_dtypes())

def get_array_1d(min_size: int = 0, unique: bool = False):
    shape = st.integers(min_value=min_size)
    return hypo_np.arrays(
            hypo_np.scalar_dtypes(),
            shape,
            elements=None,
            fill=None,
            unique=False)

def get_array_2d():
    shape = hypo_np.array_shapes(min_dims=2, max_dims=2)
    return hypo_np.arrays(
            hypo_np.scalar_dtypes(),
            shape, # shape
            elements=None,
            fill=None,
            unique=False)

def get_array_1d2d():
    return st.one_of(get_array_1d(), get_array_2d())


# index_integer = st.builds(Index, st.lists(st.integers(), unique=True))
def get_labels(min_size: int = 0):
    '''
    Labels are suitable for creating non-date Indices.
    '''
    list_integers = st.lists(st.integers(), min_size=min_size, unique=True)
    list_floats = st.lists(st.floats(), min_size=min_size, unique=True)

    list_text = st.lists(
            st.text(st.characters(min_codepoint=1)),
            min_size=min_size,
            unique=True)

    list_mixed = st.lists(st.one_of(
            st.integers(), st.floats(), st.characters(min_codepoint=1)),
            min_size=min_size, unique=True)

    return st.one_of(list_mixed, list_integers, list_floats, list_text)


if __name__ == '__main__':
    local_items = tuple(locals().items())
    for v in (v for k, v in local_items if callable(v) and k.startswith('get')):
        print(v)
        for x in range(10):
            print('\t', v().example())