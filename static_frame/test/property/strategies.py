import typing as tp

from functools import partial
from hypothesis import strategies as st
from hypothesis.extra import numpy as hypo_np

import numpy as np


def get_dtype_pairs() -> tp.Tuple[np.dtype]:
    return st.tuples(hypo_np.scalar_dtypes(), hypo_np.scalar_dtypes())

def get_array_1d(
        min_size: int = 0,
        max_size: int = 10,
        unique: bool = False):
    shape = st.integers(min_value=min_size, max_value=max_size)

    return hypo_np.arrays(
            hypo_np.scalar_dtypes(),
            shape,
            elements=None,
            fill=None,
            unique=False)


def get_shape_2d(
        min_rows=1,
        max_rows=10,
        min_columns=1,
        max_columns=10,
        ):
    return st.tuples(
            st.integers(min_value=min_rows, max_value=max_rows),
            st.integers(min_value=min_columns, max_value=max_columns)
            )

def get_array_2d(
        min_rows=1,
        max_rows=10,
        min_columns=1,
        max_columns=10,
        ):

    shape = get_shape_2d(
            min_rows=min_rows,
            max_rows=max_rows,
            min_columns=min_columns,
            max_columns=max_columns)

    return hypo_np.arrays(
            hypo_np.scalar_dtypes(),
            shape=shape,
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
    # 55203 is just before "high surrogates", and avoids this exception
    # UnicodeDecodeError: 'utf-32-le' codec can't decode bytes in position 0-3: code point in surrogate code point range(0xd800, 0xe000)

    codepoint_limits = dict(min_codepoint=1, max_codepoint=55203)

    list_integers = st.lists(st.integers(), min_size=min_size, unique=True)
    list_floats = st.lists(st.floats(), min_size=min_size, unique=True)

    list_text = st.lists(
            st.text(st.characters(**codepoint_limits)),
            min_size=min_size,
            unique=True)

    list_mixed = st.lists(st.one_of(
            st.integers(), st.floats(), st.characters(**codepoint_limits)),
            min_size=min_size, unique=True)

    return st.one_of(list_mixed, list_integers, list_floats, list_text)


if __name__ == '__main__':
    from static_frame.core.display_color import HexColor

    local_items = tuple(locals().items())
    for v in (v for k, v in local_items if callable(v) and k.startswith('get')):

        print(HexColor.format_terminal('hotpink', str(v)))


        for x in range(4    ):
            print(v().example())