import typing as tp

from functools import partial
from hypothesis import strategies as st
from hypothesis.extra import numpy as hypo_np

import numpy as np

MAX_ROWS = 10
MAX_COLUMNS = 10

def get_dtypes(min_size: int = 0) -> tp.Iterable[np.dtype]:
    return st.lists(hypo_np.scalar_dtypes(), min_size=min_size)

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
        max_rows=MAX_ROWS,
        min_columns=1,
        max_columns=MAX_COLUMNS,
        ):
    return st.tuples(
            st.integers(min_value=min_rows, max_value=max_rows),
            st.integers(min_value=min_columns, max_value=max_columns)
            )

def get_array_2d(
        min_rows=1,
        max_rows=MAX_ROWS,
        min_columns=1,
        max_columns=MAX_COLUMNS,
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

def get_arrays_2d_aligned_columns(min_size: int = 1):

    columns = st.integers(min_value=1, max_value=MAX_COLUMNS).example()

    return st.lists(
            get_array_2d(min_columns=columns, max_columns=columns),
            min_size=min_size
            )

def get_arrays_2d_aligned_rows(min_size: int = 1):

    rows = st.integers(min_value=1, max_value=MAX_ROWS).example()

    return st.lists(
            get_array_2d(min_rows=rows, max_rows=rows),
            min_size=min_size
            )



# index_integer = st.builds(Index, st.lists(st.integers(), unique=True))
def get_labels(min_size: int = 0):
    '''
    Labels are suitable for creating non-date Indices.
    '''
    # 55203 is just before "high surrogates", and avoids this exception
    # UnicodeDecodeError: 'utf-32-le' codec can't decode bytes in position 0-3: code point in surrogate code point range(0xd800, 0xe000)

    codepoint_limits = dict(min_codepoint=1, max_codepoint=55203)


    list_dates = st.lists(st.dates(), min_size=min_size, unique=True)
    list_datetimes = st.lists(st.datetimes(), min_size=min_size, unique=True)

    list_integers = st.lists(st.integers(), min_size=min_size, unique=True)
    list_floats = st.lists(st.floats(), min_size=min_size, unique=True)
    list_complex = st.lists(st.complex_numbers(), min_size=min_size, unique=True)

    list_text = st.lists(
            st.text(st.characters(**codepoint_limits)),
            min_size=min_size,
            unique=True)

    list_mixed = st.lists(st.one_of(
            st.dates(),
            st.datetimes(),
            st.integers(),
            st.floats(),
            st.complex_numbers(),
            st.characters(**codepoint_limits)),
            min_size=min_size, unique=True)

    return st.one_of(
            list_mixed,
            list_dates,
            list_datetimes,
            list_integers,
            list_floats,
            list_complex,
            list_text
            )


if __name__ == '__main__':
    from static_frame.core.display_color import HexColor

    local_items = tuple(locals().items())
    for v in (v for k, v in local_items if callable(v) and k.startswith('get')):

        print(HexColor.format_terminal('hotpink', str(v)))


        for x in range(4    ):
            print(v().example())