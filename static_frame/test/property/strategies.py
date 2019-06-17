import typing as tp

from functools import partial
from itertools import chain

from hypothesis import strategies as st
from hypothesis.extra import numpy as hypo_np

import numpy as np

MAX_ROWS = 10
MAX_COLUMNS = 20

def get_dtype():
    return hypo_np.scalar_dtypes()

def get_dtypes(min_size: int = 0) -> tp.Iterable[np.dtype]:
    return st.lists(get_dtype(), min_size=min_size)

def get_dtype_pairs() -> tp.Tuple[np.dtype]:
    return st.tuples(get_dtype(), get_dtype())

#-------------------------------------------------------------------------------
# array generation

def get_array_1d(
        min_size: int = 0,
        max_size: int = MAX_ROWS,
        unique: bool = False):
    shape = st.integers(min_value=min_size, max_value=max_size)

    return hypo_np.arrays(
            get_dtype(),
            shape,
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

def get_shape_1d2d() -> tp.Union[tp.Tuple[int], tp.Tuple[int, int]]:
    return st.one_of(get_shape_2d(), st.tuples(st.integers(min_value=1, max_value=MAX_ROWS)))

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
            get_dtype(),
            shape=shape,
            unique=False)

def get_array_1d2d(
        min_rows=1,
        max_rows=MAX_ROWS,
        min_columns=1,
        max_columns=MAX_COLUMNS,
        ):
    '''
    For convenience in building blocks, treat row constraints as 1d size constraints.
    '''
    return st.one_of(
            get_array_2d(min_rows=min_rows,
                    max_rows=max_rows,
                    min_columns=min_columns,
                    max_columns=max_columns),
            get_array_1d(min_size=min_rows, max_size=max_rows)
    )

#-------------------------------------------------------------------------------
# aligend arrays for concatenation and type blocks

def get_arrays_2d_aligned_columns(min_size: int = 1, max_size: int = 10):

    return st.integers(min_value=1, max_value=MAX_COLUMNS).flatmap(
        lambda columns: st.lists(
            get_array_2d(min_columns=columns, max_columns=columns),
            min_size=min_size,
            max_size=max_size
            )
    )


def get_arrays_2d_aligned_rows(min_size: int = 1, max_size: int = 10):

    return st.integers(min_value=1, max_value=MAX_ROWS).flatmap(
        lambda rows: st.lists(
            get_array_2d(min_rows=rows, max_rows=rows),
            min_size=min_size,
            max_size=max_size
            )
    )

def get_blocks(
        min_rows=1,
        max_rows=MAX_ROWS,
        min_columns=1,
        max_columns=MAX_COLUMNS,
        ):
    '''
    Args:
        min_columns: number of resultant columns in combination of all arrays.
    '''

    def get_arrays(pair):
        rows, columns = pair

        def is_valid(blocks):
            '''Filter to block combinations that sum to targetted columns
            '''
            return sum(1 if b.ndim == 1 else b.shape[1] for b in blocks) == columns

        return st.lists(get_array_1d2d(
                    min_rows=rows,
                    max_rows=rows,
                    min_columns=1,
                    max_columns=columns
                    ),
                min_size=1,
                max_size=columns
                ).filter(is_valid)

    return st.tuples(
            st.integers(min_value=min_rows, max_value=max_rows),
            st.integers(min_value=min_columns, max_value=max_columns)
            ).flatmap(
        get_arrays
    )

#-------------------------------------------------------------------------------
# 55203 is just before "high surrogates", and avoids this exception
# UnicodeDecodeError: 'utf-32-le' codec can't decode bytes in position 0-3: code point in surrogate code point range(0xd800, 0xe000)
ST_CODEPOINT_LIMIT = dict(min_codepoint=1, max_codepoint=55203)

ST_LABEL = (st.dates,
        st.datetimes,
        st.integers,
        st.floats,
        st.complex_numbers,
        # st.decimals,
        st.fractions,
        partial(st.characters, **ST_CODEPOINT_LIMIT),
        partial(st.text, st.characters(**ST_CODEPOINT_LIMIT))
    )

ST_VALUE = ST_LABEL + (st.booleans, st.none)


def get_value():
    '''
    Any plausible value.
    '''
    return st.one_of(strat() for strat in ST_VALUE)


def get_label():
    '''
    A hashable suitable for use in an Index.
    '''
    return st.one_of(strat() for strat in ST_LABEL)

def get_labels(min_size: int = 0):
    '''
    Labels are suitable for creating non-date Indices (though they might be dates)
    '''
    # drawing from value so as to include None and booleans
    list_mixed = st.lists(st.one_of(strat() for strat in ST_VALUE),
            min_size=min_size,
            unique=True)

    lists = chain(
            (list_mixed,),
            (st.lists(strat(), min_size=min_size, unique=True) for strat in ST_LABEL),
            )
    return st.one_of(lists)






if __name__ == '__main__':
    from static_frame.core.display_color import HexColor
    import fnmatch
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-n', '--name', default=None)
    parser.add_argument('-c', '--count', default=3, type=int)
    options = parser.parse_args()

    local_items = tuple(locals().items())
    for v in (v for k, v in local_items if callable(v) and k.startswith('get')):

        if options.name:
            if not fnmatch.fnmatch(v.__name__, options.name):
                continue

        print(HexColor.format_terminal('grey', '.' * 50))
        print(HexColor.format_terminal('hotpink', str(v.__name__)))

        for x in range(options.count):
            print(HexColor.format_terminal('grey', '.' * 50))
            print(v().example())










# columns = st.integers(min_value=1, max_value=MAX_COLUMNS).example()

# return st.lists(
#         get_array_2d(min_columns=columns, max_columns=columns),
#         min_size=min_size,
#         max_size=max_size
#         )



