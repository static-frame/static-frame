import typing as tp
import os
from pathlib import Path

import numpy as np

from static_frame.core import util


class SlicedArray(tp.NamedTuple):
    '''An array sliced into 4 parts that can be used to construct a Frame'''
    index: np.ndarray
    data: np.ndarray
    columns: np.ndarray
    top_left: np.ndarray


def read_file(p: Path):
    with open(p) as f:
        yield from f


def to_line_iter(x: util.PathSpecifierOrFileLikeOrIterator) -> tp.Iterator[str]:
    '''Unify all input types taken by StaticFrame's constructors into an iterable of lines.
    '''
    if isinstance(x, (str, os.PathLike)):
        return read_file(Path(x))
    return x


def slice_index_and_columns(
        array: np.ndarray,
        index_depth: int,
        columns_depth: int,
        skip_header: int=0,
        skip_footer: int=0,
    ) -> SlicedArray:
    '''Slice the given array into 4 parts: index, columns, data, and the unused upper left corner.

    Optionally skip header and footer rows.
    '''
    # Skip header and footer
    array = array[skip_header:array.shape[0]-skip_footer]

    # Slice into 4 quadrants, named as per compass.
    nw = array[:columns_depth, :index_depth]
    ne = array[:columns_depth, index_depth:]
    sw = array[columns_depth:, :index_depth]
    se = array[columns_depth:, index_depth:]

    return SlicedArray(
        index=sw,
        data=se,
        columns=ne,
        top_left=nw,
    )

