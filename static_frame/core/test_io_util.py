import typing as tp
from pathlib import Path
from os import PathLike
import io

import pytest
import py
import numpy as np

from static_frame.core import io_util
from static_frame.core.util import PathSpecifierOrFileLikeOrIterator


def _to_path(data: str, tmpdir: py.path.local) -> Path:
    f = tmpdir / 'file'
    f.write_text(data, 'utf8')
    return Path(f)


def _to_str(data: str, tmpdir: py.path.local) -> str:
    return str(_to_path(data, tmpdir))


def _to_textio(data: str, _: tp.Optional[py.path.local]=None) -> tp.TextIO:
    return io.StringIO(data)


def _to_str_iter(data: str, tmpdir: py.path.local) -> tp.Iterator[str]:
    # This looks kind of redundant, but may not be, as it hides methods of the io object.
    return (s for s in io.StringIO(data))


TYPE_TO_CONSTRUCTOR = {
    str: _to_str,
    PathLike: _to_path,
    tp.TextIO: _to_textio,
    tp.Iterator[str]: _to_textio,
}


@pytest.fixture(params=PathSpecifierOrFileLikeOrIterator.__args__)
def input_format_and_expected(request, tmpdir):
    expected = ('a,b,c\n', 'd,e,f')
    data = ''.join(expected)
    constructor = TYPE_TO_CONSTRUCTOR[request.param]
    yield constructor(data, tmpdir), expected


def test_to_line_iter(input_format_and_expected):
    input_, expected = input_format_and_expected

    iterator = io_util.to_line_iter(input_)
    assert tuple(iterator) == expected


def test_slice_index_and_columns():
    shape = (5, 10)
    a = np.zeros(shape, dtype=np.int8)
    r = io_util.slice_index_and_columns(a, 0, 0)
    assert r.data.shape == shape
    assert all(x.size==0 for x in (r.index, r.columns, r.top_left))

    r = io_util.slice_index_and_columns(a, 1, 0)
    assert r.data.shape == (5, 9)
    assert r.index.shape == (5, 1)
    assert all(x.size==0 for x in (r.columns, r.top_left))

    r = io_util.slice_index_and_columns(a, 0, 1)
    assert r.data.shape == (4, 10)
    assert r.columns.shape == (1, 10)
    assert all(x.size==0 for x in (r.index, r.top_left))

    r = io_util.slice_index_and_columns(a, 2, 1)
    assert r.data.shape == (4, 8)
    assert r.columns.shape == (1, 8)
    assert r.index.shape == (4, 2)
    assert r.top_left.shape == (1, 2)


def test_slice_index_and_columns_skip_header_footer():
    shape = (5, 10)
    a = np.zeros(shape, dtype=np.int8)
    r = io_util.slice_index_and_columns(a, 0, 0, 1, 1)
    assert r.data.shape == (3, 10)
    assert all(x.size==0 for x in (r.index, r.columns, r.top_left))

    r = io_util.slice_index_and_columns(a, 1, 0, 2, 0)
    assert r.data.shape == (3, 9)
    assert r.index.shape == (3, 1)
    assert all(x.size==0 for x in (r.columns, r.top_left))

    r = io_util.slice_index_and_columns(a, 0, 1, 0, 2)
    assert r.data.shape == (2, 10)
    assert r.columns.shape == (1, 10)
    assert all(x.size==0 for x in (r.index, r.top_left))

    r = io_util.slice_index_and_columns(a, 2, 1, 2, 2)
    assert r.data.shape == (0, 8)
    assert r.columns.shape == (1, 8)
    assert r.index.shape == (0, 2)
    assert r.top_left.shape == (1, 2)


def test_to_array():
    expected = ('junk','a,b,c', 'd,e,f', 'junk', 'junk,junk')
    iter_ = _to_textio('\n'.join(expected))
    array = io_util.csv_to_array(iter_, delimiter=',', skip_header=1, skip_footer=2)
    assert array.tolist() == [['a', 'b', 'c'], ['d', 'e', 'f']]
