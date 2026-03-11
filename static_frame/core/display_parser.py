"""Routines for parsing the string output of Display into Frame and Series objects."""

from __future__ import annotations

import re
import typing as tp

import numpy as np
from arraykit import iterable_str_to_array_1d

from static_frame.core.util import DTYPE_OBJECT

if tp.TYPE_CHECKING:
    from collections.abc import Iterable

    from static_frame.core.index_base import IndexBase  # pragma: no cover

# ---------------------------------------------------------------------------
# Regular expressions

ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*m')

# Matches any <...> type marker, including nested ones such as <<U1>
DTYPE_RE = re.compile(r'<[^>]*>')

# ---------------------------------------------------------------------------
# Internal helpers


def scrub_and_split(text: str) -> Iterable[str]:
    return ANSI_ESCAPE_RE.sub('', text).strip().splitlines()


def find_dtype_positions(line: str) -> tp.List[tp.Tuple[int, str]]:
    """Return ``(start_position, dtype_str)`` for every ``<dtype>`` marker in *line*."""
    return [(m.start(), m.group()) for m in DTYPE_RE.finditer(line)]


def dtype_to_np(dtype_str: str) -> np.dtype:
    # strip one leading ``<`` and one trailing ``>``
    return np.dtype(dtype_str[1:-1])


def extract_cell(line: str, start: int, end: int) -> str:
    if start >= len(line):
        return ''
    return line[start : min(end, len(line))].strip()


def make_array(values: tp.List[str], dtype: np.dtype) -> np.ndarray:
    """Return an immutable :class:`numpy.ndarray` built from *values* cast to *dtype*."""
    if dtype == DTYPE_OBJECT:
        # For object dtype, handle special string representations
        arr = np.empty(len(values), dtype=dtype)
        for i, v in enumerate(values):
            if v == 'None':
                arr[i] = None
            elif v == 'True':
                arr[i] = True
            elif v == 'False':
                arr[i] = False
            else:
                try:
                    if '.' not in v and 'e' not in v.lower():
                        arr[i] = int(v)
                    else:
                        arr[i] = float(v)
                except (ValueError, TypeError):
                    arr[i] = v  # keep as str
    else:
        arr = iterable_str_to_array_1d(values, dtype=dtype)
    arr.flags.writeable = False
    return arr


def parse_header_line(line: str) -> tp.Tuple[str, tp.Optional[str]]:
    """
    Parse a class-header line such as ``<Frame>`` or ``<Frame: myname>``.

    Returns a ``(cls_name, name_or_None)`` pair.
    """
    if not (line.startswith('<') and line.endswith('>')):
        raise ValueError(f'Invalid header line: {line!r}')
    inner = line[1:-1]  # remove outer ``<`` and ``>``
    if ': ' in inner:
        cls_name, name_str = inner.split(': ', 1)
        return cls_name, name_str
    return inner, None


def find_standalone_index_line(lines: tp.List[str]) -> int:
    """
    Return the 0-based index of the standalone index-type marker line.

    A standalone line looks like ``<Index>`` or ``<IndexHierarchy: name>``.
    It is identified by starting with ``<Index`` and containing exactly one
    ``<`` character (embedded dtype markers such as ``<<U1>`` would add more).
    """
    for i, line in enumerate(lines[1:], 1):  # skip line 0 (class header)
        stripped = line.strip()
        if (
            stripped.startswith('<Index')
            and stripped.endswith('>')
            and stripped.count('<') == 1
        ):
            return i
    raise ValueError('Could not find a standalone index type line in the display text.')


def find_index_depth(
    first_header_row: str,
    dtype_positions: tp.List[tp.Tuple[int, str]],
) -> int:
    """
    Determine the number of index columns by scanning the first column-header
    row for the first cell that contains a non-empty label.

    For a regular (depth-1) index the label at ``dtype_positions[1]`` is the
    first column label.  For an ``IndexHierarchy`` of depth *k* the cells at
    positions 1 … k-1 are blank (they fall under the merged index-type marker),
    and the first label appears at position *k*.
    """
    for i in range(1, len(dtype_positions)):
        pos = dtype_positions[i][0]
        next_pos = (
            dtype_positions[i + 1][0]
            if i + 1 < len(dtype_positions)
            else len(first_header_row)
        )
        cell = extract_cell(first_header_row, pos, next_pos)
        # strip any dtype markers that may trail at the end of the last cell
        cell = DTYPE_RE.sub('', cell).strip()
        if cell:
            return i  # first non-empty label: this is the first value column
    return len(dtype_positions)  # everything is the index (edge case)


def extract_column_header_data(
    header_rows: tp.List[str],
    dtype_positions: tp.List[tp.Tuple[int, str]],
    index_depth: int,
) -> tp.List[tp.Tuple[tp.List[str], str]]:
    """
    Extract the column labels and the trailing *columns-index dtype* string
    from each header row.

    Returns a list (one entry per display row) of ``(labels, trailing_dtype_str)``
    tuples, where *trailing_dtype_str* is the dtype of the columns index at
    that level (e.g. ``'<<U1>'`` or ``'<int64>'``).
    """
    n_value_cols = len(dtype_positions) - index_depth
    result: tp.List[tp.Tuple[tp.List[str], tp.Optional[str]]] = []

    for row in header_rows:
        row_labels: tp.List[str] = []
        trailing_dtype: str

        for i, pos_idx in enumerate(range(index_depth, len(dtype_positions))):
            pos = dtype_positions[pos_idx][0]
            next_pos = (
                dtype_positions[pos_idx + 1][0]
                if pos_idx + 1 < len(dtype_positions)
                else len(row)
            )
            cell = extract_cell(row, pos, next_pos)

            if i == n_value_cols - 1:
                # The last cell may contain the columns-index dtype at its end
                m = DTYPE_RE.search(cell)
                if m:
                    trailing_dtype = m.group()
                    cell = cell[: m.start()].strip()
                else:
                    raise ValueError('cannot find column dtype')

            row_labels.append(cell)

        result.append((row_labels, trailing_dtype))

    return result


def build_index(
    values: tp.List[str],
    dtype: np.dtype,
    index_name: tp.Optional[str],
) -> 'IndexBase':
    """
    Build an :obj:`Index` (or a datetime-specialised subclass) from string
    *values* cast to *dtype*, optionally with *index_name*.
    """
    from static_frame.core.index import Index
    from static_frame.core.index_datetime import dtype_to_index_cls

    arr = make_array(values, dtype)

    if dtype.kind == 'M':  # datetime64
        idx_cls = dtype_to_index_cls(static=True, dtype=dtype)
    else:
        idx_cls = Index
    return idx_cls(arr, name=index_name)


def build_columns(
    levels_data: tp.List[tp.Tuple[tp.List[str], str]],
) -> 'IndexBase':
    """
    Build a columns :obj:`Index` (or :obj:`IndexHierarchy` when there are
    multiple levels) from *levels_data*.

    Each entry in *levels_data* is a ``(labels, dtype_str)`` pair where
    *dtype_str* is the dtype of that level's column-index labels.  An empty
    *levels_data* list produces an empty :obj:`Index`.
    """
    from static_frame.core.index import Index
    from static_frame.core.index_hierarchy import IndexHierarchy

    if not levels_data:
        return Index(())

    if len(levels_data) == 1:
        labels, dtype_str = levels_data[0]
        arr = make_array(labels, dtype_to_np(dtype_str))
        return Index(arr)

    # Multiple levels → IndexHierarchy columns
    level_arrays: tp.List[np.ndarray] = []
    for labels, dtype_str in levels_data:
        arr = make_array(labels, dtype_to_np(dtype_str))
        level_arrays.append(arr)

    if level_arrays[0].size == 0:
        return IndexHierarchy.from_labels((), depth_reference=len(levels_data))

    return IndexHierarchy.from_values_per_depth(level_arrays)


# ---------------------------------------------------------------------------
# Low-level parsing functions (return raw data, not constructed containers)

TFrameParseResult = tp.Tuple[
    tp.List[np.ndarray],  # column arrays
    'IndexBase',  # columns index
    'IndexBase',  # row index
    tp.Optional[str],  # frame name
]


def display_parse_frame(
    display: str,
) -> TFrameParseResult:
    """
    Parse the display string of a :obj:`Frame` and return a tuple of
    ``(column_arrays, columns_index, row_index, name)`` suitable for passing
    to :meth:`Frame.from_fields`.
    """
    from static_frame.core.index_hierarchy import IndexHierarchy

    lines = scrub_and_split(display)
    if not lines:
        raise ValueError('Empty display string.')

    # 1. Parse the class-header line (line 0); ignore Frame class as will be decided by calling class
    _, frame_name = parse_header_line(lines[0])

    # 2. Locate the standalone index-type line
    standalone_idx = find_standalone_index_line(lines)
    standalone_line = lines[standalone_idx].strip()
    is_index_hierarchy = standalone_line.startswith('<IndexHierarchy')

    # The index name lives in the standalone line (e.g. '<Index: myidx>')
    _, index_name = parse_header_line(standalone_line)

    # 3. Partition the display into its sections
    col_header_rows = lines[1:standalone_idx]
    data_rows = lines[standalone_idx + 1 : -1]
    dtype_row = lines[-1]

    # 4. Determine column positions from the dtype row
    dtype_positions = find_dtype_positions(dtype_row)
    if not dtype_positions:
        raise ValueError('Could not find dtype information in the display text.')

    # 5. Determine index depth
    if col_header_rows:
        if not is_index_hierarchy:
            index_depth = 1
        else:
            index_depth = find_index_depth(col_header_rows[0], dtype_positions)
    else:  # No column-header rows → no value columns (empty frame edge case)
        index_depth = len(dtype_positions)

    # 6. Extract column labels from column header rows
    if col_header_rows:
        columns_data = extract_column_header_data(
            col_header_rows, dtype_positions, index_depth
        )
    else:
        columns_data = []

    # 7. Parse data rows: collect index-value strings and column-value strings
    positions = [p for p, _ in dtype_positions]

    index_cells: tp.List[tp.List[str]] = []
    data_cells: tp.List[tp.List[str]] = []

    for row in data_rows:
        cells = [
            extract_cell(
                row,
                positions[i],
                positions[i + 1] if i + 1 < len(positions) else len(row),
            )
            for i in range(len(positions))
        ]
        index_cells.append(cells[:index_depth])
        data_cells.append(cells[index_depth:])

    # 8. Build the row index
    index_dtypes = [dtype_to_np(dt) for _, dt in dtype_positions[:index_depth]]
    if not index_dtypes:
        raise ValueError('cannot find index dtypes')

    if is_index_hierarchy and index_depth > 1:
        level_arrays = [
            make_array([r[d] for r in index_cells], index_dtypes[d])
            for d in range(index_depth)
        ]
        if level_arrays[0].size == 0:
            row_index: IndexBase = IndexHierarchy.from_labels(
                (), depth_reference=index_depth
            )
        else:
            row_index = IndexHierarchy.from_values_per_depth(
                level_arrays, name=index_name
            )
    else:  # 1D index
        row_index = build_index([r[0] for r in index_cells], index_dtypes[0], index_name)

    # 9. Build each data column array
    n_rows = len(data_cells)
    n_cols = len(dtype_positions) - index_depth
    column_dtypes = [dtype_to_np(dt) for _, dt in dtype_positions[index_depth:]]

    arrays: tp.List[np.ndarray] = [
        make_array(
            [data_cells[r][col_idx] for r in range(n_rows)],
            column_dtypes[col_idx],
        )
        for col_idx in range(n_cols)
    ]

    # 10. Build the columns Index (or IndexHierarchy)
    columns_index: IndexBase = build_columns(columns_data)

    return arrays, columns_index, row_index, frame_name


def display_parse_series(
    display: str,
) -> tp.Tuple[np.ndarray, 'IndexBase', tp.Optional[str]]:
    """
    Parse the display string of a :obj:`Series` and return a tuple of
    ``(values_array, index, name)`` suitable for passing to the
    :obj:`Series` constructor.
    """
    from static_frame.core.index_hierarchy import IndexHierarchy

    lines = scrub_and_split(display)
    if not lines:
        raise ValueError('Empty display string.')

    # 1. Parse the class-header line (line 0)
    _, series_name = parse_header_line(lines[0])

    # 2. Locate the standalone index-type line (line 1 for a regular Series)
    standalone_idx = find_standalone_index_line(lines)
    standalone_line = lines[standalone_idx].strip()
    is_index_hierarchy = standalone_line.startswith('<IndexHierarchy')

    # Index name from the standalone line
    _, index_name = parse_header_line(standalone_line)

    # 3. Partition the display into its sections
    data_rows = lines[standalone_idx + 1 : -1]
    dtype_row = lines[-1]

    # 4. Determine column positions and dtypes from the dtype row
    dtype_positions = find_dtype_positions(dtype_row)
    if not dtype_positions:
        raise ValueError('Could not find dtype information in the display text.')

    # For a Series there is always exactly 1 value column
    index_depth = len(dtype_positions) - 1
    positions = [p for p, _ in dtype_positions]

    # 5. Parse data rows
    index_cells: tp.List[tp.List[str]] = []
    data_cells: tp.List[str] = []

    for row in data_rows:
        cells = [
            extract_cell(
                row,
                positions[i],
                positions[i + 1] if i + 1 < len(positions) else len(row),
            )
            for i in range(len(positions))
        ]
        index_cells.append(cells[:index_depth])
        data_cells.append(cells[index_depth])

    # 6. Build the index
    index_dtypes = [dtype_to_np(dt) for _, dt in dtype_positions[:index_depth]]

    if is_index_hierarchy and index_depth > 1:
        level_arrays = [
            make_array([r[d] for r in index_cells], index_dtypes[d])
            for d in range(index_depth)
        ]
        if level_arrays[0].size == 0:
            index: IndexBase = IndexHierarchy.from_labels((), depth_reference=index_depth)
        else:
            index = IndexHierarchy.from_values_per_depth(level_arrays, name=index_name)
    else:
        index_dtype = index_dtypes[0] if index_dtypes else np.dtype(object)
        index = build_index([r[0] for r in index_cells], index_dtype, index_name)

    # 7. Build the values array
    value_dtype = dtype_to_np(dtype_positions[index_depth][1])
    values_array = make_array(data_cells, value_dtype)

    return values_array, index, series_name
