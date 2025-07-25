from __future__ import annotations

import html
import inspect
import os
import platform
import re
import sys
from collections import namedtuple
from functools import partial

import numpy as np
import typing_extensions as tp

from static_frame.core.display_color import HexColor
from static_frame.core.display_config import (
    _DISPLAY_FORMAT_HTML,
    _DISPLAY_FORMAT_MAP,
    _DISPLAY_FORMAT_TERMINAL,
    DisplayConfig,
)
from static_frame.core.util import (
    COMPLEX_TYPES,
    DTYPE_INT_KINDS,
    DTYPE_STR_KINDS,
    FLOAT_TYPES,
    TCallableToIter,
    gen_skip_middle,
)

if tp.TYPE_CHECKING:
    from static_frame.core.index_base import IndexBase  # pragma: no cover
    from static_frame.core.style_config import StyleConfig  # pragma: no cover

    TNDArrayAny = np.ndarray[tp.Any, tp.Any]  # pragma: no cover
    TDtypeAny = np.dtype[tp.Any]  # pragma: no cover
    THeaderSpecifier = tp.Union[
        TDtypeAny, tp.Type[tp.Any], str, 'DisplayHeader', None
    ]  # pragma: no cover


_module = sys.modules[__name__]


# -------------------------------------------------------------------------------
# display infrastructure


# -------------------------------------------------------------------------------
class DisplayTypeCategory:
    """
    Display Type Categories are used for identifying types to which to apply specific formatting.
    """

    CONFIG_ATTR = 'type_color_default'

    @staticmethod
    def in_category(t: tp.Union[type, TDtypeAny]) -> bool:
        return True


class DisplayTypeInt(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_int'

    @staticmethod
    def in_category(t: tp.Union[type, TDtypeAny]) -> bool:
        return isinstance(t, np.dtype) and t.kind in DTYPE_INT_KINDS


class DisplayTypeFloat(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_float'

    @staticmethod
    def in_category(t: tp.Union[type, TDtypeAny]) -> bool:
        return isinstance(t, np.dtype) and t.kind == 'f'


class DisplayTypeComplex(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_complex'

    @staticmethod
    def in_category(t: tp.Union[type, TDtypeAny]) -> bool:
        return isinstance(t, np.dtype) and t.kind == 'c'


class DisplayTypeBool(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_bool'

    @staticmethod
    def in_category(t: tp.Union[type, TDtypeAny]) -> bool:
        return isinstance(t, np.dtype) and t.kind == 'b'


class DisplayTypeObject(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_object'

    @staticmethod
    def in_category(t: tp.Union[type, TDtypeAny]) -> bool:
        return isinstance(t, np.dtype) and t.kind == 'O'


class DisplayTypeStr(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_str'

    @staticmethod
    def in_category(t: tp.Union[type, TDtypeAny]) -> bool:
        return isinstance(t, np.dtype) and t.kind in DTYPE_STR_KINDS


class DisplayTypeDateTime(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_datetime'

    @staticmethod
    def in_category(t: tp.Union[type, TDtypeAny]) -> bool:
        return isinstance(t, np.dtype) and t.kind == 'M'


class DisplayTypeTimeDelta(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_timedelta'

    @staticmethod
    def in_category(t: tp.Union[type, TDtypeAny]) -> bool:
        return isinstance(t, np.dtype) and t.kind == 'm'


class DisplayTypeIndex(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_index'

    @staticmethod
    def in_category(t: tp.Union[type, TDtypeAny]) -> bool:
        from static_frame.core.index_base import IndexBase

        if not inspect.isclass(t):
            return False
        return issubclass(t, IndexBase)


class DisplayTypeSeries(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_series'

    @staticmethod
    def in_category(t: tp.Union[type, TDtypeAny]) -> bool:
        from static_frame import Series

        if not inspect.isclass(t):
            return False
        return issubclass(t, Series)


class DisplayTypeFrame(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_frame'

    @staticmethod
    def in_category(t: tp.Union[type, TDtypeAny]) -> bool:
        from static_frame import Frame

        if not inspect.isclass(t):
            return False
        return issubclass(t, Frame)


class DisplayTypeBus(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_bus'

    @staticmethod
    def in_category(t: tp.Union[type, TDtypeAny]) -> bool:
        from static_frame import Bus

        if not inspect.isclass(t):
            return False
        return issubclass(t, Bus)


class DisplayTypeQuilt(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_quilt'

    @staticmethod
    def in_category(t: tp.Union[type, TDtypeAny]) -> bool:
        from static_frame import Quilt

        if not inspect.isclass(t):
            return False
        return issubclass(t, Quilt)


class DisplayTypeCategoryFactory:
    _DISPLAY_TYPE_CATEGORIES = (
        DisplayTypeInt,
        DisplayTypeFloat,
        DisplayTypeComplex,
        DisplayTypeBool,
        DisplayTypeObject,
        DisplayTypeStr,
        DisplayTypeDateTime,
        DisplayTypeTimeDelta,
        DisplayTypeIndex,
        DisplayTypeSeries,
        DisplayTypeFrame,
        DisplayTypeBus,
        DisplayTypeQuilt,
    )

    _TYPE_TO_CATEGORY_CACHE: tp.Dict[
        tp.Union[type, TDtypeAny], tp.Type[DisplayTypeCategory]
    ] = {}

    @classmethod
    def to_category(
        cls,
        dtype: tp.Union[type, TDtypeAny],
    ) -> tp.Type[DisplayTypeCategory]:
        if dtype not in cls._TYPE_TO_CATEGORY_CACHE:
            category = None
            for dtc in cls._DISPLAY_TYPE_CATEGORIES:
                if dtc.in_category(dtype):
                    category = dtc
                    break
            if not category:
                category = DisplayTypeCategory  # type: ignore

            cls._TYPE_TO_CATEGORY_CACHE[dtype] = category  # type: ignore
            # if not match, assign default

        return cls._TYPE_TO_CATEGORY_CACHE[dtype]


# -------------------------------------------------------------------------------


def terminal_ansi(stream: tp.TextIO = sys.stdout) -> bool:
    """
    Return True if the terminal is ANSI color compatible.
    """
    environ = os.environ
    if 'ANSICON' in environ or 'PYCHARM_HOSTED' in environ:
        return True  # pragma: no cover
    if 'TERM' in environ and environ['TERM'] == 'ANSI':
        return True  # pragma: no cover
    if 'INSIDE_EMACS' in environ:
        return False  # pragma: no cover

    if getattr(stream, 'closed', False):  # if has closed attr and closed
        return False  # pragma: no cover

    if hasattr(stream, 'isatty') and stream.isatty() and platform.system() != 'Windows':
        return True  # pragma: no cover

    return False


# -------------------------------------------------------------------------------

_module._display_active = DisplayConfig()  # type: ignore


class DisplayActive:
    """Utility interface for setting and storing default display configurations."""

    FILE_NAME = '.static_frame.conf'

    @staticmethod
    def set(dc: DisplayConfig) -> None:
        _module._display_active = dc  # type: ignore

    @staticmethod
    def get(**kwargs: tp.Union[bool, int, str]) -> DisplayConfig:
        config: DisplayConfig = _module._display_active
        if not kwargs:
            return config
        args = config.to_dict()
        args.update(kwargs)
        return DisplayConfig(**args)

    @classmethod
    def update(cls, **kwargs: object) -> None:
        args = cls.get().to_dict()
        args.update(kwargs)
        cls.set(DisplayConfig(**args))

    @classmethod
    def _default_fp(cls) -> str:
        # TODO: improve cross platform support
        return os.path.join(os.path.expanduser('~'), cls.FILE_NAME)

    @classmethod
    def write(cls, fp: tp.Optional[str] = None) -> None:
        fp = fp or cls._default_fp()
        dc = cls.get()
        dc.write(fp)

    @classmethod
    def read(cls, fp: tp.Optional[str] = None) -> None:
        fp = fp or cls._default_fp()
        cls.set(DisplayConfig.from_file(fp))


# -------------------------------------------------------------------------------
class DisplayHeader:
    """
    Wrapper for passing in display header that has a name attribute, such as a :obj:`Series` or :obj:`Frame` that displays the ``name`` attribute.
    """

    __slots__ = ('cls', 'name')

    def __init__(self, cls: type, name: tp.Optional[object] = None) -> None:
        """
        Args:
            cls: the Class to be displayed.
            name: an optional name attribute stored on the instance.
        """
        self.cls = cls
        self.name = name

    def __repr__(self) -> str:
        """
        Provide string representation before additon of outer delimiters.
        """
        if self.name is not None:
            return f'{self.cls.__name__}: {self.name}'
        return self.cls.__name__


HeaderInitializer = tp.Optional[tp.Union[str, DisplayHeader]]

# An NT that stores a formatted strings ``format_str`` as well as a ``raw`` string
DisplayCell = namedtuple('DisplayCell', ('format_str', 'raw'))

FORMAT_EMPTY = '{}'


class Display:
    """
    A Display is a string representation of a table, encoded as a list of lists, where list components are equal-width strings, keyed by row index
    """

    __slots__ = (
        '_rows',
        '_config',
        '_outermost',
        '_index_depth',
        '_header_depth',
        '_style_config',
    )

    CHAR_MARGIN = 1
    CELL_EMPTY = DisplayCell(FORMAT_EMPTY, '')
    ELLIPSIS = '...'  # this string is appended to truncated entries
    CELL_ELLIPSIS = DisplayCell(FORMAT_EMPTY, ELLIPSIS)
    ELLIPSIS_CENTER_SENTINEL = object()
    ANSI_ESCAPE = re.compile(
        r'\x1B'
        r'\['
        r'(\d{1,3}(;\d{1,3})*)?'  # Optional numeric codes separated by ;
        r'[A-Za-z]'  # Letter indicating the type of code; most often "m" for colors
    )

    # ---------------------------------------------------------------------------
    # utility methods

    @staticmethod
    def type_attributes(
        type_input: THeaderSpecifier, config: DisplayConfig
    ) -> tp.Tuple[str, tp.Type[DisplayTypeCategory]]:
        """
        Return the `type_input` as a string, applying delimters to either numpy dtypes or Python classes.
        """
        type_ref: tp.Union[type, TDtypeAny]
        if isinstance(type_input, np.dtype):
            type_str = str(type_input)
            type_ref = type_input
        elif isinstance(type_input, DisplayHeader):
            type_str = repr(type_input)
            type_ref = type_input.cls
        elif inspect.isclass(type_input):
            # assert isinstance(type_input, type)
            type_str = type_input.__name__
            type_ref = type_input
        else:
            raise NotImplementedError('no handling for this input', type_input)

        type_category = DisplayTypeCategoryFactory.to_category(type_ref)

        # if config.type_delimiter_left or config.type_delimiter_right:
        left = config.type_delimiter_left or ''
        right = config.type_delimiter_right or ''
        type_label = f'{left}{type_str}{right}'

        return type_label, type_category

    @staticmethod
    def type_color_markup(
        type_category: tp.Type[DisplayTypeCategory], config: DisplayConfig
    ) -> str:
        """
        Return a format string for applying color to a type based on type category and config.

        Returns:
            A templated string with a "text" field for formatting.
        """
        color = getattr(config, type_category.CONFIG_ATTR)
        if config.display_format in _DISPLAY_FORMAT_HTML:
            return HexColor.format_html(color, FORMAT_EMPTY)

        if config.display_format in _DISPLAY_FORMAT_TERMINAL and terminal_ansi():
            return HexColor.format_terminal(color, FORMAT_EMPTY)  # pragma: no cover
            # if not a compatible terminal, return label unaltered

        return FORMAT_EMPTY

    @classmethod
    def to_cell(
        cls, value: tp.Any, config: DisplayConfig, is_dtype: bool = False
    ) -> DisplayCell:
        """
        Given a raw value, return a :obj:`DisplayCell`.
        """
        if is_dtype or inspect.isclass(value) or isinstance(value, DisplayHeader):
            type_str_raw, type_category = cls.type_attributes(value, config=config)
            if config.type_color:
                format_str = cls.type_color_markup(type_category, config)
            else:
                format_str = FORMAT_EMPTY
            return DisplayCell(format_str, type_str_raw)

        # ContainerOperand needs to import Display
        from static_frame.core.container import ContainerBase

        # handling for all other values that are stringable
        if isinstance(value, ContainerBase):
            # NOTE: we do not use type delimieters as ths is an instance, not a class
            msg = value.__class__.__name__
        else:
            msg = str(value)

        # handling for float, complex if str() produces an 'e', then we use the scientific template; otherwise, we use the postional; users can config both to be the same to always get one or the other
        if isinstance(value, FLOAT_TYPES):
            if 'e' in msg:
                msg = config.value_format_float_scientific.format(value)
            else:
                msg = config.value_format_float_positional.format(value)
        elif isinstance(value, COMPLEX_TYPES):
            if 'e' in msg:
                msg = config.value_format_complex_scientific.format(value)
            else:
                msg = config.value_format_complex_positional.format(value)
        elif isinstance(value, str) and config.display_format in _DISPLAY_FORMAT_TERMINAL:
            # When in a TERMINAL mode, separate ASCII color markup found in strings and provide the original string as the formatted string; this will permit correct spacing.
            msg, count = cls.ANSI_ESCAPE.subn('', msg)
            if count:
                return DisplayCell(value, msg)

        return DisplayCell(FORMAT_EMPTY, msg)

    # ---------------------------------------------------------------------------
    # alternate constructor

    @classmethod
    def from_values(
        cls,
        values: TNDArrayAny | tp.Sequence[tp.Any],
        *,
        header: THeaderSpecifier,
        config: tp.Optional[DisplayConfig] = None,
        outermost: bool = False,
        index_depth: int = 0,
        header_depth: int = 0,
        style_config: tp.Optional[StyleConfig] = None,
    ) -> 'Display':
        """
        Given a 1 or 2D ndarray, return a Display instance. Generally 2D arrays are passed here only from TypeBlocks.
        """
        # return a list of lists, where each inner list represents multiple columns
        config = config or DisplayActive.get()

        # create a list of lists, always starting with the header
        rows = []
        if header is not None:
            # NOTE: controlling if the header is applied with type_show is moving to display() methods; this approach will no longer be needed
            # assume that all headers are SF types; skip if type_show is False
            if config.type_show:
                rows.append([cls.to_cell(header, config=config)])
            else:
                rows.append([cls.CELL_EMPTY])

        if values.__class__ is np.ndarray and values.ndim == 2:  # type: ignore
            # NOTE: this is generally only used by TypeBlocks
            # get rows from numpy string formatting
            np_rows = np.array_str(values).split('\n')  # type: ignore
            last_idx = len(np_rows) - 1
            for idx, row in enumerate(np_rows):
                # trim brackets
                end_slice_len = 2 if idx == last_idx else 1
                row = row[2 : len(row) - end_slice_len].strip()
                rows.append([cls.to_cell(row, config=config)])
        else:
            value_gen: tp.Callable[[], tp.Iterator[tp.Any]]
            count_max = config.display_rows
            if len(values) > config.display_rows:
                data_half_count = Display.truncate_half_count(count_max)
                value_gen = partial(
                    gen_skip_middle,
                    forward_iter=values.__iter__,
                    forward_count=data_half_count,
                    reverse_iter=partial(reversed, values),
                    reverse_count=data_half_count,
                    center_sentinel=cls.ELLIPSIS_CENTER_SENTINEL,
                )
            else:
                value_gen = values.__iter__

            for v in value_gen():
                if v is cls.ELLIPSIS_CENTER_SENTINEL:  # center sentinel
                    rows.append([cls.CELL_ELLIPSIS])
                else:
                    rows.append([cls.to_cell(v, config=config)])

        # add the types to the last row
        if values.__class__ is np.ndarray and config.type_show:
            rows.append([cls.to_cell(values.dtype, config=config, is_dtype=True)])  # type: ignore

        return cls(
            rows,
            config=config,
            outermost=outermost,
            index_depth=index_depth,
            header_depth=header_depth,
            style_config=style_config,
        )

    @classmethod
    def from_params(
        cls,
        *,
        index: 'IndexBase',
        columns: 'IndexBase',
        header: DisplayHeader,
        column_forward_iter: TCallableToIter,
        column_reverse_iter: TCallableToIter,
        column_default_iter: TCallableToIter,
        config: tp.Optional[DisplayConfig] = None,
        style_config: tp.Optional[StyleConfig] = None,
    ) -> 'Display':
        """
        Args:
            {config}
        """
        config = config or DisplayActive.get()
        index_depth = index.depth if config.include_index else 0

        # always get the index Display (even if we are not going to use it) to determine how many rows we need (which may include types, as well as truncation with ellipsis).
        display_index = index.display(config)

        # header depth used for HTML and other foramtting; needs to be adjusted if removing types and/or columns and types, When showing types on a Frame, we need 2: one for the Frame type, the other for the index type.
        header_depth = (columns.depth * config.include_columns) + (2 * config.type_show)

        # create an empty display based on index display
        d = cls(
            [list() for _ in range(len(display_index))],
            config=config,
            outermost=True,
            index_depth=index_depth,
            header_depth=header_depth,
            style_config=style_config,
        )

        if config.include_index:
            # this will add more rows to accomodate the index if it is bigger due to types
            d.extend_display(display_index)
            header_column = '' if config.type_show else None
        else:
            header_column = None

        if len(columns) > config.display_columns:
            # columns as they will look after application of truncation and insertion of ellipsis
            data_half_count = cls.truncate_half_count(config.display_columns)

            column_gen = partial(
                gen_skip_middle,
                forward_iter=column_forward_iter,
                forward_count=data_half_count,
                reverse_iter=column_reverse_iter,
                reverse_count=data_half_count,
                center_sentinel=cls.ELLIPSIS_CENTER_SENTINEL,
            )
        else:
            column_gen = column_default_iter  # type: ignore

        for column in column_gen():
            if column is cls.ELLIPSIS_CENTER_SENTINEL:
                d.extend_ellipsis()
            else:
                d.extend_iterable(column, header=header_column)

        # -----------------------------------------------------------------------
        config_transpose = config.to_transpose()

        # -----------------------------------------------------------------------
        # prepare header display of container class
        header_displays = []
        if config.type_show:
            display_cls = cls.from_values((), header=header, config=config_transpose)
            header_displays.append(display_cls.flatten())

        # -----------------------------------------------------------------------
        # prepare columns display
        if config.include_columns:
            # need to apply the config_transpose such that it truncates it based on the the max columns, not the max rows
            display_columns = columns.display(config_transpose)

            if config.type_show:
                index_depth_extend = index.depth - 1
                spacer_insert_index = 1  # after the first, the name
            elif not config.type_show and config.include_index:
                index_depth_extend = index.depth
                spacer_insert_index = 0
            elif not config.include_index:  # type_show must be False
                index_depth_extend = 0
                spacer_insert_index = 0

            # add spacers to from of columns when we have a hierarchical index
            for _ in range(index_depth_extend):
                # will need a width equal to the column depth
                row = [cls.to_cell('', config=config) for _ in range(columns.depth)]
                spacer = cls([row])
                display_columns.insert_displays(spacer, insert_index=spacer_insert_index)

            if columns.depth > 1:
                display_columns_horizontal = display_columns.transform()
            else:  # can just flatten a single column into one row
                display_columns_horizontal = display_columns.flatten()

            header_displays.append(display_columns_horizontal)

        if header_displays:
            d.insert_displays(*header_displays)

        return d

    # ---------------------------------------------------------------------------
    # core cell-to-rwo expansion routines

    @staticmethod
    def truncate_half_count(count_target: int) -> int:
        """Given a target number of rows or columns, return the count of half as found in presentation where one column is used for the ellipsis. The number returned will always be odd. For example, given a target of 5 we allocate 2 per half (plus 1 reserved for middle)."""
        if count_target <= 4:
            return 1  # practical floor for all values of 4 or less
        return (count_target - 1) // 2

    @classmethod
    def _get_max_width_pad_width(
        cls,
        *,
        rows: tp.Sequence[tp.Sequence[DisplayCell]],
        col_idx_src: int,
        col_last_src: int,
        row_indices: tp.Iterable[int],
        config: tp.Optional[DisplayConfig] = None,
    ) -> tp.Tuple[int, int]:
        """
        Called once for each column to determine the maximum_width and pad_width for a particular column. All row data is passed to this function, and cell values are looked up directly with the `col_idx_src` argument.

        Args:
            rows: iterable of all rows, containing DisplayCell instances
            col_idx_src: the integer index for the currrent column
            col_last_src: the integer index for the last column
            row_indices: passed here so same range() can be reused.
        """
        config = config or DisplayActive.get()

        is_last = col_idx_src == col_last_src
        is_first = col_idx_src == 0

        if is_first:
            width_limit = config.cell_max_width_leftmost
        else:
            width_limit = config.cell_max_width

        max_width = 0
        for row_idx_src in row_indices:
            # get existing max width, up to the max
            row = rows[row_idx_src]
            if col_idx_src >= len(row):  # this row does not have this column
                continue
            cell = row[col_idx_src]
            max_width = max(max_width, len(cell.raw))
            # if already exceeded max width, stop iterating
            if max_width >= width_limit:
                break

        # get most binding constraint
        max_width = min(max_width, width_limit)

        if (config.cell_align_left is True and is_last) or (
            config.cell_align_left is False and is_first
        ):
            pad_width = max_width
        else:
            pad_width = max_width + cls.CHAR_MARGIN

        return max_width, pad_width

    @classmethod
    def _to_rows_cells(
        cls,
        display: 'Display',
        config: tp.Optional[DisplayConfig] = None,
        # index_depth: int = 0,
        # columns_depth: int = 0,
    ) -> tp.Iterable[tp.Iterable[str]]:
        """
        Given a Display object, return an iterable of iterables of strings, where each iterable contains strings for all cells in that row, with appropriate padding (if necessary) applied. Based on configruation, align cells left or right with space and return one joined string per row.

        Returns:
            Returns an iterable of formatted strings, generally one per row.
        """
        config = config or DisplayActive.get()

        # find max columns for all defined rows
        col_count_src = max(len(row) for row in display._rows)
        col_last_src = col_count_src - 1

        row_count_src = len(display._rows)
        row_indices = range(row_count_src)

        rows: tp.List[tp.List[str]] = [[] for _ in row_indices]

        # if we normalize, we truncate cells and pad
        dfc = _DISPLAY_FORMAT_MAP[config.display_format]
        is_html = config.display_format in _DISPLAY_FORMAT_HTML

        for col_idx_src in range(col_count_src):
            # for each column, get the max width
            if dfc.CELL_WIDTH_NORMALIZE:
                max_width, pad_width = cls._get_max_width_pad_width(
                    rows=display._rows,
                    col_idx_src=col_idx_src,
                    col_last_src=col_last_src,
                    row_indices=row_indices,
                    config=config,
                )

            for row_idx_src in row_indices:
                display_row = display._rows[row_idx_src]
                if col_idx_src >= len(display_row):
                    cell = cls.CELL_EMPTY
                else:
                    cell = display_row[col_idx_src]

                cell_format_str = cell.format_str
                cell_raw = cell.raw

                if dfc.CELL_WIDTH_NORMALIZE:
                    if len(cell.raw) > max_width:
                        # must truncate if cell width is greater than max width
                        width_truncate = max_width - len(cls.CELL_ELLIPSIS.raw)

                        # NOTE: this might truncate scientific notation
                        cell_raw = cell_raw[:width_truncate] + cls.ELLIPSIS
                        if is_html:
                            cell_raw = html.escape(cell_raw)

                        cell_formatted = cell_format_str.format(cell_raw)
                        cell_fill_width = cls.CHAR_MARGIN  # should only be margin left
                    else:
                        if is_html:
                            cell_raw = html.escape(cell_raw)
                        cell_formatted = cell_format_str.format(cell_raw)
                        cell_fill_width = pad_width - len(
                            cell.raw
                        )  # this includes margin

                    if config.cell_align_left:
                        # must manually add space as color chars make ljust not work
                        msg = cell_formatted + ' ' * cell_fill_width
                    else:
                        msg = ' ' * cell_fill_width + cell_formatted

                else:  # no width normalization
                    if is_html:
                        cell_raw = html.escape(cell_raw)
                    msg = cell_format_str.format(cell_raw)

                rows[row_idx_src].append(msg)

        return rows

    # ---------------------------------------------------------------------------
    def __init__(
        self,
        rows: tp.List[tp.List[DisplayCell]],
        config: tp.Optional[DisplayConfig] = None,
        outermost: bool = False,
        index_depth: int = 0,
        header_depth: int = 0,
        *,
        style_config: tp.Optional[StyleConfig] = None,
    ) -> None:
        """Define rows as a list of lists, for each row; the contained DisplayCell instances may be of different size, but they are expected to be aligned vertically in final presentation.

        Args:
            header_depth: columns depth plus any additional lines used for headers
        """
        config = config or DisplayActive.get()

        self._rows = rows
        self._config = config
        self._outermost = outermost
        self._index_depth = index_depth
        self._header_depth = header_depth
        self._style_config = style_config

    def __repr__(self) -> str:
        rows = self._to_rows_cells(self, self._config)
        dfc = _DISPLAY_FORMAT_MAP[self._config.display_format]

        if self._outermost:
            header = []
            body = []
            for idx, row in enumerate(rows):
                iloc_row = idx - self._header_depth
                row = ''.join(
                    dfc.markup_row(
                        row,
                        index_depth=self._index_depth,
                        iloc_row=iloc_row,
                        style_config=self._style_config,
                    )
                ).rstrip()
                if idx < self._header_depth:
                    header.append(row)
                else:
                    body.append(row)

            outermost = []
            if header:
                header_str = dfc.markup_header(dfc.LINE_SEP.join(header))
                outermost.append(header_str)

            body_str = dfc.markup_body(dfc.LINE_SEP.join(body))
            outermost.append(body_str)
            return dfc.markup_outermost(
                dfc.LINE_SEP.join(outermost),
                style_config=self._style_config,
            )
        # use mostly for debugging; strings in rows already have space / padding
        return dfc.LINE_SEP.join(''.join(r) for r in rows)

    def to_rows(self) -> tp.Sequence[str]:
        """
        Alternate output method for observing rows as strings within a list. Useful for testing.
        """
        post = []
        for row in self._to_rows_cells(self, self._config):
            line = ''.join(row).rstrip()
            post.append(line)
        return post

    def __iter__(self) -> tp.Iterator[tp.List[str]]:
        for row in self._rows:
            yield [cell.format_str.format(cell.raw) for cell in row]

    def __len__(self) -> int:
        return len(self._rows)

    # ---------------------------------------------------------------------------
    # in place mutation

    def extend_display(self, display: 'Display') -> None:
        """
        Mutate this display by extending to the right (adding columns) with the passed display.
        """
        # NOTE: do not want to pass config or call format here as we call this for each column or block we add
        for row_idx, row in enumerate(display._rows):
            if row_idx == len(self._rows):
                self._rows.append([])
            self._rows[row_idx].extend(row)

    def extend_iterable(
        self, iterable: tp.Sequence[tp.Any] | TNDArrayAny, header: HeaderInitializer
    ) -> None:
        """
        Add a single iterable (as a column) to the display.
        """
        row_idx_start = 0
        if header is not None:
            self._rows[0].append(self.to_cell(header, config=self._config))
            row_idx_start = 1

        # truncate iterable if necessary
        count_max = self._config.display_rows

        if len(iterable) > count_max:
            data_half_count = self.truncate_half_count(count_max)
            value_gen: tp.Callable[[], tp.Iterator[tp.Any]] = partial(
                gen_skip_middle,
                forward_iter=iterable.__iter__,
                forward_count=data_half_count,
                reverse_iter=partial(reversed, iterable),
                reverse_count=data_half_count,
                center_sentinel=self.ELLIPSIS_CENTER_SENTINEL,
            )
        else:
            value_gen = iterable.__iter__

        # start at 1 as 0 is header
        idx = 0  # store in case value gen is empty
        for idx, value in enumerate(value_gen(), start=row_idx_start):
            if value is self.ELLIPSIS_CENTER_SENTINEL:
                self._rows[idx].append(self.CELL_ELLIPSIS)
            else:
                self._rows[idx].append(self.to_cell(value, config=self._config))

        if isinstance(iterable, np.ndarray) and self._config.type_show:
            self._rows[idx + 1].append(
                self.to_cell(iterable.dtype, config=self._config, is_dtype=True)
            )

    def extend_ellipsis(self) -> None:
        """Append an ellipsis over all rows."""
        for row in self._rows:
            row.append(self.CELL_ELLIPSIS)

    def insert_displays(self, *displays: 'Display', insert_index: int = 0) -> None:
        """
        Insert rows on top of existing rows.
        args:
            Each arg in args is an instance of Display
            insert_index: the index at which to start insertion
        """
        # each arg is a list, to be a new row
        # assume each row in display becomes a column
        new_rows: tp.List[tp.List[DisplayCell]] = []
        for display in displays:
            new_rows.extend(display._rows)

        rows = []
        rows.extend(self._rows[:insert_index])
        rows.extend(new_rows)
        rows.extend(self._rows[insert_index:])
        self._rows = rows

    # ---------------------------------------------------------------------------
    # return a new display

    def flatten(self) -> 'Display':
        """
        Return a Display from this Display that is a single row, formed the contents of all rows put in a single senquece, from the top down. Through this a single-column index Display can be made into a single row Display.
        """
        row = []
        for part in self._rows:
            row.extend(part)
        rows = [row]
        return self.__class__(rows, config=self._config)

    def transform(self) -> 'Display':
        """
        Return Display transformed (rotated) on its upper left; i.e., the first column becomes the first row.
        """

        # assume first row gives us column count
        col_count = len(self._rows[0])
        rows: tp.List[tp.List[DisplayCell]] = [[] for _ in range(col_count)]
        for r in self._rows:
            for idx, cell in enumerate(r):
                rows[idx].append(cell)
        return self.__class__(rows, config=self._config)
