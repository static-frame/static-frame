import typing as tp
import sys
import json
import os
import html
import inspect
import platform

from enum import Enum

from functools import partial
from collections import namedtuple

import numpy as np


from static_frame.core.util import _gen_skip_middle
from static_frame.core.display_color import HexColor
from static_frame.core import display_html_datatables

from static_frame.core.util import _DTYPE_INT_KIND
from static_frame.core.util import _DTYPE_STR_KIND

_module = sys.modules[__name__]



ColorConstructor = tp.Union[int, str]

#-------------------------------------------------------------------------------
# display infrastructure


#-------------------------------------------------------------------------------

class DisplayTypeCategory:
    '''
    Display Type Categories are used for identifying types to which to apply specific formatting.
    '''
    CONFIG_ATTR = 'type_color_default'

    @staticmethod
    def is_dtype(t: type) -> bool:
        '''Utility method to identify NP dtypes.
        '''
        return isinstance(t, np.dtype)

    @classmethod
    def in_category(cls, t: type) -> bool:
        return True


class DisplayTypeInt(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_int'

    @classmethod
    def in_category(cls, t: type) -> bool:
        return cls.is_dtype(t) and t.kind in _DTYPE_INT_KIND

class DisplayTypeFloat(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_float'

    @classmethod
    def in_category(cls, t: type) -> bool:
        return cls.is_dtype(t) and t.kind == 'f'

class DisplayTypeComplex(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_complex'

    @classmethod
    def in_category(cls, t: type) -> bool:
        return cls.is_dtype(t) and t.kind == 'c'

class DisplayTypeBool(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_bool'

    @classmethod
    def in_category(cls, t: type) -> bool:
        return cls.is_dtype(t) and t.kind == 'b'

class DisplayTypeObject(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_object'

    @classmethod
    def in_category(cls, t: type) -> bool:
        return cls.is_dtype(t) and t.kind == 'O'

class DisplayTypeStr(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_str'

    @classmethod
    def in_category(cls, t: type) -> bool:
        return cls.is_dtype(t) and t.kind in _DTYPE_STR_KIND

class DisplayTypeDateTime(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_datetime'

    @classmethod
    def in_category(cls, t: type) -> bool:
        return cls.is_dtype(t) and t.kind == 'M'

class DisplayTypeTimeDelta(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_timedelta'

    @classmethod
    def in_category(cls, t: type) -> bool:
        return cls.is_dtype(t) and t.kind == 'm'


class DisplayTypeIndex(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_index'

    @classmethod
    def in_category(cls, t: type) -> bool:
        from static_frame.core.index_base import IndexBase
        if not inspect.isclass(t):
            return False
        return issubclass(t, IndexBase)

class DisplayTypeSeries(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_series'

    @classmethod
    def in_category(cls, t: type) -> bool:
        from static_frame import Series
        if not inspect.isclass(t):
            return False
        return issubclass(t, Series)

class DisplayTypeFrame(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_frame'

    @classmethod
    def in_category(cls, t: type) -> bool:
        from static_frame import Frame
        if not inspect.isclass(t):
            return False
        return issubclass(t, Frame)


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
            DisplayTypeFrame
            )

    _TYPE_TO_CATEGORY_CACHE = {}

    @classmethod
    def to_category(cls, dtype: type):
        if dtype not in cls._TYPE_TO_CATEGORY_CACHE:
            category = None
            for dtc in cls._DISPLAY_TYPE_CATEGORIES:
                if dtc.in_category(dtype):
                    category = dtc
                    break
            if not category:
                category = DisplayTypeCategory

            cls._TYPE_TO_CATEGORY_CACHE[dtype] = category
            # if not match, assign default

        return cls._TYPE_TO_CATEGORY_CACHE[dtype]



#-------------------------------------------------------------------------------

# NOTE: needs to be jsonable to use directly in enum

class DisplayFormats(str, Enum):
    '''
    Define display output format.
    '''

    HTML_PRE = 'html_pre'
    HTML_TABLE = 'html_table'
    HTML_DATATABLES = 'html_datatables'
    TERMINAL = 'terminal'

_DISPLAY_FORMAT_HTML = {
        DisplayFormats.HTML_PRE,
        DisplayFormats.HTML_TABLE,
        DisplayFormats.HTML_DATATABLES
        }
_DISPLAY_FORMAT_TERMINAL = {DisplayFormats.TERMINAL}


class DisplayFormat:

    CELL_WIDTH_NORMALIZE = True
    LINE_SEP = '\n'

    @staticmethod
    def markup_row(
            row: tp.Iterable[str],
            header_depth: int
            ) -> tp.Generator[str, None, None]:
        '''
        Args:
            header_depth: number of columns that should be treated as headers.
        '''
        for msg in row:
            yield msg

    @staticmethod
    def markup_header(msg: str):
        return msg

    @staticmethod
    def markup_body(msg: str):
        return msg

    @staticmethod
    def markup_outermost(msg: str,
            identifier: tp.Optional[str] = None
            ) -> str:
        return msg

class DisplayFormatTerminal(DisplayFormat):
    pass

class DisplayFormatHTMLTable(DisplayFormat):

    CELL_WIDTH_NORMALIZE = False
    LINE_SEP = ''

    @staticmethod
    def markup_row(
            row: tp.Iterable[str],
            header_depth: int) -> tp.Generator[str, None, None]:
        '''
        Args:
            header_depth: number of columsn that should be treated as headers.
        '''
        yield '<tr>'
        for count, msg in enumerate(row):
            if count < header_depth:
                yield '<th>{}</th>'.format(msg)
            else:
                yield '<td>{}</td>'.format(msg)
        yield '</tr>'

    @staticmethod
    def markup_header(msg: str):
        return '<thead>{}</thead>'.format(msg)

    @staticmethod
    def markup_body(msg: str):
        return '<tbody>{}</tbody>'.format(msg)

    @staticmethod
    def markup_outermost(msg: str,
            identifier: tp.Optional[str] = None
            ) -> str:
        if identifier:
            id_str = 'id="{}" '.format(identifier)
        else:
            id_str = ''
        return '<table {id_str}border="1">{content}</table>'.format(
                id_str=id_str,
                content=msg)


class DisplayFormatHTMLDataTables(DisplayFormatHTMLTable):

    @staticmethod
    def markup_outermost(msg: str,
            identifier: tp.Optional[str] = 'SFTable'
            ) -> str:
        # embed the table HTML in the datatables template
        html_table = DisplayFormatHTMLTable.markup_outermost(msg,
                identifier=identifier)
        return display_html_datatables.TEMPLATE(identifier, html_table)


class DisplayFormatHTMLPre(DisplayFormat):

    CELL_WIDTH_NORMALIZE = True

    @staticmethod
    def markup_outermost(msg: str,
            identifier: tp.Optional[str] = None
            ) -> str:

        style = 'style="white-space: pre; font-family: monospace"'
        id_str = 'id="{}" '.format(identifier) if identifier else ''

        return '<div {id_str}{style}>{msg}</div>'.format(
                style=style,
                id_str=id_str,
                msg=msg)


_DISPLAY_FORMAT_MAP = {
        DisplayFormats.HTML_TABLE: DisplayFormatHTMLTable,
        DisplayFormats.HTML_DATATABLES: DisplayFormatHTMLDataTables,
        DisplayFormats.HTML_PRE: DisplayFormatHTMLPre,
        DisplayFormats.TERMINAL: DisplayFormatTerminal,
        }

#-------------------------------------------------------------------------------

def terminal_ansi(stream=sys.stdout) -> bool:
    '''
    Return True if the terminal is ANSI color compatible.
    '''
    environ = os.environ
    if 'ANSICON' in environ or 'PYCHARM_HOSTED' in environ:
        return True
    if 'TERM' in environ and environ['TERM'] == 'ANSI':
        return True
    if 'INSIDE_EMACS' in environ:
        return False

    if stream.closed:
        return False

    if hasattr(stream, 'isatty') and stream.isatty() and platform.system() != 'Windows':
        return True

    return False


#-------------------------------------------------------------------------------
class DisplayConfig:
    '''
    Storage container for all display settings.
    '''

    __slots__ = (
            'type_show',
            'type_color',

            'type_color_default',
            'type_color_int',
            'type_color_float',
            'type_color_complex',
            'type_color_bool',
            'type_color_object',
            'type_color_str',
            'type_color_datetime',
            'type_color_timedelta',
            'type_color_index',
            'type_color_series',
            'type_color_frame',

            'type_delimiter_left',
            'type_delimiter_right',

            'display_format',
            'display_columns',
            'display_rows',

            'cell_max_width',
            'cell_max_width_leftmost',
            'cell_align_left',
            )

    @classmethod
    def from_json(cls, json_str) -> 'DisplayConfig':
        args = json.loads(json_str.strip())
        # filter arguments by current slots
        args_valid = {}
        for k in cls.__slots__:
            if k in args:
                args_valid[k] = args[k]
        return cls(**args_valid)

    @classmethod
    def from_file(cls, fp):
        with open(fp) as f:
            return cls.from_json(f.read())

    @classmethod
    def from_default(cls, **kwargs) -> 'DisplayConfig':
        return cls(**kwargs)

    def __init__(self, *,
            type_show: bool = True,
            type_color: bool = True,

            type_color_default: ColorConstructor = 0x505050,
            type_color_int: ColorConstructor = 0x505050,
            type_color_float: ColorConstructor = 0x505050,
            type_color_complex: ColorConstructor = 0x505050,
            type_color_bool: ColorConstructor = 0x505050,
            type_color_object: ColorConstructor = 0x505050,
            type_color_str: ColorConstructor = 0x505050,

            type_color_datetime: ColorConstructor = 0x505050,
            type_color_timedelta: ColorConstructor = 0x505050,

            type_color_index: ColorConstructor = 0x777777,
            type_color_series: ColorConstructor = 0x777777,
            type_color_frame: ColorConstructor = 0x777777,

            type_delimiter_left: str = '<',
            type_delimiter_right: str = '>',
            display_format=DisplayFormats.TERMINAL,
            display_columns: tp.Optional[int] = 12,
            display_rows: tp.Optional[int] = 36,
            cell_max_width: int = 20,
            cell_max_width_leftmost: int = 36,

            cell_align_left: bool = True
            ) -> None:

        self.type_show = type_show
        self.type_color = type_color

        self.type_color_default = type_color_default
        self.type_color_int = type_color_int
        self.type_color_float = type_color_float
        self.type_color_complex = type_color_complex
        self.type_color_bool = type_color_bool
        self.type_color_object = type_color_object
        self.type_color_str = type_color_str

        self.type_color_datetime = type_color_datetime
        self.type_color_timedelta = type_color_timedelta
        self.type_color_index = type_color_index
        self.type_color_series = type_color_series
        self.type_color_frame = type_color_frame

        self.type_delimiter_left = type_delimiter_left
        self.type_delimiter_right = type_delimiter_right

        self.display_format = display_format

        self.display_columns = display_columns
        self.display_rows = display_rows

        self.cell_max_width = cell_max_width
        self.cell_max_width_leftmost = cell_max_width_leftmost

        self.cell_align_left = cell_align_left

    def write(self, fp):
        '''Write a JSON file.
        '''
        with open(fp, 'w') as f:
            f.write(self.to_json() + '\n')

    def __repr__(self):
        return '<' + self.__class__.__name__ + ' ' + ' '.join(
                '{k}={v}'.format(k=k, v=getattr(self, k))
                for k in self.__slots__) + '>'

    def to_dict(self, **kwargs) -> dict:
        # overrides with passed in kwargs if provided
        return {k: kwargs.get(k, getattr(self, k))
                for k in self.__slots__}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def to_transpose(self) -> 'DisplayConfig':
        kwargs = self.to_dict()
        kwargs['display_columns'], kwargs['display_rows'] = (
                kwargs['display_rows'], kwargs['display_columns'])
        return self.__class__(**kwargs)

    def to_display_config(self, **kwargs) -> 'DisplayConfig':
        return self.__class__(**self.to_dict(**kwargs))

#-------------------------------------------------------------------------------
class DisplayConfigs:
    '''
    Container of common default configs.
    '''

    DEFAULT = DisplayConfig()

    HTML_PRE = DisplayConfig(
            display_format=DisplayFormats.HTML_PRE,
            type_color=True
            )

    COLOR = DisplayConfig(
            display_format=DisplayFormats.TERMINAL,
            type_color=True,
            type_color_default='gray',
            type_color_int='yellowgreen',
            type_color_float='DeepSkyBlue',
            type_color_complex='LightSkyBlue',
            type_color_bool='darkorange',
            type_color_object='DarkSlateBlue',
            type_color_str='lightcoral',
            type_color_datetime='peru',
            type_color_timedelta='sienna',
            type_color_index='DarkSlateGray',
            type_color_series='dimgray',
            type_color_frame='lightslategray',
            )

    UNBOUND = DisplayConfig(
            display_columns=np.inf,
            display_rows=np.inf,
            cell_max_width=np.inf,
            cell_max_width_leftmost=np.inf,
            )
    UNBOUND_COLUMNS = DisplayConfig(
            display_columns=np.inf,
            cell_max_width=np.inf,
            cell_max_width_leftmost=np.inf,
            )
    UNBOUND_ROWS = DisplayConfig(
            display_rows=np.inf,
            cell_max_width=np.inf,
            cell_max_width_leftmost=np.inf,
            )

#-------------------------------------------------------------------------------

_module._display_active = DisplayConfig()

class DisplayActive:
    '''Utility interface for setting module-level display configuration.
    '''
    FILE_NAME = '.static_frame.conf'

    @staticmethod
    def set(dc: DisplayConfig):
        _module._display_active = dc

    @staticmethod
    def get(**kwargs):
        config = _module._display_active
        if not kwargs:
            return config
        args = config.to_dict()
        args.update(kwargs)
        return DisplayConfig(**args)

    @classmethod
    def update(cls, **kwargs):
        args = cls.get().to_dict()
        args.update(kwargs)
        cls.set(DisplayConfig(**args))

    @classmethod
    def _default_fp(cls):
        # TODO: improve cross platform support
        return os.path.join(os.path.expanduser('~'), cls.FILE_NAME)

    @classmethod
    def write(cls, fp=None):
        fp = fp or cls._default_fp()
        dc = cls.get()
        dc.write(fp)

    @classmethod
    def read(cls, fp=None):
        fp = fp or cls._default_fp()
        cls.set(DisplayConfig.from_file(fp))


#-------------------------------------------------------------------------------


class DisplayHeader:
    '''
    Wraper for passing in display header that have a name attribute.
    '''
    __slots__ = ('cls', 'name')

    def __init__(self,
            cls,
            name: tp.Optional[object] = None):
        '''
        Args:
            cls: the Class to be displayed.
            name: an optional name attribute stored on the instance.
        '''
        self.cls = cls
        self.name = name

    def __repr__(self) -> str:
        '''
        Provide string representation before additon of outer delimiters.
        '''
        if self.name:
            return '{}: {}'.format(self.cls.__name__, self.name)
        return self.cls.__name__


HeaderInitializer = tp.Optional[tp.Union[str, DisplayHeader]]

# store formatted string (with color markup), raw width (without markup), and the original, raw string
DisplayCell = namedtuple('DisplayCell', ('format_str', 'raw'))
FORMAT_EMPTY = '{}'

class Display:
    '''
    A Display is a string representation of a table, encoded as a list of lists, where list components are equal-width strings, keyed by row index
    '''
    __slots__ = (
        '_rows',
        '_config',
        '_outermost',
        '_index_depth',
        '_columns_depth',
        )

    CHAR_MARGIN = 1
    CELL_EMPTY = DisplayCell(FORMAT_EMPTY, '')
    ELLIPSIS = '...' # this string is appended to truncated entries
    CELL_ELLIPSIS = DisplayCell(FORMAT_EMPTY, ELLIPSIS)
    # ELLIPSIS_INDICES = (None,)
    DATA_MARGINS = 2 # columns / rows that seperate data
    ELLIPSIS_CENTER_SENTINEL = object()

    #---------------------------------------------------------------------------
    # utility methods

    @staticmethod
    def type_attributes(
            type_input: tp.Union[np.dtype, DisplayHeader],
            config: DisplayConfig
            ) -> tp.Tuple[str, int, DisplayTypeCategory]:
        '''
        Apply delimteres to type, for either numpy types or Python classes.
        '''
        if isinstance(type_input, np.dtype):
            type_str = str(type_input)
            type_ref = type_input
        elif inspect.isclass(type_input):
            type_str = type_input.__name__
            type_ref = type_input
        elif isinstance(type_input, DisplayHeader):
            type_str = repr(type_input)
            type_ref = type_input.cls
        else:
            raise NotImplementedError('no handling for this input', type_input)

        type_category = DisplayTypeCategoryFactory.to_category(type_ref)

        # if config.type_delimiter_left or config.type_delimiter_right:
        left = config.type_delimiter_left or ''
        right = config.type_delimiter_right or ''
        type_label = left + type_str + right

        return type_label, type_category

    @staticmethod
    def type_color_markup(
            type_label: str,
            type_category: DisplayTypeCategory,
            config: DisplayConfig
            ) -> str:
        '''
        Return a format string for applying color to a type based on type category and config.
        '''
        color = getattr(config, type_category.CONFIG_ATTR)
        if config.display_format in _DISPLAY_FORMAT_HTML:
            return HexColor.format_html(color, FORMAT_EMPTY)

        if config.display_format in _DISPLAY_FORMAT_TERMINAL:
            if terminal_ansi():
                return HexColor.format_terminal(color, FORMAT_EMPTY)
            # if not a compatible terminal, return label unaltered
            return FORMAT_EMPTY

        raise NotImplementedError('no handling for display format:',
                config.display_format)

    @classmethod
    def to_cell(cls,
            value: object, # dtype, HeaderInitializer, or a type
            config: DisplayConfig,
            is_dtype=False) -> DisplayCell:
        '''
        Given a raw value, retrun a DisplayCell, which is defined as a pair of the string representation and the character width without any formatting markup.
        '''
        if is_dtype or inspect.isclass(value) or isinstance(value, DisplayHeader):
            type_str_raw, type_category = cls.type_attributes(
                    value,
                    config=config)
            if config.type_color:
                format_str = cls.type_color_markup(
                        type_str_raw,
                        type_category,
                        config)
            else:
                format_str = FORMAT_EMPTY
            return DisplayCell(format_str, type_str_raw)

        # handling for all other values that are stringable
        msg = str(value)
        return DisplayCell(FORMAT_EMPTY, msg)

    #---------------------------------------------------------------------------
    # aalternate constructor

    @classmethod
    def from_values(cls,
            values: np.ndarray,
            header: HeaderInitializer,
            config: DisplayConfig = None,
            outermost: bool = False,
            index_depth: int = 0,
            columns_depth: int = 0
            ) -> 'Display':
        '''
        Given a 1 or 2D ndarray, return a Display instance. Generally 2D arrays are passed here only from TypeBlocks.
        '''
        # return a list of lists, where each inner list represents multiple columns
        config = config or DisplayActive.get()

        # create a list of lists, always starting with the header
        rows = []
        if header is not None:
            rows.append([cls.to_cell(header, config=config)])

        if isinstance(values, np.ndarray) and values.ndim == 2:
            # get rows from numpy string formatting
            np_rows = np.array_str(values).split('\n')
            last_idx = len(np_rows) - 1
            for idx, row in enumerate(np_rows):
                # trim brackets
                end_slice_len = 2 if idx == last_idx else 1
                row = row[2: len(row) - end_slice_len].strip()
                rows.append([cls.to_cell(row, config=config)])
        else:
            count_max = config.display_rows - cls.DATA_MARGINS
            # print('comparing values to count_max', len(values), count_max)
            if len(values) > config.display_rows:
                data_half_count = Display.truncate_half_count(count_max)
                value_gen = partial(_gen_skip_middle,
                        forward_iter=values.__iter__,
                        forward_count=data_half_count,
                        reverse_iter=partial(reversed, values),
                        reverse_count=data_half_count,
                        center_sentinel=cls.ELLIPSIS_CENTER_SENTINEL
                        )
            else:
                value_gen = values.__iter__

            for v in value_gen():
                if v is cls.ELLIPSIS_CENTER_SENTINEL: # center sentinel
                    rows.append([cls.CELL_ELLIPSIS])
                else:
                    rows.append([cls.to_cell(v, config=config)])

        # add the types to the last row
        if isinstance(values, np.ndarray) and config.type_show:
            rows.append([cls.to_cell(values.dtype, config=config, is_dtype=True)])

        return cls(rows,
                config=config,
                outermost=outermost,
                index_depth=index_depth,
                columns_depth=columns_depth)


    #---------------------------------------------------------------------------
    # core cell-to-rwo expansion routines

    @staticmethod
    def truncate_half_count(count_target: int) -> int:
        '''Given a target number of rows or columns, return the count of half as found in presentation where one column is used for the elipsis. The number returned will always be odd. For example, given a target of 5 we allocate 2 per half (plus 1 reserved for middle).
        '''
        if count_target <= 4:
            return 1 # practical floor for all values of 4 or less
        return (count_target - 1) // 2

    # @classmethod
    # def _truncate_indices(cls, count_target: int, indices):

    #     # if have 5 data cols, 7 total, and target was 6
    #     # half count of 2, 5 total out, with 1 meta, 1 data, elipsis, data, meta

    #     # if have 5 data cols, 7 total, and target was 7
    #     # half count of 3, 7 total out, with 1 meta, 2 data, elipsis, 2 data, 1 meta

    #     # if have 6 data cols, 8 total, and target was 6
    #     # half count of 2, 5 total out, with 1 meta, 1 data, elipsis, data, meta

    #     # if have 6 data cols, 8 total, and target was 7
    #     # half count of 3, 7 total out, with 1 meta, 2 data, elipsis, 2 data, 1 meta

    #     if count_target and len(indices) > count_target:
    #         half_count = cls.truncate_half_count(count_target)
    #         # replace with array from_iter? with known size?
    #         return tuple(chain(
    #                 indices[:half_count],
    #                 cls.ELLIPSIS_INDICES,
    #                 indices[-half_count:]))
    #     return indices


    @classmethod
    def _get_max_width_pad_width(cls, *,
            rows: tp.Iterable[tp.Iterable[DisplayCell]],
            col_idx_src: int,
            col_last_src: int,
            row_indices: tp.Iterable[int],
            config: DisplayConfig = None
            ) -> tp.Tuple[int, int]:
        '''
        Called once for each column to determine the maximum_width and pad_width for a particular column. All row data is passed to this function, and cell values are looked up directly with the `col_idx_src` argument.

        Args:
            rows: iterable of all rows, containing DisplayCell instances
            col_idx_src: the integer index for the currrent column
            col_last_src: the integer index for the last column
            row_indices: passed here so same range() can be reused.
        '''
        is_last = col_idx_src == col_last_src
        is_first = col_idx_src == 0

        if is_first:
            width_limit = config.cell_max_width_leftmost
        else:
            width_limit = config.cell_max_width

        max_width = 0
        for row_idx_src in row_indices:
            # get existing max width, up to the max
            if row_idx_src is not None:
                row = rows[row_idx_src]
                if col_idx_src >= len(row): # this row does not have this column
                    continue
                cell = row[col_idx_src]
                max_width = max(max_width, len(cell.raw))
            else:
                max_width = max(max_width, len(cls.ELLIPSIS))
            # if already exceeded max width, stop iterating
            if max_width >= width_limit:
                break

        # get most binding constraint
        max_width = min(max_width, width_limit)

        if ((config.cell_align_left is True and is_last) or
                (config.cell_align_left is False and is_first)):
            pad_width = max_width
        else:
            pad_width = max_width + cls.CHAR_MARGIN

        return max_width, pad_width

    @classmethod
    def _to_rows(cls,
            display: 'Display',
            config: DisplayConfig = None,
            index_depth: int = 0,
            columns_depth: int = 0,
            ) -> tp.Iterable[str]:
        '''
        Given a Display object, return an iterable of strings, where each string is the combination of all cells in that row, and appropriate padding (if necessary), as been applied. Based on configruation, align cells left or right with space and return one joined string per row.

        Returns:
            Returns an iterable of formatted strings, generally one per row.
        '''
        config = config or DisplayActive.get()

        # find max columns for all defined rows
        col_count_src = max(len(row) for row in display._rows)
        col_last_src = col_count_src - 1

        row_count_src = len(display._rows)
        row_indices = range(row_count_src)

        rows = [[] for _ in row_indices]

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
                        config=config
                        )

            for row_idx_src in row_indices:
                row = display._rows[row_idx_src]
                if col_idx_src >= len(row):
                    cell = cls.CELL_EMPTY
                else:
                    cell = row[col_idx_src]

                cell_format_str = cell.format_str
                cell_raw = cell.raw

                if dfc.CELL_WIDTH_NORMALIZE:

                    if len(cell.raw) > max_width:
                        # must truncate if cell width is greater than max width
                        width_truncate = max_width - len(cls.CELL_ELLIPSIS.raw)

                        cell_raw = cell_raw[:width_truncate] + cls.ELLIPSIS
                        if is_html:
                            cell_raw = html.escape(cell_raw)
                        cell_formatted = cell_format_str.format(cell_raw)
                        cell_fill_width = cls.CHAR_MARGIN # should only be margin left
                    else:
                        if is_html:
                            cell_raw = html.escape(cell_raw)
                        cell_formatted = cell_format_str.format(cell_raw)
                        cell_fill_width = pad_width - len(cell.raw) # this includes margin

                    # print(col_idx, row_idx, cell, max_width, pad_width, cell_fill_width)
                    if config.cell_align_left:
                        # must manually add space as color chars make ljust not work
                        msg = cell_formatted + ' ' * cell_fill_width
                    else:
                        msg = ' ' * cell_fill_width + cell_formatted

                else: # no width normalization
                    if is_html:
                        cell_raw = html.escape(cell_raw)
                    msg = cell_format_str.format(cell_raw)

                rows[row_idx_src].append(msg)

        post = []
        for row_idx, row in enumerate(rows):
            # make sure the entire row is marked as head if row index less than column depth
            added_depth = col_count_src if row_idx < columns_depth else 0
            # rstrip to remove extra white space on last column
            post.append(''.join(dfc.markup_row(row,
                    header_depth=index_depth + added_depth)).rstrip()
                    )
        return post

    #---------------------------------------------------------------------------
    def __init__(self,
            rows: tp.List[tp.List[DisplayCell]],
            config: DisplayConfig = None,
            outermost: bool = False,
            index_depth: int = 0,
            columns_depth: int = 0,
            ) -> None:
        '''Define rows as a list of lists, for each row; the strings may be of different size, but they are expected to be aligned vertically in final presentation.
        '''
        config = config or DisplayActive.get()

        self._rows = rows
        self._config = config
        self._outermost = outermost
        self._index_depth = index_depth
        self._columns_depth = columns_depth

    def __repr__(self):
        rows = self._to_rows(self,
                self._config,
                index_depth=self._index_depth,
                columns_depth=self._columns_depth
                )

        if self._outermost:
            dfc = _DISPLAY_FORMAT_MAP[self._config.display_format]
            header = []
            body = []
            for idx, row in enumerate(rows):
                if idx < self._columns_depth:
                    header.append(row)
                else:
                    body.append(row)
            header_str = dfc.markup_header(dfc.LINE_SEP.join(header))
            body_str = dfc.markup_body(dfc.LINE_SEP.join(body))
            return dfc.markup_outermost(header_str + dfc.LINE_SEP + body_str)

        return dfc.LINE_SEP.join(rows)

    def to_rows(self) -> tp.Iterable[str]:
        return self._to_rows(self, self._config)

    def __iter__(self):
        for row in self._rows:
            yield [cell.format_str.format(cell.raw) for cell in row]

    def __len__(self):
        return len(self._rows)

    #---------------------------------------------------------------------------
    # in place mutation

    def extend_display(self, display: 'Display') -> None:
        '''
        Mutate this display by extending to the right with the passed display.
        '''
        # NOTE: do not want to pass config or call format here as we call this for each column or block we add
        for row_idx, row in enumerate(display._rows):
            if row_idx == len(self._rows):
                self._rows.append([])
            self._rows[row_idx].extend(row)

    def extend_iterable(self,
            iterable: tp.Iterable[tp.Any],
            header: HeaderInitializer) -> None:
        '''
        Add an iterable of strings as a column to the display.
        '''
        row_idx_start = 0
        if header is not None:
            self._rows[0].append(self.to_cell(header, config=self._config))
            row_idx_start = 1

        # truncate iterable if necessary
        count_max = self._config.display_rows - self.DATA_MARGINS

        if len(iterable) > count_max:
            data_half_count = self.truncate_half_count(count_max)
            value_gen = partial(_gen_skip_middle,
                    forward_iter=iterable.__iter__,
                    forward_count=data_half_count,
                    reverse_iter=partial(reversed, iterable),
                    reverse_count=data_half_count,
                    center_sentinel=self.ELLIPSIS_CENTER_SENTINEL
                    )
        else:
            value_gen = iterable.__iter__

        # start at 1 as 0 is header
        idx = 0 # store in case value gen is empty
        for idx, value in enumerate(value_gen(), start=row_idx_start):
            if value is self.ELLIPSIS_CENTER_SENTINEL:
                self._rows[idx].append(self.CELL_ELLIPSIS)
            else:
                self._rows[idx].append(self.to_cell(value, config=self._config))

        if isinstance(iterable, np.ndarray) and self._config.type_show:
            self._rows[idx + 1].append(self.to_cell(iterable.dtype,
                    config=self._config,
                    is_dtype=True))

    def extend_ellipsis(self):
        '''Append an ellipsis over all rows.
        '''
        for row in self._rows:
            row.append(self.CELL_ELLIPSIS)

    def insert_displays(self,
            *displays: tp.Iterable['Display'],
            insert_index: int = 0):
        '''
        Insert rows on top of existing rows.
        args:
            Each arg in args is an instance of Display
            insert_index: the index at which to start insertion
        '''
        # each arg is a list, to be a new row
        # assume each row in display becomes a column
        new_rows = []
        for display in displays:
            new_rows.extend(display._rows)

        rows = []
        rows.extend(self._rows[:insert_index])
        rows.extend(new_rows)
        rows.extend(self._rows[insert_index:])
        self._rows = rows

    def append_row(self, row: tp.Iterable[DisplayCell]):
        '''
        Append a row at the bottom of existing rows.
        '''
        self._rows.append(row)

    #---------------------------------------------------------------------------
    # return a new display

    def flatten(self) -> 'Display':
        '''
        Return a Display from this Display that is a single row, formed the contents of all rows put in a single senquece, from the top down. Through this a single-column index Display can be made into a single row Display.
        '''
        row = []
        for part in self._rows:
            row.extend(part)
        rows = [row]
        return self.__class__(rows, config=self._config)

    def transform(self) -> 'Display':
        '''
        Return Display transformed (rotated) on its upper left; i.e., the first column becomes the first row.
        '''

        # assume first row gives us column count
        col_count = len(self._rows[0])
        rows = [[] for _ in range(col_count)]
        for r in self._rows:
            for idx, cell in enumerate(r):
                rows[idx].append(cell)
        return self.__class__(rows, config=self._config)

