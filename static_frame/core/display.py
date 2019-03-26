import typing as tp
import sys
import json
import os
import html
import inspect
import platform

from enum import Enum

from itertools import chain
from functools import partial

import numpy as np


from static_frame.core.util import _gen_skip_middle
from static_frame.core.display_color import HexColor
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
        from static_frame import Index
        return issubclass(t, Index)

class DisplayTypeSeries(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_series'

    @classmethod
    def in_category(cls, t: type) -> bool:
        from static_frame import Series
        return issubclass(t, Series)

class DisplayTypeFrame(DisplayTypeCategory):
    CONFIG_ATTR = 'type_color_frame'

    @classmethod
    def in_category(cls, t: type) -> bool:
        from static_frame import Frame
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

# NOTE: needs to be jsonable to use an enum
class DisplayFormat:

    HTML_PRE = 'html_pre'
    HTML_TABLE = 'html_table'
    TERMINAL = 'terminal'

_DISPLAY_FORMAT_HTML = {DisplayFormat.HTML_PRE, DisplayFormat.HTML_TABLE}
_DISPLAY_FORMAT_TERMINAL = {DisplayFormat.TERMINAL}


def terminal_ansi(stream=sys.stdout) -> bool:
    '''
    Return True if the terminal is ANSI color compatible.
    '''
    if 'ANSICON' in os.environ or 'PYCHARM_HOSTED' in os.environ:
        return True
    if 'TERM' in os.environ and os.environ['TERM'] == 'ANSI':
        return True
    if hasattr(stream, "isatty") and stream.isatty() and platform.system() != 'Windows':
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
            'cell_align_left'
            )

    @classmethod
    def from_json(cls, str) -> 'DisplayConfig':
        args = json.loads(str.strip())
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

    def write(self, fp):
        '''Write a JSON file.
        '''
        with open(fp, 'w') as f:
            f.write(self.to_json() + '\n')

    @classmethod
    def from_default(cls, **kwargs):
        return cls(**kwargs)

    def __init__(self,
            type_show: bool=True,
            type_color: bool=True,

            type_color_default: ColorConstructor='gray',
            type_color_int: ColorConstructor='MidnightBlue',
            type_color_float: ColorConstructor='MidnightBlue',
            type_color_complex: ColorConstructor='MidnightBlue',
            type_color_bool: ColorConstructor='SteelBlue',
            type_color_object: ColorConstructor='DarkSlateBlue',
            type_color_str: ColorConstructor='DarkOrchid',

            type_color_datetime: ColorConstructor='DarkSlateBlue',
            type_color_timedelta: ColorConstructor='DarkSlateBlue',

            type_color_index: ColorConstructor='DarkSlateGray',
            type_color_series: ColorConstructor='dimgray',
            type_color_frame: ColorConstructor='lightslategray',

            type_delimiter_left: str='<',
            type_delimiter_right: str='>',
            display_format=DisplayFormat.TERMINAL,
            display_columns: tp.Optional[int]=12,
            display_rows: tp.Optional[int]=36,
            cell_max_width: int=20,
            cell_align_left: bool=True
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

        self.display_format  = display_format

        self.display_columns = display_columns
        self.display_rows = display_rows

        self.cell_max_width = cell_max_width
        self.cell_align_left = cell_align_left

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
        args = self.to_dict()
        args['display_columns'], args['display_rows'] = (
                args['display_rows'], args['display_columns'])
        return self.__class__(**args)

#-------------------------------------------------------------------------------
class DisplayConfigs:
    '''
    Container of common default configs.
    '''

    DEFAULT = DisplayConfig()

    HTML_PRE = DisplayConfig(
            display_format=DisplayFormat.HTML_PRE,
            type_color=True
            )

    COLOR = DisplayConfig(
            display_format=DisplayFormat.TERMINAL,
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
            )
    UNBOUND_COLUMNS = DisplayConfig(
            display_columns=np.inf,
            cell_max_width=np.inf,
            )
    UNBOUND_ROWS = DisplayConfig(
            display_rows=np.inf,
            cell_max_width=np.inf,
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
    def get():
        return _module._display_active

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
class Display:
    '''
    A Display is a string representation of a table, encoded as a list of lists, where list components are equal-width strings, keyed by row index
    '''
    __slots__ = ('_rows', '_config')

    CHAR_MARGIN = 1
    CELL_EMPTY = ('', 0)
    ELLIPSIS = '...'
    CELL_ELLIPSIS = (ELLIPSIS, len(ELLIPSIS))
    ELLIPSIS_INDICES = (None,)
    DATA_MARGINS = 2 # columns / rows that seperate data
    ELLIPSIS_CENTER_SENTINEL = object()

    @staticmethod
    def type_attributes(
            dtype: np.dtype,
            config: DisplayConfig
            ) -> tp.Tuple[str, int, DisplayTypeCategory]:
        '''
        Apply delimteres to type, for either numpy types or Python classes.
        '''
        if isinstance(dtype, np.dtype):
            type_str = str(dtype)
        elif inspect.isclass(dtype):
            type_str = dtype.__name__
        else:
            NotImplementedError('no handling for this type', dtype)

        type_category = DisplayTypeCategoryFactory.to_category(dtype)

        # if config.type_delimiter_left or config.type_delimiter_right:
        left = config.type_delimiter_left or ''
        right = config.type_delimiter_right or ''
        type_label = left + type_str + right
        # find length after stipping escape codes
        type_length = len(type_label)

        if config.display_format in _DISPLAY_FORMAT_HTML:
            type_label = html.escape(type_label)

        return type_label, type_length, type_category

    @staticmethod
    def type_color_markup(
            type_label: str,
            type_category: DisplayTypeCategory,
            config: DisplayConfig
            ):
        '''
        Return type label with markup for color.
        '''
        color = getattr(config, type_category.CONFIG_ATTR)
        if config.display_format in _DISPLAY_FORMAT_HTML:
            return HexColor.format_html(color, type_label)
        elif config.display_format in _DISPLAY_FORMAT_TERMINAL:
            if terminal_ansi():
                return HexColor.format_terminal(color, type_label)
            # if not a compatible terminal, return label unaltered
            return type_label
        raise NotImplementedError('no handling for display format:',
                config.display_format)

    @classmethod
    def to_cell(cls,
            value: tp.Any,
            config: DisplayConfig,
            is_dtype=False) -> tp.Tuple[str, int]:

        if is_dtype or inspect.isclass(value):
            type_str, type_length, type_category = cls.type_attributes(
                    value,
                    config=config)
            if config.type_color:
                type_str = cls.type_color_markup(
                        type_str,
                        type_category,
                        config)
            return (type_str, type_length)

        msg = str(value)
        return (msg, len(msg))

    @classmethod
    def from_values(cls,
            values: np.ndarray,
            header: tp.Union[str, type],
            config: DisplayConfig=None) -> 'Display':
        '''
        Given a 1 or 2D ndarray, return a Display instance. Generally 2D arrays are passed here only from TypeBlocks.
        '''
        # return a list of lists, where each inner list represents multiple columns
        config = config or DisplayActive.get()

        # create a list of lists, always starting with the header
        rows = [[cls.to_cell(header, config=config)]]

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
        else: # this is an object
            rows.append([cls.CELL_EMPTY])

        return cls(rows, config=config)


    @staticmethod
    def truncate_half_count(count_target: int) -> int:
        '''Given a target number of rows or columns, return the count of half as found in presentation where one column is used for the elipsis. The number returned will always be odd. For example, given a target of 5 we allocate 2 per half (plus 1 reserved for middle).
        '''
        if count_target <= 4:
            return 1 # practical floor for all values of 4 or less
        return (count_target - 1) // 2

    @classmethod
    def _truncate_indices(cls, count_target: int, indices):

        # if have 5 data cols, 7 total, and target was 6
        # half count of 2, 5 total out, with 1 meta, 1 data, elipsis, data, meta

        # if have 5 data cols, 7 total, and target was 7
        # half count of 3, 7 total out, with 1 meta, 2 data, elipsis, 2 data, 1 meta

        # if have 6 data cols, 8 total, and target was 6
        # half count of 2, 5 total out, with 1 meta, 1 data, elipsis, data, meta

        # if have 6 data cols, 8 total, and target was 7
        # half count of 3, 7 total out, with 1 meta, 2 data, elipsis, 2 data, 1 meta

        if count_target and len(indices) > count_target:
            half_count = cls.truncate_half_count(count_target)
            # replace with array from_iter? with known size?
            return tuple(chain(
                    indices[:half_count],
                    cls.ELLIPSIS_INDICES,
                    indices[-half_count:]))
        return indices

    @classmethod
    def _to_rows(cls,
            display: 'Display',
            config: DisplayConfig=None) -> tp.Iterable[str]:
        '''
        Given already defined rows, align them to left or right and return one joined string per row.
        '''
        config = config or DisplayActive.get()

        # find max columns for all defined rows
        col_count_src = max(len(row) for row in display._rows)
        col_last_src = col_count_src - 1

        row_count_src = len(display._rows)
        row_indices = tuple(range(row_count_src))

        rows = [[] for _ in row_indices]

        for col_idx_src in range(col_count_src):
            # for each column, get the max width
            max_width = 0
            for row_idx_src in row_indices:
                # get existing max width, up to the max
                if row_idx_src is not None:
                    row = display._rows[row_idx_src]
                    if col_idx_src >= len(row): # this row does not have this column
                        continue
                    cell = row[col_idx_src]

                    max_width = max(max_width, cell[1])
                else:
                    max_width = max(max_width, len(cls.ELLIPSIS))
                # if we have already exceeded max width, can stop iterating
                if max_width >= config.cell_max_width:
                    break
            max_width = min(max_width, config.cell_max_width)

            if ((config.cell_align_left is True and col_idx_src == col_last_src) or
                    (config.cell_align_left is False and col_idx_src == 0)):
                pad_width = max_width
            else:
                pad_width = max_width + cls.CHAR_MARGIN

            for row_idx_src in row_indices:
                row = display._rows[row_idx_src]
                if col_idx_src >= len(row):
                    cell = cls.CELL_EMPTY
                else:
                    cell = row[col_idx_src]
                # msg may have been ljusted before, so we strip again here
                # cannot use ljust here, as the cell might have more characters for coloring
                if cell[1] > max_width:
                    cell_content = cell[0].strip()[:max_width - 3] + cls.ELLIPSIS
                    cell_fill_width = cls.CHAR_MARGIN # should only be margin left
                else:
                    cell_content = cell[0].strip()
                    cell_fill_width = pad_width - cell[1] # this includes margin

                # print(col_idx, row_idx, cell, max_width, pad_width, cell_fill_width)
                if config.cell_align_left:
                    # must manually add space as color chars make ljust not
                    msg = cell_content + ' ' * cell_fill_width
                else:
                    msg = ' ' * cell_fill_width + cell_content

                rows[row_idx_src].append(msg)

        # rstrip to remove extra white space on last column
        return [''.join(row).rstrip() for row in rows]


    def __init__(self,
            rows: tp.List[tp.List[tp.Tuple[str, int]]],
            config: DisplayConfig=None) -> None:
        '''Define rows as a list of lists, for each row; the strings may be of different size, but they are expected to be aligned vertically in final presentation.
        '''
        config = config or DisplayActive.get()
        self._rows = rows
        self._config = config


    def __repr__(self):
        return '\n'.join(self._to_rows(self, self._config))

    def to_rows(self) -> tp.Iterable[str]:
        return self._to_rows(self, self._config)

    def __iter__(self):
        for row in self._rows:
            yield [cell[0] for cell in row]

    def __len__(self):
        return len(self._rows)

    #---------------------------------------------------------------------------
    # in place mutation

    def append_display(self, display: 'Display') -> None:
        '''
        Mutate this display by extending to the right with the passed display.
        '''
        # NOTE: do not want to pass config or call format here as we call this for each column or block we add
        for row_idx, row in enumerate(display._rows):
            self._rows[row_idx].extend(row)

    def append_iterable(self,
            iterable: tp.Iterable[tp.Any],
            header: str) -> None:
        '''
        Add an iterable of strings as a column to the display.
        '''
        self._rows[0].append(self.to_cell(header, config=self._config))

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
        for idx, value in enumerate(value_gen(), start=1):
            if value is self.ELLIPSIS_CENTER_SENTINEL:
                self._rows[idx].append(self.CELL_ELLIPSIS)
            else:
                self._rows[idx].append(self.to_cell(value, config=self._config))

        if isinstance(iterable, np.ndarray) and self._config.type_show:
            self._rows[idx + 1].append(self.to_cell(iterable.dtype,
                    config=self._config,
                    is_dtype=True))

    def append_ellipsis(self):
        '''Append an ellipsis over all rows.
        '''
        for row in self._rows:
            row.append(self.CELL_ELLIPSIS)

    def insert_rows(self, *displays: tp.Iterable['Display']):
        '''
        Insert rows on top of existing rows.
        args:
            Each arg in args is an instance of Display
        '''
        # each arg is a list, to be a new row
        # assume each row in display becomes a column
        new_rows = []
        for display in displays:
            new_rows.extend(display._rows)
        # slow for now: make rows a dict to make faster
        new_rows.extend(self._rows)
        self._rows = new_rows

    #---------------------------------------------------------------------------
    # return a new display

    def flatten(self) -> 'Display':
        row = []
        for part in self._rows:
            row.extend(part)
        rows = [row]
        return self.__class__(rows, config=self._config)
