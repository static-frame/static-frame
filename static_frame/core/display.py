import typing as tp
import sys
import json
import os


from itertools import chain
from functools import partial

import numpy as np


from static_frame.core.util import _gen_skip_middle

_module = sys.modules[__name__]

#-------------------------------------------------------------------------------
# display infrastructure

class DisplayConfig:

    __slots__ = (
        'type_show',
        'type_color',
        'type_delimiter',
        'display_columns',
        'display_rows',
        'cell_max_width',
        'cell_align_left'
        )

    @classmethod
    def from_json(cls, str) -> 'DisplayConfig':
        args = json.loads(str.strip())
        return cls(**args)

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
            type_show=True,
            type_color=False,
            type_delimiter='<>',
            display_columns: tp.Optional[int]=12,
            display_rows: tp.Optional[int]=36,
            cell_max_width: int=20,
            cell_align_left: bool=True
            ) -> None:
        self.type_show = type_show
        self.type_color = type_color
        self.type_delimiter = type_delimiter
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
    def type_delimiter(dtype: np.dtype, config: DisplayConfig):
        dtype_str = str(dtype) if isinstance(dtype, np.dtype) else dtype
        if config.type_delimiter:
            return config.type_delimiter[0] + dtype_str + config.type_delimiter[1]
        return dtype_str

    @staticmethod
    def type_color(dtype: np.dtype):
        dtype_str = str(dtype) if isinstance(dtype, np.dtype) else dtype
        return '\033[90m' + dtype_str + '\033[0m'

    @classmethod
    def to_cell(cls,
            value: tp.Any,
            config: DisplayConfig,
            is_dtype=False) -> tp.Tuple[str, int]:

        if is_dtype:
            type_str = cls.type_delimiter(value, config=config)
            type_length = len(type_str)
            if config.type_color:
                type_str = cls.type_color(type_str)
            return (type_str, type_length)

        msg = str(value)
        return (msg, len(msg))

    @classmethod
    def from_values(cls,
            values: np.ndarray,
            header: str,
            config: DisplayConfig=None) -> 'Display':
        '''
        Given a 1 or 2D ndarray, return a Display instance. Generally 2D arrays are passed here only from TypeBlocks.
        '''
        # return a list of lists, where each inner list represents multiple columns
        config = config or DisplayActive.get()

        msg = header.strip()

        # create a list of lists, always starting with the header
        rows = [[(msg, len(msg))]]

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
        Mutate this display by appending the passed display.
        '''
        # NOTE: do not want to pass config or call format here as we call this for each column or block we add
        for row_idx, row in enumerate(display._rows):
            self._rows[row_idx].extend(row)

    def append_iterable(self,
            iterable: tp.Iterable[tp.Any],
            header: str) -> None:
        '''
        Add an iterable of strings
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

        if isinstance(iterable, np.ndarray):
            if self._config.type_show:
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
