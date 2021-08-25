from enum import Enum
import typing as tp
import re
import json

import numpy as np

from static_frame.core.interface_meta import InterfaceMeta
from static_frame.core import display_html_datatables
from static_frame.core.style_config import StyleConfig


ColorConstructor = tp.Union[int, str]

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
    RST = 'rst'
    MARKDOWN = 'markdown'
    LATEX = 'latex'

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
            index_depth: int, #pylint: disable=W0613
            iloc_row: int,
            style_config: tp.Optional[StyleConfig] = None,
            ) -> tp.Iterator[str]:
        '''
        Called with each row, post cell-width normalization (if enabled).

        Args:
            index_depth: number of elements to be marked as header; value can be np.inf to mark all values as header.
            iloc_row: iloc position of rows in the table; if negative, this is a columns row.
        '''
        for msg in row:
            yield msg

    @staticmethod
    def markup_header(msg: str) -> str:
        '''
        Called with all `LINE_SEP` joined header lines..
        '''
        return msg

    @staticmethod
    def markup_body(msg: str) -> str:
        '''
        Called with all `LINE_SEP` joined body lines.
        '''
        return msg

    @staticmethod
    def markup_outermost(msg: str,
            identifier: tp.Optional[str] = None, #pylint: disable=W0613
            style_config: tp.Optional[StyleConfig] = None,
            ) -> str:
        '''
        Called with combination of header and body joined with `LINE_SEP`.
        '''
        return msg


class DisplayFormatTerminal(DisplayFormat):
    pass

class DisplayFormatHTMLTable(DisplayFormat):

    CELL_WIDTH_NORMALIZE = False
    LINE_SEP = ''

    @staticmethod
    def markup_row(
            row: tp.Iterable[str],
            index_depth: int, # this can be renamed index_depth
            iloc_row: int,
            style_config: tp.Optional[StyleConfig] = None,
            ) -> tp.Iterator[str]:

        yield '<tr>'
        style = ''
        for count, msg in enumerate(row):
            # header depth here refers potentially to a header that is the index
            iloc_column = count - index_depth

            if iloc_row < 0 and iloc_column < 0:
                is_header = True
                if style_config:
                    coordinates = (iloc_row, iloc_column)
                    msg, style = style_config.apex(msg, coordinates)
            elif iloc_column < 0 and iloc_row >= 0:
                is_header = True
                if style_config:
                    msg, style = style_config.index(msg)
            elif iloc_column >= 0 and iloc_row < 0:
                is_header = True
                if style_config:
                    msg, style = style_config.columns(msg)
            else:
                is_header = False
                if style_config:
                    coordinates = (iloc_row, iloc_column)
                    msg, style = style_config.values(msg, coordinates)

            if is_header:
                yield f'<th{style}>{msg}</th>'
            else:
                yield f'<td{style}>{msg}</td>'
        yield '</tr>'

    @staticmethod
    def markup_header(msg: str) -> str:
        return f'<thead>{msg}</thead>'

    @staticmethod
    def markup_body(msg: str) -> str:
        return f'<tbody>{msg}</tbody>'

    @staticmethod
    def markup_outermost(msg: str,
            identifier: tp.Optional[str] = None,
            style_config: tp.Optional[StyleConfig] = None,
            ) -> str:
        style = style_config.frame() if style_config else ''
        id_str = f' id="{identifier}"' if identifier else ''
        return f'<table{id_str}{style}>{msg}</table>'


class DisplayFormatHTMLDataTables(DisplayFormatHTMLTable):

    @staticmethod
    def markup_outermost(msg: str,
            identifier: tp.Optional[str] = 'SFTable',
            style_config: tp.Optional[StyleConfig] = None,
            ) -> str:
        # embed the table HTML in the datatables template
        html_table = DisplayFormatHTMLTable.markup_outermost(msg,
                identifier=identifier)
        return display_html_datatables.TEMPLATE(identifier, html_table)


class DisplayFormatHTMLPre(DisplayFormat):

    CELL_WIDTH_NORMALIZE = True

    @staticmethod
    def markup_outermost(msg: str,
            identifier: tp.Optional[str] = None,
            style_config: tp.Optional[StyleConfig] = None,
            ) -> str:

        style = 'style="white-space: pre; font-family: monospace"'
        id_str = 'id="{}" '.format(identifier) if identifier else ''
        return f'<div {id_str}{style}>{msg}</div>'


class DisplayFormatRST(DisplayFormat):

    CELL_WIDTH_NORMALIZE = True
    LINE_SEP = '\n'
    _RE_NOT_PIPE = re.compile(r'[^|]')

    @staticmethod
    def markup_row(
            row: tp.Iterable[str],
            index_depth: int,
            iloc_row: int,
            style_config: tp.Optional[StyleConfig] = None,
            ) -> tp.Iterator[str]:

        yield f"|{'|'.join(row)}|"

    @classmethod
    def markup_header(cls, msg: str) -> str:
        # header does boundary lines above, and one additional = line below
        def lines() -> tp.Iterator[str]:
            for line in msg.split(cls.LINE_SEP):
                yield cls._RE_NOT_PIPE.sub('-', line).replace('|', '+')
                yield line
            yield cls._RE_NOT_PIPE.sub('=', line).replace('|', '+')

        return cls.LINE_SEP.join(lines())

    @classmethod
    def markup_body(cls, msg: str) -> str:
        # body lines add boundary lines below
        def lines() -> tp.Iterator[str]:
            for line in msg.split(cls.LINE_SEP):
                yield line
                yield cls._RE_NOT_PIPE.sub('-', line).replace('|', '+')

        return cls.LINE_SEP.join(lines())


class DisplayFormatMarkdown(DisplayFormat):

    CELL_WIDTH_NORMALIZE = True
    LINE_SEP = '\n'
    _RE_NOT_PIPE = re.compile(r'[^|]')

    @staticmethod
    def markup_row(
            row: tp.Iterable[str],
            index_depth: int,
            iloc_row: int,
            style_config: tp.Optional[StyleConfig] = None,
            ) -> tp.Iterator[str]:
        yield f"|{'|'.join(row)}|"

    @classmethod
    def markup_header(cls, msg: str) -> str:
        # header does boundary lines above, and one additional = line below
        def lines() -> tp.Iterator[str]:
            for line in msg.split(cls.LINE_SEP):
                yield line
            yield cls._RE_NOT_PIPE.sub('-', line)

        return cls.LINE_SEP.join(lines())



class DisplayFormatLaTeX(DisplayFormat):

    CELL_WIDTH_NORMALIZE = True
    LINE_SEP = '\n'
    _CELL_SEP = ' & '

    @classmethod
    def markup_row(cls,
            row: tp.Iterable[str],
            index_depth: int,
            iloc_row: int,
            style_config: tp.Optional[StyleConfig] = None,
            ) -> tp.Iterator[str]:
        yield f'{cls._CELL_SEP.join(row)} \\\\' # need 2 backslashes

    @classmethod
    def markup_header(cls, msg: str) -> str:
        # assume that the header is small and the wasteful split is acceptable
        def lines() -> tp.Iterator[str]:
            lines_header = msg.split('\n')
            col_count = lines_header[0].count(cls._CELL_SEP) + 1
            col_spec = ' '.join('c' * col_count)
            yield f'\\begin{{tabular}}{{{col_spec}}}'
            yield r'\hline\hline'
            yield from lines_header
            yield r'\hline'

        return cls.LINE_SEP.join(lines())

    @classmethod
    def markup_body(cls,
            msg: str) -> str:
        return msg + cls.LINE_SEP + r'\hline\end{tabular}'

    @classmethod
    def markup_outermost(cls,
            msg: str,
            identifier: tp.Optional[str] = None,
            style_config: tp.Optional[StyleConfig] = None,
            ) -> str:

        def lines() -> tp.Iterator[str]:
            yield r'\begin{table}[ht]'
            # if caption:
            #     yield f'\caption{{caption}}'
            yield r'\centering'
            yield msg
            if identifier:
                yield f'\\label{{table:{identifier}}}'
            yield r'\end{table}'

        return cls.LINE_SEP.join(lines())

_DISPLAY_FORMAT_MAP: tp.Dict[str, tp.Type[DisplayFormat]] = {
        DisplayFormats.HTML_TABLE: DisplayFormatHTMLTable,
        DisplayFormats.HTML_DATATABLES: DisplayFormatHTMLDataTables,
        DisplayFormats.HTML_PRE: DisplayFormatHTMLPre,
        DisplayFormats.TERMINAL: DisplayFormatTerminal,
        DisplayFormats.RST: DisplayFormatRST,
        DisplayFormats.MARKDOWN: DisplayFormatMarkdown,
        DisplayFormats.LATEX: DisplayFormatLaTeX,
        }


#-------------------------------------------------------------------------------
class DisplayConfig(metaclass=InterfaceMeta):
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

            'value_format_float_positional',
            'value_format_float_scientific',
            'value_format_complex_positional',
            'value_format_complex_scientific',

            'display_format',
            'display_columns',
            'display_rows',

            'include_columns',
            'include_index',

            'cell_max_width',
            'cell_max_width_leftmost',
            'cell_align_left',
            )

    @classmethod
    def from_json(cls, json_str: str) -> 'DisplayConfig':
        args = json.loads(json_str.strip())
        # filter arguments by current slots
        args_valid = {}
        for k in cls.__slots__:
            if k in args:
                args_valid[k] = args[k]
        return cls(**args_valid)

    @classmethod
    def from_file(cls, fp: str) -> 'DisplayConfig':
        with open(fp) as f:
            return cls.from_json(f.read())

    @classmethod
    def from_default(cls, **kwargs: tp.Any) -> 'DisplayConfig':
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

            # for positional, default to {} to avoid a fixed floating point size
            value_format_float_positional: str = '{}',
            value_format_float_scientific: str = '{:.8e}',
            value_format_complex_positional: str = '{}',
            value_format_complex_scientific: str = '{:.2e}',

            display_format: str = DisplayFormats.TERMINAL,
            display_columns: int = 12,
            display_rows: int = 36,

            include_columns: bool = True,
            include_index: bool = True,

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

        self.value_format_float_positional = value_format_float_positional
        self.value_format_float_scientific = value_format_float_scientific
        self.value_format_complex_positional = value_format_complex_positional
        self.value_format_complex_scientific = value_format_complex_scientific

        self.display_format = display_format
        self.display_columns = display_columns
        self.display_rows = display_rows

        self.include_columns = include_columns
        self.include_index = include_index

        self.cell_max_width = cell_max_width
        self.cell_max_width_leftmost = cell_max_width_leftmost
        self.cell_align_left = cell_align_left

        #-----------------------------------------------------------------------
        # handle any inter-dependent configurations

        if not self.include_columns or not self.include_index:
            if self.type_show:
                raise RuntimeError('cannot show types if not including columns or index.')


    def write(self, fp: str) -> None:
        '''Write a JSON file.
        '''
        with open(fp, 'w') as f:
            f.write(self.to_json() + '\n')

    def __repr__(self) -> str:
        return '<' + self.__class__.__name__ + ' ' + ' '.join(
                '{k}={v}'.format(k=k, v=getattr(self, k))
                for k in self.__slots__) + '>'

    def to_dict(self, **kwargs: object) -> tp.Dict[str, tp.Any]:
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

    def to_display_config(self, **kwargs: object) -> 'DisplayConfig':
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
