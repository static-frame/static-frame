
import typing as tp
from itertools import chain

import numpy as np # type: ignore


from static_frame.core.util import DTYPE_INT_KIND
from static_frame.core.util import DTYPE_STR_KIND
from static_frame.core.util import DTYPE_NAN_KIND
# from static_frame.core.util import DTYPE_DATETIME_KIND
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import _DT64_S

from static_frame.core.frame import Frame
# from static_frame.core.exception import ErrorInitStore
# from static_frame.core.util import PathSpecifier

from static_frame.core.store import Store


if tp.TYPE_CHECKING:
    from xlsxwriter.worksheet import Worksheet # type: ignore # pylint: disable=W0611
    from xlsxwriter.workbook import Workbook # type: ignore  # pylint: disable=W0611
    from xlsxwriter.format import Format # type: ignore # pylint: disable=W0611



class StoreXLSX(Store):

    _EXT: str = '.xlsx'


    @staticmethod
    def _dtype_to_writer_attr(dtype: np.dtype) -> str:
        kind = dtype.kind
        if dtype == _DT64_S:
            return 'write_datetime'
        elif dtype == DTYPE_BOOL:
            return 'write_boolean'
        elif kind in DTYPE_STR_KIND:
            return 'write_string'
        elif kind in DTYPE_INT_KIND or kind in DTYPE_NAN_KIND:
            return 'write_number'
        return 'write'

    @staticmethod
    def _get_format_or_default(
            workbook: 'Workbook', # do not want module level import o
            format_specifier: tp.Optional[tp.Dict[str, tp.Any]]
            ) -> 'Format':
        if format_specifier:
            return workbook.add_format(format_specifier)
        else:
            f = workbook.add_format()
            f.set_bold()
            return f

    @classmethod
    def _frame_to_worksheet(cls,
            frame: Frame,
            ws: 'Worksheet',
            *,
            include_columns: bool,
            include_index: bool,
            format_columns: 'Format',
            format_index: 'Format',
            merge_hierarchical_labels: bool
            ) -> None:

        # iterating by columns avoids type coercion
        if include_index:
            index_values = frame._index.values
            index_depth = frame._index.depth

            if index_depth == 1:
                columns_iter = enumerate(chain(
                        (index_values,),
                        frame._blocks.axis_values(0)
                        ))
            else:
                # this approach is the same as IndexHierarchy.values_at_depth
                columns_iter = enumerate(chain(
                        (index_values[:, d] for d in range(index_depth)),
                        frame._blocks.axis_values(0)
                        ))
        else:
            index_depth = 0
            # avoid creating a Series per column by going to blocks
            columns_iter = enumerate(frame._blocks.axis_values(0))

        if include_columns:
            columns_depth = frame._columns.depth
            columns_values = frame._columns.values

        # TODO: need to determine if .name attr on index or columns should be populated in upper left corner "dead" zone.

        for col, values in columns_iter:

            # get writer for values; this may not be correct for column labels, but will include the index arrays
            # NOTE: not sure if this is suffient to handle problematic types
            writer = getattr(ws, cls._dtype_to_writer_attr(values.dtype))

            if include_columns:
                # col integers will include index depth
                if col >= index_depth:
                    if columns_depth == 1:
                        ws.write(0, col, columns_values[col - index_depth], format_columns)
                    else:
                        for i in range(columns_depth):
                            # here, row selection is column count, column selection is depth
                            ws.write(i,
                                    col,
                                    columns_values[col - index_depth, i],
                                    format_columns
                                    )
            # start enumeration of row after the columns
            for row, v in enumerate(values, columns_depth):
                writer(row,
                        col,
                        v,
                        format_index if col < index_depth else None)

        # post process to merge cells; need to get width of at depth
        if merge_hierarchical_labels and columns_depth > 1:
            for depth in range(columns_depth-1): # never most deep
                row = depth
                col = index_depth # start after index
                for label, width in frame._columns.label_widths_at_depth(depth):
                    ws.merge_range(row, col, row, col + width - 1, label, format_columns)
                    col += width

        if merge_hierarchical_labels and index_depth > 1:
            for depth in range(index_depth-1): # never most deep
                row = columns_depth
                col = depth
                for label, width in frame._index.label_widths_at_depth(depth):
                    ws.merge_range(row, col, row + width - 1, col, label, format_columns)
                    row += width


    def write(self,
            items: tp.Iterable[tp.Tuple[str, Frame]],
            *,
            include_index: bool = True,
            include_columns: bool = True,
            format_index: tp.Optional[tp.Dict[str, tp.Any]] = None,
            format_columns: tp.Optional[tp.Dict[str, tp.Any]] = None,
            merge_hierarchical_labels: bool = True
            ) -> None:
        '''
        Args:
            include_index: Boolean to determine if the ``index`` is included in output.
            include_columns: Boolean to determine if the ``columns`` is included in output.
            format_index: dictionary of XlsxWriter format specfications.
            format_columns: dictionary of XlsxWriter format specfications.
        '''
        # format_data: tp.Optional[tp.Dict[tp.Hashable, tp.Dict[str, tp.Any]]]
        # format_data: dictionary of dictionaries, keyed by column label, that contains dictionaries of XlsxWriter format specifications.

        import xlsxwriter # type: ignore

        wb = xlsxwriter.Workbook(self._fp)

        format_columns = self._get_format_or_default(wb, format_columns)
        format_index = self._get_format_or_default(wb, format_index)

        for label, frame in items:
            ws = wb.add_worksheet(label)
            self._frame_to_worksheet(frame,
                    ws,
                    format_columns=format_columns,
                    format_index=format_index,
                    include_index=include_index,
                    include_columns=include_columns,
                    merge_hierarchical_labels=merge_hierarchical_labels
                    )

        wb.close()


    def read(self,
            label: str,
            *,
            index_depth: int=1,
            columns_depth: int=1,
            ) -> Frame:

        import openpyxl # type: ignore
        # from openpyxl.utils.cell import coordinate_from_string
        # from openpyxl.utils.cell import column_index_from_string

        wb = openpyxl.load_workbook(filename=self._fp, read_only=True)
        ws = wb[label]

        if ws.max_column <= 1 or ws.max_row <= 1:
            # https://openpyxl.readthedocs.io/en/stable/optimized.html
            # says that some clients might not repare correct dimensions; not sure what conditions are best to show this
            ws.calculate_dimension()

        # can iter a column, but produces a tuple for each row; probably not efficient
        # [x for x in ws.iter_rows(min_row=0, max_row=ws.max_row, min_col=1, max_col=1)]
        coord_index = dict(
                min_row=columns_depth + 1,
                max_row=ws.max_row,
                min_col=1,
                max_col=index_depth
                )
        coord_columns = dict(
                min_row=1,
                max_row=columns_depth,
                min_col=index_depth + 1,
                max_col=ws.max_column
                )
        coord_data = dict(
                min_row=columns_depth + 1,
                max_row=ws.max_row,
                min_col=index_depth + 1,
                max_col=ws.max_column
                )

        data_rows = tuple(tuple(cell.value for cell in row) for row in ws.iter_rows(**coord_data))
        # import ipdb; ipdb.set_trace()

    def labels(self) -> tp.Iterator[str]:

        import openpyxl # type: ignore

        wb = openpyxl.load_workbook(filename=self._fp, read_only=True)
        return tuple(wb.get_sheet_names()) # comes as a list


# p q I I II II
# r s A B A  B
# 1 A 0 0 0  0
# 1 B 0 0 0  0
