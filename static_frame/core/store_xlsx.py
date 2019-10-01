
import typing as tp
from itertools import chain

import numpy as np # type: ignore

from static_frame.core.util import DtypesSpecifier

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
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.index import Index
from static_frame.core.index_base import IndexBase

from static_frame.core.frame import Frame

from static_frame.core.doc_str import doc_inject

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


    @staticmethod
    def _load_workbook(fp: str) -> 'Workbook':
        import openpyxl # type: ignore
        return openpyxl.load_workbook(
                filename=fp,
                read_only=True,
                data_only=True
                )

    @doc_inject(selector='constructor_frame')
    def read(self,
            label: str,
            *,
            index_depth: int=1,
            columns_depth: int=1,
            dtypes: DtypesSpecifier = None,
            ) -> Frame:
        '''
        Args:
            {dtypes}
        '''
        wb = self._load_workbook(self._fp)
        ws = wb[label]

        if ws.max_column <= 1 or ws.max_row <= 1:
            # https://openpyxl.readthedocs.io/en/stable/optimized.html
            # says that some clients might not repare correct dimensions; not sure what conditions are best to show this
            ws.calculate_dimension()

        max_column = ws.max_column
        max_row = ws.max_row
        name = ws.title

        index_values: tp.List[tp.Any] = []
        columns_values: tp.List[tp.Any] = []

        # print()
        # for row in ws.iter_rows():
        #     print(tuple(str(c.value).ljust(10) for c in row))

        data = []

        for row_count, row in enumerate(ws.iter_rows()): # cannot use values_only on 2.5.4
            row = tuple(c.value for c in row)
            if row_count <= columns_depth - 1:
                if columns_depth == 1:
                    columns_values.extend(row[index_depth:])
                elif columns_depth > 1:
                    # NOTE: this orientation will need to be rotated
                    columns_values.append(row[index_depth:])
                continue
            else:
                if index_depth == 0:
                    data.append(row)
                elif index_depth == 1:
                    index_values.append(row[0])
                    data.append(row[1:])
                else:
                    index_values.append(row[:index_depth])
                    data.append(row[index_depth:])

        wb.close()

        index: tp.Optional[IndexBase] = None
        if index_depth == 1:
            index = Index(index_values)
        elif index_depth > 1:
            index = IndexHierarchy.from_labels(
                    index_values,
                    continuation_token=None
                    )

        columns: tp.Optional[IndexBase] = None
        if columns_depth == 1:
            columns = Index(columns_values)
        elif columns_depth > 1:
            columns = IndexHierarchy.from_labels(
                    zip(*columns_values),
                    continuation_token=None
                    )

        return tp.cast(Frame, Frame.from_records(data,
                index=index,
                columns=columns,
                dtypes=dtypes,
                own_index=True,
                own_columns=True,
                name=name
                ))

    def labels(self) -> tp.Iterator[str]:

        import openpyxl

        wb = self._load_workbook(self._fp)
        labels = tuple(wb.sheetnames)
        wb.close()
        yield from labels


# p q I I II II
# r s A B A  B
# 1 A 0 0 0  0
# 1 B 0 0 0  0
