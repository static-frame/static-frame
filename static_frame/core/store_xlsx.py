
import typing as tp
from itertools import chain

import numpy as np


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


    def write(self,
            items: tp.Iterable[tp.Tuple[str, Frame]],
            *,
            include_index: bool = True,
            include_columns: bool = True,
            format_index: tp.Optional[tp.Dict[str, tp.Any]] = None,
            format_columns: tp.Optional[tp.Dict[str, tp.Any]] = None,
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

        import xlsxwriter

        wb = xlsxwriter.Workbook(self._fp)

        if format_index:
            format_index = wb.add_format(format_index)
        else:
            format_index = wb.add_format()
            format_index.set_bold()

        if format_columns:
            format_columns = wb.add_format(format_columns)
        else:
            format_columns = wb.add_format()
            format_columns.set_bold()

        for label, frame in items:
            ws = wb.add_worksheet(label)

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

                # get writer for values; this may no be correct for column labels
                writer = getattr(ws, self._dtype_to_writer_attr(values.dtype))

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
                    if col < index_depth:
                        f = format_index
                    else:
                        f = None
                    writer(row,
                            col,
                            v,
                            f)

            # post process to merge cells; need to get width of at depth


        wb.close()


# p q I I II II
# r s A B A  B
# 1 A 0 0 0  0
# 1 B 0 0 0  0
