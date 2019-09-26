
import typing as tp
from itertools import chain

from static_frame.core.frame import Frame
# from static_frame.core.exception import ErrorInitStore
# from static_frame.core.util import PathSpecifier

from static_frame.core.store import Store

class StoreXLSX(Store):

    _EXT: str = '.xlsx'



    def write(self,
            items: tp.Iterable[tp.Tuple[str, Frame]],
            *,
            include_index: bool = True,
            include_columns: bool = True,

            ) -> None:

        import xlsxwriter

        wb = xlsxwriter.Workbook(self._fp)

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
                # index_depth_shift = index_depth + 1
            # else:
            #     columns_depth = 0

            for col, values in columns_iter:
                if include_columns:
                    # col integers will include index depth
                    if col >= index_depth:
                        if columns_depth == 1:
                            ws.write(0, col, columns_values[col - index_depth])
                        else:
                            for i in range(columns_depth):
                                # here, row selection is column count, column selection is depth
                                ws.write(i, col, columns_values[col - index_depth, i])
                for row, v in enumerate(values, columns_depth):
                    # start afterc columns, if shwon
                    ws.write(row, col, v)

        wb.close()



