
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
                # avoid creating a per column by going to blocks
                columns_iter = enumerate(frame._blocks.axis_values(0))

            for col, values in columns_iter:
                for row, v in enumerate(values):
                    ws.write(row, col, v)

        wb.close()



