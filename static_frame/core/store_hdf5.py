
import typing as tp

# from itertools import chain

# import numpy as np # type: ignore

from static_frame.core.frame import Frame
from static_frame.core.store import Store

# from static_frame.core.index_hierarchy import IndexHierarchy

# from static_frame.core.store_filter import StoreFilter
# from static_frame.core.store_filter import STORE_FILTER_DEFAULT

from static_frame.core.doc_str import doc_inject

from static_frame.core.util import DtypesSpecifier
# from static_frame.core.util import DTYPE_INT_KIND
# from static_frame.core.util import DTYPE_STR_KIND
# from static_frame.core.util import DTYPE_NAN_KIND
# from static_frame.core.util import DTYPE_DATETIME_KIND
# from static_frame.core.util import DTYPE_BOOL
# from static_frame.core.util import BOOL_TYPES
# from static_frame.core.util import NUMERIC_TYPES




class StoreHDF5(Store):

    _EXT: str = '.hdf5'



    # @classmethod
    # def _frame_to_table(cls,
    #         *,
    #         frame: Frame,
    #         label: tp.Optional[str], # can be None
    #         cursor: sqlite3.Cursor,
    #         include_columns: bool,
    #         include_index: bool,
    #         # store_filter: tp.Optional[StoreFilter]
    #         ) -> None:

    #     # for interface compatibility with StoreXLSX, where label can be None
    #     if label is None:
    #         label = 'None'

    #     index = frame.index
    #     columns = frame.columns

    #     if not include_index:
    #         dtypes = frame._blocks.dtypes
    #         if include_columns:
    #             field_names = columns.values
    #         else: # name fields with integers?
    #             field_names = range(frame._blocks.shape[1])
    #         # no primary key possible
    #         create_primary_key = ''
    #     else:
    #         if index.depth == 1:
    #             dtypes = [index.dtype]
    #             # cannot use index as it is a keyword in sqlite
    #             field_names = [index.name if index.name else 'index0']
    #         else:
    #             assert isinstance(index, IndexHierarchy) # for typing
    #             dtypes = index.dtypes.values.tolist()
    #             # TODO: use index .name attribute if available
    #             field_names = [f'index{d}' for d in range(index.depth)]

    #         # add fram dtypes tp those from index
    #         dtypes.extend(frame._blocks.dtypes)

    #         # add index names in front of column names
    #         if include_columns:
    #             field_names.extend(columns)
    #         else: # name fields with integers?
    #             field_names.extend(range(frame._blocks.shape[1]))


    #     field_name_to_field_type = (
    #             (field, cls._dtype_to_affinity_type(dtype))
    #             for field, dtype in zip(field_names, dtypes)
    #             )


    #     if include_index:
    #         index_values = index.values
    #         def values() -> tp.Iterator[tp.Sequence[tp.Any]]:
    #             for idx, row in enumerate(frame.iter_array(1)):
    #                 if index.depth > 1:
    #                     yield tuple(chain(index_values[idx], row))
    #                 else:
    #                     row_final = [index_values[idx]]
    #                     row_final.extend(row)
    #                     yield row_final
    #     else:
    #         values = partial(frame.iter_array, 1) #type: ignore

    #     # numpy types go in as blobs if they are not individuall converted tp python types
    #     cursor.executemany(insert, values())


    def write(self,
            items: tp.Iterable[tp.Tuple[tp.Optional[str], Frame]],
            *,
            include_index: bool = True,
            include_columns: bool = True,
            # store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ) -> None:

        import tables # type: ignore

        with tables.open_file(self._fp, mode='w') as file:
            for label, frame in items:
                group = file.create_group('/', label)

                # note: need to handle hiararchical columns
                description = {k: tables.Col.from_dtype(v) for k, v in frame.dtypes.items()}
                table = file.create_table(group, label, description)

                for row_src in frame.iter_array(1):
                    row_dst = table.row
                    for k, v in zip(frame._columns, row_src):
                        row_dst[k] = v
                    row_dst.append()

                table.flush()


    @doc_inject(selector='constructor_frame')
    def read(self,
            label: tp.Optional[str] = None,
            *,
            index_depth: int=1,
            columns_depth: int=1,
            dtypes: DtypesSpecifier = None,
            ) -> Frame:
        '''
        Args:
            {dtypes}
        '''


    def labels(self) -> tp.Iterator[str]:
        pass