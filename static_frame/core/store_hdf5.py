
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

    _EXT: tp.FrozenSet[str] =  frozenset(('.h5', '.hdf5'))


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
                # should all tables be under a common group?
                group = file.create_group('/', label)

                field_names, dtypes = self._get_field_names_and_dtypes(
                        frame=frame,
                        include_index=include_index,
                        include_columns=include_columns
                        )

                # note: need to handle hiararchical columns
                description = {k: tables.Col.from_dtype(v) for k, v in zip(field_names, dtypes)}
                table = file.create_table(group, label, description)

                values = self._get_row_iterator(frame=frame, include_index=include_index)

                for row_src in values():
                    row_dst = table.row
                    for k, v in zip(field_names, row_src):
                        row_dst[k] = v # pylint: disable=E1137
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