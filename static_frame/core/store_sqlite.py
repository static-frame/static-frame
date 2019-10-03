

import sqlite3
import typing as tp



import numpy as np # type: ignore

from static_frame.core.frame import Frame
from static_frame.core.store import Store


from static_frame.core.store_filter import StoreFilter
from static_frame.core.store_filter import STORE_FILTER_DEFAULT

from static_frame.core.doc_str import doc_inject

from static_frame.core.util import DtypesSpecifier
from static_frame.core.util import DTYPE_INT_KIND
from static_frame.core.util import DTYPE_STR_KIND
from static_frame.core.util import DTYPE_NAN_KIND
# from static_frame.core.util import DTYPE_DATETIME_KIND
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import BOOL_TYPES
from static_frame.core.util import NUMERIC_TYPES




class StoreSQLite(Store):

    _EXT: str = '.sqlite'

    @staticmethod
    def _dtype_to_affinity_type(
            dtype: np.dtype,
            ) -> str:
        '''
        Return a pair of writer function, Boolean, where Boolean denotes if replacements need be applied.
        '''
        kind = dtype.kind
        if dtype == DTYPE_BOOL:
            return 'BOOLEAN' # maps to NUMERIC
        elif kind in DTYPE_STR_KIND:
            return 'TEXT'
        elif kind in DTYPE_INT_KIND:
            return 'INTEGER'
        elif kind in DTYPE_NAN_KIND:
            return 'REAL'
        return 'NONE'

    @classmethod
    def _frame_to_table(cls,
            *,
            frame: Frame,
            label: str,
            cursor: sqlite3.Cursor,
            include_columns: bool,
            include_index: bool,
            store_filter: tp.Optional[StoreFilter]
            ) -> None:

        field_name_to_field_type = (
                (field, cls._dtype_to_affinity_type(dtype))
                for field, dtype in frame.dtypes.items())
        create_fields = ', '.join(f'{k} {v}' for k, v in field_name_to_field_type)
        create = f'CREATE TABLE {label} ({create_fields})'

        cursor.execute(create)

        if frame.columns.depth == 1:
            insert_fields = ', '.join(f'{k}' for k in frame.columns)
        else:
            # insert_fields = ', '.join(' '.join(f"'{sub}'" for sub in k) for k in frame.columns)
            insert_fields = ', '.join(str(k) for k in frame.columns)

        values = frame.iter_array(1)
        insert_template = ', '.join('?' for _ in frame.columns)
        insert = f'insert into {label} ({insert_fields}) values ({insert_template})'

        # import ipdb; ipdb.set_trace()
        # cursor.execute("PRAGMA table_info(f3)")

        # numpy types go in as blobs if they are not individuall converted tp python types
        cursor.executemany(insert, list(tuple((int(x) for x in v)) for v in values))


    def write(self,
            items: tp.Iterable[tp.Tuple[tp.Optional[str], Frame]],
            *,
            include_index: bool = True,
            include_columns: bool = True,
            store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ) -> None:


        # NOTE: register adapters for NP types:
        # sqlite3.register_adapter(np.int64, int)

        # hierarchical columns might be stored as tuples
        conn = sqlite3.connect(self._fp)
        cursor = conn.cursor()

        for label, frame in items:
            self._frame_to_table(frame=frame,
                    label=label,
                    cursor=cursor,
                    include_columns=include_columns,
                    include_index=include_index,
                    store_filter=store_filter
                    )

        conn.commit()
        conn.close()



    @doc_inject(selector='constructor_frame')
    def read(self,
            label: tp.Optional[str] = None,
            *,
            index_depth: int=1,
            columns_depth: int=1,
            dtypes: DtypesSpecifier = None,
            store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ) -> Frame:
        '''
        Args:
            {dtypes}
        '''
        pass