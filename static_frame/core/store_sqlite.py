
import sqlite3
import typing as tp
from fractions import Fraction

import numpy as np


from static_frame.core.doc_str import doc_inject
from static_frame.core.frame import Frame
from static_frame.core.store import Store
from static_frame.core.store import store_coherent_non_write
from static_frame.core.store import store_coherent_write
from static_frame.core.store import StoreConfig
from static_frame.core.store import StoreConfigMap
from static_frame.core.store import StoreConfigMapInitializer

# from static_frame.core.store_filter import STORE_FILTER_DEFAULT
# from static_frame.core.store_filter import StoreFilter

from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import DTYPE_INT_KINDS
from static_frame.core.util import DTYPE_INEXACT_KINDS
from static_frame.core.util import DTYPE_STR_KINDS


class StoreSQLite(Store):

    _EXT: tp.FrozenSet[str] =  frozenset(('.db', '.sqlite'))

    # _EXT: str = '.sqlite'
    _BYTES_ONE = b'1'

    # _BYTES_NONE = b'None'
    # _BYTES_NEGINF = b'-Inf'
    # _BYTES_POSINF = b'Inf'

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
        elif kind in DTYPE_STR_KINDS:
            return 'TEXT'
        elif kind in DTYPE_INT_KINDS:
            return 'INTEGER'
        elif kind in DTYPE_INEXACT_KINDS:
            return 'REAL'
        return 'NONE'

    @classmethod
    def _frame_to_table(cls,
            *,
            frame: Frame,
            label: tp.Optional[str], # can be None
            cursor: sqlite3.Cursor,
            include_columns: bool,
            include_index: bool,
            # store_filter: tp.Optional[StoreFilter]
            ) -> None:

        # here we provide a row-based represerntation that is externally usable as an slqite db; an alternative approach would be to store one cell pre column, where the column iststored as as binary BLOB; see here https://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database

        # for interface compatibility with StoreXLSX, where label can be None
        if label is None:
            label = 'None'

        field_names, dtypes = cls.get_field_names_and_dtypes(
                frame=frame,
                include_index=include_index,
                include_columns=include_columns,
                force_brackets=True # needed for having nu7mbers as field names
                )

        index = frame._index
        columns = frame._columns

        if not include_index:
            create_primary_key = ''
        else:
            primary_fields = ', '.join(field_names[:index.depth])
            # need leading comma
            create_primary_key = f', PRIMARY KEY ({primary_fields})'

        field_name_to_field_type = (
                (field, cls._dtype_to_affinity_type(dtype))
                for field, dtype in zip(field_names, dtypes)
                )

        create_fields = ', '.join(f'{k} {v}' for k, v in field_name_to_field_type)
        create = f'CREATE TABLE "{label}" ({create_fields}{create_primary_key})'
        cursor.execute(create)

        # works for IndexHierarchy too
        insert_fields = ', '.join(f'{k}' for k in field_names)
        insert_template = ', '.join('?' for _ in field_names)
        insert = f'INSERT INTO "{label}" ({insert_fields}) VALUES ({insert_template})'

        values = cls._get_row_iterator(frame=frame, include_index=include_index)
        cursor.executemany(insert, values())

    @store_coherent_write
    def write(self,
            items: tp.Iterable[tp.Tuple[tp.Optional[str], Frame]],
            *,
            config: StoreConfigMapInitializer = None,
            # store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT,
            ) -> None:

        config_map = StoreConfigMap.from_initializer(config)

        # NOTE: register adapters for NP types:
        # numpy types go in as blobs if they are not individualy converted tp python types
        sqlite3.register_adapter(np.int64, int)
        sqlite3.register_adapter(np.int32, int)
        sqlite3.register_adapter(np.int16, int)
        # common python types
        sqlite3.register_adapter(Fraction, str)
        sqlite3.register_adapter(complex, lambda x: f'{x.real}:{x.imag}')


        # hierarchical columns might be stored as tuples
        with sqlite3.connect(self._fp, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            cursor = conn.cursor()
            for label, frame in items:
                c = config_map[label]

                self._frame_to_table(frame=frame,
                        label=label,
                        cursor=cursor,
                        include_columns=c.include_columns,
                        include_index=c.include_index,
                        # store_filter=store_filter
                        )

            conn.commit()
            # conn.close()



    @doc_inject(selector='constructor_frame')
    @store_coherent_non_write
    def read(self,
            label: tp.Optional[str] = None,
            *,
            config: tp.Optional[StoreConfig] = None,
            container_type: tp.Type[Frame] = Frame,
            # store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ) -> Frame:
        '''
        Args:
            {dtypes}
        '''
        if config is None:
            config = StoreConfig() # get default

        sqlite3.register_converter('BOOLEAN', lambda x: x == self._BYTES_ONE)

        # def bytes_to_types(x):
        #     if x == self._BYTES_NONE:
        #         return None
        #     elif x == self._BYTES_NEGINF:
        #         return -np.inf
        #     elif x == self._BYTES_POSINF:
        #         return np.inf
        #     # import ipdb; ipdb.set_trace()
        #     return x.decode()
            # return x
        # sqlite3.register_converter('NONE', bytes_to_types)

        with sqlite3.connect(self._fp,
                detect_types=sqlite3.PARSE_DECLTYPES
                ) as conn:
            # cursor = conn.cursor()
            query = f'SELECT * from "{label}"'
            return tp.cast(Frame, container_type.from_sql(query=query,
                    connection=conn,
                    index_depth=config.index_depth,
                    columns_depth=config.columns_depth,
                    dtypes=config.dtypes,
                    name=label,
                    consolidate_blocks=config.consolidate_blocks
                    ))

    @store_coherent_non_write
    def labels(self, strip_ext: bool = True) -> tp.Iterator[str]:
        with sqlite3.connect(self._fp) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            for row in cursor:
                yield row[0]

