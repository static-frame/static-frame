from __future__ import annotations

import os
import sqlite3
from contextlib import suppress
from fractions import Fraction

import numpy as np
import typing_extensions as tp

from static_frame.core.db_util import dtype_to_type_decl_sqlite
# from static_frame.core.doc_str import doc_inject
from static_frame.core.frame import Frame
from static_frame.core.store import Store
from static_frame.core.store import store_coherent_non_write
from static_frame.core.store import store_coherent_write
from static_frame.core.store_config import StoreConfigMap
from static_frame.core.store_config import StoreConfigMapInitializer

if tp.TYPE_CHECKING:
    from static_frame.core.util import TLabel  # pragma: no cover
    TDtypeAny = np.dtype[tp.Any] #pragma: no cover

TFrameAny = Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]]  #pragma: no cover


class StoreSQLite(Store):

    _EXT: tp.FrozenSet[str] =  frozenset(('.db', '.sqlite'))
    _BYTES_ONE = b'1'

    @classmethod
    def _frame_to_table(cls,
            *,
            frame: TFrameAny,
            label: str, # can be None
            cursor: sqlite3.Cursor,
            include_columns: bool,
            include_index: bool,
            # store_filter: tp.Optional[StoreFilter]
            ) -> None:

        # here we provide a row-based representation that is externally usable as an slqite db; an alternative approach would be to store one cell pre column, where the column isstored as as binary BLOB; see here https://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database

        field_names, dtypes = cls.get_field_names_and_dtypes(
                frame=frame,
                include_index=include_index,
                include_index_name=True,
                include_columns=include_columns,
                include_columns_name=False,
                force_brackets=True # needed for having numbers as field names
                )

        index = frame._index
        # columns = frame._columns

        if not include_index:
            create_primary_key = ''
        else:
            primary_fields = ', '.join(field_names[:index.depth])
            # need leading comma
            create_primary_key = f', PRIMARY KEY ({primary_fields})'

        field_name_to_field_type = (
                (field, dtype_to_type_decl_sqlite(dtype))
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
            items: tp.Iterable[tp.Tuple[TLabel, TFrameAny]],
            *,
            config: StoreConfigMapInitializer = None,
            # store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT,
            ) -> None:

        config_map = StoreConfigMap.from_initializer(config)

        # NOTE: register adapters for NP types:
        # numpy scalar types go in as blobs if they are not individually converted tp python types
        sqlite3.register_adapter(np.int64, int)
        sqlite3.register_adapter(np.int32, int)
        sqlite3.register_adapter(np.int16, int)
        sqlite3.register_adapter(np.bool_, bool)
        # common python types
        sqlite3.register_adapter(Fraction, str)
        sqlite3.register_adapter(complex, lambda x: f'{x.real}:{x.imag}')

        # SQLite will naturally try to update, not replace, a DB found at an FP; this is not how all other stores work, so best to remove the file first.
        with suppress(FileNotFoundError):
            os.remove(self._fp)

        with sqlite3.connect(self._fp, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            cursor = conn.cursor()
            for label, frame in items:
                c = config_map[label]
                # if label is STORE_LABEL_DEFAULT this will raise
                label = config_map.default.label_encode(label)

                self._frame_to_table(frame=frame,
                        label=label,
                        cursor=cursor,
                        include_columns=c.include_columns,
                        include_index=c.include_index,
                        # store_filter=store_filter
                        )

            conn.commit()

    @store_coherent_non_write
    def read_many(self,
            labels: tp.Iterable[TLabel],
            *,
            config: StoreConfigMapInitializer = None,
            container_type: tp.Type[TFrameAny] = Frame,
            ) -> tp.Iterator[TFrameAny]:

        config_map = StoreConfigMap.from_initializer(config)
        sqlite3.register_converter('BOOLEAN', lambda x: x == self._BYTES_ONE)

        with sqlite3.connect(self._fp,
                detect_types=sqlite3.PARSE_DECLTYPES
                ) as conn:

            for label in labels:
                c = config_map[label]
                label_encoded = config_map.default.label_encode(label)
                name = label
                query = f'SELECT * from "{label_encoded}"'
                f = container_type.from_sql(query,
                        connection=conn,
                        index_depth=c.index_depth,
                        index_constructors=c.index_constructors,
                        columns_depth=c.columns_depth,
                        columns_select=c.columns_select,
                        columns_constructors=c.columns_constructors,
                        dtypes=c.dtypes,
                        name=name,
                        consolidate_blocks=c.consolidate_blocks
                        )
                if c.read_frame_filter is not None:
                    yield c.read_frame_filter(label, f)
                else:
                    yield f

    @store_coherent_non_write
    def labels(self, *,
            config: StoreConfigMapInitializer = None,
            strip_ext: bool = True,
            ) -> tp.Iterator[TLabel]:

        config_map = StoreConfigMap.from_initializer(config)

        with sqlite3.connect(self._fp) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            for row in cursor:
                yield config_map.default.label_decode(row[0])
