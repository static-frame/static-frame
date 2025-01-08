from __future__ import annotations

import sqlite3
import typing as tp
from collections.abc import Mapping
from enum import Enum
from functools import partial
from itertools import chain

import numpy as np

from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import DTYPE_INEXACT_KINDS
from static_frame.core.util import DTYPE_INT_KINDS
from static_frame.core.util import DTYPE_NAT_KINDS
from static_frame.core.util import DTYPE_STR_KINDS
from static_frame.core.util import TLabel

TDtypeAny = np.dtype[tp.Any]


if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame  # pragma: no cover

#-------------------------------------------------------------------------------

def dtype_to_type_decl_sqlite(
        dtype: TDtypeAny,
        ) -> str:
    kind = dtype.kind
    if kind == "S":
        return "BLOB"
    elif dtype == DTYPE_BOOL:
        return 'BOOLEAN'
    elif kind in DTYPE_STR_KINDS:
        return 'TEXT'
    elif kind in DTYPE_INT_KINDS:
        return 'INTEGER'
    elif kind in DTYPE_INEXACT_KINDS:
        return 'REAL'
    elif kind in DTYPE_NAT_KINDS:
        return "TEXT"
    return 'NONE'


def _dtype_to_type_decl_many(
        dtype: TDtypeAny,
        is_postgres: bool,
        ) -> str:
    '''
    Handle postgresql, mysql, mariadb
    '''
    kind = dtype.kind
    itemsize = dtype.itemsize

    if kind == 'i':  # Integer types
        if itemsize <= 2:
            return 'SMALLINT'
        elif itemsize <= 4:
            return 'INTEGER' if is_postgres else 'INT'
        return 'BIGINT'
    if kind == 'u':  # Unsigned integer types
        if itemsize <= 2:
            return 'SMALLINT' if is_postgres else 'SMALLINT UNSIGNED'
        elif itemsize <= 4:
            return 'INTEGER' if is_postgres else 'INT UNSIGNED'
        return 'BIGINT UNSIGNED' if not is_postgres else 'BIGINT'
    if kind == 'f':  # Floating-point types
        if itemsize <= 4:
            return 'REAL' if is_postgres else 'FLOAT'
        return 'DOUBLE PRECISION' if is_postgres else 'DOUBLE'
    if kind == 'c':  # Complex numbers
        return 'JSONB' if is_postgres else 'JSON'
    if kind == 'b':  # Boolean types
        return 'BOOLEAN' if is_postgres else 'TINYINT(1)'
    if kind == 'S':
        return 'BYTEA' if is_postgres else 'BLOB'
    if kind == 'U':
        return 'TEXT'

    if kind in DTYPE_NAT_KINDS:
        datetime_unit = np.datetime_data(dtype)[0]
        if datetime_unit in {'Y', 'M'}:  # Year or month precision
            return 'DATE'
        elif datetime_unit == 'D':  # Day precision
            return 'DATE'
        elif datetime_unit in ('h', 'm', 's'):
            return 'TIMESTAMP' if is_postgres else 'DATETIME'
        elif datetime_unit in ('ms', 'us', 'ns'):
            return 'TIMESTAMP' if is_postgres else 'DATETIME(6)'
    raise ValueError(f'Unsupported dtype: {dtype}')

dtype_to_type_decl_postgresql = partial(
        _dtype_to_type_decl_many,
        is_postgres=True,
        )
dtype_to_type_decl_mysql = partial(
        _dtype_to_type_decl_many,
        is_postgres=False,
        )
dtype_to_type_decl_mariadb = partial(
        _dtype_to_type_decl_many,
        is_postgres=False,
        )

#-------------------------------------------------------------------------------

TDtypeToTypeDeclFunc = tp.Callable[[TDtypeAny], str]
TDtypeToTypeDecl = Mapping[TDtypeAny, str]

class DTypeToTypeDecl(TDtypeToTypeDecl):
    '''Trivial wrapper of a lookup function to look like a Mapping.
    '''
    def __init__(self, func: TDtypeToTypeDeclFunc):
        self._func = func

    def __getitem__(self, key: TDtypeAny) -> str:
        return self._func(key)

    def __iter__(self) -> tp.Iterator[TDtypeAny]:
        raise NotImplementedError() #pragma: no cover

    def __len__(self) -> int:
        raise NotImplementedError() #pragma: no cover

#-------------------------------------------------------------------------------

class DBType(Enum):
    SQLITE = 0
    POSTGRESQL = 1
    MYSQL = 2
    MARIADB = 3
    UNKNOWN = 4

    @classmethod
    def from_connection(cls, conn: tp.Any) -> DBType:
        if isinstance(conn, sqlite3.Connection):
            return DBType.SQLITE

        cursor = conn.cursor()
        result = None

        # postgres
        try:
            cursor.execute("SELECT version();")
            result = cursor.fetchone()
        except Exception: # pragma: no cover
            pass # pragma: no cover
        if result and "postgresql" in result[0].lower():
            return DBType.POSTGRESQL

        # mysql, mariadb
        try:
            cursor.execute("SHOW VARIABLES LIKE 'version_comment'")
            result = cursor.fetchone()
        except Exception: # pragma: no cover
            pass # pragma: no cover

        if result:
            version_comment = result[1].lower()
            if "mysql" in version_comment:
                return DBType.MYSQL
            if "mariadb" in version_comment:
                return DBType.MARIADB

        return DBType.UNKNOWN #pragma: no cover

    #---------------------------------------------------------------------------
    def to_placeholder(self) -> str:
        if self in (DBType.SQLITE,):
            return '?'
        elif self in (DBType.POSTGRESQL, DBType.MYSQL, DBType.MARIADB):
            return '%s'
        return '%s'

    def to_dytpe_to_type_decl(self) -> TDtypeToTypeDecl:
        if self == DBType.SQLITE:
            return DTypeToTypeDecl(dtype_to_type_decl_sqlite)
        elif self == DBType.POSTGRESQL:
            return DTypeToTypeDecl(dtype_to_type_decl_postgresql)
        elif self == DBType.MYSQL:
            return DTypeToTypeDecl(dtype_to_type_decl_mysql)
        elif self == DBType.MARIADB:
            return DTypeToTypeDecl(dtype_to_type_decl_mariadb)
        raise NotImplementedError('A dtype to type declaration mapping must be provided.')

    def supports_lazy_parameters(self) -> bool:
        if self == DBType.POSTGRESQL or self == DBType.SQLITE:
            return True
        return False

#-------------------------------------------------------------------------------

class DBQuery:
    __slots__ = (
        '_connection',
        '_db_type',
        '_placeholder',
        '_dtype_to_type_decl',
        )

    @classmethod
    def from_defaults(cls,
            connection: sqlite3.Connection,
            placeholder: str = '',
            dtype_to_type_decl: TDtypeToTypeDecl | None = None,
            ) -> tp.Self:
        db_type = DBType.from_connection(connection)
        ph = (placeholder if placeholder
                else db_type.to_placeholder())
        dttd = (dtype_to_type_decl if dtype_to_type_decl
                else db_type.to_dytpe_to_type_decl())
        return cls(connection, db_type, ph, dttd)

    @classmethod
    def from_db_type(cls,
        connection: sqlite3.Connection,
        db_type: DBType,
        ) -> tp.Self:
        return cls(connection,
                db_type,
                db_type.to_placeholder(),
                db_type.to_dytpe_to_type_decl(),
                )

    def __init__(self,
            connection: sqlite3.Connection,
            db_type: DBType,
            placeholder: str,
            dtype_to_type_decl: TDtypeToTypeDecl,
            ) -> None:
        self._connection = connection
        self._db_type = db_type
        self._placeholder = placeholder
        self._dtype_to_type_decl = dtype_to_type_decl

    def _sql_create(self, *,
            frame: Frame,
            label: TLabel,
            schema: str,
            include_index: bool = True,
            ) -> str:
        index = frame._index
        if include_index and index.ndim == 1:
            columns = chain(index.names, frame._columns)
            col_type_pair = ((c, self._dtype_to_type_decl[dt]) for c, dt in zip(
                    columns,
                    chain((index.dtype,), frame._blocks.dtypes) #type: ignore
                    ))

        elif include_index and index.ndim == 2:
            columns = chain(index.names, frame._columns)
            col_type_pair = ((c, self._dtype_to_type_decl[dt]) for c, dt in zip(
                    columns,
                    chain(index.dtypes.values, frame._blocks.dtypes) # type: ignore
                    ))
        else:
            col_type_pair = ((c, self._dtype_to_type_decl[dt]) for c, dt in zip(
                    frame._columns,
                    frame._blocks.dtypes,
                    ))
        create_body = ', '.join(f'{p[0]} {p[1]}' for p in col_type_pair)
        table_name = ((schema + '.') if schema else '') + str(label)
        return f'CREATE TABLE IF NOT EXISTS {table_name} ({create_body});'

    def _sql_insert(self, *,
            frame: Frame,
            label: TLabel,
            schema: str,
            include_index: bool = True,
            scalars: bool = False,
            eager: bool = False,
            ) -> tuple[str, tp.Iterable[tp.Sequence[tp.Any]]]:
        '''
        Args:
            eager: If True, return parameters as realized list, not an iterator
            scalars: Providing scalars is experimental and only appears to work with SQLite, where scalar values are written as bytes.
        '''
        index = frame._index
        row_iter: tp.Iterable[tp.Sequence[tp.Any]]
        index_iter: tp.Iterable[TLabel]
        if scalars:
            # Forcing row-tuple creation will retain Scalar element types
            row_iter = frame._blocks.iter_row_tuples(None)
            index_iter = index
        else:
            row_iter = frame._blocks.iter_row_lists()
            if include_index:
                if index.ndim == 1:
                    index_iter = index.values.tolist()
                else:
                    # NOTE: do not need to check recach as index is immutable by now
                    index_iter = index._blocks.iter_row_lists() # type: ignore

        parameters: tp.Iterable[tp.Sequence[tp.Any]]

        if include_index and index.ndim == 1:
            columns = chain(index.names, frame._columns)
            count = len(frame._columns) + 1
            parameters = ((label, *record)
                    for label, record in zip(index_iter, row_iter))
        elif include_index and index.ndim == 2:
            columns = chain(index.names, frame._columns)
            count = len(frame._columns) + index.depth
            parameters = ((*labels, *record) # pyright: ignore
                    for labels, record in zip(index_iter, row_iter))
        else:
            columns = frame._columns # type: ignore
            count = len(frame._columns)
            parameters = row_iter

        ph = self._placeholder
        table_name = ((schema + '.') if schema else '') + str(label)
        query = f'''INSERT INTO {table_name} ({','.join(str(c) for c in columns)})
        VALUES ({','.join(ph for _ in range(count))});
        '''

        if eager:
            parameters = list(parameters)

        return query, parameters

    def execute(self, *,
            frame: Frame,
            label: TLabel,
            schema: str = '',
            include_index: bool = True,
            create: bool = True,
            scalars: bool = False,
            eager: bool = False,
            ) -> None:
        if create:
            query_create = self._sql_create(frame=frame,
                    label=label,
                    schema=schema,
                    include_index=include_index,
                    )
            # print(query_create)
        query_insert, parameters = self._sql_insert(frame=frame,
                label=label,
                schema=schema,
                include_index=include_index,
                scalars=scalars,
                eager=eager,
                )

        cursor: sqlite3.Cursor | None = None
        try:
            cursor = self._connection.cursor()
            if create:
                cursor.execute(query_create)
            cursor.executemany(query_insert, parameters)
        finally:
            if cursor:
                cursor.close()

    def execute_db_type(self, *,
            frame: Frame,
            label: TLabel,
            schema: str,
            include_index: bool,
            ) -> None:
        '''Entry point that fixes configuration based on the stored DBType.
        '''
        scalars = False # only works with SQLite, and badly
        create = True
        eager = not self._db_type.supports_lazy_parameters()

        self.execute(frame=frame,
                label=label,
                schema=schema,
                include_index=include_index,
                create=create,
                scalars=scalars,
                eager=eager,
                )


