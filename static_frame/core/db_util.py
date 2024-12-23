
import sqlite3
import typing as tp
from collections.abc import Mapping
from enum import Enum
from functools import partial

import numpy as np

from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import DTYPE_INEXACT_KINDS
from static_frame.core.util import DTYPE_INT_KINDS
from static_frame.core.util import DTYPE_NAT_KINDS
from static_frame.core.util import DTYPE_STR_KINDS

TDtypeAny = np.dtype[tp.Any] #pragma: no cover

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
    Handle postgresql and mysql
    '''
    kind = dtype.kind
    itemsize = dtype.itemsize

    if kind == "i":  # Integer types
        if itemsize <= 2:
            return "SMALLINT"
        elif itemsize <= 4:
            return "INTEGER" if is_postgres else "INT"
        else:
            return "BIGINT"
    elif kind == "u":  # Unsigned integer types
        if itemsize <= 2:
            return "SMALLINT" if is_postgres else "SMALLINT UNSIGNED"
        elif itemsize <= 4:
            return "INTEGER" if is_postgres else "INT UNSIGNED"
        else:
            return "BIGINT UNSIGNED" if not is_postgres else "BIGINT"
    elif kind == "f":  # Floating-point types
        if itemsize <= 4:
            return "REAL" if is_postgres else "FLOAT"
        else:
            return "DOUBLE PRECISION" if is_postgres else "DOUBLE"
    elif kind == "c":  # Complex numbers
        return "JSONB" if is_postgres else "JSON"
    elif kind == "b":  # Boolean types
        return "BOOLEAN" if is_postgres else "TINYINT(1)"
    elif kind == "S":
        return "BYTEA" if is_postgres else "BLOB"
    elif kind == "U":
        return "TEXT"
    elif kind in DTYPE_NAT_KINDS:
        datetime_unit = np.datetime_data(dtype)[0]
        if datetime_unit in {"Y", "M"}:  # Year or month precision
            return "DATE"
        elif datetime_unit == "D":  # Day precision
            return "DATE"
        elif datetime_unit in {"h", "m", "s"}:
            return "TIMESTAMP" if is_postgres else "DATETIME"
        elif datetime_unit in {"ms", "us", "ns"}:
            return "TIMESTAMP" if is_postgres else "DATETIME(6)"
    raise ValueError(f"Unsupported dtype: {dtype}")

dtype_to_type_decl_postgresql = partial(
        _dtype_to_type_decl_many,
        is_postgres=True,
        )
dtype_to_type_decl_mysql = partial(
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
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

#-------------------------------------------------------------------------------

class DBType(Enum):
    SQLITE = 0
    POSTGRESQL = 1
    MYSQL = 2
    UNKNOWN = 3

    def to_placeholder(self) -> str:
        if self in (DBType.SQLITE,):
            return '?'
        elif self in (DBType.POSTGRESQL, DBType.MYSQL):
            return '%s'
        return '%s'

    def to_dytpe_to_type_decl(self) -> TDtypeToTypeDecl:
        if self == DBType.SQLITE:
            return DTypeToTypeDecl(dtype_to_type_decl_sqlite)
        elif self == DBType.POSTGRESQL:
            return DTypeToTypeDecl(dtype_to_type_decl_postgresql)
        elif self == DBType.MYSQL:
            return DTypeToTypeDecl(dtype_to_type_decl_mysql)
        raise NotImplementedError('A dtype to type declaration mapping must be provided.')


def connection_to_db(conn: tp.Any) -> DBType:
    if isinstance(conn, sqlite3.Connection):
        return DBType.SQLITE
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT version();")  # PostgreSQL and MySQL
        result = cursor.fetchone()
    except Exception:
        result = ''

    if result:
        version_info = result[0].lower()
        if "postgresql" in version_info:
            return DBType.POSTGRESQL
        elif "mysql" in version_info or "mariadb" in version_info:
            return DBType.MYSQL
    return DBType.UNKNOWN


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
        db_type = connection_to_db(connection)
        ph = (placeholder if placeholder
                else db_type.to_placeholder())
        dttd = (dtype_to_type_decl if dtype_to_type_decl
                else db_type.to_dytpe_to_type_decl())
        return cls(connection, db_type, ph, dttd)

    def __init__(self,
            connection: sqlite3.Connection,
            db_type: DBType,
            placeholder: str,
            dtype_to_type_decl: TDtypeToTypeDecl,
            ) -> None:
        self._conection = connection
        self._db_type = db_type
        self._placeholder = placeholder
        self._dtype_to_type_decl = dtype_to_type_decl

