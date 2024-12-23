
import typing as tp
from enum import Enum

import numpy as np
import sqlite


from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import DTYPE_STR_KINDS
from static_frame.core.util import DTYPE_INT_KINDS
from static_frame.core.util import DTYPE_INEXACT_KINDS


TDtypeAny = np.dtype[tp.Any] #pragma: no cover




#-------------------------------------------------------------------------------

def dtype_to_db_type(dtype: TDtypeAny) -> str:
    kind = dtype.kind
    if dtype == DTYPE_BOOL:
        return 'BOOLEAN' # maps to NUMERIC
    elif kind in DTYPE_STR_KINDS:
        return 'TEXT'
    elif kind in DTYPE_INT_KINDS:
        return 'INTEGER'
    elif kind in DTYPE_INEXACT_KINDS:
        return 'REAL'
    elif kind == "M":  # Datetime types
        return "TEXT"  # ISO 8601 string representation
    return 'NONE'


def numpy_dtype_to_sqlite(dtype):
    kind = dtype.kind
    if kind == "i":  # Integer types
        return "INTEGER"
    elif kind == "u":  # Unsigned integer types
        return "INTEGER"
    elif kind == "f":  # Floating-point types
        return "REAL"
    elif kind == "c":  # Complex numbers
        return "TEXT"  # Store complex numbers as string representations
    elif kind == "b":  # Boolean types
        return "INTEGER"  # SQLite has no native BOOLEAN type; use INTEGER
    elif kind == "O":  # Object types (e.g., Python objects)
        return "TEXT"
    elif kind == "S":  # Byte string
        return "BLOB"
    elif kind == "U":  # Unicode string
        return "TEXT"
    elif kind == "M":  # Datetime types
        return "TEXT"  # ISO 8601 string representation
    raise ValueError(f"Unsupported dtype: {dtype}")




def numpy_dtype_to_sql(dtype, is_postgres):
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
    elif kind == "O":  # Object types (e.g., Python objects)
        return "TEXT"
    elif kind == "S":
        return "BYTEA" if is_postgres else "BLOB"
    elif kind == "U":
        return "TEXT"
    elif kind == "M":
        datetime_unit = np.datetime_data(dtype)[0]
        if datetime_unit in {"Y", "M"}:  # Year or month precision
            return "DATE"
        elif datetime_unit == "D":  # Day precision
            return "DATE"
        elif datetime_unit in {"h", "m", "s"}:
            return "TIMESTAMP" if is_postgres else "DATETIME"
        elif datetime_unit in {"ms", "us", "ns"}:
            return "TIMESTAMP" if is_postgres else "DATETIME(6)"
        else:
            raise ValueError(f"Unsupported datetime unit: {datetime_unit}")
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


#-------------------------------------------------------------------------------

class DBType(Enum):
    SQLITE = 0
    POSTGRESQL = 1
    MYSQL = 2
    UNKNOWN = 3

    def to_placeholder(self):
        if self in (DBType.SQLITE,):
            return '?'
        elif self in (DBType.POSTGRESQL, DBType.MYSQL):
            return '%s'
        return '%s'

    def to_dytpe_to_type_decl(self):
        if self in (DBType.SQLITE,):
            return '?'
        elif self in (DBType.POSTGRESQL, DBType.MYSQL):
            return '%s'
        raise NotImplementedError()

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

TDtypeToTypeDecl = dict[str, TDtypeAny]

class DBQuery:
    __slots__ = (
        '_connection',
        'db_type',
        '_placeholder',
        '_dtype_to_type_decl',
        )

    @classmethod
    def from_defaults(cls,
        connection: sqlite3.Connection,
        palceholder: str = '',
        dtype_to_type_decl: TDtypeToTypeDecl | None = None,
        ):
        if not placeholder or not dtype_to_type_decl:
            db_type = connection_to_db(connection)
        else:
            db_type = DBType.UNKNOWN
        ph = palceholder if placeholder else db_type.to_placeholder()
        dttd = dtype_to_db_type if dtype_to_db_type else db_type.to_dytpe_to_type_decl()


    def __init__(self,
            connection: sqlite3.Connection,
            db_type: DBType,
            placeholder: str,
            dtype_to_type_decl: TDtypeToTypeDecl,
            ):
        self._conection = connection
        self._placeholder = placeholder
        self._dtype_to_type_decl = dtype_to_type_decl

