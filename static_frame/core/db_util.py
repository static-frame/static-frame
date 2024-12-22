
import typing as tp
from enum import Enum

import numpy as np
import sqlite


class DBType(Enum):
    SQLITE = 0
    POSTGRESQL = 1
    MYSQL = 2
    UNKNOWN = 3


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
    return 'NONE'




def numpy_dtype_to_sqlite(dtype):
    """
    Convert a NumPy dtype to the best-fit SQLite type.

    Parameters:
        dtype (np.dtype): The NumPy dtype to map.

    Returns:
        str: The SQLite type corresponding to the NumPy dtype.
    """
    kind = dtype.kind
    itemsize = dtype.itemsize

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
    elif kind == "S" or kind == "a":  # Byte string
        return "BLOB"
    elif kind == "U":  # Unicode string
        return "TEXT"
    elif kind == "M":  # Datetime types
        return "TEXT"  # ISO 8601 string representation
    else:
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



        # if placeholder:
        #     ph = placeholder
        # elif isinstance(connection, sqlite3.Connection):
        #     ph = '?'
        # else: # psycopg2, PyMySQL
        #     ph = '%s'
