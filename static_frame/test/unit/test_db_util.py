import sqlite3

import numpy as np
import pytest

from static_frame.core.db_util import DBQuery
from static_frame.core.db_util import DBType
from static_frame.core.db_util import dtype_to_type_decl_mysql
from static_frame.core.db_util import dtype_to_type_decl_postgresql
from static_frame.core.db_util import dtype_to_type_decl_sqlite
from static_frame.core.db_util import mysql_type_decl_to_dtype
from static_frame.core.db_util import postgresql_type_decl_to_dtype
from static_frame.core.frame import Frame
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.index_hierarchy import IndexHierarchyGO
from static_frame.test.test_case import temp_file

#-------------------------------------------------------------------------------

def test_dt_to_td_sqlite_a():
    assert dtype_to_type_decl_sqlite(np.dtype(np.int64)) == 'INTEGER'
    assert dtype_to_type_decl_sqlite(np.dtype(np.int32)) == 'INTEGER'
    assert dtype_to_type_decl_sqlite(np.dtype(np.int16)) == 'INTEGER'
    assert dtype_to_type_decl_sqlite(np.dtype(np.int8)) == 'INTEGER'


def test_dt_to_td_postgres_a1():
    assert dtype_to_type_decl_postgresql(np.dtype(np.int64)) == 'BIGINT'
    assert dtype_to_type_decl_postgresql(np.dtype(np.int32)) == 'INTEGER'
    assert dtype_to_type_decl_postgresql(np.dtype(np.int16)) == 'SMALLINT'
    assert dtype_to_type_decl_postgresql(np.dtype(np.int8)) == 'SMALLINT'

def test_dt_to_td_postgres_a2():
    assert dtype_to_type_decl_postgresql(np.dtype(np.uint64)) == 'BIGINT'
    assert dtype_to_type_decl_postgresql(np.dtype(np.uint32)) == 'BIGINT'
    assert dtype_to_type_decl_postgresql(np.dtype(np.uint16)) == 'INTEGER'
    assert dtype_to_type_decl_postgresql(np.dtype(np.uint8)) == 'SMALLINT'


def test_dt_to_td_mysql_a():
    assert dtype_to_type_decl_mysql(np.dtype(np.int64)) == 'BIGINT'
    assert dtype_to_type_decl_mysql(np.dtype(np.int32)) == 'INT'
    assert dtype_to_type_decl_mysql(np.dtype(np.int16)) == 'SMALLINT'
    assert dtype_to_type_decl_mysql(np.dtype(np.int8)) == 'TINYINT'

def test_dt_to_td_mysql_b():
    assert dtype_to_type_decl_mysql(np.dtype(np.uint64)) == 'BIGINT UNSIGNED'
    assert dtype_to_type_decl_mysql(np.dtype(np.uint32)) == 'INT UNSIGNED'
    assert dtype_to_type_decl_mysql(np.dtype(np.uint16)) == 'SMALLINT UNSIGNED'
    assert dtype_to_type_decl_mysql(np.dtype(np.uint8)) == 'TINYINT UNSIGNED'

def test_dt_to_td_sqlite_b():
    assert dtype_to_type_decl_sqlite(np.dtype(np.float64)) == 'REAL'
    assert dtype_to_type_decl_sqlite(np.dtype(np.float32)) == 'REAL'

def test_dt_to_td_postgres_b():
    assert dtype_to_type_decl_postgresql(np.dtype(np.float64)) == 'DOUBLE PRECISION'
    assert dtype_to_type_decl_postgresql(np.dtype(np.float32)) == 'REAL'

def test_dt_to_td_mysql_c1():
    assert dtype_to_type_decl_mysql(np.dtype(np.float64)) == 'DOUBLE'
    assert dtype_to_type_decl_mysql(np.dtype(np.float32)) == 'FLOAT'


def test_dt_to_td_sqlite_c():
    assert dtype_to_type_decl_sqlite(np.dtype(np.bool_)) == 'BOOLEAN'

def test_dt_to_td_postgres_c():
    assert dtype_to_type_decl_postgresql(np.dtype(np.bool_)) == 'BOOLEAN'

def test_dt_to_td_mysql_c2():
    assert dtype_to_type_decl_mysql(np.dtype(np.bool_)) == 'TINYINT(1)'


def test_dt_to_td_sqlite_d():
    assert dtype_to_type_decl_sqlite(np.dtype('U10')) == 'TEXT'
    assert dtype_to_type_decl_sqlite(np.dtype('S10')) == 'BLOB'

def test_dt_to_td_postgres_d():
    assert dtype_to_type_decl_postgresql(np.dtype('U10')) == 'TEXT'
    assert dtype_to_type_decl_postgresql(np.dtype('S10')) == 'BYTEA'

def test_dt_to_td_mysql_d():
    assert dtype_to_type_decl_mysql(np.dtype('U10')) == 'TEXT'
    assert dtype_to_type_decl_mysql(np.dtype('S10')) == 'BLOB'


def test_dt_to_td_sqlite_e():
    assert dtype_to_type_decl_sqlite(np.dtype('datetime64[ns]')) == 'TEXT'
    assert dtype_to_type_decl_sqlite(np.dtype('timedelta64[ns]')) == 'TEXT'

def test_dt_to_td_postgres_e():
    assert dtype_to_type_decl_postgresql(np.dtype('datetime64[Y]')) == 'DATE'
    assert dtype_to_type_decl_postgresql(np.dtype('datetime64[M]')) == 'DATE'
    assert dtype_to_type_decl_postgresql(np.dtype('datetime64[D]')) == 'DATE'
    assert dtype_to_type_decl_postgresql(np.dtype('datetime64[h]')) == 'TIMESTAMP'
    assert dtype_to_type_decl_postgresql(np.dtype('datetime64[ms]')) == 'TIMESTAMP'
    assert dtype_to_type_decl_postgresql(np.dtype('timedelta64[s]')) == 'TIMESTAMP'

def test_dt_to_td_mysql_e():
    assert dtype_to_type_decl_mysql(np.dtype('datetime64[Y]')) == 'DATE'
    assert dtype_to_type_decl_mysql(np.dtype('datetime64[M]')) == 'DATE'
    assert dtype_to_type_decl_mysql(np.dtype('datetime64[D]')) == 'DATE'
    assert dtype_to_type_decl_mysql(np.dtype('datetime64[h]')) == 'DATETIME'
    assert dtype_to_type_decl_mysql(np.dtype('datetime64[ms]')) == 'DATETIME(6)'
    assert dtype_to_type_decl_mysql(np.dtype('timedelta64[s]')) == 'DATETIME'


def test_dt_to_td_sqlite_f():
    assert dtype_to_type_decl_sqlite(np.dtype('O')) == 'NONE'
    assert dtype_to_type_decl_sqlite(np.dtype('complex64')) == 'REAL'
    assert dtype_to_type_decl_sqlite(np.dtype('complex128')) == 'REAL'
    assert dtype_to_type_decl_sqlite(np.dtype('V10')) == 'NONE'

def test_dt_to_td_postgres_f():
    assert dtype_to_type_decl_postgresql(np.dtype('complex64')) == 'JSONB'
    assert dtype_to_type_decl_postgresql(np.dtype('complex128')) == 'JSONB'
    with pytest.raises(ValueError):
        dtype_to_type_decl_postgresql(np.dtype('O'))
        dtype_to_type_decl_postgresql(np.dtype('V10'))

def test_dt_to_td_mysql_f():
    assert dtype_to_type_decl_mysql(np.dtype('complex64')) == 'JSON'
    assert dtype_to_type_decl_mysql(np.dtype('complex128')) == 'JSON'
    with pytest.raises(ValueError):
        dtype_to_type_decl_mysql(np.dtype('O'))
        dtype_to_type_decl_mysql(np.dtype('V10'))

#-------------------------------------------------------------------------------

def test_db_type_a():
    dbt = DBType.SQLITE
    assert dbt.to_placeholder() == '?'

    tttd = dbt.to_dytpe_to_type_decl()
    assert tttd[np.dtype(np.int64)] == 'INTEGER'

def test_db_type_b():
    dbt = DBType.POSTGRESQL
    assert dbt.to_placeholder() == '%s'

    tttd = dbt.to_dytpe_to_type_decl()
    assert tttd[np.dtype(np.int64)] == 'BIGINT'

def test_db_type_c1():
    dbt = DBType.MYSQL
    assert dbt.to_placeholder() == '%s'

    tttd = dbt.to_dytpe_to_type_decl()
    assert tttd[np.dtype(np.int64)] == 'BIGINT'

def test_db_type_c2():
    dbt = DBType.MARIADB
    assert dbt.to_placeholder() == '%s'

    tttd = dbt.to_dytpe_to_type_decl()
    assert tttd[np.dtype(np.int64)] == 'BIGINT'

def test_db_type_d():
    dbt = DBType.UNKNOWN
    assert dbt.to_placeholder() == '%s'

    with pytest.raises(NotImplementedError):
        _ = dbt.to_dytpe_to_type_decl()


#-------------------------------------------------------------------------------

def test_dbquery_create_a1():
    f = Frame.from_records([('a', 3, False), ('b', 0, True)],
            columns=('x', 'y', 'z'),
            name='foo',
            dtypes=(np.str_, np.int64, np.bool_),
            )

    dbq = DBQuery.from_db_type(None, DBType.SQLITE)
    post = dbq._sql_create(frame=f, label=f.name, schema='', include_index=False)
    assert post == 'CREATE TABLE IF NOT EXISTS foo (x TEXT, y INTEGER, z BOOLEAN);'

def test_dbquery_create_a2():
    f = Frame.from_records([('a', 3, False), ('b', 0, True)],
            columns=('x', 'y', 'z'),
            name='foo',
            dtypes=(np.str_, np.int64, np.bool_),
            )

    dbq = DBQuery.from_db_type(None, DBType.SQLITE)
    post = dbq._sql_create(frame=f, label=f.name, schema='', include_index=True)
    assert post == 'CREATE TABLE IF NOT EXISTS foo (__index0__ INTEGER, x TEXT, y INTEGER, z BOOLEAN);'

def test_dbquery_create_a3():
    f = Frame.from_records([('a', 3, False), ('b', -20, True)],
            columns=('x', 'y', 'z'),
            name='foo',
            dtypes=(np.str_, np.int64, np.bool_),
            )

    # with TemporaryDirectory() as fp_dir:
    with temp_file('.db') as fp:
        # fp = Path(fp_dir) / 'temp.db'
        conn = sqlite3.connect(fp)
        dbq = DBQuery.from_defaults(conn)
        dbq.execute(frame=f, label=f.name, include_index=False, scalars=False, eager=False)
        post = list(conn.cursor().execute(f'select * from {f.name}'))
        assert post == [('a', 3, 0), ('b', -20, 1)]

def test_dbquery_create_a4():
    f = Frame.from_records([('a', 3, False), ('b', -20, True)],
            columns=('x', 'y', 'z'),
            name='foo',
            dtypes=(np.str_, np.int64, np.bool_),
            )
    with temp_file('.db') as fp:
        conn = sqlite3.connect(fp)
        dbq = DBQuery.from_defaults(conn)
        query, parameters = dbq._sql_insert(frame=f, label=f.name, schema='', include_index=False, scalars=True, eager=True)
        assert query == 'INSERT INTO foo (x,y,z)\n        VALUES (?,?,?);\n        '
        assert parameters == [('a', 3, False), ('b', -20, True)]
        assert parameters[0][1].__class__ == np.int64

def test_dbquery_create_a5():
    f = Frame.from_records([('a', 3, False), ('b', -20, True)],
            columns=('x', 'y', 'z'),
            index=('p', 'q'),
            name='foo',
            dtypes=(np.str_, np.int64, np.bool_),
            )

    # with TemporaryDirectory() as fp_dir:
    with temp_file('.db') as fp:
        # fp = Path(fp_dir) / 'temp.db'
        conn = sqlite3.connect(fp)
        dbq = DBQuery.from_defaults(conn)
        dbq.execute(frame=f, label=f.name, include_index=True, scalars=False)
        result = conn.cursor().execute(f'select * from {f.name}')
        assert [d[0] for d in result.description] == ['__index0__', 'x', 'y', 'z']
        post = list(result)
        assert post == [('p', 'a', 3, 0), ('q', 'b', -20, 1)]

def test_dbquery_create_a6():
    f = Frame.from_records([('a', 3, False), ('b', -20, True)],
            columns=('x', 'y', 'z'),
            index=IndexHierarchy.from_labels([('p', 100), ('q', 200)], name=('v', 'w')),
            name='foo',
            dtypes=(np.str_, np.int64, np.bool_),
            )
    with temp_file('.db') as fp:
        conn = sqlite3.connect(fp)
        dbq = DBQuery.from_defaults(conn)
        dbq.execute(frame=f, label=f.name, include_index=True, scalars=False)
        result = conn.cursor().execute(f'select * from {f.name}')
        assert [d[0] for d in result.description] == ['v', 'w', 'x', 'y', 'z']
        post = list(result)
        assert post == [('p', 100, 'a', 3, 0), ('q', 200, 'b', -20, 1)]

def test_dbquery_create_a7():
    ih = IndexHierarchyGO.from_labels([('p', 100), ('q', 200)], name=('v', 'w'))
    ih.append(('r', 300))

    f = Frame.from_records(
            [('a', 3, False), ('b', -20, True), ('c', 5, True)],
            columns=('x', 'y', 'z'),
            index=ih,
            name='foo',
            dtypes=(np.str_, np.int64, np.bool_),
            )
    with temp_file('.db') as fp:
        conn = sqlite3.connect(fp)
        dbq = DBQuery.from_defaults(conn)
        dbq.execute(frame=f, label=f.name, include_index=True, scalars=False)
        result = conn.cursor().execute(f'select * from {f.name}')
        assert [d[0] for d in result.description] == ['v', 'w', 'x', 'y', 'z']
        post = list(result)
        assert post == [('p', 100, 'a', 3, 0), ('q', 200, 'b', -20, 1), ('r', 300, 'c', 5, 1)]


#-------------------------------------------------------------------------------
def test_dbquery_create_b1():
    f = Frame.from_records([('a', 3, False), ('b', 0, True)],
            columns=('x', 'y', 'z'),
            name='foo',
            dtypes=(np.str_, np.int64, np.bool_),
            )

    dbq = DBQuery.from_db_type(None, DBType.POSTGRESQL)
    post = dbq._sql_create(frame=f, label=f.name, schema='', include_index=False)
    assert post == 'CREATE TABLE IF NOT EXISTS foo (x TEXT, y BIGINT, z BOOLEAN);'

def test_dbquery_create_b2():
    f = Frame.from_records([('a', 3, False), ('b', 0, True)],
            columns=('x', 'y', 'z'),
            name='foo',
            dtypes=(np.str_, np.int64, np.bool_),
            )

    dbq = DBQuery.from_db_type(None, DBType.POSTGRESQL)
    post = dbq._sql_create(frame=f, label=f.name, schema='public', include_index=False)
    assert post == 'CREATE TABLE IF NOT EXISTS public.foo (x TEXT, y BIGINT, z BOOLEAN);'


def test_dbquery_create_c():
    f = Frame.from_records([('a', 3, False), ('b', 0, True)],
            columns=('x', 'y', 'z'),
            name='foo',
            dtypes=(np.str_, np.int64, np.bool_),
            )

    dbq = DBQuery.from_db_type(None, DBType.MYSQL)
    post = dbq._sql_create(frame=f, label=f.name, schema='', include_index=False)
    assert post == 'CREATE TABLE IF NOT EXISTS foo (x TEXT, y BIGINT, z TINYINT(1));'


#-------------------------------------------------------------------------------



### PostgreSQL Tests ###
def test_postgresql_type_decl_to_dtype():
    assert postgresql_type_decl_to_dtype("SMALLINT") == np.dtype(np.int16)
    assert postgresql_type_decl_to_dtype("INTEGER") == np.dtype(np.int32)
    assert postgresql_type_decl_to_dtype("INT") == np.dtype(np.int32)
    assert postgresql_type_decl_to_dtype("BIGINT") == np.dtype(np.int64)

    assert postgresql_type_decl_to_dtype("REAL") == np.dtype(np.float32)
    assert postgresql_type_decl_to_dtype("FLOAT") == np.dtype(np.float32)
    assert postgresql_type_decl_to_dtype("DOUBLE PRECISION") == np.dtype(np.float64)

    assert postgresql_type_decl_to_dtype("BOOLEAN") == np.dtype(np.bool_)

    assert postgresql_type_decl_to_dtype("TEXT") == np.dtype(np.str_)
    assert postgresql_type_decl_to_dtype("BYTEA") == np.dtype(np.bytes_)

    assert postgresql_type_decl_to_dtype("JSONB") == np.dtype(np.complex128)
    assert postgresql_type_decl_to_dtype("JSON") == np.dtype(np.complex128)

    assert postgresql_type_decl_to_dtype("DATE") == np.dtype("datetime64[D]")

    assert postgresql_type_decl_to_dtype("TIME") is None
    assert postgresql_type_decl_to_dtype("TIME(3)") is None
    assert postgresql_type_decl_to_dtype("TIME(6)") is None

    assert postgresql_type_decl_to_dtype("TIMESTAMP") == np.dtype("datetime64[s]")
    assert postgresql_type_decl_to_dtype("TIMESTAMP(3)") == np.dtype("datetime64[ms]")
    assert postgresql_type_decl_to_dtype("TIMESTAMP(6)") == np.dtype("datetime64[us]")
    assert postgresql_type_decl_to_dtype("TIMESTAMP(9)") == np.dtype("datetime64[ns]")

    assert postgresql_type_decl_to_dtype("UNKNOWN_TYPE") is None


def test_mysql_type_decl_to_dtype():
    assert mysql_type_decl_to_dtype("TINYINT") == np.dtype(np.int8)
    assert mysql_type_decl_to_dtype("SMALLINT") == np.dtype(np.int16)
    assert mysql_type_decl_to_dtype("INTEGER") == np.dtype(np.int32)
    assert mysql_type_decl_to_dtype("INT") == np.dtype(np.int32)
    assert mysql_type_decl_to_dtype("BIGINT") == np.dtype(np.int64)

    assert mysql_type_decl_to_dtype("TINYINT UNSIGNED") == np.dtype(np.uint8)
    assert mysql_type_decl_to_dtype("SMALLINT UNSIGNED") == np.dtype(np.uint16)
    assert mysql_type_decl_to_dtype("INTEGER UNSIGNED") == np.dtype(np.uint32)
    assert mysql_type_decl_to_dtype("INT UNSIGNED") == np.dtype(np.uint32)
    assert mysql_type_decl_to_dtype("BIGINT UNSIGNED") == np.dtype(np.uint64)

    assert mysql_type_decl_to_dtype("REAL") == np.dtype(np.float32)
    assert mysql_type_decl_to_dtype("FLOAT") == np.dtype(np.float32)
    assert mysql_type_decl_to_dtype("DOUBLE") == np.dtype(np.float64)

    assert mysql_type_decl_to_dtype("BOOLEAN") == np.dtype(np.bool_)
    assert mysql_type_decl_to_dtype("TINYINT(1)") == np.dtype(np.bool_)

    assert mysql_type_decl_to_dtype("TEXT") == np.dtype(np.str_)
    assert mysql_type_decl_to_dtype("BLOB") == np.dtype(np.bytes_)

    assert mysql_type_decl_to_dtype("DATE") == np.dtype("datetime64[D]")

    assert mysql_type_decl_to_dtype("DATETIME") == np.dtype("datetime64[s]")
    assert mysql_type_decl_to_dtype("DATETIME(3)") == np.dtype("datetime64[ms]")
    assert mysql_type_decl_to_dtype("DATETIME(6)") == np.dtype("datetime64[us]")

    assert mysql_type_decl_to_dtype("TIMESTAMP") == np.dtype("datetime64[s]")  # MySQL TIMESTAMP is always in seconds

    assert mysql_type_decl_to_dtype("UNKNOWN_TYPE") is None



#-------------------------------------------------------------------------------
def test_cursor_to_dtypes_a():

    with temp_file('.db') as fp:
        db_conn = sqlite3.connect(fp)
        dbt = DBType.from_connection(db_conn)
        with pytest.raises(ValueError):
            _ = dbt.cursor_to_dtypes(db_conn.cursor())


