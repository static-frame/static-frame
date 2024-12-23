import numpy as np
import pytest

from static_frame.core.db_util import dtype_to_type_decl_mysql
from static_frame.core.db_util import dtype_to_type_decl_postgresql
from static_frame.core.db_util import dtype_to_type_decl_sqlite
from static_frame.core.db_util import DBType

#-------------------------------------------------------------------------------

def test_dt_to_td_sqlite_a():
    assert dtype_to_type_decl_sqlite(np.dtype(np.int64)) == 'INTEGER'
    assert dtype_to_type_decl_sqlite(np.dtype(np.int32)) == 'INTEGER'
    assert dtype_to_type_decl_sqlite(np.dtype(np.int16)) == 'INTEGER'
    assert dtype_to_type_decl_sqlite(np.dtype(np.int8)) == 'INTEGER'

def test_dt_to_td_postgres_a():
    assert dtype_to_type_decl_postgresql(np.dtype(np.int64)) == 'BIGINT'
    assert dtype_to_type_decl_postgresql(np.dtype(np.int32)) == 'INTEGER'
    assert dtype_to_type_decl_postgresql(np.dtype(np.int16)) == 'SMALLINT'
    assert dtype_to_type_decl_postgresql(np.dtype(np.int8)) == 'SMALLINT'

def test_dt_to_td_mysql_a():
    assert dtype_to_type_decl_mysql(np.dtype(np.int64)) == 'BIGINT'
    assert dtype_to_type_decl_mysql(np.dtype(np.int32)) == 'INT'
    assert dtype_to_type_decl_mysql(np.dtype(np.int16)) == 'SMALLINT'
    assert dtype_to_type_decl_mysql(np.dtype(np.int8)) == 'SMALLINT'


def test_dt_to_td_sqlite_b():
    assert dtype_to_type_decl_sqlite(np.dtype(np.float64)) == 'REAL'
    assert dtype_to_type_decl_sqlite(np.dtype(np.float32)) == 'REAL'

def test_dt_to_td_postgres_b():
    assert dtype_to_type_decl_postgresql(np.dtype(np.float64)) == 'DOUBLE PRECISION'
    assert dtype_to_type_decl_postgresql(np.dtype(np.float32)) == 'REAL'

def test_dt_to_td_mysql_b():
    assert dtype_to_type_decl_mysql(np.dtype(np.float64)) == 'DOUBLE'
    assert dtype_to_type_decl_mysql(np.dtype(np.float32)) == 'FLOAT'


def test_dt_to_td_sqlite_c():
    assert dtype_to_type_decl_sqlite(np.dtype(np.bool_)) == 'BOOLEAN'

def test_dt_to_td_postgres_c():
    assert dtype_to_type_decl_postgresql(np.dtype(np.bool_)) == 'BOOLEAN'

def test_dt_to_td_mysql_c():
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

def test_db_type_c():
    dbt = DBType.MYSQL
    assert dbt.to_placeholder() == '%s'

    tttd = dbt.to_dytpe_to_type_decl()
    assert tttd[np.dtype(np.int64)] == 'BIGINT'

def test_db_type_d():
    dbt = DBType.UNKNOWN
    assert dbt.to_placeholder() == '%s'

    with pytest.raises(NotImplementedError):
        _ = dbt.to_dytpe_to_type_decl()