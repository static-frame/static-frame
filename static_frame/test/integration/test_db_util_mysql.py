import datetime
import subprocess
import time
from functools import partial

import frame_fixtures as ff
import mysql.connector
import numpy as np
import pytest

from static_frame.core.db_util import DBQuery
from static_frame.core.db_util import DBType
from static_frame.core.frame import Frame
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.test.test_case import skip_mac_gha
from static_frame.test.test_case import skip_win
from static_frame.test.test_images import IMAGE_MARIADB
from static_frame.test.test_images import IMAGE_MYSQL

DB_USER = "testuser"
DB_PASSWORD = "testpass" # noqa: S105
DB_NAME = "testdb"

PORT_MYSQL = 3306
PORT_MARIADB = 3307

connect = partial(mysql.connector.connect,
        host="127.0.0.1", # NOTE: cannot use "localhost": force TCP conn
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        )

def wait_for_db(port: int):
    for _ in range(10):
        try:
            conn = connect(port=port)
            conn.close()
            return
        except mysql.connector.Error as e:
            time.sleep(3)
    raise RuntimeError("DB did not become ready in time.")


@pytest.fixture(scope="session", autouse=True)
def start_mysql_container():
    name = "test-mysql"

    cmd = [
        'docker', 'run', '--rm',
        '--name', name,
        '-e', f'MYSQL_USER={DB_USER}',
        '-e', f'MYSQL_PASSWORD={DB_PASSWORD}',
        '-e', f'MYSQL_DATABASE={DB_NAME}',
        '-e', 'MYSQL_ROOT_PASSWORD=rootpass',
        '-p', f'{PORT_MYSQL}:3306',
        '-d',
        IMAGE_MYSQL,
        '--innodb-flush-method=nosync',
        '--skip-innodb-doublewrite',
        ]
    try:
        subprocess.run(cmd, check=True)
        wait_for_db(PORT_MYSQL)
        yield
    finally:
        subprocess.run(["docker", "stop", name], check=True)

@pytest.fixture(scope="session", autouse=True)
def start_mariadb_container():
    name = "test-mariadb"

    cmd = [
        'docker', 'run', '--rm',
        '--name', name,
        '-e', f'MYSQL_USER={DB_USER}',
        '-e', f'MYSQL_PASSWORD={DB_PASSWORD}',
        '-e', f'MYSQL_DATABASE={DB_NAME}',
        '-e', 'MYSQL_ROOT_PASSWORD=rootpass',
        '-p', f'{PORT_MARIADB}:3306',
        '-d',
        IMAGE_MARIADB,
        '--innodb-flush-method=nosync',
        '--skip-innodb-doublewrite',
        ]
    try:
        subprocess.run(cmd, check=True)
        wait_for_db(PORT_MARIADB)
        yield
    finally:
        subprocess.run(["docker", "stop", name], check=True)

@pytest.fixture
def conn_mysql():
    conn = connect(port=PORT_MYSQL)
    yield conn
    conn.close()

@pytest.fixture
def conn_mariadb():
    conn = connect(port=PORT_MARIADB)
    yield conn
    conn.close()

#-------------------------------------------------------------------------------
@skip_win
@skip_mac_gha
def test_dbq_mysql_execuate_a(conn_mysql):
    f = Frame.from_records([('a', 3, False), ('b', -20, True)],
            columns=('x', 'y', 'z'),
            index=IndexHierarchy.from_labels([('p', 100), ('q', 200)], name=('v', 'w')),
            name='f1',
            dtypes=(np.str_, np.int64, np.bool_),
            )

    dbq = DBQuery.from_defaults(conn_mysql)
    assert dbq._db_type == DBType.MYSQL

    dbq.execute(frame=f, label=f.name, include_index=True, scalars=False, eager=True)

    cur = conn_mysql.cursor()
    cur.execute(f'select * from {f.name}')
    post = list(cur)
    assert post == [('p', 100, 'a', 3, 0), ('q', 200, 'b', -20, 1)]

    cur.execute(f'drop table if exists {f.name}')

@skip_win
@skip_mac_gha
def test_dbq_mysql_execuate_b(conn_mysql):
    f = Frame.from_records([('a', 3, False), ('b', 8, True)],
            columns=('x', 'y', 'z'),
            name='f2',
            dtypes=(np.str_, np.uint8, np.bool_),
            )

    dbq = DBQuery.from_defaults(conn_mysql)
    assert dbq._db_type == DBType.MYSQL

    dbq.execute(frame=f, label=f.name, include_index=False, scalars=False, eager=True)

    cur = conn_mysql.cursor()
    cur.execute(f'select * from {f.name}')
    post = list(cur)
    assert post == [('a', 3, 0), ('b', 8, 1)]

    cur.execute(f'drop table if exists {f.name}')

#-------------------------------------------------------------------------------
@skip_win
@skip_mac_gha
def test_dbq_mariadb_execuate_a(conn_mariadb):
    f = Frame.from_records([('a', 3, False), ('b', -20, True)],
            columns=('x', 'y', 'z'),
            index=IndexHierarchy.from_labels([('p', 100), ('q', 200)], name=('v', 'w')),
            name='f1',
            dtypes=(np.str_, np.int64, np.bool_),
            )

    dbq = DBQuery.from_defaults(conn_mariadb)
    assert dbq._db_type == DBType.MARIADB

    dbq.execute(frame=f, label=f.name, include_index=True, scalars=False, eager=True)

    cur = conn_mariadb.cursor()
    cur.execute(f'select * from {f.name}')
    post = list(cur)
    assert post == [('p', 100, 'a', 3, 0), ('q', 200, 'b', -20, 1)]

    cur.execute(f'drop table if exists {f.name}')


#-------------------------------------------------------------------------------
@skip_win
@skip_mac_gha
def test_dbq_mysql_to_sql_a(conn_mysql):

    f = Frame.from_fields(((10, 2, 8, 3), (False, True, True, False), ('1517-01-01', '1517-04-01', '1517-12-31', '1517-06-30')), columns=('a', 'b', 'c'), dtypes=dict(c=np.datetime64), name='x')
    f.to_sql(conn_mysql, include_index=False)

    cur = conn_mysql.cursor()
    cur.execute(f'select * from {f.name}')
    post = list(cur)

    # NOTE: bools converted to int
    assert post == [(10, 0, datetime.date(1517, 1, 1)), (2, 1, datetime.date(1517, 4, 1)), (8, 1, datetime.date(1517, 12, 31)), (3, 0, datetime.date(1517, 6, 30))]

    cur.execute(f'drop table if exists {f.name}')


@skip_win
@skip_mac_gha
def test_dbq_mysql_to_sql_b(conn_mysql):

    f = ff.parse('s(3,6)|v(int32, uint8, int64, float, str, bool)').rename('f1', index='x').relabel(columns=('a', 'b', 'c', 'd', 'e', 'f'))
    f.to_sql(conn_mysql, include_index=False)

    cur = conn_mysql.cursor()
    cur.execute(f'select * from {f.name}')

    post = list(cur)
    assert post == [(-88017, 150, -3648, 1080.4, 'zDVQ', 0), (92867, 250, 91301, 2580.34, 'z5hI', 1), (84967, 100, 30205, 700.42, 'zyT8', 0)]

    f.to_sql(conn_mysql, include_index=False)

    cur = conn_mysql.cursor()
    cur.execute(f'select * from {f.name}')

    post = list(cur)
    assert post == [(-88017, 150, -3648, 1080.4, 'zDVQ', 0), (92867, 250, 91301, 2580.34, 'z5hI', 1), (84967, 100, 30205, 700.42, 'zyT8', 0), (-88017, 150, -3648, 1080.4, 'zDVQ', 0), (92867, 250, 91301, 2580.34, 'z5hI', 1), (84967, 100, 30205, 700.42, 'zyT8', 0)]

    cur.execute(f'drop table if exists {f.name}')

#-------------------------------------------------------------------------------

@skip_win
@skip_mac_gha
def test_from_sql_a(conn_mysql):
    f1 = Frame.from_records([('a', 3, False), ('b', 8, True)],
            columns=('x', 'y', 'z'),
            name='f1',
            dtypes=(np.str_, np.uint8, np.bool_),
            )
    f1.to_sql(conn_mysql)
    dbt = DBType.from_connection(conn_mysql)

    # f2 = Frame.from_sql('select * from f1', connection=conn_mysql, index_depth=1)
    # assert f1.equals(f2)
    cur = conn_mysql.cursor()
    cur.execute('select * from f1')

    post = dict(dbt.cursor_to_dtypes(cur))
    assert post == {'__index0__': np.dtype('int64'), 'x': np.dtype('<U'), 'y': np.dtype('int16'), 'z': np.dtype('int8')}
    _ = list(cur)
    cur.execute(f'drop table if exists {f1.name}')


@skip_win
@skip_mac_gha
def test_from_sql_b(conn_mysql):
    f1 = Frame.from_records([('a', 3.3, 3), ('b', 8.2, 4)],
            columns=('x', 'y', 'z'),
            name='f1',
            dtypes=(np.str_, np.float64, np.int16),
            )
    f1.to_sql(conn_mysql)

    dbt = DBType.from_connection(conn_mysql)
    # f2 = Frame.from_sql('select * from f1', connection=conn_mysql, index_depth=1)
    # assert f1.equals(f2)
    cur = conn_mysql.cursor()
    cur.execute(f'select * from f1')

    post = dict(dbt.cursor_to_dtypes(cur))
    assert post == {'__index0__': np.dtype('int64'), 'x': np.dtype('<U'), 'y': np.dtype('float64'), 'z': np.dtype('int16')}
    _ = list(cur)
    cur.execute(f'drop table if exists {f1.name}')

