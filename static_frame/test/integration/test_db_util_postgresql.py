import datetime
import subprocess
import time
from functools import partial

import frame_fixtures as ff
import numpy as np
import psycopg2
import pytest

from static_frame.core.db_util import DBQuery
from static_frame.core.db_util import DBType
from static_frame.core.frame import Frame
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.test.test_case import skip_mac_gha
from static_frame.test.test_case import skip_win
from static_frame.test.test_images import IMAGE_POSTGRESQL

DB_USER = "testuser"
DB_PASSWORD = "testpass" # noqa: S105
DB_NAME = "testdb"

DB_PORT = '15432'

connect = partial(psycopg2.connect,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host="localhost",
        port=DB_PORT
        )

# NOTE: on MacOS need to run `brew install --cask docker` first, and then run `open /Applications/Docker.app` to configure Docker Desktop. Using docker in this manner does not seem to be supported on GitHub Action MacOS runners.
# NOTE: on Linux, the follow is necessary on dev systems: (1) `sudo apt install docker-compose` (2) `sudo usermod -aG docker $USER; newgrp docker`. Using docker in GitHub Action Linux runners requires no configuration.

def wait_for_db():
    for _ in range(10):
        try:
            conn = connect()
            conn.close()
            return
        except psycopg2.OperationalError:
            time.sleep(1)
    raise RuntimeError("PostgreSQL did not become ready in time.")

@pytest.fixture(scope='session', autouse=True)
def start_postgres_container():
    name = 'test-postgres'
    cmd = [
        'docker', 'run', '--rm', '--name', name,
        '-e', f'POSTGRES_USER={DB_USER}',
        '-e', f'POSTGRES_PASSWORD={DB_PASSWORD}',
        '-e', f'POSTGRES_DB={DB_NAME}',
        '-p', f'{DB_PORT}:5432',
        '-d',
        IMAGE_POSTGRESQL
        ]
    try:
        subprocess.run(cmd, check=True)
        wait_for_db()
        yield  # run tests
    finally:
        subprocess.run(['docker', 'stop', name], check=True)

@pytest.fixture
def db_conn():
    conn = connect()
    yield conn
    conn.close()

#-------------------------------------------------------------------------------
@skip_win
@skip_mac_gha
def test_dbq_postgres_execuate_a(db_conn):
    f = Frame.from_records([('a', 3, False), ('b', -20, True)],
            columns=('x', 'y', 'z'),
            index=IndexHierarchy.from_labels([('p', 100), ('q', 200)], name=('v', 'w')),
            name='f1',
            dtypes=(np.str_, np.int64, np.bool_),
            )

    dbq = DBQuery.from_defaults(db_conn)
    assert dbq._db_type == DBType.POSTGRESQL

    dbq.execute(frame=f, label=f.name, include_index=True, scalars=False)

    cur = db_conn.cursor()
    cur.execute(f'select * from {f.name}')
    post = list(cur)
    assert post == [('p', 100, 'a', 3, 0), ('q', 200, 'b', -20, 1)]

    cur.execute(f'drop table if exists {f.name}')


@skip_win
@skip_mac_gha
def test_dbq_postgres_execuate_b(db_conn):
    f = Frame.from_records([('a', 3, False), ('b', -20, True)],
            columns=('x', 'y', 'z'),
            name='f2',
            dtypes=(np.str_, np.int8, np.bool_),
            )

    dbq = DBQuery.from_defaults(db_conn)
    assert dbq._db_type == DBType.POSTGRESQL

    dbq.execute(frame=f, label=f.name, include_index=False, scalars=False)
    dbq.execute(frame=f, label=f.name, include_index=False, scalars=False)
    dbq.execute(frame=f, label=f.name, include_index=False, scalars=False)

    cur = db_conn.cursor()
    cur.execute(f'select * from {f.name}')
    post = list(cur)
    assert post == [('a', 3, False), ('b', -20, True), ('a', 3, False), ('b', -20, True), ('a', 3, False), ('b', -20, True)]

    cur.execute(f'drop table if exists {f.name}')


#-------------------------------------------------------------------------------
@skip_win
@skip_mac_gha
def test_dbq_postgres_to_sql_a(db_conn):

    f = Frame.from_fields(((10, 2, 8, 3), (False, True, True, False), ('1517-01-01', '1517-04-01', '1517-12-31', '1517-06-30')), columns=('a', 'b', 'c'), dtypes=dict(c=np.datetime64), name='x')
    f.to_sql(db_conn, include_index=False)

    cur = db_conn.cursor()
    cur.execute(f'select * from {f.name}')
    post = list(cur)
    assert post == [(10, False, datetime.date(1517, 1, 1)), (2, True, datetime.date(1517, 4, 1)), (8, True, datetime.date(1517, 12, 31)), (3, False, datetime.date(1517, 6, 30))]

    cur.execute(f'drop table if exists {f.name}')



@skip_win
@skip_mac_gha
def test_dbq_postgres_to_sql_b(db_conn):

    f = ff.parse('s(3,6)|v(int32, uint8, int64, float, str, bool)').rename('f1', index='x').relabel(columns=('a', 'b', 'c', 'd', 'e', 'f'))
    f.to_sql(db_conn, include_index=False)

    cur = db_conn.cursor()
    cur.execute(f'select * from {f.name}')
    post = list(cur)
    assert list(f.iter_tuple(axis=1, constructor=tuple)) == post

    cur.execute(f'drop table if exists {f.name}')


@skip_win
@skip_mac_gha
def test_dbq_postgres_to_sql_c(db_conn):

    f = ff.parse('s(2,3)|v(int)').rename('f1', index='x').relabel(columns=('c', 'd', 'e'))

    cur = db_conn.cursor()
    cur.execute(f'create table {f.name} (a INTEGER, b INTEGER, c INTEGER, d INTEGER, e INTEGER)')

    f.to_sql(db_conn, include_index=False)

    cur = db_conn.cursor()
    cur.execute(f'select * from {f.name}')
    post = list(cur)
    assert post == [(None, None, -88017, 162197, -3648), (None, None, 92867, -41157, 91301)]


@skip_win
@skip_mac_gha
def test_dbq_postgres_to_sql_d(db_conn):

    f = ff.parse('s(2,3)|v(int)').rename('f1', index='x').relabel(columns=('c', 'd', 'e'))

    cur = db_conn.cursor()
    cur.execute(f'create table {f.name} (a INTEGER, b INTEGER, c INTEGER)')

    with pytest.raises(psycopg2.errors.UndefinedColumn):
        f.to_sql(db_conn, include_index=False)

@skip_win
@skip_mac_gha
def test_dbq_postgres_to_sql_e(db_conn):

    f = ff.parse('s(2,3)|v(int)').rename('f1', index='x').relabel(columns=('a', 'b', 'c'))

    cur = db_conn.cursor()
    cur.execute(f'create table {f.name} (x SERIAL PRIMARY KEY, a INTEGER, b INTEGER, c INTEGER)')
    f.to_sql(db_conn, include_index=False)

    cur = db_conn.cursor()
    cur.execute(f'select * from {f.name}')
    post = list(cur)
    assert post == [(1, -88017, 162197, -3648), (2, 92867, -41157, 91301)]


@skip_win
@skip_mac_gha
def test_dbq_postgres_to_sql_f(db_conn):

    f = ff.parse('s(2,3)|v(int)').rename('f1', index='x').relabel(columns=('a', 'b', 'c'))

    cur = db_conn.cursor()
    cur.execute(f'create table {f.name} (a INTEGER, b INTEGER, c INTEGER)')
    f.to_sql(db_conn, include_index=False, schema='public')

    cur = db_conn.cursor()
    cur.execute(f'select * from {f.name}')
    post = list(cur)
    assert post == [(-88017, 162197, -3648), (92867, -41157, 91301)]

    with pytest.raises(psycopg2.errors.InvalidSchemaName):
        f.to_sql(db_conn, include_index=False, schema='foo')


#-------------------------------------------------------------------------------
# adhoc test to show this works with SQLAlchemy.

# @skip_win
# @skip_mac_gha
# def test_dbq_sqlalchemy_to_sql_a(db_conn):
#     from sqlalchemy import create_engine, text

#     f = Frame.from_fields(((10, 2, 8, 3), (False, True, True, False), ('1517-01-01', '1517-04-01', '1517-12-31', '1517-06-30')), columns=('a', 'b', 'c'), dtypes=dict(c=np.datetime64), name='x')
#     f.to_sql(db_conn, include_index=False)

#     engine = create_engine(
#         "postgresql+psycopg2://",
#         creator=lambda: db_conn,
#         )

#     with engine.connect() as conn:
#         f.to_sql(conn.connection, include_index=False)

#         post = list(conn.execute(text(f'select * from {f.name}')))
#         assert post == [(10, False, datetime.date(1517, 1, 1)), (2, True, datetime.date(1517, 4, 1)), (8, True, datetime.date(1517, 12, 31)), (3, False, datetime.date(1517, 6, 30))]

#-------------------------------------------------------------------------------

@skip_win
@skip_mac_gha
def test_from_sql_a(db_conn):
    f1 = Frame.from_records([('a', 3, False), ('b', 8, True)],
            columns=('x', 'y', 'z'),
            name='f1',
            dtypes=(np.str_, np.uint8, np.bool_),
            )
    f1.to_sql(db_conn)

    cur = db_conn.cursor()
    cur.execute('select * from f1')

    dbt = DBType.from_connection(db_conn)
    post = dict(dbt.cursor_to_dtypes(cur))
    assert post == {'__index0__': np.dtype('int64'), 'x': np.dtype('<U'), 'y': np.dtype('int16'), 'z': np.dtype('bool')}
    cur.execute(f'drop table if exists {f1.name}')

@skip_win
@skip_mac_gha
def test_from_sql_b(db_conn):
    f1 = Frame.from_records([('a', 3.3, 3), ('b', 8.2, 4)],
            columns=('x', 'y', 'z'),
            name='f1',
            dtypes=(np.str_, np.float64, np.int16),
            )
    f1.to_sql(db_conn)

    cur = db_conn.cursor()
    cur.execute('select * from f1')

    dbt = DBType.from_connection(db_conn)
    post = dict(dbt.cursor_to_dtypes(cur))
    assert post == {'__index0__': np.dtype('int64'), 'x': np.dtype('<U'), 'y': np.dtype('float64'), 'z': np.dtype('int16')}
    cur.execute(f'drop table if exists {f1.name}')
