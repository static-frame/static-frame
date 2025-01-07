import datetime
import subprocess
import time
from functools import partial

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
DB_PASSWORD = "testpass"
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
