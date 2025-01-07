import subprocess
import time
from functools import partial

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

# MYSQL_IMAGE = "mysql:8.0" # "mariadb:10.6"
MYSQL_USER = "testuser"
MYSQL_PASSWORD = "testpass"
MYSQL_DB = "testdb"
MYSQL_PORT = 3306
MARIADB_PORT = 3307

MAX_RETRIES = 10
RETRY_DELAY = 3  # seconds

connect = partial(mysql.connector.connect,
        host="127.0.0.1", # NOTE: cannot use "localhost": force TCP conn
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        )

def wait_for_db(port: int):
    for _ in range(MAX_RETRIES):
        try:
            conn = connect(port=port)
            conn.close()
            return
        except mysql.connector.Error as e:
            time.sleep(RETRY_DELAY)
    raise RuntimeError("DB did not become ready in time.")


@pytest.fixture(scope="session", autouse=True)
def start_mysql_container():
    name = "test-mysql"

    cmd = [
        'docker', 'run', '--rm',
        '--name', name,
        '-e', f'MYSQL_USER={MYSQL_USER}',
        '-e', f'MYSQL_PASSWORD={MYSQL_PASSWORD}',
        '-e', f'MYSQL_DATABASE={MYSQL_DB}',
        '-e', 'MYSQL_ROOT_PASSWORD=rootpass',
        '-p', f'{MYSQL_PORT}:3306',
        '-d',
        IMAGE_MYSQL,
        '--innodb-flush-method=nosync',
        '--skip-innodb-doublewrite',
        ]
    try:
        subprocess.run(cmd, check=True)
        wait_for_db(MYSQL_PORT)
        yield
    finally:
        subprocess.run(["docker", "stop", name], check=True)

@pytest.fixture(scope="session", autouse=True)
def start_mariadb_container():
    name = "test-mariadb"

    cmd = [
        'docker', 'run', '--rm',
        '--name', name,
        '-e', f'MYSQL_USER={MYSQL_USER}',
        '-e', f'MYSQL_PASSWORD={MYSQL_PASSWORD}',
        '-e', f'MYSQL_DATABASE={MYSQL_DB}',
        '-e', 'MYSQL_ROOT_PASSWORD=rootpass',
        '-p', f'{MARIADB_PORT}:3306',
        '-d',
        IMAGE_MARIADB,
        '--innodb-flush-method=nosync',
        '--skip-innodb-doublewrite',
        ]
    try:
        subprocess.run(cmd, check=True)
        wait_for_db(MARIADB_PORT)
        yield
    finally:
        subprocess.run(["docker", "stop", name], check=True)

@pytest.fixture
def conn_mysql():
    conn = connect(port=MYSQL_PORT)
    yield conn
    conn.close()

@pytest.fixture
def conn_mariadb():
    conn = connect(port=MARIADB_PORT)
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



