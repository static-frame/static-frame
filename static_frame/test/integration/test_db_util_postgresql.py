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

POSTGRES_CONTAINER_NAME = 'test-postgres'
POSTGRES_IMAGE = 'postgres:14'
POSTGRES_USER = 'postgres'
POSTGRES_PASSWORD = 'secret'
POSTGRES_DB = 'postgres'
POSTGRES_PORT = '15432'
MAX_RETRIES = 10
RETRY_DELAY = 1  # seconds

connect = partial(psycopg2.connect,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host="localhost",
        port=POSTGRES_PORT
        )

# NOTE: on MacOS need to run `brew install --cask docker` first, and then run `open /Applications/Docker.app` to configure Docker Desktop. Using docker in this manner does not seem to be supported on GitHub Action MacOS runners.
# NOTE: on Linux, the follow is necessary on dev systems: (1) `sudo apt install docker-compose` (2) `sudo usermod -aG docker $USER; newgrp docker`. Using docker in GitHub Action Linux runners requires no configuration.

def wait_for_postgres():
    for _ in range(MAX_RETRIES):
        try:
            conn = connect()
            conn.close()
            return
        except psycopg2.OperationalError:
            time.sleep(RETRY_DELAY)
    raise RuntimeError("PostgreSQL did not become ready in time.")

@pytest.fixture(scope='session', autouse=True)
def start_postgres_container():
    cmd = [
        'docker', 'run', '--rm', '--name', POSTGRES_CONTAINER_NAME,
        '-e', f'POSTGRES_USER={POSTGRES_USER}',
        '-e', f'POSTGRES_PASSWORD={POSTGRES_PASSWORD}',
        '-e', f'POSTGRES_DB={POSTGRES_DB}',
        '-p', f'{POSTGRES_PORT}:5432',
        '-d',
        POSTGRES_IMAGE
        ]
    try:
        subprocess.run(cmd, check=True)
        wait_for_postgres()
        yield  # run tests
    finally:
        subprocess.run(['docker', 'stop', POSTGRES_CONTAINER_NAME], check=True)


@pytest.fixture
def db_conn():
    conn = connect()
    yield conn
    conn.close()

@skip_win
@skip_mac_gha
def test_create_and_insert(db_conn):
    f = Frame.from_records([('a', 3, False), ('b', -20, True)],
            columns=('x', 'y', 'z'),
            index=IndexHierarchy.from_labels([('p', 100), ('q', 200)], name=('v', 'w')),
            name='foo',
            dtypes=(np.str_, np.int64, np.bool_),
            )

    dbq = DBQuery.from_defaults(db_conn)
    assert dbq._db_type == DBType.POSTGRESQL

    dbq.execute(frame=f, label=f.name, include_index=True, scalars=False)

    cur = db_conn.cursor()
    cur.execute(f'select * from {f.name}')
    post = list(cur)
    assert post == [('p', 100, 'a', 3, 0), ('q', 200, 'b', -20, 1)]

