import subprocess
import time
import pytest
import psycopg2

POSTGRES_CONTAINER_NAME = 'test-postgres'
POSTGRES_IMAGE = 'postgres:14'
POSTGRES_USER = 'postgres'
POSTGRES_PASSWORD = 'secret'
POSTGRES_DB = 'postgres'
POSTGRES_PORT = '15432'

# NOTE: no MacOS need to run `brew install --cask docker` first, and then run `open /Applications/Docker.app` to configure Docker Desktop.

@pytest.fixture(scope='session', autouse=True)
def start_postgres_container():
    cmd = [
        'docker', 'run', '--rm', '--name', POSTGRES_CONTAINER_NAME,
        '-e', f'POSTGRES_USER={POSTGRES_USER}',
        '-e', f'POSTGRES_PASSWORD={POSTGRES_PASSWORD}',
        '-e', f'POSTGRES_DB={POSTGRES_DB}',
        '-p', f'{POSTGRES_PORT}:5432',
        '-d', POSTGRES_IMAGE
        ]
    try:
        subprocess.run(cmd, check=True)
        # Wait for PostgreSQL replaceable with a health check
        time.sleep(5)
        yield  # run tests
    finally:
        subprocess.run(['docker', 'stop', POSTGRES_CONTAINER_NAME], check=True)


@pytest.fixture
def db_connection():
    """Provide a connection to the test database."""
    conn = psycopg2.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host="localhost",
        port=POSTGRES_PORT
    )
    yield conn
    conn.close()


def test_create_and_insert(db_connection):
    with db_connection.cursor() as cur:
        cur.execute("CREATE TABLE test_table (id SERIAL PRIMARY KEY, name TEXT NOT NULL)")
        cur.execute("INSERT INTO test_table (name) VALUES (%s)", ("test_name",))
        cur.execute("SELECT COUNT(*) FROM test_table")
        assert cur.fetchone()[0] == 1


