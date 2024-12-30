import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

import frame_fixtures as ff

from static_frame.core.frame import Frame


def test_frame_to_sql_a():

    with TemporaryDirectory() as fp_dir:
        fp = Path(fp_dir) / 'temp.db'
        conn = sqlite3.connect(fp)

        f = ff.parse('s(2,3)|v(str)').rename('f1', index='x').relabel(columns=('a', 'b', 'c'))
        f.to_sql(conn)
        post = list(conn.cursor().execute('select * from f1'))
        assert post == [(0, 'zjZQ', 'zaji', 'ztsv'), (1, 'zO5l', 'zJnC', 'zUvW')]
