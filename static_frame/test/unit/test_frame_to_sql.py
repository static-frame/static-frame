import sqlite3

import frame_fixtures as ff
import numpy as np
import pytest

from static_frame.core.frame import Frame
from static_frame.test.test_case import temp_file


def test_frame_to_sql_a():

    with temp_file('.db') as fp:
        conn = sqlite3.connect(fp)

        f = ff.parse('s(2,3)|v(str)').rename('f1', index='x').relabel(columns=('a', 'b', 'c'))
        f.to_sql(conn)
        post = list(conn.cursor().execute('select * from f1'))
        assert post == [(0, 'zjZQ', 'zaji', 'ztsv'), (1, 'zO5l', 'zJnC', 'zUvW')]




def test_frame_to_sql_b():

    with temp_file('.db') as fp:
        conn = sqlite3.connect(fp)

        f = ff.parse('s(2,3)|v(str)').relabel(columns=('a', 'b', 'c'))

        with pytest.raises(RuntimeError):
            f.to_sql(conn)


def test_frame_to_sql_c():


    with temp_file('.db') as fp:
        conn = sqlite3.connect(fp)
        f1 = Frame.from_fields(((10, 2, 8, 3), (False, True, True, False), ('1517-01-01', '1517-04-01', '1517-12-31', '1517-06-30')), columns=('a', 'b', 'c'), dtypes=dict(c=np.datetime64), name='x')

        f1.to_sql(conn, include_index=False)
        post = list(conn.cursor().execute('select * from x'))

        # NOTE: bools converted to ints, dates converted to strings
        assert post == [(10, 0, '1517-01-01'), (2, 1, '1517-04-01'), (8, 1, '1517-12-31'), (3, 0, '1517-06-30')]
