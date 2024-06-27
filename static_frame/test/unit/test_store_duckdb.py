import frame_fixtures as ff
import numpy as np

from static_frame.core.frame import Frame
from static_frame.core.store_duckdb import frame_to_connection


def test_store_duckd_a():

    import duckdb

    f1 = ff.parse('s(6,3)|v(int64,str,bool)|c(I,str)')
    conn = duckdb.connect()
    post = frame_to_connection(frame=f1, connection=conn)
    f2 = Frame.from_pandas(post.df())
    assert (f2.to_pairs() ==
            (('zZbu', ((0, -88017), (1, 92867), (2, 84967), (3, 13448), (4, 175579), (5, 58768))), ('ztsv', ((0, 'zaji'), (1, 'zJnC'), (2, 'zDdR'), (3, 'zuVU'), (4, 'zKka'), (5, 'zJXD'))), ('zUvW', ((0, True), (1, False), (2, False), (3, True), (4, False), (5, False))))
            )
