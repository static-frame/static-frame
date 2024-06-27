import numpy as np
import frame_fixtures as ff

from static_frame.core.store_duckdb import frame_to_connection



def test_store_duckd_a():

    import duckdb

    f1 = ff.parse('s(6,3)|v(int64,str,bool)|c(I,str)')
    conn = duckdb.connect()
    post = frame_to_connection(frame=f1, connection=conn)
    # f2 = f.Frame.from_pandas(post.df())

    # import ipdb; ipdb.set_trace()