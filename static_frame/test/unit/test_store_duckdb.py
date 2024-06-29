import frame_fixtures as ff

from static_frame.core.frame import Frame
from static_frame.core.store_duckdb import StoreDuckDB

# import numpy as np

def test_store_duckd_a():

    import duckdb

    f1 = ff.parse('s(6,3)|v(int64,str,bool)|c(I,str)')
    conn = duckdb.connect()
    post = StoreDuckDB._frame_to_connection(frame=f1,
            label='foo', connection=conn, include_index=False, include_columns=True)

    f2 = Frame.from_pandas(post.query('select * from foo').df())
    assert (f2.to_pairs() ==
            (('zZbu', ((0, -88017), (1, 92867), (2, 84967), (3, 13448), (4, 175579), (5, 58768))), ('ztsv', ((0, 'zaji'), (1, 'zJnC'), (2, 'zDdR'), (3, 'zuVU'), (4, 'zKka'), (5, 'zJXD'))), ('zUvW', ((0, True), (1, False), (2, False), (3, True), (4, False), (5, False))))
            )

    f3 = StoreDuckDB._connection_to_frame(connection=conn, label='foo')
    assert f3.equals(f1, compare_name=False, compare_dtype=True, compare_class=True)


def test_store_duckd_b():

    import duckdb

    f1 = ff.parse('s(6,3)|v(float64)|c(I,str)')
    conn = duckdb.connect()
    post = StoreDuckDB._frame_to_connection(frame=f1,
            label='foo',
            connection=conn,
            include_index=False,
            include_columns=True,
            )
    f2 = StoreDuckDB._connection_to_frame(
            connection=conn,
            label='foo',
            consolidate_blocks=True,
            )
    assert f2._blocks.unified == True
    assert f2.name == 'foo'


def test_store_duckd_c():

    import duckdb

    f1 = ff.parse('s(6,3)|v(int64)|i(I,str)|c(I,str)')
    conn = duckdb.connect()
    post = StoreDuckDB._frame_to_connection(frame=f1,
            label='foo',
            connection=conn,
            include_index=True,
            include_columns=True,
            )
    f2 = StoreDuckDB._connection_to_frame(
            connection=conn,
            label='foo',
            index_depth=1,
            )
    f1.equals(f2, compare_name=False, compare_dtype=True, compare_class=True)


def test_store_duckd_d():

    import duckdb

    f1 = ff.parse('s(6,3)|v(int64)|i(I,str)')
    f1 = f1.rename(index='a')
    f1 = f1.relabel(columns=('b', 'c', 'd'))
    conn = duckdb.connect()
    post = StoreDuckDB._frame_to_connection(frame=f1,
            label='foo',
            connection=conn,
            include_index=True,
            include_columns=True,
            )
    f2 = StoreDuckDB._connection_to_frame(
            connection=conn,
            label='foo',
            index_depth=0,
            )
    assert f1.columns.values.tolist(), ['a', 'b', 'c', 'd']


def test_store_duckd_e():

    import duckdb

    f1 = ff.parse('s(6,3)|v(int64)|i((I,I),(str,str))')
    f1 = f1.rename('foo', index=('a', 'b'))
    f1 = f1.relabel(columns=('c', 'd', 'e'))
    conn = duckdb.connect()
    post = StoreDuckDB._frame_to_connection(frame=f1,
            label='foo',
            connection=conn,
            include_index=True,
            include_columns=True,
            )
    f2 = StoreDuckDB._connection_to_frame(
            connection=conn,
            label='foo',
            index_depth=2,
            )
    f1.equals(f2, compare_name=True, compare_dtype=True, compare_class=True)


def test_store_duckd_f():

    import duckdb

    f1 = ff.parse('s(6,3)|v(int64)|i((I,I),(str,str))|c((I,I),(str,str))')
    f1 = f1.rename(index=('a', 'b'))
    conn = duckdb.connect()
    post = StoreDuckDB._frame_to_connection(frame=f1,
            label='foo',
            connection=conn,
            include_index=True,
            include_columns=True,
            )
    f2 = StoreDuckDB._connection_to_frame(
            connection=conn,
            label='foo',
            index_depth=2,
            columns_depth=2,
            )
    f1.equals(f2, compare_name=True, compare_dtype=True, compare_class=True)

