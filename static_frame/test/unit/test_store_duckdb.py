import os
from tempfile import TemporaryDirectory

import frame_fixtures as ff

from static_frame.core.frame import Frame
from static_frame.core.store_config import StoreConfig
from static_frame.core.store_duckdb import StoreDuckDB

# from static_frame.test.test_case import temp_file

# import numpy as np

def test_store_duckdb_a():

    import duckdb

    f1 = ff.parse('s(6,3)|v(int64,str,bool)|c(I,str)')
    conn = duckdb.connect()
    post = StoreDuckDB._frame_to_connection(frame=f1,
            label='foo', connection=conn, include_index=False, include_columns=True)

    f2 = Frame.from_pandas(post.query('select * from foo').df())
    assert (f2.to_pairs() ==
            (('zZbu', ((0, -88017), (1, 92867), (2, 84967), (3, 13448), (4, 175579), (5, 58768))), ('ztsv', ((0, 'zaji'), (1, 'zJnC'), (2, 'zDdR'), (3, 'zuVU'), (4, 'zKka'), (5, 'zJXD'))), ('zUvW', ((0, True), (1, False), (2, False), (3, True), (4, False), (5, False))))
            )

    f3 = StoreDuckDB._connection_to_frame(container_type=Frame, connection=conn, label='foo')
    assert f3.equals(f1, compare_name=False, compare_dtype=False, compare_class=True)


def test_store_duckdb_b():

    import duckdb

    f1 = ff.parse('s(6,3)|v(float64)|c(I,str)')
    conn = duckdb.connect()
    post = StoreDuckDB._frame_to_connection(frame=f1,
            label='foo',
            connection=conn,
            include_index=False,
            include_columns=True,
            )
    f2 = StoreDuckDB._connection_to_frame(container_type=Frame,
            connection=conn,
            label='foo',
            consolidate_blocks=True,
            )
    assert f2._blocks.unified
    assert f2.name == 'foo'


def test_store_duckdb_c():

    import duckdb

    f1 = ff.parse('s(6,3)|v(int64)|i(I,str)|c(I,str)')
    conn = duckdb.connect()
    post = StoreDuckDB._frame_to_connection(frame=f1,
            label='foo',
            connection=conn,
            include_index=True,
            include_columns=True,
            )
    f2 = StoreDuckDB._connection_to_frame(container_type=Frame,
            connection=conn,
            label='foo',
            index_depth=1,
            )
    f1.equals(f2, compare_name=False, compare_dtype=True, compare_class=True)


def test_store_duckdb_d():

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
    f2 = StoreDuckDB._connection_to_frame(container_type=Frame,
            connection=conn,
            label='foo',
            index_depth=0,
            )
    assert f1.columns.values.tolist(), ['a', 'b', 'c', 'd']


def test_store_duckdb_e():

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
    f2 = StoreDuckDB._connection_to_frame(container_type=Frame,
            connection=conn,
            label='foo',
            index_depth=2,
            )
    f1.equals(f2, compare_name=True, compare_dtype=True, compare_class=True)


def test_store_duckdb_f():

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
    f2 = StoreDuckDB._connection_to_frame(container_type=Frame,
            connection=conn,
            label='foo',
            index_depth=2,
            columns_depth=2,
            )
    f1.equals(f2, compare_name=True, compare_dtype=True, compare_class=True)


def test_store_duckdb_labels_a():
    import duckdb

    # NOTE: normal temp file generation is not working
    with TemporaryDirectory() as fp_dir:
        fp = os.path.join(fp_dir, 'test.db')

        conn = duckdb.connect(fp)
        f1 = ff.parse('s(6,3)|v(int64)|i(I,str)|c(I,str)')
        _ = StoreDuckDB._frame_to_connection(frame=f1,
            label='foo',
            connection=conn,
            include_index=True,
            include_columns=True,
            )
        _ = StoreDuckDB._frame_to_connection(frame=f1,
            label='bar',
            connection=conn,
            include_index=True,
            include_columns=True,
            )
        conn.close()
        st = StoreDuckDB(fp)
        assert list(st.labels()) == ['bar', 'foo']



def test_store_duckdb_write_a():
    import duckdb

    f1 = ff.parse('s(6,3)|v(int64)|i(I,str)|c(I,str)')
    f2 = ff.parse('s(4,5)|v(float64)|i(I,str)|c(I,str)')

    config = StoreConfig.from_frame(f1)

    # NOTE: normal temp file generation is not working
    with TemporaryDirectory() as fp_dir:
        fp = os.path.join(fp_dir, 'test.db')
        st = StoreDuckDB(fp)
        st.write((('a', f1), ('b', f2)), config=config)
        assert list(st.labels()) == ['a', 'b']

        post = list(st.read_many(('a', 'b'), config=config))
        assert post[0].equals(f1, compare_dtype=False)
        assert post[1].equals(f2, compare_dtype=False)



def test_store_duckdb_read_a():
    import duckdb

    f1 = ff.parse('s(6,3)|v(int64)|i(I,str)|c(I,str)')
    f2 = ff.parse('s(4,5)|v(float64)|i(I,str)|c(I,str)')

    config = StoreConfig.from_frame(f1)

    # NOTE: normal temp file generation is not working
    with TemporaryDirectory() as fp_dir:
        fp = os.path.join(fp_dir, 'test.db')
        st = StoreDuckDB(fp)
        st.write((('a', f1), ('b', f2)), config=config)

        f3 = st.read('b', config=config)
        assert f3.name == 'b'
        assert f3.shape == (4, 5)

        f4 = st.read('a', config=config)
        assert (f4.to_pairs() ==
                (('zZbu', (('zZbu', -88017), ('ztsv', 92867), ('zUvW', 84967), ('zkuW', 13448), ('zmVj', 175579), ('z2Oo', 58768))), ('ztsv', (('zZbu', 162197), ('ztsv', -41157), ('zUvW', 5729), ('zkuW', -168387), ('zmVj', 140627), ('z2Oo', 66269))), ('zUvW', (('zZbu', -3648), ('ztsv', 91301), ('zUvW', 30205), ('zkuW', 54020), ('zmVj', 129017), ('z2Oo', 35021)))))

