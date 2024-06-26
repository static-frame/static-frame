
from stiatc_frame.core.generic_aliases import TFrameAny


# '''
# WITH
# t1 AS (SELECT ROW_NUMBER() OVER() AS rownum, * FROM a1),
# t2 AS (SELECT ROW_NUMBER() OVER() AS rownum, * FROM a2),
# t3 AS (SELECT ROW_NUMBER() OVER() AS rownum, * FROM a3)
# SELECT t1.column0 AS int_column, t2.column0 AS bool_column, t3.column0 AS str_column
# FROM t1
# JOIN t2 ON t1.rownum = t2.rownum
# JOIN t3 ON t1.rownum = t3.rownum
# '''



def frame_to_table(
        *,
        frame: TFrameAny,
        label: str, # can be None
        connection,
        include_columns: bool,
        include_index: bool,
        ):
    '''
    Args:
        label: string to be used as the table name.
    '''


if __name__ == '__main__':
    import duckdb
    f = ff.parse('s(6,3)|v(int64,str,bool)c(I,str)')
    conn = duckdb.connect()
    import frame_fixtures as ff
    frame_to_table(frame=f, connection=conn)