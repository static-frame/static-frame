import typing_extensions as tp

from static_frame.core.generic_aliases import TFrameAny

if tp.TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


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

def frame_to_connection(
        *,
        frame: TFrameAny,
        # label: str, # can be None
        connection: 'DuckDBPyConnection',
        # include_index: bool,
        ) -> 'DuckDBPyConnection':
    '''
    Args:
        label: string to be used as the table name.
    '''

    query = ['WITH']
    select = []
    for i, array in enumerate(frame._blocks.iter_columns_arrays()):
        exec(f'a{i} = array')
        select.append(f't{i} AS (SELECT ROW_NUMBER() OVER() AS rownum, * FROM a{i})')
    query.append(', '.join(select))

    select = ['SELECT']
    for i, col in enumerate(frame.columns):
        select.append(f't{i}.column0 AS {col},')
    select.append('from t0')
    query.extend(select)

    r = range(frame.shape[1])
    for i, j in zip(r[:-1], r[1:]):
        query.append(f'join t{j} on t{i}.rownum = t{j}.rownum')

    msg = ' '.join(query)
    return connection.execute(msg)
