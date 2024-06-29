from functools import partial

import numpy as np
import typing_extensions as tp

from static_frame.core.container_util import constructor_from_optional_constructors
from static_frame.core.container_util import index_from_optional_constructors
from static_frame.core.generic_aliases import TFrameAny
from static_frame.core.index import Index
from static_frame.core.index_base import IndexBase
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.store import Store
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.util import NAME_DEFAULT
from static_frame.core.util import TIndexCtorSpecifier
from static_frame.core.util import TIndexCtorSpecifiers
from static_frame.core.util import TIndexHierarchyCtor
from static_frame.core.util import TIndexInitializer
from static_frame.core.util import TLabel
from static_frame.core.util import TNDArrayAny

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

class StoreDuckDB(Store):

    @classmethod
    def _frame_to_connection(cls,
            *,
            frame: TFrameAny,
            label: str,
            connection: 'DuckDBPyConnection',
            include_index: bool,
            include_columns: bool,
            ) -> 'DuckDBPyConnection':
        '''
        Args:
            label: string to be used as the table name.
        '''
        field_names, _ = cls.get_field_names_and_dtypes(
                frame=frame,
                include_index=include_index,
                include_index_name=True,
                include_columns=include_columns,
                include_columns_name=False,
                force_brackets=False
                )
        label_arrays = zip(field_names,
                cls.get_column_iterator(frame, include_index=include_index)
                )
        query = [f'CREATE TABLE {label} AS WITH']
        w = []
        s = ['SELECT']

        for i, (label, array) in enumerate(label_arrays):
            # TEMP: waiting on duckdb to support registering arrays
            exec(f'a{i} = array')
            w.append(f't{i} AS (SELECT ROW_NUMBER() OVER() AS rownum, * FROM a{i})')
            s.append(f't{i}.column0 AS "{label}",')

        query.append(', '.join(w))
        s.append('from t0')
        query.extend(s)

        r = range(len(field_names))
        for i, j in zip(r[:-1], r[1:]):
            query.append(f'join t{j} on t{i}.rownum = t{j}.rownum')

        msg = ' '.join(query)
        return connection.execute(msg)

    @classmethod
    def _connection_to_frame(cls,
            *,
            label: str,
            connection: 'DuckDBPyConnection',
            index_depth: int = 0,
            index_constructors: TIndexCtorSpecifiers = None,
            columns_depth: int = 1,
            columns_constructors: TIndexCtorSpecifiers = None,
            name: TLabel = NAME_DEFAULT,
            consolidate_blocks: bool = False,
            ) -> TFrameAny:

        from static_frame.core.frame import Frame

        labels: tp.List[TLabel] = []
        arrays: tp.List[TNDArrayAny] = []
        for l, a in connection.query(
                f'select * from {label}').fetchnumpy().items():
            labels.append(l)
            if a.__class__ is not np.ndarray:
                # assume we have a categorical of strings
                a = a.to_numpy().astype(str)
            a.flags.writeable = False
            arrays.append(a)

        index: tp.Optional[TIndexInitializer]
        index_constructor: TIndexCtorSpecifier

        if index_depth == 0:
            index = None
            index_constructor = None
        elif index_depth == 1:
            index = arrays[0]
            index_name = labels[0]
            arrays = arrays[1:]
            labels = labels[1:]
            index_constructor = constructor_from_optional_constructors(
                    depth=index_depth,
                    default_constructor=Index,
                    explicit_constructors=index_constructors,
                    )
        else:
            index = arrays[:index_depth] # type: ignore
            index_name = tuple(labels[:index_depth])
            arrays = arrays[index_depth:]
            labels = labels[index_depth:]

            def index_default_constructor(values: tp.Iterable[TNDArrayAny],
                    *,
                    index_constructors: TIndexCtorSpecifiers = None,
                    ) -> IndexBase:
                return IndexHierarchy._from_type_blocks(
                    TypeBlocks.from_blocks(values),
                    name=index_name,
                    index_constructors=index_constructors,
                    own_blocks=True,
                    )
            index_constructor = constructor_from_optional_constructors(
                    depth=index_depth,
                    default_constructor=index_default_constructor,
                    explicit_constructors=index_constructors,
                    )

        if columns_depth == 1:
            columns, own_columns = index_from_optional_constructors(
                    labels,
                    depth=columns_depth,
                    default_constructor=Frame._COLUMNS_CONSTRUCTOR,
                    explicit_constructors=columns_constructors, # cannot supply name
                    )
        elif columns_depth > 1:
            # NOTE: we only support loading in IH if encoded in each header with a space delimiter
            columns_constructor: TIndexHierarchyCtor = partial(
                    Frame._COLUMNS_HIERARCHY_CONSTRUCTOR.from_labels_delimited,
                    delimiter=' ',
                    )
            columns, own_columns = index_from_optional_constructors(
                    labels,
                    depth=columns_depth,
                    default_constructor=columns_constructor,
                    explicit_constructors=columns_constructors,
                    )

        if consolidate_blocks:
            arrays = TypeBlocks.consolidate_blocks(arrays) # type: ignore
        tb = TypeBlocks.from_blocks(arrays)

        name = label if name is NAME_DEFAULT else name

        return Frame(tb,
                index=index,
                index_constructor=index_constructor,
                columns=columns,
                own_columns=own_columns,
                own_data=True,
                name=name,
                )



