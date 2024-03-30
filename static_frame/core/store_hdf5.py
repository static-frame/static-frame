from __future__ import annotations

import numpy as np
import typing_extensions as tp

# from static_frame.core.doc_str import doc_inject
from static_frame.core.frame import Frame
from static_frame.core.store import Store
from static_frame.core.store import store_coherent_non_write
from static_frame.core.store import store_coherent_write
from static_frame.core.store_config import StoreConfigMap
from static_frame.core.store_config import StoreConfigMapInitializer
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.util import DTYPE_STR_KINDS
from static_frame.core.util import TLabel
from static_frame.core.util import WarningsSilent

if tp.TYPE_CHECKING:
    TNDArrayAny = np.ndarray[tp.Any, tp.Any] #pragma: no cover
TFrameAny = Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]]

class StoreHDF5(Store):

    _EXT: tp.FrozenSet[str] =  frozenset(('.h5', '.hdf5'))

    @store_coherent_write
    def write(self,
            items: tp.Iterable[tp.Tuple[TLabel, TFrameAny]],
            *,
            config: StoreConfigMapInitializer = None,
            # store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ) -> None:

        config_map = StoreConfigMap.from_initializer(config)

        with WarningsSilent():
            import tables

            # silence: DeprecationWarning: `np.typeDict` is a deprecated alias for `np.sctypeDict`.

        with tables.open_file(self._fp, mode='w') as file, WarningsSilent():
            # silence NaturalNameWarning: object name is not a valid Python identifier:

            for label, frame in items:
                c = config_map[label]
                label = config_map.default.label_encode(label)

                # should all tables be under a common group?
                field_names, dtypes = self.get_field_names_and_dtypes(
                        frame=frame,
                        include_index=c.include_index,
                        include_index_name=True,
                        include_columns=c.include_columns,
                        include_columns_name=False,
                        )

                # Must set pos to have stable position
                description = {}
                for i, (k, v) in enumerate(zip(field_names, dtypes)):
                    if v == object:
                        raise RuntimeError('cannot store object dtypes in HDF5')
                    description[k] = tables.Col.from_dtype(v, pos=i)

                table = file.create_table('/', # create off root from sring
                        name=label,
                        description=description,
                        expectedrows=len(frame),
                        )

                values = self._get_row_iterator(frame=frame,
                        include_index=c.include_index)
                table.append(tuple(values()))
                table.flush()

    @store_coherent_non_write
    def read_many(self,
            labels: tp.Iterable[TLabel],
            *,
            config: StoreConfigMapInitializer = None,
            container_type: tp.Type[TFrameAny] = Frame,
            ) -> tp.Iterator[TFrameAny]:
        import tables
        config_map = StoreConfigMap.from_initializer(config)

        with tables.open_file(self._fp, mode='r') as file:
            for label in labels:
                c = config_map[label]
                label_encoded = config_map.default.label_encode(label)

                index_depth = c.index_depth
                index_constructors = c.index_constructors
                columns_depth = c.columns_depth
                columns_constructors = c.columns_constructors
                consolidate_blocks = c.consolidate_blocks
                if c.dtypes:
                    raise NotImplementedError('using config.dtypes on HDF5 not yet supported')

                index_arrays = []
                columns_labels = []

                table = file.get_node(f'/{label_encoded}')
                colnames = table.cols._v_colnames # pyright: ignore

                def blocks() -> tp.Iterator[TNDArrayAny]:
                    for col_idx, colname in enumerate(colnames):
                        # can also do: table.read(field=colname)
                        array = table.col(colname) # pyright: ignore
                        if array.dtype.kind in DTYPE_STR_KINDS:
                            array = array.astype(str)
                        array.flags.writeable = False

                        if col_idx < index_depth:
                            index_arrays.append(array)
                            continue
                        # only store column labels for those yielded
                        columns_labels.append(colname)
                        yield array

                if consolidate_blocks:
                    data = TypeBlocks.from_blocks(TypeBlocks.consolidate_blocks(blocks()))
                else:
                    data = TypeBlocks.from_blocks(blocks())

                # this will own_data in subsequent constructor call
                yield container_type._from_data_index_arrays_column_labels(
                        data=data,
                        index_depth=index_depth,
                        index_arrays=index_arrays,
                        index_constructors=index_constructors,
                        columns_depth=columns_depth,
                        columns_labels=columns_labels,
                        columns_constructors=columns_constructors,
                        name=label,
                        )

    @store_coherent_non_write
    def labels(self, *,
            config: StoreConfigMapInitializer = None,
            strip_ext: bool = True,
            ) -> tp.Iterator[TLabel]:
        '''
        Iterator of labels.
        '''
        import tables

        config_map = StoreConfigMap.from_initializer(config)

        with tables.open_file(self._fp, mode='r') as file:
            for node in file.iter_nodes(where='/',
                    classname=tables.Table.__name__):
                # NOTE: this is not the complete path
                yield config_map.default.label_decode(node.name)

