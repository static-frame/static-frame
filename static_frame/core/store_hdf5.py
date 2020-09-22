
import typing as tp

import numpy as np


from static_frame.core.doc_str import doc_inject
from static_frame.core.frame import Frame
from static_frame.core.store import Store
from static_frame.core.store import store_coherent_non_write
from static_frame.core.store import store_coherent_write
from static_frame.core.store import StoreConfig
from static_frame.core.store import StoreConfigMap
from static_frame.core.store import StoreConfigMapInitializer
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.util import DTYPE_STR_KINDS

# from static_frame.core.store_filter import STORE_FILTER_DEFAULT
# from static_frame.core.store_filter import StoreFilter

class StoreHDF5(Store):

    _EXT: tp.FrozenSet[str] =  frozenset(('.h5', '.hdf5'))

    @store_coherent_write
    def write(self,
            items: tp.Iterable[tp.Tuple[tp.Optional[str], Frame]],
            *,
            config: StoreConfigMapInitializer = None,
            # store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ) -> None:

        config_map = StoreConfigMap.from_initializer(config)

        import tables

        with tables.open_file(self._fp, mode='w') as file:
            for label, frame in items:
                c = config_map[label]

                # should all tables be under a common group?
                field_names, dtypes = self.get_field_names_and_dtypes(
                        frame=frame,
                        include_index=c.include_index,
                        include_columns=c.include_columns
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


    @doc_inject(selector='constructor_frame')
    @store_coherent_non_write
    def read(self,
            label: tp.Optional[str] = None,
            *,
            config: tp.Optional[StoreConfig] = None,
            container_type: tp.Type[Frame] = Frame,
            # store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ) -> Frame:
        '''
        Args:
            {dtypes}
        '''
        import tables

        if config is None:
            config = StoreConfig() # get default
        if config.dtypes:
            raise NotImplementedError('using config.dtypes on HDF5 not yet supported')

        index_depth = config.index_depth
        columns_depth = config.columns_depth

        index_arrays = []
        columns_labels = []

        with tables.open_file(self._fp, mode='r') as file:
            table = file.get_node(f'/{label}')
            colnames = table.cols._v_colnames

            def blocks() -> tp.Iterator[np.ndarray]:
                for col_idx, colname in enumerate(colnames):

                    # can also do: table.read(field=colname)
                    array = table.col(colname)

                    if array.dtype.kind in DTYPE_STR_KINDS:
                        array = array.astype(str)
                    array.flags.writeable = False

                    if col_idx < index_depth:
                        index_arrays.append(array)
                        continue
                    # only store column labels for those yielded
                    columns_labels.append(colname)
                    yield array

            if config.consolidate_blocks:
                data = TypeBlocks.from_blocks(TypeBlocks.consolidate_blocks(blocks()))
            else:
                data = TypeBlocks.from_blocks(blocks())

        return container_type._from_data_index_arrays_column_labels(
                data=data,
                index_depth=index_depth,
                index_arrays=index_arrays,
                columns_depth=columns_depth,
                columns_labels=columns_labels,
                name=tp.cast(tp.Hashable, label) # not sure why this is necessary
                )


    @store_coherent_non_write
    def labels(self, strip_ext: bool = True) -> tp.Iterator[str]:
        '''
        Iterator of labels.
        '''
        import tables

        with tables.open_file(self._fp, mode='r') as file:
            for node in file.iter_nodes(where='/',
                    classname=tables.Table.__name__):
                # NOTE: this is not the complete path
                yield node.name
