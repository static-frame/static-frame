
import typing as tp

# from itertools import chain

from static_frame.core.frame import Frame
from static_frame.core.store import Store

# from static_frame.core.index_hierarchy import IndexHierarchy

# from static_frame.core.store_filter import StoreFilter
# from static_frame.core.store_filter import STORE_FILTER_DEFAULT

from static_frame.core.doc_str import doc_inject

# from static_frame.core.util import DtypesSpecifier
# from static_frame.core.util import DTYPE_INT_KIND
from static_frame.core.util import DTYPE_STR_KIND
# from static_frame.core.util import DTYPE_NAN_KIND
# from static_frame.core.util import DTYPE_DATETIME_KIND
# from static_frame.core.util import DTYPE_BOOL
# from static_frame.core.util import BOOL_TYPES
# from static_frame.core.util import NUMERIC_TYPES




class StoreHDF5(Store):

    _EXT: tp.FrozenSet[str] =  frozenset(('.h5', '.hdf5'))


    def write(self,
            items: tp.Iterable[tp.Tuple[tp.Optional[str], Frame]],
            *,
            include_index: bool = True,
            include_columns: bool = True,
            # store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ) -> None:

        import tables

        with tables.open_file(self._fp, mode='w') as file:
            for label, frame in items:
                # should all tables be under a common group?
                field_names, dtypes = self.get_field_names_and_dtypes(
                        frame=frame,
                        include_index=include_index,
                        include_columns=include_columns
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
                        include_index=include_index)
                table.append(tuple(values()))
                table.flush()


    @doc_inject(selector='constructor_frame')
    def read(self,
            label: tp.Optional[str] = None,
            *,
            index_depth: int=1,
            columns_depth: int=1,
            ) -> Frame:
        '''
        Args:
            {dtypes}
        '''
        import tables

        with tables.open_file(self._fp, mode='r') as file:
            table = file.get_node(f'/{label}')
            array = table.read()
            array.flags.writeable = False

            # Discover all string dtypes and replace the dtype with a generic `str` function; first element in values array is the dtype object.
            dtypes =  {k: str
                    for i, (k, v) in enumerate(array.dtype.fields.items())
                    if v[0].kind in DTYPE_STR_KIND
                    }
            # this works, but does not let us pull off columns yet
            f = tp.cast(Frame,
                    Frame.from_structured_array(
                            array,
                            name=label,
                            index_depth=index_depth,
                            columns_depth=columns_depth,
                            dtypes=dtypes,
                    ))
            return f

    def labels(self) -> tp.Iterator[str]:
        '''
        Iterator of labels.
        '''
        import tables

        with tables.open_file(self._fp, mode='r') as file:
            for node in file.iter_nodes(where='/',
                    classname=tables.Table.__name__):
                # NOTE: this is not the complete path
                yield node.name
