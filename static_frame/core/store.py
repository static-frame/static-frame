

import typing as tp
import os

from itertools import chain
from functools import partial
from functools import wraps
import numpy as np

from static_frame.core.interface_meta import InterfaceMeta

from static_frame.core.exception import ErrorInitStore
from static_frame.core.exception import ErrorInitStoreConfig
from static_frame.core.exception import StoreFileMutation
from static_frame.core.frame import Frame
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.util import AnyCallable
from static_frame.core.util import DtypesSpecifier
from static_frame.core.util import path_filter
from static_frame.core.util import PathSpecifier
from static_frame.core.util import DepthLevelSpecifier


#-------------------------------------------------------------------------------
class StoreConfig(metaclass=InterfaceMeta):
    '''
    A read-only container of parameters used by :obj:`Store` subclasses for reading from and writing to multi-table storage formats.
    '''

    index_depth: int
    columns_depth: int
    dtypes: DtypesSpecifier
    include_index: bool
    include_columns: bool
    merge_hierarchical_labels: bool

    __slots__ = (
            'index_depth',
            'index_name_depth_level',
            'columns_depth',
            'columns_name_depth_level',
            'dtypes',
            'consolidate_blocks',
            'skip_header',
            'skip_footer',
            'trim_nadir',
            'include_index',
            'include_index_name',
            'include_columns',
            'include_columns_name',
            'merge_hierarchical_labels',
            )

    @classmethod
    def from_frame(cls, frame: Frame) -> 'StoreConfig':
        '''Derive a config from a Frame.
        '''
        include_index = frame.index.depth > 1 or not frame.index._map is None
        index_depth = 0 if not include_index else frame.index.depth

        include_columns = frame.columns.depth > 1 or not frame.columns._map is None
        columns_depth = 0 if not include_columns else frame.columns.depth

        return cls(
                index_depth=index_depth,
                columns_depth=columns_depth,
                include_index=include_index,
                include_columns=include_columns
                )

    def __init__(self, *,
            # constructors
            index_depth: int = 0, # this default does not permit round trip
            index_name_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            columns_depth: int = 1,
            columns_name_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            dtypes: DtypesSpecifier = None,
            consolidate_blocks: bool = False,
            # not used by all constructors
            skip_header: int = 0,
            skip_footer: int = 0,
            trim_nadir: bool = False,
            # exporters
            include_index: bool = True,
            include_index_name: bool = True,
            include_columns: bool = True,
            include_columns_name: bool = False,
            # not used by all exporters
            merge_hierarchical_labels: bool = True,
            ):
        '''
        Args:
            include_index: Boolean to determine if the ``index`` is included in output.
            include_columns: Boolean to determine if the ``columns`` is included in output.
        '''
        # constructor
        self.index_depth = index_depth
        self.index_name_depth_level = index_name_depth_level
        self.columns_depth = columns_depth
        self.columns_name_depth_level = columns_name_depth_level
        self.dtypes = dtypes
        self.consolidate_blocks = consolidate_blocks
        self.skip_header = skip_header
        self.skip_footer = skip_footer
        self.trim_nadir = trim_nadir

        # exporter
        self.include_index = include_index
        self.include_index_name = include_index_name
        self.include_columns = include_columns
        self.include_columns_name = include_columns_name

        # self.format_index = format_index
        # self.format_columns = format_columns
        self.merge_hierarchical_labels = merge_hierarchical_labels

# NOTE: key should be tp.Optional[str], but cannot get mypy to accept
SCMMapType = tp.Mapping[tp.Any, StoreConfig]
SCMMapInitializer = tp.Optional[SCMMapType]

StoreConfigMapInitializer = tp.Union[
        StoreConfig,
        SCMMapInitializer,
        'StoreConfigMap'
        ]


class StoreConfigMap:
    '''
    Container of one or more StoreConfig, with the optional specification of a default StoreConfig. Assumed immutable over the life of the instance.
    '''
    __slots__ = (
            '_map',
            '_default'
            )

    _DEFAULT: StoreConfig = StoreConfig()

    @classmethod
    def from_frames(cls, frames: tp.Iterable[Frame]) -> 'StoreConfigMap':
        '''
        Derive a config map from an iterable of Frames
        '''
        config_map = {f.name: StoreConfig.from_frame(f) for f in frames}
        return cls(config_map, own_config_map=True)
    @classmethod
    def from_config(cls, config: StoreConfig) -> 'StoreConfigMap':
        return cls(default=config)

    @classmethod
    def from_initializer(
            cls,
            initializer: StoreConfigMapInitializer
            ) -> 'StoreConfigMap':
        if isinstance(initializer, StoreConfig):
            return cls.from_config(initializer)
        if isinstance(initializer, cls):
            # return same instance
            return initializer
        if initializer is None: # will get default configuration
            return cls()
        assert isinstance(initializer, dict)
        return cls(initializer)

    def __init__(self,
            config_map: SCMMapInitializer = None,
            default: tp.Optional[StoreConfig] = None,
            own_config_map: bool = False
            ):

        # initialize new dict and transfer to support checking Config classes
        self._map: SCMMapType = {}

        if own_config_map and config_map is not None:
            self._map = config_map
        elif config_map:
            for label, config in config_map.items():
                if not isinstance(config, self._DEFAULT.__class__):
                    raise ErrorInitStoreConfig(
                        f'unspported class {config}, must be {self._DEFAULT.__class__}')
                self._map[label] = config

        if default is None:
            self._default = self._DEFAULT
        elif not isinstance(default, StoreConfig):
            raise ErrorInitStoreConfig(
                f'unspported class {default}, must be {StoreConfig}')
        else:
            self._default = default

    def __getitem__(self, key: tp.Optional[str]) -> StoreConfig:
        return self._map.get(key, self._default)



#-------------------------------------------------------------------------------
class Store:

    _EXT: tp.FrozenSet[str]

    __slots__ = (
            '_fp',
            '_last_modified'
            )

    def __init__(self, fp: PathSpecifier):
        # Redefine fp variable as only string after the filter.
        fp = tp.cast(str, path_filter(fp))

        if not os.path.splitext(fp)[1] in self._EXT:
            raise ErrorInitStore(
                    f'file path {fp} does not match one of the required extensions: {self._EXT}')

        self._fp: str = fp

        self._last_modified = np.nan
        self._mtime_update()


    def _mtime_update(self) -> None:
        if os.path.exists(self._fp):
            self._last_modified = os.path.getmtime(self._fp)
        else:
            self._last_modified = np.nan

    def _mtime_coherent(self) -> None:
        '''Raise if a file exists at self._fp and its mtime is not as expected
        '''
        if os.path.exists(self._fp):
            if os.path.getmtime(self._fp) != self._last_modified:
                raise StoreFileMutation(f'file {self._fp} was unexpectedly changed')
        elif not np.isnan(self._last_modified):
            # file existed previously and we got a modification time, but now it does not exist
            raise StoreFileMutation(f'expected file {self._fp} no longer exists')


    #---------------------------------------------------------------------------
    @staticmethod
    def get_field_names_and_dtypes(*,
            frame: Frame,
            include_index: bool,
            include_columns: bool,
            force_str_names: bool = False,
            force_brackets: bool = False
            ) -> tp.Tuple[tp.Sequence[str], tp.Sequence[np.dtype]]:

        index = frame.index
        columns = frame.columns
        columns_values = columns.values

        if columns.depth > 1:
            # The str() of an array produces a space-delimited representation that includes list brackets; we could trim these brackets here, but need them for SQLite usage; thus, clients will have to trim if necessary.
            columns_values = tuple(str(c) for c in columns_values)

        if not include_index:
            dtypes = frame._blocks.dtypes
            if include_columns:
                field_names = columns_values
            else: # name fields with integers?
                field_names = range(frame._blocks.shape[1])
        else:
            if index.depth == 1:
                dtypes = [index.dtype]
            else:
                assert isinstance(index, IndexHierarchy) # for typing
                dtypes = index.dtypes.values.tolist()
            # Get a list to mutate.
            field_names = list(index.names)

            # add fram dtypes tp those from index
            dtypes.extend(frame._blocks.dtypes)

            # add index names in front of column names
            if include_columns:
                field_names.extend(columns_values)
            else: # name fields with integers?
                field_names.extend(range(frame._blocks.shape[1]))

        if force_str_names:
            field_names = [str(n) for n in field_names]
        if force_brackets:
            def gen() -> tp.Iterator[str]:
                for name in field_names:
                    name = str(name)
                    if name.startswith('[') and name.endswith(']'):
                        yield name
                    else:
                        yield f'[{name}]'
            field_names = tuple(gen())

        return field_names, dtypes

    @staticmethod
    def _get_row_iterator(
            frame: Frame,
            include_index: bool
            ) -> tp.Callable[[], tp.Iterator[tp.Sequence[tp.Any]]]:

        if include_index:
            index = frame._index
            index_values = index.values

            def values() -> tp.Iterator[tp.Sequence[tp.Any]]:
                for idx, row in enumerate(frame.iter_array(axis=1)):
                    if index.depth > 1:
                        index_row = index_values[idx] # this is an array
                    else:
                        index_row = (index_values[idx],)
                    yield tuple(chain(index_row, row))
            return values

        return partial(frame.iter_array, 1) #type: ignore

    @staticmethod
    def get_column_iterator(
            frame: Frame,
            include_index: bool
            ) -> tp.Iterator[np.ndarray]:
        if include_index:
            index_values = frame._index.values
            index_depth = frame._index.depth

            if index_depth == 1:
                return chain(
                        (index_values,),
                        frame._blocks.axis_values(0)
                        )
            # this approach is the same as IndexHierarchy.values_at_depth
            return chain(
                    (index_values[:, d] for d in range(index_depth)),
                    frame._blocks.axis_values(0)
                    )
        # avoid creating a Series per column by going to blocks
        return frame._blocks.axis_values(0)

    #---------------------------------------------------------------------------
    def read(self,
            label: str,
            *,
            config: tp.Optional[StoreConfig] = None,
            container_type: tp.Type[Frame] = Frame,
            ) -> Frame:
        '''Read a single Frame, given by `label`, from the Store. Return an instance of `container_type`.
        '''
        raise NotImplementedError() #pragma: no cover

    def write(self,
            items: tp.Iterable[tp.Tuple[str, Frame]],
            *,
            config: StoreConfigMapInitializer = None
            ) -> None:
        '''Write all ``Frames`` in the Store.
        '''
        raise NotImplementedError() #pragma: no cover

    def labels(self, strip_ext: bool = True) -> tp.Iterator[str]:
        raise NotImplementedError() #pragma: no cover



def store_coherent_non_write(f: AnyCallable) -> AnyCallable:

    @wraps(f)
    def wrapper(self: Store, *args: tp.Any, **kwargs: tp.Any) -> Frame:
        '''Decprator for derived Store class implementation of reaad(), labels().
        '''
        self._mtime_coherent()
        return f(self, *args, **kwargs) # type: ignore

    return wrapper


def store_coherent_write(f: AnyCallable) -> AnyCallable:
    '''Decorator for dervied Store classes implementation of write()
    '''
    @wraps(f)
    def wrapper(self: Store, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        post = f(self,  *args, **kwargs)
        self._mtime_update()
        return post

    return wrapper
