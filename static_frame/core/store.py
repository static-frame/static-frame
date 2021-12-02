

import typing as tp
import os

from itertools import chain
from functools import partial
from functools import wraps
from weakref import WeakValueDictionary
import numpy as np

from static_frame.core.interface_meta import InterfaceMeta

from static_frame.core.exception import ErrorInitStore
from static_frame.core.exception import ErrorInitStoreConfig
from static_frame.core.exception import StoreFileMutation
from static_frame.core.exception import StoreParameterConflict

from static_frame.core.frame import Frame
from static_frame.core.util import AnyCallable
from static_frame.core.util import DtypesSpecifier
from static_frame.core.util import path_filter
from static_frame.core.util import PathSpecifier
from static_frame.core.util import DepthLevelSpecifier
from static_frame.core.util import IndexConstructors

#-------------------------------------------------------------------------------

class StoreConfigHE(metaclass=InterfaceMeta):
    '''
    A read-only, hashable container used by :obj:`Store` subclasses for reading from and writing to multi-table storage formats.
    '''

    index_depth: int
    index_name_depth_level: tp.Optional[DepthLevelSpecifier]
    index_constructors: IndexConstructors
    columns_depth: int
    columns_name_depth_level: tp.Optional[DepthLevelSpecifier]
    columns_constructors: IndexConstructors
    columns_select: tp.Optional[tp.Iterable[str]]
    dtypes: DtypesSpecifier
    consolidate_blocks: bool
    skip_header: int
    skip_footer: int
    trim_nadir: bool
    include_index: bool
    include_index_name: bool
    include_columns: bool
    include_columns_name: bool
    merge_hierarchical_labels: bool
    read_max_workers: tp.Optional[int]
    read_chunksize: int
    write_max_workers: tp.Optional[int]
    write_chunksize: int
    _hash: tp.Optional[int]

    __slots__ = (
            'index_depth',
            'index_name_depth_level',
            'index_constructors',
            'columns_depth',
            'columns_name_depth_level',
            'columns_constructors',
            'columns_select',
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
            'read_max_workers',
            'read_chunksize',
            'write_max_workers',
            'write_chunksize',
            '_hash'
            )

    def __init__(self, *,
            # constructors
            index_depth: int = 0, # this default does not permit round trip
            index_name_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            index_constructors: IndexConstructors = None,
            columns_depth: int = 1,
            columns_name_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            columns_constructors: IndexConstructors = None,
            columns_select: tp.Optional[tp.Iterable[str]] = None,
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
            # multiprocessing configuration
            read_max_workers: tp.Optional[int] = None,
            read_chunksize: int = 1,
            write_max_workers: tp.Optional[int] = None,
            write_chunksize: int = 1,
            ):
        '''
        Args:
            include_index: Boolean to determine if the ``index`` is included in output.
            include_columns: Boolean to determine if the ``columns`` is included in output.
        '''
        # constructor
        self.index_depth = index_depth
        self.index_name_depth_level = index_name_depth_level
        self.index_constructors = index_constructors
        self.columns_depth = columns_depth
        self.columns_name_depth_level = columns_name_depth_level
        self.columns_constructors = columns_constructors
        self.columns_select = columns_select
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

        self.read_max_workers = read_max_workers
        self.read_chunksize = read_chunksize
        self.write_max_workers = write_max_workers
        self.write_chunksize = write_chunksize

        self._hash = None

    def __eq__(self, other: tp.Any) -> bool:
        if not isinstance(other, StoreConfigHE):
            return False

        for attr in self.__slots__:
            if attr.startswith('_'):
                continue
            if getattr(self, attr) != getattr(other, attr):
                return False

        return True

    def __ne__(self, other: tp.Any) -> bool:
        return not self.__eq__(other)

    @staticmethod
    def _hash_depth_specifier(depth_specifier: tp.Optional[DepthLevelSpecifier]) -> tp.Hashable:
        if depth_specifier is None or isinstance(depth_specifier, int):
            return depth_specifier
        return tuple(depth_specifier)

    @staticmethod
    def _hash_dtypes_specifier(dtypes_specifier: DtypesSpecifier) -> tp.Hashable:
        if dtypes_specifier is None :
            return dtypes_specifier
        if isinstance(dtypes_specifier, dict):
            return tuple(dtypes_specifier.items())
        if isinstance(dtypes_specifier, list):
            return tuple(dtypes_specifier)
        return dtypes_specifier # type: ignore [return-value]

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((
                    self.index_depth, # int
                    self._hash_depth_specifier(self.index_name_depth_level),
                    self.index_constructors, # class or callable
                    self.columns_depth, # int
                    self._hash_depth_specifier(self.columns_name_depth_level),
                    self.columns_constructors, # class or callable
                    self.columns_select if self.columns_select is None else tuple(self.columns_select),
                    self._hash_dtypes_specifier(self.dtypes),
                    self.consolidate_blocks, # bool
                    self.skip_header, # int
                    self.skip_footer, # int
                    self.trim_nadir, # bool
                    self.include_index, # bool
                    self.include_index_name, # bool
                    self.include_columns, # bool
                    self.include_columns_name, # bool
                    self.merge_hierarchical_labels, # bool
                    self.read_max_workers, # Optional[int]
                    self.read_chunksize, # int
                    self.write_max_workers, # Optional[int]
                    self.write_chunksize, # int
            ))
        return self._hash


class StoreConfig(StoreConfigHE):
    '''
    A read-only container of parameters used by :obj:`Store` subclasses for reading from and writing to multi-table storage formats.
    '''
    label_encoder: tp.Optional[tp.Callable[[tp.Hashable], str]]
    label_decoder: tp.Optional[tp.Callable[[str], tp.Hashable]]

    __slots__ = (
            'label_encoder',
            'label_decoder',
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
            index_depth: int = 0,
            index_name_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            index_constructors: IndexConstructors = None,
            columns_depth: int = 1,
            columns_name_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            columns_constructors: IndexConstructors = None,
            columns_select: tp.Optional[tp.Iterable[str]] = None,
            dtypes: DtypesSpecifier = None,
            consolidate_blocks: bool = False,
            skip_header: int = 0,
            skip_footer: int = 0,
            trim_nadir: bool = False,
            include_index: bool = True,
            include_index_name: bool = True,
            include_columns: bool = True,
            include_columns_name: bool = False,
            merge_hierarchical_labels: bool = True,
            label_encoder: tp.Optional[tp.Callable[[tp.Hashable], str]] = None,
            label_decoder: tp.Optional[tp.Callable[[str], tp.Hashable]] = None,
            read_max_workers: tp.Optional[int] = None,
            read_chunksize: int = 1,
            write_max_workers: tp.Optional[int] = None,
            write_chunksize: int = 1,
            ):
        StoreConfigHE.__init__(self,
                index_depth=index_depth,
                index_name_depth_level=index_name_depth_level,
                index_constructors=index_constructors,
                columns_depth=columns_depth,
                columns_name_depth_level=columns_name_depth_level,
                columns_constructors=columns_constructors,
                columns_select=columns_select,
                dtypes=dtypes,
                consolidate_blocks=consolidate_blocks,
                skip_header=skip_header,
                skip_footer=skip_footer,
                trim_nadir=trim_nadir,
                include_index=include_index,
                include_index_name=include_index_name,
                include_columns=include_columns,
                include_columns_name=include_columns_name,
                merge_hierarchical_labels=merge_hierarchical_labels,
                read_max_workers=read_max_workers,
                read_chunksize=read_chunksize,
                write_max_workers=write_max_workers,
                write_chunksize=write_chunksize,
        )
        # NOTE: if only encode is provide, should we raise?
        self.label_encoder = label_encoder
        self.label_decoder = label_decoder

    def label_encode(self, label: tp.Hashable) -> str:
        if self.label_encoder:
            label = self.label_encoder(label)
        if not isinstance(label, str):
            raise RuntimeError(f'Store label {label} is not a string; provide a label_encoder to StoreConfig')
        return label

    def label_decode(self, label: str) -> tp.Hashable:
        if self.label_decoder:
            return self.label_decoder(label)
        return label

    def to_store_config_he(self) -> 'StoreConfigHE':
        '''
        Return a ``StoreConfigHE`` version of this StoreConfig.
        '''
        return StoreConfigHE(**{attr: getattr(self, attr)
            for attr in StoreConfigHE.__slots__ if not attr.startswith('_')})

    def __eq__(self, other: tp.Any) -> bool:
        if not isinstance(other, StoreConfig):
            return False
        return id(self) == id(other)

    def __hash__(self) -> int:
        raise NotImplementedError()


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

    # These attrs (when set) must align with default
    _ALIGN_WITH_DEFAULT_ATTRS = (
            'label_encoder',
            'label_decoder',
            'read_max_workers',
            'read_chunksize',
            'write_max_workers',
            'write_chunksize',
    )

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
            *,
            default: tp.Optional[StoreConfig] = None,
            own_config_map: bool = False
            ):

        if default is None:
            self._default = self._DEFAULT
        elif not isinstance(default, StoreConfig):
            raise ErrorInitStoreConfig(
                f'unspported class {default}, must be {StoreConfig}')
        else:
            self._default = default

        # initialize new dict and transfer to support checking Config classes
        self._map: SCMMapType = {}

        if own_config_map and config_map is not None:
            self._map = config_map
        elif config_map:
            for label, config in config_map.items():
                if not isinstance(config, self._DEFAULT.__class__):
                    raise ErrorInitStoreConfig(
                        f'unspported class {config}, must be {self._DEFAULT.__class__}')

                for attr in self._ALIGN_WITH_DEFAULT_ATTRS:
                    if getattr(config, attr) != getattr(self._default, attr):
                        raise ErrorInitStoreConfig(f'config {label} has {attr} inconsistent with default; align values and/or pass a default StoreConfig.')

                self._map[label] = config

    def __getitem__(self, key: tp.Optional[tp.Hashable]) -> StoreConfig:
        return self._map.get(key, self._default)

    @property
    def default(self) -> StoreConfig:
        return self._default


#-------------------------------------------------------------------------------
# decorators

def store_coherent_non_write(f: AnyCallable) -> AnyCallable:

    @wraps(f)
    def wrapper(self: 'Store', *args: tp.Any, **kwargs: tp.Any) -> Frame:
        '''Decprator for derived Store class implementation of read(), labels().
        '''
        self._mtime_coherent()
        return f(self, *args, **kwargs) # type: ignore

    return wrapper


def store_coherent_write(f: AnyCallable) -> AnyCallable:
    '''Decorator for derived Store classes implementation of write()
    '''
    @wraps(f)
    def wrapper(self: 'Store', *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        post = f(self,  *args, **kwargs)
        self._mtime_update()
        return post

    return wrapper

#-------------------------------------------------------------------------------
class Store:

    _EXT: tp.FrozenSet[str]

    __slots__ = (
            '_fp',
            '_last_modified',
            '_weak_cache',
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
        self._weak_cache: tp.MutableMapping[tp.Hashable, Frame] = WeakValueDictionary()

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

    # def __copy__(self) -> 'Store':
    #     '''
    #     Return a new Store instance linked to the same file.
    #     '''
    #     return self.__class__(fp=self._fp)

    #---------------------------------------------------------------------------
    @staticmethod
    def get_field_names_and_dtypes(*,
            frame: Frame,
            include_index: bool,
            include_index_name: bool,
            include_columns: bool,
            include_columns_name: bool,
            force_str_names: bool = False,
            force_brackets: bool = False
            ) -> tp.Tuple[tp.Sequence[str], tp.Sequence[np.dtype]]:

        index = frame.index
        columns = frame.columns
        columns_values = columns.values

        if include_index_name and include_columns_name:
            raise StoreParameterConflict('cannot include_index_name and include_columns_name with this Store')

        if columns.depth > 1:
            # The str() of an array produces a space-delimited representation that includes list brackets; we could trim these brackets here, but need them for SQLite usage; thus, clients will have to trim if necessary.
            columns_values = tuple(str(c) for c in columns_values)

        if not include_index:
            if include_columns_name:
                raise StoreParameterConflict('cannot include_columns_name when include_index is False')
            dtypes = frame._blocks.dtypes.tolist()
            if include_columns:
                field_names = columns_values
            else: # name fields with integers?
                field_names = range(frame._blocks.shape[1])
        else:
            if index.depth == 1:
                dtypes = [index.dtype]
            else:
                dtypes = index.dtypes.values.tolist() #type: ignore [attr-defined]
            # Get a list to mutate.
            if include_index_name:
                field_names = list(index.names)
            elif include_columns_name and index.depth == columns.depth:
                field_names = list(columns.names)
            elif include_columns_name and index.depth == 1 and columns.depth > 1:
                field_names = [tuple(str(c) for c in columns.names),]
            else: # if index_depth > 1 and not equal t columns_depth
                raise StoreParameterConflict('cannot determine field names over index; set one of include_index_name or include_columns_name')

            # add frame dtypes t0 those from index
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
                    name_str = str(name)
                    if name_str.startswith('[') and name_str.endswith(']'):
                        yield name_str
                    elif isinstance(name, tuple):
                        # make look like it came from the array
                        yield f'[{" ".join((repr(n) for n in name))}]'
                    else:
                        yield f'[{name_str}]'
            field_names = list(gen())

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

        return partial(frame.iter_array, axis=1) #type: ignore

    @staticmethod
    def get_column_iterator(
            frame: Frame,
            include_index: bool
            ) -> tp.Iterator[np.ndarray]:
        if include_index:
            index_depth = frame._index.depth

            if index_depth == 1:
                index_values = frame._index.values
                return chain(
                        (index_values,),
                        frame._blocks.axis_values(0)
                        )
            return chain(
                    (frame._index.values_at_depth(d) for d in range(index_depth)),
                    frame._blocks.axis_values(0)
                    )
        # avoid creating a Series per column by going to blocks
        return frame._blocks.axis_values(0)

    #---------------------------------------------------------------------------
    def read_many(self,
            labels: tp.Iterable[tp.Hashable],
            *,
            config: StoreConfigMapInitializer = None,
            container_type: tp.Type[Frame] = Frame,
            ) -> tp.Iterator[Frame]:
        '''Read many Frame, given by `labels`, from the Store. Return an iterator of instances of `container_type`.
        '''
        raise NotImplementedError() #pragma: no cover

    @store_coherent_non_write
    def read(self,
            label: tp.Hashable,
            *,
            config: tp.Optional[StoreConfig] = None,
            container_type: tp.Type[Frame] = Frame,
            ) -> Frame:
        '''Read a single Frame, given by `label`, from the Store. Return an instance of `container_type`. This is a convenience method using ``read_many``.
        '''
        return next(self.read_many((label,), config=config, container_type=container_type))

    def write(self,
            items: tp.Iterable[tp.Tuple[str, Frame]],
            *,
            config: StoreConfigMapInitializer = None
            ) -> None:
        '''Write all ``Frames`` in the Store.
        '''
        raise NotImplementedError() #pragma: no cover

    def labels(self, *,
            config: StoreConfigMapInitializer = None,
            strip_ext: bool = True,
            ) -> tp.Iterator[tp.Hashable]:
        raise NotImplementedError() #pragma: no cover

