from __future__ import annotations

import typing_extensions as tp

from static_frame.core.exception import ErrorInitStoreConfig
from static_frame.core.frame import Frame
from static_frame.core.interface_meta import InterfaceMeta
from static_frame.core.util import DTYPE_STR_KINDS
from static_frame.core.util import TDepthLevel
from static_frame.core.util import TDtypesSpecifier
from static_frame.core.util import TIndexCtorSpecifiers
from static_frame.core.util import TLabel

TFrameAny = Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]]


def label_encode_tuple(source: tp.Tuple[tp.Any]) -> str:
    '''For encoding tuples of NumPy scalars in strings that can use literal_eval to re-evaluate
    '''
    parts = []
    for obj in source:
        if dt := getattr(obj, 'dtype', None): # a NumPy scalar
            if dt.kind in DTYPE_STR_KINDS:
                parts.append(f"'{obj}'")
            else: # str, not repr, must be used
                parts.append(str(obj))
        elif isinstance(obj, str):
            parts.append(repr(obj))
        else:
            parts.append(str(obj))
    return f"({', '.join(parts)})"


#-------------------------------------------------------------------------------

class StoreConfigHE(metaclass=InterfaceMeta):
    '''
    A read-only, hashable container used by :obj:`Store` subclasses for reading from and writing to multi-table storage formats.
    '''

    index_depth: int
    index_name_depth_level: tp.Optional[TDepthLevel]
    index_constructors: TIndexCtorSpecifiers
    columns_depth: int
    columns_name_depth_level: tp.Optional[TDepthLevel]
    columns_constructors: TIndexCtorSpecifiers
    columns_select: tp.Optional[tp.Iterable[str]]
    dtypes: TDtypesSpecifier
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
    mp_context: tp.Optional[str]
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
            'mp_context',
            '_hash'
            )

    def __init__(self, *,
            # constructors
            index_depth: int = 0, # this default does not permit round trip
            index_name_depth_level: tp.Optional[TDepthLevel] = None,
            index_constructors: TIndexCtorSpecifiers = None,
            columns_depth: int = 1,
            columns_name_depth_level: tp.Optional[TDepthLevel] = None,
            columns_constructors: TIndexCtorSpecifiers = None,
            columns_select: tp.Optional[tp.Iterable[str]] = None,
            dtypes: TDtypesSpecifier = None,
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
            mp_context: tp.Optional[str] = None,
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
        self.mp_context = mp_context
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
    def _hash_depth_specifier(depth_specifier: tp.Optional[TDepthLevel]) -> TLabel:
        if depth_specifier is None or isinstance(depth_specifier, int):
            return depth_specifier
        return tuple(depth_specifier)

    @staticmethod
    def _hash_dtypes_specifier(dtypes_specifier: TDtypesSpecifier) -> TLabel:
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
                    self.mp_context,
            ))
        return self._hash


class StoreConfig(StoreConfigHE):
    '''
    A read-only container of parameters used by :obj:`Store` subclasses for reading from and writing to multi-table storage formats.
    '''
    label_encoder: tp.Optional[tp.Callable[[TLabel], str]]
    label_decoder: tp.Optional[tp.Callable[[str], TLabel]]

    __slots__ = (
            'label_encoder',
            'label_decoder',
            )

    @classmethod
    def from_frame(cls, frame: TFrameAny) -> 'StoreConfig':
        '''Derive a config from a Frame.
        '''
        include_index = frame.index.depth > 1 or not frame.index._map is None # type: ignore
        index_depth = 0 if not include_index else frame.index.depth

        include_columns = frame.columns.depth > 1 or not frame.columns._map is None # type: ignore
        columns_depth = 0 if not include_columns else frame.columns.depth

        return cls(
                index_depth=index_depth,
                columns_depth=columns_depth,
                include_index=include_index,
                include_columns=include_columns
                )

    def __init__(self, *,
            index_depth: int = 0,
            index_name_depth_level: tp.Optional[TDepthLevel] = None,
            index_constructors: TIndexCtorSpecifiers = None,
            columns_depth: int = 1,
            columns_name_depth_level: tp.Optional[TDepthLevel] = None,
            columns_constructors: TIndexCtorSpecifiers = None,
            columns_select: tp.Optional[tp.Iterable[str]] = None,
            dtypes: TDtypesSpecifier = None,
            consolidate_blocks: bool = False,
            skip_header: int = 0,
            skip_footer: int = 0,
            trim_nadir: bool = False,
            include_index: bool = True,
            include_index_name: bool = True,
            include_columns: bool = True,
            include_columns_name: bool = False,
            merge_hierarchical_labels: bool = True,
            label_encoder: tp.Optional[tp.Callable[[TLabel], str]] = None,
            label_decoder: tp.Optional[tp.Callable[[str], TLabel]] = None,
            read_max_workers: tp.Optional[int] = None,
            read_chunksize: int = 1,
            write_max_workers: tp.Optional[int] = None,
            write_chunksize: int = 1,
            mp_context: tp.Optional[str] = None,
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
                mp_context=mp_context,
        )
        self.label_encoder = label_encoder
        self.label_decoder = label_decoder

    def label_encode(self, label: TLabel) -> str:
        if self.label_encoder is str and isinstance(label, tuple):
            # with NumPy2, str() of a tuple of NumPy scalars returns (np.str_('a'), np.int64(1)), not ('a', 1)
            label = label_encode_tuple(label) # type: ignore
        elif self.label_encoder:
            label = self.label_encoder(label)
        if not isinstance(label, str):
            raise RuntimeError(f'Store label {label!r} is not a string; provide a label_encoder to StoreConfig')
        return label

    def label_decode(self, label: str) -> TLabel:
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
    def from_frames(cls, frames: tp.Iterable[TFrameAny]) -> 'StoreConfigMap':
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

    def __getitem__(self, key: tp.Optional[TLabel]) -> StoreConfig:
        return self._map.get(key, self._default)

    @property
    def default(self) -> StoreConfig:
        return self._default

