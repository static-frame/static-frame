from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Mapping

import typing_extensions as tp

from static_frame.core.exception import ErrorInitStoreConfig
from static_frame.core.frame import Frame
from static_frame.core.util import DTYPE_STR_KINDS

if tp.TYPE_CHECKING:
    from static_frame.core.generic_aliases import TFrameAny
    from static_frame.core.store_filter import StoreFilter
    from static_frame.core.util import (
        TCallableAny,
        TDepthLevel,
        TDtypesSpecifier,
        TIndexCtorSpecifiers,
        TLabel,
        TMpContext,
    )


def validate_func_and_store_config(
    func: TCallableAny,
    store_config: type[StoreConfig],
) -> None:
    defined = {field.name for field in dataclasses.fields(store_config)}
    required = {
        name
        for name, param in inspect.signature(func).parameters.items()
        if param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD)
    }

    if missing := defined - required:
        raise ValueError(
            f'The following fields are defined on {store_config.__name__} but are '
            f"not part of {func.__name__}'s signature: {missing}"
        )


def label_encode_tuple(source: tuple[tp.Any, ...]) -> str:
    """For encoding tuples of NumPy scalars in strings that can use literal_eval to re-evaluate"""
    parts = []
    for obj in source:
        if dt := getattr(obj, 'dtype', None):  # a NumPy scalar
            if dt.kind in DTYPE_STR_KINDS:
                parts.append(f"'{obj}'")
            else:  # str, not repr, must be used
                parts.append(str(obj))
        elif isinstance(obj, str):
            parts.append(repr(obj))
        else:
            parts.append(str(obj))
    return f'({", ".join(parts)})'


def from_frame(cls, frame: TFrameAny) -> 'StoreConfig':
    """Derive a config from a Frame."""
    include_index = frame.index.depth > 1 or frame.index._map is not None  # type: ignore
    index_depth = 0 if not include_index else frame.index.depth

    include_columns = frame.columns.depth > 1 or frame.columns._map is not None  # type: ignore
    columns_depth = 0 if not include_columns else frame.columns.depth

    return cls(
        index_depth=index_depth,
        columns_depth=columns_depth,
        include_index=include_index,
        include_columns=include_columns,
    )


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
    unsafe_hash=True,
)
class StoreConfig:
    """
    A read-only, hashable container used by :obj:`Store` subclasses for reading from and writing to multi-table storage formats.

    This base class contains arguments common to all storage formats.
    """

    label_encoder: tp.Callable[[TLabel], str] | None = None
    label_decoder: tp.Callable[[str], TLabel] | None = None
    read_frame_filter: tp.Callable[[TLabel, TFrameAny], TFrameAny] | None = None
    read_max_workers: int | None = None
    read_chunksize: int = 1
    write_max_workers: int | None = None
    write_chunksize: int = 1
    mp_context: TMpContext | None = None

    _CONSTRUCTOR: tp.ClassVar[tp.Callable[..., TFrameAny]]

    @classmethod
    def __init_subclass__(cls, complete: bool = True) -> None:
        if complete:
            validate_func_and_store_config(cls._CONSTRUCTOR, cls)

    def for_frame_construction_only(self) -> tp.Self:
        # This base config contains only information relevant to the frame construction process.
        return dataclasses.replace(
            self,
            **dict.fromkeys((field.name for field in dataclasses.fields(StoreConfig))),
        )

    def label_encode(self, label: TLabel) -> str:
        if self.label_encoder is str and isinstance(label, tuple):
            # with NumPy2, str() of a tuple of NumPy scalars returns (np.str_('a'), np.int64(1)), not ('a', 1)
            label = label_encode_tuple(label)

        elif self.label_encoder:
            label = self.label_encoder(label)

        if not isinstance(label, str):
            raise RuntimeError(
                f'Store label {label!r} is not a string; provide a label_encoder to StoreConfig'
            )

        return label

    def label_decode(self, label: str) -> TLabel:
        if self.label_decoder:
            return self.label_decoder(label)

        return label


DEFAULT_STORE_CONFIG = StoreConfig()


@dataclasses.dataclass(frozen=True, kw_only=True)
class StoreConfigDelimited(StoreConfig, complete=False):
    # Constructors
    index_depth: int = 0  # this default does not permit round trip
    index_name_depth_level: TDepthLevel | None = None
    index_constructors: TIndexCtorSpecifiers = None
    columns_depth: int = 1
    columns_name_depth_level: TDepthLevel | None = None
    columns_constructors: TIndexCtorSpecifiers = None
    dtypes: TDtypesSpecifier = None
    consolidate_blocks: bool = False

    # Exporters
    include_index: bool = True
    include_index_name: bool = True
    include_columns: bool = True
    include_columns_name: bool = False

    # Both
    store_filter: StoreFilter | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class StoreConfigTSV(StoreConfigDelimited):
    _CONSTRUCTOR = Frame.from_tsv


@dataclasses.dataclass(frozen=True, kw_only=True)
class StoreConfigCSV(StoreConfigDelimited):
    _CONSTRUCTOR = Frame.from_csv


@dataclasses.dataclass(frozen=True, kw_only=True)
class StoreConfigPickle(StoreConfig):
    _CONSTRUCTOR = Frame.from_pickle


@dataclasses.dataclass(frozen=True, kw_only=True)
class StoreConfigNPZ(StoreConfig):
    # Exporters
    include_index: bool = True
    include_columns: bool = True
    consolidate_blocks: bool = False

    _CONSTRUCTOR = Frame.from_npz


@dataclasses.dataclass(frozen=True, kw_only=True)
class StoreConfigParquet(StoreConfig):
    # Constructors
    index_depth: int = 0  # this default does not permit round trip
    index_name_depth_level: TDepthLevel | None = None
    index_constructors: TIndexCtorSpecifiers = None
    columns_depth: int = 1
    columns_name_depth_level: TDepthLevel | None = None
    columns_constructors: TIndexCtorSpecifiers = None
    columns_select: tp.Iterable[str] | None = None
    dtypes: TDtypesSpecifier = None
    consolidate_blocks: bool = False

    # Exporters
    include_index: bool = True
    include_index_name: bool = True
    include_columns: bool = True
    include_columns_name: bool = False

    _CONSTRUCTOR = Frame.from_parquet


@dataclasses.dataclass(frozen=True, kw_only=True)
class StoreConfigNPY(StoreConfig):
    # Exporters
    include_index: bool = True
    include_columns: bool = True
    consolidate_blocks: bool = False

    _CONSTRUCTOR = Frame.from_npy


TVStoreConfig = tp.TypeVar('TVStoreConfig', bound=StoreConfig, default=StoreConfig)


@tp.final
class StoreConfigMap(tp.Generic[TVStoreConfig]):
    """
    Immutable defaultdict-like mapping to StoreConfigs.
    """

    __slots__ = (
        '_map',
        '_default',
    )

    _ALIGN_WITH_DEFAULT_ATTRS = tuple(
        field.name for field in dataclasses.fields(StoreConfig)
    )

    @classmethod
    def _from_config(cls, config: TVStoreConfig) -> tp.Self:
        return cls(default=config)

    @classmethod
    def from_initializer(
        cls, initializer: TVStoreConfigMapInitializer[TVStoreConfig]
    ) -> tp.Self:
        if initializer is None:
            return cls()

        if isinstance(initializer, StoreConfig):
            return cls._from_config(initializer)

        if isinstance(initializer, cls):
            return initializer

        if not isinstance(initializer, Mapping):
            raise ErrorInitStoreConfig(
                f'Unsupported initializer type: {type(initializer)}'
            )

        return cls(initializer)

    def __init__(
        self,
        config_map: tp.Mapping[tp.Any, TVStoreConfig] | None = None,
        *,
        default: TVStoreConfig | None = None,
    ) -> None:
        self._map: tp.Mapping[tp.Any, TVStoreConfig] = {}

        if config_map:
            config_types: set[type[TVStoreConfig]] = set(map(type, config_map.values()))

            if len(config_types) == 1:
                config_type = config_types.pop()

                # Better default!
                if default is None:
                    default = config_type()
                elif not isinstance(default, config_type):
                    raise ErrorInitStoreConfig(
                        'Default & Mapping config classes must be the same!'
                    )
            else:
                raise ErrorInitStoreConfig('Multiple config types present!')

            for label, config in config_map.items():
                for attr in self._ALIGN_WITH_DEFAULT_ATTRS:
                    if getattr(config, attr) != getattr(default, attr):
                        raise ErrorInitStoreConfig(
                            f'config {label} has {attr} inconsistent with default; align values and/or pass a default StoreConfig.'
                        )

                self._map[label] = config

        if default is None:
            # Mapping is empty and no default provided; use global default!
            default = DEFAULT_STORE_CONFIG  # type: ignore

        # Either we have empty map & default config, or properly validated map & default
        self._default: TVStoreConfig = default  # type: ignore

    def __getitem__(self, key: tp.Any) -> TVStoreConfig:
        return self._map.get(key, self._default)

    @property
    def default(self) -> TVStoreConfig:
        return self._default


TVStoreConfigMapInitializer = tp.TypeAliasType(
    'TVStoreConfigMapInitializer',
    TVStoreConfig
    | StoreConfigMap[TVStoreConfig]
    | None
    | tp.Mapping[tp.Any, TVStoreConfig],
    type_params=(TVStoreConfig,),
)
