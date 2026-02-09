from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import ClassVar

import typing_extensions as tp

from static_frame.core.exception import (
    ErrorInitStoreConfig,
    ErrorInitStoreMapConfig,
)
from static_frame.core.frame import Frame
from static_frame.core.store_filter import STORE_FILTER_DEFAULT
from static_frame.core.util import DTYPE_STR_KINDS

if tp.TYPE_CHECKING:
    from static_frame.core.generic_aliases import TFrameAny
    from static_frame.core.store_filter import StoreFilter
    from static_frame.core.util import (
        TDepthLevel,
        TDtypesSpecifier,
        TIndexCtor,
        TIndexCtorSpecifiers,
        TLabel,
        TMpContext,
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


TStoreConfig = tp.TypeVar('TStoreConfig', bound='StoreConfig', default='StoreConfig')


def _hash_depth_specifier(
    depth_specifier: TDepthLevel | None,
) -> int | tuple[int, ...] | None:
    if isinstance(depth_specifier, Iterable):
        # If already a tuple, this is a no-op in modern Python.
        return tuple(depth_specifier)

    return depth_specifier


def _hash_index_constructors_specifier(
    ctor_specifier: TIndexCtorSpecifiers | None,
) -> TIndexCtor | tuple[TIndexCtor, ...] | None:
    if isinstance(ctor_specifier, Iterable):
        # If already a tuple, this is a no-op in modern Python.
        return tuple(ctor_specifier)  # type: ignore

    return ctor_specifier  # type: ignore


def _hash_dtypes_specifier(dtypes_specifier: TDtypesSpecifier) -> tp.Hashable:
    if dtypes_specifier is None:
        return dtypes_specifier

    if isinstance(dtypes_specifier, dict):
        return tuple(dtypes_specifier.items())

    if isinstance(dtypes_specifier, Iterable):
        # If already a tuple, this is a no-op in modern Python.
        return tuple(dtypes_specifier)

    return dtypes_specifier


_HASH_HELPERS: dict[str, tp.Callable[[tp.Any], tp.Any]] = dict(
    index_name_depth_level=_hash_depth_specifier,
    index_constructors=_hash_index_constructors_specifier,
    columns_name_depth_level=_hash_depth_specifier,
    columns_constructors=_hash_index_constructors_specifier,
    columns_select=lambda v: None if v is None else tuple(v),
    dtypes=_hash_dtypes_specifier,
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class StoreConfig:
    """
    A read-only container used by :obj:`Store` subclasses for reading from and writing to multi-table storage formats.

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

    _CONSTRUCTOR: ClassVar[tp.Callable[..., TFrameAny]]

    def for_frame_construction_only(self) -> tp.Self:
        # All frame-construction-relevant fields are defined in subclasses.
        to_replace = dict.fromkeys(
            field.name for field in dataclasses.fields(StoreConfig)
        )

        undefined = object()

        for attr, handler in _HASH_HELPERS.items():
            if (value := getattr(self, attr, undefined)) is not undefined:
                to_replace[attr] = handler(value)

        return dataclasses.replace(self, **to_replace)  # type: ignore

    def label_encode(self, label: TLabel) -> str:
        if self.label_encoder is str and isinstance(label, tuple):
            # with NumPy2, str() of a tuple of NumPy scalars returns (np.str_('a'), np.int64(1)), not ('a', 1)
            label = label_encode_tuple(label)

        elif self.label_encoder:
            label = self.label_encoder(label)

        if not isinstance(label, str):
            raise RuntimeError(
                f'Store label {label!r} is not a string; provide a label_encoder to {self.__class__.__name__}.'
            )

        return label

    def label_decode(self, label: str) -> TLabel:
        if self.label_decoder:
            return self.label_decoder(label)

        return label


STORE_CONFIG_DEFAULT = StoreConfig()


@dataclasses.dataclass(frozen=True, kw_only=True)
class StoreConfigFromFrameMixin:
    # Constructors
    index_depth: int = 0  # this default does not permit round trip
    columns_depth: int = 1

    # Exporters
    include_index: bool = True
    include_index_name: bool = True
    include_columns: bool = True
    include_columns_name: bool = False

    @classmethod
    def from_frame(cls, frame: TFrameAny) -> tp.Self:
        """Derive a config from a Frame."""
        include_index = frame.index.depth > 1 or frame.index._map is not None  # type: ignore
        include_columns = frame.columns.depth > 1 or frame.columns._map is not None  # type: ignore

        index_depth = 0 if not include_index else frame.index.depth
        columns_depth = 0 if not include_columns else frame.columns.depth

        return cls(
            index_depth=index_depth,
            columns_depth=columns_depth,
            include_index=include_index,
            include_columns=include_columns,
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class StoreConfigDelimited(StoreConfigFromFrameMixin, StoreConfig):
    # Constructors
    index_name_depth_level: TDepthLevel | None = None
    index_constructors: TIndexCtorSpecifiers = None
    columns_name_depth_level: TDepthLevel | None = None
    columns_constructors: TIndexCtorSpecifiers = None
    dtypes: TDtypesSpecifier = None
    consolidate_blocks: bool = False

    # Both
    store_filter: StoreFilter | None = STORE_FILTER_DEFAULT


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
class StoreConfigParquet(StoreConfigFromFrameMixin, StoreConfig):
    # Constructors
    index_name_depth_level: TDepthLevel | None = None
    index_constructors: TIndexCtorSpecifiers = None
    columns_name_depth_level: TDepthLevel | None = None
    columns_constructors: TIndexCtorSpecifiers = None
    columns_select: tp.Iterable[str] | None = None
    dtypes: TDtypesSpecifier = None
    consolidate_blocks: bool = False

    _CONSTRUCTOR = Frame.from_parquet


@dataclasses.dataclass(frozen=True, kw_only=True)
class StoreConfigNPY(StoreConfig):
    # Exporters
    include_index: bool = True
    include_columns: bool = True
    consolidate_blocks: bool = False


@dataclasses.dataclass(frozen=True, kw_only=True)
class StoreConfigXLSX(StoreConfigFromFrameMixin, StoreConfig):
    # Constructors
    index_name_depth_level: TDepthLevel | None = None
    index_constructors: TIndexCtorSpecifiers = None
    columns_name_depth_level: TDepthLevel | None = None
    columns_constructors: TIndexCtorSpecifiers = None
    columns_select: tp.Iterable[str] | None = None
    dtypes: TDtypesSpecifier = None
    consolidate_blocks: bool = False
    skip_header: int = 0
    skip_footer: int = 0
    trim_nadir: bool = False

    # Exporters
    merge_hierarchical_labels: bool = True

    # Both
    store_filter: StoreFilter | None = STORE_FILTER_DEFAULT

    _CONSTRUCTOR = Frame.from_xlsx


@dataclasses.dataclass(frozen=True, kw_only=True)
class StoreConfigSQLite(StoreConfigFromFrameMixin, StoreConfig):
    # Constructors
    index_constructors: TIndexCtorSpecifiers = None
    columns_select: tp.Iterable[str] | None = None
    columns_constructors: TIndexCtorSpecifiers = None
    dtypes: TDtypesSpecifier = None
    consolidate_blocks: bool = False

    _CONSTRUCTOR = Frame.from_sqlite


TVStoreConfig = tp.TypeVar('TVStoreConfig', bound=StoreConfig, default=StoreConfig)


class FromFrameProtocol(tp.Protocol[TVStoreConfig]):
    @classmethod
    def from_frame(cls, frame: TFrameAny) -> TVStoreConfig: ...


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
    def _infer_default_from_typehint(cls) -> type[TVStoreConfig]:
        for base in tp.get_original_bases(cls):
            if tp.get_origin(base) is None:
                continue

            match tp.get_args(base):
                case (config_type,) if isinstance(config_type, type) and issubclass(
                    config_type, StoreConfig
                ):
                    return config_type  # type: ignore
                case _:
                    pass

        # This branch is only hit when: StoreConfigMap.from_initializer(None)
        # User should not build StoreConfigMap's this way!
        raise ErrorInitStoreMapConfig(
            'Disallowed construction; cannot infer StoreConfigMap default type from None. '
            'To get a default, simply construct the StoreConfig subclass without args (e.g. StoreConfigCSV())'
        )

    @staticmethod
    def from_frames(
        frames: tp.Iterable[TFrameAny], *, config_class: FromFrameProtocol[TVStoreConfig]
    ) -> StoreConfigMap[TVStoreConfig]:
        config_map: dict[tp.Any, TVStoreConfig] = {}

        for f in frames:
            config_map[f.name] = config_class.from_frame(f)

        return StoreConfigMap[TVStoreConfig](config_map=config_map)

    @staticmethod
    def from_initializer(
        initializer: TVStoreConfigMapInitializer[TVStoreConfig],
    ) -> StoreConfigMap[TVStoreConfig]:
        if initializer is None:
            return StoreConfigMap[TVStoreConfig]()

        if isinstance(initializer, StoreConfig):
            return StoreConfigMap[TVStoreConfig](default=initializer)

        if isinstance(initializer, StoreConfigMap):
            return initializer

        if not isinstance(initializer, Mapping):
            raise ErrorInitStoreConfig(
                f'Unsupported initializer type: {type(initializer)}'
            )

        return StoreConfigMap[TVStoreConfig](config_map=initializer)

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

                if not isinstance(config_type, type) or not issubclass(
                    config_type, StoreConfig
                ):
                    raise ErrorInitStoreConfig(
                        'Mapping config values must be subclasses of StoreConfig!'
                    )

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
                            f'config {label!r} has {attr} inconsistent with default; align values and/or pass a default StoreConfig.'
                        )

                self._map[label] = config

        if default is None:
            # Infer the default type from the typeint
            self._default = self._infer_default_from_typehint()
        else:
            if not isinstance(default, StoreConfig):
                raise ErrorInitStoreConfig(
                    'Default config must be a StoreConfig instance!'
                )

            self._default = default  # type: ignore

    def __getitem__(self, key: TLabel | None) -> TVStoreConfig:
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
