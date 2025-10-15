from __future__ import annotations

import io
import os
import pickle
import zipfile
from contextlib import contextmanager
from functools import partial

import typing_extensions as tp

from static_frame.core.archive_npy import ArchiveFrameConverter, ArchiveZipWrapper
from static_frame.core.archive_zip import zip_namelist
from static_frame.core.container_util import container_to_exporter_attr
from static_frame.core.exception import ErrorNPYEncode, store_label_non_unique_factory
from static_frame.core.frame import Frame
from static_frame.core.store import Store, store_coherent_non_write, store_coherent_write
from static_frame.core.store_config import (
    StoreConfig,
    StoreConfigHE,
    StoreConfigMap,
    StoreConfigMapInitializer,
)
from static_frame.core.util import (
    NOT_IN_CACHE_SENTINEL,
    TCallableAny,
    TLabel,
    get_concurrent_executor,
)


class WriteFrameBytes(tp.Protocol):
    def __call__(
        self, frame: TFrameAny, f: tp.IO[bytes], /
    ) -> None: ...  # pragma: no cover


# class TCallableAny(tp.Protocol):
#     def __call__(self, frame: TFrameAny, f: tp.IO[bytes] | tp.IO[str], /, **kwargs: tp.Any) -> None: ...  # pragma: no cover


TFrameAny = Frame[tp.Any, tp.Any, tp.Unpack[tuple[tp.Any, ...]]]
FrameConstructor = tp.Callable[..., TFrameAny]
LabelAndBytes = tuple[TLabel, bytes]
IteratorItemsLabelOptionalFrame = tp.Iterator[tuple[TLabel, TFrameAny | None]]


class PayloadBytesToFrame(tp.NamedTuple):
    """
    Defines the necessary objects to construct a Frame. Used for multiprocessing.
    """

    src: bytes
    name: TLabel
    config: StoreConfigHE
    constructor: FrameConstructor


class PayloadFrameToBytes(tp.NamedTuple):
    """
    Defines the necessary objects to construct writeable Frame bytes. Used for multiprocessing.
    """

    name: TLabel
    config: StoreConfigHE
    frame: TFrameAny
    exporter: TCallableAny


@contextmanager
def bytes_io_to_str_io(f: tp.IO[bytes]) -> tp.Generator[io.TextIOWrapper, None, None]:
    # Ensure we have a buffered binary stream for TextIOWrapper
    if isinstance(f, io.BufferedIOBase):
        buf = f
    else:
        # wrap raw/writable binary stream in a buffer
        buf = io.BufferedWriter(f)  # type: ignore

    try:
        tw = io.TextIOWrapper(buf, encoding='utf-8', newline='')
        yield tw
    finally:
        tw.flush()
        tw.detach()


class _StoreZip(Store):
    _EXT: frozenset[str] = frozenset(('.zip',))
    _EXT_CONTAINED: str = ''
    _EXPORTER: TCallableAny

    @classmethod
    def _container_type_to_constructor(
        cls, container_type: type[TFrameAny]
    ) -> FrameConstructor:
        raise NotImplementedError

    @staticmethod
    def _build_frame(
        src: bytes,
        name: TLabel,
        config: StoreConfigHE | StoreConfig,
        constructor: FrameConstructor,
    ) -> TFrameAny:
        raise NotImplementedError

    @classmethod
    def _payload_to_frame(cls, payload: PayloadBytesToFrame) -> TFrameAny:
        """
        Single argument wrapper for _build_frame().
        """
        return cls._build_frame(
            src=payload.src,
            name=payload.name,
            config=payload.config,
            constructor=payload.constructor,
        )

    @staticmethod
    def _set_container_type(
        frame: TFrameAny, container_type: type[TFrameAny]
    ) -> TFrameAny:
        """
        Helper method to coerce a frame to the expected type, or return it as is
        if the type is already correct
        """
        if frame.__class__ is not container_type:
            return frame._to_frame(container_type)
        return frame

    @store_coherent_non_write
    def labels(
        self,
        *,
        config: StoreConfigMapInitializer = None,
        strip_ext: bool = True,
    ) -> tp.Iterator[TLabel]:
        config_map = StoreConfigMap.from_initializer(config)

        for name in zip_namelist(self._fp):
            if strip_ext:
                name = name.replace(self._EXT_CONTAINED, '')
            # always use default decoder
            yield config_map.default.label_decode(name)

    @store_coherent_non_write
    def _read_many_single_thread(
        self,
        labels: tp.Iterable[TLabel],
        *,
        config_map: StoreConfigMap,
        constructor: FrameConstructor,
        container_type: type[TFrameAny],
    ) -> tp.Iterator[TFrameAny]:
        """
        Simplified logic path for reading many frames in a single thread, using
        the weak_cache when possible.
        """
        with zipfile.ZipFile(self._fp) as zf:
            for label in labels:
                # Since the value can be deallocated between lookup & extraction,
                # we have to handle it with `get`` & a sentinel to ensure we
                # don't have a race condition
                cache_lookup = self._weak_cache.get(label, NOT_IN_CACHE_SENTINEL)
                if cache_lookup is not NOT_IN_CACHE_SENTINEL:
                    yield self._set_container_type(cache_lookup, container_type)  # type: ignore
                    continue

                c: StoreConfig = config_map[label]

                label_encoded: str = config_map.default.label_encode(label)
                # NOTE: bytes read here are decompressed and CRC checked when using ZipFile; the resulting bytes, downstream, are treated as an uncompressed zip
                src: bytes = zf.read(label_encoded + self._EXT_CONTAINED)

                f = self._build_frame(
                    src=src,
                    name=label,
                    config=c,
                    constructor=constructor,
                )
                if c.read_frame_filter is not None:
                    f = c.read_frame_filter(label, f)
                # Newly read frame, add it to our weak_cache
                self._weak_cache[label] = f
                yield f

    @store_coherent_non_write
    def read_many(
        self,
        labels: tp.Iterable[TLabel],
        *,
        config: StoreConfigMapInitializer = None,
        container_type: type[TFrameAny] = Frame,
    ) -> tp.Iterator[TFrameAny]:
        config_map = StoreConfigMap.from_initializer(config)
        multiprocess: bool = config_map.default.read_max_workers is not None
        constructor: FrameConstructor = self._container_type_to_constructor(
            container_type
        )

        if not multiprocess:
            yield from self._read_many_single_thread(
                labels=labels,
                config_map=config_map,
                constructor=constructor,
                container_type=container_type,
            )
            return

        count_cache: int = 0
        if self._weak_cache:
            count_labels: int = 0
            results: dict[TLabel, TFrameAny | None] = {}
            for label in labels:
                count_labels += 1
                cache_lookup = self._weak_cache.get(label, NOT_IN_CACHE_SENTINEL)
                if cache_lookup is not NOT_IN_CACHE_SENTINEL:
                    results[label] = self._set_container_type(
                        cache_lookup,  # type: ignore
                        container_type,
                    )
                    count_cache += 1
                else:
                    results[label] = None

            def results_items() -> IteratorItemsLabelOptionalFrame:
                yield from results.items()
        else:
            labels = list(labels)

            def results_items() -> IteratorItemsLabelOptionalFrame:
                for label in labels:
                    yield label, None

        # Avoid spinning up a process pool if all requested labels had weakrefs
        if count_cache and count_cache == count_labels:
            for _, frame in results_items():
                assert frame is not None  # mypy
                yield frame
            return

        def gen() -> tp.Iterator[PayloadBytesToFrame]:
            """
            This method is synchronized with the following `for label in results_items` loop, as they both share the same necessary & initial condition: `if cached_frame is not None`.
            """
            with zipfile.ZipFile(self._fp) as zf:
                for label, cached_frame in results_items():
                    if cached_frame is not None:
                        continue

                    c: StoreConfig = config_map[label]
                    label_encoded: str = config_map.default.label_encode(label)
                    src: bytes = zf.read(label_encoded + self._EXT_CONTAINED)

                    yield PayloadBytesToFrame(
                        src=src,
                        name=label,
                        config=c.to_store_config_he(),
                        constructor=constructor,
                    )

        chunksize = config_map.default.read_chunksize
        pool_executor = get_concurrent_executor(
            use_threads=False,
            max_workers=config_map.default.read_max_workers,
            mp_context=config_map.default.mp_context,
        )

        with pool_executor() as executor:
            frame_gen = executor.map(self._payload_to_frame, gen(), chunksize=chunksize)

            for label, cached_frame in results_items():
                if cached_frame is not None:
                    yield cached_frame
                else:
                    f = next(frame_gen)
                    c: StoreConfig = config_map[label]
                    if c.read_frame_filter is not None:
                        f = c.read_frame_filter(label, f)
                    # Newly read frame, add it to our weak_cache
                    self._weak_cache[label] = f
                    yield f

    # --------------------------------------------------------------------------

    @classmethod
    def _payload_to_bytes(cls, payload: PayloadFrameToBytes) -> LabelAndBytes:
        dst = io.BytesIO()
        cls._build_exporter(payload.config)(payload.frame, dst)
        return payload.name, dst.getvalue()

    @classmethod
    def _build_exporter(cls, config: StoreConfigHE | StoreConfig) -> WriteFrameBytes:
        raise NotImplementedError('implement on derived class')  # pragma: no cover

    @classmethod
    def _build_and_verify_zf_label(
        cls, label: TLabel, existing_labels: set[str], config_map: StoreConfigMap
    ) -> str:
        label_encoded = config_map.default.label_encode(label)

        if label_encoded in existing_labels:
            raise store_label_non_unique_factory(label_encoded)

        existing_labels.add(label_encoded)
        return label_encoded + cls._EXT_CONTAINED

    @classmethod
    def _write_into_zip(
        cls,
        fp: str,
        compression: int,
        zf_write_loop: tp.Callable[[zipfile.ZipFile], None],
    ) -> None:
        try:
            with zipfile.ZipFile(
                fp, mode='w', compression=compression, allowZip64=True
            ) as zf:
                zf_write_loop(zf)

        except ErrorNPYEncode:
            # NOTE: catch NPY failures and remove fp to not leave a malformed zip
            if os.path.exists(fp):
                os.remove(fp)
            raise

    def _write_single_process(
        self,
        *,
        items: tp.Iterable[tuple[TLabel, TFrameAny]],
        compression: int,
        config_map: StoreConfigMap,
    ) -> None:
        def zf_write_loop(zf: zipfile.ZipFile) -> None:
            existing_labels: set[str] = set()

            for label, frame in items:
                zf_label = self._build_and_verify_zf_label(
                    label, existing_labels, config_map
                )
                exporter = self._build_exporter(config_map[label])

                with zf.open(zf_label, 'w', force_zip64=True) as f:
                    exporter(frame, f)

        self._write_into_zip(
            fp=self._fp, compression=compression, zf_write_loop=zf_write_loop
        )

    def _write_multi_process(
        self,
        *,
        items: tp.Iterable[tuple[TLabel, TFrameAny]],
        compression: int,
        config_map: StoreConfigMap,
    ) -> None:
        def gen_payloads() -> tp.Iterable[PayloadFrameToBytes]:
            for label, frame in items:
                yield PayloadFrameToBytes(
                    name=label,
                    config=config_map[label].to_store_config_he(),
                    frame=frame,
                    exporter=self.__class__._EXPORTER,
                )

        pool_executor = get_concurrent_executor(
            use_threads=False,
            max_workers=config_map.default.write_max_workers,
            mp_context=config_map.default.mp_context,
        )

        def zf_write_loop(zf: zipfile.ZipFile) -> None:
            existing_labels: set[str] = set()

            with pool_executor() as executor:
                for label, frame_bytes in executor.map(
                    self._payload_to_bytes,
                    gen_payloads(),
                    chunksize=config_map.default.write_chunksize,
                ):
                    zf_label = self._build_and_verify_zf_label(
                        label, existing_labels, config_map
                    )

                    zf.writestr(zf_label, frame_bytes)

        self._write_into_zip(
            fp=self._fp, compression=compression, zf_write_loop=zf_write_loop
        )

    @store_coherent_write
    def write(
        self,
        items: tp.Iterable[tuple[TLabel, TFrameAny]],
        *,
        config: StoreConfigMapInitializer = None,
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> None:
        config_map = StoreConfigMap.from_initializer(config)
        multiprocess = (
            config_map.default.write_max_workers is not None
            and config_map.default.write_max_workers > 1
        )

        if multiprocess:
            func = self._write_multi_process
        else:
            func = self._write_single_process

        func(items=items, compression=compression, config_map=config_map)


class _StoreZipDelimited(_StoreZip):
    # store attribute of passed-in container_type to use for construction
    _EXPORTER_ATTR: str
    _CONSTRUCTOR_ATTR: str

    @classmethod
    def _container_type_to_constructor(
        cls, container_type: type[TFrameAny]
    ) -> FrameConstructor:
        return getattr(container_type, cls._CONSTRUCTOR_ATTR)  # type: ignore

    @staticmethod
    def _build_frame(
        src: bytes,
        name: TLabel,
        config: StoreConfigHE | StoreConfig,
        constructor: FrameConstructor,
    ) -> TFrameAny:
        return constructor(
            io.StringIO(src.decode()),
            index_depth=config.index_depth,
            index_name_depth_level=config.index_name_depth_level,
            index_constructors=config.index_constructors,
            columns_depth=config.columns_depth,
            columns_name_depth_level=config.columns_name_depth_level,
            columns_constructors=config.columns_constructors,
            dtypes=config.dtypes,
            name=name,
            consolidate_blocks=config.consolidate_blocks,
        )

    @classmethod
    def _build_exporter(cls, config: StoreConfigHE | StoreConfig) -> WriteFrameBytes:
        def exporter(frame: TFrameAny, f: tp.IO[bytes], /) -> None:
            with bytes_io_to_str_io(f) as text_wrapper:
                cls._EXPORTER(
                    frame,
                    text_wrapper,
                    include_index=config.include_index,
                    include_index_name=config.include_index_name,
                    include_columns=config.include_columns,
                    include_columns_name=config.include_columns_name,
                )

        return exporter


class StoreZipTSV(_StoreZipDelimited):
    """
    Store of TSV files contained within a ZIP file.
    """

    _EXT_CONTAINED = '.txt'
    _EXPORTER = Frame.to_tsv
    _CONSTRUCTOR_ATTR = Frame.from_tsv.__name__


class StoreZipCSV(_StoreZipDelimited):
    """
    Store of CSV files contained within a ZIP file.
    """

    _EXT_CONTAINED = '.csv'
    _EXPORTER = Frame.to_csv
    _CONSTRUCTOR_ATTR = Frame.from_csv.__name__


# -------------------------------------------------------------------------------


class StoreZipPickle(_StoreZip):
    """A zip of pickles, permitting incremental loading of Frames."""

    _EXT_CONTAINED = '.pickle'
    # NOTE: might be able to use to_pickle
    _EXPORTER = pickle.dumps

    @classmethod
    def _container_type_to_constructor(
        cls, container_type: type[TFrameAny]
    ) -> FrameConstructor:
        return pickle.loads

    @staticmethod
    def _build_frame(
        src: bytes,
        name: TLabel,
        config: StoreConfigHE | StoreConfig,
        constructor: FrameConstructor,
    ) -> TFrameAny:
        return constructor(src)

    @store_coherent_non_write
    def read_many(
        self,
        labels: tp.Iterable[TLabel],
        *,
        config: StoreConfigMapInitializer = None,
        container_type: type[TFrameAny] = Frame,
    ) -> tp.Iterator[TFrameAny]:
        exporter = container_to_exporter_attr(container_type)

        for frame in _StoreZip.read_many(
            self,
            labels,
            config=config,
            container_type=container_type,
        ):
            if frame.__class__ is container_type:
                yield frame
            else:
                yield getattr(frame, exporter)()

    @classmethod
    def _build_exporter(cls, config: StoreConfigHE | StoreConfig) -> WriteFrameBytes:
        def exporter(frame: TFrameAny, f: tp.IO[bytes], /) -> None:
            f.write(cls._EXPORTER(frame))

        return exporter


# -------------------------------------------------------------------------------
class StoreZipNPZ(_StoreZip):
    """A zip of npz files, permitting incremental loading of Frames."""

    _EXT_CONTAINED = '.npz'
    _EXPORTER = Frame.to_npz

    @classmethod
    def _container_type_to_constructor(
        cls, container_type: type[TFrameAny]
    ) -> FrameConstructor:
        return container_type.from_npz

    @staticmethod
    def _build_frame(
        src: bytes,
        name: TLabel,
        config: StoreConfigHE | StoreConfig,
        constructor: FrameConstructor,
    ) -> TFrameAny:
        return constructor(
            io.BytesIO(src),
        )

    @classmethod
    def _build_exporter(cls, config: StoreConfigHE | StoreConfig) -> WriteFrameBytes:
        return partial(
            cls._EXPORTER,
            include_index=config.include_index,
            include_columns=config.include_columns,
        )


# -------------------------------------------------------------------------------


class StoreZipParquet(_StoreZip):
    """A zip of parquet files, permitting incremental loading of Frames."""

    _EXT_CONTAINED = '.parquet'
    _EXPORTER = Frame.to_parquet

    @classmethod
    def _container_type_to_constructor(
        cls, container_type: type[TFrameAny]
    ) -> FrameConstructor:
        return container_type.from_parquet

    @staticmethod
    def _build_frame(
        src: bytes,
        name: TLabel,
        config: StoreConfigHE | StoreConfig,
        constructor: FrameConstructor,
    ) -> TFrameAny:
        return constructor(
            io.BytesIO(src),
            index_depth=config.index_depth,
            index_name_depth_level=config.index_name_depth_level,
            index_constructors=config.index_constructors,
            columns_depth=config.columns_depth,
            columns_name_depth_level=config.columns_name_depth_level,
            columns_constructors=config.columns_constructors,
            columns_select=config.columns_select,
            dtypes=config.dtypes,
            name=name,
            consolidate_blocks=config.consolidate_blocks,
        )

    @classmethod
    def _build_exporter(cls, config: StoreConfigHE | StoreConfig) -> WriteFrameBytes:
        return partial(
            cls._EXPORTER,
            include_index=config.include_index,
            include_index_name=config.include_index_name,
            include_columns=config.include_columns,
            include_columns_name=config.include_columns_name,
        )


# -------------------------------------------------------------------------------
class StoreZipNPY(Store):
    """A zip of NPY files. This does not presently support multi-processing."""

    _EXT: frozenset[str] = frozenset(('.zip',))
    _DELIMITER = '/'

    @store_coherent_write
    def write(
        self,
        items: tp.Iterable[tuple[TLabel, TFrameAny]],
        *,
        config: StoreConfigMapInitializer = None,
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> None:
        config_map = StoreConfigMap.from_initializer(config)

        try:
            with zipfile.ZipFile(
                self._fp,
                mode='w',
                compression=compression,
                allowZip64=True,
            ) as zf:
                archive = ArchiveZipWrapper(
                    zf,
                    writeable=True,
                    memory_map=False,
                    delimiter=self._DELIMITER,
                )
                for label, frame in items:
                    c: StoreConfig = config_map[label]
                    archive.prefix = config_map.default.label_encode(label)  # mutate
                    ArchiveFrameConverter.frame_encode(
                        archive=archive,
                        frame=frame,
                        include_index=c.include_index,
                        include_columns=c.include_columns,
                        consolidate_blocks=c.consolidate_blocks,
                    )
        except ErrorNPYEncode:
            # NOTE: catch NPY failures and remove self._fp to not leave a malformed zip
            if os.path.exists(self._fp):
                os.remove(self._fp)
            raise

    @store_coherent_non_write
    def labels(
        self,
        *,
        config: StoreConfigMapInitializer = None,
        strip_ext: bool = True,  # not used
    ) -> tp.Iterator[TLabel]:
        config_map = StoreConfigMap.from_initializer(config)

        with zipfile.ZipFile(self._fp) as zf:
            archive = ArchiveZipWrapper(
                zf,
                writeable=False,
                memory_map=False,
                delimiter=self._DELIMITER,
            )
            # NOTE: this labels() delivers directories of NPY, not individual NPY
            yield from (
                config_map.default.label_decode(name) for name in archive.labels()
            )

    @store_coherent_non_write
    def read_many(
        self,
        labels: tp.Iterable[TLabel],
        *,
        config: StoreConfigMapInitializer = None,
        container_type: type[TFrameAny] = Frame,
    ) -> tp.Iterator[TFrameAny]:
        config_map = StoreConfigMap.from_initializer(config)

        with zipfile.ZipFile(self._fp) as zf:
            archive = ArchiveZipWrapper(
                zf,
                writeable=False,
                memory_map=False,
                delimiter=self._DELIMITER,
            )
            for label in labels:
                cache_lookup = self._weak_cache.get(label, NOT_IN_CACHE_SENTINEL)
                if cache_lookup is not NOT_IN_CACHE_SENTINEL:
                    yield _StoreZip._set_container_type(cache_lookup, container_type)  # type: ignore
                    continue

                archive.prefix = config_map.default.label_encode(label)  # mutate
                f = ArchiveFrameConverter.frame_decode(
                    archive=archive,
                    constructor=container_type,
                )
                # Newly read frame, add it to our weak_cache
                c: StoreConfig = config_map[label]
                if c.read_frame_filter is not None:
                    f = c.read_frame_filter(label, f)
                self._weak_cache[label] = f
                yield f
