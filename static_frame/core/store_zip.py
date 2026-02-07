from __future__ import annotations

import dataclasses
import io
import os
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
    StoreConfigCSV,
    StoreConfigDelimited,
    StoreConfigNPY,
    StoreConfigNPZ,
    StoreConfigParquet,
    StoreConfigPickle,
    StoreConfigTSV,
    TVStoreConfig,
)
from static_frame.core.util import (
    NOT_IN_CACHE_SENTINEL,
    TLabel,
    get_concurrent_executor,
)

if tp.TYPE_CHECKING:
    from static_frame.core.generic_aliases import TFrameAny

    FrameConstructor: tp.TypeAlias = tp.Callable[..., TFrameAny]
    IteratorItemsLabelOptionalFrame: tp.TypeAlias = tp.Iterator[
        tuple[TLabel, TFrameAny | None]
    ]


class WriteFrameBytes(tp.Protocol):
    def __call__(
        self, frame: TFrameAny, f: tp.IO[bytes], /
    ) -> None: ...  # pragma: no cover


@dataclasses.dataclass(frozen=True, kw_only=True)
class PayloadBytesToFrame(tp.Generic[TVStoreConfig]):
    """
    Defines the necessary objects to construct a Frame. Used for multiprocessing.
    """

    src: bytes
    label: TLabel
    config: TVStoreConfig


@dataclasses.dataclass(frozen=True, kw_only=True)
class PayloadFrameToBytes(tp.Generic[TVStoreConfig]):
    """
    Defines the necessary objects to construct writeable Frame bytes. Used for multiprocessing.
    """

    name: TLabel
    config: TVStoreConfig
    frame: TFrameAny


@contextmanager
def bytes_io_to_str_io(
    bytes_io: tp.IO[bytes],
) -> tp.Generator[io.TextIOWrapper, None, None]:
    """
    A helper context manager that provides a TextIOWrapper around a binary stream.

    Necessary because not all exporters expect a binary stream, but we always write
    to a binary stream when writing to a zip file.
    """
    try:
        tw = io.TextIOWrapper(bytes_io, encoding='utf-8', newline='')
        yield tw
    finally:
        tw.flush()
        tw.detach()


class _StoreZip(Store[TVStoreConfig]):
    _EXT: frozenset[str] = frozenset(('.zip',))
    _EXT_CONTAINED: str = ''

    @classmethod
    def _build_frame(
        cls,
        src: bytes,
        label: TLabel,
        config: TVStoreConfig,
    ) -> TFrameAny:
        raise NotImplementedError

    @classmethod
    def _payload_to_frame(cls, payload: PayloadBytesToFrame[TVStoreConfig]) -> TFrameAny:
        """
        Single argument wrapper for _build_frame().
        """
        return cls._build_frame(
            src=payload.src,
            label=payload.label,
            config=payload.config,
        )

    @store_coherent_non_write
    def labels(
        self,
        *,
        strip_ext: bool = True,
    ) -> tp.Iterator[TLabel]:
        for name in zip_namelist(self._fp):
            if strip_ext:
                name = name.replace(self._EXT_CONTAINED, '')

            # always use default decoder
            yield self._config.default.label_decode(name)

    @store_coherent_non_write
    def _read_many_single_thread(
        self,
        labels: tp.Iterable[TLabel],
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
                    yield cache_lookup  # pyright: ignore
                    continue

                label_encoded: str = self._config.default.label_encode(label)
                # NOTE: bytes read here are decompressed and CRC checked when using ZipFile; the resulting bytes, downstream, are treated as an uncompressed zip
                src: bytes = zf.read(label_encoded + self._EXT_CONTAINED)

                f = self._build_frame(
                    src=src,
                    label=label,
                    config=self._config[label],
                )
                if self._config.default.read_frame_filter is not None:
                    f = self._config.default.read_frame_filter(label, f)
                # Newly read frame, add it to our weak_cache
                self._weak_cache[label] = f
                yield f

    @store_coherent_non_write
    def read_many(self, labels: tp.Iterable[TLabel]) -> tp.Iterator[TFrameAny]:
        multiprocess: bool = self._config.default.read_max_workers is not None

        if not multiprocess:
            yield from self._read_many_single_thread(labels=labels)
            return

        count_cache: int = 0
        if self._weak_cache:
            count_labels: int = 0
            results: dict[TLabel, TFrameAny | None] = {}
            for label in labels:
                count_labels += 1
                cache_lookup = self._weak_cache.get(label, NOT_IN_CACHE_SENTINEL)
                if cache_lookup is not NOT_IN_CACHE_SENTINEL:
                    results[label] = cache_lookup  # type: ignore
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

        def gen() -> tp.Iterator[PayloadBytesToFrame[TVStoreConfig]]:
            """
            This method is synchronized with the following `for label in results_items` loop, as they both share the same necessary & initial condition: `if cached_frame is not None`.
            """
            with zipfile.ZipFile(self._fp) as zf:
                for label, cached_frame in results_items():
                    if cached_frame is not None:
                        continue

                    label_encoded: str = self._config.default.label_encode(label)
                    src: bytes = zf.read(label_encoded + self._EXT_CONTAINED)

                    yield PayloadBytesToFrame(
                        src=src,
                        label=label,
                        config=self._config[label].for_frame_construction_only(),
                    )

        chunksize = self._config.default.read_chunksize
        pool_executor = get_concurrent_executor(
            use_threads=False,
            max_workers=self._config.default.read_max_workers,
            mp_context=self._config.default.mp_context,
        )

        with pool_executor() as executor:
            frame_gen = executor.map(self._payload_to_frame, gen(), chunksize=chunksize)

            for label, cached_frame in results_items():
                if cached_frame is not None:
                    yield cached_frame
                else:
                    f = next(frame_gen)
                    if self._config.default.read_frame_filter is not None:
                        f = self._config.default.read_frame_filter(label, f)
                    # Newly read frame, add it to our weak_cache
                    self._weak_cache[label] = f
                    yield f

    # --------------------------------------------------------------------------

    @classmethod
    def _payload_to_bytes(
        cls,
        payload: PayloadFrameToBytes[TVStoreConfig],
    ) -> tuple[TLabel, bytes]:
        """Export the payload frame to bytes. Necessary for multi-processing"""
        dst = io.BytesIO()
        cls._partial_exporter(payload.config)(payload.frame, dst)
        return payload.name, dst.getvalue()

    @classmethod
    def _partial_exporter(cls, config: TVStoreConfig) -> WriteFrameBytes:
        raise NotImplementedError('implement on derived class')  # pragma: no cover

    @classmethod
    def _encode_and_verify_zf_label(
        cls,
        label: TLabel,
        existing_labels: set[str],
        default_config: TVStoreConfig,
    ) -> str:
        """
        NOTE: this mutates `existing_labels` in place.
        """
        label_encoded = default_config.label_encode(label)

        if label_encoded in existing_labels:
            raise store_label_non_unique_factory(label_encoded)

        existing_labels.add(label_encoded)
        return label_encoded + cls._EXT_CONTAINED

    @classmethod
    def _write_into_zip(
        cls,
        fp: str,
        compression: int,
        write_items_into_zf: tp.Callable[[zipfile.ZipFile], None],
    ) -> None:
        try:
            with zipfile.ZipFile(
                fp, mode='w', compression=compression, allowZip64=True
            ) as zf:
                write_items_into_zf(zf)

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
    ) -> None:
        def write_items_into_zf(zf: zipfile.ZipFile) -> None:
            existing_labels: set[str] = set()

            for label, frame in items:
                zf_label = self._encode_and_verify_zf_label(
                    label, existing_labels, self._config.default
                )
                exporter = self._partial_exporter(self._config[label])

                with zf.open(zf_label, 'w', force_zip64=True) as f:
                    exporter(frame, f)

        self._write_into_zip(
            fp=self._fp,
            compression=compression,
            write_items_into_zf=write_items_into_zf,
        )

    def _write_multi_process(
        self,
        *,
        items: tp.Iterable[tuple[TLabel, TFrameAny]],
        compression: int,
    ) -> None:
        def gen_payloads() -> tp.Iterable[PayloadFrameToBytes[TVStoreConfig]]:
            for label, frame in items:
                yield PayloadFrameToBytes(
                    name=label,
                    config=self._config[label].for_frame_construction_only(),
                    frame=frame,
                )

        pool_executor = get_concurrent_executor(
            use_threads=False,
            max_workers=self._config.default.write_max_workers,
            mp_context=self._config.default.mp_context,
        )

        def write_items_into_zf(zf: zipfile.ZipFile) -> None:
            existing_labels: set[str] = set()

            with pool_executor() as executor:
                for label, frame_bytes in executor.map(
                    self._payload_to_bytes,
                    gen_payloads(),
                    chunksize=self._config.default.write_chunksize,
                ):
                    zf_label = self._encode_and_verify_zf_label(
                        label, existing_labels, self._config.default
                    )

                    zf.writestr(zf_label, frame_bytes)

        self._write_into_zip(
            fp=self._fp,
            compression=compression,
            write_items_into_zf=write_items_into_zf,
        )

    @store_coherent_write
    def write(
        self,
        items: tp.Iterable[tuple[TLabel, TFrameAny]],
        *,
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> None:
        multiprocess = (
            self._config.default.write_max_workers is not None
            and self._config.default.write_max_workers > 1
        )

        write_items = (
            self._write_multi_process if multiprocess else self._write_single_process
        )
        write_items(items=items, compression=compression)


TVStoreConfigDelimited = tp.TypeVar(
    'TVStoreConfigDelimited',
    bound=StoreConfigDelimited,
)


class _StoreZipDelimited(
    tp.Generic[TVStoreConfigDelimited],
    _StoreZip[TVStoreConfigDelimited],
):
    @classmethod
    def _build_frame(
        cls,
        src: bytes,
        label: TLabel,
        config: TVStoreConfigDelimited,
    ) -> TFrameAny:
        return cls._STORE_CONFIG_CLASS._CONSTRUCTOR(
            io.StringIO(src.decode()),
            name=label,
            index_depth=config.index_depth,
            index_name_depth_level=config.index_name_depth_level,
            index_constructors=config.index_constructors,
            columns_depth=config.columns_depth,
            columns_name_depth_level=config.columns_name_depth_level,
            columns_constructors=config.columns_constructors,
            dtypes=config.dtypes,
            consolidate_blocks=config.consolidate_blocks,
            store_filter=config.store_filter,
        )

    @classmethod
    def _partial_exporter(cls, config: TVStoreConfigDelimited) -> WriteFrameBytes:
        def exporter(frame: TFrameAny, f: tp.IO[bytes], /) -> None:
            with bytes_io_to_str_io(f) as text_wrapper:
                cls._EXPORTER(
                    frame,
                    text_wrapper,
                    include_index=config.include_index,
                    include_index_name=config.include_index_name,
                    include_columns=config.include_columns,
                    include_columns_name=config.include_columns_name,
                    store_filter=config.store_filter,
                )

        return exporter


class StoreZipTSV(_StoreZipDelimited[StoreConfigTSV]):
    """
    Store of TSV files contained within a ZIP file.
    """

    _EXT_CONTAINED = '.txt'
    _STORE_CONFIG_CLASS = StoreConfigTSV
    _EXPORTER = Frame.to_tsv


class StoreZipCSV(_StoreZipDelimited[StoreConfigCSV]):
    """
    Store of CSV files contained within a ZIP file.
    """

    _EXT_CONTAINED = '.csv'
    _STORE_CONFIG_CLASS = StoreConfigCSV
    _EXPORTER = Frame.to_csv


# -------------------------------------------------------------------------------


class StoreZipPickle(_StoreZip[StoreConfigPickle]):
    """A zip of pickles, permitting incremental loading of Frames."""

    _EXT_CONTAINED = '.pickle'
    _STORE_CONFIG_CLASS = StoreConfigPickle
    _EXPORTER = Frame.to_pickle

    @classmethod
    def _build_frame(
        cls,
        src: bytes,
        label: TLabel,
        config: StoreConfigPickle,
    ) -> TFrameAny:
        frame = cls._STORE_CONFIG_CLASS._CONSTRUCTOR(src)  # type: ignore[arg-type]

        if frame.name is None:
            frame = frame.rename(label)

        return frame

    @classmethod
    def _partial_exporter(cls, config: StoreConfigPickle) -> WriteFrameBytes:
        return cls._EXPORTER  # type: ignore


class StoreZipNPZ(_StoreZip[StoreConfigNPZ]):
    """A zip of npz files, permitting incremental loading of Frames."""

    _EXT_CONTAINED = '.npz'
    _STORE_CONFIG_CLASS = StoreConfigNPZ
    _EXPORTER = Frame.to_npz

    @classmethod
    def _build_frame(
        cls,
        src: bytes,
        label: TLabel,
        config: StoreConfigNPZ,
    ) -> TFrameAny:
        frame = cls._STORE_CONFIG_CLASS._CONSTRUCTOR(io.BytesIO(src))

        if frame.name is None:
            frame = frame.rename(label)

        return frame

    @classmethod
    def _partial_exporter(cls, config: StoreConfigNPZ) -> WriteFrameBytes:
        return partial(
            cls._EXPORTER,
            include_index=config.include_index,
            include_columns=config.include_columns,
            consolidate_blocks=config.consolidate_blocks,
        )


# -------------------------------------------------------------------------------


class StoreZipParquet(_StoreZip[StoreConfigParquet]):
    """A zip of parquet files, permitting incremental loading of Frames."""

    _EXT_CONTAINED = '.parquet'
    _STORE_CONFIG_CLASS = StoreConfigParquet
    _EXPORTER = Frame.to_parquet

    @classmethod
    def _build_frame(
        cls,
        src: bytes,
        label: TLabel,
        config: StoreConfigParquet,
    ) -> TFrameAny:
        return cls._STORE_CONFIG_CLASS._CONSTRUCTOR(
            io.BytesIO(src),  # type: ignore
            index_depth=config.index_depth,
            index_name_depth_level=config.index_name_depth_level,
            index_constructors=config.index_constructors,
            columns_depth=config.columns_depth,
            columns_name_depth_level=config.columns_name_depth_level,
            columns_constructors=config.columns_constructors,
            columns_select=config.columns_select,
            dtypes=config.dtypes,
            name=label,
            consolidate_blocks=config.consolidate_blocks,
        )

    @classmethod
    def _partial_exporter(cls, config: StoreConfigParquet) -> WriteFrameBytes:
        return partial(
            cls._EXPORTER,
            include_index=config.include_index,
            include_index_name=config.include_index_name,
            include_columns=config.include_columns,
            include_columns_name=config.include_columns_name,
        )


# -------------------------------------------------------------------------------
class StoreZipNPY(Store[StoreConfigNPY]):
    """A zip of NPY files. This does not presently support multi-processing."""

    _EXT: frozenset[str] = frozenset(('.zip',))
    _DELIMITER = '/'
    _STORE_CONFIG_CLASS = StoreConfigNPY

    @store_coherent_write
    def write(
        self,
        items: tp.Iterable[tuple[TLabel, TFrameAny]],
        *,
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> None:
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
                    c = self._config[label]
                    archive.prefix = self._config.default.label_encode(label)  # mutate
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
        strip_ext: bool = True,
    ) -> tp.Iterator[TLabel]:
        with zipfile.ZipFile(self._fp) as zf:
            archive = ArchiveZipWrapper(
                zf,
                writeable=False,
                memory_map=False,
                delimiter=self._DELIMITER,
            )
            # NOTE: this labels() delivers directories of NPY, not individual NPY
            yield from (
                self._config.default.label_decode(name) for name in archive.labels()
            )

    @store_coherent_non_write
    def read_many(self, labels: tp.Iterable[TLabel]) -> tp.Iterator[TFrameAny]:
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
                    yield cache_lookup  # pyright: ignore
                    continue

                archive.prefix = self._config.default.label_encode(label)  # mutate
                f = ArchiveFrameConverter.frame_decode(
                    archive=archive,
                    constructor=Frame,
                )
                # Newly read frame, add it to our weak_cache
                if self._config.default.read_frame_filter is not None:
                    f = self._config.default.read_frame_filter(label, f)
                self._weak_cache[label] = f
                yield f
