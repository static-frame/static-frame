from __future__ import annotations

import os
import pickle
import zipfile
from io import BytesIO
from io import StringIO

import typing_extensions as tp

from static_frame.core.archive_npy import ArchiveFrameConverter
from static_frame.core.archive_npy import ArchiveZipWrapper
from static_frame.core.archive_zip import zip_namelist
from static_frame.core.container_util import container_to_exporter_attr
from static_frame.core.exception import ErrorNPYEncode
from static_frame.core.exception import StoreLabelNonUnique
from static_frame.core.frame import Frame
from static_frame.core.store import Store
from static_frame.core.store import store_coherent_non_write
from static_frame.core.store import store_coherent_write
from static_frame.core.store_config import StoreConfig
from static_frame.core.store_config import StoreConfigHE
from static_frame.core.store_config import StoreConfigMap
from static_frame.core.store_config import StoreConfigMapInitializer
from static_frame.core.util import NOT_IN_CACHE_SENTINEL
from static_frame.core.util import TCallableAny
from static_frame.core.util import TLabel
from static_frame.core.util import get_concurrent_executor

TFrameAny = Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]]
FrameExporter = TCallableAny # Protocol not supported yet...
FrameConstructor = tp.Callable[..., TFrameAny]
LabelAndBytes = tp.Tuple[TLabel, tp.Union[str, bytes]]
IteratorItemsLabelOptionalFrame = tp.Iterator[tp.Tuple[TLabel, tp.Optional[TFrameAny]]]

class PayloadBytesToFrame(tp.NamedTuple):
    '''
    Defines the necessary objects to construct a Frame. Used for multiprocessing.
    '''
    src: bytes
    name: TLabel
    config: StoreConfigHE
    constructor: FrameConstructor

class PayloadFrameToBytes(tp.NamedTuple):
    '''
    Defines the necessary objects to construct writeable Frame bytes. Used for multiprocessing.
    '''
    name: TLabel
    config: StoreConfigHE
    frame: TFrameAny
    exporter: FrameExporter


class _StoreZip(Store):

    _EXT: tp.FrozenSet[str] = frozenset(('.zip',))
    _EXT_CONTAINED: str = ''
    _EXPORTER: TCallableAny

    @classmethod
    def _container_type_to_constructor(cls, container_type: tp.Type[TFrameAny]) -> FrameConstructor:
        raise NotImplementedError

    @staticmethod
    def _build_frame(
            src: bytes,
            name: TLabel,
            config: tp.Union[StoreConfigHE, StoreConfig],
            constructor: FrameConstructor,
            ) -> TFrameAny:
        raise NotImplementedError

    @classmethod
    def _payload_to_frame(cls, payload: PayloadBytesToFrame) -> TFrameAny:
        '''
        Single argument wrapper for _build_frame().
        '''
        return cls._build_frame(
                src=payload.src,
                name=payload.name,
                config=payload.config,
                constructor=payload.constructor,
                )

    @staticmethod
    def _set_container_type(frame: TFrameAny, container_type: tp.Type[TFrameAny]) -> TFrameAny:
        '''
        Helper method to coerce a frame to the expected type, or return it as is
        if the type is already correct
        '''
        if frame.__class__ is not container_type:
            return frame._to_frame(container_type)
        return frame

    @store_coherent_non_write
    def labels(self, *,
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
    def _read_many_single_thread(self,
            labels: tp.Iterable[TLabel],
            *,
            config_map: StoreConfigMap,
            constructor: FrameConstructor,
            container_type: tp.Type[TFrameAny],
            ) -> tp.Iterator[TFrameAny]:
        '''
        Simplified logic path for reading many frames in a single thread, using
        the weak_cache when possible.
        '''
        with zipfile.ZipFile(self._fp) as zf:
            for label in labels:
                # Since the value can be deallocated between lookup & extraction,
                # we have to handle it with `get`` & a sentinel to ensure we
                # don't have a race condition
                cache_lookup = self._weak_cache.get(label, NOT_IN_CACHE_SENTINEL)
                if cache_lookup is not NOT_IN_CACHE_SENTINEL:
                    yield self._set_container_type(cache_lookup, container_type) # type: ignore
                    continue

                c: StoreConfig = config_map[label]

                label_encoded: str = config_map.default.label_encode(label)
                # NOTE: bytes read here are decompressed and CRC checked when using ZipFile; the resulting bytes, downstream, are treated as an uncompressed zip
                src: bytes = zf.read(label_encoded + self._EXT_CONTAINED)

                frame = self._build_frame(
                        src=src,
                        name=label,
                        config=c,
                        constructor=constructor,
                )
                # Newly read frame, add it to our weak_cache
                self._weak_cache[label] = frame
                yield frame

    @store_coherent_non_write
    def read_many(self,
            labels: tp.Iterable[TLabel],
            *,
            config: StoreConfigMapInitializer = None,
            container_type: tp.Type[TFrameAny] = Frame,
            ) -> tp.Iterator[TFrameAny]:

        config_map = StoreConfigMap.from_initializer(config)
        multiprocess: bool = config_map.default.read_max_workers is not None
        constructor: FrameConstructor = self._container_type_to_constructor(container_type)

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
            results: tp.Dict[TLabel, tp.Optional[TFrameAny]] = {}
            for label in labels:
                count_labels += 1
                cache_lookup = self._weak_cache.get(label, NOT_IN_CACHE_SENTINEL)
                if cache_lookup is not NOT_IN_CACHE_SENTINEL:
                    results[label] = self._set_container_type(cache_lookup, container_type) # type: ignore
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
            '''
            This method is synchronized with the following `for label in results_items` loop, as they both share the same necessary & initial condition: `if cached_frame is not None`.
            '''
            with zipfile.ZipFile(self._fp) as zf:
                for label, cached_frame in results_items():
                    if cached_frame is not None:
                        continue

                    c: StoreConfig = config_map[label]
                    label_encoded: str = config_map.default.label_encode(label)
                    src: bytes = zf.read(label_encoded + self._EXT_CONTAINED)

                    yield PayloadBytesToFrame( # pylint: disable=no-value-for-parameter
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
                    frame = next(frame_gen)
                    # Newly read frame, add it to our weak_cache
                    self._weak_cache[label] = frame
                    yield frame

    # --------------------------------------------------------------------------

    @staticmethod
    def _payload_to_bytes(payload: PayloadFrameToBytes) -> LabelAndBytes:
        raise NotImplementedError('implement on derived class') #pragma: no cover

    @store_coherent_write
    def write(self,
            items: tp.Iterable[tp.Tuple[TLabel, TFrameAny]],
            *,
            config: StoreConfigMapInitializer = None,
            compression: int = zipfile.ZIP_DEFLATED,
            ) -> None:
        config_map = StoreConfigMap.from_initializer(config)
        multiprocess = (config_map.default.write_max_workers is not None and
                        config_map.default.write_max_workers > 1)

        def gen() -> tp.Iterable[PayloadFrameToBytes]:
            for label, frame in items:
                yield PayloadFrameToBytes( # pylint: disable=no-value-for-parameter
                        name=label,
                        config=config_map[label].to_store_config_he(),
                        frame=frame,
                        exporter=self.__class__._EXPORTER,
                        )

        if multiprocess:
            pool_executor = get_concurrent_executor(
                    use_threads=False,
                    max_workers=config_map.default.write_max_workers,
                    mp_context=config_map.default.mp_context,
                    )
            def label_and_bytes() -> tp.Iterator[LabelAndBytes]:
                with pool_executor() as executor:
                    yield from executor.map(self._payload_to_bytes,
                            gen(),
                            chunksize=config_map.default.write_chunksize)
        else:
            label_and_bytes = lambda: (self._payload_to_bytes(x) for x in gen())

        labels_encoded = set() # track uniqueness post encoding

        try:
            with zipfile.ZipFile(self._fp,
                    mode='w',
                    compression=compression,
                    allowZip64=True,
                    ) as zf:
                for label, frame_bytes in label_and_bytes():
                    label_encoded = config_map.default.label_encode(label)

                    if label_encoded in labels_encoded:
                        raise StoreLabelNonUnique(label_encoded)
                    labels_encoded.add(label_encoded)

                    zf.writestr(label_encoded + self._EXT_CONTAINED, frame_bytes)
        except ErrorNPYEncode:
            # NOTE: catch NPY failures and remove self._fp to not leave a malformed zip
            if os.path.exists(self._fp):
                os.remove(self._fp)
            raise


class _StoreZipDelimited(_StoreZip):
    # store attribute of passed-in container_type to use for construction
    _EXPORTER_ATTR: str
    _CONSTRUCTOR_ATTR: str

    @classmethod
    def _container_type_to_constructor(cls, container_type: tp.Type[TFrameAny]) -> FrameConstructor:
        return getattr(container_type, cls._CONSTRUCTOR_ATTR) # type: ignore

    @staticmethod
    def _build_frame(
            src: bytes,
            name: TLabel,
            config: tp.Union[StoreConfigHE, StoreConfig],
            constructor: FrameConstructor,
            ) -> TFrameAny:

        return constructor(
            StringIO(src.decode()),
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

    @staticmethod
    def _payload_to_bytes(payload: PayloadFrameToBytes) -> LabelAndBytes:
        c = payload.config

        dst = StringIO()
        # call from class to explicitly pass self as frame
        payload.exporter(payload.frame,
                dst,
                include_index=c.include_index,
                include_index_name=c.include_index_name,
                include_columns=c.include_columns,
                include_columns_name=c.include_columns_name
                )
        return payload.name, dst.getvalue()


class StoreZipTSV(_StoreZipDelimited):
    '''
    Store of TSV files contained within a ZIP file.
    '''
    _EXT_CONTAINED = '.txt'
    _EXPORTER = Frame.to_tsv
    _CONSTRUCTOR_ATTR = Frame.from_tsv.__name__


class StoreZipCSV(_StoreZipDelimited):
    '''
    Store of CSV files contained within a ZIP file.
    '''
    _EXT_CONTAINED = '.csv'
    _EXPORTER = Frame.to_csv
    _CONSTRUCTOR_ATTR = Frame.from_csv.__name__


#-------------------------------------------------------------------------------

class StoreZipPickle(_StoreZip):
    '''A zip of pickles, permitting incremental loading of Frames.
    '''
    _EXT_CONTAINED = '.pickle'
    _EXPORTER = pickle.dumps # NOTE: might be able to use to_pickle

    @classmethod
    def _container_type_to_constructor(cls, container_type: tp.Type[TFrameAny]) -> FrameConstructor:
        return pickle.loads

    @staticmethod
    def _build_frame(
            src: bytes,
            name: TLabel,
            config: tp.Union[StoreConfigHE, StoreConfig],
            constructor: FrameConstructor,
        ) -> TFrameAny:
        return constructor(src)

    @store_coherent_non_write
    def read_many(self,
            labels: tp.Iterable[TLabel],
            *,
            config: StoreConfigMapInitializer = None,
            container_type: tp.Type[TFrameAny] = Frame,
            ) -> tp.Iterator[TFrameAny]:

        exporter = container_to_exporter_attr(container_type)

        for frame in _StoreZip.read_many(self,
                labels,
                config=config,
                container_type=container_type,
                ):
            if frame.__class__ is container_type:
                yield frame
            else:
                yield getattr(frame, exporter)()

    @staticmethod
    def _payload_to_bytes(payload: PayloadFrameToBytes) -> LabelAndBytes:
        return payload.name, payload.exporter(payload.frame)

#-------------------------------------------------------------------------------
class StoreZipNPZ(_StoreZip):
    '''A zip of npz files, permitting incremental loading of Frames.
    '''
    _EXT_CONTAINED = '.npz'
    _EXPORTER = Frame.to_npz

    @classmethod
    def _container_type_to_constructor(cls, container_type: tp.Type[TFrameAny]) -> FrameConstructor:
        return container_type.from_npz

    @staticmethod
    def _build_frame(
            src: bytes,
            name: TLabel,
            config: tp.Union[StoreConfigHE, StoreConfig],
            constructor: FrameConstructor,
        ) -> TFrameAny:
        return constructor(
            BytesIO(src),
            )

    @staticmethod
    def _payload_to_bytes(payload: PayloadFrameToBytes) -> LabelAndBytes:
        c = payload.config
        dst = BytesIO()
        payload.exporter(payload.frame,
                dst,
                include_index=c.include_index,
                include_columns=c.include_columns,
                )
        return payload.name, dst.getvalue()

#-------------------------------------------------------------------------------

class StoreZipParquet(_StoreZip):
    '''A zip of parquet files, permitting incremental loading of Frames.
    '''
    _EXT_CONTAINED = '.parquet'
    _EXPORTER = Frame.to_parquet

    @classmethod
    def _container_type_to_constructor(cls, container_type: tp.Type[TFrameAny]) -> FrameConstructor:
        return container_type.from_parquet

    @staticmethod
    def _build_frame(
            src: bytes,
            name: TLabel,
            config: tp.Union[StoreConfigHE, StoreConfig],
            constructor: FrameConstructor,
        ) -> TFrameAny:
        return constructor(
            BytesIO(src),
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

    @staticmethod
    def _payload_to_bytes(payload: PayloadFrameToBytes) -> LabelAndBytes:
        c = payload.config

        dst = BytesIO()
        payload.exporter(payload.frame,
                dst,
                include_index=c.include_index,
                include_index_name=c.include_index_name,
                include_columns=c.include_columns,
                include_columns_name=c.include_columns_name,
                )
        return payload.name, dst.getvalue()

#-------------------------------------------------------------------------------
class StoreZipNPY(Store):
    '''A zip of NPY files. This does not presently support multi-processing.
    '''
    _EXT: tp.FrozenSet[str] = frozenset(('.zip',))
    _DELIMITER = '/'

    @store_coherent_write
    def write(self,
            items: tp.Iterable[tp.Tuple[TLabel, TFrameAny]],
            *,
            config: StoreConfigMapInitializer = None,
            compression: int = zipfile.ZIP_DEFLATED,
            ) -> None:
        config_map = StoreConfigMap.from_initializer(config)

        try:
            with zipfile.ZipFile(self._fp,
                    mode='w',
                    compression=compression,
                    allowZip64=True,
                    ) as zf:
                archive = ArchiveZipWrapper(zf,
                        writeable=True,
                        memory_map=False,
                        delimiter=self._DELIMITER,
                        )
                for label, frame in items:
                    c: StoreConfig = config_map[label]
                    archive.prefix = config_map.default.label_encode(label) # mutate
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
    def labels(self, *,
            config: StoreConfigMapInitializer = None,
            strip_ext: bool = True, # not used
            ) -> tp.Iterator[TLabel]:

        config_map = StoreConfigMap.from_initializer(config)

        with zipfile.ZipFile(self._fp) as zf:
            archive = ArchiveZipWrapper(zf,
                    writeable=False,
                    memory_map=False,
                    delimiter=self._DELIMITER,
                    )
            # NOTE: this labels() delivers directories of NPY, not individual NPY
            yield from (config_map.default.label_decode(name)
                    for name in archive.labels())

    @store_coherent_non_write
    def read_many(self,
            labels: tp.Iterable[TLabel],
            *,
            config: StoreConfigMapInitializer = None,
            container_type: tp.Type[TFrameAny] = Frame,
            ) -> tp.Iterator[TFrameAny]:

        config_map = StoreConfigMap.from_initializer(config)

        with zipfile.ZipFile(self._fp) as zf:
            archive = ArchiveZipWrapper(zf,
                    writeable=False,
                    memory_map=False,
                    delimiter=self._DELIMITER,
                    )
            for label in labels:
                cache_lookup = self._weak_cache.get(label, NOT_IN_CACHE_SENTINEL)
                if cache_lookup is not NOT_IN_CACHE_SENTINEL:
                    yield _StoreZip._set_container_type(cache_lookup, container_type) # type: ignore
                    continue

                archive.prefix = config_map.default.label_encode(label) # mutate
                frame = ArchiveFrameConverter.frame_decode(
                            archive=archive,
                            constructor=container_type,
                            )
                # Newly read frame, add it to our weak_cache
                self._weak_cache[label] = frame
                yield frame
