import typing as tp
import zipfile
import pickle
from io import StringIO
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor

from static_frame.core.frame import Frame
from static_frame.core.store import Store
from static_frame.core.store import store_coherent_non_write
from static_frame.core.store import store_coherent_write
from static_frame.core.store import StoreConfigMap
from static_frame.core.store import StoreConfig
from static_frame.core.store import StoreConfigHE
from static_frame.core.store import StoreConfigMapInitializer
from static_frame.core.util import AnyCallable
from static_frame.core.container_util import container_to_exporter_attr


FrameConstructor = tp.Callable[[tp.Any], Frame]


class DeferredFrameInitPayload(tp.NamedTuple):
    '''
    Defines the necessary objects to construct a frame. Used for multiprocessing.
    '''
    src: bytes
    name: tp.Hashable
    config: StoreConfigHE
    constructor: FrameConstructor


class _StoreZip(Store):

    _EXT: tp.FrozenSet[str] = frozenset(('.zip',))
    _EXT_CONTAINED: str = ''

    @classmethod
    def _container_type_to_constructor(cls, container_type: tp.Type[Frame]) -> FrameConstructor:
        raise NotImplementedError

    @classmethod
    def _build_frame(cls,
            src: bytes,
            name: tp.Hashable,
            config: tp.Union[StoreConfigHE, StoreConfig],
            constructor: FrameConstructor,
            ) -> Frame:
        raise NotImplementedError

    @classmethod
    def _build_frame_from_payload(cls, payload: DeferredFrameInitPayload) -> Frame:
        return cls._build_frame(
                src=payload.src,
                name=payload.name,
                config=payload.config,
                constructor=payload.constructor,
        )

    @store_coherent_non_write
    def labels(self, *,
            config: StoreConfigMapInitializer = None,
            strip_ext: bool = True,
            ) -> tp.Iterator[tp.Hashable]:

        config_map = StoreConfigMap.from_initializer(config)

        with zipfile.ZipFile(self._fp) as zf:
            for name in zf.namelist():
                if strip_ext:
                    name = name.replace(self._EXT_CONTAINED, '')
                # always use default decoder
                yield config_map.default.label_decode(name)

    @store_coherent_non_write
    def read_many(self,
            labels: tp.Iterable[tp.Hashable],
            *,
            config: StoreConfigMapInitializer = None,
            container_type: tp.Type[Frame] = Frame,
        ) -> tp.Iterator[Frame]:

        config_map = StoreConfigMap.from_initializer(config)
        multiprocess: bool = config_map.default.read_max_workers is not None
        constructor: FrameConstructor = self._container_type_to_constructor(container_type)

        def gen() -> tp.Iterable[tp.Union[DeferredFrameInitPayload, Frame]]:
            with zipfile.ZipFile(self._fp) as zf:
                for label in labels:
                    c: StoreConfig = config_map[label]

                    label_encoded: str = config_map.default.label_encode(label)
                    src: bytes = zf.read(label_encoded + self._EXT_CONTAINED)

                    if multiprocess:
                        yield DeferredFrameInitPayload( # pylint: disable=no-value-for-parameter
                                src=src,
                                name=label,
                                config=c.to_store_config_he(),
                                constructor=constructor,
                        )
                    else:
                        yield self._build_frame(
                                src=src,
                                name=label,
                                config=c,
                                constructor=constructor,
                        )

        if multiprocess:
            chunksize = config_map.default.read_chunksize

            with ProcessPoolExecutor(max_workers=config_map.default.read_max_workers) as executor:
                yield from executor.map(self._build_frame_from_payload, gen(), chunksize=chunksize)
        else:
            yield from gen() # type: ignore


class _StoreZipDelimited(_StoreZip):
    # store attribute of passed-in container_type to use for construction
    _EXPORTER: AnyCallable
    _CONSTRUCTOR_ATTR: str

    @classmethod
    def _container_type_to_constructor(cls, container_type: tp.Type[Frame]) -> FrameConstructor:
        return getattr(container_type, cls._CONSTRUCTOR_ATTR) # type: ignore

    @classmethod
    def _build_frame(cls,
            src: bytes,
            name: tp.Hashable,
            config: tp.Union[StoreConfigHE, StoreConfig],
            constructor: FrameConstructor,
        ) -> Frame:
        return constructor( # type: ignore
            StringIO(src.decode()),
            index_depth=config.index_depth,
            columns_depth=config.columns_depth,
            dtypes=config.dtypes,
            name=name,
            consolidate_blocks=config.consolidate_blocks,
        )

    @store_coherent_write
    def write(self,
            items: tp.Iterable[tp.Tuple[tp.Hashable, Frame]],
            config: StoreConfigMapInitializer = None
            ) -> None:

        # will create default from None, will pass let a map pass through
        config_map = StoreConfigMap.from_initializer(config)

        with zipfile.ZipFile(self._fp, 'w', zipfile.ZIP_DEFLATED) as zf:
            for label, frame in items:
                c = config_map[label]
                label_encoded = config_map.default.label_encode(label)

                dst = StringIO()
                # call from class to explicitly pass self as frame
                self.__class__._EXPORTER(frame,
                        dst,
                        include_index=c.include_index,
                        include_index_name=c.include_index_name,
                        include_columns=c.include_columns,
                        include_columns_name=c.include_columns_name
                        )
                dst.seek(0)
                # this will write it without a container
                zf.writestr(label_encoded + self._EXT_CONTAINED, dst.read())


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

    @classmethod
    def _container_type_to_constructor(cls, container_type: tp.Type[Frame]) -> FrameConstructor:
        return pickle.loads

    @classmethod
    def _build_frame(cls,
            src: bytes,
            name: tp.Hashable,
            config: tp.Union[StoreConfigHE, StoreConfig],
            constructor: FrameConstructor,
        ) -> Frame:
        return constructor(src)

    @store_coherent_non_write
    def read_many(self,
            labels: tp.Iterable[tp.Hashable],
            *,
            config: StoreConfigMapInitializer = None,
            container_type: tp.Type[Frame] = Frame,
            ) -> tp.Iterator[Frame]:

        exporter = container_to_exporter_attr(container_type)

        for frame in super().read_many(labels, config=config, container_type=container_type):
            if frame.__class__ is container_type:
                yield frame
            else:
                yield getattr(frame, exporter)()

    @store_coherent_write
    def write(self,
            items: tp.Iterable[tp.Tuple[tp.Hashable, Frame]],
            *,
            config: StoreConfigMapInitializer = None
            ) -> None:

        config_map = StoreConfigMap.from_initializer(config)

        with zipfile.ZipFile(self._fp, 'w', zipfile.ZIP_DEFLATED) as zf:
            for label, frame in items:
                label_encoded = config_map.default.label_encode(label)
                zf.writestr(label_encoded + self._EXT_CONTAINED, pickle.dumps(frame))


#-------------------------------------------------------------------------------

class StoreZipParquet(_StoreZip):
    '''A zip of parquet files, permitting incremental loading of Frames.
    '''

    _EXT_CONTAINED = '.parquet'

    @classmethod
    def _container_type_to_constructor(cls, container_type: tp.Type[Frame]) -> FrameConstructor:
        return container_type.from_parquet

    @classmethod
    def _build_frame(cls,
            src: bytes,
            name: tp.Hashable,
            config: tp.Union[StoreConfigHE, StoreConfig],
            constructor: FrameConstructor,
        ) -> Frame:
        return constructor( # type: ignore
            BytesIO(src),
            index_depth=config.index_depth,
            index_name_depth_level=config.index_name_depth_level,
            columns_depth=config.columns_depth,
            columns_name_depth_level=config.columns_name_depth_level,
            columns_select=config.columns_select,
            dtypes=config.dtypes,
            name=name,
            consolidate_blocks=config.consolidate_blocks,
        )

    @store_coherent_write
    def write(self,
            items: tp.Iterable[tp.Tuple[tp.Hashable, Frame]],
            *,
            config: StoreConfigMapInitializer = None
            ) -> None:

        config_map = StoreConfigMap.from_initializer(config)

        with zipfile.ZipFile(self._fp, 'w', zipfile.ZIP_DEFLATED) as zf:
            for label, frame in items:
                c = config_map[label]
                label_encoded = config_map.default.label_encode(label)

                dst = BytesIO()
                # call from class to explicitly pass self as frame
                frame.to_parquet(
                        dst,
                        include_index=c.include_index,
                        include_index_name=c.include_index_name,
                        include_columns=c.include_columns,
                        include_columns_name=c.include_columns_name,
                        )
                dst.seek(0)
                # this will write it without a container
                zf.writestr(label_encoded + self._EXT_CONTAINED, dst.read())
