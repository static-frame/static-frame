import typing as tp
import zipfile
import pickle
from io import StringIO
from io import BytesIO


from static_frame.core.exception import ErrorInitStore
from static_frame.core.frame import Frame
from static_frame.core.frame import FrameGO
from static_frame.core.store import Store
from static_frame.core.store import store_coherent_non_write
from static_frame.core.store import store_coherent_write
from static_frame.core.store import StoreConfig
from static_frame.core.store import StoreConfigMap
from static_frame.core.store import StoreConfigMapInitializer
from static_frame.core.util import AnyCallable


class _StoreZip(Store):

    _EXT: tp.FrozenSet[str] = frozenset(('.zip',))
    _EXT_CONTAINED: str = ''

    @store_coherent_non_write
    def labels(self, strip_ext: bool = True) -> tp.Iterator[str]:
        with zipfile.ZipFile(self._fp) as zf:
            for name in zf.namelist():
                if strip_ext:
                    yield name.replace(self._EXT_CONTAINED, '')
                else:
                    yield name

class _StoreZipDelimited(_StoreZip):
    # store attribute of passed-in container_type to use for construction
    _EXPORTER: AnyCallable
    _CONSTRUCTOR_ATTR: str

    @store_coherent_non_write
    def read(self,
            label: str,
            config: tp.Optional[StoreConfig] = None,
            container_type: tp.Type[Frame] = Frame,
            ) -> Frame:

        if config is None:
            raise ErrorInitStore('a StoreConfig is required on delimited Stores')

        # NOTE: labels need to be strings
        with zipfile.ZipFile(self._fp) as zf:
            src = StringIO()
            # labels may not be present
            src.write(zf.read(label + self._EXT_CONTAINED).decode())
            src.seek(0)
            # call from class to explicitly pass self as frame
            constructor = getattr(container_type, self._CONSTRUCTOR_ATTR)
            return tp.cast(Frame, constructor(src,
                    index_depth=config.index_depth,
                    columns_depth=config.columns_depth,
                    dtypes=config.dtypes,
                    name=label,
                    consolidate_blocks=config.consolidate_blocks
                    ))

    @store_coherent_write
    def write(self,
            items: tp.Iterable[tp.Tuple[str, Frame]],
            config: StoreConfigMapInitializer = None
            ) -> None:

        # will create default from None, will pass let a map pass through
        config_map = StoreConfigMap.from_initializer(config)

        with zipfile.ZipFile(self._fp, 'w', zipfile.ZIP_DEFLATED) as zf:
            for label, frame in items:
                c = config_map[label]
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
                zf.writestr(label + self._EXT_CONTAINED, dst.read())


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

    @store_coherent_non_write
    def read(self,
            label: str,
            *,
            config: tp.Optional[StoreConfig] = None,
            container_type: tp.Type[Frame] = Frame,
            ) -> Frame:
        # config does not do anything for pickles
        # if config is not None:
        #     raise ErrorInitStore('cannot use a StoreConfig on pickled Stores')

        with zipfile.ZipFile(self._fp) as zf:
            frame = pickle.loads(zf.read(label + self._EXT_CONTAINED))

            # assume the stored frame is not a FrameGO
            if issubclass(container_type, FrameGO):
                frame = frame.to_frame_go()

            return tp.cast(Frame, frame)

    @store_coherent_write
    def write(self,
            items: tp.Iterable[tp.Tuple[str, Frame]],
            *,
            config: StoreConfigMapInitializer = None
            ) -> None:

        # if config is not None:
        #     raise ErrorInitStore('cannot use a StoreConfig on pickled Stores')

        with zipfile.ZipFile(self._fp, 'w', zipfile.ZIP_DEFLATED) as zf:
            for label, frame in items:
                if isinstance(frame, FrameGO):
                    raise NotImplementedError('convert FrameGO to Frame before pickling.')
                zf.writestr(label + self._EXT_CONTAINED, pickle.dumps(frame))




#-------------------------------------------------------------------------------

class StoreZipParquet(_StoreZip):
    '''A zip of parquet files, permitting incremental loading of Frames.
    '''

    _EXT_CONTAINED = '.parquet'

    @store_coherent_non_write
    def read(self,
            label: str,
            *,
            config: tp.Optional[StoreConfig] = None,
            container_type: tp.Type[Frame] = Frame,
            ) -> Frame:

        if config is None:
            raise ErrorInitStore('a StoreConfig is required on parquet Stores')

        with zipfile.ZipFile(self._fp) as zf:
            src = BytesIO(zf.read(label + self._EXT_CONTAINED))
            frame = container_type.from_parquet(
                    src,
                    index_depth=config.index_depth,
                    columns_depth=config.columns_depth,
                    dtypes=config.dtypes,
                    name=label,
                    consolidate_blocks=config.consolidate_blocks,
                    )
        return tp.cast(Frame, frame)

    @store_coherent_write
    def write(self,
            items: tp.Iterable[tp.Tuple[str, Frame]],
            *,
            config: StoreConfigMapInitializer = None
            ) -> None:

        config_map = StoreConfigMap.from_initializer(config)

        with zipfile.ZipFile(self._fp, 'w', zipfile.ZIP_DEFLATED) as zf:
            for label, frame in items:
                c = config_map[label]
                dst = BytesIO()
                # call from class to explicitly pass self as frame
                frame.to_parquet(
                        dst,
                        include_index=c.include_index,
                        include_columns=c.include_columns
                        )
                dst.seek(0)
                # this will write it without a container
                zf.writestr(label + self._EXT_CONTAINED, dst.read())



