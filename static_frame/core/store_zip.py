import typing as tp
import zipfile
import pickle
from io import StringIO
from io import BytesIO


from static_frame.core.frame import Frame
from static_frame.core.store import Store
from static_frame.core.store import store_coherent_non_write
from static_frame.core.store import store_coherent_write
from static_frame.core.store import StoreConfigMap
from static_frame.core.store import StoreConfigMapInitializer
from static_frame.core.util import AnyCallable
from static_frame.core.container_util import container_to_exporter_attr

class _StoreZip(Store):

    _EXT: tp.FrozenSet[str] = frozenset(('.zip',))
    _EXT_CONTAINED: str = ''

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


class _StoreZipDelimited(_StoreZip):
    # store attribute of passed-in container_type to use for construction
    _EXPORTER: AnyCallable
    _CONSTRUCTOR_ATTR: str

    @store_coherent_non_write
    def read_many(self,
            labels: tp.Iterable[tp.Hashable],
            *,
            config: StoreConfigMapInitializer = None,
            container_type: tp.Type[Frame] = Frame,
            ) -> tp.Iterator[Frame]:

        config_map = StoreConfigMap.from_initializer(config)

        with zipfile.ZipFile(self._fp) as zf:
            for label in labels:
                c = config_map[label]
                label_encoded = config_map.default.label_encode(label)

                src = StringIO()
                src.write(zf.read(label_encoded + self._EXT_CONTAINED).decode())
                src.seek(0)
                # call from class to explicitly pass self as frame
                constructor = getattr(container_type, self._CONSTRUCTOR_ATTR)
                yield constructor(src,
                        index_depth=c.index_depth,
                        columns_depth=c.columns_depth,
                        dtypes=c.dtypes,
                        name=label,
                        consolidate_blocks=c.consolidate_blocks
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

    @store_coherent_non_write
    def read_many(self,
            labels: tp.Iterable[tp.Hashable],
            *,
            config: StoreConfigMapInitializer = None,
            container_type: tp.Type[Frame] = Frame,
            ) -> tp.Iterator[Frame]:

        config_map = StoreConfigMap.from_initializer(config)
        exporter = container_to_exporter_attr(container_type)

        with zipfile.ZipFile(self._fp) as zf:
            for label in labels:
                label_encoded = config_map.default.label_encode(label)
                frame = pickle.loads(zf.read(label_encoded + self._EXT_CONTAINED))
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

    @store_coherent_non_write
    def read_many(self,
            labels: tp.Iterable[tp.Hashable],
            *,
            config: StoreConfigMapInitializer = None,
            container_type: tp.Type[Frame] = Frame,
            ) -> tp.Iterator[Frame]:

        config_map = StoreConfigMap.from_initializer(config)

        with zipfile.ZipFile(self._fp) as zf:
            for label in labels:
                c = config_map[label]
                label_encoded = config_map.default.label_encode(label)

                src = BytesIO(zf.read(label_encoded + self._EXT_CONTAINED))
                yield container_type.from_parquet(
                        src,
                        index_depth=c.index_depth,
                        columns_depth=c.columns_depth,
                        columns_select=c.columns_select,
                        dtypes=c.dtypes,
                        name=label,
                        consolidate_blocks=c.consolidate_blocks,
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
                        include_columns=c.include_columns,
                        )
                dst.seek(0)
                # this will write it without a container
                zf.writestr(label_encoded + self._EXT_CONTAINED, dst.read())



