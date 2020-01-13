import typing as tp
import zipfile
import pickle
from io import StringIO

from static_frame.core.util import AnyCallable
from static_frame.core.store import Store
from static_frame.core.store import StoreConfig
from static_frame.core.store import StoreConfigMapInitializer
from static_frame.core.store import StoreConfigMap

from static_frame.core.store import store_coherent_non_write
from static_frame.core.store import store_coherent_write

from static_frame.core.frame import  Frame
from static_frame.core.exception import ErrorInitStore

class _StoreZip(Store):

    _EXT: tp.FrozenSet[str] = frozenset(('.zip',))
    _EXT_CONTAINED: str = ''

    _EXPORTER: AnyCallable
    _CONSTRUCTOR: tp.Callable[..., Frame]

    @store_coherent_non_write
    def labels(self, strip_ext: bool = True) -> tp.Iterator[str]:
        with zipfile.ZipFile(self._fp) as zf:
            for name in zf.namelist():
                if strip_ext:
                    yield name.replace(self._EXT_CONTAINED, '')
                else:
                    yield name

class _StoreZipDelimited(_StoreZip):

    @store_coherent_non_write
    def read(self,
            label: str,
            config: tp.Optional[StoreConfig] = None,
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
            return self.__class__._CONSTRUCTOR(src,
                    index_depth=config.index_depth,
                    columns_depth=config.columns_depth,
                    dtypes=config.dtypes,
                    name=label
                    )

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
                        include_columns=c.include_columns
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
    _CONSTRUCTOR = Frame.from_tsv

class StoreZipCSV(_StoreZipDelimited):
    '''
    Store of CSV files contained within a ZIP file.
    '''
    _EXT_CONTAINED = '.csv'
    _EXPORTER = Frame.to_csv
    _CONSTRUCTOR = Frame.from_csv


class StoreZipPickle(_StoreZip):
    '''A zip of pickles, permitting incremental loading of Frames.
    '''

    _EXT_CONTAINED = '.pickle'

    @store_coherent_non_write
    def read(self,
            label: str,
            *,
            config: tp.Optional[StoreConfig] = None,
            ) -> Frame:
        # config does not do anything for pickles
        # if config is not None:
        #     raise ErrorInitStore('cannot use a StoreConfig on pickled Stores')

        with zipfile.ZipFile(self._fp) as zf:
            return tp.cast(Frame, pickle.loads(zf.read(label + self._EXT_CONTAINED)))

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
                zf.writestr(label + self._EXT_CONTAINED, pickle.dumps(frame))



