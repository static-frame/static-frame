import typing as tp
import zipfile
import pickle
from io import StringIO

from static_frame.core.util import AnyCallable
from static_frame.core.store import Store
from static_frame.core.store import StoreConfigConstructor
from static_frame.core.store import StoreConfigExporterInitializer
from static_frame.core.store import StoreConfigs

from static_frame.core.frame import  Frame

class _StoreZip(Store):

    _EXT: tp.FrozenSet[str] =  frozenset(('.zip',))

    _EXT_CONTAINED: str = ''

    _EXPORTER: AnyCallable
    _CONSTRUCTOR: tp.Callable[..., Frame]

    def labels(self, strip_ext: bool = True) -> tp.Iterator[str]:
        with zipfile.ZipFile(self._fp) as zf:
            for name in zf.namelist():
                if strip_ext:
                    yield name.replace(self._EXT_CONTAINED, '')
                else:
                    yield name

class _StoreZipDelimited(_StoreZip):

    def read(self,
            label: str,
            config: StoreConfigConstructor = StoreConfigs.DEFAULT_CONSTRUCTOR,
            ) -> Frame:
        # NOTE: labels need to be strings
        with zipfile.ZipFile(self._fp) as zf:
            src = StringIO()
            # labels may not be present
            src.write(zf.read(label + self._EXT_CONTAINED).decode())
            src.seek(0)
            # call from class to explicitly pass self as frame
            # NOTE: assuming single index, single columns; this will not be valid for IndexHierarchy
            return self.__class__._CONSTRUCTOR(src, index_depth=1)

    def write(self,
            items: tp.Iterable[tp.Tuple[str, Frame]],
            config: StoreConfigExporterInitializer = StoreConfigs.DEFAULT_EXPORTER
            ) -> None:

        with zipfile.ZipFile(self._fp, 'w', zipfile.ZIP_DEFLATED) as zf:
            for label, frame in items:
                dst = StringIO()
                # call from class to explicitly pass self as frame
                self.__class__._EXPORTER(frame, dst)
                dst.seek(0)
                # this will write it without a container
                zf.writestr(label + self._EXT_CONTAINED, dst.read())


class StoreZipTSV(_StoreZipDelimited):
    '''
    Store of TSV files contained within a ZIP file.
    '''
    _EXT_CONTAINED = '.txt'
    # by defualt this will write include the index and columns
    _EXPORTER = Frame.to_tsv
    _CONSTRUCTOR = Frame.from_tsv

class StoreZipCSV(_StoreZipDelimited):
    '''
    Store of CSV files contained within a ZIP file.
    '''
    _EXT_CONTAINED = '.csv'
    # NOTE: defaults may not result in intended index
    _EXPORTER = Frame.to_csv
    _CONSTRUCTOR = Frame.from_csv


class StoreZipPickle(_StoreZip):
    '''A zip of pickles, permitting incremental loading of Frames.
    '''

    _EXT_CONTAINED = '.pickle'

    def read(self,
            label: str,
            config: StoreConfigConstructor = StoreConfigs.DEFAULT_CONSTRUCTOR,
            ) -> Frame:
        with zipfile.ZipFile(self._fp) as zf:
            return tp.cast(Frame, pickle.loads(zf.read(label + self._EXT_CONTAINED)))

    def write(self,
            items: tp.Iterable[tp.Tuple[str, Frame]],
            config: StoreConfigExporterInitializer = StoreConfigs.DEFAULT_EXPORTER
            ) -> None:

        with zipfile.ZipFile(self._fp, 'w', zipfile.ZIP_DEFLATED) as zf:
            for label, frame in items:
                zf.writestr(label + self._EXT_CONTAINED, pickle.dumps(frame))



