

import typing as tp
import zipfile
from io import StringIO

from static_frame.core.frame import Frame




class Store:

    __slots__ = (
            '_fp',
            )

    def __init__(self, fp: str):
        self._fp = fp


#-------------------------------------------------------------------------------

class _StoreZip(Store):

    _EXT_CONTAINED: str = '' # extension of contained files

    def labels(self, strip_ext: bool = True) -> tp.Iterator[str]:
        with zipfile.ZipFile(self._fp) as zf:
            for name in zf.namelist():
                if strip_ext:
                    yield name.replace(self._EXT_CONTAINED, '')
                else:
                    yield name

class _StoreZipDelimited(_StoreZip):

    def read(self, label: str) -> Frame:
        # NOTE: labels need to be strings
        with zipfile.ZipFile(self._fp) as zf:
            src = StringIO()
            # labels may not be present
            src.write(zf.read(label + self._EXT_CONTAINED).decode())
            src.seek(0)
            # NOTE: how to handle index hiearchy?
            return self.__class__._CONSTRUCTOR(src, index_column=0)

    def write(self,
            items: tp.Iterable[tp.Tuple[str, Frame]]
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
    Store of TSV files contained within a ZIP file. Incremental loading is supported.
    '''

    _EXT_CONTAINED = '.txt'
    _EXPORTER = Frame.to_tsv
    _CONSTRUCTOR = Frame.from_tsv

class StoreZipCSV(_StoreZipDelimited):
    '''
    Store of CSV files contained within a ZIP file. Incremental loading is supported.
    '''

    _EXT_CONTAINED = '.csv'
    _EXPORTER = Frame.to_csv
    _CONSTRUCTOR = Frame.from_csv


class StoreZipPickle(_StoreZip):
    '''A zip of pickles, permitting incremental loading of Frames.
    '''

    _EXT_CONTAINED = '.p'

    def read(self, label: str) -> Frame:
        with zipfile.ZipFile(self._fp) as zf:
            pass

    def write(self,
            items: tp.Iterable[tp.Tuple[str, Frame]]
            ) -> None:

        with zipfile.ZipFile(self._fp, 'w', zipfile.ZIP_DEFLATED) as zf:
            for label, frame in items:
                pass