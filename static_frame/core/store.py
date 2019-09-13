

import typing as tp
import zipfile
from io import StringIO

from static_frame.core.frame import Frame




class Store:
    INCREMENTAL: bool = True # define if this Store can do incremental loads


class _StoreZipDelimited(Store):

    INCREMENTAL = True
    _EXT_CONTAINED: str = ''

    def __init__(self, fp: str):
        self._fp = fp

    def read(self, label: str) -> Frame:
        with zipfile.ZipFile(self._fp) as zf:
            src = StringIO()
            # labels may not be present
            src.write(zf.read(label + self._EXT_CONTAINED).decode())
            src.seek(0)
            # NOTE: how to handle index hiearchy?
            return Frame.from_tsv(src, index_column=0)

    def write(self,
            items: tp.Iterable[tp.Tuple[str, Frame]]
            ) -> None:

        with zipfile.ZipFile(self._fp, 'w',zipfile.ZIP_DEFLATED) as zf:
            for label, frame in items:
                dst = StringIO()
                frame.to_tsv(dst)
                dst.seek(0)
                # this will write it without a container
                zf.writestr(label + self._EXT_CONTAINED, dst.read())


    def labels(self) -> tp.Iterator[str]:
        with zipfile.ZipFile(self._fp) as zf:
            for name in zf.namelist():
                yield name.replace(self._EXT_CONTAINED, '')



class StoreZipTSV(_StoreZipDelimited):
    '''
    Store of TSV files contained within a ZIP file. Incremental loading is supported.
    '''
    _EXT_CONTAINED = '.txt'

class StoreZipCSV(_StoreZipDelimited):
    _EXT_CONTAINED = '.csv'
    '''
    Store of CSV files contained within a ZIP file. Incremental loading is supported.
    '''

class StorePickle(Store):
    pass
