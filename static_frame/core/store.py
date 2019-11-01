

import typing as tp
import zipfile
import pickle
import os
from io import StringIO

from itertools import chain
from functools import partial

import numpy as np


from static_frame.core.frame import Frame
from static_frame.core.exception import ErrorInitStore
from static_frame.core.util import PathSpecifier
from static_frame.core.util import path_filter
# from static_frame.core.util import array2d_to_tuples
from static_frame.core.index_hierarchy import IndexHierarchy



class Store:

    _EXT: tp.FrozenSet[str]

    __slots__ = (
            '_fp',
            )

    def __init__(self, fp: PathSpecifier):
        # Redefine fp variable as only string after the filter.
        fp = tp.cast(str, path_filter(fp))

        if not os.path.splitext(fp)[1] in self._EXT:
            raise ErrorInitStore(
                    f'file path {fp} does not match one of the required extensions: {self._EXT}')

        self._fp: str = fp

    def read(self, label: str) -> Frame:
        raise NotImplementedError()

    def write(self,
            items: tp.Iterable[tp.Tuple[str, Frame]]
            ) -> None:
        raise NotImplementedError()

    def labels(self) -> tp.Iterator[str]:
        raise NotImplementedError()

    @staticmethod
    def get_field_names_and_dtypes(
            frame: Frame,
            include_index: bool,
            include_columns: bool,
            force_str_names: bool = False,
            force_brackets: bool = False
            ) -> tp.Tuple[tp.Sequence[str], tp.Sequence[np.dtype]]:

        index = frame.index
        columns = frame.columns
        columns_values = columns.values

        if columns.depth > 1:
            # The str() of an array produces a space-delimited representation that includes list brackets; we could trim these brackets here, but need them for SQLite usage; thus, clients will have to trim if necessary.
            columns_values = tuple(str(c) for c in columns_values)

        if not include_index:
            dtypes = frame._blocks.dtypes
            if include_columns:
                field_names = columns_values
            else: # name fields with integers?
                field_names = range(frame._blocks.shape[1])
        else:
            if index.depth == 1:
                dtypes = [index.dtype]
            else:
                assert isinstance(index, IndexHierarchy) # for typing
                dtypes = index.dtypes.values.tolist()
            # Get a list to mutate.
            field_names = list(index.names)

            # add fram dtypes tp those from index
            dtypes.extend(frame._blocks.dtypes)

            # add index names in front of column names
            if include_columns:
                field_names.extend(columns_values)
            else: # name fields with integers?
                field_names.extend(range(frame._blocks.shape[1]))

        if force_str_names:
            field_names = [str(n) for n in field_names]
        if force_brackets:
            def gen() -> tp.Iterator[str]:
                for name in field_names:
                    name = str(name)
                    if name.startswith('[') and name.endswith(']'):
                        yield name
                    else:
                        yield f'[{name}]'
            field_names = tuple(gen())

        return field_names, dtypes

    @staticmethod
    def _get_row_iterator(
            frame: Frame,
            include_index: bool
            ) -> tp.Callable[[], tp.Iterator[tp.Sequence[tp.Any]]]:

        if include_index:
            index = frame._index
            index_values = index.values

            def values() -> tp.Iterator[tp.Sequence[tp.Any]]:
                for idx, row in enumerate(frame.iter_array(1)):
                    if index.depth > 1:
                        index_row = index_values[idx] # this is an array
                    else:
                        index_row = (index_values[idx],)
                    yield tuple(chain(index_row, row))
            return values

        return partial(frame.iter_array, 1) #type: ignore

    @staticmethod
    def get_column_iterator(
            frame: Frame,
            include_index: bool
            ) -> tp.Iterator[np.ndarray]:
        if include_index:
            index_values = frame._index.values
            index_depth = frame._index.depth

            if index_depth == 1:
                return chain(
                        (index_values,),
                        frame._blocks.axis_values(0)
                        )
            # this approach is the same as IndexHierarchy.values_at_depth
            return chain(
                    (index_values[:, d] for d in range(index_depth)),
                    frame._blocks.axis_values(0)
                    )
        # avoid creating a Series per column by going to blocks
        return frame._blocks.axis_values(0)


#-------------------------------------------------------------------------------
from static_frame.core.util import AnyCallable

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

    def read(self, label: str) -> Frame:
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

    def read(self, label: str) -> Frame:
        with zipfile.ZipFile(self._fp) as zf:
            return tp.cast(Frame, pickle.loads(zf.read(label + self._EXT_CONTAINED)))

    def write(self,
            items: tp.Iterable[tp.Tuple[str, Frame]]
            ) -> None:

        with zipfile.ZipFile(self._fp, 'w', zipfile.ZIP_DEFLATED) as zf:
            for label, frame in items:
                zf.writestr(label + self._EXT_CONTAINED, pickle.dumps(frame))



