from __future__ import annotations

import os
from functools import partial
from functools import wraps
from itertools import chain
from weakref import WeakValueDictionary

import numpy as np
import typing_extensions as tp

from static_frame.core.exception import ErrorInitStore
from static_frame.core.exception import StoreFileMutation
from static_frame.core.exception import StoreParameterConflict
from static_frame.core.frame import Frame
from static_frame.core.store_config import StoreConfig
from static_frame.core.store_config import StoreConfigMapInitializer
from static_frame.core.util import TCallableAny
from static_frame.core.util import TLabel
from static_frame.core.util import TPathSpecifier
from static_frame.core.util import path_filter

if tp.TYPE_CHECKING:
    TNDArrayAny = np.ndarray[tp.Any, tp.Any] #pragma: no cover
    TDtypeAny = np.dtype[tp.Any] #pragma: no cover

TFrameAny = Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]]

#-------------------------------------------------------------------------------
# decorators

def store_coherent_non_write(f: TCallableAny) -> TCallableAny:

    @wraps(f)
    def wrapper(self: 'Store', *args: tp.Any, **kwargs: tp.Any) -> TFrameAny:
        '''Decprator for derived Store class implementation of read(), labels().
        '''
        self._mtime_coherent()
        return f(self, *args, **kwargs) # type: ignore

    return wrapper


def store_coherent_write(f: TCallableAny) -> TCallableAny:
    '''Decorator for derived Store classes implementation of write()
    '''
    @wraps(f)
    def wrapper(self: 'Store', *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        post = f(self,  *args, **kwargs)
        self._mtime_update()
        return post

    return wrapper

#-------------------------------------------------------------------------------
class Store:

    _EXT: tp.FrozenSet[str]

    __slots__ = (
            '_fp',
            '_last_modified',
            '_weak_cache',
            )

    def __init__(self, fp: TPathSpecifier):
        # Redefine fp variable as only string after the filter.
        fp = tp.cast(str, path_filter(fp))

        if not os.path.splitext(fp)[1] in self._EXT:
            raise ErrorInitStore(
                    f'file path {fp} does not match one of the required extensions: {self._EXT}')

        self._fp: str = fp
        self._last_modified = np.nan
        self._mtime_update()
        self._weak_cache: tp.MutableMapping[TLabel, TFrameAny] = WeakValueDictionary()

    def _mtime_update(self) -> None:
        if os.path.exists(self._fp):
            self._last_modified = os.path.getmtime(self._fp)
        else:
            self._last_modified = np.nan

    def _mtime_coherent(self) -> None:
        '''Raise if a file exists at self._fp and its mtime is not as expected
        '''
        if os.path.exists(self._fp):
            if os.path.getmtime(self._fp) != self._last_modified:
                raise StoreFileMutation(f'file {self._fp} was unexpectedly changed')
        elif not np.isnan(self._last_modified):
            # file existed previously and we got a modification time, but now it does not exist
            raise StoreFileMutation(f'expected file {self._fp} no longer exists')

    def __getstate__(self) -> tp.Tuple[None, tp.Dict[str, tp.Any]]:
        # https://docs.python.org/3/library/pickle.html#object.__getstate__
        # staying consistent with __slots__ only objects by using None as first value in tuple
        return (
            None,
            {
                attr: getattr(self, attr)
                for attr in self.__slots__
                if attr != '_weak_cache'
            }
        )

    def __setstate__(self, state: tp.Tuple[None, tp.Dict[str, tp.Any]]) -> None:
        for key, value in state[1].items():
            setattr(self, key, value)
        self._weak_cache = WeakValueDictionary()

    # def __copy__(self) -> 'Store':
    #     '''
    #     Return a new Store instance linked to the same file.
    #     '''
    #     return self.__class__(fp=self._fp)

    #---------------------------------------------------------------------------
    @staticmethod
    def get_field_names_and_dtypes(*,
            frame: TFrameAny,
            include_index: bool,
            include_index_name: bool,
            include_columns: bool,
            include_columns_name: bool,
            force_str_names: bool = False,
            force_brackets: bool = False
            ) -> tp.Tuple[tp.Sequence[str], tp.Sequence[TDtypeAny]]:

        index = frame.index
        columns = frame.columns
        columns_values: tp.Sequence[TLabel] = columns.values # type: ignore

        if include_index_name and include_columns_name:
            raise StoreParameterConflict('cannot include_index_name and include_columns_name with this Store')

        if columns.depth > 1:
            # The str() of an array produces a space-delimited representation that includes list brackets; we could trim these brackets here, but need them for SQLite usage; thus, clients will have to trim if necessary.
            columns_values = tuple(str(c) for c in columns_values)

        field_names: tp.Sequence[TLabel]
        dtypes: tp.List[TDtypeAny]

        if not include_index:
            if include_columns_name:
                raise StoreParameterConflict('cannot include_columns_name when include_index is False')
            dtypes = frame._blocks.dtypes.tolist()
            if include_columns:
                field_names = columns_values
            else: # name fields with integers?
                field_names = range(frame._blocks.shape[1])
        else:
            if index.depth == 1:
                dtypes = [index.dtype] # type: ignore
            else:
                dtypes = index.dtypes.values.tolist() #type: ignore [attr-defined]
            # Get a list to mutate.
            if include_index_name:
                field_names = list(index.names)
            elif include_columns_name and index.depth == columns.depth:
                field_names = list(columns.names)
            elif include_columns_name and index.depth == 1 and columns.depth > 1:
                field_names = [tuple(str(c) for c in columns.names),]
            else: # if index_depth > 1 and not equal t columns_depth
                raise StoreParameterConflict('cannot determine field names over index; set one of include_index_name or include_columns_name')

            # add frame dtypes t0 those from index
            dtypes.extend(frame._blocks.dtypes)

            # add index names in front of column names
            if include_columns:
                field_names.extend(columns_values) # pyright: ignore
            else: # name fields with integers?
                field_names.extend(range(frame._blocks.shape[1])) # pyright: ignore

        field_names_post: tp.Sequence[str]
        if force_str_names:
            field_names_post = [str(n) for n in field_names]
        elif force_brackets:
            def gen() -> tp.Iterator[str]:
                for name in field_names:
                    name_str = str(name)
                    if name_str.startswith('[') and name_str.endswith(']'):
                        yield name_str
                    elif isinstance(name, tuple):
                        # make look like it came from the array
                        yield f'[{" ".join((repr(n) for n in name))}]'
                    else:
                        yield f'[{name_str}]'
            field_names_post = list(gen())
        else:
            field_names_post = field_names # type: ignore

        return field_names_post, dtypes

    @staticmethod
    def _get_row_iterator(
            frame: TFrameAny,
            include_index: bool
            ) -> tp.Callable[[], tp.Iterator[tp.Sequence[tp.Any]]]:

        if include_index:
            index = frame._index
            index_values = index.values

            def values() -> tp.Iterator[tp.Sequence[tp.Any]]:
                for idx, row in enumerate(frame.iter_array(axis=1)):
                    if index.depth > 1:
                        index_row = index_values[idx] # this is an array
                    else:
                        index_row = (index_values[idx],)
                    yield tuple(chain(index_row, row))
            return values

        return partial(frame.iter_array, axis=1) #type: ignore

    @staticmethod
    def get_column_iterator(
            frame: TFrameAny,
            include_index: bool
            ) -> tp.Iterator[TNDArrayAny]:
        if include_index:
            index_depth = frame._index.depth

            if index_depth == 1:
                index_values = frame._index.values
                return chain(
                        (index_values,),
                        frame._blocks.iter_columns_arrays()
                        )
            return chain(
                    (frame._index.values_at_depth(d) for d in range(index_depth)),
                    frame._blocks.iter_columns_arrays()
                    )
        # avoid creating a Series per column by going to blocks
        return frame._blocks.iter_columns_arrays()

    #---------------------------------------------------------------------------
    def read_many(self,
            labels: tp.Iterable[TLabel],
            *,
            config: StoreConfigMapInitializer = None,
            container_type: tp.Type[TFrameAny] = Frame,
            ) -> tp.Iterator[TFrameAny]:
        '''Read many Frame, given by `labels`, from the Store. Return an iterator of instances of `container_type`.
        '''
        raise NotImplementedError() #pragma: no cover

    @store_coherent_non_write
    def read(self,
            label: TLabel,
            *,
            config: tp.Optional[StoreConfig] = None,
            container_type: tp.Type[TFrameAny] = Frame,
            ) -> TFrameAny:
        '''Read a single Frame, given by `label`, from the Store. Return an instance of `container_type`. This is a convenience method using ``read_many``.
        '''
        return next(self.read_many((label,), config=config, container_type=container_type))

    def write(self,
            items: tp.Iterable[tp.Tuple[str, TFrameAny]],
            *,
            config: StoreConfigMapInitializer = None
            ) -> None:
        '''Write all ``Frames`` in the Store.
        '''
        raise NotImplementedError() #pragma: no cover

    def labels(self, *,
            config: StoreConfigMapInitializer = None,
            strip_ext: bool = True,
            ) -> tp.Iterator[TLabel]:
        raise NotImplementedError() #pragma: no cover

