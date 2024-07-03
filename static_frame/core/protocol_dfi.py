from __future__ import annotations

import numpy as np
import typing_extensions as tp

from static_frame.core.protocol_dfi_abc import Buffer
from static_frame.core.protocol_dfi_abc import CategoricalDescription
from static_frame.core.protocol_dfi_abc import Column
from static_frame.core.protocol_dfi_abc import ColumnBuffers
from static_frame.core.protocol_dfi_abc import ColumnNullType
from static_frame.core.protocol_dfi_abc import DataFrame
from static_frame.core.protocol_dfi_abc import DlpackDeviceType
from static_frame.core.protocol_dfi_abc import Dtype
from static_frame.core.protocol_dfi_abc import DtypeKind
from static_frame.core.util import NAT
from static_frame.core.util import NULL_SLICE

if tp.TYPE_CHECKING:
    from static_frame import Frame  # pragma: no cover
    from static_frame.core.index_base import IndexBase  # pragma: no cover
    TNDArrayAny = np.ndarray[tp.Any, tp.Any] #pragma: no cover
    TDtypeAny = np.dtype[tp.Any] #pragma: no cover
    TFrameAny = Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]] #pragma: no cover

NP_KIND_TO_DFI_KIND = {
    'i': DtypeKind.INT,
    'u': DtypeKind.UINT,
    'f': DtypeKind.FLOAT,
    'b': DtypeKind.BOOL,
    'U': DtypeKind.STRING,
    'M': DtypeKind.DATETIME,
    'm': DtypeKind.DATETIME,
}

class ArrowCType:
    """
    Enum for Apache Arrow C type format strings.
    The Arrow C data interface:
    https://arrow.apache.org/docs/format/CDataInterface.html#data-type-description-format-strings
    """

    KIND_ITEMSIZE_TO_FORMAT = {
        ('b', 1): 'b',

        ('i', 1): 'c',
        ('i', 2): 's',
        ('i', 4): 'i',
        ('i', 8): 'l',

        ('u', 1): 'C',
        ('u', 2): 'S',
        ('u', 4): 'I',
        ('u', 8): 'L',

        ('f', 2): 'e',
        ('f', 4): 'f',
        ('f', 8): 'g',
    }

    NP_DT64_UNIT_TO_RESOLUTION = {
        's': 's',
        'ms': 'm',
        'us': 'u',
        'ns': 'n',
    }

    @classmethod
    def from_dtype(cls, dtype: TDtypeAny) -> str:
        kind = dtype.kind
        if kind == 'O':
            raise NotImplementedError('no support for object arrays')
        elif kind in ('U', 'S'):
            return 'u'
        elif kind in ('M', 'm'): # dt64
            prefix = 'tt' if kind == 'M' else 'tD' # timestamp or delta
            unit = np.datetime_data(dtype)[0]
            if unit == 'D': # a day
                return 'tdm' # NOTE: there is also tdD, date32

            try:
                res = cls.NP_DT64_UNIT_TO_RESOLUTION[unit]
            except KeyError as e:
                raise NotImplementedError(f'no support for datetime64 unit: {dtype}') from e
            return f'{prefix}{res}'

        try:
            return cls.KIND_ITEMSIZE_TO_FORMAT[(kind, dtype.itemsize)] # pyright: ignore
        except KeyError as e:
            raise NotImplementedError(f'no support for dtype: {dtype}') from e


def np_dtype_to_dfi_dtype(dtype: TDtypeAny) -> Dtype:
    return (NP_KIND_TO_DFI_KIND[dtype.kind],
            dtype.itemsize * 8, # bits!
            ArrowCType.from_dtype(dtype),
            '=',
            )


class DFIBuffer(Buffer):
    __slots__ = (
        '_array',
        )

    def __init__(self, array: TNDArrayAny) -> None:
        self._array = array

        # NOTE: would expect transformation upstream to avoid reproducing the same contiguous buffer on repeated calls;expect 1D arrays to have the same value for C_CONTIGUOUS and F_CONTIGUOUS
        if self._array.ndim != 1 or not self._array.flags['F_CONTIGUOUS']:
            raise ValueError('provided array is not contiguous')

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: shape={self._array.shape} dtype={self._array.dtype.str}>'

    def __array__(self, dtype: TDtypeAny | None = None) -> TNDArrayAny:
        '''
        Support the __array__ interface, returning an array of values.
        '''
        if dtype is None:
            return self._array
        return self._array.astype(dtype)

    @property
    def bufsize(self) -> int:
        return self._array.nbytes

    @property
    def ptr(self) -> int:
        return self._array.__array_interface__['data'][0] # type: ignore

    def __dlpack__(self) -> tp.Any:
        raise NotImplementedError("__dlpack__")

    def __dlpack_device__(self) -> tp.Tuple[DlpackDeviceType, tp.Optional[int]]:
        return (DlpackDeviceType.CPU, None)


#-------------------------------------------------------------------------------
class DFIColumn(Column):
    __slots__ = (
            '_array',
            '_index',
            )

    def __init__(self,
            array: TNDArrayAny,
            index: 'IndexBase',
            ):
        assert len(array) == len(index)
        # NOTE: for efficiency, we do not store a Series, but just an array and the index (for metadata)
        self._array = array # always a 1D array
        self._index = index

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: shape={self._array.shape} dtype={self._array.dtype.str}>'

    def __array__(self, dtype: TDtypeAny | None = None) -> TNDArrayAny:
        '''
        Support the __array__ interface, returning an array of values.
        '''
        if dtype is None:
            return self._array
        return self._array.astype(dtype)

    def size(self) -> int:
        return self._array.size

    @property
    def offset(self) -> int:
        return 0

    @property
    def dtype(self) -> Dtype:
        return np_dtype_to_dfi_dtype(self._array.dtype)

    @property
    def describe_categorical(self) -> CategoricalDescription:
        raise TypeError('no categorical dtypes')

    @property
    def describe_null(self) -> tp.Tuple[ColumnNullType, tp.Any]:
        kind = self._array.dtype.kind
        if kind in ('f', 'c'):
            return (ColumnNullType.USE_NAN, None)
        elif kind in ('m', 'M'):
            return (ColumnNullType.USE_SENTINEL, NAT)
        return (ColumnNullType.NON_NULLABLE, None)

    @property
    def null_count(self) -> tp.Optional[int]:
        kind = self._array.dtype.kind
        if kind in ('f', 'c', 'm', 'M'):
            return np.isnan(self._array).sum() # type: ignore
        return 0

    @property
    def metadata(self) -> tp.Dict[str, tp.Any]:
        return {'static-frame.index': self._index}

    def num_chunks(self) -> int:
        return 1

    def get_chunks(self, n_chunks: tp.Optional[int] = None) -> tp.Iterable["DFIColumn"]:
        if n_chunks and n_chunks > 1:
            size = self._array.shape[0]
            step = size // n_chunks

            # adjust for an incomplete chunk
            if size % n_chunks != 0:
                step += 1

            for start in range(0, step * n_chunks, step):
                yield DFIColumn(
                        self._array[start: start + step],
                        self._index.iloc[start: start + step],
                        )
        else:
            yield self

    def get_buffers(self) -> ColumnBuffers:
        kind = self._array.dtype.kind
        if kind in ('f', 'c', 'm', 'M'):
            va = np.isnan(self._array)
            validity = (DFIBuffer(va), np_dtype_to_dfi_dtype(va.dtype))
        else:
            validity = None

        return dict(
                data=(DFIBuffer(self._array), self.dtype),
                validity=validity,
                offsets=None,
                )


#-------------------------------------------------------------------------------
class DFIDataFrame(DataFrame):
    __slots__ = (
            '_frame',
            '_nan_as_null',
            '_allow_copy',
            )

    def __init__(self,
            frame: TFrameAny,
            *,
            nan_as_null: bool = False,
            allow_copy: bool = True,
            recast_blocks: bool = True,
            ):
        '''
        Args:
            nan_as_null: "...intended for the consumer to tell the producer to overwrite null values in the data with ``NaN``"; this is not relevant of StaticFrame does not use bit / byte masks.
            allow_copy: "... that defines whether or not the library is allowed to make a copy of the data"; this is always the case with StaticFrame as data will be made columnar contiguous.
        '''
        # NOTE: we recast internal blocks in be all contiguous columnar, meaning either 1D arrays or 2D arrays in Fortran ordering (which reduces overhead while permitting contiguous columnar slices)
        self._nan_as_null = nan_as_null
        self._allow_copy = allow_copy

        if recast_blocks:
            from static_frame.core.frame import Frame
            self._frame: TFrameAny = Frame(
                    frame._blocks.contiguous_columnar(),
                    index=frame._index,
                    columns=frame._columns,
                    name=frame._name,
                    own_data=True,
                    own_index=True,
                    own_columns=frame.STATIC,
                    )
        else:
            assert frame.STATIC
            self._frame = frame

    def __dataframe__(self,
            nan_as_null: bool = False,
            allow_copy: bool = True,
            ) -> "DFIDataFrame":
        return self.__class__(self._frame,
                nan_as_null=self._nan_as_null,
                allow_copy=self._allow_copy,
                recast_blocks=False,
                )

    def __array__(self, dtype: TDtypeAny | None = None) -> TNDArrayAny:
        '''
        Support the __array__ interface, returning an array of values.
        '''
        return self._frame.__array__(dtype)

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: shape={self._frame.shape}>'

    @property
    def metadata(self) -> tp.Dict[str, tp.Any]:
        return {'static-frame.index': self._frame._index,
                'static-frame.name': self._frame._name,
                }

    def num_columns(self) -> int:
        return self._frame._blocks.shape[1]

    def num_rows(self) -> tp.Optional[int]:
        return self._frame._blocks.shape[0]

    def num_chunks(self) -> int:
        return 1

    def column_names(self) -> tp.Iterable[str]:
        # NOTE: Pandas does not enforce that these are strings
        return self._frame.columns # type: ignore

    def get_column(self, i: int) -> DFIColumn:
        return DFIColumn(self._frame._blocks._extract_array_column(i), self._frame._index)

    def get_column_by_name(self, name: str) -> DFIColumn:
        return self.get_column(self._frame._columns.loc_to_iloc(name)) # type: ignore

    def get_columns(self) -> tp.Iterable[DFIColumn]:
        index = self._frame._index
        yield from (DFIColumn(a, index) for a in self._frame._blocks.iter_columns_arrays())

    def select_columns(self, indices: tp.Sequence[int]) -> "DFIDataFrame":
        if not isinstance(indices, list):
            indices = list(indices)

        return self.__class__(
                self._frame.iloc[NULL_SLICE, indices],
                nan_as_null=self._nan_as_null,
                allow_copy=self._allow_copy,
                recast_blocks=False,
                )

    def select_columns_by_name(self, names: tp.Sequence[str]) -> "DFIDataFrame":
        if not isinstance(names, list):
            names = list(names)

        return self.select_columns(self._frame.columns.loc_to_iloc(names)) # type: ignore

    def get_chunks(self, n_chunks: tp.Optional[int] = None) -> tp.Iterable["DFIDataFrame"]:

        if n_chunks and n_chunks > 1:
            size = self._frame._blocks.shape[0]
            step = size // n_chunks

            # adjust for an incomplete chunk
            if size % n_chunks != 0:
                step += 1

            for start in range(0, step * n_chunks, step):
                yield DFIDataFrame(
                        self._frame.iloc[start: start + step, NULL_SLICE],
                        nan_as_null=self._nan_as_null,
                        allow_copy=self._allow_copy,
                        recast_blocks=True,
                        )
        else:
            yield self


