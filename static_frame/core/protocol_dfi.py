import typing as tp

import numpy as np

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
    from static_frame import Frame  # pylint: disable=W0611 #pragma: no cover
    from static_frame import Index  # pylint: disable=W0611 #pragma: no cover


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
    def from_dtype(cls, dtype: np.dtype) -> str:
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
            return cls.KIND_ITEMSIZE_TO_FORMAT[(kind, dtype.itemsize)]
        except KeyError as e:
            raise NotImplementedError(f'no support for dtype: {dtype}') from e


def np_dtype_to_dfi_dtype(dtype: np.dtype) -> Dtype:
    return (NP_KIND_TO_DFI_KIND[dtype.kind],
            dtype.itemsize * 8, # bits!
            ArrowCType.from_dtype(dtype),
            '=',
            )


class DFIBuffer(Buffer):
    __slots__ = (
        '_array',
        )

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

        # always one dimensional
        assert self._array.ndim == 1

        # NOTE: woud be better to do this transformation upstream to avoid reproducing the same contiguous buffer on repeated calls
        if not self._array.data.contiguous:
            self._array = np.ascontiguousarray(self._array)
            self._array.flags.writeable = False

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: shape={self._array.shape} dtype={self._array.dtype.str}>'

    def __array__(self, dtype: np.dtype = None) -> np.ndarray:
        '''
        Support the __array__ interface, returning an array of values.
        '''
        if dtype is None:
            return self._array
        return self._array.astype(dtype)

    @property
    def bufsize(self) -> int:
        return self._array.nbytes # type: ignore

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
            array: np.ndarray,
            index: 'Index',
            ):
        assert len(array) == len(index)
        # NOTE: for efficiency, we do not store a Series, but just an array and the index (for metadata)
        self._array = array # always a 1D array
        self._index = index

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: shape={self._array.shape} dtype={self._array.dtype.str}>'

    def __array__(self, dtype: np.dtype = None) -> np.ndarray:
        '''
        Support the __array__ interface, returning an array of values.
        '''
        if dtype is None:
            return self._array
        return self._array.astype(dtype)

    def size(self) -> int:
        return self._array.size # type: ignore

    @property
    def offset(self) -> int:
        return 0

    @property
    def dtype(self) -> Dtype:
        return np_dtype_to_dfi_dtype(self._array.dtype) # type: ignore

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
            frame: 'Frame',
            nan_as_null: bool = False,
            allow_copy: bool = True,
            ):
        self._frame = frame
        self._nan_as_null = nan_as_null
        self._allow_copy = allow_copy

    def __dataframe__(self,
            nan_as_null: bool = False,
            allow_copy: bool = True,
            ) -> "DFIDataFrame":
        return self.__class__(self._frame, nan_as_null, allow_copy)

    def __array__(self, dtype: np.dtype = None) -> np.ndarray:
        '''
        Support the __array__ interface, returning an array of values.
        '''
        return self._frame.__array__(dtype)

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: shape={self._frame.shape}>'

    @property
    def metadata(self) -> tp.Dict[str, tp.Any]:
        return {'static-frame.index': self._frame._index}

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
        return self.get_column(self._frame._columns.loc_to_iloc(name))

    def get_columns(self) -> tp.Iterable[DFIColumn]:
        index = self._frame._index
        yield from (DFIColumn(a, index) for a in self._frame._blocks.axis_values(0))

    def select_columns(self, indices: tp.Sequence[int]) -> "DFIDataFrame":
        if not isinstance(indices, list):
            indices = list(indices)

        return self.__class__(
                self._frame.iloc[NULL_SLICE, indices],
                self._nan_as_null,
                self._allow_copy,
                )

    def select_columns_by_name(self, names: tp.Sequence[str]) -> "DFIDataFrame":
        if not isinstance(names, list):
            names = list(names)

        return self.select_columns(self._frame.columns.loc_to_iloc(names))

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
                        self._nan_as_null,
                        self._allow_copy,
                        )
        else:
            yield self



# examples of Pandas implementation

# >>> df = ff.parse('s(4,6)|v(int,bool,float,str,dtns)|i(I,str)|c(I,str)').to_pandas()
# >>> df
#        zZbu   ztsv     zUvW  zkuW                          zmVj    z2Oo
# zZbu -88017  False   694.30  z2Oo 1970-01-01 00:00:00.000058768   84967
# ztsv  92867  False   -72.96  z5l6 1970-01-01 00:00:00.000146284   13448
# zUvW  84967  False  1826.02  zCE3 1970-01-01 00:00:00.000170440  175579
# zkuW  13448  False   604.10  zr4u 1970-01-01 00:00:00.000032395   58768

# >>> df.__dataframe__()
# <pandas.core.interchange.dataframe.PandasDataFrameXchg object at 0x7f2d14cb4a00>
# >>> dfi = df.__dataframe__()
# >>> dfi.metadata()
# Traceback (most recent call last):
#   File "<console>", line 1, in <module>
# TypeError: 'dict' object is not callable
# >>> dfi.metadata
# {'pandas.index': Index(['zZbu', 'ztsv', 'zUvW', 'zkuW'], dtype='object')}
# >>> dfi.num_columns
# <bound method PandasDataFrameXchg.num_columns of <pandas.core.interchange.dataframe.PandasDataFrameXchg object at 0x7f2d14cb4bb0>>
# >>> dfi.num_columns()
# 6
# >>> dfi.num_rows()
# 4
# >>> dfi.num_chunks()
# 1
# >>> dfi.column_names()
# Index(['zZbu', 'ztsv', 'zUvW', 'zkuW', 'zmVj', 'z2Oo'], dtype='object')
# >>> dfi.get_column(0)
# <pandas.core.interchange.column.PandasColumn object at 0x7f2d14cb4af0>

# NOTE: a new instance is created each time
# >>> dfi.get_column_by_name('zUvW')
# <pandas.core.interchange.column.PandasColumn object at 0x7f2d147349a0>
# >>>
# >>> dfi.get_column_by_name('zUvW')
# <pandas.core.interchange.column.PandasColumn object at 0x7f2d14cb4910>

# NOTE: pandas is not lazy; spec says iterator
# >>> dfi.get_columns()
# [<pandas.core.interchange.column.PandasColumn object at 0x7f2d14734c70>, <pandas.core.interchange.column.PandasColumn object at 0x7f2d14734ee0>, <pandas.core.interchange.column.PandasColumn object at 0x7f2d14734e20>, <pandas.core.interchange.column.PandasColumn object at 0x7f2d14718100>, <pandas.core.interchange.column.PandasColumn object at 0x7f2d147182b0>, <pandas.core.interchange.column.PandasColumn object at 0x7f2d14718430>]


# >>> dfi.select_columns([1, 4])
# <pandas.core.interchange.dataframe.PandasDataFrameXchg object at 0x7f2d14cb4af0>

# >>> dfi.select_columns([1, 4]).get_columns()
# [<pandas.core.interchange.column.PandasColumn object at 0x7f2d14734430>, <pandas.core.interchange.column.PandasColumn object at 0x7f2d14718100>]

# >>> dfi.select_columns_by_name(('zmVj',))
# <pandas.core.interchange.dataframe.PandasDataFrameXchg object at 0x7f2d147349a0>
# >>> dfi.select_columns_by_name(('zmVj', 'zUvW'))
# <pandas.core.interchange.dataframe.PandasDataFrameXchg object at 0x7f2d14cb4910>


# >>> tuple(dfi.get_chunks())[0]
# <pandas.core.interchange.dataframe.PandasDataFrameXchg object at 0x7f2d14cb4bb0>
# >>> tuple(dfi.get_chunks())[0].column_names()
# Index(['zZbu', 'ztsv', 'zUvW', 'zkuW', 'zmVj', 'z2Oo'], dtype='object')

