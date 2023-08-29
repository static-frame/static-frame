
import typing_extensions as tpe
import typing as tp

import numpy as np
from static_frame.core.index_base import IndexBase
from static_frame.core.container import ContainerOperand






TDtype = tpe.TypeVar('TDtype', bound=np.dtype, default=tp.Any)
class Index(IndexBase, tp.Generic[TDtype]):
    ...

class IndexDate(Index, tp.Generic[TDtype]):
    ...

x: Index[np.bool_]
y: IndexDate[np.datetime64]






TIndices = tpe.TypeVarTuple('TIndices', default=tp.Any) # bound=IndexBase,
class IndexHierarchy(IndexBase, tp.Generic[*TIndices]):
    ...

z: IndexHierarchy[Index[np.int64], Index[np.unicode_]]

# NOTE: including Index[dt] instead of just dt becomes more useful in the case of Series and Frame, where it becomes much more readable what are indices and dtypes






TIndex = tpe.TypeVar('TIndex', bound=IndexBase, default=tp.Any)
TDtype = tpe.TypeVar('TDtype', bound=np.dtype, default=tp.Any)

class Series(ContainerOperand, tp.Generic[TIndex, TDtype]):
    ...

a: Series[Index[np.int64], np.unicode_]
b: Series[IndexHierarchy[IndexDate[np.datetime64], Index[np.int64]], np.float64]







TIndex = tpe.TypeVar('TIndex', bound=IndexBase, default=tp.Any)
TColumns = tpe.TypeVar('TColumns', bound=IndexBase, default=tp.Any)
TDtypes = tpe.TypeVarTuple('TDtype', default=tp.Any) # bound=np.dtype

class Frame(ContainerOperand, tp.Generic[TIndex, TColumns, *TDtypes]):
    ...


q: Frame[Index[np.int64], Index[np.unicode_], np.bool_, np.unicode_, np.int64]
r: Frame[
        IndexHierarchy[IndexDate[np.datetime64], Index[np.int64]],
        Index[np.unicode_],
        np.bool_,
        np.unicode_,
        np.int64,
        ]





def proc(
    f: Frame[
        IndexHierarchy[IndexDate[np.datetime64], Index[np.int64]],
        Index[np.unicode_],
        np.unicode_,
        np.bool_,
        np.float64,
        ]
    ) -> Series[IndexDate[np.datetime64], np.float64]:
    ...









