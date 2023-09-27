
import numpy as np
import numpy.typing as npt
import typing_extensions as tp

from static_frame.core.container import ContainerOperand
from static_frame.core.index_base import IndexBase

a1: np.ndarray[tp.Any, np.int64]

dt1: = np.dtype[np.int64]

a2: NDArray[np.int64]


# NOTE: flattening of types means that NumPy's shape-first approach is awkward




TDtype = tp.TypeVar('TDtype', bound=np.dtype, default=tp.Any)
class Index(IndexBase, tp.Generic[TDtype]):
    ...

class IndexDate(Index):
    ...

x: Index[np.bool_]
y: IndexDate




TIndices = tp.TypeVarTuple('TIndices', default=tp.Any) # bound=IndexBase,
class IndexHierarchy(IndexBase, tp.Generic[*TIndices]):
    ...

z: IndexHierarchy[Index[np.int64], Index[np.unicode_]]
z: IndexHierarchy[Index[np.int64], Index[np.unicode_], Index[np.float256]]

# NOTE: including Index[dt] instead of just dt becomes more useful in the case of Series and Frame, where it becomes much more readable what are indices and dtypes



TIndex = tp.TypeVar('TIndex', bound=IndexBase, default=tp.Any)
TDtype = tp.TypeVar('TDtype', bound=np.dtype, default=tp.Any)

class Series(ContainerOperand, tp.Generic[TIndex, TDtype]):
    ...

a: Series[Index[np.int64], np.unicode_]
b: Series[IndexHierarchy[IndexDate[np.datetime64], Index[np.int64]], np.float64]


TIndex = tp.TypeVar('TIndex', bound=IndexBase, default=tp.Any)
TColumns = tp.TypeVar('TColumns', bound=IndexBase, default=tp.Any)
TDtypes = tp.TypeVarTuple('TDtype', default=tp.Any) # bound=np.dtype

class Frame(ContainerOperand, tp.Generic[TIndex, TColumns, *TDtypes]):
    ...


# q: Frame[Index[np.int64], Index[np.unicode_], np.bool_, np.unicode_, np.int64]

# r: Frame[tp.Any, tp.Any, *tp.Tuple[np.float64, ...]]

# r2: Frame[tp.Any, tp.Any, np.float64, np.int64]

# s: Frame[
#         IndexHierarchy[IndexDate, Index[np.int64]],
#         Index[np.unicode_],
#         np.bool_,
#         np.unicode_,
#         *tp.Tuple[np.int64, ...],
#         np.bool_,
#         ]


# def proc1(
#     f: Frame[
#         IndexHierarchy[IndexDate, Index[np.int64]],
#         Index[np.str_],
#         np.float64,
#         np.float64,
#         ]
#     ) -> Series[IndexDate, np.float64]:
#     ...



# @sf.validate(
#         Frame.validate(
#         name='scores',
#         columns=Index.validate('code', 'included', ..., 'signal', ...),
#         shape=(20,)),
#         )
# def proc(
#     f: Frame[
#         IndexHierarchy[IndexDate[np.datetime64], Index[np.int64]],
#         Index[np.unicode_],
#         np.unicode_,
#         np.bool_,
#         *tp.Tuple[np.float64, ...]
#         ]
#     ) -> Series[IndexDate[np.datetime64], np.float64]:
#     ...

# # NOTE: could use tp.Annotated to bundle type with a field name

# @sf.validate(
#         x = Series.validate(index=bb_index, shape=len(bb_index)),
#         y = Series.validate(index=bb_index),
#         )







