
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



# basic SF containers

>>> idx1 = sf.Index(('a', 'b', 'c', 'd'))
>>> idx2 = sf.Index((3, 4, 5, -1))

>>> check_type(idx2, sf.Index[np.int_])


# support for built-in containers, Union

>>> check_type((idx1, idx2), tp.Tuple[sf.Index[np.str_], sf.Index[np.int_]])
>>> check_type((idx1, idx2), tp.Tuple[sf.Index[np.str_], sf.Index[np.int_], int]) # fails

# union

>>> check_type(dict(a=idx1, b=idx2), tp.Dict[str, tp.Union[sf.Index[np.str_], sf.Index[np.int_]]])
>>> check_type(dict(a=idx1, b=idx2), tp.Dict[str, tp.Union[sf.Index[np.float64], sf.Index[np.complex128]]]) # fails


# literals

>>> check_type(dict(a=idx1, b=idx2), tp.Dict[tp.Literal['a', 'b'], tp.Union[sf.Index[np.str_], sf.Index[np.int_]]])

>>> check_type(dict(a=idx1, b=idx2), tp.Dict[tp.Literal['a', 'x'], tp.Union[sf.Index[np.str_], sf.Index[np.int_]]]) # fails


# using an annotation for additional checks

>>> check_type(idx1, tp.Annotated[sf.Index[np.str_], Len(4)])
>>> check_type(idx1, tp.Annotated[sf.Index[np.str_], Len(2)]) # fails


>>> check_type(idx1, tp.Annotated[sf.Index[np.str_], Labels(('a', ..., 'd'))])
>>> check_type(idx1, tp.Annotated[sf.Index[np.str_], Labels(('a', ..., 'c', 'd'))])
>>> check_type(idx1, tp.Annotated[sf.Index[np.str_], Labels(('a', 'b', ...))])

>>> check_type(idx1, tp.Annotated[sf.Index[np.str_], Validator(lambda idx: 'c' in idx)])
>>> check_type(idx1, tp.Annotated[sf.Index[np.str_], Validator(lambda idx: 'q' in idx)]) # fails


# skip over IndexHierarchy, Series


>>> f = sf.Frame.from_records(((10.4, False), (30.2, True)), index=('a', 'b'), columns=sf.IndexYear(('1543', '1533')))

>>> check_type(f, sf.Frame[sf.Index[np.str_], sf.IndexYear, np.float64, np.bool_])
>>> check_type(f, sf.Frame[sf.Index[np.str_], sf.IndexDate, np.float64, np.bool_]) # fails
>>> check_type(f, sf.Frame[tp.Annotated[sf.Index[np.str_], Labels(('a', ...))], sf.IndexYear, np.float64, np.bool_])


>>> u1 = tp.Union[str, int]
>>> u2 = str | int
>>>
>>> type(u1)
typing._UnionGenericAlias
>>> type(u2)
types.UnionType
>>>
>>>
>>> tp.get_origin(u1)
typing.Union
>>> tp.get_args(u1)
(str, int)


>>> hf = sf.Frame[sf.Index[np.str_], sf.IndexDate, np.float64, np.bool_]
>>> type(hf)
typing._GenericAlias
>>> tp.get_origin(hf)
static_frame.core.frame.Frame
>>> tp.get_args(hf)
(static_frame.core.index.Index[numpy.str_],
 static_frame.core.index_datetime.IndexDate,
 numpy.float64,
 numpy.bool_)
>>> tp.get_args(tp.get_args(hf)[0])
(numpy.str_,)



# Implementation of _iter_errors


# index: q, r, a, b
# hint: ..., a, b
# comparisons
# q, ..., look ahead to a
# r, ..., look ahead to a
# a, ..., look ahead to a, validate a, advance two
# b,


# a, b, c, d
# a, ...




# index: a, b, c, d
# a, ..., d

a -> a
b -> ...
c -> ...
d -> ...



































































































