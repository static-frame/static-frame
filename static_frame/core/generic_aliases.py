import typing_extensions as tp

from static_frame.core.frame import Frame
from static_frame.core.frame import FrameGO
from static_frame.core.frame import FrameHE
from static_frame.core.index import Index
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.series import Series

TIndexAny = Index[tp.Any]
TIndexHierarchyAny = IndexHierarchy[tp.Unpack[tp.Tuple[tp.Any, ...]]] # type: ignore[type-arg]

TNDIndexAny = tp.Union[TIndexAny, TIndexHierarchyAny]


TSeriesAny = Series[TNDIndexAny, tp.Any]



TFrameAny = Frame[TNDIndexAny, TNDIndexAny, tp.Unpack[tp.Tuple[tp.Any, ...]]] # type: ignore[type-arg] # pylint: disable=W0611 #pragma: no cover
TFrameGOAny = FrameGO[TNDIndexAny, TNDIndexAny, tp.Unpack[tp.Tuple[tp.Any, ...]]] # type: ignore[type-arg] # pylint: disable=W0611 #pragma: no cover
TFrameHEAny = FrameHE[TNDIndexAny, TNDIndexAny, tp.Unpack[tp.Tuple[tp.Any, ...]]] # type: ignore[type-arg] # pylint: disable=W0611 #pragma: no cover
