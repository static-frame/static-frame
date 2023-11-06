import typing_extensions as tp

from static_frame.core.bus import Bus
from static_frame.core.frame import Frame
from static_frame.core.frame import FrameGO
from static_frame.core.frame import FrameHE
from static_frame.core.index import Index
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.series import Series
from static_frame.core.series import SeriesHE

TIndexAny = Index[tp.Any]
TIndexHierarchyAny = IndexHierarchy[tp.Unpack[tp.Tuple[tp.Any, ...]]] # type: ignore[type-arg]

TSeriesAny = Series[tp.Any, tp.Any]
TSeriesHEAny = SeriesHE[tp.Any, tp.Any]


TFrameAny = Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]] # type: ignore[type-arg] # pylint: disable=W0611 #pragma: no cover
TFrameGOAny = FrameGO[tp.Any, tp.Any] # pylint: disable=W0611 #pragma: no cover
TFrameHEAny = FrameHE[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]] # type: ignore[type-arg] # pylint: disable=W0611 #pragma: no cover

TBusAny = Bus[tp.Any] # pylint: disable=W0611 #pragma: no cover
