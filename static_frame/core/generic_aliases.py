import numpy as np
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
TIndexIntDefault = Index[np.int64]
TIndexHierarchyAny = IndexHierarchy[tp.Unpack[tp.Tuple[tp.Any, ...]]]

TSeriesAny = Series[tp.Any, tp.Any]
TSeriesObject = Series[tp.Any, np.object_]
TSeriesHEAny = SeriesHE[tp.Any, tp.Any]


TFrameAny = Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]]  #pragma: no cover
TFrameGOAny = FrameGO[tp.Any, tp.Any] #pragma: no cover
TFrameHEAny = FrameHE[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]]  #pragma: no cover

TBusAny = Bus[tp.Any] #pragma: no cover
