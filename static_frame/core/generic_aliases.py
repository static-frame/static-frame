import numpy as np
import typing_extensions as tp

from static_frame.core.bus import Bus
from static_frame.core.frame import Frame, FrameGO, FrameHE
from static_frame.core.index import Index
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.series import Series, SeriesHE

TIndexAny: tp.TypeAlias = Index[tp.Any]
TIndexIntDefault: tp.TypeAlias = Index[np.int64]
TIndexHierarchyAny: tp.TypeAlias = IndexHierarchy[tp.Unpack[tuple[tp.Any, ...]]]

TSeriesAny: tp.TypeAlias = Series[tp.Any, tp.Any]
TSeriesObject: tp.TypeAlias = Series[tp.Any, np.object_]
TSeriesHEAny: tp.TypeAlias = SeriesHE[tp.Any, tp.Any]


TFrameAny: tp.TypeAlias = Frame[tp.Any, tp.Any, tp.Unpack[tuple[tp.Any, ...]]]
TFrameGOAny: tp.TypeAlias = FrameGO[tp.Any, tp.Any]
TFrameHEAny: tp.TypeAlias = FrameHE[tp.Any, tp.Any, tp.Unpack[tuple[tp.Any, ...]]]

TBusAny: tp.TypeAlias = Bus[tp.Any]
