__version__ = '3.2.0'

# We import the names "as" themselves here (and here only) to tell linting tools
# that they are explicitly being exported here (and not just unused).
from arraykit import ErrorInitTypeBlocks as ErrorInitTypeBlocks
from arraykit import isna_element as isna_element
from arraykit import mloc as mloc

from static_frame.core.archive_npy import NPY as NPY
from static_frame.core.archive_npy import NPZ as NPZ
from static_frame.core.batch import Batch as Batch
from static_frame.core.bus import Bus as Bus
from static_frame.core.container import ContainerBase as ContainerBase
from static_frame.core.display import Display as Display
from static_frame.core.display import DisplayActive as DisplayActive
from static_frame.core.display_config import DisplayConfig as DisplayConfig
from static_frame.core.display_config import DisplayConfigs as DisplayConfigs
from static_frame.core.display_config import DisplayFormats as DisplayFormats
from static_frame.core.exception import AxisInvalid as AxisInvalid
from static_frame.core.exception import ErrorInit as ErrorInit
from static_frame.core.exception import ErrorInitBus as ErrorInitBus
from static_frame.core.exception import ErrorInitColumns as ErrorInitColumns
from static_frame.core.exception import ErrorInitFrame as ErrorInitFrame
from static_frame.core.exception import ErrorInitIndex as ErrorInitIndex
from static_frame.core.exception import ErrorInitSeries as ErrorInitSeries
from static_frame.core.exception import ErrorInitStore as ErrorInitStore
from static_frame.core.exception import ErrorInitStoreConfig as ErrorInitStoreConfig
from static_frame.core.exception import LocEmpty as LocEmpty
from static_frame.core.exception import LocInvalid as LocInvalid
from static_frame.core.exception import StoreFileMutation as StoreFileMutation
from static_frame.core.fill_value_auto import FillValueAuto as FillValueAuto
from static_frame.core.frame import Frame as Frame
from static_frame.core.frame import FrameAssign as FrameAssign
from static_frame.core.frame import FrameAssignBLoc as FrameAssignBLoc
from static_frame.core.frame import FrameAssignILoc as FrameAssignILoc
from static_frame.core.frame import FrameGO as FrameGO
from static_frame.core.frame import FrameHE as FrameHE
from static_frame.core.generic_aliases import TBusAny as TBusAny
from static_frame.core.generic_aliases import TFrameAny as TFrameAny
from static_frame.core.generic_aliases import TFrameGOAny as TFrameGOAny
from static_frame.core.generic_aliases import TFrameHEAny as TFrameHEAny
from static_frame.core.generic_aliases import TIndexAny as TIndexAny
from static_frame.core.generic_aliases import TIndexHierarchyAny as TIndexHierarchyAny
from static_frame.core.generic_aliases import TSeriesAny as TSeriesAny
from static_frame.core.generic_aliases import TSeriesHEAny as TSeriesHEAny
from static_frame.core.hloc import HLoc as HLoc
from static_frame.core.index import ILoc as ILoc
from static_frame.core.index import Index as Index
from static_frame.core.index import IndexGO as IndexGO
from static_frame.core.index_auto import (
    IndexAutoConstructorFactory as IndexAutoConstructorFactory,
)
from static_frame.core.index_auto import IndexAutoFactory as IndexAutoFactory
from static_frame.core.index_auto import IndexAutoInitializer as IndexAutoInitializer
from static_frame.core.index_auto import (
    IndexDefaultConstructorFactory as IndexDefaultConstructorFactory,
)
from static_frame.core.index_auto import TIndexAutoFactory as TIndexAutoFactory
from static_frame.core.index_datetime import IndexDate as IndexDate
from static_frame.core.index_datetime import IndexDateGO as IndexDateGO
from static_frame.core.index_datetime import IndexHour as IndexHour
from static_frame.core.index_datetime import IndexHourGO as IndexHourGO
from static_frame.core.index_datetime import IndexMicrosecond as IndexMicrosecond
from static_frame.core.index_datetime import IndexMicrosecondGO as IndexMicrosecondGO
from static_frame.core.index_datetime import IndexMillisecond as IndexMillisecond
from static_frame.core.index_datetime import IndexMillisecondGO as IndexMillisecondGO
from static_frame.core.index_datetime import IndexMinute as IndexMinute
from static_frame.core.index_datetime import IndexMinuteGO as IndexMinuteGO
from static_frame.core.index_datetime import IndexNanosecond as IndexNanosecond
from static_frame.core.index_datetime import IndexNanosecondGO as IndexNanosecondGO
from static_frame.core.index_datetime import IndexSecond as IndexSecond
from static_frame.core.index_datetime import IndexSecondGO as IndexSecondGO
from static_frame.core.index_datetime import IndexYear as IndexYear
from static_frame.core.index_datetime import IndexYearGO as IndexYearGO
from static_frame.core.index_datetime import IndexYearMonth as IndexYearMonth
from static_frame.core.index_datetime import IndexYearMonthGO as IndexYearMonthGO
from static_frame.core.index_hierarchy import IndexHierarchy as IndexHierarchy
from static_frame.core.index_hierarchy import IndexHierarchyGO as IndexHierarchyGO
from static_frame.core.interface_meta import InterfaceMeta as InterfaceMeta
from static_frame.core.memory_measure import MemoryDisplay as MemoryDisplay
from static_frame.core.node_dt import InterfaceBatchDatetime as InterfaceBatchDatetime
from static_frame.core.node_dt import InterfaceDatetime as InterfaceDatetime
from static_frame.core.node_fill_value import InterfaceFillValue as InterfaceFillValue
from static_frame.core.node_hashlib import InterfaceHashlib as InterfaceHashlib
from static_frame.core.node_iter import IterNodeApplyType as IterNodeApplyType
from static_frame.core.node_iter import IterNodeAxis as IterNodeAxis
from static_frame.core.node_iter import IterNodeDelegate as IterNodeDelegate
from static_frame.core.node_iter import IterNodeDelegateMapable as IterNodeDelegateMapable
from static_frame.core.node_iter import (
    IterNodeDelegateReducible as IterNodeDelegateReducible,
)
from static_frame.core.node_iter import IterNodeDepthLevel as IterNodeDepthLevel
from static_frame.core.node_iter import IterNodeDepthLevelAxis as IterNodeDepthLevelAxis
from static_frame.core.node_iter import IterNodeGroup as IterNodeGroup
from static_frame.core.node_iter import IterNodeGroupAxis as IterNodeGroupAxis
from static_frame.core.node_iter import IterNodeNoArgMapable as IterNodeNoArgMapable
from static_frame.core.node_iter import IterNodeWindow as IterNodeWindow
from static_frame.core.node_re import InterfaceRe as InterfaceRe
from static_frame.core.node_selector import (
    InterfaceAssignQuartet as InterfaceAssignQuartet,
)
from static_frame.core.node_selector import InterfaceAssignTrio as InterfaceAssignTrio
from static_frame.core.node_selector import InterfaceBatchAsType as InterfaceBatchAsType
from static_frame.core.node_selector import InterfaceConsolidate as InterfaceConsolidate
from static_frame.core.node_selector import InterfaceFrameAsType as InterfaceFrameAsType
from static_frame.core.node_selector import (
    InterfaceIndexHierarchyAsType as InterfaceIndexHierarchyAsType,
)
from static_frame.core.node_selector import InterfacePersist as InterfacePersist
from static_frame.core.node_selector import InterfaceSelectDuo as InterfaceSelectDuo
from static_frame.core.node_selector import (
    InterfaceSelectQuartet as InterfaceSelectQuartet,
)
from static_frame.core.node_selector import InterfaceSelectTrio as InterfaceSelectTrio
from static_frame.core.node_selector import (
    InterGetItemLocReduces as InterGetItemLocReduces,
)
from static_frame.core.node_str import InterfaceBatchString as InterfaceBatchString
from static_frame.core.node_str import InterfaceString as InterfaceString
from static_frame.core.node_transpose import (
    InterfaceBatchTranspose as InterfaceBatchTranspose,
)
from static_frame.core.node_transpose import InterfaceTranspose as InterfaceTranspose
from static_frame.core.node_values import InterfaceBatchValues as InterfaceBatchValues
from static_frame.core.node_values import InterfaceValues as InterfaceValues
from static_frame.core.platform import Platform as Platform
from static_frame.core.quilt import Quilt as Quilt
from static_frame.core.reduce import InterfaceBatchReduceDispatch
from static_frame.core.reduce import ReduceDispatch as ReduceDispatch
from static_frame.core.reduce import ReduceDispatchAligned as ReduceDispatchAligned
from static_frame.core.reduce import ReduceDispatchUnaligned as ReduceDispatchUnaligned
from static_frame.core.series import Series as Series
from static_frame.core.series import SeriesAssign as SeriesAssign
from static_frame.core.series import SeriesHE as SeriesHE
from static_frame.core.series_mapping import SeriesMapping as SeriesMapping
from static_frame.core.store_config import StoreConfig as StoreConfig
from static_frame.core.store_config import StoreConfigMap as StoreConfigMap
from static_frame.core.store_filter import StoreFilter as StoreFilter
from static_frame.core.type_blocks import TypeBlocks as TypeBlocks
from static_frame.core.type_clinic import CallGuard as CallGuard
from static_frame.core.type_clinic import ClinicError as ClinicError
from static_frame.core.type_clinic import ClinicResult as ClinicResult
from static_frame.core.type_clinic import Require as Require
from static_frame.core.type_clinic import TypeClinic as TypeClinic
from static_frame.core.util import IterNodeType as IterNodeType
from static_frame.core.util import TCallableOrMapping as TCallableOrMapping
from static_frame.core.util import TDtypeSpecifier as TDtypeSpecifier
from static_frame.core.util import TFrameInitializer as TFrameInitializer
from static_frame.core.util import TIndexInitializer as TIndexInitializer
from static_frame.core.util import TIndexSpecifier as TIndexSpecifier
from static_frame.core.util import TKeyOrKeys as TKeyOrKeys
from static_frame.core.util import TLocSelector as TLocSelector
from static_frame.core.util import TLocSelectorCompound as TLocSelectorCompound
from static_frame.core.util import TPathSpecifierOrBinaryIO as TPathSpecifierOrBinaryIO
from static_frame.core.util import TPathSpecifierOrTextIO as TPathSpecifierOrTextIO
from static_frame.core.util import TSeriesInitializer as TSeriesInitializer
from static_frame.core.www import WWW as WWW
from static_frame.core.yarn import Yarn as Yarn
