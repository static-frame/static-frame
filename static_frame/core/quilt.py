from __future__ import annotations

from functools import partial
from itertools import chain, repeat, zip_longest

import numpy as np
import typing_extensions as tp

from static_frame.core.axis_map import bus_to_hierarchy, get_extractor
from static_frame.core.bus import Bus
from static_frame.core.container import ContainerBase
from static_frame.core.container_util import (
    axis_window_items,
    iter_component_signature_bytes,
)
from static_frame.core.display import Display, DisplayActive, DisplayHeader
from static_frame.core.doc_str import doc_inject, doc_update
from static_frame.core.exception import (
    AxisInvalid,
    ErrorInitIndexNonUnique,
    ErrorInitQuilt,
    NotImplementedAxis,
    immutable_type_error_factory,
)
from static_frame.core.frame import Frame
from static_frame.core.hloc import HLoc
from static_frame.core.index_auto import IndexAutoConstructorFactory as IACF
from static_frame.core.index_hierarchy import IndexHierarchy, TTreeNode
from static_frame.core.node_iter import (
    IterNodeApplyType,
    IterNodeAxis,
    IterNodeConstructorAxis,
    IterNodeWindow,
)
from static_frame.core.node_selector import (
    InterGetItemILocCompoundReduces,
    InterGetItemLocCompoundReduces,
)
from static_frame.core.series import Series
from static_frame.core.store_client_mixin import StoreClientMixin
from static_frame.core.store_sqlite import StoreSQLite
from static_frame.core.store_xlsx import StoreXLSX
from static_frame.core.store_zip import (
    StoreZipCSV,
    StoreZipNPY,
    StoreZipNPZ,
    StoreZipParquet,
    StoreZipPickle,
    StoreZipTSV,
)
from static_frame.core.util import (
    INT_TYPES,
    NULL_SLICE,
    IterNodeType,
    TCallableAny,
    TILocSelector,
    TILocSelectorCompound,
    TILocSelectorMany,
    TILocSelectorOne,
    TLabel,
    TLocSelector,
    TLocSelectorCompound,
    TLocSelectorMany,
    TName,
    TPathSpecifier,
    concat_resolved,
    get_tuple_constructor,
)
from static_frame.core.yarn import Yarn

if tp.TYPE_CHECKING:
    from static_frame.core.display_config import DisplayConfig  # pragma: no cover
    from static_frame.core.index import Index  # pragma: no cover
    from static_frame.core.index_base import IndexBase  # pragma: no cover
    from static_frame.core.store import Store  # pragma: no cover
    from static_frame.core.store_config import (
        StoreConfigMapInitializer,
    )  # pragma: no cover
    from static_frame.core.style_config import StyleConfig  # pragma: no cover

    TNDArrayAny = np.ndarray[tp.Any, tp.Any]  # pragma: no cover
    TDtypeAny = np.dtype[tp.Any]  # pragma: no cover

TSeriesAny = Series[tp.Any, tp.Any]
TFrameAny = Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]]
TBusAny = Bus[tp.Any]
TYarnAny = Yarn[tp.Any]


class Quilt(ContainerBase, StoreClientMixin):
    """
    A :obj:`Frame`-like view of the contents of a :obj:`Bus` or :obj:`Yarn`. With the Quilt, :obj:`Frame` contained in a :obj:`Bus` or :obj:`Yarn` can be conceived as stacking vertically (primary axis 0) or horizontally (primary axis 1). If the labels of the primary axis are unique across all contained :obj:`Frame`, ``retain_labels`` can be set to ``False`` and underlying labels are simply concatenated; otherwise, ``retain_labels`` must be set to ``True`` and an additional depth-level is added to the primary axis labels. A :obj:`Quilt` can only be created if labels of the opposite axis of all contained :obj:`Frame` are aligned.
    """

    __slots__ = (
        '_bus',
        '_axis',
        '_axis_hierarchy',
        '_retain_labels',
        '_axis_opposite',
        '_assign_axis',
        '_columns',
        '_index',
        '_deepcopy_from_bus',
    )

    _bus: tp.Union[TBusAny, TYarnAny]
    _axis: int
    _axis_hierarchy: tp.Optional[IndexHierarchy]
    _axis_opposite: tp.Optional[IndexBase]
    _columns: IndexBase
    _index: IndexBase
    _assign_axis: bool

    _NDIM: int = 2

    @classmethod
    def from_frame(
        cls,
        frame: TFrameAny,
        /,
        *,
        chunksize: int,
        retain_labels: bool,
        axis: int = 0,
        name: TName = None,
        label_extractor: tp.Optional[tp.Callable[[IndexBase], TLabel]] = None,
        config: StoreConfigMapInitializer = None,
        deepcopy_from_bus: bool = False,
    ) -> 'Quilt':
        """
        Given a :obj:`Frame`, create a :obj:`Quilt` by partitioning it along the specified ``axis`` in units of ``chunksize``, where ``axis`` 0 partitions vertically (retaining aligned columns) and 1 partions horizontally (retaining aligned index).

        Args:
            label_extractor: Function that, given the partitioned index component along the specified axis, returns a string label for that chunk.
        """
        if axis == 0 and frame._index._NDIM != 1:
            raise ValueError('Index must be 1D.')
        elif axis == 1 and frame._columns._NDIM != 1:
            raise ValueError('Columns must be 1D.')

        vector = frame._index if axis == 0 else frame._columns
        vector_len = len(vector)

        starts = range(0, vector_len, chunksize)
        if len(starts) == 1:
            ends: tp.Iterable[int] = (vector_len,)
        else:
            ends = range(starts[1], vector_len, chunksize)

        if label_extractor is None:
            label_extractor = lambda x: x.iloc[0]

        axis_map_components: TTreeNode = {}
        opposite = None

        def values() -> tp.Iterator[TFrameAny]:
            nonlocal opposite

            for start, end in zip_longest(starts, ends, fillvalue=vector_len):
                # NOTE: index / columns cannot be IndexHierarchy
                if axis == 0:  # along rows
                    f = frame.iloc[start:end]
                    label = label_extractor(f._index)
                    axis_map_components[label] = f._index  # type: ignore[assignment]
                    if opposite is None:
                        opposite = f._columns
                elif axis == 1:  # along columns
                    f = frame.iloc[NULL_SLICE, start:end]
                    label = label_extractor(f._columns)
                    axis_map_components[label] = f._columns  # type: ignore[assignment]
                    if opposite is None:
                        opposite = f.index
                else:
                    raise AxisInvalid(f'invalid axis {axis}')
                yield f.rename(label)

        name = name if name else frame.name
        bus = Bus.from_frames(values(), config=config, name=name)

        axis_hierarchy = IndexHierarchy.from_tree(
            axis_map_components, index_constructors=IACF
        )

        return cls(
            bus,
            axis=axis,
            axis_hierarchy=axis_hierarchy,
            axis_opposite=opposite,
            retain_labels=retain_labels,
            deepcopy_from_bus=deepcopy_from_bus,
        )

    # ---------------------------------------------------------------------------
    # constructors by data format

    @classmethod
    def _from_store(
        cls,
        store: Store,
        *,
        config: StoreConfigMapInitializer = None,
        axis: int = 0,
        retain_labels: bool,
        deepcopy_from_bus: bool = False,
        max_persist: tp.Optional[int] = None,
    ) -> 'Quilt':
        bus = Bus._from_store(
            store=store,
            config=config,
            max_persist=max_persist,  # None is default
        )
        return cls(
            bus,
            axis=axis,
            retain_labels=retain_labels,
            deepcopy_from_bus=deepcopy_from_bus,
        )

    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_zip_tsv(
        cls,
        fp: TPathSpecifier,
        /,
        *,
        config: StoreConfigMapInitializer = None,
        axis: int = 0,
        retain_labels: bool,
        deepcopy_from_bus: bool = False,
        max_persist: tp.Optional[int] = None,
    ) -> 'Quilt':
        """
        Given a file path to zipped TSV :obj:`Quilt` store, return a :obj:`Quilt` instance.

        {args}
        """
        store = StoreZipTSV(fp)
        return cls._from_store(
            store,
            config=config,
            axis=axis,
            retain_labels=retain_labels,
            deepcopy_from_bus=deepcopy_from_bus,
            max_persist=max_persist,
        )

    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_zip_csv(
        cls,
        fp: TPathSpecifier,
        /,
        *,
        config: StoreConfigMapInitializer = None,
        axis: int = 0,
        retain_labels: bool,
        deepcopy_from_bus: bool = False,
        max_persist: tp.Optional[int] = None,
    ) -> 'Quilt':
        """
        Given a file path to zipped CSV :obj:`Quilt` store, return a :obj:`Quilt` instance.

        {args}
        """
        store = StoreZipCSV(fp)
        return cls._from_store(
            store,
            config=config,
            axis=axis,
            retain_labels=retain_labels,
            deepcopy_from_bus=deepcopy_from_bus,
            max_persist=max_persist,
        )

    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_zip_pickle(
        cls,
        fp: TPathSpecifier,
        /,
        *,
        config: StoreConfigMapInitializer = None,
        axis: int = 0,
        retain_labels: bool,
        deepcopy_from_bus: bool = False,
        max_persist: tp.Optional[int] = None,
    ) -> 'Quilt':
        """
        Given a file path to zipped pickle :obj:`Quilt` store, return a :obj:`Quilt` instance.

        {args}
        """
        store = StoreZipPickle(fp)
        return cls._from_store(
            store,
            config=config,
            axis=axis,
            retain_labels=retain_labels,
            deepcopy_from_bus=deepcopy_from_bus,
            max_persist=max_persist,
        )

    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_zip_npz(
        cls,
        fp: TPathSpecifier,
        /,
        *,
        config: StoreConfigMapInitializer = None,
        axis: int = 0,
        retain_labels: bool,
        deepcopy_from_bus: bool = False,
        max_persist: tp.Optional[int] = None,
    ) -> 'Quilt':
        """
        Given a file path to zipped NPZ :obj:`Quilt` store, return a :obj:`Quilt` instance.

        {args}
        """
        store = StoreZipNPZ(fp)
        return cls._from_store(
            store,
            config=config,
            axis=axis,
            retain_labels=retain_labels,
            deepcopy_from_bus=deepcopy_from_bus,
            max_persist=max_persist,
        )

    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_zip_npy(
        cls,
        fp: TPathSpecifier,
        /,
        *,
        config: StoreConfigMapInitializer = None,
        axis: int = 0,
        retain_labels: bool,
        deepcopy_from_bus: bool = False,
        max_persist: tp.Optional[int] = None,
    ) -> 'Quilt':
        """
        Given a file path to zipped NPY :obj:`Quilt` store, return a :obj:`Quilt` instance.

        {args}
        """
        store = StoreZipNPY(fp)
        return cls._from_store(
            store,
            config=config,
            axis=axis,
            retain_labels=retain_labels,
            deepcopy_from_bus=deepcopy_from_bus,
            max_persist=max_persist,
        )

    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_zip_parquet(
        cls,
        fp: TPathSpecifier,
        /,
        *,
        config: StoreConfigMapInitializer = None,
        axis: int = 0,
        retain_labels: bool,
        deepcopy_from_bus: bool = False,
        max_persist: tp.Optional[int] = None,
    ) -> 'Quilt':
        """
        Given a file path to zipped parquet :obj:`Quilt` store, return a :obj:`Quilt` instance.

        {args}
        """
        store = StoreZipParquet(fp)
        return cls._from_store(
            store,
            config=config,
            axis=axis,
            retain_labels=retain_labels,
            deepcopy_from_bus=deepcopy_from_bus,
            max_persist=max_persist,
        )

    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_xlsx(
        cls,
        fp: TPathSpecifier,
        /,
        *,
        config: StoreConfigMapInitializer = None,
        axis: int = 0,
        retain_labels: bool,
        deepcopy_from_bus: bool = False,
        max_persist: tp.Optional[int] = None,
    ) -> 'Quilt':
        """
        Given a file path to an XLSX :obj:`Quilt` store, return a :obj:`Quilt` instance.

        {args}
        """
        # how to pass configuration for multiple sheets?
        store = StoreXLSX(fp)
        return cls._from_store(
            store,
            config=config,
            axis=axis,
            retain_labels=retain_labels,
            deepcopy_from_bus=deepcopy_from_bus,
            max_persist=max_persist,
        )

    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_sqlite(
        cls,
        fp: TPathSpecifier,
        /,
        *,
        config: StoreConfigMapInitializer = None,
        axis: int = 0,
        retain_labels: bool,
        deepcopy_from_bus: bool = False,
        max_persist: tp.Optional[int] = None,
    ) -> 'Quilt':
        """
        Given a file path to an SQLite :obj:`Quilt` store, return a :obj:`Quilt` instance.

        {args}
        """
        store = StoreSQLite(fp)
        return cls._from_store(
            store,
            config=config,
            axis=axis,
            retain_labels=retain_labels,
            deepcopy_from_bus=deepcopy_from_bus,
            max_persist=max_persist,
        )

    # ---------------------------------------------------------------------------

    @classmethod
    def from_items(
        cls,
        items: tp.Iterable[tp.Tuple[TLabel, TFrameAny]],
        /,
        *,
        axis: int = 0,
        name: TName = None,
        retain_labels: bool,
        deepcopy_from_bus: bool = False,
    ) -> 'Quilt':
        """
        Given an iterable of pairs of label, :obj:`Frame`, create a :obj:`Quilt`.
        """
        bus = Bus.from_items(items, name=name)
        return cls(
            bus,
            axis=axis,
            retain_labels=retain_labels,
            deepcopy_from_bus=deepcopy_from_bus,
        )

    @classmethod
    def from_frames(
        cls,
        frames: tp.Iterable[TFrameAny],
        /,
        *,
        axis: int = 0,
        name: TName = None,
        retain_labels: bool,
        deepcopy_from_bus: bool = False,
    ) -> 'Quilt':
        """Return a :obj:`Quilt` from an iterable of :obj:`Frame`; labels will be drawn from :obj:`Frame.name`."""
        bus = Bus.from_frames(frames, name=name)
        return cls(
            bus,
            axis=axis,
            retain_labels=retain_labels,
            deepcopy_from_bus=deepcopy_from_bus,
        )

    # ---------------------------------------------------------------------------
    def __init__(
        self,
        bus: tp.Union[TBusAny, TYarnAny],
        /,
        *,
        axis: int = 0,
        retain_labels: bool,
        axis_hierarchy: tp.Optional[IndexHierarchy] = None,
        axis_opposite: tp.Optional[IndexBase] = None,
        deepcopy_from_bus: bool = False,
    ) -> None:
        """
        {args}
        """
        self._bus = bus
        self._axis = axis
        self._retain_labels = retain_labels
        self._deepcopy_from_bus = deepcopy_from_bus

        if (axis_hierarchy is None) ^ (axis_opposite is None):
            raise ErrorInitQuilt('if supplying axis_hierarchy, supply axis_opposite')

        # can creation until needed
        self._axis_hierarchy = axis_hierarchy
        self._axis_opposite = axis_opposite
        self._assign_axis = True  # Boolean to control deferred axis index creation

    # ---------------------------------------------------------------------------
    # deferred loading of axis info
    @staticmethod
    def _error_update_axis_labels(axis: int) -> ErrorInitQuilt:
        axis_label = 'index' if axis == 0 else 'column'
        axis_labels = 'indices' if axis == 0 else 'columns'
        err_msg = f'Duplicate {axis_label} labels across frames. Either ensure all {axis_labels} are unique for all frames, or set retain_labels=True to obtain an IndexHierarchy'
        return ErrorInitQuilt(err_msg)

    def _update_axis_labels(self) -> None:
        if self._axis_hierarchy is None or self._axis_opposite is None:
            self._axis_hierarchy, self._axis_opposite = bus_to_hierarchy(
                self._bus,
                axis=self._axis,
                deepcopy_from_bus=self._deepcopy_from_bus,
                init_exception_cls=ErrorInitQuilt,
            )
        if self._axis == 0:
            if not self._retain_labels:
                try:
                    self._index = self._axis_hierarchy.level_drop(1)
                except ErrorInitIndexNonUnique:
                    raise self._error_update_axis_labels(self._axis) from None
            else:  # get hierarchical
                self._index = self._axis_hierarchy
            self._columns = self._axis_opposite  # type: ignore
        else:
            if not self._retain_labels:
                try:
                    self._columns = self._axis_hierarchy.level_drop(1)
                except ErrorInitIndexNonUnique:
                    raise self._error_update_axis_labels(self._axis) from None
            else:
                self._columns = self._axis_hierarchy
            self._index = self._axis_opposite  # type: ignore
        self._assign_axis = False

    def unpersist(self) -> None:
        """For the :obj:`Bus` or :obj:`Yarn` contained in this object, replace all loaded :obj:`Frame` with :obj:`FrameDeferred`."""
        self._bus.unpersist()

    # ---------------------------------------------------------------------------
    # name interface

    @property
    @doc_inject()
    def name(self) -> TName:
        """{}"""
        return self._bus.name

    def rename(
        self,
        name: TName,
        /,
    ) -> 'Quilt':
        """
        Return a new :obj:`Quilt` with an updated name attribute.

        Args:
            name
        """
        return self.__class__(
            self._bus.rename(name),
            axis=self._axis,
            retain_labels=self._retain_labels,
            deepcopy_from_bus=self._deepcopy_from_bus,
            axis_hierarchy=self._axis_hierarchy,
            axis_opposite=self._axis_opposite,
        )

    # ---------------------------------------------------------------------------

    @doc_inject()
    def display(
        self,
        config: tp.Optional[DisplayConfig] = None,
        /,
        *,
        style_config: tp.Optional[StyleConfig] = None,
    ) -> Display:
        """{doc}

        Args:
            {config}
        """
        if self._assign_axis:
            self._update_axis_labels()

        config = config or DisplayActive.get()

        drop_column_dtype = False
        if self._axis == 0:
            if not self._retain_labels:
                index = self.index.rename('Concatenated')
            else:
                index = self._bus.index.rename('Frames')
            columns = self.columns.rename('Aligned')
        else:
            index = self.index.rename('Aligned')
            if not self._retain_labels:
                columns = self.columns.rename('Concatenated')
            else:
                columns = self._bus.index.rename('Frames')
                drop_column_dtype = True

        def placeholder_gen() -> tp.Iterator[tp.Iterable[tp.Any]]:
            assert config is not None
            yield from repeat(
                tuple(repeat(config.cell_placeholder, times=len(index))),
                times=len(columns),
            )

        d = Display.from_params(
            index=index,
            columns=columns,
            header=DisplayHeader(self.__class__, self.name),
            column_forward_iter=placeholder_gen,
            column_reverse_iter=placeholder_gen,
            column_default_iter=placeholder_gen,
            config=config,
            style_config=style_config,
        )

        # Strip out the dtype information!
        if config.type_show:
            if drop_column_dtype:
                # First Column Row -> last element is the dtype of the column
                # Guaranteed to not be index hierarchy as buses cannot have index hierarchies
                d._rows[1].pop()

            # Since placeholder_gen is not a ndarray, there is no dtype to append in the final row
            # However, in the case of a center ellipsis being added, an ellipsis will be
            # awkwardly placed direclty adjacent to the index dtype information.
            if d._rows[-1][-1] == Display.CELL_ELLIPSIS:
                d._rows[-1].pop()
        return d

    # ---------------------------------------------------------------------------
    # accessors

    @property
    @doc_inject(selector='values_2d', class_name='Quilt')
    def values(self) -> TNDArrayAny:
        """
        {}
        """
        if self._assign_axis:
            self._update_axis_labels()
        return self.to_frame().values

    @property
    def index(self) -> IndexBase:
        """The ``IndexBase`` instance assigned for row labels."""
        if self._assign_axis:
            self._update_axis_labels()
        return self._index

    @property
    def columns(self) -> IndexBase:
        """The ``IndexBase`` instance assigned for column labels."""
        if self._assign_axis:
            self._update_axis_labels()
        return self._columns

    @property
    def bus(self) -> tp.Union[TBusAny, TYarnAny]:
        """The ``Bus`` instance assigned to this ``Quilt``."""
        return self._bus

    # ---------------------------------------------------------------------------

    @property
    def shape(self) -> tp.Tuple[int, int]:
        """
        Return a tuple describing the shape of the underlying NumPy array.

        Returns:
            :obj:`tp.Tuple[int]`
        """
        if self._assign_axis:
            self._update_axis_labels()
        return len(self._index), len(self._columns)

    @property
    def ndim(self) -> int:
        """
        Return the number of dimensions, which for a `Frame` is always 2.

        Returns:
            :obj:`int`
        """
        return self._NDIM

    @property
    def size(self) -> int:
        """
        Return the size of the underlying NumPy array.

        Returns:
            :obj:`int`
        """
        if self._assign_axis:
            self._update_axis_labels()
        return len(self._index) * len(self._columns)

    @property
    def nbytes(self) -> int:
        """
        Return the total bytes of the underlying NumPy arrays.

        Returns:
            :obj:`int`
        """
        # return self._blocks.nbytes
        if self._assign_axis:
            self._update_axis_labels()
        return sum(f.nbytes for _, f in self._bus.items())

    @property
    def status(self) -> TFrameAny:
        """
        Return a :obj:`Frame` indicating loaded status, size, bytes, and shape of all loaded :obj:`Frame` in the contained :obj:`Quilt`.
        """
        return self._bus.status

    @property
    def inventory(self) -> TFrameAny:
        """
        Return a :obj:`Frame` indicating file_path, last-modified time, and size of underlying disk-based data stores if used for this :obj:`Quilt`.
        """
        return self._bus.inventory

    # ---------------------------------------------------------------------------
    # dictionary-like interface

    def keys(self) -> tp.Iterable[TLabel]:
        """Iterator of column labels."""
        if self._assign_axis:
            self._update_axis_labels()
        return self._columns

    def __iter__(self) -> tp.Iterable[TLabel]:
        """
        Iterator of column labels, same as :py:meth:`Frame.keys`.
        """
        if self._assign_axis:
            self._update_axis_labels()
        return self._columns.__iter__()

    def __contains__(
        self,
        value: TLabel,
        /,
    ) -> bool:
        """
        Inclusion of value in column labels.
        """
        if self._assign_axis:
            self._update_axis_labels()
        return self._columns.__contains__(value)

    def items(self) -> tp.Iterator[tp.Tuple[TLabel, TSeriesAny]]:
        """Iterator of pairs of column label and corresponding column :obj:`Series`."""
        if self._assign_axis:
            self._update_axis_labels()
        yield from self._axis_series_items(axis=0)  # iterate columns

    def get(
        self,
        key: TLabel,
        default: tp.Optional[TSeriesAny] = None,
    ) -> tp.Optional[TSeriesAny]:
        """
        Return the value found at the columns key, else the default if the key is not found. This method is implemented to complete the dictionary-like interface.
        """
        if self._assign_axis:
            self._update_axis_labels()
        if key not in self._columns:
            return default
        return self.__getitem__(key)

    # ---------------------------------------------------------------------------
    # compatibility with StoreClientMixin

    def _items_store(self) -> tp.Iterator[tp.Tuple[TLabel, TFrameAny]]:
        """Iterator of pairs of :obj:`Quilt` label and contained :obj:`Frame`."""
        yield from self._bus.items()

    # ---------------------------------------------------------------------------
    # axis iterators

    def _axis_array(self, axis: int) -> tp.Iterator[TNDArrayAny]:
        """Generator of arrays across an axis

        Args:
            axis: 0 iterates over columns, 1 iterates over rows
        """
        extractor = get_extractor(
            self._deepcopy_from_bus,
            is_array=True,
            memo_active=False,
        )

        if axis == 1:  # iterate over rows
            if self._axis == 0:  # bus components aligned vertically
                for _, component in self._bus.items():
                    for array in component._blocks.axis_values(axis):
                        yield extractor(array)
            else:  # bus components aligned horizontally
                raise NotImplementedAxis()
        elif axis == 0:  # iterate over columns
            if self._axis == 1:  # bus components aligned horizontally
                for _, component in self._bus.items():
                    for array in component._blocks.axis_values(axis):
                        yield extractor(array)
            else:  # bus components aligned horizontally
                raise NotImplementedAxis()
        else:
            raise AxisInvalid(f'no support for axis {axis}')

    def _axis_array_items(self, axis: int) -> tp.Iterator[tp.Tuple[TLabel, TNDArrayAny]]:
        keys = self._index if axis == 1 else self._columns
        yield from zip(keys, self._axis_array(axis))

    def _axis_tuple(
        self,
        *,
        axis: int,
        constructor: tp.Optional[tp.Type[tp.NamedTuple]] = None,
    ) -> tp.Iterator[tp.NamedTuple]:
        """Generator of named tuples across an axis.

        Args:
            axis: 0 iterates over columns (index axis), 1 iterates over rows (column axis)
        """
        if constructor is None:
            if axis == 1:
                labels = self._columns.values
            elif axis == 0:
                labels = self._index.values
            else:
                raise AxisInvalid(f'no support for axis {axis}')
            # uses _make method to call with iterable
            constructor = get_tuple_constructor(labels)  # type: ignore
        elif (
            isinstance(constructor, type)
            and issubclass(constructor, tuple)
            and hasattr(constructor, '_make')
        ):
            constructor = constructor._make  # type: ignore

        assert constructor is not None

        for axis_values in self._axis_array(axis):
            yield constructor(axis_values)  # type: ignore

    def _axis_tuple_items(
        self,
        *,
        axis: int,
        constructor: tp.Optional[tp.Type[tp.NamedTuple]] = None,
    ) -> tp.Iterator[tp.Tuple[TLabel, tp.NamedTuple]]:
        keys = self._index if axis == 1 else self._columns
        yield from zip(keys, self._axis_tuple(axis=axis, constructor=constructor))

    def _axis_series(self, axis: int) -> tp.Iterator[TSeriesAny]:
        """Generator of Series across an axis"""
        index = self._index if axis == 0 else self._columns
        for label, axis_values in self._axis_array_items(axis):
            yield Series(axis_values, index=index, name=label, own_index=True)

    def _axis_series_items(self, axis: int) -> tp.Iterator[tp.Tuple[TLabel, TSeriesAny]]:
        keys = self._index if axis == 1 else self._columns
        yield from zip(keys, self._axis_series(axis=axis))

    # ---------------------------------------------------------------------------
    def _axis_window_items(
        self,
        *,
        size: int,
        axis: int = 0,
        step: int = 1,
        window_sized: bool = True,
        window_func: tp.Optional[TCallableAny] = None,
        window_valid: tp.Optional[TCallableAny] = None,
        label_shift: int = 0,
        label_missing_skips: bool = True,
        label_missing_raises: bool = False,
        start_shift: int = 0,
        size_increment: int = 0,
        as_array: bool = False,
    ) -> tp.Iterator[tp.Tuple[TLabel, tp.Any]]:
        """Generator of index, processed-window pairs."""
        # NOTE: this will use _extract, _extract_array to get results, thus we do not need an extractor
        yield from axis_window_items(
            source=self,
            size=size,
            axis=axis,
            step=step,
            window_sized=window_sized,
            window_func=window_func,
            window_valid=window_valid,
            label_shift=label_shift,
            label_missing_skips=label_missing_skips,
            label_missing_raises=label_missing_raises,
            start_shift=start_shift,
            size_increment=size_increment,
            derive_label=True,
            as_array=as_array,
        )

    def _axis_window(
        self,
        *,
        size: int,
        axis: int = 0,
        step: int = 1,
        window_sized: bool = True,
        window_func: tp.Optional[TCallableAny] = None,
        window_valid: tp.Optional[TCallableAny] = None,
        label_shift: int = 0,
        label_missing_skips: bool = True,
        label_missing_raises: bool = False,
        start_shift: int = 0,
        size_increment: int = 0,
        as_array: bool = False,
    ) -> tp.Iterator[TFrameAny]:
        yield from (
            x
            for _, x in axis_window_items(
                source=self,
                size=size,
                axis=axis,
                step=step,
                window_sized=window_sized,
                window_func=window_func,
                window_valid=window_valid,
                label_shift=label_shift,
                label_missing_skips=label_missing_skips,
                label_missing_raises=label_missing_raises,
                start_shift=start_shift,
                size_increment=size_increment,
                derive_label=False,
                as_array=as_array,
            )
        )

    # ---------------------------------------------------------------------------
    def _extract_array(
        self,
        row_key: TILocSelector = None,
        column_key: TILocSelector = None,
    ) -> TNDArrayAny:
        """
        Extract a consolidated array based on iloc selection.
        """
        assert self._axis_hierarchy is not None  # mypy

        extractor: tp.Callable[..., TNDArrayAny] = get_extractor(
            self._deepcopy_from_bus,
            is_array=True,
            memo_active=False,
        )

        row_key = NULL_SLICE if row_key is None else row_key
        column_key = NULL_SLICE if column_key is None else column_key

        if row_key == NULL_SLICE and column_key == NULL_SLICE:
            if len(self._bus) == 1:
                return extractor(self._bus.iloc[0].values)  # type: ignore

            # NOTE: do not need to call extractor when concatenate is called, as a new array is always allocated.
            arrays = [f.values for _, f in self._bus.items()]
            return concat_resolved(
                arrays,
                axis=self._axis,
            )

        parts: tp.List[TNDArrayAny] = []
        bus_keys: tp.Iterable[TLabel]

        if self._axis == 0:
            sel_key = row_key
            opposite_key = column_key
        else:
            sel_key = column_key
            opposite_key = row_key

        sel_reduces = isinstance(sel_key, INT_TYPES)
        opposite_reduces = isinstance(opposite_key, INT_TYPES)

        sel = np.full(len(self._axis_hierarchy), False)
        sel[sel_key] = True

        # get ordered unique Bus labels
        axis_map_sub = self._axis_hierarchy.iloc[sel_key]
        if isinstance(axis_map_sub, tuple):
            bus_keys = (axis_map_sub[0],)
        else:
            bus_keys = axis_map_sub.unique(0, order_by_occurrence=True)

        for key in bus_keys:
            sel_component = sel[self._axis_hierarchy._loc_to_iloc(HLoc[key])]

            if self._axis == 0:
                component = self._bus.loc[key]._extract_array(sel_component, opposite_key)  # type: ignore
                if sel_reduces:
                    component = component[0]
            else:
                component = self._bus.loc[key]._extract_array(opposite_key, sel_component)  # type: ignore
                if sel_reduces:
                    if component.ndim == 1:
                        component = component[0]
                    elif component.ndim == 2:
                        component = component[NULL_SLICE, 0]

            parts.append(component)

        if sel_reduces and opposite_reduces:  # we have an element
            return parts.pop()

        # we call extractor() when we might be referencing data to control if we give a slice or a deepcopy
        if len(parts) == 1:
            return extractor(parts.pop())

        # NOTE: concatenate always allocates a new array, thus no need for extractor above
        if sel_reduces or opposite_reduces:
            # NOTE: not sure if concat_resolved is needed here
            return concat_resolved(parts)
        return concat_resolved(parts, axis=self._axis)

    @staticmethod
    def _relabel(
        labels_is_ih: bool,
        label: TLabel,
        component: Frame | Series,
        axis: int,
    ) -> Frame | Series:
        """Relabel a component given a label that might be a tuple from an IH. If a tuple from an IH, produce a new IndexHierarchy."""
        if labels_is_ih:  # label is a tuple
            idx = component.index if axis == 0 else component.columns  # type: ignore [union-attr]
            size = component.shape[axis]
            values = [np.full(size, v) for v in label]  # type: ignore [union-attr]
            values.extend(idx.values_at_depth(x) for x in range(idx.depth))
            ih = IndexHierarchy.from_values_per_depth(values, index_constructors=IACF)
            if axis == 0:
                return component.relabel(index=ih)
            return component.relabel(columns=ih)  # type: ignore [call-arg]
        if axis == 0:
            return component.relabel_level_add(label, index_constructor=IACF)  # type: ignore
        return component.relabel_level_add(
            columns=label,  # pyright: ignore
            columns_constructor=IACF,  # type: ignore
        )

    @tp.overload
    def _extract(self, row_key: TILocSelectorOne) -> TSeriesAny: ...

    @tp.overload
    def _extract(self, row_key: TILocSelectorMany) -> TFrameAny: ...

    @tp.overload
    def _extract(self, row_key: None, column_key: TILocSelectorOne) -> TSeriesAny: ...

    @tp.overload
    def _extract(self, row_key: None, column_key: TILocSelectorMany) -> TFrameAny: ...

    @tp.overload
    def _extract(
        self, row_key: TILocSelectorMany, column_key: TILocSelectorOne
    ) -> TSeriesAny: ...

    @tp.overload
    def _extract(
        self, row_key: TILocSelectorOne, column_key: TILocSelectorMany
    ) -> TSeriesAny: ...

    @tp.overload
    def _extract(
        self, row_key: TILocSelectorMany, column_key: TILocSelectorMany
    ) -> TFrameAny: ...

    @tp.overload
    def _extract(
        self, row_key: TILocSelectorOne, column_key: TILocSelectorOne
    ) -> tp.Any: ...

    @tp.overload
    def _extract(self, row_key: TILocSelector) -> tp.Any: ...

    def _extract(
        self,
        row_key: TILocSelector = None,
        column_key: TILocSelector = None,
    ) -> tp.Any:
        """
        Extract Container based on iloc selection.
        """
        assert self._axis_hierarchy is not None  # mypy

        extractor = get_extractor(
            self._deepcopy_from_bus,
            is_array=False,
            memo_active=False,
        )

        row_key = NULL_SLICE if row_key is None else row_key
        row_key_is_array = isinstance(row_key, np.ndarray)
        column_key = NULL_SLICE if column_key is None else column_key
        column_key_is_array = isinstance(column_key, np.ndarray)

        labels_is_ih = self._bus._index._NDIM == 2

        # if doing a full extraction: both row and columns are NULL_SLICE
        if (
            not row_key_is_array
            and row_key == NULL_SLICE
            and not column_key_is_array
            and column_key == NULL_SLICE
        ):
            if self._retain_labels:
                frames = (
                    extractor(self._relabel(labels_is_ih, k, f, self._axis))
                    for k, f in self._bus.items()
                )
            else:
                frames = (extractor(f) for _, f in self._bus.items())
            return Frame.from_concat(
                frames,
                axis=self._axis,
            )

        parts: tp.List[tp.Any] = []
        frame_labels: tp.Iterable[TLabel]
        opposite_key: TILocSelector
        sel_key: TILocSelector

        if self._axis == 0:
            sel_key = row_key
            opposite_key = column_key
        else:
            sel_key = column_key
            opposite_key = row_key

        sel_reduces = isinstance(sel_key, INT_TYPES)
        sel = np.full(len(self._axis_hierarchy), False)
        sel[sel_key] = True

        # get ordered unique Bus labels
        axis_map_sub = self._axis_hierarchy.iloc[sel_key]
        if isinstance(axis_map_sub, tuple):
            frame_labels = (axis_map_sub[0],)
        else:
            # get the outer level, or just the unique frame labels needed
            frame_labels = axis_map_sub.unique(0, order_by_occurrence=True)

        component: tp.Any
        for key_count, key in enumerate(frame_labels):
            # get Boolean segment for this Frame
            sel_component = sel[self._axis_hierarchy._loc_to_iloc(HLoc[key, NULL_SLICE])]

            if self._axis == 0:
                component = self._bus.loc[key].iloc[sel_component, opposite_key]  # pyright: ignore
                if key_count == 0:
                    component_is_series = isinstance(component, Series)
                if self._retain_labels:
                    component = self._relabel(labels_is_ih, key, component, 0)
                if sel_reduces:  # make Frame into a Series, Series into an element
                    component = component.iloc[0]
            else:
                component = self._bus.loc[key].iloc[opposite_key, sel_component]  # pyright: ignore
                if key_count == 0:
                    component_is_series = isinstance(component, Series)
                if self._retain_labels:
                    component = self._relabel(
                        labels_is_ih,
                        key,
                        component,
                        0 if component_is_series else 1,
                    )
                if sel_reduces:  # make Frame into a Series, Series into an element
                    if component_is_series:
                        component = component.iloc[0]
                    else:
                        component = component.iloc[NULL_SLICE, 0]

            parts.append(extractor(component))

        if len(parts) == 1:
            return parts.pop()

        # NOTE: Series/Frame from_concat will attempt to re-use ndarrays, and thus using extractor above is appropriate
        if component_is_series:
            return Series.from_concat(parts)
        return Frame.from_concat(parts, axis=self._axis)

    # ---------------------------------------------------------------------------
    @doc_inject(selector='sample')
    def sample(
        self,
        index: tp.Optional[int] = None,
        columns: tp.Optional[int] = None,
        *,
        seed: tp.Optional[int] = None,
    ) -> TFrameAny:
        """
        {doc}

        Args:
            {index}
            {columns}
            {seed}
        """
        if self._assign_axis:
            self._update_axis_labels()

        if index is not None:
            _, index_key = self._index._sample_and_key(count=index, seed=seed)
        else:
            index_key = None

        if columns is not None:
            _, columns_key = self._columns._sample_and_key(count=columns, seed=seed)
        else:
            columns_key = None

        return self._extract(row_key=index_key, column_key=columns_key)

    # ---------------------------------------------------------------------------

    def _extract_iloc(self, key: TILocSelectorCompound) -> tp.Any:
        """
        Give a compound key, return a new Frame. This method simply handles the variabiliyt of single or compound selectors.
        """
        if self._assign_axis:
            self._update_axis_labels()
        if isinstance(key, tuple):
            r, c = key
            return self._extract(r, c)
        return self._extract(key)

    def _compound_loc_to_iloc(
        self, key: TLocSelectorCompound
    ) -> tp.Tuple[TILocSelector, TILocSelector]:
        """
        Given a compound iloc key, return a tuple of row, column keys. Assumes the first argument is always a row extractor.
        """
        if isinstance(key, tuple):
            loc_row_key, loc_column_key = key  # pyright: ignore
            iloc_column_key = self._columns._loc_to_iloc(loc_column_key)
        else:
            loc_row_key = key
            iloc_column_key = None

        iloc_row_key = self._index._loc_to_iloc(loc_row_key)
        return iloc_row_key, iloc_column_key

    def _extract_loc(self, key: TLocSelectorCompound) -> tp.Any:
        if self._assign_axis:
            self._update_axis_labels()
        r, c = self._compound_loc_to_iloc(key)
        return self._extract(r, c)

    def _compound_loc_to_getitem_iloc(
        self, key: TLocSelectorCompound
    ) -> tp.Tuple[TILocSelector, TILocSelector]:
        """Handle a potentially compound key in the style of __getitem__. This will raise an appropriate exception if a two argument loc-style call is attempted."""
        iloc_column_key = self._columns._loc_to_iloc(key)
        return None, iloc_column_key

    @tp.overload
    def __getitem__(self, key: TLabel) -> TSeriesAny: ...

    @tp.overload
    def __getitem__(self, key: TLocSelectorMany) -> TFrameAny: ...

    @doc_inject(selector='selector')
    def __getitem__(self, key: TLocSelector) -> tp.Union[TFrameAny, TSeriesAny]:
        """Selector of columns by label.

        Args:
            key: {key_loc}
        """
        if self._assign_axis:
            self._update_axis_labels()
        r, c = self._compound_loc_to_getitem_iloc(key)
        return self._extract(r, c)

    def __setitem__(self, key: TLabel, value: tp.Any) -> None:
        raise immutable_type_error_factory(self.__class__, '', key, value)

    # ---------------------------------------------------------------------------
    # interfaces

    @property
    def loc(self) -> InterGetItemLocCompoundReduces[TFrameAny]:
        return InterGetItemLocCompoundReduces(self._extract_loc)

    @property
    def iloc(self) -> InterGetItemILocCompoundReduces[TFrameAny]:
        return InterGetItemILocCompoundReduces(self._extract_iloc)

    # ---------------------------------------------------------------------------
    # iterators

    @property
    def iter_array(self) -> IterNodeAxis['Quilt']:
        """
        Iterator of :obj:`np.array`, where arrays are drawn from columns (axis=0) or rows (axis=1)
        """
        if self._assign_axis:
            self._update_axis_labels()
        return IterNodeAxis(
            container=self,
            function_values=self._axis_array,
            function_items=self._axis_array_items,
            yield_type=IterNodeType.VALUES,
            apply_type=IterNodeApplyType.SERIES_VALUES,
        )

    @property
    def iter_array_items(self) -> IterNodeAxis['Quilt']:
        """
        Iterator of pairs of label, :obj:`np.array`, where arrays are drawn from columns (axis=0) or rows (axis=1)
        """
        if self._assign_axis:
            self._update_axis_labels()
        return IterNodeAxis(
            container=self,
            function_values=self._axis_array,
            function_items=self._axis_array_items,
            yield_type=IterNodeType.ITEMS,
            apply_type=IterNodeApplyType.SERIES_VALUES,
        )

    @property
    def iter_tuple(self) -> IterNodeConstructorAxis['Quilt']:
        """
        Iterator of :obj:`NamedTuple`, where tuples are drawn from columns (axis=0) or rows (axis=1). An optional ``constructor`` callable can be used to provide a :obj:`NamedTuple` class (or any other constructor called with a single iterable) to be used to create each yielded axis value.
        """
        if self._assign_axis:
            self._update_axis_labels()
        return IterNodeConstructorAxis(
            container=self,
            function_values=self._axis_tuple,
            function_items=self._axis_tuple_items,
            yield_type=IterNodeType.VALUES,
            apply_type=IterNodeApplyType.SERIES_VALUES,
        )

    @property
    def iter_tuple_items(self) -> IterNodeConstructorAxis['Quilt']:
        """
        Iterator of pairs of label, :obj:`NamedTuple`, where tuples are drawn from columns (axis=0) or rows (axis=1)
        """
        if self._assign_axis:
            self._update_axis_labels()
        return IterNodeConstructorAxis(
            container=self,
            function_values=self._axis_tuple,
            function_items=self._axis_tuple_items,
            yield_type=IterNodeType.ITEMS,
            apply_type=IterNodeApplyType.SERIES_VALUES,
        )

    @property
    def iter_series(self) -> IterNodeAxis['Quilt']:
        """
        Iterator of :obj:`Series`, where :obj:`Series` are drawn from columns (axis=0) or rows (axis=1)
        """
        if self._assign_axis:
            self._update_axis_labels()
        return IterNodeAxis(
            container=self,
            function_values=self._axis_series,
            function_items=self._axis_series_items,
            yield_type=IterNodeType.VALUES,
            apply_type=IterNodeApplyType.SERIES_VALUES,
        )

    @property
    def iter_series_items(self) -> IterNodeAxis['Quilt']:
        """
        Iterator of pairs of label, :obj:`Series`, where :obj:`Series` are drawn from columns (axis=0) or rows (axis=1)
        """
        if self._assign_axis:
            self._update_axis_labels()
        return IterNodeAxis(
            container=self,
            function_values=self._axis_series,
            function_items=self._axis_series_items,
            yield_type=IterNodeType.ITEMS,
            apply_type=IterNodeApplyType.SERIES_VALUES,
        )

    # ---------------------------------------------------------------------------

    @property
    @doc_inject(selector='window')
    def iter_window(self) -> IterNodeWindow['Quilt']:
        """
        Iterator of windowed values, where values are given as a :obj:`Frame`.

        {args}
        """
        if self._assign_axis:
            self._update_axis_labels()
        function_values = partial(self._axis_window, as_array=False)
        function_items = partial(self._axis_window_items, as_array=False)
        return IterNodeWindow(
            container=self,
            function_values=function_values,
            function_items=function_items,
            yield_type=IterNodeType.VALUES,
            apply_type=IterNodeApplyType.SERIES_ITEMS,
        )

    @property
    @doc_inject(selector='window')
    def iter_window_items(self) -> IterNodeWindow['Quilt']:
        """
        Iterator of pairs of label, windowed values, where values are given as a :obj:`Frame`.

        {args}
        """
        if self._assign_axis:
            self._update_axis_labels()
        function_values = partial(self._axis_window, as_array=False)
        function_items = partial(self._axis_window_items, as_array=False)
        return IterNodeWindow(
            container=self,
            function_values=function_values,
            function_items=function_items,
            yield_type=IterNodeType.ITEMS,
            apply_type=IterNodeApplyType.SERIES_ITEMS,
        )

    @property
    @doc_inject(selector='window')
    def iter_window_array(self) -> IterNodeWindow['Quilt']:
        """
        Iterator of windowed values, where values are given as a :obj:`np.array`.

        {args}
        """
        if self._assign_axis:
            self._update_axis_labels()
        function_values = partial(self._axis_window, as_array=True)
        function_items = partial(self._axis_window_items, as_array=True)
        return IterNodeWindow(
            container=self,
            function_values=function_values,
            function_items=function_items,
            yield_type=IterNodeType.VALUES,
            apply_type=IterNodeApplyType.SERIES_ITEMS,
        )

    @property
    @doc_inject(selector='window')
    def iter_window_array_items(self) -> IterNodeWindow['Quilt']:
        """
        Iterator of pairs of label, windowed values, where values are given as a :obj:`np.array`.

        {args}
        """
        if self._assign_axis:
            self._update_axis_labels()
        function_values = partial(self._axis_window, as_array=True)
        function_items = partial(self._axis_window_items, as_array=True)
        return IterNodeWindow(
            container=self,
            function_values=function_values,
            function_items=function_items,
            yield_type=IterNodeType.ITEMS,
            apply_type=IterNodeApplyType.SERIES_ITEMS,
        )

    # ---------------------------------------------------------------------------
    # transformations resulting in changed dimensionality
    @doc_inject(selector='head', class_name='Quilt')
    def head(
        self,
        count: int = 5,
        /,
    ) -> TFrameAny:
        """{doc}

        Args:
            {count}
        """
        return self.iloc[:count]

    @doc_inject(selector='tail', class_name='Quilt')
    def tail(
        self,
        count: int = 5,
        /,
    ) -> TFrameAny:
        """{doc}

        Args:
            {count}
        """
        return self.iloc[-count:]

    # ---------------------------------------------------------------------------
    @doc_inject()
    def equals(
        self,
        other: tp.Any,
        /,
        *,
        compare_name: bool = False,
        compare_dtype: bool = False,
        compare_class: bool = False,
        skipna: bool = True,
    ) -> bool:
        """
        {doc}

        Note: this will attempt to load and compare all Frame managed by the Bus stored within this Quilt.

        Args:
            {compare_name}
            {compare_dtype}
            {compare_class}
            {skipna}
        """
        if id(other) == id(self):
            return True

        if compare_class and self.__class__ != other.__class__:
            return False
        elif not isinstance(other, Quilt):
            return False

        if self._axis != other._axis:
            return False

        if self._retain_labels != other._retain_labels:
            return False

        if compare_name and self.name != other.name:
            return False

        if self._assign_axis:
            self._update_axis_labels()
        if other._assign_axis:
            other._update_axis_labels()

        if not self._axis_hierarchy.equals(  # type: ignore
            other._axis_hierarchy,
            compare_name=compare_name,
            compare_dtype=compare_dtype,
            compare_class=compare_class,
            skipna=skipna,
        ):
            return False

        if not self._axis_opposite.equals(  # type: ignore
            other._axis_opposite,
            compare_name=compare_name,
            compare_dtype=compare_dtype,
            compare_class=compare_class,
            skipna=skipna,
        ):
            return False

        if not self._bus.equals(
            other._bus,
            compare_name=compare_name,
            compare_dtype=compare_dtype,
            compare_class=compare_class,
            skipna=skipna,
        ):
            return False

        return True

    # ---------------------------------------------------------------------------
    def to_frame(self) -> TFrameAny:
        """
        Return a consolidated :obj:`Frame`.
        """
        if self._assign_axis:
            self._update_axis_labels()
        return self._extract(NULL_SLICE, NULL_SLICE)

    def _to_signature_bytes(
        self,
        include_name: bool = True,
        include_class: bool = True,
        encoding: str = 'utf-8',
    ) -> bytes:
        if self._assign_axis:
            self._update_axis_labels()

        return b''.join(
            chain(
                iter_component_signature_bytes(
                    self,
                    include_name=include_name,
                    include_class=include_class,
                    encoding=encoding,
                ),
                (
                    self._axis_hierarchy._to_signature_bytes(  # type: ignore
                        include_name=include_name,
                        include_class=include_class,
                        encoding=encoding,
                    ),
                    self._axis_opposite._to_signature_bytes(  # type: ignore
                        include_name=include_name,
                        include_class=include_class,
                        encoding=encoding,
                    ),
                    self._bus._to_signature_bytes(
                        include_name=include_name,
                        include_class=include_class,
                        encoding=encoding,
                    ),
                ),
            )
        )


doc_update(Quilt.__init__, selector='quilt_init')
