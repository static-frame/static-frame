import typing as tp
from itertools import repeat
from itertools import zip_longest
from functools import partial

import numpy as np

from static_frame.core.bus import Bus
from static_frame.core.container import ContainerBase
from static_frame.core.container_util import axis_window_items
from static_frame.core.display import Display
from static_frame.core.display import DisplayHeader
from static_frame.core.display_config import DisplayConfig
from static_frame.core.doc_str import doc_inject
from static_frame.core.exception import AxisInvalid
from static_frame.core.exception import ErrorInitQuilt
from static_frame.core.exception import ErrorInitIndexNonUnique
from static_frame.core.exception import NotImplementedAxis
from static_frame.core.frame import Frame
from static_frame.core.hloc import HLoc
from static_frame.core.index_base import IndexBase
from static_frame.core.node_iter import IterNodeAxis
from static_frame.core.node_iter import IterNodeConstructorAxis
from static_frame.core.node_iter import IterNodeType
from static_frame.core.node_iter import IterNodeWindow
from static_frame.core.node_iter import IterNodeApplyType
from static_frame.core.node_selector import InterfaceGetItem
from static_frame.core.series import Series
from static_frame.core.store import Store
from static_frame.core.store import StoreConfigMapInitializer
from static_frame.core.store_client_mixin import StoreClientMixin
from static_frame.core.store_hdf5 import StoreHDF5
from static_frame.core.store_sqlite import StoreSQLite
from static_frame.core.store_xlsx import StoreXLSX
from static_frame.core.store_zip import StoreZipCSV
from static_frame.core.store_zip import StoreZipParquet
from static_frame.core.store_zip import StoreZipPickle
from static_frame.core.store_zip import StoreZipTSV
from static_frame.core.store_zip import StoreZipNPZ
from static_frame.core.util import AnyCallable
from static_frame.core.util import ShapeType
from static_frame.core.util import get_tuple_constructor
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import GetItemKeyTypeCompound
from static_frame.core.util import INT_TYPES
from static_frame.core.util import NameType
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import PathSpecifier
from static_frame.core.util import concat_resolved
from static_frame.core.style_config import StyleConfig
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.yarn import Yarn

from static_frame.core.axis_map import build_quilt_indices
from static_frame.core.axis_map import get_extractor


Q = tp.TypeVar('Q', bound='Quilt')


class Quilt(ContainerBase, StoreClientMixin):
    '''
    A :obj:`Frame`-like view of the contents of a :obj:`Bus` or :obj:`Yarn`. With the Quilt, :obj:`Frame` contained in a :obj:`Bus` or :obj:`Yarn` can be conceived as stacking vertically (primary axis 0) or horizontally (primary axis 1). If the labels of the primary axis are unique accross all contained :obj:`Frame`, ``retain_labels`` can be set to ``False`` and underlying labels are simply concatenated; otherwise, ``retain_labels`` must be set to ``True`` and an additional depth-level is added to the primary axis labels. A :obj:`Quilt` can only be created if labels of the opposite axis of all contained :obj:`Frame` are aligned.
    '''

    __slots__ = (
            '_bus',
            '_axis',
            '_index',
            '_columns',
            '_primary_index',
            '_secondary_index',
            '_retain_labels',
            '_include_index',
            '_deepcopy_from_bus',
            '_assign_axis',
            '_iloc_to_frame_label',
            '_frame_label_offset',
            )

    _bus: tp.Union[Bus, Yarn]
    _axis: int
    _index: IndexBase
    _columns: IndexBase
    _primary_index: tp.Optional[IndexBase]
    _secondary_index: tp.Optional[IndexBase]
    _retain_labels: bool
    _include_index: bool
    _deepcopy_from_bus: bool
    _assign_axis: bool
    _iloc_to_frame_label: tp.Optional[Series]
    _frame_label_offset: tp.Optional[Series]

    _NDIM: int = 2

    @classmethod
    def from_frame(cls: tp.Type[Q],
            frame: Frame,
            *,
            chunksize: int,
            retain_labels: bool,
            axis: int = 0,
            name: NameType = None,
            label_extractor: tp.Optional[tp.Callable[[IndexBase], tp.Hashable]] = None,
            config: StoreConfigMapInitializer = None,
            deepcopy_from_bus: bool = False,
            ) -> Q:
        '''
        Given a :obj:`Frame`, create a :obj:`Quilt` by partitioning it along the specified ``axis`` in units of ``chunksize``, where ``axis`` 0 partitions vertically (retaining aligned columns) and 1 partions horizontally (retaining aligned index).

        Args:
            label_extractor: Function that, given the partitioned index component along the specified axis, returns a string label for that chunk.
        '''
        vector = frame._index if axis == 0 else frame._columns
        vector_len = len(vector)

        starts = range(0, vector_len, chunksize)
        if len(starts) == 1:
            ends: tp.Iterable[int] = (vector_len,)
        else:
            ends = range(starts[1], vector_len, chunksize)

        if label_extractor is None:
            label_extractor = lambda x: x.iloc[0] # type: ignore

        axis_map_components: tp.Dict[tp.Hashable, IndexBase] = {}
        secondary_index = None

        def values() -> tp.Iterator[Frame]:
            nonlocal secondary_index

            for start, end in zip_longest(starts, ends, fillvalue=vector_len):
                if axis == 0: # along rows
                    f = frame.iloc[start:end]
                    label = label_extractor(f.index) # type: ignore
                    axis_map_components[label] = f.index
                    if secondary_index is None:
                        secondary_index = f.columns
                elif axis == 1: # along columns
                    f = frame.iloc[NULL_SLICE, start:end]
                    label = label_extractor(f.columns) # type: ignore
                    axis_map_components[label] = f.columns
                    if secondary_index is None:
                        secondary_index = f.index
                else:
                    raise AxisInvalid(f'invalid axis {axis}')
                yield f.rename(label)

        name = name if name else frame.name
        bus = Bus.from_frames(values(), config=config, name=name)

	primary_index = IndexHierarchy.from_tree(axis_map_components,
                index_constructors=IndexAutoConstructorFactory)

        return cls(bus,
                axis=axis,
                primary_index=primary_index,
                secondary_index=secondary_index,
                retain_labels=retain_labels,
                deepcopy_from_bus=deepcopy_from_bus,
                include_index=True,
                )

    #---------------------------------------------------------------------------
    # constructors by data format

    @classmethod
    def _from_store(cls: tp.Type[Q],
            store: Store,
            *,
            config: StoreConfigMapInitializer = None,
            axis: int = 0,
            retain_labels: bool,
            include_index: bool = True,
            deepcopy_from_bus: bool = False,
            max_persist: tp.Optional[int] = None,
            ) -> Q:
        bus = Bus._from_store(store=store,
                config=config,
                max_persist=max_persist, # None is default
                )
        return cls(bus,
                axis=axis,
                retain_labels=retain_labels,
                include_index=include_index,
                deepcopy_from_bus=deepcopy_from_bus,
                )

    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_zip_tsv(cls: tp.Type[Q],
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            axis: int = 0,
            retain_labels: bool,
            include_index: bool = True,
            deepcopy_from_bus: bool = False,
            max_persist: tp.Optional[int] = None,
            ) -> Q:
        '''
        Given a file path to zipped TSV :obj:`Quilt` store, return a :obj:`Quilt` instance.

        {args}
        '''
        store = StoreZipTSV(fp)
        return cls._from_store(store,
                config=config,
                axis=axis,
                retain_labels=retain_labels,
                include_index=include_index,
                deepcopy_from_bus=deepcopy_from_bus,
                max_persist=max_persist,
                )

    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_zip_csv(cls: tp.Type[Q],
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            axis: int = 0,
            retain_labels: bool,
            include_index: bool = True,
            deepcopy_from_bus: bool = False,
            max_persist: tp.Optional[int] = None,
            ) -> Q:
        '''
        Given a file path to zipped CSV :obj:`Quilt` store, return a :obj:`Quilt` instance.

        {args}
        '''
        store = StoreZipCSV(fp)
        return cls._from_store(store,
                config=config,
                axis=axis,
                retain_labels=retain_labels,
                include_index=include_index,
                deepcopy_from_bus=deepcopy_from_bus,
                max_persist=max_persist,
                )

    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_zip_pickle(cls: tp.Type[Q],
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            axis: int = 0,
            retain_labels: bool,
            include_index: bool = True,
            deepcopy_from_bus: bool = False,
            max_persist: tp.Optional[int] = None,
            ) -> Q:
        '''
        Given a file path to zipped pickle :obj:`Quilt` store, return a :obj:`Quilt` instance.

        {args}
        '''
        store = StoreZipPickle(fp)
        return cls._from_store(store,
                config=config,
                axis=axis,
                retain_labels=retain_labels,
                include_index=include_index,
                deepcopy_from_bus=deepcopy_from_bus,
                max_persist=max_persist,
                )

    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_zip_npz(cls: tp.Type[Q],
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            axis: int = 0,
            retain_labels: bool,
            include_index: bool = True,
            deepcopy_from_bus: bool = False,
            max_persist: tp.Optional[int] = None,
            ) -> Q:
        '''
        Given a file path to zipped parquet :obj:`Quilt` store, return a :obj:`Quilt` instance.

        {args}
        '''
        store = StoreZipNPZ(fp)
        return cls._from_store(store,
                config=config,
                axis=axis,
                retain_labels=retain_labels,
                include_index=include_index,
                deepcopy_from_bus=deepcopy_from_bus,
                max_persist=max_persist,
                )

    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_zip_parquet(cls: tp.Type[Q],
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            axis: int = 0,
            retain_labels: bool,
            include_index: bool = True,
            deepcopy_from_bus: bool = False,
            max_persist: tp.Optional[int] = None,
            ) -> Q:
        '''
        Given a file path to zipped parquet :obj:`Quilt` store, return a :obj:`Quilt` instance.

        {args}
        '''
        store = StoreZipParquet(fp)
        return cls._from_store(store,
                config=config,
                axis=axis,
                retain_labels=retain_labels,
                include_index=include_index,
                deepcopy_from_bus=deepcopy_from_bus,
                max_persist=max_persist,
                )

    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_xlsx(cls: tp.Type[Q],
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            axis: int = 0,
            retain_labels: bool,
            include_index: bool = True,
            deepcopy_from_bus: bool = False,
            max_persist: tp.Optional[int] = None,
            ) -> Q:
        '''
        Given a file path to an XLSX :obj:`Quilt` store, return a :obj:`Quilt` instance.

        {args}
        '''
        # how to pass configuration for multiple sheets?
        store = StoreXLSX(fp)
        return cls._from_store(store,
                config=config,
                axis=axis,
                retain_labels=retain_labels,
                include_index=include_index,
                deepcopy_from_bus=deepcopy_from_bus,
                max_persist=max_persist,
                )

    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_sqlite(cls: tp.Type[Q],
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            axis: int = 0,
            retain_labels: bool,
            include_index: bool = True,
            deepcopy_from_bus: bool = False,
            max_persist: tp.Optional[int] = None,
            ) -> Q:
        '''
        Given a file path to an SQLite :obj:`Quilt` store, return a :obj:`Quilt` instance.

        {args}
        '''
        store = StoreSQLite(fp)
        return cls._from_store(store,
                config=config,
                axis=axis,
                retain_labels=retain_labels,
                include_index=include_index,
                deepcopy_from_bus=deepcopy_from_bus,
                max_persist=max_persist,
                )

    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_hdf5(cls: tp.Type[Q],
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            axis: int = 0,
            retain_labels: bool,
            include_index: bool = True,
            deepcopy_from_bus: bool = False,
            max_persist: tp.Optional[int] = None,
            ) -> Q:
        '''
        Given a file path to a HDF5 :obj:`Quilt` store, return a :obj:`Quilt` instance.

        {args}
        '''
        store = StoreHDF5(fp)
        return cls._from_store(store,
                config=config,
                axis=axis,
                retain_labels=retain_labels,
                include_index=include_index,
                deepcopy_from_bus=deepcopy_from_bus,
                max_persist=max_persist,
                )

    #---------------------------------------------------------------------------

    @classmethod
    def from_items(cls: tp.Type[Q],
            items: tp.Iterable[tp.Tuple[tp.Hashable, Frame]],
            *,
            axis: int = 0,
            name: NameType = None,
            retain_labels: bool,
            include_index: bool = True,
            deepcopy_from_bus: bool = False,
            ) -> Q:
        '''
        Given an iterable of pairs of label, :obj:`Frame`, create a :obj:`Quilt`.
        '''
        bus = Bus.from_items(items, name=name)
        return cls(bus,
                axis=axis,
                retain_labels=retain_labels,
                include_index=include_index,
                deepcopy_from_bus=deepcopy_from_bus,
                )

    @classmethod
    def from_frames(cls: tp.Type[Q],
            frames: tp.Iterable[Frame],
            *,
            axis: int = 0,
            name: NameType = None,
            retain_labels: bool,
            include_index: bool = True,
            deepcopy_from_bus: bool = False,
            ) -> Q:
        '''Return a :obj:`Quilt` from an iterable of :obj:`Frame`; labels will be drawn from :obj:`Frame.name`.
        '''
        bus = Bus.from_frames(frames, name=name)
        return cls(bus,
                axis=axis,
                retain_labels=retain_labels,
                include_index=include_index,
                deepcopy_from_bus=deepcopy_from_bus,
                )

    #---------------------------------------------------------------------------
    @doc_inject(selector='quilt_init')
    def __init__(self: Q,
            bus: tp.Union[Bus, Yarn],
            *,
            axis: int = 0,
            retain_labels: bool,
            include_index: bool = True,
            deepcopy_from_bus: bool = False,
            primary_index: tp.Optional[IndexBase] = None,
            secondary_index: tp.Optional[IndexBase] = None,
            iloc_to_frame_label: tp.Optional[Series] = None,
            frame_label_offset: tp.Optional[Series] = None,
            ) -> None:
        '''
        {args}
        '''
        self._bus = bus
        self._axis = axis
        self._retain_labels = retain_labels if include_index else False
        self._deepcopy_from_bus = deepcopy_from_bus
        self._include_index = include_index

        if (primary_index is None) ^ (secondary_index is None):
            raise ErrorInitQuilt('if supplying primary_index, supply secondary_index')

        self._primary_index = primary_index
        self._secondary_index = secondary_index
        self._iloc_to_frame_label = iloc_to_frame_label
        self._frame_label_offset = frame_label_offset

        if primary_index is not None and secondary_index is not None:
            self._update_axis_labels()
        else:
            self._assign_axis = True

    #---------------------------------------------------------------------------
    # deferred loading of axis info

    @staticmethod
    def _error_update_axis_labels(axis: int) -> ErrorInitQuilt:
        axis_label = 'index' if axis == 0 else 'column'
        axis_labels = 'indices' if axis == 0 else 'columns'
        err_msg = f'Duplicate {axis_label} labels across frames. Either ensure all {axis_labels} are unique for all frames, set retain_labels=True to obtain an IndexHierarchy, or set include_index=False to use an auto-incremented index.'
        return ErrorInitQuilt(err_msg)

    def _update_axis_labels(self: Q) -> None:
        if self._primary_index is None or self._secondary_index is None:
            primary, self._secondary_index = build_quilt_indices(
                    self._bus,
                    axis=self._axis,
                    include_index=self._include_index,
                    deepcopy_from_bus=self._deepcopy_from_bus,
                    init_exception_cls=ErrorInitQuilt,
                    )

            if self._include_index:
                self._primary_index = primary # type: ignore
                self._iloc_to_frame_label = None
                self._frame_label_offset = None
            else:
                self._primary_index = primary.index # type: ignore
                self._iloc_to_frame_label = primary # type: ignore
                s: Series = self._iloc_to_frame_label.drop_duplicated(exclude_first=True) # type: ignore
                self._frame_label_offset = Series(s.index, index=s.values)

        if self._axis == 0:
            if not self._include_index or self._retain_labels:
                self._index = self._primary_index # type: ignore
            else:
                try:
                    self._index = self._primary_index.level_drop(1) # type: ignore
                except ErrorInitIndexNonUnique:
                    raise self._error_update_axis_labels(self._axis) from None

            self._columns = self._secondary_index
        else:
            if not self._include_index or self._retain_labels:
                self._columns = self._primary_index # type: ignore
            else:
                try:
                    self._columns = self._primary_index.level_drop(1) # type: ignore
                except ErrorInitIndexNonUnique:
                    raise self._error_update_axis_labels(self._axis) from None

            self._index = self._secondary_index

        self._assign_axis = False

    def unpersist(self: Q) -> None:
        '''For the :obj:`Bus` or :obj:`Yarn` contained in this object, replace all loaded :obj:`Frame` with :obj:`FrameDeferred`.
        '''
        self._bus.unpersist()

    #---------------------------------------------------------------------------
    # name interface

    @property # type: ignore
    @doc_inject()
    def name(self: Q) -> NameType:
        '''{}'''
        return self._bus.name # type: ignore

    def rename(self: Q,
            name: NameType,
            ) -> Q:
        '''
        Return a new :obj:`Quilt` with an updated name attribute.

        Args:
            name
        '''
        if not self._assign_axis:
            additional_kwargs = dict(
                    primary_index=self._primary_index,
                    secondary_index=self._secondary_index,
                    iloc_to_frame_label=self._iloc_to_frame_label,
                    frame_label_offset=self._frame_label_offset,
                    )
        else:
            additional_kwargs = {}

        return self.__class__(self._bus.rename(name),
                axis=self._axis,
                retain_labels=self._retain_labels,
                include_index=self._include_index,
                deepcopy_from_bus=self._deepcopy_from_bus,
                **additional_kwargs,
                )

    #---------------------------------------------------------------------------

    @doc_inject()
    def display(self: Q,
            config: tp.Optional[DisplayConfig] = None,
            *,
            style_config: tp.Optional[StyleConfig] = None,
            ) -> Display:
        '''{doc}

        Args:
            {config}
        '''
        if self._assign_axis:
            self._update_axis_labels()

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

        config = config or DisplayConfig()

        def placeholder_gen() -> tp.Iterator[tp.Iterable[tp.Any]]:
            assert config is not None
            yield from repeat(
                    tuple(
                        repeat(config.cell_placeholder, times=len(index))
                        ),
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

    #---------------------------------------------------------------------------
    # accessors

    @property # type: ignore
    @doc_inject(selector='values_2d', class_name='Quilt')
    def values(self: Q) -> np.ndarray:
        '''
        {}
        '''
        if self._assign_axis:
            self._update_axis_labels()

        return self.to_frame().values

    @property
    def index(self: Q) -> IndexBase:
        '''The ``IndexBase`` instance assigned for row labels.
        '''
        if self._assign_axis:
            self._update_axis_labels()

        return self._index

    @property
    def columns(self: Q) -> IndexBase:
        '''The ``IndexBase`` instance assigned for column labels.
        '''
        if self._assign_axis:
            self._update_axis_labels()

        return self._columns

    #---------------------------------------------------------------------------

    @property
    def shape(self: Q) -> ShapeType:
        '''
        Return a tuple describing the shape of the underlying NumPy array.

        Returns:
            :obj:`tp.Tuple[int]`
        '''
        if self._assign_axis:
            self._update_axis_labels()

        return len(self._index), len(self._columns)

    @property
    def ndim(self: Q) -> int:
        '''
        Return the number of dimensions, which for a `Frame` is always 2.

        Returns:
            :obj:`int`
        '''
        return self._NDIM

    @property
    def size(self: Q) -> int:
        '''
        Return the size of the underlying NumPy array.

        Returns:
            :obj:`int`
        '''
        if self._assign_axis:
            self._update_axis_labels()

        return len(self._index) * len(self._columns)

    @property
    def nbytes(self: Q) -> int:
        '''
        Return the total bytes of the underlying NumPy arrays.

        Returns:
            :obj:`int`
        '''
        # return self._blocks.nbytes
        if self._assign_axis:
            self._update_axis_labels()

        return sum(f.nbytes for f in self._bus.values)

    @property
    def status(self: Q) -> Frame:
        '''
        Return a :obj:`Frame` indicating loaded status, size, bytes, and shape of all loaded :obj:`Frame` in the contained :obj:`Quilt`.
        '''
        return self._bus.status

    #---------------------------------------------------------------------------
    # dictionary-like interface

    def keys(self: Q) -> tp.Iterable[tp.Hashable]:
        '''Iterator of column labels.
        '''
        if self._assign_axis:
            self._update_axis_labels()

        return self._columns

    def __iter__(self: Q) -> tp.Iterable[tp.Hashable]:
        '''
        Iterator of column labels, same as :py:meth:`Frame.keys`.
        '''
        if self._assign_axis:
            self._update_axis_labels()

        return self._columns.__iter__()

    def __contains__(self: Q,
            value: tp.Hashable,
            ) -> bool:
        '''
        Inclusion of value in column labels.
        '''
        if self._assign_axis:
            self._update_axis_labels()

        return self._columns.__contains__(value)

    def items(self: Q) -> tp.Iterator[tp.Tuple[tp.Hashable, Series]]:
        '''Iterator of pairs of column label and corresponding column :obj:`Series`.
        '''
        if self._assign_axis:
            self._update_axis_labels()

        yield from self._axis_series_items(axis=0) # iterate columns

    def get(self: Q,
            key: tp.Hashable,
            default: tp.Optional[tp.Any] = None,
            ) -> tp.Optional[tp.Union[tp.Any, Series, Frame]]:
        '''
        Return the value found at the columns key, else the default if the key is not found. This method is implemented to complete the dictionary-like interface.
        '''
        if self._assign_axis:
            self._update_axis_labels()

        if key not in self._columns:
            return default
        return self.__getitem__(key)

    #---------------------------------------------------------------------------
    # compatibility with StoreClientMixin

    def _items_store(self: Q) -> tp.Iterator[tp.Tuple[tp.Hashable, Frame]]:
        '''Iterator of pairs of :obj:`Quilt` label and contained :obj:`Frame`.
        '''
        yield from self._bus.items()

    #---------------------------------------------------------------------------
    # axis iterators

    def _axis_array(self: Q,
            axis: int,
            ) -> tp.Iterator[np.ndarray]:
        '''Generator of arrays across an axis

        Args:
            axis: 0 iterates over columns, 1 iterates over rows
        '''
        extractor = get_extractor(
                self._deepcopy_from_bus,
                is_array=True,
                memo_active=False,
                )

        if axis == self._axis:
            raise NotImplementedAxis()

        if axis >= 2 or axis < 0:
            raise AxisInvalid(f'no support for axis {axis}')

        for component in self._bus.values:
            for array in component._blocks.axis_values(axis):
                yield extractor(array)

    def _axis_array_items(self: Q,
            axis: int,
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, np.ndarray]]:
        keys = self._index if axis == 1 else self._columns
        yield from zip(keys, self._axis_array(axis))

    def _axis_tuple(self: Q,
            *,
            axis: int,
            constructor: tp.Optional[tp.Type[tp.NamedTuple]] = None,
            ) -> tp.Iterator[tp.NamedTuple]:
        '''Generator of named tuples across an axis.

        Args:
            axis: 0 iterates over columns (index axis), 1 iterates over rows (column axis)
        '''
        tuple_constructor: tp.Callable[[tp.Sequence[tp.Any]], tp.NamedTuple]
        if constructor is None:
            if axis == 1:
                labels = self._columns.values
            elif axis == 0:
                labels = self._index.values
            else:
                raise AxisInvalid(f'no support for axis {axis}')
            # uses _make method to call with iterable
            tuple_constructor = get_tuple_constructor(labels) # type: ignore

        elif (isinstance(constructor, type) and
                issubclass(constructor, tuple) and
                hasattr(constructor, '_make')):
            tuple_constructor = constructor._make

        else:
            tuple_constructor = constructor

        for axis_values in self._axis_array(axis):
            yield tuple_constructor(axis_values)

    def _axis_tuple_items(self: Q,
            *,
            axis: int,
            constructor: tp.Optional[tp.Type[tp.NamedTuple]] = None,
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, tp.NamedTuple]]:
        keys = self._index if axis == 1 else self._columns
        yield from zip(keys, self._axis_tuple(axis=axis, constructor=constructor))

    def _axis_series(self: Q,
            axis: int,
            ) -> tp.Iterator[Series]:
        '''Generator of Series across an axis
        '''
        index = self._index if axis == 0 else self._columns
        for label, axis_values in self._axis_array_items(axis):
            yield Series(axis_values, index=index, name=label, own_index=True)

    def _axis_series_items(self: Q,
            axis: int,
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, np.ndarray]]:
        keys = self._index if axis == 1 else self._columns
        yield from zip(keys, self._axis_series(axis=axis))

    #---------------------------------------------------------------------------

    def _axis_window_items(self: Q,
            *,
            size: int,
            axis: int = 0,
            step: int = 1,
            window_sized: bool = True,
            window_func: tp.Optional[AnyCallable] = None,
            window_valid: tp.Optional[AnyCallable] = None,
            label_shift: int = 0,
            start_shift: int = 0,
            size_increment: int = 0,
            as_array: bool = False,
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, tp.Any]]:
        '''Generator of index, processed-window pairs.
        '''
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
                start_shift=start_shift,
                size_increment=size_increment,
                as_array=as_array
                )

    def _axis_window(self: Q,
            *,
            size: int,
            axis: int = 0,
            step: int = 1,
            window_sized: bool = True,
            window_func: tp.Optional[AnyCallable] = None,
            window_valid: tp.Optional[AnyCallable] = None,
            label_shift: int = 0,
            start_shift: int = 0,
            size_increment: int = 0,
            as_array: bool = False,
            ) -> tp.Iterator[Frame]:
        yield from (x for _, x in self._axis_window_items(
                size=size,
                axis=axis,
                step=step,
                window_sized=window_sized,
                window_func=window_func,
                window_valid=window_valid,
                label_shift=label_shift,
                start_shift=start_shift,
                size_increment=size_increment,
                as_array=as_array
                ))

    #---------------------------------------------------------------------------

    def _extract_null_slice_array(self: Q,
            extractor: AnyCallable,
            ) -> np.ndarray:
        if len(self._bus) == 1:
            return extractor(self._bus.iloc[0].values)

        # NOTE: concatenate allocates a new array, meaning we don't need to use extractor
        arrays = [f.values for f in self._bus.values]
        return concat_resolved(
                arrays,
                axis=self._axis,
                )

    def _extract_no_hierarchy_array(self: Q,
            primary_index_sel: tp.Union[int, IndexBase],
            sel_reduces: bool,
            opposite_reduces: bool,
            extractor: AnyCallable,
            opposite_key: GetItemKeyType,
            ) -> tp.Union[tp.Any, Frame, Series]:
        '''Specialized path for when `_include_index` is False'''
        if sel_reduces:
            primary_index_sel = (primary_index_sel,) # type: ignore

        components: tp.List[tp.Union[tp.Any, np.ndarray]] = []

        for iloc_key in primary_index_sel: # type: ignore
            frame_label = self._iloc_to_frame_label[iloc_key] # type: ignore
            sel_component = iloc_key - self._frame_label_offset[frame_label] # type: ignore

            func = self._bus.loc[frame_label]._extract_array # type: ignore
            if sel_reduces:
                components.append(func(sel_component, opposite_key))
            else:
                components.append(func([sel_component], opposite_key))

        if sel_reduces and opposite_reduces:
            return components[0]

        if sel_reduces or opposite_reduces:
            return concat_resolved(components)
        return concat_resolved(components, axis=self._axis)

    def _extract_hierarchy_array(self: Q,
            sel_key: GetItemKeyType,
            opposite_key: GetItemKeyType,
            sel_reduces: bool,
            opposite_reduces: bool,
            primary_index_sel: tp.Union[tp.Tuple[tp.Hashable, ...], IndexBase],
            extractor: AnyCallable,
            ) -> tp.Union[tp.Any, np.ndarray]:
        sel = np.full(len(self._primary_index), False)
        sel[sel_key] = True

        # get ordered unique Bus labels
        frame_labels: tp.Iterable[tp.Hashable]

        if isinstance(primary_index_sel, tuple):
            frame_labels = (primary_index_sel[0],)
        else:
            frame_labels = primary_index_sel._get_unique_labels_in_occurence_order(depth=0) # type: ignore

        parts: tp.List[np.ndarray] = []

        for frame_label in frame_labels:
            sel_component = sel[self._primary_index._loc_to_iloc(HLoc[frame_label])] # type: ignore

            if self._axis == 0:
                component = self._bus.loc[frame_label]._extract_array(sel_component, opposite_key) # type: ignore
                if sel_reduces:
                    component = component[0]
            else:
                component = self._bus.loc[frame_label]._extract_array(opposite_key, sel_component) # type: ignore
                if sel_reduces:
                    if component.ndim == 1:
                        component = component[0]
                    elif component.ndim == 2:
                        component = component[NULL_SLICE, 0]

            parts.append(component)

        if len(parts) == 1:
            return extractor(parts.pop())

        # NOTE: concatenate allocates a new array, meaning we don't need to use extractor
        if sel_reduces or opposite_reduces:
            return concat_resolved(parts)
        return concat_resolved(parts, axis=self._axis)

    #---------------------------------------------------------------------------

    def _extract_null_slice(self: Q,
            extractor: AnyCallable,
            ) -> Frame:
        if self._retain_labels and self._axis == 0:
            frames = (
                    extractor(frame.relabel_level_add(index=label))
                    for label, frame in self._bus.items()
                    )
        elif self._retain_labels and self._axis == 1:
            frames = (
                    extractor(frame.relabel_level_add(columns=label))
                    for label, frame in self._bus.items()
                    )
        else:
            frames = (extractor(frame) for frame in self._bus.values)

        if not self._include_index:
            return Frame.from_concat(frames, axis=self._axis, index=self._primary_index) # type: ignore

        return Frame.from_concat(frames, axis=self._axis) # type: ignore

    def _extract_no_hierarchy(self: Q,
            primary_index_sel: tp.Union[int, IndexBase],
            sel_reduces: bool,
            opposite_reduces: bool,
            extractor: AnyCallable,
            opposite_key: GetItemKeyType,
            ) -> tp.Union[tp.Any, Frame, Series]:
        '''Specialized path for when `_include_index` is False'''
        if sel_reduces:
            primary_index_sel = (primary_index_sel,) # type: ignore

        def gen_components() -> tp.Iterator[tp.Union[tp.Any, Frame, Series]]:
            for iloc_key in primary_index_sel: # type: ignore
                frame_label = self._iloc_to_frame_label[iloc_key] # type: ignore
                sel_component = iloc_key - self._frame_label_offset[frame_label] # type: ignore

                component = self._bus.loc[frame_label].iloc[sel_component, opposite_key]
                if opposite_reduces:
                    yield extractor(component)
                else:
                    yield extractor(component.rename(iloc_key))

        if sel_reduces and opposite_reduces:
            return next(gen_components())

        if opposite_reduces:
            return Series(
                    gen_components(),
                    index=primary_index_sel,
                    name=self._secondary_index[opposite_key], # type: ignore
                    )

        if sel_reduces:
            return Series.from_concat(gen_components())

        return Frame.from_concat(
                gen_components(),
                axis=self._axis,
                )

    def _extract_hierarchy(self: Q,
            sel_key: GetItemKeyType,
            opposite_key: GetItemKeyType,
            sel_reduces: bool,
            opposite_reduces: bool,
            primary_index_sel: tp.Union[tp.Tuple[tp.Hashable, ...], IndexBase],
            extractor: AnyCallable,
            ) -> tp.Union[tp.Any, Series, Frame]:
        sel = np.full(len(self._primary_index), False)
        sel[sel_key] = True

        def get_component(frame_label: tp.Hashable) -> tp.Any:
            sel_component = sel[self._primary_index._loc_to_iloc(HLoc[frame_label])] # type: ignore
            if self._axis == 0:
                return self._bus.loc[frame_label].iloc[sel_component, opposite_key]

            return self._bus.loc[frame_label].iloc[opposite_key, sel_component]

        if sel_reduces and opposite_reduces:
            return get_component(primary_index_sel[0]).iloc[0]

        frame_label_components: tp.Iterable[tp.Tuple[tp.Hashable, tp.Union[tp.Any, Frame, Series]]]

        if isinstance(primary_index_sel, tuple):
            frame_label = primary_index_sel[0]
            frame_label_components = ((frame_label, get_component(frame_label)),)
        else:
            # get the outer level, or just the unique frame labels needed
            frame_labels = primary_index_sel._get_unique_labels_in_occurence_order(depth=0) # type: ignore
            frame_label_components = (
                    (frame_label, get_component(frame_label))
                    for frame_label in frame_labels
                    )

        # Short-circuit if there is no relabeling or decomposition needed
        if not sel_reduces and not opposite_reduces and not self._retain_labels:
            return Frame.from_concat(
                    (component for _, component in frame_label_components),
                    axis=self._axis,
                    )

        # Finally, process the components to feed correctly relabeled/decomposed components
        def gen_components() -> tp.Iterator[tp.Union[tp.Any, Frame, Series]]:
            for frame_label, component in frame_label_components:
                if self._retain_labels:
                    if self._axis == 0 or opposite_reduces:
                        # Component is either a Series or Frame, which means we can call the same with first arg
                        component = component.relabel_level_add(frame_label)
                    else:
                        component = component.relabel_level_add(columns=frame_label)

                if sel_reduces:
                    # make Frame into a Series
                    if self._axis == 0:
                        component = component.iloc[0]
                    else:
                        component = component.iloc[NULL_SLICE, 0]

                # NOTE: from_concat will attempt to re-use ndarrays, and thus extractor is needed
                yield extractor(component)

        if sel_reduces or opposite_reduces:
            return Series.from_concat(gen_components())

        return Frame.from_concat(gen_components(), axis=self._axis)

    #---------------------------------------------------------------------------

    def _extract_helper(self: Q,
            is_array: bool,
            row_key: GetItemKeyType = None,
            column_key: GetItemKeyType = None,
            ) -> tp.Any:
        if self._assign_axis:
            self._update_axis_labels()

        if is_array:
            null_slice_extractor = self._extract_null_slice_array
            no_hierarchy_extractor = self._extract_no_hierarchy_array
            hierarchy_extractor = self._extract_hierarchy_array
        else:
            null_slice_extractor = self._extract_null_slice
            no_hierarchy_extractor = self._extract_no_hierarchy
            hierarchy_extractor = self._extract_hierarchy

        extractor = get_extractor(
                self._deepcopy_from_bus,
                is_array=is_array,
                memo_active=False,
                )

        row_key = NULL_SLICE if row_key is None else row_key
        column_key = NULL_SLICE if column_key is None else column_key

        if (
            row_key.__class__ is slice and row_key == NULL_SLICE and
            column_key.__class__ is slice and column_key == NULL_SLICE
            ):
            return null_slice_extractor(extractor)

        if self._axis == 0:
            sel_key = row_key
            opposite_key = column_key
        else:
            sel_key = column_key
            opposite_key = row_key

        sel_reduces = isinstance(sel_key, INT_TYPES)
        opposite_reduces = isinstance(opposite_key, INT_TYPES)

        # get ordered unique Bus labels
        primary_index_sel = self._primary_index.iloc[sel_key] # type: ignore

        if not self._include_index:
            return no_hierarchy_extractor(
                primary_index_sel=primary_index_sel,
                sel_reduces=sel_reduces,
                opposite_reduces=opposite_reduces,
                extractor=extractor,
                opposite_key=opposite_key,
            )

        return hierarchy_extractor(
                sel_key=sel_key,
                opposite_key=opposite_key,
                sel_reduces=sel_reduces,
                opposite_reduces=opposite_reduces,
                primary_index_sel=primary_index_sel,
                extractor=extractor,
                )

    def _extract_array(self: Q,
            row_key: GetItemKeyType = None,
            column_key: GetItemKeyType = None,
            ) -> tp.Union[tp.Any, np.ndarray]:
        '''
        Extract a consolidated array based on iloc selection.
        '''
        return self._extract_helper(
                is_array=True,
                row_key=row_key,
                column_key=column_key,
                )

    def _extract(self: Q,
            row_key: GetItemKeyType = None,
            column_key: GetItemKeyType = None,
            ) -> tp.Union[tp.Any, Frame, Series]:
        '''
        Extract Container based on iloc selection.
        '''
        return self._extract_helper(
                is_array=False,
                row_key=row_key,
                column_key=column_key,
                )

    #---------------------------------------------------------------------------

    @doc_inject(selector='sample')
    def sample(self: Q,
            index: tp.Optional[int] = None,
            columns: tp.Optional[int] = None,
            *,
            seed: tp.Optional[int] = None,
            ) -> Frame:
        '''
        {doc}

        Args:
            {index}
            {columns}
            {seed}
        '''
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

        return self._extract(row_key=index_key, column_key=columns_key) # type: ignore

    #---------------------------------------------------------------------------

    def _extract_iloc(self: Q,
            key: GetItemKeyTypeCompound,
            ) -> tp.Union[Series, Frame]:
        '''
        Give a compound key, return a new Frame. This method simply handles the variability of single or compound selectors.
        '''
        if self._assign_axis:
            self._update_axis_labels()

        if isinstance(key, tuple):
            return self._extract(*key)
        return self._extract(row_key=key)

    def _compound_loc_to_iloc(self: Q,
            key: GetItemKeyTypeCompound,
            ) -> tp.Tuple[GetItemKeyType, GetItemKeyType]:
        '''
        Given a compound iloc key, return a tuple of row, column keys. Assumes the first argument is always a row extractor.
        '''
        if isinstance(key, tuple):
            loc_row_key, loc_column_key = key
            iloc_column_key = self._columns._loc_to_iloc(loc_column_key)
        else:
            loc_row_key = key
            iloc_column_key = None

        iloc_row_key = self._index._loc_to_iloc(loc_row_key)
        return iloc_row_key, iloc_column_key

    def _extract_loc(self: Q,
            key: GetItemKeyTypeCompound,
            ) -> tp.Union[Series, Frame]:
        if self._assign_axis:
            self._update_axis_labels()

        return self._extract(*self._compound_loc_to_iloc(key))

    def _compound_loc_to_getitem_iloc(self: Q,
            key: GetItemKeyTypeCompound,
            ) -> tp.Tuple[GetItemKeyType, GetItemKeyType]:
        '''Handle a potentially compound key in the style of __getitem__. This will raise an appropriate exception if a two argument loc-style call is attempted.
        '''
        iloc_column_key = self._columns._loc_to_iloc(key)
        return None, iloc_column_key

    @doc_inject(selector='selector')
    def __getitem__(self: Q,
            key: GetItemKeyType,
            ) -> tp.Union[tp.Any, Frame, Series]:
        '''Selector of columns by label.

        Args:
            key: {key_loc}
        '''
        if self._assign_axis:
            self._update_axis_labels()

        return self._extract(*self._compound_loc_to_getitem_iloc(key))

    #---------------------------------------------------------------------------
    # interfaces

    @property
    def loc(self: Q) -> InterfaceGetItem[Frame]:
        return InterfaceGetItem(self._extract_loc)

    @property
    def iloc(self: Q) -> InterfaceGetItem[Frame]:
        return InterfaceGetItem(self._extract_iloc)

    #---------------------------------------------------------------------------
    # iterators

    @property
    def iter_array(self: Q) -> IterNodeAxis['Quilt']:
        '''
        Iterator of :obj:`np.array`, where arrays are drawn from columns (axis=0) or rows (axis=1)
        '''
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
    def iter_array_items(self: Q) -> IterNodeAxis['Quilt']:
        '''
        Iterator of pairs of label, :obj:`np.array`, where arrays are drawn from columns (axis=0) or rows (axis=1)
        '''
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
    def iter_tuple(self: Q) -> IterNodeConstructorAxis['Quilt']:
        '''
        Iterator of :obj:`NamedTuple`, where tuples are drawn from columns (axis=0) or rows (axis=1). An optional ``constructor`` callable can be used to provide a :obj:`NamedTuple` class (or any other constructor called with a single iterable) to be used to create each yielded axis value.
        '''
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
    def iter_tuple_items(self: Q) -> IterNodeConstructorAxis['Quilt']:
        '''
        Iterator of pairs of label, :obj:`NamedTuple`, where tuples are drawn from columns (axis=0) or rows (axis=1)
        '''
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
    def iter_series(self: Q) -> IterNodeAxis['Quilt']:
        '''
        Iterator of :obj:`Series`, where :obj:`Series` are drawn from columns (axis=0) or rows (axis=1)
        '''
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
    def iter_series_items(self: Q) -> IterNodeAxis['Quilt']:
        '''
        Iterator of pairs of label, :obj:`Series`, where :obj:`Series` are drawn from columns (axis=0) or rows (axis=1)
        '''
        if self._assign_axis:
            self._update_axis_labels()

        return IterNodeAxis(
                container=self,
                function_values=self._axis_series,
                function_items=self._axis_series_items,
                yield_type=IterNodeType.ITEMS,
                apply_type=IterNodeApplyType.SERIES_VALUES,
                )

    #---------------------------------------------------------------------------
    def _iter_window(self: Q,
            as_array: bool,
            yield_type: IterNodeType,
            ) -> IterNodeWindow['Quilt']:
        if self._assign_axis:
            self._update_axis_labels()

        function_values = partial(self._axis_window, as_array=as_array)
        function_items = partial(self._axis_window_items, as_array=as_array)

        return IterNodeWindow(
                container=self,
                function_values=function_values,
                function_items=function_items,
                yield_type=yield_type,
                apply_type=IterNodeApplyType.SERIES_ITEMS,
                )

    @property
    @doc_inject(selector='window')
    def iter_window(self: Q) -> IterNodeWindow['Quilt']:
        '''
        Iterator of windowed values, where values are given as a :obj:`Frame`.

        {args}
        '''
        return self._iter_window(False, IterNodeType.VALUES)

    @property
    @doc_inject(selector='window')
    def iter_window_items(self: Q) -> IterNodeWindow['Quilt']:
        '''
        Iterator of pairs of label, windowed values, where values are given as a :obj:`Frame`.

        {args}
        '''
        return self._iter_window(False, IterNodeType.ITEMS)

    @property
    @doc_inject(selector='window')
    def iter_window_array(self: Q) -> IterNodeWindow['Quilt']:
        '''
        Iterator of windowed values, where values are given as a :obj:`np.array`.

        {args}
        '''
        return self._iter_window(True, IterNodeType.VALUES)

    @property
    @doc_inject(selector='window')
    def iter_window_array_items(self: Q) -> IterNodeWindow['Quilt']:
        '''
        Iterator of pairs of label, windowed values, where values are given as a :obj:`np.array`.

        {args}
        '''
        return self._iter_window(True, IterNodeType.ITEMS)

    #---------------------------------------------------------------------------
    # transformations resulting in changed dimensionality

    @doc_inject(selector='head', class_name='Quilt')
    def head(self: Q,
            count: int = 5,
            ) -> Frame:
        '''{doc}

        Args:
            {count}
        '''
        return self.iloc[:count]

    @doc_inject(selector='tail', class_name='Quilt')
    def tail(self: Q,
            count: int = 5,
            ) -> Frame:
        '''{doc}

        Args:
            {count}
        '''
        return self.iloc[-count:]

    #---------------------------------------------------------------------------

    @doc_inject()
    def equals(self: Q,
            other: tp.Any,
            *,
            compare_name: bool = False,
            compare_dtype: bool = False,
            compare_class: bool = False,
            skipna: bool = True,
            ) -> bool:
        '''
        {doc}

        Note: this will attempt to load and compare all Frame managed by the Bus stored within this Quilt.

        Args:
            {compare_name}
            {compare_dtype}
            {compare_class}
            {skipna}
        '''
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

        if self._include_index != other._include_index:
            return False

        if compare_name and self.name != other.name:
            return False

        if self._assign_axis:
            self._update_axis_labels()

        if other._assign_axis:
            other._update_axis_labels()

        if not self._index.equals(
                other._index,
                compare_name=compare_name,
                compare_dtype=compare_dtype,
                compare_class=compare_class,
                skipna=skipna,
                ):
            return False

        if not self._columns.equals(
                other._columns,
                compare_name=compare_name,
                compare_dtype=compare_dtype,
                compare_class=compare_class,
                skipna=skipna,
                ):
            return False

        if not self._bus.equals(other._bus,
                compare_name=compare_name,
                compare_dtype=compare_dtype,
                compare_class=compare_class,
                skipna=skipna,
                ):
            return False

        return True

    #---------------------------------------------------------------------------

    def to_frame(self: Q) -> Frame:
        '''
        Return a consolidated :obj:`Frame`.
        '''
        if self._assign_axis:
            self._update_axis_labels()
        return self._extract(NULL_SLICE, NULL_SLICE) # type: ignore
