import typing as tp
from itertools import zip_longest
from functools import partial

import numpy as np

from static_frame.core.bus import Bus
from static_frame.core.container import ContainerBase
from static_frame.core.container_util import axis_window_items
from static_frame.core.display import Display
from static_frame.core.display_config import DisplayConfig
from static_frame.core.doc_str import doc_inject
from static_frame.core.exception import AxisInvalid
from static_frame.core.exception import ErrorInitQuilt
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
from static_frame.core.util import AnyCallable
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

from static_frame.core.axis_map import bus_to_hierarchy
from static_frame.core.axis_map import get_extractor

class Quilt(ContainerBase, StoreClientMixin):
    '''
    A :obj:`Frame`-like view of the contents of a :obj:`Bus`. With the Quilt, :obj:`Frame` contained in a :obj:`Bus` can be conceived as stacking vertically (primary axis 0) or horizontally (primary axis 1). If the labels of the primary axis are unique accross all contained :obj:`Frame`, ``retain_labels`` can be set to ``False`` and underlying labels are simply concatenated; otherwise, ``retain_labels`` must be set to ``True`` and an additional depth-level is added to the primary axis labels. A :obj:`Quilt` can only be created if labels of the opposite axis of all contained :obj:`Frame` are aligned.
    '''

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

    _bus: Bus
    _axis: int
    _axis_hierarchy: tp.Optional[IndexHierarchy]
    _axis_opposite: tp.Optional[IndexBase]
    _columns: IndexBase
    _index: IndexBase
    _assign_axis: bool

    _NDIM: int = 2

    @classmethod
    def from_frame(cls,
            frame: Frame,
            *,
            chunksize: int,
            retain_labels: bool,
            axis: int = 0,
            name: NameType = None,
            label_extractor: tp.Optional[tp.Callable[[IndexBase], tp.Hashable]] = None,
            config: StoreConfigMapInitializer = None,
            deepcopy_from_bus: bool = False,
            ) -> 'Quilt':
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
            label_extractor = lambda x: x.iloc[0] #type: ignore

        axis_map_components: tp.Dict[tp.Hashable, IndexBase] = {}
        opposite = None

        def values() -> tp.Iterator[Frame]:
            nonlocal opposite

            for start, end in zip_longest(starts, ends, fillvalue=vector_len):
                if axis == 0: # along rows
                    f = frame.iloc[start:end]
                    label = label_extractor(f.index) #type: ignore
                    axis_map_components[label] = f.index
                    if opposite is None:
                        opposite = f.columns
                elif axis == 1: # along columns
                    f = frame.iloc[:, start:end]
                    label = label_extractor(f.columns) #type: ignore
                    axis_map_components[label] = f.columns
                    if opposite is None:
                        opposite = f.index
                else:
                    raise AxisInvalid(f'invalid axis {axis}')
                yield f.rename(label)

        name = name if name else frame.name
        bus = Bus.from_frames(values(), config=config, name=name)

        axis_hierarchy = IndexHierarchy.from_tree(axis_map_components)

        return cls(bus,
                axis=axis,
                axis_hierarchy=axis_hierarchy,
                axis_opposite=opposite,
                retain_labels=retain_labels,
                deepcopy_from_bus=deepcopy_from_bus,
                )

    #---------------------------------------------------------------------------
    # constructors by data format

    @classmethod
    def _from_store(cls,
            store: Store,
            *,
            config: StoreConfigMapInitializer = None,
            axis: int = 0,
            retain_labels: bool,
            deepcopy_from_bus: bool = False,
            max_persist: tp.Optional[int] = None,
            ) -> 'Quilt':
        bus = Bus._from_store(store=store,
                config=config,
                max_persist=max_persist, # None is default
                )
        return cls(bus,
                axis=axis,
                retain_labels=retain_labels,
                deepcopy_from_bus=deepcopy_from_bus,
                )


    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_zip_tsv(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            axis: int = 0,
            retain_labels: bool,
            deepcopy_from_bus: bool = False,
            max_persist: tp.Optional[int] = None,
            ) -> 'Quilt':
        '''
        Given a file path to zipped TSV :obj:`Quilt` store, return a :obj:`Quilt` instance.

        {args}
        '''
        store = StoreZipTSV(fp)
        return cls._from_store(store,
                config=config,
                axis=axis,
                retain_labels=retain_labels,
                deepcopy_from_bus=deepcopy_from_bus,
                max_persist=max_persist,
                )

    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_zip_csv(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            axis: int = 0,
            retain_labels: bool,
            deepcopy_from_bus: bool = False,
            max_persist: tp.Optional[int] = None,
            ) -> 'Quilt':
        '''
        Given a file path to zipped CSV :obj:`Quilt` store, return a :obj:`Quilt` instance.

        {args}
        '''
        store = StoreZipCSV(fp)
        return cls._from_store(store,
                config=config,
                axis=axis,
                retain_labels=retain_labels,
                deepcopy_from_bus=deepcopy_from_bus,
                max_persist=max_persist,
                )

    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_zip_pickle(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            axis: int = 0,
            retain_labels: bool,
            deepcopy_from_bus: bool = False,
            max_persist: tp.Optional[int] = None,
            ) -> 'Quilt':
        '''
        Given a file path to zipped pickle :obj:`Quilt` store, return a :obj:`Quilt` instance.

        {args}
        '''
        store = StoreZipPickle(fp)
        return cls._from_store(store,
                config=config,
                axis=axis,
                retain_labels=retain_labels,
                deepcopy_from_bus=deepcopy_from_bus,
                max_persist=max_persist,
                )


    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_zip_parquet(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            axis: int = 0,
            retain_labels: bool,
            deepcopy_from_bus: bool = False,
            max_persist: tp.Optional[int] = None,
            ) -> 'Quilt':
        '''
        Given a file path to zipped parquet :obj:`Quilt` store, return a :obj:`Quilt` instance.

        {args}
        '''
        store = StoreZipParquet(fp)
        return cls._from_store(store,
                config=config,
                axis=axis,
                retain_labels=retain_labels,
                deepcopy_from_bus=deepcopy_from_bus,
                max_persist=max_persist,
                )


    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_xlsx(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            axis: int = 0,
            retain_labels: bool,
            deepcopy_from_bus: bool = False,
            max_persist: tp.Optional[int] = None,
            ) -> 'Quilt':
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
                deepcopy_from_bus=deepcopy_from_bus,
                max_persist=max_persist,
                )


    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_sqlite(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            axis: int = 0,
            retain_labels: bool,
            deepcopy_from_bus: bool = False,
            max_persist: tp.Optional[int] = None,
            ) -> 'Quilt':
        '''
        Given a file path to an SQLite :obj:`Quilt` store, return a :obj:`Quilt` instance.

        {args}
        '''
        store = StoreSQLite(fp)
        return cls._from_store(store,
                config=config,
                axis=axis,
                retain_labels=retain_labels,
                deepcopy_from_bus=deepcopy_from_bus,
                max_persist=max_persist,
                )


    @classmethod
    @doc_inject(selector='quilt_constructor')
    def from_hdf5(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            axis: int = 0,
            retain_labels: bool,
            deepcopy_from_bus: bool = False,
            max_persist: tp.Optional[int] = None,
            ) -> 'Quilt':
        '''
        Given a file path to a HDF5 :obj:`Quilt` store, return a :obj:`Quilt` instance.

        {args}
        '''
        store = StoreHDF5(fp)
        return cls._from_store(store,
                config=config,
                axis=axis,
                retain_labels=retain_labels,
                deepcopy_from_bus=deepcopy_from_bus,
                max_persist=max_persist,
                )

    #---------------------------------------------------------------------------

    @classmethod
    def from_items(cls,
            items: tp.Iterable[tp.Tuple[tp.Hashable, Frame]],
            *,
            axis: int = 0,
            name: NameType = None,
            retain_labels: bool,
            deepcopy_from_bus: bool = False,
            ) -> 'Quilt':
        '''
        Given an iterable of pairs of label, :obj:`Frame`, create a :obj:`Quilt`.
        '''
        bus = Bus.from_items(items, name=name)
        return cls(bus,
                axis=axis,
                retain_labels=retain_labels,
                deepcopy_from_bus=deepcopy_from_bus,
                )

    @classmethod
    def from_frames(cls,
            frames: tp.Iterable[Frame],
            *,
            axis: int = 0,
            name: NameType = None,
            retain_labels: bool,
            deepcopy_from_bus: bool = False,
            ) -> 'Quilt':
        '''Return a :obj:`Quilt` from an iterable of :obj:`Frame`; labels will be drawn from :obj:`Frame.name`.
        '''
        bus = Bus.from_frames(frames, name=name)
        return cls(bus,
                axis=axis,
                retain_labels=retain_labels,
                deepcopy_from_bus=deepcopy_from_bus,
                )

    #---------------------------------------------------------------------------
    @doc_inject(selector='quilt_init')
    def __init__(self,
            bus: Bus,
            *,
            axis: int = 0,
            retain_labels: bool,
            axis_hierarchy: tp.Optional[IndexHierarchy] = None,
            axis_opposite: tp.Optional[IndexBase] = None,
            deepcopy_from_bus: bool = False,
            ) -> None:
        '''
        {args}
        '''
        self._bus = bus
        self._axis = axis
        self._retain_labels = retain_labels
        self._deepcopy_from_bus = deepcopy_from_bus

        if (axis_hierarchy is None) ^ (axis_opposite is None):
            raise ErrorInitQuilt('if supplying axis_hierarchy, supply axis_opposite')

        # can creation until needed
        self._axis_hierarchy = axis_hierarchy
        self._axis_opposite = axis_opposite
        # will be set with re-axis
        # self._index = None
        # self._columns = None
        self._assign_axis = True # Boolean to controll deferred axis index creation

    #---------------------------------------------------------------------------
    # deferred loading of axis info

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
                self._index = self._axis_hierarchy.level_drop(1)
            else: # get hierarchical
                self._index = self._axis_hierarchy
            self._columns = self._axis_opposite
        else:
            if not self._retain_labels:
                self._columns = self._axis_hierarchy.level_drop(1)
            else:
                self._columns = self._axis_hierarchy
            self._index = self._axis_opposite
        self._assign_axis = False

    #---------------------------------------------------------------------------
    # name interface

    @property #type: ignore
    @doc_inject()
    def name(self) -> NameType:
        '''{}'''
        return self._bus._series._name

    def rename(self, name: NameType) -> 'Quilt':
        '''
        Return a new :obj:`Quilt` with an updated name attribute.

        Args:
            name
        '''
        return self.__class__(self._bus.rename(name),
                axis=self._axis,
                retain_labels=self._retain_labels,
                deepcopy_from_bus=self._deepcopy_from_bus,
                axis_hierarchy=self._axis_hierarchy,
                axis_opposite=self._axis_opposite,
                )

    #---------------------------------------------------------------------------

    def __repr__(self) -> str:
        '''Provide a display of the :obj:`Quilt` that does not realize the entire :obj:`Frame`.
        '''
        if self.name:
            header = f'{self.__class__.__name__}: {self.name}'
        else:
            header = self.__class__.__name__
        return f'<{header} at {hex(id(self))}>'

    def display(self,
            config: tp.Optional[DisplayConfig] = None,
            *,
            style_config: tp.Optional[StyleConfig] = None,
            ) -> Display:
        '''Provide a :obj:`Frame`-style display of the :obj:`Quilt`.
        '''
        if self._assign_axis:
            self._update_axis_labels()
        return self.to_frame().display( #type: ignore
                config,
                style_config=style_config,
                )

    #---------------------------------------------------------------------------
    # accessors

    @property #type: ignore
    @doc_inject(selector='values_2d', class_name='Quilt')
    def values(self) -> np.ndarray:
        '''
        {}
        '''
        if self._assign_axis:
            self._update_axis_labels()
        return self.to_frame().values

    @property
    def index(self) -> IndexBase:
        '''The ``IndexBase`` instance assigned for row labels.
        '''
        if self._assign_axis:
            self._update_axis_labels()
        return self._index

    @property
    def columns(self) -> IndexBase:
        '''The ``IndexBase`` instance assigned for column labels.
        '''
        if self._assign_axis:
            self._update_axis_labels()
        return self._columns

    #---------------------------------------------------------------------------

    @property
    def shape(self) -> tp.Tuple[int, int]:
        '''
        Return a tuple describing the shape of the underlying NumPy array.

        Returns:
            :obj:`tp.Tuple[int]`
        '''
        if self._assign_axis:
            self._update_axis_labels()
        return len(self._index), len(self._columns)

    @property
    def ndim(self) -> int:
        '''
        Return the number of dimensions, which for a `Frame` is always 2.

        Returns:
            :obj:`int`
        '''
        return self._NDIM

    @property
    def size(self) -> int:
        '''
        Return the size of the underlying NumPy array.

        Returns:
            :obj:`int`
        '''
        if self._assign_axis:
            self._update_axis_labels()
        return len(self._index) * len(self._columns)

    @property
    def nbytes(self) -> int:
        '''
        Return the total bytes of the underlying NumPy arrays.

        Returns:
            :obj:`int`
        '''
        # return self._blocks.nbytes
        if self._assign_axis:
            self._update_axis_labels()
        return sum(f.nbytes for _, f in self._bus.items())

    @property
    def status(self) -> Frame:
        '''
        Return a :obj:`Frame` indicating loaded status, size, bytes, and shape of all loaded :obj:`Frame` in the contained :obj:`Quilt`.
        '''
        return self._bus.status

    #---------------------------------------------------------------------------
    # dictionary-like interface

    def keys(self) -> tp.Iterable[tp.Hashable]:
        '''Iterator of column labels.
        '''
        if self._assign_axis:
            self._update_axis_labels()
        return self._columns

    def __iter__(self) -> tp.Iterable[tp.Hashable]:
        '''
        Iterator of column labels, same as :py:meth:`Frame.keys`.
        '''
        if self._assign_axis:
            self._update_axis_labels()
        return self._columns.__iter__()

    def __contains__(self, value: tp.Hashable) -> bool:
        '''
        Inclusion of value in column labels.
        '''
        if self._assign_axis:
            self._update_axis_labels()
        return self._columns.__contains__(value)

    def items(self) -> tp.Iterator[tp.Tuple[tp.Hashable, Series]]:
        '''Iterator of pairs of column label and corresponding column :obj:`Series`.
        '''
        if self._assign_axis:
            self._update_axis_labels()
        yield from self._axis_series_items(axis=0) # iterate columns

    def get(self,
            key: tp.Hashable,
            default: tp.Optional[Series] = None,
            ) -> tp.Optional[Series]:
        '''
        Return the value found at the columns key, else the default if the key is not found. This method is implemented to complete the dictionary-like interface.
        '''
        if self._assign_axis:
            self._update_axis_labels()
        if key not in self._columns:
            return default
        return self.__getitem__(key) #type: ignore

    #---------------------------------------------------------------------------
    # compatibility with StoreClientMixin

    def _items_store(self) -> tp.Iterator[tp.Tuple[tp.Hashable, Frame]]:
        '''Iterator of pairs of :obj:`Quilt` label and contained :obj:`Frame`.
        '''
        yield from self._bus.items()


    #---------------------------------------------------------------------------
    # axis iterators

    def _axis_array(self, axis: int) -> tp.Iterator[np.ndarray]:
        '''Generator of arrays across an axis

        Args:
            axis: 0 iterates over columns, 1 iterates over rows
        '''
        extractor = get_extractor(
                self._deepcopy_from_bus,
                is_array=True,
                memo_active=False,
                )

        if axis == 1: # iterate over rows
            if self._axis == 0: # bus components aligned vertically
                for _, component in self._bus.items():
                    for array in component._blocks.axis_values(axis):
                        yield extractor(array)
            else: # bus components aligned horizontally
                raise NotImplementedAxis()
        elif axis == 0: # iterate over columns
            if self._axis == 1: # bus components aligned horizontally
                for _, component in self._bus.items():
                    for array in component._blocks.axis_values(axis):
                        yield extractor(array)
            else: # bus components aligned horizontally
                raise NotImplementedAxis()
        else:
            raise AxisInvalid(f'no support for axis {axis}')

    def _axis_array_items(self, axis: int) -> tp.Iterator[tp.Tuple[tp.Hashable, np.ndarray]]:
        keys = self._index if axis == 1 else self._columns
        yield from zip(keys, self._axis_array(axis))


    def _axis_tuple(self, *,
            axis: int,
            constructor: tp.Optional[tp.Type[tp.NamedTuple]] = None,
            ) -> tp.Iterator[tp.NamedTuple]:
        '''Generator of named tuples across an axis.

        Args:
            axis: 0 iterates over columns (index axis), 1 iterates over rows (column axis)
        '''
        if constructor is None:
            if axis == 1:
                labels = self._columns.values
            elif axis == 0:
                labels = self._index.values
            else:
                raise AxisInvalid(f'no support for axis {axis}')
            # uses _make method to call with iterable
            constructor = get_tuple_constructor(labels) #type: ignore
        elif (isinstance(constructor, type) and
                issubclass(constructor, tuple) and
                hasattr(constructor, '_make')):
            constructor = constructor._make #type: ignore

        assert constructor is not None

        for axis_values in self._axis_array(axis):
            yield constructor(axis_values)

    def _axis_tuple_items(self, *,
            axis: int,
            constructor: tp.Optional[tp.Type[tp.NamedTuple]] = None,
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, tp.NamedTuple]]:
        keys = self._index if axis == 1 else self._columns
        yield from zip(keys, self._axis_tuple(axis=axis, constructor=constructor))


    def _axis_series(self, axis: int) -> tp.Iterator[Series]:
        '''Generator of Series across an axis
        '''
        index = self._index if axis == 0 else self._columns
        for label, axis_values in self._axis_array_items(axis):
            yield Series(axis_values, index=index, name=label, own_index=True)

    def _axis_series_items(self, axis: int) -> tp.Iterator[tp.Tuple[tp.Hashable, np.ndarray]]:
        keys = self._index if axis == 1 else self._columns
        yield from zip(keys, self._axis_series(axis=axis))

    #---------------------------------------------------------------------------
    def _axis_window_items(self, *,
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

    def _axis_window(self, *,
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
            ) -> tp.Iterator['Frame']:
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
    def _extract_array(self,
            row_key: GetItemKeyType = None,
            column_key: GetItemKeyType = None,
            ) -> np.ndarray:
        '''
        Extract a consolidated array based on iloc selection.
        '''
        assert self._axis_hierarchy is not None #mypy

        extractor = get_extractor(
                self._deepcopy_from_bus,
                is_array=True,
                memo_active=False,
                )

        row_key = NULL_SLICE if row_key is None else row_key
        column_key = NULL_SLICE if column_key is None else column_key

        if row_key == NULL_SLICE and column_key == NULL_SLICE:
            if len(self._bus) == 1:
                return extractor(self._bus.iloc[0].values)

            # NOTE: do not need to call extractor when concatenate is called, as a new array is always allocated.
            arrays = [f.values for _, f in self._bus.items()]
            return concat_resolved(
                    arrays,
                    axis=self._axis,
                    )

        parts: tp.List[np.ndarray] = []
        bus_keys: tp.Iterable[tp.Hashable]

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
        if isinstance(axis_map_sub, tuple): # type: ignore
            bus_keys = (axis_map_sub[0],) #type: ignore
        else:
            bus_keys = axis_map_sub._levels.index

        for key_count, key in enumerate(bus_keys):
            sel_component = sel[self._axis_hierarchy._loc_to_iloc(HLoc[key])]

            if self._axis == 0:
                component = self._bus.loc[key]._extract_array(sel_component, opposite_key) #type: ignore [attr-defined]
                if sel_reduces:
                    component = component[0]
            else:
                component = self._bus.loc[key]._extract_array(opposite_key, sel_component) #type: ignore [attr-defined]
                if sel_reduces:
                    if component.ndim == 1:
                        component = component[0]
                    elif component.ndim == 2:
                        component = component[NULL_SLICE, 0]

            parts.append(component)

        if len(parts) == 1:
            return extractor(parts.pop())

        # NOTE: concatenate always allocates a new array, thus no need for extractor above
        if sel_reduces or opposite_reduces:
            # NOTE: not sure if concat_resolved is needed here
            return concat_resolved(parts)
        return concat_resolved(parts, axis=self._axis)

    def _extract(self,
            row_key: GetItemKeyType = None,
            column_key: GetItemKeyType = None,
            ) -> tp.Union[Frame, Series]:
        '''
        Extract Container based on iloc selection.
        '''
        assert self._axis_hierarchy is not None #mypy

        extractor = get_extractor(
                self._deepcopy_from_bus,
                is_array=False,
                memo_active=False,
                )

        row_key = NULL_SLICE if row_key is None else row_key
        row_key_is_array = isinstance(row_key, np.ndarray)
        column_key = NULL_SLICE if column_key is None else column_key
        column_key_is_array = isinstance(column_key, np.ndarray)

        if (not row_key_is_array and row_key == NULL_SLICE
                and not column_key_is_array and column_key == NULL_SLICE):
            if self._retain_labels and self._axis == 0:
                frames = (extractor(f.relabel_level_add(index=k))
                        for k, f in self._bus.items())
            elif self._retain_labels and self._axis == 1:
                frames = (extractor(f.relabel_level_add(columns=k))
                        for k, f in self._bus.items())
            else:
                frames = (extractor(f) for _, f in self._bus.items())

            return Frame.from_concat( #type: ignore
                    frames,
                    axis=self._axis,
                    )

        parts: tp.List[tp.Any] = []
        bus_keys: tp.Iterable[tp.Hashable]

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
        if isinstance(axis_map_sub, tuple): #type: ignore
            bus_keys = (axis_map_sub[0],) #type: ignore
        else:
            bus_keys = axis_map_sub._levels.index

        for key_count, key in enumerate(bus_keys):
            sel_component = sel[self._axis_hierarchy._loc_to_iloc(HLoc[key])]

            if self._axis == 0:
                component = self._bus.loc[key].iloc[sel_component, opposite_key]
                if key_count == 0:
                    component_is_series = isinstance(component, Series)
                if self._retain_labels:
                    # component might be a Series, can call the same with first arg
                    component = component.relabel_level_add(key)
                if sel_reduces: # make Frame into a Series, Series into an element
                    component = component.iloc[0]
            else:
                component = self._bus.loc[key].iloc[opposite_key, sel_component]
                if key_count == 0:
                    component_is_series = isinstance(component, Series)
                if self._retain_labels:
                    if component_is_series:
                        component = component.relabel_level_add(key)
                    else:
                        component = component.relabel_level_add(columns=key)
                if sel_reduces: # make Frame into a Series, Series into an element
                    if component_is_series:
                        component = component.iloc[0]
                    else:
                        component = component.iloc[NULL_SLICE, 0]

            parts.append(extractor(component))

        if len(parts) == 1:
            return parts.pop() #type: ignore

        # NOTE: Series/Frame from_concate will attempt to re-use ndarrays, and thus using extractor above is appropriate
        if component_is_series:
            return Series.from_concat(parts)
        return Frame.from_concat(parts, axis=self._axis) #type: ignore

    #---------------------------------------------------------------------------

    def _extract_iloc(self, key: GetItemKeyTypeCompound) -> tp.Union[Series, Frame]:
        '''
        Give a compound key, return a new Frame. This method simply handles the variabiliyt of single or compound selectors.
        '''
        if self._assign_axis:
            self._update_axis_labels()
        if isinstance(key, tuple):
            return self._extract(*key)
        return self._extract(row_key=key)

    def _compound_loc_to_iloc(self,
            key: GetItemKeyTypeCompound) -> tp.Tuple[GetItemKeyType, GetItemKeyType]:
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

    def _extract_loc(self, key: GetItemKeyTypeCompound) -> tp.Union[Series, Frame]:
        if self._assign_axis:
            self._update_axis_labels()
        return self._extract(*self._compound_loc_to_iloc(key))

    def _compound_loc_to_getitem_iloc(self,
            key: GetItemKeyTypeCompound) -> tp.Tuple[GetItemKeyType, GetItemKeyType]:
        '''Handle a potentially compound key in the style of __getitem__. This will raise an appropriate exception if a two argument loc-style call is attempted.
        '''
        iloc_column_key = self._columns._loc_to_iloc(key)
        return None, iloc_column_key

    @doc_inject(selector='selector')
    def __getitem__(self, key: GetItemKeyType) -> tp.Union[Frame, Series]:
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
    def loc(self) -> InterfaceGetItem['Frame']:
        return InterfaceGetItem(self._extract_loc) #type: ignore

    @property
    def iloc(self) -> InterfaceGetItem['Frame']:
        return InterfaceGetItem(self._extract_iloc) #type: ignore

    #---------------------------------------------------------------------------
    # iterators

    @property
    def iter_array(self) -> IterNodeAxis['Quilt']:
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
    def iter_array_items(self) -> IterNodeAxis['Quilt']:
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
    def iter_tuple(self) -> IterNodeConstructorAxis['Quilt']:
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
    def iter_tuple_items(self) -> IterNodeConstructorAxis['Quilt']:
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
    def iter_series(self) -> IterNodeAxis['Quilt']:
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
    def iter_series_items(self) -> IterNodeAxis['Quilt']:
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

    @property #type: ignore
    @doc_inject(selector='window')
    def iter_window(self) -> IterNodeWindow['Quilt']:
        '''
        Iterator of windowed values, where values are given as a :obj:`Frame`.

        {args}
        '''
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

    @property #type: ignore
    @doc_inject(selector='window')
    def iter_window_items(self) -> IterNodeWindow['Quilt']:
        '''
        Iterator of pairs of label, windowed values, where values are given as a :obj:`Frame`.

        {args}
        '''
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

    @property #type: ignore
    @doc_inject(selector='window')
    def iter_window_array(self) -> IterNodeWindow['Quilt']:
        '''
        Iterator of windowed values, where values are given as a :obj:`np.array`.

        {args}
        '''
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

    @property #type: ignore
    @doc_inject(selector='window')
    def iter_window_array_items(self) -> IterNodeWindow['Quilt']:
        '''
        Iterator of pairs of label, windowed values, where values are given as a :obj:`np.array`.

        {args}
        '''
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

    #---------------------------------------------------------------------------
    # transformations resulting in changed dimensionality
    @doc_inject(selector='head', class_name='Quilt')
    def head(self, count: int = 5) -> 'Frame':
        '''{doc}

        Args:
            {count}
        '''
        return self.iloc[:count]

    @doc_inject(selector='tail', class_name='Quilt')
    def tail(self, count: int = 5) -> 'Frame':
        '''{doc}

        Args:
            {count}
        '''
        return self.iloc[-count:]


    #---------------------------------------------------------------------------
    def to_frame(self) -> Frame:
        '''
        Return a consolidated :obj:`Frame`.
        '''
        if self._assign_axis:
            self._update_axis_labels()
        return self._extract(NULL_SLICE, NULL_SLICE) #type: ignore
