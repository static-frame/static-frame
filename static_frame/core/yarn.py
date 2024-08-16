from __future__ import annotations

from collections.abc import Set
from functools import partial
from itertools import chain

import numpy as np
import typing_extensions as tp

from static_frame.core.axis_map import buses_to_iloc_hierarchy
from static_frame.core.axis_map import buses_to_loc_hierarchy
from static_frame.core.bus import FrameDeferred
from static_frame.core.container import ContainerBase
from static_frame.core.container_util import index_from_optional_constructor
from static_frame.core.container_util import index_many_concat
from static_frame.core.container_util import iter_component_signature_bytes
from static_frame.core.container_util import rehierarch_from_index_hierarchy
from static_frame.core.container_util import sort_index_for_order
from static_frame.core.display import Display
from static_frame.core.display import DisplayActive
from static_frame.core.display import DisplayHeader
from static_frame.core.display_config import DisplayConfig
from static_frame.core.doc_str import doc_inject
from static_frame.core.exception import ErrorInitYarn
from static_frame.core.exception import RelabelInvalid
from static_frame.core.frame import Frame
from static_frame.core.generic_aliases import TBusAny
from static_frame.core.generic_aliases import TFrameAny
from static_frame.core.generic_aliases import TIndexAny
from static_frame.core.generic_aliases import TIndexIntDefault
from static_frame.core.generic_aliases import TSeriesAny
from static_frame.core.generic_aliases import TSeriesObject
from static_frame.core.index import Index
from static_frame.core.index_auto import IndexAutoConstructorFactory
from static_frame.core.index_auto import IndexAutoFactory
from static_frame.core.index_auto import TIndexAutoFactory
from static_frame.core.index_auto import TRelabelInput
from static_frame.core.index_base import IndexBase
from static_frame.core.index_correspondence import IndexCorrespondence
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.node_iter import IterNodeApplyType
from static_frame.core.node_iter import IterNodeNoArg
from static_frame.core.node_selector import InterfaceSelectTrio
from static_frame.core.node_selector import InterGetItemILocReduces
from static_frame.core.node_selector import InterGetItemLocReduces
from static_frame.core.series import Series
from static_frame.core.store_client_mixin import StoreClientMixin
from static_frame.core.style_config import StyleConfig
from static_frame.core.util import BOOL_TYPES
from static_frame.core.util import DEFAULT_SORT_KIND
from static_frame.core.util import DTYPE_INT_DEFAULT
from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import EMPTY_SLICE
from static_frame.core.util import INT_TYPES
from static_frame.core.util import NAME_DEFAULT
from static_frame.core.util import IterNodeType
from static_frame.core.util import PositionsAllocator
from static_frame.core.util import TBoolOrBools
from static_frame.core.util import TDtypeObject
from static_frame.core.util import TILocSelector
from static_frame.core.util import TILocSelectorMany
from static_frame.core.util import TILocSelectorOne
from static_frame.core.util import TIndexCtorSpecifier
from static_frame.core.util import TIndexCtorSpecifiers
from static_frame.core.util import TIndexInitializer
from static_frame.core.util import TLabel
from static_frame.core.util import TLocSelector
from static_frame.core.util import TName
from static_frame.core.util import TNDArrayAny
from static_frame.core.util import TNDArrayIntDefault
from static_frame.core.util import TNDArrayObject
from static_frame.core.util import TSortKinds
from static_frame.core.util import array_shift
from static_frame.core.util import is_callable_or_mapping
from static_frame.core.util import iterable_to_array_1d

#-------------------------------------------------------------------------------
TIHInternal = IndexHierarchy[TIndexIntDefault, TIndexAny]

TVIndex = tp.TypeVar('TVIndex', bound=IndexBase, default=tp.Any) # pylint: disable=E1123

class Yarn(ContainerBase, StoreClientMixin, tp.Generic[TVIndex]):
    '''
    A :obj:`Series`-like container made of an ordered collection of :obj:`Bus`. :obj:`Yarn` can be indexed independently of the contained :obj:`Bus`, permitting independent labels per contained :obj:`Frame`.
    '''

    __slots__ = (
            '_values',
            '_hierarchy',
            '_index',
            '_indexer',
            '_name',
            '_deepcopy_from_bus',
            )

    _values: TNDArrayObject
    _hierarchy: TIHInternal
    _index: IndexBase
    _indexer: TNDArrayIntDefault
    _name: TName
    _deepcopy_from_bus: bool

    _NDIM: int = 1

    @classmethod
    def from_buses(cls,
            buses: tp.Iterable[TBusAny],
            *,
            name: TName = None,
            retain_labels: bool,
            deepcopy_from_bus: bool = False,
            ) -> tp.Self:
        '''Return a :obj:`Yarn` from an iterable of :obj:`Bus`; labels will be drawn from :obj:`Bus.name`.
        '''
        values, _ = iterable_to_array_1d(buses, dtype=DTYPE_OBJECT)

        hierarchy = buses_to_iloc_hierarchy(
                values,
                deepcopy_from_bus=deepcopy_from_bus,
                init_exception_cls=ErrorInitYarn,
                )

        if retain_labels:
            index = buses_to_loc_hierarchy(
                    values,
                    deepcopy_from_bus=deepcopy_from_bus,
                    init_exception_cls=ErrorInitYarn,
                    )
        else:
            index = hierarchy.level_drop(1) #type: ignore

        return cls(values,
                hierarchy=hierarchy,
                index=index,
                name=name,
                deepcopy_from_bus=deepcopy_from_bus,
                )

    @classmethod
    def from_concat(cls,
            containers: tp.Iterable[TYarnAny],
            *,
            index: tp.Optional[tp.Union[TIndexInitializer, TIndexAutoFactory]] = None,
            name: TName = NAME_DEFAULT,
            deepcopy_from_bus: bool = False,
            ) -> tp.Self:
        '''
        Concatenate multiple :obj:`Yarn` into a new :obj:`Yarn`. Loaded status of :obj:`Frame` within each :obj:`Bus` will not be altered.

        Args:
            containers:
            index: Optionally provide new labels for the result of the concatenation.
            name:
            deepcopy_from_bus:
        '''
        values_components: tp.List[TNDArrayObject] = []
        indexer_components: tp.List[TNDArrayIntDefault] = []
        index_components: tp.Optional[tp.List[IndexBase]] = None if index is not None else []
        labels = [] # for new hierarchy

        bus_count = 0
        hierarchy_count = 0

        for y in containers:
            if not isinstance(y, Yarn):
                raise NotImplementedError(f'Cannot concatenate from {type(y)}')

            b_pos: int
            for b_pos, frame_label in y._hierarchy: # type: ignore[assignment]
                labels.append((b_pos + bus_count, frame_label))

            values_components.append(y._values)
            indexer_components.append(y._indexer + hierarchy_count)

            bus_count += len(y._values)
            hierarchy_count += len(y._hierarchy)

            if index_components is not None: # only accumulate if index not provided
                index_components.append(y.index)

        values = np.concatenate(values_components, dtype=DTYPE_OBJECT) # pylint: disable=E1123
        indexer = np.concatenate(indexer_components, dtype=DTYPE_INT_DEFAULT) # pylint: disable=E1123

        ctor: tp.Callable[..., IndexBase] = partial(Index, dtype=DTYPE_INT_DEFAULT)
        ctors: TIndexCtorSpecifiers = [ctor, IndexAutoConstructorFactory] # type: ignore[list-item]
        hierarchy: TIHInternal = IndexHierarchy.from_labels(labels,
                index_constructors=ctors,
                )

        if index_components is not None:
            index = index_many_concat(index_components, Index)
            own_index = True
        else: # provided index must be evaluated
            own_index = False

        return cls(values,
                index=index,
                deepcopy_from_bus=deepcopy_from_bus,
                indexer=indexer,
                hierarchy=hierarchy,
                name=name if name is not NAME_DEFAULT else None,
                own_index=own_index,
                )

    #---------------------------------------------------------------------------
    def __init__(self,
            series: tp.Union[TSeriesObject, tp.Iterable[TBusAny]], # rename: values
            *,
            index: TIndexInitializer | TIndexAutoFactory | None = None,
            index_constructor: tp.Optional[TIndexCtorSpecifier] = None,
            deepcopy_from_bus: bool = False,
            indexer: tp.Optional[TNDArrayIntDefault] = None,
            hierarchy: tp.Optional[TIHInternal] = None,
            name: TName = None,
            own_index: bool = False,
            ) -> None:
        '''
        Args:
            series: An iterable (or :obj:`Series`) of :obj:`Bus`. The length of this container may not be the same as ``index``, if provided.
            index: Optionally provide an index for the :obj:`Frame` contained in all :obj:`Bus`.
            index_constructor:
            deepcopy_from_bus:
            hierarchy: Optionally provide a depth-two `IndexHierarchy` constructed from `Bus` integer positions on the outer level, and contained `Frame` labels on the inner level.
            indexer: For each `Frame` referenced by the index, provide the location within the internal `IndexHierarchy`.
            name:
            own_index:
        '''

        if isinstance(series, Series):
            if series.dtype != DTYPE_OBJECT:
                raise ErrorInitYarn(
                        f'Series passed to initializer must have dtype object, not {series.dtype}')
            self._values = series.values
        else:
            try:
                self._values, _ = iterable_to_array_1d(series, dtype=DTYPE_OBJECT)
            except RuntimeError as e:
                raise ErrorInitYarn(e) from None

        self._name = name
        self._deepcopy_from_bus = deepcopy_from_bus

        if hierarchy is None:
            self._hierarchy = buses_to_iloc_hierarchy(
                    self._values,
                    deepcopy_from_bus=self._deepcopy_from_bus,
                    init_exception_cls=ErrorInitYarn,
                    )
        else: # NOTE: we assume this hierarchy is well-formed
            self._hierarchy = hierarchy

        self._index: IndexBase
        if own_index:
            self._index = index #type: ignore
        elif index is None or index is IndexAutoFactory:
            self._index = IndexAutoFactory.from_optional_constructor(
                    len(self._hierarchy),
                    default_constructor=Index,
                    explicit_constructor=index_constructor
                    )
        else: # an iterable of labels or an Index
            self._index = index_from_optional_constructor(index,
                    default_constructor=Index,
                    explicit_constructor=index_constructor
                    )

        if len(self._index) > len(self._hierarchy): # pyright: ignore
            raise ErrorInitYarn(f'Length of supplied index ({len(self._index)}) not of sufficient size ({len(self._hierarchy)}).') # pyright: ignore

        self._indexer: TNDArrayIntDefault
        if indexer is None:
            self._indexer = PositionsAllocator.get(len(self._index)) # pyright: ignore
        else:
            self._indexer = indexer
            if len(self._indexer) != len(self._index): # pyright: ignore
                raise ErrorInitYarn(f'Length of supplied indexer ({len(self._indexer)}) not of sufficient size ({len(self._index)}).') # pyright: ignore


    #---------------------------------------------------------------------------
    # deferred loading of axis info

    def unpersist(self) -> None:
        '''For the :obj:`Bus` contained in this object, replace all loaded :obj:`Frame` with :obj:`FrameDeferred`.
        '''
        for b in self._values:
            if b is not None:
                b.unpersist()

    #---------------------------------------------------------------------------
    def __reversed__(self) -> tp.Iterator[TLabel]:
        '''
        Returns a reverse iterator on the :obj:`Yarn` index.

        Returns:
            :obj:`Index`
        '''
        return reversed(self._index)

    #---------------------------------------------------------------------------
    # name interface

    @property
    @doc_inject()
    def name(self) -> TName:
        '''{}'''
        return self._name

    def rename(self, name: TName) -> tp.Self:
        '''
        Return a new :obj:`Yarn` with an updated name attribute.

        Args:
            name
        '''
        # NOTE: do not need to call _update_index_labels; can continue to defer
        return self.__class__(self._values,
                index=self._index,
                hierarchy=self._hierarchy,
                indexer=self._indexer,
                name=name,
                deepcopy_from_bus=self._deepcopy_from_bus,
                own_index=True,
                )

    #---------------------------------------------------------------------------
    # interfaces

    @property
    def loc(self) -> InterGetItemLocReduces[TYarnAny, np.object_]:
        return InterGetItemLocReduces(self._extract_loc) # type: ignore

    @property
    def iloc(self) -> InterGetItemILocReduces[TYarnAny, np.object_]:
        return InterGetItemILocReduces(self._extract_iloc)

    @property
    def drop(self) -> InterfaceSelectTrio[TYarnAny]:
        '''
        Interface for dropping elements from :obj:`Yarn`.
        '''
        return InterfaceSelectTrio( #type: ignore
                func_iloc=self._drop_iloc,
                func_loc=self._drop_loc,
                func_getitem=self._drop_loc
                )

    #---------------------------------------------------------------------------
    @property
    def iter_element(self) -> IterNodeNoArg[TYarnAny]:
        '''
        Iterator of elements.
        '''
        return IterNodeNoArg(
                container=self,
                function_items=self._axis_element_items,
                function_values=self._axis_element,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.SERIES_VALUES,
                )

    @property
    def iter_element_items(self) -> IterNodeNoArg[TYarnAny]:
        '''
        Iterator of label, element pairs.
        '''
        return IterNodeNoArg(
                container=self,
                function_items=self._axis_element_items,
                function_values=self._axis_element,
                yield_type=IterNodeType.ITEMS,
                apply_type=IterNodeApplyType.SERIES_VALUES,
                )

    #---------------------------------------------------------------------------
    # extraction

    @tp.overload
    def _extract_iloc(self, key: TILocSelectorMany) -> tp.Self: ...

    @tp.overload
    def _extract_iloc(self, key: TILocSelectorOne) -> TFrameAny: ...

    def _extract_iloc(self, key: TILocSelector) -> tp.Self | TFrameAny:
        '''
        Returns:
            Yarn or, if an element is selected, a Frame
        '''
        indexer: tp.Union[TNDArrayIntDefault, int] = self._indexer[key]

        sel_hierarchy = self._hierarchy._extract_iloc(indexer)

        if isinstance(indexer, INT_TYPES):
            # got a single element, return a Frame
            b_pos, frame_label = sel_hierarchy # always two-item tuple
            f: Frame = self._values[b_pos]._extract_loc(frame_label) # pyright: ignore
            return f

        # NOTE: identify Bus that are no longer needed, and remove them from the values such that they can be GCd if necessary; for now, we leave the hierarchy (and the position numbers) unchanged
        bus_pos = self._hierarchy.index_at_depth(depth_level=0)
        sel_bus_pos = sel_hierarchy.index_at_depth(depth_level=0)
        if len(sel_bus_pos) < len(bus_pos):
            values = self._values.copy() # becomes mutable
            for pos in bus_pos.difference(sel_bus_pos):
                values[pos] = None
            values.flags.writeable = False
        else:
            values = self._values

        return self.__class__(values,
                index=self._index.iloc[key],
                deepcopy_from_bus=self._deepcopy_from_bus,
                hierarchy=self._hierarchy,
                indexer=indexer,
                name=self._name,
                own_index=True,
                )

    def _extract_loc(self, key: TLocSelector) -> TYarnAny | TFrameAny:
        # use the index active for this Yarn
        key_iloc = self._index._loc_to_iloc(key)
        return self._extract_iloc(key_iloc)

    @doc_inject(selector='selector')
    def __getitem__(self, key: TLocSelector) -> TYarnAny | TFrameAny:
        '''Selector of values by label.

        Args:
            key: {key_loc}
        '''
        return self._extract_loc(key)

    #---------------------------------------------------------------------------
    # utilities for alternate extraction: drop

    def _drop_iloc(self, key: TILocSelector) -> tp.Self:
        invalid = np.full(len(self._index), True)
        invalid[key] = False
        return self._extract_iloc(invalid)

    def _drop_loc(self, key: TLocSelector) -> tp.Self:
        return self._drop_iloc(self._index._loc_to_iloc(key))

    #---------------------------------------------------------------------------
    # axis functions

    def _axis_element_items(self,
            ) -> tp.Iterator[tp.Tuple[TLabel, tp.Any]]:
        '''Generator of index, value pairs, equivalent to Series.items(). Repeated to have a common signature as other axis functions.
        '''
        yield from self.items()

    def _axis_element(self,
            ) -> tp.Iterator[TFrameAny]:
        for b_pos, frame_label in self._hierarchy._extract_iloc(self._indexer):
            yield self._values[b_pos]._extract_loc(frame_label) # pyright: ignore

    #---------------------------------------------------------------------------
    # index manipulation

    @doc_inject(selector='reindex', class_name='Bus')
    def reindex(self,
            index: TIndexInitializer,
            *,
            fill_value: tp.Any = None,
            own_index: bool = False,
            check_equals: bool = True
            ) -> tp.Self:
        '''
        {doc}

        Args:
            index: {index_initializer}
            columns: {index_initializer}
            {fill_value}
            {own_index}
        '''
        index_owned: IndexBase
        if own_index:
            index_owned = index # type: ignore
        else:
            index_owned = index_from_optional_constructor(index,
                    default_constructor=Index)

        if check_equals and self._index.equals(index_owned):
            # if labels are equal (even if a different Index subclass), we can simply use the new Index
            return self.__class__(self._values,
                    index=index_owned,
                    hierarchy=self._hierarchy,
                    indexer=self._indexer,
                    name=self._name,
                    deepcopy_from_bus=self._deepcopy_from_bus,
                    own_index=True,
                    )

        ic = IndexCorrespondence.from_correspondence(self._index, index_owned)
        if not ic.size:
            return self._extract_iloc(EMPTY_SLICE)

        if ic.is_subset: # must have some common
            indexer = self._indexer[ic.iloc_src]
            indexer.flags.writeable = False

            return self.__class__(self._values,
                    index=index_owned,
                    hierarchy=self._hierarchy,
                    indexer=indexer,
                    name=self._name,
                    deepcopy_from_bus=self._deepcopy_from_bus,
                    own_index=True,
                    )

        raise NotImplementedError('Reindex operations that are not strict subsets are not supported by `Yarn`')

    @doc_inject(selector='relabel', class_name='Yarn')
    def relabel(self,
            index: tp.Optional[TRelabelInput]
            ) -> tp.Self:
        '''
        {doc}

        Args:
            index: {relabel_input_index}
        '''
        #NOTE: we name the parameter index for alignment with the corresponding Frame method
        own_index = False
        if index is IndexAutoFactory:
            index_init = None
        elif index is None:
            index_init = self._index
        elif is_callable_or_mapping(index):
            index_init = self._index.relabel(index)
            own_index = True
        elif isinstance(index, Set):
            raise RelabelInvalid()
        else:
            index_init = index #type: ignore

        return self.__class__(self._values, # no change to Buses
                index=index_init, # pyright: ignore
                deepcopy_from_bus=self._deepcopy_from_bus,
                hierarchy=self._hierarchy, # no change
                indexer=self._indexer,
                own_index=own_index,
                )

    @doc_inject(selector='relabel_flat', class_name='Yarn')
    def relabel_flat(self) -> tp.Self:
        '''
        {doc}
        '''
        if not isinstance(self._index, IndexHierarchy):
            raise RuntimeError('cannot flatten an Index that is not an IndexHierarchy')

        return self.__class__(self._values, # no change to Buses
                index=self._index.flat(),
                deepcopy_from_bus=self._deepcopy_from_bus,
                hierarchy=self._hierarchy, # no change
                indexer=self._indexer,
                own_index=True,
                )

    @doc_inject(selector='relabel_level_add', class_name='Yarn')
    def relabel_level_add(self,
            level: TLabel
            ) -> tp.Self:
        '''
        {doc}

        Args:
            level: {level}
        '''
        return self.__class__(self._values, # no change to Buses
                index=self._index.level_add(level),
                deepcopy_from_bus=self._deepcopy_from_bus,
                hierarchy=self._hierarchy, # no change
                indexer=self._indexer,
                own_index=True,
                )

    @doc_inject(selector='relabel_level_drop', class_name='Yarn')
    def relabel_level_drop(self,
            count: int = 1
            ) -> tp.Self:
        '''
        {doc}

        Args:
            count: {count}
        '''
        if not isinstance(self._index, IndexHierarchy):
            raise RuntimeError('cannot drop level of an Index that is not an IndexHierarchy')

        return self.__class__(self._values, # no change to Buses
                index=self._index.level_drop(count),
                deepcopy_from_bus=self._deepcopy_from_bus,
                hierarchy=self._hierarchy, # no change
                indexer=self._indexer,
                own_index=True,
                )

    def rehierarch(self,
            depth_map: tp.Sequence[int],
            *,
            index_constructors: TIndexCtorSpecifiers = None,
            ) -> tp.Self:
        '''
        Return a new :obj:`Series` with new a hierarchy based on the supplied ``depth_map``.
        '''
        if self.index.depth == 1:
            raise RuntimeError('cannot rehierarch when there is no hierarchy')

        index, iloc_map = rehierarch_from_index_hierarchy(
                labels=self._index, #type: ignore
                depth_map=depth_map,
                index_constructors=index_constructors,
                name=self._index.name,
                )

        return self._extract_iloc(iloc_map).relabel(index)

    #---------------------------------------------------------------------------

    def items(self) -> tp.Iterator[tp.Tuple[TLabel, TFrameAny]]:
        '''Iterator of pairs of :obj:`Yarn` label and contained :obj:`Frame`.
        '''
        labels = iter(self._index)
        for b_pos, frame_label in self._hierarchy._extract_iloc(self._indexer):
            # NOTE: missing optimization to read multiple Frame from Bus in one extraction
            yield next(labels), self._values[b_pos]._extract_loc(frame_label) # pyright: ignore

    _items_store = items

    @property
    def values(self) -> TNDArrayObject:
        '''A 1D object array of all :obj:`Frame` contained in all contained :obj:`Bus`.
        '''
        array = np.empty(shape=len(self._index), dtype=DTYPE_OBJECT)

        for i, (b_pos, frame_label) in enumerate(
                self._hierarchy._extract_iloc(self._indexer)):
            array[i] = self._values[b_pos]._extract_loc(frame_label) # pyright: ignore

        array.flags.writeable = False
        return array

    #---------------------------------------------------------------------------

    def __len__(self) -> int:
        '''Length of values.
        '''
        return self._index.__len__()

    @doc_inject()
    def display(self,
            config: tp.Optional[DisplayConfig] = None,
            *,
            style_config: tp.Optional[StyleConfig] = None,
            ) -> Display:
        '''{doc}

        Args:
            {config}
        '''
        # NOTE: the key change over serires is providing the Bus as the displayed class
        config = config or DisplayActive.get()
        display_cls = Display.from_values((),
                header=DisplayHeader(self.__class__, self._name),
                config=config)

        array = np.empty(shape=len(self._index), dtype=DTYPE_OBJECT)

        for i, (b_pos, frame_label) in enumerate(
                self._hierarchy._extract_iloc(self._indexer)):
            b = self._values[b_pos]
            # NOTE: do not load FrameDeferred
            array[i] = b._values_mutable[b.index.loc_to_iloc(frame_label)] # pyright: ignore

        array.flags.writeable = False

        # create temporary series just for display
        series: TSeriesObject = Series(array, index=self._index, own_index=True)
        return series._display(config,
                display_cls=display_cls,
                style_config=style_config,
                )

    #---------------------------------------------------------------------------
    # extended discriptors; in general, these do not force loading Frame

    @property
    def mloc(self) -> TSeriesObject:
        '''Returns a :obj:`Series` showing a tuple of memory locations within each loaded Frame.
        '''
        mlocs = [(b.mloc if b is not None else None) for b in self._values]
        array = np.empty(shape=len(self._index), dtype=DTYPE_OBJECT)

        for i, (b_pos, frame_label) in enumerate(
                self._hierarchy._extract_iloc(self._indexer)):
            array[i] = mlocs[b_pos]._extract_loc(frame_label)

        array.flags.writeable = False
        return Series(array, index=self._index, own_index=True, name='mloc')

    @property
    def dtypes(self) -> TFrameAny:
        '''Returns a Frame of dtypes for all loaded Frames.
        '''
        deferred_dtypes = Series((None,))

        def gen() -> tp.Iterator[TSeriesObject]:
            for b_pos, frame_label in self._hierarchy._extract_iloc(self._indexer):
                b = self._values[b_pos]
                f = b._values_mutable[b.index.loc_to_iloc(frame_label)] # pyright: ignore
                if f is FrameDeferred:
                    yield deferred_dtypes
                else:
                    yield f.dtypes

        return Frame.from_concat(gen(), index=self._index, fill_value=None)

    @property
    def shapes(self) -> TSeriesObject:
        '''A :obj:`Series` describing the shape of each loaded :obj:`Frame`. Unloaded :obj:`Frame` will have a shape of None.

        Returns:
            :obj:`tp.Series`
        '''
        # collect shape Series
        shapes = [(b.shapes if b is not None else None) for b in self._values]
        array = np.empty(shape=len(self._index), dtype=DTYPE_OBJECT)

        for i, (b_pos, frame_label) in enumerate(
                self._hierarchy._extract_iloc(self._indexer)):
            array[i] = shapes[b_pos][frame_label] # pyright: ignore

        array.flags.writeable = False
        return Series(array, index=self._index, own_index=True, name='shape')

    @property
    def nbytes(self) -> int:
        '''Total bytes of data currently loaded in :obj:`Frame` contained in this :obj:`Yarn`.
        '''
        post = 0
        for b_pos, frame_label in self._hierarchy._extract_iloc(self._indexer):
            b = self._values[b_pos]
            f = b._values_mutable[b.index.loc_to_iloc(frame_label)] # pyright: ignore
            if f is not FrameDeferred:
                post += f.nbytes

        return post

    @property
    def status(self) -> TFrameAny:
        '''
        Return a :obj:`Frame` indicating loaded status, size, bytes, and shape of all loaded :obj:`Frame` in :obj:`Bus` contined in this :obj:`Yarn`.
        '''
        # collect status Frame
        status = [(b.status if b is not None else None) for b in self._values]

        def gen() -> tp.Iterator[TNDArrayObject]:
            for b_pos, frame_label in self._hierarchy._extract_iloc(self._indexer):
                f = status[b_pos]
                yield f._extract_array(f.index.loc_to_iloc(frame_label))

        return Frame.from_records(gen(),
                index=self._index,
                columns=('loaded', 'size', 'nbytes', 'shape'))

    #---------------------------------------------------------------------------
    # common attributes from the numpy array

    @property
    def dtype(self) -> TDtypeObject:
        '''
        Return the dtype of the realized NumPy array.

        Returns:
            :obj:`numpy.dtype`
        '''
        return DTYPE_OBJECT # always dtype object

    @property
    def shape(self) -> tp.Tuple[int]:
        '''
        Return a tuple describing the shape of the realized NumPy array.

        Returns:
            :obj:`Tuple[int]`
        '''
        return (self._index.__len__(),)

    @property
    def ndim(self) -> int:
        '''
        Return the number of dimensions, which for a :obj:`Yarn` is always 1.

        Returns:
            :obj:`int`
        '''
        return self._NDIM

    @property
    def size(self) -> int:
        '''
        Return the size.

        Returns:
            :obj:`int`
        '''
        return self._index.__len__()

    #---------------------------------------------------------------------------

    @property
    def index(self) -> IndexBase:
        '''
        The index instance assigned to this container.

        Returns:
            :obj:`Index`
        '''
        return self._index

    #---------------------------------------------------------------------------
    # dictionary-like interface

    def keys(self) -> IndexBase:
        '''
        Iterator of index labels.

        Returns:
            :obj:`Iterator[Hashable]`
        '''
        return self._index

    def __iter__(self) -> tp.Iterator[TLabel]:
        '''
        Iterator of index labels, same as :obj:`static_frame.Series.keys`.

        Returns:
            :obj:`Iterator[Hashasble]`
        '''
        return self._index.__iter__()

    def __contains__(self, value: TLabel) -> bool:
        '''
        Inclusion of value in index labels.

        Returns:
            :obj:`bool`
        '''
        return self._index.__contains__(value)

    def get(self, key: TLabel,
            default: tp.Any = None,
            ) -> tp.Any:
        '''
        Return the value found at the index key, else the default if the key is not found.

        Returns:
            :obj:`Any`
        '''
        if key not in self._index:
            return default
        return self.__getitem__(key)

    #---------------------------------------------------------------------------
    @doc_inject()
    def equals(self,
            other: tp.Any,
            *,
            compare_name: bool = False,
            compare_dtype: bool = False,
            compare_class: bool = False,
            skipna: bool = True,
            ) -> bool:
        '''
        {doc}

        Note: this will attempt to load and compare all Frame managed by the Bus.

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
        elif not isinstance(other, Yarn):
            return False

        if compare_name and self._name != other._name:
            return False

        # length of series in Yarn might be different but may still have the same frames, so look at realized length
        if len(self) != len(other):
            return False

        if not self._index.equals(
                other.index, # call property to force index creation
                compare_name=compare_name,
                compare_dtype=compare_dtype,
                compare_class=compare_class,
                skipna=skipna,
                ):
            return False

        # can zip because length of Series already match
        # using .values will force loading all Frame into memory; better to use items() to permit collection
        for (_, frame_self), (_, frame_other) in zip(self.items(), other.items()):
            if not frame_self.equals(frame_other,
                    compare_name=compare_name,
                    compare_dtype=compare_dtype,
                    compare_class=compare_class,
                    skipna=skipna,
                    ):
                return False

        return True

    #---------------------------------------------------------------------------
    # transformations resulting in changed dimensionality

    @doc_inject(selector='head', class_name='Yarn')
    def head(self, count: int = 5) -> TYarnAny:
        '''{doc}

        Args:
            {count}

        Returns:
            :obj:`Yarn`
        '''
        return self.iloc[:count]

    @doc_inject(selector='tail', class_name='Yarn')
    def tail(self, count: int = 5) -> TYarnAny:
        '''{doc}s

        Args:
            {count}

        Returns:
            :obj:`Yarn`
        '''
        return self.iloc[-count:]

    #---------------------------------------------------------------------------
    # transformations resulting in the same dimensionality

    @doc_inject(selector='sort')
    def sort_index(self,
            *,
            ascending: TBoolOrBools = True,
            kind: TSortKinds = DEFAULT_SORT_KIND,
            key: tp.Optional[tp.Callable[[IndexBase], tp.Union[TNDArrayAny, IndexBase]]] = None,
            ) -> tp.Self:
        '''
        Return a new Yarn ordered by the sorted Index.

        Args:
            *
            {ascendings}
            {kind}
            {key}

        Returns:
            :obj:`Yarn`
        '''
        order = sort_index_for_order(self._index,
                kind=kind,
                ascending=ascending,
                key=key,
                )
        return self._extract_iloc(order)

    @doc_inject(selector='sort')
    def sort_values(self,
            *,
            ascending: bool = True,
            kind: TSortKinds = DEFAULT_SORT_KIND,
            key: tp.Callable[[TYarnAny], tp.Union[TNDArrayAny, TSeriesAny]],
            ) -> tp.Self:
        '''
        Return a new Yarn ordered by the sorted values. Note that as a Yarn contains Frames, a `key` argument must be provided to extract a sortable value, and this key function will process a :obj:`Series` of :obj:`Frame`.

        Args:
            *
            {ascending}
            {kind}
            {key}

        Returns:
            :obj:`Yarn`
        '''
        cfs = key(self)
        cfs_values: TNDArrayAny = cfs if cfs.__class__ is np.ndarray else cfs.values # type: ignore
        asc_is_element = isinstance(ascending, BOOL_TYPES)
        if not asc_is_element:
            raise RuntimeError('Multiple ascending values not permitted.')

        # argsort lets us do the sort once and reuse the results
        order = np.argsort(cfs_values, kind=kind)
        if not ascending:
            order = order[::-1]

        return self._extract_iloc(order)

    def roll(self,
            shift: int,
            *,
            include_index: bool = False,
            ) -> tp.Self:
        '''Return a Yarn with values rotated forward and wrapped around the index (with a positive shift) or backward and wrapped around the index (with a negative shift).

        Args:
            shift: Positive or negative integer shift.
            include_index: Determine if the Index is shifted with the underlying data.

        Returns:
            :obj:`Yarn`
        '''
        if shift % len(self._indexer):
            indexer = array_shift(
                    array=self._indexer,
                    shift=shift,
                    axis=0,
                    wrap=True)
            indexer.flags.writeable = False
        else:
            indexer = self._indexer

        if include_index:
            index = self._index.roll(shift=shift)
            own_index = True
        else:
            index = self._index
            own_index = False

        return self.__class__(self._values,
                index=index,
                own_index=own_index,
                indexer=indexer,
                hierarchy=self._hierarchy,
                name=self._name,
                deepcopy_from_bus=self._deepcopy_from_bus,
                )

    def shift(self,
            shift: int,
            *,
            fill_value: tp.Any,
            ) -> tp.Self:
        '''Return a :obj:`Yarn` with values shifted forward on the index (with a positive shift) or backward on the index (with a negative shift).

        Args:
            shift: Positive or negative integer shift.
            fill_value: Value to be used to fill data missing after the shift.

        Returns:
            :obj:`Yarn`
        '''
        raise NotImplementedError('A `Yarn` cannot be shifted as newly created missing values cannot be filled without replacing stored `Bus`.')

    #---------------------------------------------------------------------------
    # exporter

    def to_series(self) -> TSeriesObject: # can get generic Bus index
        '''Return a :obj:`Series` with the :obj:`Frame` contained in all contained :obj:`Bus`.
        '''
        # NOTE: this will load all deferred Frame
        return Series(self.values,
                index=self._index,
                own_index=True,
                name=self._name,
                )

    def _to_signature_bytes(self,
            include_name: bool = True,
            include_class: bool = True,
            encoding: str = 'utf-8',
            ) -> bytes:

        # For a Yarn, the signature bytes need only contain the signature of the associated Frame and the index; all else are internal implementation mechanisms

        v = (f._to_signature_bytes(
                include_name=include_name,
                include_class=include_class,
                encoding=encoding,
                ) for f in self._axis_element())

        return b''.join(chain(
                iter_component_signature_bytes(self,
                        include_name=include_name,
                        include_class=include_class,
                        encoding=encoding),
                (self._index._to_signature_bytes(
                        include_name=include_name,
                        include_class=include_class,
                        encoding=encoding),),
                v))


TYarnAny = Yarn[tp.Any]


