import typing as tp
from functools import partial
from itertools import chain

import numpy as np
from numpy.ma import MaskedArray #type: ignore

from static_frame.core.assign import Assign

from static_frame.core.container import ContainerOperand
from static_frame.core.container_util import apply_binary_operator
from static_frame.core.container_util import axis_window_items
from static_frame.core.container_util import index_from_optional_constructor
from static_frame.core.container_util import matmul
from static_frame.core.container_util import pandas_to_numpy
from static_frame.core.container_util import pandas_version_under_1
from static_frame.core.container_util import rehierarch_from_index_hierarchy
from static_frame.core.container_util import index_many_set
from static_frame.core.container_util import index_many_concat

from static_frame.core.display import Display
from static_frame.core.display import DisplayActive
from static_frame.core.display_config import DisplayConfig
from static_frame.core.display_config import DisplayFormats
from static_frame.core.display import DisplayHeader
from static_frame.core.doc_str import doc_inject
from static_frame.core.exception import AxisInvalid
from static_frame.core.exception import ErrorInitSeries

from static_frame.core.index import Index
from static_frame.core.index_auto import IndexAutoFactory
from static_frame.core.index_auto import IndexAutoFactoryType
from static_frame.core.index_base import IndexBase
from static_frame.core.index_auto import RelabelInput
from static_frame.core.index_correspondence import IndexCorrespondence
from static_frame.core.index_hierarchy import IndexHierarchy

from static_frame.core.node_dt import InterfaceDatetime
from static_frame.core.node_iter import IterNodeApplyType
from static_frame.core.node_iter import IterNodeDepthLevel
from static_frame.core.node_iter import IterNodeGroup
from static_frame.core.node_iter import IterNodeNoArg
from static_frame.core.node_iter import IterNodeType
from static_frame.core.node_iter import IterNodeWindow
from static_frame.core.node_selector import InterfaceAssignTrio
from static_frame.core.node_selector import InterfaceGetItem
from static_frame.core.node_selector import InterfaceSelectTrio
from static_frame.core.node_str import InterfaceString

from static_frame.core.util import AnyCallable
from static_frame.core.util import argmax_1d
from static_frame.core.util import argmin_1d
from static_frame.core.util import array_shift
from static_frame.core.util import array_to_duplicated
from static_frame.core.util import array_to_groups_and_locations
from static_frame.core.util import array2d_to_tuples
from static_frame.core.util import binary_transition
from static_frame.core.util import concat_resolved
from static_frame.core.util import DEFAULT_SORT_KIND
from static_frame.core.util import DepthLevelSpecifier
from static_frame.core.util import dtype_to_fill_value
from static_frame.core.util import dtype_from_element
from static_frame.core.util import DtypeSpecifier
from static_frame.core.util import EMPTY_TUPLE
from static_frame.core.util import FLOAT_TYPES
from static_frame.core.util import full_for_fill
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import immutable_filter
from static_frame.core.util import IndexConstructor
from static_frame.core.util import IndexInitializer
from static_frame.core.util import INT_TYPES
from static_frame.core.util import intersect1d
from static_frame.core.util import is_callable_or_mapping
from static_frame.core.util import isin
from static_frame.core.util import isna_array
from static_frame.core.util import iterable_to_array_1d
from static_frame.core.util import mloc
from static_frame.core.util import NAME_DEFAULT
from static_frame.core.util import name_filter
from static_frame.core.util import NameType
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import PathSpecifierOrFileLike
from static_frame.core.util import resolve_dtype
from static_frame.core.util import SeriesInitializer
from static_frame.core.util import slices_from_targets
from static_frame.core.util import ufunc_axis_skipna
from static_frame.core.util import ufunc_unique
from static_frame.core.util import write_optional_file
from static_frame.core.util import UFunc
from static_frame.core.util import dtype_kind_to_na


if tp.TYPE_CHECKING:
    from static_frame import Frame # pylint: disable=W0611 #pragma: no cover
    from static_frame import FrameGO # pylint: disable=W0611 #pragma: no cover
    import pandas # pylint: disable=W0611 #pragma: no cover




#-------------------------------------------------------------------------------
class Series(ContainerOperand):
    '''A one-dimensional, ordered, labelled container, immutable and of fixed size.
    '''

    __slots__ = (
            'values',
            '_index',
            '_name',
            )


    values: np.ndarray

    _index: IndexBase

    _NDIM: int = 1

    #---------------------------------------------------------------------------
    @classmethod
    def from_element(cls,
            element: tp.Any,
            *,
            index: IndexInitializer,
            dtype: DtypeSpecifier = None,
            name: NameType = None,
            index_constructor: tp.Optional[IndexConstructor] = None,
            own_index: bool = False,
            ) -> 'Series':
        '''
        Create a :obj:`static_frame.Series` from a single element. The size of the resultant container will be determined by the ``index`` argument.

        Returns:
            :obj:`static_frame.Series`
        '''
        if own_index:
            index_final = index
        else:
            index_final = index_from_optional_constructor(index,
                    default_constructor=Index,
                    explicit_constructor=index_constructor
                    )
        array = np.full(
                len(index_final), #type: ignore
                fill_value=element,
                dtype=dtype)
        array.flags.writeable = False
        return cls(array,
                index=index_final,
                name=name,
                own_index=True,
                )


    @classmethod
    def from_items(cls,
            pairs: tp.Iterable[tp.Tuple[tp.Hashable, tp.Any]],
            *,
            dtype: DtypeSpecifier = None,
            name: NameType = None,
            index_constructor: tp.Optional[tp.Callable[..., IndexBase]] = None
            ) -> 'Series':
        '''Series construction from an iterator or generator of pairs, where the first pair value is the index and the second is the value.

        Args:
            pairs: Iterable of pairs of index, value.
            dtype: dtype or valid dtype specifier.

        Returns:
            :obj:`static_frame.Series`
        '''
        index = []
        def values() -> tp.Iterator[tp.Any]:
            for k, v in pairs:
                # populate index list as side effect of iterating values
                index.append(k)
                yield v

        return cls(values(),
                index=index,
                dtype=dtype,
                name=name,
                index_constructor=index_constructor)


    @classmethod
    def from_dict(cls,
            mapping: tp.Dict[tp.Hashable, tp.Any],
            *,
            dtype: DtypeSpecifier = None,
            name: NameType = None,
            index_constructor: tp.Optional[tp.Callable[..., IndexBase]] = None
            ) -> 'Series':
        '''Series construction from a dictionary, where the first pair value is the index and the second is the value.

        Args:
            mapping: a dictionary or similar mapping interface.
            dtype: dtype or valid dtype specifier.

        Returns:
            :obj:`static_frame.Series`
        '''
        return cls.from_items(mapping.items(),
                name=name,
                dtype=dtype,
                index_constructor=index_constructor)

    @classmethod
    def from_concat(cls,
            containers: tp.Iterable['Series'],
            *,
            index: tp.Optional[tp.Union[IndexInitializer, IndexAutoFactoryType]] = None,
            name: NameType = None
            ) -> 'Series':
        '''
        Concatenate multiple :obj:`Series` into a new :obj:`Series`.

        Args:
            containers: Iterable of ``Series`` from which values in the new ``Series`` are drawn.
            index: If None, the resultant index will be the concatenation of all indices (assuming they are unique in combination). If ``IndexAutoFactory``, the resultant index is a auto-incremented integer index. Otherwise, the value is used as a index initializer.

        Returns:
            :obj:`static_frame.Series`
        '''
        array_values = []
        if index is None:
            indices = []

        for c in containers:
            array_values.append(c.values)
            if index is None:
                indices.append(c.index)

        # End quickly if empty iterable
        if not array_values:
            return cls(EMPTY_TUPLE, index=index, name=name)

        # returns immutable arrays
        values = concat_resolved(array_values)

        if index is None:
            index = index_many_concat(indices, cls_default=Index)
        elif index is IndexAutoFactory:
            # set index arg to None to force IndexAutoFactory usage in creation
            index = None

        return cls(values, index=index, name=name)

    @classmethod
    def from_concat_items(cls,
            items: tp.Iterable[tp.Tuple[tp.Hashable, 'Series']]
            ) -> 'Series':
        '''
        Produce a :obj:`Series` with a hierarchical index from an iterable of pairs of labels, :obj:`Series`. The :obj:`IndexHierarchy` is formed from the provided labels and the :obj:`Index` if each :obj:`Series`.

        Args:
            items: Iterable of pairs of label, :obj:`Series`

        Returns:
            :obj:`static_frame.Series`
        '''
        array_values = []

        def gen() -> tp.Iterator[tp.Tuple[tp.Hashable, IndexBase]]:
            for label, series in items:
                array_values.append(series.values)
                yield label, series._index

        try:
            # populates array_values as side effect
            ih = IndexHierarchy.from_index_items(gen()) #type: ignore
            # returns immutable array
            values = concat_resolved(array_values)
            own_index = True
        except StopIteration:
            # Default to empty when given an empty iterable
            ih = None #type: ignore
            values = EMPTY_TUPLE
            own_index= False

        return cls(values, index=ih, own_index=own_index)

    @classmethod
    def from_overlay(cls,
            containers: tp.Iterable['Series'],
            *,
            index: tp.Optional[IndexInitializer] = None,
            union: bool = True,
            name: NameType = None,
            ) -> 'Series':
        '''Return a new :obj:`Series` made by overlaying containers, filling in missing values (None or NaN) with aligned values from subsequent containers.

        Args:
            containers: Iterable of :obj:`Series`.
            index: An :obj:`Index` or :obj:`IndexHierarchy`, or index initializer, to be used as the index upon which all containers are aligned. :obj:`IndexAutoFactory` is not supported.
            union: If True, and no ``index`` argument is supplied, a union index from ``containers`` will be used; if False, the intersection index will be used.
        '''
        if not hasattr(containers, '__len__'):
            containers = tuple(containers) # exhaust a generator

        if index is None:
            index = index_many_set(
                    (c.index for c in containers),
                    cls_default=Index,
                    union=union,
                    )
        else: # construct an index if not an index
            if not isinstance(index, IndexBase):
                index = Index(index)

        container_iter = iter(containers)
        container_first = next(container_iter)

        if container_first.index.equals(index):
            post = cls(container_first.values, index=index, own_index=True, name=name)
        else:
            fill_value = dtype_kind_to_na(container_first.dtype.kind)
            post = container_first.reindex(index, fill_value=fill_value).rename(name)

        for container in container_iter:
            post = post.fillna(container)
            if not post.isna().any(): # NOTE: should we short circuit, or get more out of fillna?
                break

        return post



    @classmethod
    @doc_inject()
    def from_pandas(cls,
            value: 'pandas.Series',
            *,
            index_constructor: tp.Optional[tp.Union[IndexConstructor, IndexAutoFactoryType]] = None,
            name: NameType = NAME_DEFAULT,
            own_data: bool = False) -> 'Series':
        '''Given a Pandas Series, return a Series.

        Args:
            value: Pandas Series.
            {own_data}
            {own_index}

        Returns:
            :obj:`static_frame.Series`
        '''
        import pandas
        if not isinstance(value, pandas.Series):
            raise ErrorInitSeries(f'from_pandas must be called with a Pandas Series object, not: {type(value)}')

        if pandas_version_under_1():
            if own_data:
                data = value.values
                data.flags.writeable = False
            else:
                data = immutable_filter(value.values)
        else:
            data = pandas_to_numpy(value, own_data=own_data)

        name = name if name is not NAME_DEFAULT else value.name

        own_index = True
        if index_constructor is IndexAutoFactory:
            index = None
            own_index = False
        elif index_constructor is not None:
            index = index_constructor(value.index) #type: ignore
        else: # if None
            index = Index.from_pandas(value.index)

        return cls(data,
                index=index,
                name=name,
                own_index=own_index
                )


    #---------------------------------------------------------------------------
    @doc_inject(selector='container_init', class_name='Series')
    def __init__(self,
            values: SeriesInitializer,
            *,
            index: tp.Union[IndexInitializer, IndexAutoFactoryType, None] = None,
            name: NameType = NAME_DEFAULT,
            dtype: DtypeSpecifier = None,
            index_constructor: tp.Optional[IndexConstructor] = None,
            own_index: bool = False
            ) -> None:
        '''Initializer.

        Args:
            values: An iterable of values to be aligned with the supplied (or automatically generated) index.
            {index}
            {own_index}
        '''

        if own_index and index is None:
            raise ErrorInitSeries('cannot own_index if no index is provided.')

        #-----------------------------------------------------------------------
        # values assignment

        values_constructor = None # if deferred

        if not isinstance(values, np.ndarray):
            if isinstance(values, dict):
                raise ErrorInitSeries('use Series.from_dict to create a Series from a mapping.')
            elif isinstance(values, Series):
                self.values = values.values # take immutable array
                if dtype is not None and dtype != values.dtype:
                    raise ErrorInitSeries(f'when supplying values via Series, the dtype argument is not required; if provided ({dtype}), it must agree with the dtype of the Series ({values.dtype})')
                if index is None and index_constructor is None:
                    # set up for direct assignment below; index is always immutable
                    index = values.index
                    own_index = True
                if name is NAME_DEFAULT:
                    name = values.name # propagate Series.name
            elif hasattr(values, '__iter__') and not isinstance(values, str):
                # returned array is already immutable
                self.values, _ = iterable_to_array_1d(values, dtype=dtype) #type: ignore
            else: # it must be an element, or a string
                raise ErrorInitSeries('Use Series.from_element to create a Series from an element.')

        else: # is numpy array
            if dtype is not None and dtype != values.dtype:
                raise ErrorInitSeries(f'when supplying values via array, the dtype argument is not required; if provided ({dtype}), it must agree with the dtype of the array ({values.dtype})')

            if values.shape == (): # handle special case of NP element
                def values_constructor(count: int) -> None: #pylint: disable=E0102
                    self.values = np.repeat(values, count)
                    self.values.flags.writeable = False
            else:
                self.values = immutable_filter(values)

        self._name = None if name is NAME_DEFAULT else name_filter(name)

        #-----------------------------------------------------------------------
        # index assignment

        if own_index:
            self._index = index #type: ignore
        elif index is None or index is IndexAutoFactory:
            # if a values constructor is defined, self.values is not yet defined, and no index is supplied, the resultant shape will be of length 1. (If an index is supplied, the shape might be larger than one if an array element was given
            if values_constructor:
                value_count = 1
            else:
                value_count = len(self.values)
            self._index = IndexAutoFactory.from_optional_constructor(
                    value_count,
                    default_constructor=Index,
                    explicit_constructor=index_constructor
                    )
        else: # an iterable of labels, or an index subclass
            self._index = index_from_optional_constructor(index, #type: ignore
                    default_constructor=Index,
                    explicit_constructor=index_constructor
                    )
        index_count = self._index.__len__()

        if not self._index.STATIC:
            raise ErrorInitSeries('non-static index cannot be assigned to Series')

        if values_constructor:
            values_constructor(index_count) # updates self.values
            # must update after calling values constructor
        value_count = len(self.values)

        #-----------------------------------------------------------------------
        # final evaluation

        if self.values.ndim != self._NDIM:
            raise ErrorInitSeries('dimensionality of final values not supported')
        if value_count != index_count:
            raise ErrorInitSeries(
                f'Index has incorrect size (got {index_count}, expected {value_count})'
                )

    # ---------------------------------------------------------------------------
    def __reversed__(self) -> tp.Iterator[tp.Hashable]:
        '''
        Returns a reverse iterator on the series' index.

        Returns:
            :obj:`static_frame.Series`
        '''
        return reversed(self._index) #type: ignore

    #---------------------------------------------------------------------------
    def __setstate__(self, state: tp.Any) -> None:
        '''
        Ensure that reanimated NP arrays are set not writeable.
        '''
        for key, value in state[1].items():
            setattr(self, key, value)
        self.values.flags.writeable = False

    #---------------------------------------------------------------------------
    # name interface

    @property #type: ignore
    @doc_inject()
    def name(self) -> NameType:
        '''{}'''
        return self._name

    def rename(self, name: NameType) -> 'Series':
        '''
        Return a new Series with an updated name attribute.
        '''
        return self.__class__(self.values,
                index=self._index,
                name=name,
                )

    #---------------------------------------------------------------------------
    # interfaces

    @property
    def loc(self) -> InterfaceGetItem['Series']:
        '''
        Interface for label-based selection.
        '''
        return InterfaceGetItem(self._extract_loc)

    @property
    def iloc(self) -> InterfaceGetItem['Series']:
        '''
        Interface for position-based selection.
        '''
        return InterfaceGetItem(self._extract_iloc)

    @property
    def drop(self) -> InterfaceSelectTrio['Series']:
        '''
        Interface for dropping elements from :obj:`static_frame.Series`.
        '''
        return InterfaceSelectTrio( #type: ignore
                func_iloc=self._drop_iloc,
                func_loc=self._drop_loc,
                func_getitem=self._drop_loc
                )

    @property
    def mask(self) -> InterfaceSelectTrio['Series']:
        '''
        Interface for extracting Boolean :obj:`static_frame.Series`.
        '''
        return InterfaceSelectTrio( #type: ignore
                func_iloc=self._extract_iloc_mask,
                func_loc=self._extract_loc_mask,
                func_getitem=self._extract_loc_mask
                )

    @property
    def masked_array(self) -> InterfaceSelectTrio['Series']:
        '''
        Interface for extracting NumPy Masked Arrays.
        '''
        return InterfaceSelectTrio(
                func_iloc=self._extract_iloc_masked_array,
                func_loc=self._extract_loc_masked_array,
                func_getitem=self._extract_loc_masked_array
                )

    @property
    def assign(self) -> InterfaceAssignTrio['Series']:
        '''
        Interface for doing assignment-like selection and replacement.
        '''
        # NOTE: this is not a InterfaceAssignQuartet, like on Frame
        return InterfaceAssignTrio( #type: ignore
                func_iloc=self._extract_iloc_assign,
                func_loc=self._extract_loc_assign,
                func_getitem=self._extract_loc_assign,
                delegate=SeriesAssign
                )

    #---------------------------------------------------------------------------
    @property
    def via_str(self) -> InterfaceString['Series']:
        '''
        Interface for applying string methods to elements in this container.
        '''
        blocks = (self.values,)

        def blocks_to_container(blocks: tp.Iterator[np.ndarray]) -> 'Series':
            return self.__class__(
                next(blocks), # assume only one
                index=self._index,
                name=self._name,
                own_index=True,
                )

        return InterfaceString(
                blocks=blocks,
                blocks_to_container=blocks_to_container,
                )

    @property
    def via_dt(self) -> InterfaceDatetime['Series']:
        '''
        Interface for applying datetime properties and methods to elements in this container.
        '''
        blocks = (self.values,)

        def blocks_to_container(blocks: tp.Iterator[np.ndarray]) -> 'Series':
            return self.__class__(
                next(blocks), # assume only one
                index=self._index,
                name=self._name,
                own_index=True,
                )

        return InterfaceDatetime(
                blocks=blocks,
                blocks_to_container=blocks_to_container,
                )

    #---------------------------------------------------------------------------
    @property
    def iter_group(self) -> IterNodeGroup['Series']:
        '''
        Iterator of :obj:`static_frame.Series`, where each :obj:`static_frame.Series` is matches unique values.
        '''
        return IterNodeGroup(
                container=self,
                function_items=self._axis_group_items,
                function_values=self._axis_group,
                yield_type=IterNodeType.VALUES
                )

    @property
    def iter_group_items(self) -> IterNodeGroup['Series']:
        return IterNodeGroup(
                container=self,
                function_items=self._axis_group_items,
                function_values=self._axis_group,
                yield_type=IterNodeType.ITEMS
                )

    #---------------------------------------------------------------------------
    @property
    def iter_group_labels(self) -> IterNodeDepthLevel['Series']:
        return IterNodeDepthLevel(
                container=self,
                function_items=self._axis_group_labels_items,
                function_values=self._axis_group_labels,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.SERIES_ITEMS_FLAT
                )

    @property
    def iter_group_labels_items(self) -> IterNodeDepthLevel['Series']:
        return IterNodeDepthLevel(
                container=self,
                function_items=self._axis_group_labels_items,
                function_values=self._axis_group_labels,
                yield_type=IterNodeType.ITEMS,
                apply_type=IterNodeApplyType.SERIES_ITEMS_FLAT
                )

    #---------------------------------------------------------------------------
    @property
    def iter_element(self) -> IterNodeNoArg['Series']:
        '''
        Iterator of elements.
        '''
        return IterNodeNoArg(
                container=self,
                function_items=self._axis_element_items,
                function_values=self._axis_element,
                yield_type=IterNodeType.VALUES
                )

    @property
    def iter_element_items(self) -> IterNodeNoArg['Series']:
        '''
        Iterator of label, element pairs.
        '''
        return IterNodeNoArg(
                container=self,
                function_items=self._axis_element_items,
                function_values=self._axis_element,
                yield_type=IterNodeType.ITEMS
                )

    #---------------------------------------------------------------------------
    @property
    def iter_window(self) -> IterNodeWindow['Series']:
        function_values = partial(self._axis_window, as_array=False)
        function_items = partial(self._axis_window_items, as_array=False)
        return IterNodeWindow(
                container=self,
                function_values=function_values,
                function_items=function_items,
                yield_type=IterNodeType.VALUES
                )

    @property
    def iter_window_items(self) -> IterNodeWindow['Series']:
        function_values = partial(self._axis_window, as_array=False)
        function_items = partial(self._axis_window_items, as_array=False)
        return IterNodeWindow(
                container=self,
                function_values=function_values,
                function_items=function_items,
                yield_type=IterNodeType.ITEMS
                )


    @property
    def iter_window_array(self) -> IterNodeWindow['Series']:
        function_values = partial(self._axis_window, as_array=True)
        function_items = partial(self._axis_window_items, as_array=True)
        return IterNodeWindow(
                container=self,
                function_values=function_values,
                function_items=function_items,
                yield_type=IterNodeType.VALUES
                )

    @property
    def iter_window_array_items(self) -> IterNodeWindow['Series']:
        function_values = partial(self._axis_window, as_array=True)
        function_items = partial(self._axis_window_items, as_array=True)
        return IterNodeWindow(
                container=self,
                function_values=function_values,
                function_items=function_items,
                yield_type=IterNodeType.ITEMS
                )
    #---------------------------------------------------------------------------
    # index manipulation

    def _reindex_other_like_iloc(self,
            value: 'Series',
            iloc_key: GetItemKeyType,
            fill_value: tp.Any = np.nan,
            ) -> 'Series':
        '''Given a value that is a Series, reindex it to the index components, drawn from this Series, that are specified by the iloc_key.
        '''
        return value.reindex( #type: ignore
                self._index._extract_iloc(iloc_key),
                fill_value=fill_value
                )

    @doc_inject(selector='reindex', class_name='Series')
    def reindex(self,
            index: IndexInitializer,
            *,
            fill_value: tp.Any = np.nan,
            own_index: bool = False,
            check_equals: bool = True
            ) -> 'Series':
        '''
        {doc}

        Args:
            index: {index_initializer}
            columns: {index_initializer}
            {fill_value}
            {own_index}
        '''
        if not own_index:
            index = index_from_optional_constructor(index,
                    default_constructor=Index)

        # NOTE: it is assumed that the equals comparison is faster than continuing with this method
        if check_equals and self._index.equals(index):
            # if labels are equal (even if a different Index subclass), we can simply use the new Index
            return self.__class__(
                    self.values,
                    index=index,
                    own_index=True,
                    name=self._name)

        ic = IndexCorrespondence.from_correspondence(self._index, index) #type: ignore

        if ic.is_subset: # must have some common
            values = self.values[ic.iloc_src]
            values.flags.writeable = False
            return self.__class__(
                    values,
                    index=index,
                    own_index=True,
                    name=self._name)

        values = full_for_fill(self.values.dtype, len(index), fill_value) #type: ignore
        # if some intersection of values
        if ic.has_common:
            values[ic.iloc_dst] = self.values[ic.iloc_src]
        values.flags.writeable = False

        return self.__class__(values,
                index=index,
                own_index=True,
                name=self._name)

    @doc_inject(selector='relabel', class_name='Series')
    def relabel(self,
            index: RelabelInput
            ) -> 'Series':
        '''
        {doc}

        Args:
            index: {relabel_input}
        '''
        #NOTE: we name the parameter index for alignment with the corresponding Frame method

        own_index = False
        if index is IndexAutoFactory:
            index_init = None
        elif is_callable_or_mapping(index): #type: ignore
            index_init = self._index.relabel(index)
            own_index = True
        elif index is None:
            index_init = self._index
        else:
            index_init = index #type: ignore

        return self.__class__(self.values,
                index=index_init,
                own_index=own_index,
                name=self._name)

    @doc_inject(selector='relabel_flat', class_name='Series')
    def relabel_flat(self) -> 'Series':
        '''
        {doc}
        '''
        if not isinstance(self._index, IndexHierarchy):
            raise RuntimeError('cannot flatten an Index that is not an IndexHierarchy')

        return self.__class__(self.values,
                index=self._index.flat(),
                name=self._name)

    @doc_inject(selector='relabel_level_add', class_name='Series')
    def relabel_level_add(self,
            level: tp.Hashable
            ) -> 'Series':
        '''
        {doc}

        Args:
            level: {level}
        '''
        return self.__class__(self.values,
                index=self._index.level_add(level),
                name=self._name)

    @doc_inject(selector='relabel_level_drop', class_name='Series')
    def relabel_level_drop(self,
            count: int = 1
            ) -> 'Series':
        '''
        {doc}

        Args:
            count: {count}
        '''
        if not isinstance(self._index, IndexHierarchy):
            raise RuntimeError('cannot drop level of an Index that is not an IndexHierarchy')

        return self.__class__(self.values,
                index=self._index.level_drop(count),
                name=self._name)


    def rehierarch(self,
            depth_map: tp.Sequence[int]
            ) -> 'Series':
        '''
        Return a new :obj:`Series` with new a hierarchy based on the supplied ``depth_map``.
        '''
        if self.index.depth == 1:
            raise RuntimeError('cannot rehierarch when there is no hierarchy')

        index, iloc_map = rehierarch_from_index_hierarchy(
                labels=self._index, #type: ignore
                depth_map=depth_map,
                name=self._index.name,
                )
        values = self.values[iloc_map]
        values.flags.writeable = False
        return self.__class__(values,
                index=index,
                name=self._name)


    #---------------------------------------------------------------------------
    # na handling

    def isna(self) -> 'Series':
        '''
        Return a same-indexed, Boolean ``Series`` indicating which values are NaN or None.
        '''
        # consider returning self if not values.any()?
        values = isna_array(self.values)
        values.flags.writeable = False
        return self.__class__(values, index=self._index)

    def notna(self) -> 'Series':
        '''
        Return a same-indexed, Boolean Series indicating which values are NaN or None.
        '''
        values = np.logical_not(isna_array(self.values))
        values.flags.writeable = False
        return self.__class__(values, index=self._index)

    def dropna(self) -> 'Series':
        '''
        Return a new :obj:`static_frame.Series` after removing values of NaN or None.
        '''
        # get positions that we want to keep
        sel = np.logical_not(isna_array(self.values))
        if not np.any(sel):
            return self.__class__(())

        values = self.values[sel]
        values.flags.writeable = False

        return self.__class__(values,
                index=self._index.loc[sel],
                name=self._name,
                own_index=True)

    @doc_inject(selector='fillna')
    def fillna(self,
            value: tp.Any # an element or a Series
            ) -> 'Series':
        '''Return a new :obj:`Series` after replacing null (NaN or None) with the supplied value. The value can be an element or :obj:`Series`.

        Args:
            {value}
        '''
        values = self.values
        sel = isna_array(values)
        if not np.any(sel):
            return self

        if hasattr(value, '__iter__') and not isinstance(value, str):
            if not isinstance(value, Series):
                raise RuntimeError('unlabeled iterables cannot be used for fillna: use a Series')
            value_dtype = value.dtype
            # choose a fill value that will not force a type coercion
            fill_value = dtype_to_fill_value(value_dtype)
            # find targets that are NaN in self and have labels in value; otherwise, might fill values after reindexing, and end up filling a fill_value rather than keeping original (na) value
            labels_common = intersect1d(self.index.values[sel], value.index.values)
            sel = self.index.isin(labels_common)
            if not np.any(sel): # avoid copying, retyping
                return self

            # must reindex to align ordering; just get array
            value = self._reindex_other_like_iloc(value,
                    sel,
                    fill_value=fill_value).values
        else:
            value_dtype = dtype_from_element(value)

        assignable_dtype = resolve_dtype(value_dtype, values.dtype)

        if values.dtype == assignable_dtype:
            assigned = values.copy()
        else:
            assigned = values.astype(assignable_dtype)

        assigned[sel] = value
        assigned.flags.writeable = False

        return self.__class__(assigned,
                index=self._index,
                name=self._name)



    @staticmethod
    def _fillna_directional(
            array: np.ndarray,
            directional_forward: bool,
            limit: int = 0) -> np.ndarray:
        '''Return a new ``Series`` after feeding forward the last non-null (NaN or None) observation across contiguous nulls.

        Args:
            count: Set the limit of nan values to be filled per nan region. A value of 0 is equivalent to no limit.
        '''
        sel = isna_array(array)
        if not np.any(sel):
            return array

        def slice_condition(target_slice: slice) -> bool:
            # NOTE: start is never None
            return sel[target_slice.start] #type: ignore

        # type is already compatible, no need for check
        assigned = array.copy()
        target_index = binary_transition(sel)
        target_values = array[target_index]
        length = len(array)

        for target_slice, value in slices_from_targets(
                target_index=target_index,
                target_values=target_values,
                length=length,
                directional_forward=directional_forward,
                limit=limit,
                slice_condition=slice_condition # isna True in region
                ):
            assigned[target_slice] = value

        assigned.flags.writeable = False
        return assigned

    @doc_inject(selector='fillna')
    def fillna_forward(self, limit: int = 0) -> 'Series':
        '''Return a new ``Series`` after feeding forward the last non-null (NaN or None) observation across contiguous nulls.

        Args:
            {limit}
        '''
        return self.__class__(self._fillna_directional(
                    array=self.values,
                    directional_forward=True,
                    limit=limit),
                index=self._index,
                name=self._name)

    @doc_inject(selector='fillna')
    def fillna_backward(self, limit: int = 0) -> 'Series':
        '''Return a new ``Series`` after feeding backward the last non-null (NaN or None) observation across contiguous nulls.

        Args:
            {limit}
        '''
        return self.__class__(self._fillna_directional(
                    array=self.values,
                    directional_forward=False,
                    limit=limit),
                index=self._index,
                name=self._name)


    @staticmethod
    def _fillna_sided(array: np.ndarray,
            value: tp.Any,
            sided_leading: bool,
            ) -> np.ndarray:
        '''
        Args:
            sided_leading: True sets the side to fill is the leading side; False sets the side to fill to the trailiing side.
        '''
        sel = isna_array(array)

        if not np.any(sel):
            return array

        sided_index = 0 if sided_leading else -1

        if not sel[sided_index]:
            # sided value is not null: nothing to do
            return array # assume immutable

        if isinstance(value, np.ndarray):
            raise RuntimeError('cannot assign an array to fillna')

        assignable_dtype = resolve_dtype(
                dtype_from_element(value),
                array.dtype)

        if array.dtype == assignable_dtype:
            assigned = array.copy()
        else:
            assigned = array.astype(assignable_dtype)

        targets = np.nonzero(~sel)[0] # as 1D, can just take index 0 resuilts
        if len(targets):
            if sided_leading:
                sel_slice = slice(0, targets[0])
            else: # trailing
                sel_slice = slice(targets[-1]+1, None)
        else: # all are NaN
            sel_slice = NULL_SLICE

        assigned[sel_slice] = value
        assigned.flags.writeable = False
        return assigned

    @doc_inject(selector='fillna')
    def fillna_leading(self, value: tp.Any) -> 'Series':
        '''Return a new ``Series`` after filling leading (and only leading) null (NaN or None) with the supplied value.

        Args:
            {value}
        '''
        return self.__class__(self._fillna_sided(
                    array=self.values,
                    value=value,
                    sided_leading=True),
                index=self._index,
                name=self._name)

    @doc_inject(selector='fillna')
    def fillna_trailing(self, value: tp.Any) -> 'Series':
        '''Return a new ``Series`` after filling trailing (and only trailing) null (NaN or None) with the supplied value.

        Args:
            {value}
        '''
        return self.__class__(self._fillna_sided(
                    array=self.values,
                    value=value,
                    sided_leading=False),
                index=self._index,
                name=self._name)

    #---------------------------------------------------------------------------
    # operators

    def _ufunc_unary_operator(self, operator: UFunc) -> 'Series':
        '''
        For unary operations, the `name` attribute propagates.
        '''
        values = operator(self.values)
        return self.__class__(values,
                index=self._index,
                dtype=values.dtype, # some operators might change the dtype
                name=self._name
                )

    def _ufunc_binary_operator(self, *,
            operator: UFunc,
            other: tp.Any,
            ) -> 'Series':
        '''
        For binary operations, the `name` attribute does not propagate unless other is a scalar.
        '''
        # get both reverse and regular
        if operator.__name__ == 'matmul':
            return matmul(self, other) #type: ignore
        elif operator.__name__ == 'rmatmul':
            return matmul(other, self) #type: ignore

        values = self.values
        index = self._index
        other_is_array = False

        if isinstance(other, Series):
            name = None
            other_is_array = True
            if not self._index.equals(other._index):
                # if not equal, create a new Index by forming the union
                index = self._index.union(other._index)
                # now need to reindex the Series
                values = self.reindex(index, own_index=True, check_equals=False).values
                other = other.reindex(index, own_index=True, check_equals=False).values
            else:
                other = other.values
        elif isinstance(other, np.ndarray):
            name = None
            other_is_array = True
            if other.ndim > 1:
                raise NotImplementedError('Operator application to greater dimensionalities will result in an array with more than 1 dimension.')
        else:
            name = self._name

        result = apply_binary_operator(
                values=values,
                other=other,
                other_is_array=other_is_array,
                operator=operator,
                )
        return self.__class__(result, index=index, name=name)

    def _ufunc_axis_skipna(self, *,
            axis: int,
            skipna: bool,
            ufunc: UFunc,
            ufunc_skipna: UFunc,
            composable: bool,
            dtypes: tp.Tuple[np.dtype, ...],
            size_one_unity: bool
            ) -> np.ndarray:
        '''
        For a Series, all functions of this type reduce the single axis of the Series to a single element, so Index has no use here.

        Args:
            dtype: not used, part of signature for a common interface
        '''
        return ufunc_axis_skipna(
                array=self.values,
                skipna=skipna,
                axis=0,
                ufunc=ufunc,
                ufunc_skipna=ufunc_skipna
                )

    def _ufunc_shape_skipna(self, *,
            axis: int,
            skipna: bool,
            ufunc: UFunc,
            ufunc_skipna: UFunc,
            composable: bool,
            dtypes: tp.Tuple[np.dtype, ...],
            size_one_unity: bool
            ) -> 'Series':
        '''
        NumPy ufunc proccessors that retain the shape of the processed.

        Args:
            dtypes: not used, part of signature for a common interface
        '''
        values = ufunc_axis_skipna(
                array=self.values,
                skipna=skipna,
                axis=0,
                ufunc=ufunc,
                ufunc_skipna=ufunc_skipna
                )
        values.flags.writeable = False
        return self.__class__(values, index=self._index)

    #---------------------------------------------------------------------------
    def __len__(self) -> int:
        '''Length of values.
        '''
        return self.values.__len__() #type: ignore

    def _display(self,
            config: DisplayConfig,
            display_cls: Display,
            ) -> Display:
        '''
        Private display interface to be shared by Bus and Series.
        '''
        index_depth = self._index.depth if config.include_index else 0
        display_index = self._index.display(config=config)

        # When showing type we need 2: one for the Series type, the other for the index type.
        header_depth = 2 * config.type_show

        # create an empty display based on index display
        d = Display([list() for _ in range(len(display_index))],
                config=config,
                outermost=True,
                index_depth=index_depth,
                header_depth=header_depth
                )

        if config.include_index:
            d.extend_display(display_index)
            header_values = '' if config.type_show else None
        else:
            header_values = None

        d.extend_display(Display.from_values(
                self.values,
                header=header_values,
                config=config))

        if config.type_show:
            d.insert_displays(display_cls.flatten())

        return d

    @doc_inject()
    def display(self,
            config: tp.Optional[DisplayConfig] = None
            ) -> Display:
        '''{doc}

        Args:
            {config}
        '''
        config = config or DisplayActive.get()
        display_cls = Display.from_values((),
                header=DisplayHeader(self.__class__, self._name),
                config=config)
        return self._display(config, display_cls)

        # config = config or DisplayActive.get()
        # index_depth = self._index.depth if config.include_index else 0
        # display_index = self._index.display(config=config)

        # # When showing type we need 2: one for the Series type, the other for the index type.
        # header_depth = 2 * config.type_show

        # # create an empty display based on index display
        # d = Display([list() for _ in range(len(display_index))],
        #         config=config,
        #         outermost=True,
        #         index_depth=index_depth,
        #         header_depth=header_depth
        #         )

        # if config.include_index:
        #     d.extend_display(display_index)
        #     header_values = '' if config.type_show else None
        # else:
        #     header_values = None

        # d.extend_display(Display.from_values(
        #         self.values,
        #         header=header_values,
        #         config=config))

        # if config.type_show:
        #     display_cls = Display.from_values((),
        #             header=DisplayHeader(self.__class__, self._name),
        #             config=config)
        #     d.insert_displays(display_cls.flatten())

        # return d

    #---------------------------------------------------------------------------
    # common attributes from the numpy array

    @property #type: ignore
    @doc_inject()
    def mloc(self) -> int:
        '''{doc_int}
        '''
        return mloc(self.values)

    @property
    def dtype(self) -> np.dtype:
        '''
        Return the dtype of the underlying NumPy array.

        Returns:
            :obj:`numpy.dtype`
        '''
        return self.values.dtype

    @property
    def shape(self) -> tp.Tuple[int]:
        '''
        Return a tuple describing the shape of the underlying NumPy array.

        Returns:
            :obj:`Tuple[int]`
        '''
        return self.values.shape #type: ignore

    @property
    def ndim(self) -> int:
        '''
        Return the number of dimensions, which for a `Series` is always 1.

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
        return self.values.size #type: ignore

    @property
    def nbytes(self) -> int:
        '''
        Return the total bytes of the underlying NumPy array.

        Returns:
            :obj:`int`
        '''
        return self.values.nbytes #type: ignore

    # def __bool__(self) -> bool:
    #     '''
    #     True if this container has size.
    #     '''
    #     return bool(self.values.size)


    #---------------------------------------------------------------------------
    # extraction

    def _extract_iloc(self, key: GetItemKeyType) -> 'Series':
        # iterable selection should be handled by NP
        values = self.values[key]

        if not isinstance(values, np.ndarray): # if we have a single element
            return values #type: ignore
        return self.__class__(
                values,
                index=self._index.iloc[key],
                name=self._name)

    def _extract_loc(self, key: GetItemKeyType) -> 'Series':
        '''
        Compatibility:
            Pandas supports taking in iterables of keys, where some keys are not found in the index; a Series is returned as if a reindex operation was performed. This is undesirable. Better instead is to use reindex()
        '''
        iloc_key = self._index.loc_to_iloc(key)
        values = self.values[iloc_key]

        if not isinstance(values, np.ndarray): # if we have a single element
            # NOTE: this branch is not encountered and may not be necessary
            # if isinstance(key, HLoc) and key.has_key_multiple():
            #     # must return a Series, even though we do not have an array
            #     values = np.array(values)
            #     values.flags.writeable = False
            return values #type: ignore

        return self.__class__(values,
                index=self._index.iloc[iloc_key],
                own_index=True,
                name=self._name)

    @doc_inject(selector='selector')
    def __getitem__(self, key: GetItemKeyType) -> 'Series':
        '''Selector of values by label.

        Args:
            key: {key_loc}

        Compatibility:
            Pandas supports using both loc and iloc style selections with the __getitem__ interface on Series. This is undesirable, so here we only expose the loc interface (making the Series dictionary like, but unlike the Index, where __getitem__ is an iloc).
        '''
        return self._extract_loc(key)

    #---------------------------------------------------------------------------
    # utilites for alternate extraction: drop, mask and assignment

    def _drop_iloc(self, key: GetItemKeyType) -> 'Series':
        if isinstance(key, np.ndarray) and key.dtype == bool:
            # use Boolean array to select indices from Index positions, as np.delete does not work with arrays
            values = np.delete(self.values, self._index.positions[key])
        else:
            values = np.delete(self.values, key)
        values.flags.writeable = False

        index = self._index._drop_iloc(key)

        return self.__class__(values,
                index=index,
                name=self._name,
                own_index=True
                )

    def _drop_loc(self, key: GetItemKeyType) -> 'Series':
        return self._drop_iloc(self._index.loc_to_iloc(key))

    #---------------------------------------------------------------------------

    def _extract_iloc_mask(self, key: GetItemKeyType) -> 'Series':
        '''Produce a new boolean Series of the same shape, where the values selected via iloc selection are True. The `name` attribute is not propagated.
        '''
        mask = np.full(self.values.shape, False, dtype=bool)
        mask[key] = True
        mask.flags.writeable = False
        return self.__class__(mask, index=self._index)

    def _extract_loc_mask(self, key: GetItemKeyType) -> 'Series':
        '''Produce a new boolean Series of the same shape, where the values selected via loc selection are True. The `name` attribute is not propagated.
        '''
        iloc_key = self._index.loc_to_iloc(key)
        return self._extract_iloc_mask(key=iloc_key)

    #---------------------------------------------------------------------------

    def _extract_iloc_masked_array(self, key: GetItemKeyType) -> MaskedArray:
        '''Produce a new boolean Series of the same shape, where the values selected via iloc selection are True.
        '''
        mask = self._extract_iloc_mask(key=key)
        return MaskedArray(data=self.values, mask=mask.values)

    def _extract_loc_masked_array(self, key: GetItemKeyType) -> MaskedArray:
        '''Produce a new boolean Series of the same shape, where the values selected via loc selection are True.
        '''
        iloc_key = self._index.loc_to_iloc(key)
        return self._extract_iloc_masked_array(key=iloc_key)

    #---------------------------------------------------------------------------

    def _extract_iloc_assign(self, key: GetItemKeyType) -> 'SeriesAssign':
        return SeriesAssign(self, iloc_key=key)

    def _extract_loc_assign(self, key: GetItemKeyType) -> 'SeriesAssign':
        iloc_key = self._index.loc_to_iloc(key)
        return SeriesAssign(self, iloc_key=iloc_key)

    #---------------------------------------------------------------------------
    # axis functions

    def _axis_group_items(self, *,
            axis: int = 0
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, 'Series']]:
        if axis != 0:
            raise AxisInvalid(f'invalid axis {axis}')

        groups, locations = array_to_groups_and_locations(self.values)
        for idx, g in enumerate(groups):
            selection = locations == idx
            yield g, self._extract_iloc(selection)

    def _axis_group(self, *,
            axis: int = 0
            ) -> tp.Iterator['Series']:
        yield from (x for _, x in self._axis_group_items(axis=axis))


    def _axis_element_items(self,
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, tp.Any]]:
        '''Generator of index, value pairs, equivalent to Series.items(). Rpeated to have a common signature as other axis functions.
        '''
        yield from zip(self._index, self.values)

    def _axis_element(self,
            ) -> tp.Iterator[tp.Any]:
        yield from self.values



    def _axis_group_labels_items(self,
            depth_level: DepthLevelSpecifier = 0,
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, 'Series']]:

        values = self.index.values_at_depth(depth_level)
        group_to_tuple = values.ndim == 2
        groups, locations = array_to_groups_and_locations(
                values)

        for idx, g in enumerate(groups):
            selection = locations == idx
            if group_to_tuple:
                g = tuple(g)
            yield g, self._extract_iloc(selection)

    def _axis_group_labels(self,
            depth_level: DepthLevelSpecifier = 0,
            ) -> tp.Iterator[tp.Hashable]:
        yield from (x for _, x in self._axis_group_labels_items(
                depth_level=depth_level))



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
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, tp.Union[np.ndarray, 'Series']]]:
        '''Generator of index, processed-window pairs.
        '''
        yield from axis_window_items(
                source=self,
                axis=axis,
                size=size,
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
            ) -> tp.Iterator[tp.Union[np.ndarray, 'Series']]:
        yield from (x for _, x in self._axis_window_items(
                axis=axis,
                size=size,
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

    @property
    def index(self) -> IndexBase:
        '''
        The index instance assigned to this container.

        Returns:
            :obj:`static_frame.Index`
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

    def __iter__(self) -> tp.Iterator[tp.Hashable]:
        '''
        Iterator of index labels, same as :obj:`static_frame.Series.keys`.

        Returns:
            :obj:`Iterator[Hashasble]`
        '''
        return self._index.__iter__()

    def __contains__(self, value: tp.Hashable) -> bool:
        '''
        Inclusion of value in index labels.

        Returns:
            :obj:`bool`
        '''
        return self._index.__contains__(value)

    def items(self) -> tp.Iterator[tp.Tuple[tp.Any, tp.Any]]:
        '''Iterator of pairs of index label and value.

        Returns:
            :obj:`Iterator[Tuple[Hashable, Any]]`
        '''
        return zip(self._index.values, self.values)

    def get(self, key: tp.Hashable,
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
    # transformations resulting in the same dimensionality

    def sort_index(self,
            *,
            ascending: bool = True,
            kind: str = DEFAULT_SORT_KIND
            ) -> 'Series':
        '''
        Return a new Series ordered by the sorted Index.

        Args:
            *
            ascending: if True, values are sorted low to high
            kind: sort algorithm

        Returns:
            :obj:`static_frame.Series`
        '''
        # argsort lets us do the sort once and reuse the results
        if self._index.depth > 1:
            v = self._index.values
            order = np.lexsort([v[:, i] for i in range(v.shape[1]-1, -1, -1)])
        else:
            # this technique does not work when values is a 2d array
            order = np.argsort(self._index.values, kind=kind)

        if not ascending:
            order = order[::-1]

        index_values = self._index.values[order]
        index_values.flags.writeable = False
        index = self._index.from_labels(index_values, name=self._index._name)

        values = self.values[order]
        values.flags.writeable = False

        return self.__class__(values,
                index=index,
                name=self._name,
                own_index=True
                )

    def sort_values(self,
            *,
            ascending: bool = True,
            kind: str = DEFAULT_SORT_KIND
            ) -> 'Series':
        '''
        Return a new Series ordered by the sorted values.

        Returns:
            :obj:`Series`
        '''
        # argsort lets us do the sort once and reuse the results
        order = np.argsort(self.values, kind=kind)
        if not ascending:
            order = order[::-1]

        index_values = self._index.values[order]
        index_values.flags.writeable = False
        index = self._index.from_labels(index_values, name=self._index._name)

        values = self.values[order]
        values.flags.writeable = False

        return self.__class__(values,
                index=index,
                name=self._name,
                own_index=True
                )

    def isin(self, other: tp.Iterable[tp.Any]) -> 'Series':
        '''
        Return a same-sized Boolean Series that shows if the same-positioned element is in the iterable passed to the function.

        Returns:
            :obj:`Series`
        '''
        array = isin(self.values, other)
        return self.__class__(array, index=self._index, name=self._name)

    @doc_inject(class_name='Series')
    def clip(self, *,
            lower: tp.Optional[tp.Union[float, 'Series']] = None,
            upper: tp.Optional[tp.Union[float, 'Series']] = None,
            ) -> 'Series':
        '''{}

        Args:
            lower: value or ``Series`` to define the inclusive lower bound.
            upper: value or ``Series`` to define the inclusive upper bound.

        Returns:
            :obj:`Series`
        '''
        args = [lower, upper]
        for idx, arg in enumerate(args):
            # arg = args[idx]
            if isinstance(arg, Series):
                # after reindexing, strip away the index
                # NOTE: using the bound forces going to a float type; this may not be the best approach
                bound = -np.inf if idx == 0 else np.inf
                args[idx] = arg.reindex(self.index).fillna(bound).values
            elif hasattr(arg, '__iter__'):
                raise RuntimeError('only Series are supported as iterable lower/upper arguments')
            # assume single value otherwise, no change necessary

        array = np.clip(self.values, *args)
        array.flags.writeable = False
        return self.__class__(array, index=self._index, name=self._name)

    def transpose(self) -> 'Series':
        '''Transpose. For a 1D immutable container, this returns a reference to self.

        Returns:
            :obj:`Series`
        '''
        return self

    @property
    def T(self) -> 'Series':
        '''Transpose. For a 1D immutable container, this returns a reference to self.

        Returns:
            :obj:`Series`
        '''
        return self.transpose()

    @doc_inject(selector='duplicated')
    def duplicated(self, *,
            exclude_first: bool = False,
            exclude_last: bool = False,
            ) -> np.ndarray:
        '''
        Return a same-sized Boolean Series that shows True for all b values that are duplicated.

        Args:
            {exclude_first}
            {exclude_last}

        Returns:
            :obj:`numpy.ndarray`
        '''
        duplicates = array_to_duplicated(self.values,
                exclude_first=exclude_first,
                exclude_last=exclude_last)
        duplicates.flags.writeable = False
        return self.__class__(duplicates, index=self._index)

    @doc_inject(selector='duplicated')
    def drop_duplicated(self, *,
            exclude_first: bool = False,
            exclude_last: bool = False
            ) -> 'Series':
        '''
        Return a Series with duplicated values removed.

        Args:
            {exclude_first}
            {exclude_last}

        Returns:
            :obj:`Series`
        '''
        duplicates = array_to_duplicated(self.values,
                exclude_first=exclude_first,
                exclude_last=exclude_last)
        keep = ~duplicates
        return self.__class__(self.values[keep],
                index=self._index[keep],
                name=self._name
                )

    @doc_inject(select='astype')
    def astype(self, dtype: DtypeSpecifier) -> 'Series':
        '''
        Return a Series with type determined by `dtype` argument. Note that for Series, this is a simple function, whereas for ``Frame``, this is an interface exposing both a callable and a getitem interface.

        Args:
            {dtype}

        Returns:
            :obj:`Series`
        '''
        return self.__class__(
                self.values.astype(dtype),
                index=self._index,
                name=self._name
                )

    def __round__(self, decimals: int = 0) -> 'Series':
        '''
        Return a Series rounded to the given decimals. Negative decimals round to the left of the decimal point.

        Args:
            decimals: number of decimals to round to.

        Returns:
            :obj:`Series`
        '''
        return self.__class__(
                np.round(self.values, decimals),
                index=self._index,
                name=self._name
                )

    def roll(self,
            shift: int,
            *,
            include_index: bool = False) -> 'Series':
        '''Return a Series with values rotated forward and wrapped around the index (with a postive shift) or backward and wrapped around the index (with a negative shift).

        Args:
            shift: Postive or negative integer shift.
            include_index: Determine if the Index is shifted with the underlying data.

        Returns:
            :obj:`Series`
        '''
        if shift % len(self.values):
            values = array_shift(
                    array=self.values,
                    shift=shift,
                    axis=0,
                    wrap=True)
            values.flags.writeable = False
        else:
            values = self.values

        if include_index:
            index = self._index.roll(shift=shift)
            own_index = True
        else:
            index = self._index
            own_index = False

        return self.__class__(values,
                index=index,
                name=self._name,
                own_index=own_index)


    def shift(self,
            shift: int,
            *,
            fill_value: tp.Any = np.nan) -> 'Series':
        '''Return a Series with values shifted forward on the index (with a postive shift) or backward on the index (with a negative shift).

        Args:
            shift: Postive or negative integer shift.
            fill_value: Value to be used to fill data missing after the shift.

        Returns:
            :obj:`Series`
        '''

        if shift:
            values = array_shift(
                    array=self.values,
                    shift=shift,
                    axis=0,
                    wrap=False,
                    fill_value=fill_value)
            values.flags.writeable = False
        else:
            values = self.values

        return self.__class__(values,
                index=self._index,
                name=self._name)


    #---------------------------------------------------------------------------
    # transformations resulting in changed dimensionality

    @doc_inject(selector='head', class_name='Series')
    def head(self, count: int = 5) -> 'Series':
        '''{doc}

        Args:
            {count}

        Returns:
            :obj:`Series`
        '''
        return self.iloc[:count]

    @doc_inject(selector='tail', class_name='Series')
    def tail(self, count: int = 5) -> 'Series':
        '''{doc}s

        Args:
            {count}

        Returns:
            :obj:`Series`
        '''
        return self.iloc[-count:]

    @doc_inject(selector='argminmax')
    def loc_min(self, *,
            skipna: bool = True
            ) -> tp.Hashable:
        '''
        Return the label corresponding to the minimum value found.

        Args:
            {skipna}

        Returns:
            tp.Hashable
        '''
        # if skipna is False and a NaN is returned, this will raise
        post = argmin_1d(self.values, skipna=skipna)
        if isinstance(post, FLOAT_TYPES): # NaN was returned
            raise RuntimeError('cannot produce loc representation from NaN')
        return self.index[post]

    @doc_inject(selector='argminmax')
    def iloc_min(self, *,
            skipna: bool = True,
            ) -> int:
        '''
        Return the integer index corresponding to the minimum value found.

        Args:
            {skipna}

        Returns:
            int
        '''
        return argmin_1d(self.values, skipna=skipna) #type: ignore

    @doc_inject(selector='argminmax')
    def loc_max(self, *,
            skipna: bool = True,
            ) -> tp.Hashable:
        '''
        Return the label corresponding to the maximum value found.

        Args:
            {skipna}

        Returns:
            tp.Hashable
        '''
        post = argmax_1d(self.values, skipna=skipna)
        if isinstance(post, FLOAT_TYPES): # NaN was returned
            raise RuntimeError('cannot produce loc representation from NaN')
        return self.index[post]

    @doc_inject(selector='argminmax')
    def iloc_max(self, *,
                skipna: bool = True,
                ) -> int:
        '''
        Return the integer index corresponding to the maximum value.

        Args:
            {skipna}

        Returns:
            int
        '''
        return argmax_1d(self.values, skipna=skipna) #type: ignore

    #---------------------------------------------------------------------------
    def _insert(self,
            key: int, # iloc positions
            container: 'Series',
            ) -> 'Series':
        if not isinstance(container, Series):
            raise NotImplementedError(
                    f'No support for inserting with {type(container)}')

        if not len(container.index): # must be empty data, empty index container
            return self

        dtype = resolve_dtype(self.values.dtype, container.dtype)
        values = np.empty(len(self) + len(container), dtype=dtype)
        key_end = key + len(container)

        values_prior = self.values

        values[:key] = values_prior[:key]
        values[key: key_end] = container.values
        values[key_end:] = values_prior[key:]
        values.flags.writeable = False

        labels_prior = self._index.values

        index = self._index.__class__.from_labels(chain(
                labels_prior[:key],
                container._index.__iter__(), #type: ignore
                labels_prior[key:],
                ))

        return self.__class__(values,
                index=index,
                name=self._name,
                own_index=True,
                )

    @doc_inject(selector='insert')
    def insert_before(self,
            key: tp.Hashable,
            container: 'Series',
            ) -> 'Series':
        '''
        Create a new :obj:`Series` by inserting a :obj:`Series` at the position before the label specified by ``key``.

        Args:
            {key_before}
            {container}

        Returns:
            :obj:`Series`
        '''
        iloc_key = self._index.loc_to_iloc(key)
        if not isinstance(iloc_key, INT_TYPES):
            raise RuntimeError(f'Unsupported key type: {key}')
        return self._insert(iloc_key, container)

    @doc_inject(selector='insert')
    def insert_after(self,
            key: tp.Hashable, # iloc positions
            container: 'Series',
            ) -> 'Series':
        '''
        Create a new :obj:`Series` by inserting a :obj:`Series` at the position after the label specified by ``key``.

        Args:
            {key_after}
            {container}

        Returns:
            :obj:`Series`
        '''
        iloc_key = self._index.loc_to_iloc(key)
        if not isinstance(iloc_key, INT_TYPES):
            raise RuntimeError(f'Unsupported key type: {key}')
        return self._insert(iloc_key + 1, container)





    #---------------------------------------------------------------------------
    # utility function to numpy array or other types

    def unique(self) -> np.ndarray:
        '''
        Return a NumPy array of unqiue values.

        Returns:
            :obj:`numpy.ndarray`
        '''
        return ufunc_unique(self.values)

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

        Args:
            {compare_name}
            {compare_dtype}
            {compare_class}
            {skipna}
        '''
        if id(other) == id(self):
            return True

        # NOTE: there are presently no Series subclasses, but better to be consistent
        if compare_class and self.__class__ != other.__class__:
            return False
        elif not isinstance(other, Series):
            return False

        if len(self.values) != len(other.values):
            return False
        if compare_name and self._name != other._name:
            return False
        if compare_dtype and self.values.dtype != other.values.dtype:
            return False

        eq = self.values == other.values

        # NOTE: will only be False, or an array
        if eq is False:
            return eq #type: ignore

        if skipna:
            isna_both = (isna_array(self.values, include_none=False) &
                    isna_array(other.values, include_none=False))
            eq[isna_both] = True

        if not eq.all():
            return False

        return self._index.equals(other._index,
                compare_name=compare_name,
                compare_dtype=compare_dtype,
                compare_class=compare_class,
                skipna=skipna,
                )

    #---------------------------------------------------------------------------
    # export

    def to_pairs(self) -> tp.Iterable[tp.Tuple[tp.Hashable, tp.Any]]:
        '''
        Return a tuple of tuples, where each inner tuple is a pair of index label, value.

        Returns:
            tp.Iterable[tp.Tuple[tp.Hashable, tp.Any]]
        '''
        if isinstance(self._index, IndexHierarchy):
            index_values = list(array2d_to_tuples(self._index.values))
        else:
            index_values = self._index.values

        return tuple(zip(index_values, self.values))



    def _to_frame(self,
            constructor: tp.Type['Frame'],
            axis: int = 1
            ) -> 'Frame':
        '''
        Common Frame construction utilities.
        '''
        from static_frame import TypeBlocks
        columns: tp.Optional[IndexInitializer]
        index: tp.Optional[IndexInitializer]

        if axis == 1:
            # present as a column
            def block_gen() -> tp.Iterator[np.ndarray]:
                yield self.values

            index = self._index
            own_index = True
            columns = None if self._name is None else (self._name,)
            own_columns = False
        elif axis == 0:
            def block_gen() -> tp.Iterator[np.ndarray]:
                yield self.values.reshape((1, self.values.shape[0]))

            index = None if self._name is None else (self._name,)
            own_index = False
            columns = self._index
            # if column constuctor is static, we can own the static index
            own_columns = constructor._COLUMNS_CONSTRUCTOR.STATIC
        else:
            raise NotImplementedError(f'no handling for axis {axis}')

        return constructor(
                TypeBlocks.from_blocks(block_gen()),
                index=index,
                columns=columns,
                own_data=True,
                own_index=own_index,
                own_columns=own_columns,
                )


    def to_frame(self, axis: int = 1) -> 'Frame':
        '''
        Return a :obj:`Frame` view of this :obj:`Series`. As underlying data is immutable, this is a no-copy operation.

        Returns:
            :obj:`Frame`
        '''
        from static_frame import Frame
        return self._to_frame(constructor=Frame, axis=axis)

    def to_frame_go(self, axis: int = 1) -> 'FrameGO':
        '''
        Return :obj:`FrameGO` view of this :obj:`Series`. As underlying data is immutable, this is a no-copy operation.

        Returns:
            :obj:`FrameGO`
        '''
        from static_frame import FrameGO
        return self._to_frame(constructor=FrameGO, axis=axis) #type: ignore

    def to_pandas(self) -> 'pandas.Series':
        '''
        Return a Pandas Series.

        Returns:
            :obj:`pandas.Series`
        '''
        import pandas
        return pandas.Series(self.values.copy(),
                index=self._index.to_pandas(),
                name=self._name)

    @doc_inject(class_name='Series')
    def to_html(self,
            config: tp.Optional[DisplayConfig] = None
            ) -> str:
        '''
        {}
        '''
        config = config or DisplayActive.get(type_show=False)
        config = config.to_display_config(
                display_format=DisplayFormats.HTML_TABLE,
                )
        return repr(self.display(config))

    @doc_inject(class_name='Series')
    def to_html_datatables(self,
            fp: tp.Optional[PathSpecifierOrFileLike] = None,
            show: bool = True,
            config: tp.Optional[DisplayConfig] = None
            ) -> tp.Optional[str]:
        '''
        {}
        '''
        config = config or DisplayActive.get(type_show=False)
        config = config.to_display_config(
                display_format=DisplayFormats.HTML_DATATABLES,
                )
        content = repr(self.display(config))
        # path_filter applied in call
        fp = write_optional_file(content=content, fp=fp)

        if show:
            assert isinstance(fp, str) #pragma: no cover
            import webbrowser #pragma: no cover
            webbrowser.open_new_tab(fp) #pragma: no cover
        return fp


#-------------------------------------------------------------------------------
class SeriesAssign(Assign):
    __slots__ = ('container', 'iloc_key')

    def __init__(self,
            container: Series,
            iloc_key: GetItemKeyType
            ) -> None:
        self.container = container
        self.iloc_key = iloc_key

    def __call__(self,
            value: tp.Any, # any possible assignment type
            fill_value: tp.Any = np.nan
            ) -> Series:
        '''
        Assign the ``value`` in the position specified by the selector. The `name` attribute is propagated to the returned container.

        Args:
            value:  Value to assign, which can be a :obj:`Series`, np.ndarray, or element.
            fill_value: If the ``value`` parameter has to be reindexed, this element will be used to fill newly created elements.
        '''
        if isinstance(value, Series):
            # instead of using fill_value here, might be better to use dtype_to_fill_value, so as to not coerce the type of the value to be assigned
            value = self.container._reindex_other_like_iloc(value,
                    self.iloc_key,
                    fill_value=fill_value).values

        if isinstance(value, np.ndarray):
            value_dtype = value.dtype
        elif hasattr(value, '__len__') and not isinstance(value, str):
            value, _ = iterable_to_array_1d(value)
            value_dtype = value.dtype
        else:
            value_dtype = dtype_from_element(value)

        dtype = resolve_dtype(self.container.dtype, value_dtype)

        # create or copy the array to return
        if dtype == self.container.dtype:
            array = self.container.values.copy()
        else:
            array = self.container.values.astype(dtype)

        array[self.iloc_key] = value
        array.flags.writeable = False

        return self.container.__class__(array,
                index=self.container._index,
                name=self.container._name)
