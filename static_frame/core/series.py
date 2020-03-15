import typing as tp
from functools import partial

import numpy as np

from numpy.ma import MaskedArray

from static_frame.core.util import DEFAULT_SORT_KIND
from static_frame.core.util import BOOL_TYPES
from static_frame.core.util import FLOAT_TYPES
from static_frame.core.util import EMPTY_TUPLE

from static_frame.core.util import GetItemKeyType
from static_frame.core.util import resolve_dtype
from static_frame.core.util import isna_array
from static_frame.core.util import iterable_to_array_1d
from static_frame.core.util import array_to_groups_and_locations
from static_frame.core.util import array_to_duplicated
from static_frame.core.util import full_for_fill
from static_frame.core.util import mloc
from static_frame.core.util import immutable_filter
from static_frame.core.util import name_filter
from static_frame.core.util import ufunc_axis_skipna
from static_frame.core.util import array2d_to_tuples
from static_frame.core.util import array_shift
from static_frame.core.util import write_optional_file
from static_frame.core.util import ufunc_unique
from static_frame.core.util import concat_resolved
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import binary_transition
from static_frame.core.util import isin
from static_frame.core.util import slices_from_targets
from static_frame.core.util import is_callable_or_mapping

from static_frame.core.util import AnyCallable
from static_frame.core.util import CallableOrMapping
from static_frame.core.util import SeriesInitializer
from static_frame.core.util import PathSpecifierOrFileLike
from static_frame.core.util import DepthLevelSpecifier

from static_frame.core.util import DtypeSpecifier
from static_frame.core.util import IndexInitializer
from static_frame.core.util import IndexConstructor
from static_frame.core.util import dtype_to_na

from static_frame.core.selector_node import InterfaceGetItem
from static_frame.core.selector_node import InterfaceSelection2D

from static_frame.core.util import argmin_1d
from static_frame.core.util import argmax_1d
from static_frame.core.util import intersect1d

from static_frame.core.index_correspondence import IndexCorrespondence
from static_frame.core.container import ContainerOperand

from static_frame.core.display import DisplayConfig
from static_frame.core.display import DisplayActive
from static_frame.core.display import Display
from static_frame.core.display import DisplayFormats
from static_frame.core.display import DisplayHeader

from static_frame.core.iter_node import IterNodeType
# from static_frame.core.iter_node import IterNode
from static_frame.core.iter_node import IterNodeGroup
from static_frame.core.iter_node import IterNodeDepthLevel
from static_frame.core.iter_node import IterNodeWindow
from static_frame.core.iter_node import IterNodeNoArg

from static_frame.core.iter_node import IterNodeApplyType

from static_frame.core.index import Index

# from static_frame.core.index_hierarchy import HLoc
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.index_base import IndexBase

from static_frame.core.container_util import index_from_optional_constructor
from static_frame.core.container_util import matmul
from static_frame.core.container_util import axis_window_items
from static_frame.core.container_util import rehierarch_and_map
from static_frame.core.container_util import pandas_version_under_1
from static_frame.core.container_util import pandas_to_numpy

from static_frame.core.index_auto import IndexAutoFactory
from static_frame.core.index_auto import IndexAutoFactoryType

from static_frame.core.exception import ErrorInitSeries
from static_frame.core.exception import AxisInvalid

from static_frame.core.doc_str import doc_inject

if tp.TYPE_CHECKING:
    from static_frame import Frame # pylint: disable=W0611 #pragma: no cover
    from pandas import DataFrame # pylint: disable=W0611 #pragma: no cover


RelabelInput = tp.Union[CallableOrMapping, IndexAutoFactoryType, IndexInitializer]


#-------------------------------------------------------------------------------
@doc_inject(selector='container_init', class_name='Series')
class Series(ContainerOperand):
    '''
    A one-dimensional ordered, labelled collection, immutable and of fixed size.

    Args:
        values: An iterable of values, or a single object, to be aligned with the supplied (or automatically generated) index. Alternatively, a dictionary of index / value pairs can be provided.
        {index}
        {own_index}
    '''

    __slots__ = (
            'values',
            '_index',
            '_name',
            )

    sum: tp.Callable[['Series'], tp.Any]
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
            name: tp.Hashable = None,
            index_constructor: IndexConstructor = None,
            own_index: bool = False,
            ):

        if own_index:
            index_final = index
        else:
            index_final = index_from_optional_constructor(index,
                    default_constructor=Index,
                    explicit_constructor=index_constructor
                    )
        array = np.full(
                len(index_final),
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
            name: tp.Hashable = None,
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
        def values():
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
            name: tp.Hashable = None,
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
            index: tp.Union[IndexInitializer, IndexAutoFactoryType] = None,
            name: tp.Hashable = None
            ) -> 'Series':
        '''
        Concatenate multiple :obj:`Series` into a new :obj:`Series`.

        Args:
            containers: Iterable of ``Series`` from which values in the new ``Series`` are drawn.
            index: If None, the resultant index will be the concatenation of all indices (assuming they are unique in combination). If ``IndexAutoFactory``, the resultant index is a auto-incremented integer index. Otherwise, the value is used as a index initializer.
        '''
        array_values = []
        if index is None:
            array_index = []
        for c in containers:
            array_values.append(c.values)
            if index is None:
                array_index.append(c.index.values)

        # End quickly if empty iterable
        if not array_values:
            return cls(EMPTY_TUPLE, index=index, name=name)

        # returns immutable arrays
        values = concat_resolved(array_values)

        if index is None:
            index = concat_resolved(array_index)
            if index.ndim == 2:
                index = IndexHierarchy.from_labels(index)
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
        '''
        array_values = []

        def gen():
            for label, series in items:
                array_values.append(series.values)
                yield label, series._index

        try:
            # populates array_values as side effect
            ih = IndexHierarchy.from_index_items(gen())
            # returns immutable array
            values = concat_resolved(array_values)
            own_index = True
        except StopIteration:
            # Default to empty when given an empty iterable
            ih = None
            values = EMPTY_TUPLE
            own_index= False

        return cls(values, index=ih, own_index=own_index)



    @classmethod
    @doc_inject()
    def from_pandas(cls,
            value,
            *,
            index_constructor: IndexConstructor = None,
            own_data: bool = False) -> 'Series':
        '''Given a Pandas Series, return a Series.

        Args:
            value: Pandas Series.
            {own_data}
            {own_index}

        Returns:
            :obj:`static_frame.Series`
        '''
        if pandas_version_under_1():
            if own_data:
                data = value.values
                data.flags.writeable = False
            else:
                data = immutable_filter(value.values)
        else:
            data = pandas_to_numpy(value, own_data=own_data)

        own_index = True
        if index_constructor is IndexAutoFactory:
            index = None
            own_index = False
        elif index_constructor is not None:
            index = index_constructor(value.index)
        else: # if None
            index = Index.from_pandas(value.index)

        return cls(data,
                index=index,
                name=value.name,
                own_index=own_index
                )


    #---------------------------------------------------------------------------
    def __init__(self,
            values: SeriesInitializer,
            *,
            index: tp.Union[IndexInitializer, IndexAutoFactoryType] = None,
            name: tp.Hashable = None,
            dtype: DtypeSpecifier = None,
            index_constructor: IndexConstructor = None,
            own_index: bool = False
            ) -> None:
        # doc string at class definition
        self._name = name if name is None else name_filter(name)

        if own_index and index is None:
            raise ErrorInitSeries('cannot own_index if no index is provided.')

        #-----------------------------------------------------------------------
        # values assignment

        values_constructor = None # if deferred

        if not isinstance(values, np.ndarray):
            if isinstance(values, dict):
                raise ErrorInitSeries('use Series.from_dict to create a Series from a mapping.')
            elif hasattr(values, '__iter__') and not isinstance(values, str):
                # returned array is already immutable
                self.values, _ = iterable_to_array_1d(values, dtype=dtype)
            else: # it must be an element, or a string
                raise ErrorInitSeries('Use Series.from_element to create a Series from an element.')

        else: # is numpy array
            if dtype is not None and dtype != values.dtype:
                raise ErrorInitSeries(f'when supplying values via array, the dtype argument is not required; if provided ({dtype}), it must agree with the dtype of the array ({values.dtype})')
            if values.shape == (): # handle special case of NP element
                def values_constructor(shape): #pylint: disable=E0102
                    self.values = np.repeat(values, shape)
                    self.values.flags.writeable = False
            else:
                self.values = immutable_filter(values)

        #-----------------------------------------------------------------------
        # index assignment

        if own_index:
            self._index = index
        elif index is None or index is IndexAutoFactory:
            # if a values constructor is defined, self.values is not yet defined, and we have a single element or string; if index is None or empty, we auto-supply a shape of 1; otherwise, take len of self.values
            if values_constructor:
                shape = 1
            else:
                shape = len(self.values)

            self._index = IndexAutoFactory.from_optional_constructor(
                    shape,
                    default_constructor=Index,
                    explicit_constructor=index_constructor
                    )
        else: # an iterable of labels, or an index subclass
            self._index = index_from_optional_constructor(index,
                    default_constructor=Index,
                    explicit_constructor=index_constructor
                    )

        if not self._index.STATIC:
            raise ErrorInitSeries('non-static index cannot be assigned to Series')

        shape = self._index.__len__()

        if values_constructor:
            values_constructor(shape) # updates self.values

        #-----------------------------------------------------------------------
        # final evaluation

        if self.values.ndim != self._NDIM:
            raise ErrorInitSeries('dimensionality of final values not supported')


    # ---------------------------------------------------------------------------
    def __reversed__(self) -> tp.Iterator[tp.Hashable]:
        '''
        Returns a reverse iterator on the series' index.
        '''
        return reversed(self._index)

    #---------------------------------------------------------------------------
    def __setstate__(self, state):
        '''
        Ensure that reanimated NP arrays are set not writeable.
        '''
        for key, value in state[1].items():
            setattr(self, key, value)
        self.values.flags.writeable = False

    #---------------------------------------------------------------------------
    # name interface

    @property
    @doc_inject()
    def name(self) -> tp.Hashable:
        '''{}'''
        return self._name

    def rename(self, name: tp.Hashable) -> 'Series':
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
    def loc(self) -> InterfaceGetItem:
        '''
        Interface for label-based selection.
        '''
        return InterfaceGetItem(self._extract_loc)

    @property
    def iloc(self) -> InterfaceGetItem:
        '''
        Interface for position-based selection.
        '''
        return InterfaceGetItem(self._extract_iloc)

    # NOTE: this could be ExtractInterfacd1D, but are consistent with what is done on the base name space: loc and getitem duplicate each other.

    @property
    def drop(self) -> InterfaceSelection2D:
        '''
        Interface for dropping elements from :obj:`static_frame.Series`.
        '''
        return InterfaceSelection2D(
                func_iloc=self._drop_iloc,
                func_loc=self._drop_loc,
                func_getitem=self._drop_loc
                )

    @property
    def mask(self) -> InterfaceSelection2D:
        '''
        Interface for extracting Boolean :obj:`static_frame.Series`.
        '''
        return InterfaceSelection2D(
                func_iloc=self._extract_iloc_mask,
                func_loc=self._extract_loc_mask,
                func_getitem=self._extract_loc_mask
                )

    @property
    def masked_array(self) -> InterfaceSelection2D:
        '''
        Interface for extracting NumPy Masked Arrays.
        '''
        return InterfaceSelection2D(
                func_iloc=self._extract_iloc_masked_array,
                func_loc=self._extract_loc_masked_array,
                func_getitem=self._extract_loc_masked_array
                )

    @property
    def assign(self) -> InterfaceSelection2D:
        '''
        Interface for doing assignment-like selection and replacement.
        '''
        return InterfaceSelection2D(
                func_iloc=self._extract_iloc_assign,
                func_loc=self._extract_loc_assign,
                func_getitem=self._extract_loc_assign
                )

    #---------------------------------------------------------------------------
    @property
    def iter_group(self) -> IterNodeGroup:
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
    def iter_group_items(self) -> IterNodeGroup:
        return IterNodeGroup(
                container=self,
                function_items=self._axis_group_items,
                function_values=self._axis_group,
                yield_type=IterNodeType.ITEMS
                )

    #---------------------------------------------------------------------------
    @property
    def iter_group_labels(self) -> IterNodeDepthLevel:
        return IterNodeDepthLevel(
                container=self,
                function_items=self._axis_group_labels_items,
                function_values=self._axis_group_labels,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.SERIES_ITEMS_FLAT
                )

    @property
    def iter_group_labels_items(self) -> IterNodeDepthLevel:
        return IterNodeDepthLevel(
                container=self,
                function_items=self._axis_group_labels_items,
                function_values=self._axis_group_labels,
                yield_type=IterNodeType.ITEMS,
                apply_type=IterNodeApplyType.SERIES_ITEMS_FLAT
                )

    #---------------------------------------------------------------------------
    @property
    def iter_element(self) -> IterNodeNoArg:
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
    def iter_element_items(self) -> IterNodeNoArg:
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
    def iter_window(self) -> IterNodeWindow:
        function_values = partial(self._axis_window, as_array=False)
        function_items = partial(self._axis_window_items, as_array=False)
        return IterNodeWindow(
                container=self,
                function_values=function_values,
                function_items=function_items,
                yield_type=IterNodeType.VALUES
                )

    @property
    def iter_window_items(self) -> IterNodeWindow:
        function_values = partial(self._axis_window, as_array=False)
        function_items = partial(self._axis_window_items, as_array=False)
        return IterNodeWindow(
                container=self,
                function_values=function_values,
                function_items=function_items,
                yield_type=IterNodeType.ITEMS
                )


    @property
    def iter_window_array(self) -> IterNodeWindow:
        function_values = partial(self._axis_window, as_array=True)
        function_items = partial(self._axis_window_items, as_array=True)
        return IterNodeWindow(
                container=self,
                function_values=function_values,
                function_items=function_items,
                yield_type=IterNodeType.VALUES
                )

    @property
    def iter_window_array_items(self) -> IterNodeWindow:
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
            fill_value=np.nan) -> 'Series':
        '''Given a value that is a Series, reindex it to the index components, drawn from this Series, that are specified by the iloc_key.
        '''
        return value.reindex(
                self._index._extract_iloc(iloc_key),
                fill_value=fill_value
                )

    @doc_inject(selector='reindex', class_name='Series')
    def reindex(self,
            index: IndexInitializer,
            fill_value=np.nan,
            own_index: bool = False
            ) -> 'Series':
        '''
        {doc}

        Args:
            index: {index_initializer}
            columns: {index_initializer}
            {fill_value}
            {own_index}
        '''
        if isinstance(index, IndexBase):
            if not own_index:
                # use the Index constructor for safe reuse when possible
                index = index.__class__(index)
        else: # create the Index if not already an index, assume 1D
            index = Index(index)

        ic = IndexCorrespondence.from_correspondence(self.index, index)

        if ic.is_subset: # must have some common
            return self.__class__(self.values[ic.iloc_src],
                    index=index,
                    own_index=True,
                    name=self._name)

        values = full_for_fill(self.values.dtype, len(index), fill_value)

        # if some intersection of values
        if ic.has_common:
            values[ic.iloc_dst] = self.values[ic.iloc_src]

        # make immutable so a copy is not made
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
        own_index = False
        if index is IndexAutoFactory:
            index = None
        elif is_callable_or_mapping(index):
            index = self._index.relabel(index)
            own_index = True
        elif index is None:
            index = self._index
        # else: # assume index IndexInitializer
        #     index = index

        return self.__class__(self.values,
                index=index,
                own_index=own_index,
                name=self._name)

    @doc_inject(selector='relabel_flat', class_name='Series')
    def relabel_flat(self) -> 'Series':
        '''
        {doc}
        '''
        return self.__class__(self.values,
                index=self._index.flat(),
                name=self._name)

    @doc_inject(selector='relabel_add_level', class_name='Series')
    def relabel_add_level(self,
            level: tp.Hashable
            ) -> 'Series':
        '''
        {doc}

        Args:
            level: {level}
        '''
        return self.__class__(self.values,
                index=self._index.add_level(level),
                name=self._name)

    @doc_inject(selector='relabel_drop_level', class_name='Series')
    def relabel_drop_level(self,
            count: int = 1
            ) -> 'Series':
        '''
        {doc}

        Args:
            count: {count}
        '''
        return self.__class__(self.values,
                index=self._index.drop_level(count),
                name=self._name)


    def rehierarch(self,
            depth_map: tp.Iterable[int]
            ) -> 'Series':
        '''
        Return a new :obj:`Series` with new a hierarchy based on the supplied ``depth_map``.
        '''
        if self.index.depth == 1:
            raise RuntimeError('cannot rehierarch when there is no hierarchy')

        # index, iloc_map = self.index._rehierarch_and_map(depth_map=depth_map)

        index, iloc_map = rehierarch_and_map(
                labels=self._index.values,
                depth_map=depth_map,
                index_constructor=self._index.from_labels,
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
        '''Return a new :obj:`static_frame.Series` after replacing null (NaN or None) with the supplied value. The value can be element or

        Args:
            {value}
        '''
        sel = isna_array(self.values)
        if not np.any(sel):
            return self

        if hasattr(value, '__iter__') and not isinstance(value, str):
            if not isinstance(value, Series):
                raise RuntimeError('unlabeled iterables cannot be used for fillna: use a Series')
            value_dtype = value.dtype
            # choose a fill value that will not force a type coercion
            fill_value = dtype_to_na(value_dtype)
            # find targets that are NaN in self and have labels in value; otherwise, might fill values after reindexing, and end up filling a fill_value rather than keeping original (na) value
            sel = self.index.isin(
                    intersect1d(self.index.values[sel], value.index.values))
            if not np.any(sel): # avoid copying, retyping
                return self

            # must reindex to align ordering; just get array
            value = self._reindex_other_like_iloc(value,
                    sel,
                    fill_value=fill_value).values
        else:
            value_dtype = np.array(value).dtype

        assignable_dtype = resolve_dtype(value_dtype, self.values.dtype)

        if self.values.dtype == assignable_dtype:
            assigned = self.values.copy()
        else:
            assigned = self.values.astype(assignable_dtype)

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
            return array # assume immutable

        def slice_condition(target_slice: slice) -> bool:
            # NOTE: start is never None
            return sel[target_slice.start]

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
            sided_leading: bool):
        '''
        Args:
            sided_leading: True sets the side to fill is the leading side; False sets the side to fill to the trailiing side.
        '''
        sel = isna_array(array)

        if not np.any(sel):
            return array

        sided_index = 0 if sided_leading else -1

        if sel[sided_index] == False:
            # sided value is not null: nothing to do
            return array # assume immutable

        if isinstance(value, np.ndarray):
            raise RuntimeError('cannot assign an array to fillna')

        assignable_dtype = resolve_dtype(np.array(value).dtype, array.dtype)

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

    def _ufunc_unary_operator(self, operator: tp.Callable) -> 'Series':
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
            operator: tp.Callable,
            other
            ) -> 'Series':
        '''
        For binary operations, the `name` attribute does not propagate.
        '''
        # get both reverse and regular
        if operator.__name__ == 'matmul':
            return matmul(self, other)
        elif operator.__name__ == 'rmatmul':
            return matmul(other, self)

        values = self.values
        index = self._index

        if isinstance(other, Series):
            # if indices are the same, we can simply set other to values and fallback on NP
            if len(self.index) != len(other.index) or (
                    self.index != other.index).any():
                index = self.index.union(other.index)
                # now need to reindex the Series
                values = self.reindex(index).values
                other = other.reindex(index).values
            else:
                other = other.values

        # if its an np array, we simply fall back on np behavior
        elif isinstance(other, np.ndarray):
            if other.ndim > 1:
                raise NotImplementedError('Operator application to greater dimensionalities will result in an array with more than 1 dimension; it is not clear how such an array should be indexed.')
        # permit single value constants; not sure about filtering other types

        # we want the dtype to be the result of applying the operator; this happends by default
        result = operator(values, other)

        if not isinstance(result, np.ndarray):
            # in comparison to Booleans, if values is of length 1 and a character type, we will get a Boolean back, not an array; this issues the following warning: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
            if isinstance(result, BOOL_TYPES):
                # return a Boolean at the same size as the original Series; this works, but means that we will mask that, if the arguement is a tuple of length equal to an erray, NP will perform element wise comparison; but if the argment is a tuple of length greater or equal, each value in value will be compared to that tuple
                result = np.full(len(values), result)
            else:
                raise RuntimeError('unexpected branch from non-array result of operator application to array') #pragma: no cover

        result.flags.writeable = False
        return self.__class__(result, index=index)

    def _ufunc_axis_skipna(self, *,
            axis: int,
            skipna: bool,
            ufunc,
            ufunc_skipna,
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
            ufunc,
            ufunc_skipna,
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
        return self.values.__len__()

    @doc_inject()
    def display(self,
            config: tp.Optional[DisplayConfig] = None
            ) -> Display:
        '''{doc}

        Args:
            {config}
        '''
        config = config or DisplayActive.get()

        d = Display([],
                config=config,
                outermost=True,
                index_depth=1,
                header_depth=2) # series and index header

        display_index = self._index.display(config=config)
        d.extend_display(display_index)

        d.extend_display(Display.from_values(
                self.values,
                header='',
                config=config))

        display_cls = Display.from_values((),
                header=DisplayHeader(self.__class__, self._name),
                config=config)
        d.insert_displays(display_cls.flatten())
        return d

    def _repr_html_(self):
        '''
        Provide HTML representation for Jupyter Notebooks.
        '''
        # modify the active display to be fore HTML
        config = DisplayActive.get(
                display_format=DisplayFormats.HTML_TABLE,
                type_show=False
                )
        return repr(self.display(config))

    #---------------------------------------------------------------------------
    # common attributes from the numpy array

    @property
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
            :obj:`tp.Tuple[int]`
        '''
        return self.values.shape

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
        return self.values.size

    @property
    def nbytes(self) -> int:
        '''
        Return the total bytes of the underlying NumPy array.

        Returns:
            :obj:`int`
        '''
        return self.values.nbytes

    def __bool__(self) -> bool:
        '''
        True if this container has size.
        '''
        return bool(self.values.size)


    #---------------------------------------------------------------------------
    # extraction

    def _extract_iloc(self, key: GetItemKeyType) -> 'Series':
        # iterable selection should be handled by NP
        values = self.values[key]

        if not isinstance(values, np.ndarray): # if we have a single element
            return values
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
            return values

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
            # use Boolean area to select indices from Index positions, as np.delete does not work with arrays
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
            ):
        if axis != 0:
            raise AxisInvalid(f'invalid axis {axis}')

        groups, locations = array_to_groups_and_locations(self.values)
        for idx, g in enumerate(groups):
            selection = locations == idx
            yield g, self._extract_iloc(selection)

    def _axis_group(self, *,
            axis: int = 0
            ):
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
            ):

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
            ):
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
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, tp.Any]]:
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
            ):
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
    def index(self):
        '''
        The ``IndexBase`` instance assigned for labels.
        '''
        return self._index

    #---------------------------------------------------------------------------
    # dictionary-like interface

    def keys(self) -> Index:
        '''
        Iterator of index labels.
        '''
        return self._index

    def __iter__(self):
        '''
        Iterator of index labels, same as :py:meth:`Series.keys`.
        '''
        return self._index.__iter__()

    def __contains__(self, value) -> bool:
        '''
        Inclusion of value in index labels.
        '''
        return self._index.__contains__(value)

    def items(self) -> tp.Iterator[tp.Tuple[tp.Any, tp.Any]]:
        '''Iterator of pairs of index label and value.
        '''
        return zip(self._index.values, self.values)

    def get(self, key: tp.Hashable, default=None) -> tp.Any:
        '''
        Return the value found at the index key, else the default if the key is not found.
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

    def isin(self, other) -> 'Series':
        '''
        Return a same-sized Boolean Series that shows if the same-positioned element is in the iterable passed to the function.
        '''
        array = isin(self.values, other)
        return self.__class__(array, index=self._index, name=self._name)

    @doc_inject(class_name='Series')
    def clip(self, *,
            lower=None,
            upper=None
            ):
        '''{}

        Args:
            lower: value or ``Series`` to define the inclusive lower bound.
            upper: value or ``Series`` to define the inclusive upper bound.
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
        '''
        return self

    @property
    def T(self):
        '''Transpose. For a 1D immutable container, this returns a reference to self.
        '''
        return self.transpose()

    @doc_inject(selector='duplicated')
    def duplicated(self, *,
            exclude_first=False,
            exclude_last=False) -> np.ndarray:
        '''
        Return a same-sized Boolean Series that shows True for all b values that are duplicated.

        Args:
            {exclude_first}
            {exclude_last}
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
        '''
        return self.__class__(
                self.values.astype(dtype),
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
            fill_value=np.nan) -> 'Series':
        '''Return a Series with values shifted forward on the index (with a postive shift) or backward on the index (with a negative shift).

        Args:
            shift: Postive or negative integer shift.
            fill_value: Value to be used to fill data missing after the shift.
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
    # transformations resulting in reduced dimensionality
    @doc_inject(selector='head', class_name='Series')
    def head(self, count: int = 5) -> 'Series':
        '''{doc}

        Args:
            {count}
        '''
        return self.iloc[:count]

    @doc_inject(selector='tail', class_name='Series')
    def tail(self, count: int = 5) -> 'Series':
        '''{doc}

        Args:
            {count}
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
        '''
        return argmin_1d(self.values, skipna=skipna)

    @doc_inject(selector='argminmax')
    def loc_max(self, *,
            skipna: bool = True,
            ) -> tp.Hashable:
        '''
        Return the label corresponding to the maximum value found.

        Args:
            {skipna}
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
        '''
        return argmax_1d(self.values, skipna=skipna)


    #---------------------------------------------------------------------------
    # utility function to numpy array

    def unique(self) -> np.ndarray:
        '''
        Return a NumPy array of unqiue values.
        '''
        return ufunc_unique(self.values)

    #---------------------------------------------------------------------------
    # export

    def to_pairs(self) -> tp.Iterable[tp.Tuple[tp.Hashable, tp.Any]]:
        '''
        Return a tuple of tuples, where each inner tuple is a pair of index label, value.
        '''
        if isinstance(self._index, IndexHierarchy):
            index_values = list(array2d_to_tuples(self._index.values))
        else:
            index_values = self._index.values

        return tuple(zip(index_values, self.values))



    def _to_frame(self, constructor, axis: int = 1):
        '''
        Common Frame construction utilities.
        '''
        from static_frame import TypeBlocks

        if axis == 1:
            # present as a column
            def block_gen():
                yield self.values

            index = self._index
            own_index = True
            columns = None if self._name is None else (self._name,)
            own_columns = False
        elif axis == 0:
            def block_gen():
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


    def to_frame(self, axis: int = 1):
        '''
        Return a :obj:`static_frame.Frame` view of this :obj:`static_frame.Series`. As underlying data is immutable, this is a no-copy operation.
        '''
        from static_frame import Frame
        return self._to_frame(constructor=Frame, axis=axis)

    def to_frame_go(self, axis: int = 1):
        '''
        Return :obj:`static_frame.FrameGO` view of this :obj:`static_frame.Series`. As underlying data is immutable, this is a no-copy operation.
        '''
        from static_frame import FrameGO
        return self._to_frame(constructor=FrameGO, axis=axis)

    def to_pandas(self) -> 'DataFrame':
        '''
        Return a Pandas Series.
        '''
        import pandas
        return pandas.Series(self.values.copy(),
                index=self._index.to_pandas(),
                name=self._name)

    @doc_inject(class_name='Series')
    def to_html(self,
            config: tp.Optional[DisplayConfig] = None
            ):
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
            ) -> str:
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
            import webbrowser #pragma: no cover
            webbrowser.open_new_tab(fp) #pragma: no cover
        return fp


#-------------------------------------------------------------------------------
class SeriesAssign:
    __slots__ = ('container', 'iloc_key')

    def __init__(self,
            container: Series,
            iloc_key: GetItemKeyType
            ) -> None:
        self.container = container
        self.iloc_key = iloc_key

    def __call__(self,
            value, # any possible assignment type
            fill_value=np.nan
            ):
        '''
        Calling with a value performs the assignment. The `name` attribute is propagated.
        '''
        if isinstance(value, Series):
            # instead of using fill_value here, might be better to use dtype_to_na, so as to not coerce the type of the value to be assigned
            value = self.container._reindex_other_like_iloc(value,
                    self.iloc_key,
                    fill_value=fill_value).values

        if isinstance(value, np.ndarray):
            value_dtype = value.dtype
        else:
            value_dtype = np.array(value).dtype

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
