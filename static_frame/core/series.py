import typing as tp


import numpy as np
from numpy.ma import MaskedArray

from static_frame.core.util import DEFAULT_SORT_KIND
from static_frame.core.util import _BOOL_TYPES
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import _resolve_dtype
from static_frame.core.util import _isna
from static_frame.core.util import iterable_to_array
from static_frame.core.util import _array_to_groups_and_locations
from static_frame.core.util import _array_to_duplicated
from static_frame.core.util import resolve_dtype_iter
from static_frame.core.util import full_for_fill
from static_frame.core.util import mloc
from static_frame.core.util import immutable_filter
from static_frame.core.util import name_filter
from static_frame.core.util import ufunc_skipna_1d
from static_frame.core.util import _dict_to_sorted_items
from static_frame.core.util import array2d_to_tuples
from static_frame.core.util import array_shift
from static_frame.core.util import write_optional_file
from static_frame.core.util import ufunc_unique
from static_frame.core.util import concat_resolved

from static_frame.core.util import CallableOrMapping
from static_frame.core.util import SeriesInitializer
from static_frame.core.util import FilePathOrFileLike
from static_frame.core.util import DepthLevelSpecifier

from static_frame.core.util import DtypeSpecifier
from static_frame.core.util import IndexInitializer
from static_frame.core.util import STATIC_ATTR

from static_frame.core.util import GetItem
from static_frame.core.util import InterfaceSelection2D
from static_frame.core.util import IndexCorrespondence

from static_frame.core.operator_delegate import MetaOperatorDelegate

from static_frame.core.display import DisplayConfig
from static_frame.core.display import DisplayActive
from static_frame.core.display import Display
from static_frame.core.display import DisplayFormats
from static_frame.core.display import DisplayHeader

from static_frame.core.iter_node import IterNodeType
from static_frame.core.iter_node import IterNode

from static_frame.core.index import Index
from static_frame.core.index_hierarchy import HLoc
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.index_base import IndexBase

from static_frame.core.doc_str import doc_inject

if tp.TYPE_CHECKING:
    from static_frame import Frame

#-------------------------------------------------------------------------------
@doc_inject(selector='container_init', class_name='Series')
class Series(metaclass=MetaOperatorDelegate):
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

    @classmethod
    def from_items(cls,
            pairs: tp.Iterable[tp.Tuple[tp.Hashable, tp.Any]],
            *,
            dtype: DtypeSpecifier = None,
            name: tp.Hashable = None
            ) -> 'Series':
        '''Series construction from an iterator or generator of pairs, where the first pair value is the index and the second is the value.

        Args:
            pairs: Iterable of pairs of index, value.
            dtype: dtype or valid dtype specifier.

        Returns:
            :py:class:`static_frame.Series`
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
                name=name)

    @classmethod
    def from_concat(cls,
            containers: tp.Iterable['Series'],
            *,
            name: tp.Hashable = None
            ):
        '''
        Concatenate multiple Series into a new Series, assuming the combination of all Indices result in a unique Index.
        '''
        array_values = []
        array_index = []
        for c in containers:
            array_values.append(c.values)
            array_index.append(c.index.values)

        # returns immutable arrays
        values = concat_resolved(array_values)
        index = concat_resolved(array_index)

        if index.ndim == 2:
            index = IndexHierarchy.from_labels(index)

        return cls(values, index=index, name=name)


    @classmethod
    @doc_inject()
    def from_pandas(cls,
            value,
            *,
            own_data: bool = False) -> 'Series':
        '''Given a Pandas Series, return a Series.

        Args:
            value: Pandas Series.
            {own_data}
            {own_index}

        Returns:
            :py:class:`static_frame.Series`
        '''
        if own_data:
            data = value.values
            data.flags.writeable = False
        else:
            data = immutable_filter(value.values)

        return cls(data,
                index=IndexBase.from_pandas(value.index),
                name=value.name,
                own_index=True
                )

    def __init__(self,
            values: SeriesInitializer,
            *,
            index: IndexInitializer = None,
            name: tp.Hashable = None,
            dtype: DtypeSpecifier = None,
            own_index: bool = False
            ) -> None:

        # TODO: support construction from another Series, propagate name attr
        self._name = name if name is None else name_filter(name)

        #-----------------------------------------------------------------------
        # values assignment

        values_constructor = None # if deferred

        if not isinstance(values, np.ndarray):
            if isinstance(values, dict):
                # not sure if we should sort; not sure what to do if index is provided
                if index is not None:
                    raise Exception('cannot create a Series from a dictionary when an index is defined')
                index = []
                def values_gen():
                    for k, v in _dict_to_sorted_items(values):
                        # populate index as side effect of iterating values
                        index.append(k)
                        yield v
                if dtype and dtype != object:
                    # fromiter does not work with object types
                    self.values = np.fromiter(values_gen(),
                            dtype=dtype,
                            count=len(values))
                else:
                    self.values = np.array(tuple(values_gen()), dtype=dtype)
                self.values.flags.writeable = False

            # NOTE: not sure if we need to check __iter__ here
            elif (dtype and dtype != object and dtype != str
                    and hasattr(values, '__iter__')
                    and hasattr(values, '__len__')):
                self.values = np.fromiter(values, dtype=dtype, count=len(values))
                self.values.flags.writeable = False
            elif hasattr(values, '__len__') and not isinstance(values, str):
                self.values = np.array(values, dtype=dtype)
                self.values.flags.writeable = False
            elif hasattr(values, '__next__'): # a generator-like
                self.values = np.array(tuple(values), dtype=dtype)
                self.values.flags.writeable = False
            else: # it must be a single item
                # we cannot create the values until we realize the index, which might be hierarchical and not have final size equal to length
                def values_constructor(shape):
                    self.values = np.full(shape, values, dtype=dtype)
                    self.values.flags.writeable = False
        else: # is numpy
            if dtype is not None and dtype != values.dtype:
                raise Exception('when supplying values via array, the dtype argument is not required; if provided, it must agree with the dtype of the array')
            if values.shape == (): # handle special case of NP element
                def values_constructor(shape):
                    self.values = np.repeat(values, shape)
                    self.values.flags.writeable = False
            else:
                self.values = immutable_filter(values)

        #-----------------------------------------------------------------------
        # index assignment
        # NOTE: this generally must be done after values assignment, as from_items (for example) needs the values generator to be exhausted before looking to index

        if index is None or (hasattr(index, '__len__') and len(index) == 0):
            # create an integer index; we specify dtype for windows
            self._index = Index(range(len(self.values)),
                    loc_is_iloc=True,
                    dtype=np.int64)
        elif own_index:
            self._index = index
        elif hasattr(index, STATIC_ATTR):
            if index.STATIC:
                self._index = index
            else:
                raise RuntimeError('non-static index cannot be assigned to Series')
        else: # let index handle instantiation
            if isinstance(index, (Index, IndexHierarchy)):
                # call with the class of the passed-in index, in case it is hierarchical
                self._index = index.__class__(index)
            else:
                self._index = Index(index)

        shape = self._index.__len__()

        if values_constructor:
            values_constructor(shape) # updates self.values

        if len(self.values) != shape:
            raise RuntimeError('values and index do not match length')

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
    def name(self) -> tp.Hashable:
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
    def loc(self) -> GetItem:
        return GetItem(self._extract_loc)

    @property
    def iloc(self) -> GetItem:
        return GetItem(self._extract_iloc)

    # NOTE: this could be ExtractInterfacd1D, but are consistent with what is done on the base name space: loc and getitem duplicate each other.

    @property
    def drop(self) -> InterfaceSelection2D:
        return InterfaceSelection2D(
                func_iloc=self._drop_iloc,
                func_loc=self._drop_loc,
                func_getitem=self._drop_loc
                )

    @property
    def mask(self) -> InterfaceSelection2D:
        return InterfaceSelection2D(
                func_iloc=self._extract_iloc_mask,
                func_loc=self._extract_loc_mask,
                func_getitem=self._extract_loc_mask
                )

    @property
    def masked_array(self) -> InterfaceSelection2D:
        return InterfaceSelection2D(
                func_iloc=self._extract_iloc_masked_array,
                func_loc=self._extract_loc_masked_array,
                func_getitem=self._extract_loc_masked_array
                )

    @property
    def assign(self) -> InterfaceSelection2D:
        return InterfaceSelection2D(
                func_iloc=self._extract_iloc_assign,
                func_loc=self._extract_loc_assign,
                func_getitem=self._extract_loc_assign
                )

    @property
    def iter_group(self) -> IterNode:
        return IterNode(
                container=self,
                function_items=self._axis_group_items,
                function_values=self._axis_group,
                yield_type=IterNodeType.VALUES
                )

    @property
    def iter_group_items(self) -> IterNode:
        return IterNode(
                container=self,
                function_items=self._axis_group_items,
                function_values=self._axis_group,
                yield_type=IterNodeType.ITEMS
                )


    @property
    def iter_group_index(self) -> IterNode:
        return IterNode(
                container=self,
                function_items=self._axis_group_index_items,
                function_values=self._axis_group_index,
                yield_type=IterNodeType.VALUES
                )

    @property
    def iter_group_index_items(self) -> IterNode:
        return IterNode(
                container=self,
                function_items=self._axis_group_index_items,
                function_values=self._axis_group_index,
                yield_type=IterNodeType.ITEMS
                )


    @property
    def iter_element(self) -> IterNode:
        return IterNode(
                container=self,
                function_items=self._axis_element_items,
                function_values=self._axis_element,
                yield_type=IterNodeType.VALUES
                )

    @property
    def iter_element_items(self) -> IterNode:
        return IterNode(
                container=self,
                function_items=self._axis_element_items,
                function_values=self._axis_element,
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
        return value.reindex(self._index._extract_iloc(iloc_key), fill_value=fill_value)

    def reindex(self,
            index: tp.Union[Index, tp.Sequence[tp.Any]],
            fill_value=np.nan) -> 'Series':
        '''
        Return a new Series based on the passed index.

        Args:
            fill_value: attempted to be used, but may be coerced by the dtype of this Series. `
        '''
        if isinstance(index, (Index, IndexHierarchy)):
            # always use the Index constructor for safe reuse when possible
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

    def relabel(self, mapper: CallableOrMapping) -> 'Series':
        '''
        Return a new Series based on a mapping (or callable) from old to new index values.
        '''
        return self.__class__(self.values,
                index=self._index.relabel(mapper),
                own_index=True,
                name=self._name)

    def reindex_flat(self):
        '''
        Return a new Series, where a ``IndexHierarchy`` (if deifined) is replaced with a flat, one-dimension index of tuples.
        '''
        return self.__class__(self.values,
                index=self._index.flat(),
                name=self._name)

    def reindex_add_level(self, level: tp.Hashable):
        '''
        Return a new Series, adding a new root level to an ``IndexHierarchy``.
        '''
        return self.__class__(self.values,
                index=self._index.add_level(level),
                name=self._name)

    @doc_inject(selector='reindex')
    def reindex_drop_level(self, count: int = 1):
        '''
        Return a new Series, dropping one or more levels from an ``IndexHierarchy``. {count}
        '''
        return self.__class__(self.values,
                index=self._index.drop_level(count),
                name=self._name)


    #---------------------------------------------------------------------------
    # na handling

    def isna(self) -> 'Series':
        '''
        Return a same-indexed, Boolean Series indicating which values are NaN or None.
        '''
        # consider returning self if not values.any()?
        values = _isna(self.values)
        values.flags.writeable = False
        return self.__class__(values, index=self._index)

    def notna(self) -> 'Series':
        '''
        Return a same-indexed, Boolean Series indicating which values are NaN or None.
        '''
        values = np.logical_not(_isna(self.values))
        values.flags.writeable = False
        return self.__class__(values, index=self._index)

    def dropna(self) -> 'Series':
        '''
        Return a new Series after removing values of NaN or None.
        '''
        # get positions that we want to keep
        sel = np.logical_not(_isna(self.values))
        if not np.any(sel):
            return self.__class__(())

        values = self.values[sel]
        values.flags.writeable = False

        return self.__class__(values,
                index=self._index.loc[sel],
                name=self._name)

    def fillna(self, value) -> 'Series':
        '''Return a new Series after replacing NaN or None values with the supplied value.
        '''
        sel = _isna(self.values)
        if not np.any(sel):
            return self

        if isinstance(value, np.ndarray):
            raise Exception('cannot assign an array to fillna')

        value_dtype = np.array(value).dtype
        assigned_dtype = _resolve_dtype(value_dtype, self.values.dtype)

        if self.values.dtype == assigned_dtype:
            assigned = self.values.copy()
        else:
            assigned = self.values.astype(assigned_dtype)

        assigned[sel] = value
        assigned.flags.writeable = False

        return self.__class__(assigned,
                index=self._index,
                name=self._name)

    #---------------------------------------------------------------------------
    # operators

    def _ufunc_unary_operator(self, operator: tp.Callable) -> 'Series':
        '''
        For unary operations, the `name` attribute propagates.
        '''
        return self.__class__(operator(self.values),
                index=self._index,
                dtype=self.dtype,
                name=self._name)

    def _ufunc_binary_operator(self, *, operator: tp.Callable, other) -> 'Series':
        '''
        For binary operations, the `name` attribute does not propagate.
        '''
        values = self.values
        index = self._index

        if isinstance(other, Series):
            # if indices are the same, we can simply set other to values and fallback on NP
            if len(self.index) != len(other.index) or (self.index != other.index).any():
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
            if isinstance(result, _BOOL_TYPES):
                # return a Boolean at the same size as the original Series; this works, but means that we will mask that, if the arguement is a tuple of length equalt to an erray, NP will perform element wise comparison; bit if the arguemtn is a tuple of length greater or eqial, each value in value will be compared to that tuple
                result = np.full(len(values), result)
            else:
                raise Exception('unexpected branch from non-array result of operator application to array')

        result.flags.writeable = False
        return self.__class__(result, index=index)

    def _ufunc_axis_skipna(self, *, axis, skipna, ufunc, ufunc_skipna, dtype=None):
        '''For a Series, all functions of this type reduce the single axis of the Series to 1d, so Index has no use here.

        Args:
            dtype: not used, part of signature for a commin interface
        '''
        return ufunc_skipna_1d(
                array=self.values,
                skipna=skipna,
                ufunc=ufunc,
                ufunc_skipna=ufunc_skipna)

    #---------------------------------------------------------------------------
    def __len__(self) -> int:
        '''Length of values.
        '''
        return self.values.__len__()

    def display(self,
            config: tp.Optional[DisplayConfig] = None
            ) -> Display:
        '''Return a Display of the Series.
        '''
        config = config or DisplayActive.get()

        d = Display([],
                config=config,
                outermost=True,
                index_depth=1,
                columns_depth=2) # series and index header

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

    def __repr__(self):
        return repr(self.display())

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
    def mloc(self):
        return mloc(self.values)

    @property
    def dtype(self) -> np.dtype:
        '''
        Return the dtype of the underlying NumPy array.

        Returns:
            :py:class:`numpy.dtype`
        '''
        return self.values.dtype

    @property
    def shape(self) -> tp.Tuple[int]:
        '''
        Return a tuple describing the shape of the underlying NumPy array.

        Returns:
            :py:class:`tp.Tuple[int]`
        '''
        return self.values.shape

    @property
    def ndim(self) -> int:
        '''
        Return the number of dimensions, which for a `Series` is always 1.

        Returns:
            :py:class:`int`
        '''
        return self.values.ndim

    @property
    def size(self) -> int:
        '''
        Return the size of the underlying NumPy array.

        Returns:
            :py:class:`int`
        '''
        return self.values.size

    @property
    def nbytes(self) -> int:
        '''
        Return the total bytes of the underlying NumPy array.

        Returns:
            :py:class:`int`
        '''
        return self.values.nbytes

    #---------------------------------------------------------------------------
    # extraction

    def _extract_iloc(self, key: GetItemKeyType) -> 'Series':
        # iterable selection should be handled by NP (but maybe not if a tuple)
        values = self.values[key]
        if not isinstance(values, np.ndarray): # if we have a single element
            return values
        return self.__class__(
                self.values[key],
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
            if isinstance(key, HLoc) and key.has_key_multiple():
                # must return a Series, even though we do not have an array
                values = np.array(values)
                values.flags.writeable = False
            else:
                return values

        return self.__class__(values,
                index=self._index.iloc[iloc_key],
                own_index=True,
                name=self._name)

    def __getitem__(self, key: GetItemKeyType) -> 'Series':
        '''A Loc selection (by index labels).

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

    def _axis_group_items(self, *, axis=0):
        groups, locations = _array_to_groups_and_locations(self.values)
        for idx, g in enumerate(groups):
            selection = locations == idx
            yield g, self._extract_iloc(selection)

    def _axis_group(self, *, axis=0):
        yield from (x for _, x in self._axis_group_items(axis=axis))

    def _axis_element_items(self, *, axis=0):
        '''Generator of index, value pairs, equivalent to Series.items(). Rpeated to have a common signature as other axis functions.
        '''
        return zip(self._index.values, self.values)

    def _axis_element(self, *, axis=0):
        yield from (x for _, x in self._axis_element_items(axis=axis))


    def _axis_group_index_items(self,
            depth_level: DepthLevelSpecifier = 0,
            ):

        values = self.index.values_at_depth(depth_level)
        group_to_tuple = values.ndim == 2

        groups, locations = _array_to_groups_and_locations(
                values)

        for idx, g in enumerate(groups):
            selection = locations == idx
            if group_to_tuple:
                g = tuple(g)
            yield g, self._extract_iloc(selection)

    def _axis_group_index(self,
            depth_level: DepthLevelSpecifier = 0,
            ):
        yield from (x for _, x in self._axis_group_index_items(
                depth_level=depth_level))


    #---------------------------------------------------------------------------

    @property
    def index(self):
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

    def items(self) -> tp.Generator[tp.Tuple[tp.Any, tp.Any], None, None]:
        '''Iterator of pairs of index label and value.
        '''
        return zip(self._index.values, self.values)

    def get(self, key, default=None):
        '''
        Return the value found at the index key, else the default if the key is not found.
        '''
        if key not in self._index:
            return default
        return self.__getitem__(key)

    #---------------------------------------------------------------------------
    # transformations resulting in the same dimensionality

    def sort_index(self,
            ascending: bool = True,
            kind: str = DEFAULT_SORT_KIND) -> 'Series':
        '''
        Return a new Series ordered by the sorted Index.
        '''
        # argsort lets us do the sort once and reuse the results
        order = np.argsort(self._index.values, kind=kind)
        if not ascending:
            order = order[::-1]

        index_values = self._index.values[order]
        index_values.flags.writeable = False
        values = self.values[order]
        values.flags.writeable = False
        return self.__class__(values, index=index_values)

    def sort_values(self,
            ascending: bool = True,
            kind: str = DEFAULT_SORT_KIND) -> 'Series':
        '''
        Return a new Series ordered by the sorted values.
        '''
        # argsort lets us do the sort once and reuse the results
        order = np.argsort(self.values, kind=kind)
        if not ascending:
            order = order[::-1]

        index_values = self._index.values[order]
        index_values.flags.writeable = False
        values = self.values[order]
        values.flags.writeable = False
        return self.__class__(values, index=index_values)


    def isin(self, other) -> 'Series':
        '''
        Return a same-sized Boolean Series that shows if the same-positoined element is in the iterable passed to the function.
        '''
        # cannot use assume_unique because do not know if values is unique
        v, _ = iterable_to_array(other)
        # NOTE: could identify empty iterable and create False array
        array = np.in1d(self.values, v)
        array.flags.writeable = False
        return self.__class__(array, index=self._index)

    @doc_inject(class_name='Series')
    def clip(self, lower=None, upper=None):
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
        return self.__class__(array, index=self._index)

    def transpose(self) -> 'Series':
        '''The transpositon of a Series is itself.
        '''
        return self

    @property
    def T(self):
        return self.transpose()


    def duplicated(self,
            exclude_first=False,
            exclude_last=False) -> np.ndarray:
        '''
        Return a same-sized Boolean Series that shows True for all b values that are duplicated.
        '''
        # TODO: might be able to do this witnout calling .values and passing in TypeBlocks, but TB needs to support roll
        duplicates = _array_to_duplicated(self.values,
                exclude_first=exclude_first,
                exclude_last=exclude_last)
        duplicates.flags.writeable = False
        return self.__class__(duplicates, index=self._index)

    def drop_duplicated(self,
            exclude_first=False,
            exclude_last=False
            ):
        '''
        Return a Series with duplicated values removed.
        '''
        duplicates = _array_to_duplicated(self.values,
                exclude_first=exclude_first,
                exclude_last=exclude_last)
        keep = ~duplicates
        return self.__class__(self.values[keep], index=self._index[keep])

    def astype(self, dtype: DtypeSpecifier) -> 'Series':
        '''
        Return a Series with type determined by `dtype` argument. Note that for Series, this is a simple function, whereas for Frame, this is an interface exposing both a callable and a getitem interface.
        '''
        return self.__class__(
                self.values.astype(dtype),
                index=self._index,
                name=self._name
                )


    def roll(self,
            shift: int,
            include_index: bool = False) -> 'Series':
        '''Return a Series with values rotated forward and wrapped around the index (with a postive shift) or backward and wrapped around the index (with a negative shift).

        Args:
            shift: Postive or negative integer shift.
            include_index: Determine if the Index is shifted with the underlying data.
        '''
        if shift % len(self.values):
            values = array_shift(self.values,
                    shift,
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
            fill_value=np.nan) -> 'Series':
        '''Return a Series with values shifted forward on the index (with a postive shift) or backward on the index (with a negative shift).

        Args:
            shift: Postive or negative integer shift.
            fill_value: Value to be used to fill data missing after the shift.
        '''

        if shift:
            values = array_shift(self.values,
                    shift,
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

    def head(self, count: int = 5) -> 'Series':
        '''Return a Series consisting only of the top elements as specified by ``count``.

        Args:
            count: Number of elements to be returned from the top of the Series.
        '''
        return self.iloc[:count]

    def tail(self, count: int = 5) -> 'Series':
        '''Return a Series consisting only of the bottom elements as specified by ``count``.

        Args:
            count: Number of elements to be returned from the bottom of the Series.
        '''
        return self.iloc[-count:]


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
            own_columns = True # index is immutable
        else:
            raise NotImplementedError('no handling for axis', axis)

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
        Return a :py:class:`static_frame.Frame` view of this :py:class:`static_frame.Series`. As underlying data is immutable, this is a no-copy operation.
        '''
        from static_frame import Frame
        return self._to_frame(constructor=Frame, axis=axis)

    def to_frame_go(self, axis: int = 1):
        '''
        Return :py:class:`static_frame.FrameGO` view of this :py:class:`static_frame.Series`. As underlying data is immutable, this is a no-copy operation.
        '''
        from static_frame import FrameGO
        return self._to_frame(constructor=FrameGO, axis=axis)

    def to_pandas(self):
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
            fp: tp.Optional[FilePathOrFileLike] = None,
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
        fp = write_optional_file(content=content, fp=fp)

        if show:
            import webbrowser
            webbrowser.open_new_tab(fp)
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
            fill_value=np.nan):
        '''
        Calling with a value performs the assignment. The `name` attribute is propagated.
        '''
        if isinstance(value, Series):
            value = self.container._reindex_other_like_iloc(value,
                    self.iloc_key,
                    fill_value=fill_value).values

        if isinstance(value, np.ndarray):
            value_dtype = value.dtype
        else:
            value_dtype = np.array(value).dtype

        dtype = _resolve_dtype(self.container.dtype, value_dtype)

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



