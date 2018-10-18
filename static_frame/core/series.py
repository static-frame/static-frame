import typing as tp


import numpy as np
from numpy.ma import MaskedArray

from static_frame.core.util import _DEFAULT_SORT_KIND
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import _resolve_dtype
from static_frame.core.util import _isna
from static_frame.core.util import _iterable_to_array
from static_frame.core.util import _array_to_groups_and_locations
from static_frame.core.util import _array_to_duplicated
from static_frame.core.util import _full_for_fill
from static_frame.core.util import mloc
from static_frame.core.util import immutable_filter
from static_frame.core.util import _ufunc_skipna_1d
from static_frame.core.util import _dict_to_sorted_items
from static_frame.core.util import _array2d_to_tuples


from static_frame.core.util import CallableOrMapping
from static_frame.core.util import SeriesInitializer

from static_frame.core.util import DtypeSpecifier
from static_frame.core.util import IndexInitializer

from static_frame.core.util import GetItem
from static_frame.core.util import ExtractInterface
from static_frame.core.util import IndexCorrespondence

from static_frame.core.operator_delegate import MetaOperatorDelegate

from static_frame.core.display import DisplayConfig
from static_frame.core.display import DisplayActive
from static_frame.core.display import Display

from static_frame.core.iter_node import IterNodeType
from static_frame.core.iter_node import IterNode

from static_frame.core.index import Index
from static_frame.core.index import IndexGO
from static_frame.core.index_hierarchy import IndexHierarchy




#-------------------------------------------------------------------------------
class Series(metaclass=MetaOperatorDelegate):
    '''
    A one-dimensional ordered, labelled collection, immutable and of fixed size.

    Args:
        values: An iterable of values, or a single object, to be aligned with the supplied (or automatically generated) index. Alternatively, a dictionary of index / value pairs can be provided.
        index: Option index initializer. If provided, lenght must be equal to length of values.
        own_index: Flag index as ownable by Series; primarily for internal clients.
    '''

    __slots__ = (
        'values',
        '_index',
        'iloc',
        'loc',
        'mask',
        'masked_array',
        'assign',
        'iter_group',
        'iter_group_items',
        'iter_element',
        'iter_element_items',
        )

    @classmethod
    def from_items(cls,
            pairs: tp.Iterable[tp.Tuple[tp.Hashable, tp.Any]],
            dtype: DtypeSpecifier=None) -> 'Series':
        '''Series construction from an iterator or generator of pairs, where the first value is the index and the second value is the value.

        Args:
            pairs: Iterable of pairs of index, value.
            dtype: dtype or valid dtype specifier.
        '''
        index = []
        def values():
            for pair in pairs:
                # populate index as side effect of iterating values
                index.append(pair[0])
                yield pair[1]

        return cls(values(), index=index, dtype=dtype)

    @classmethod
    def from_pandas(cls,
            value,
            *,
            own_data: bool=False,
            own_index: bool=False) -> 'Series':
        '''Given a Pandas Series, return a Series.

        Args:
            own_data: If True, the underlying NumPy data array will be made immutable and used without a copy.
            own_index: If True, the underlying NumPy index label array will be made immutable and used without a copy.
        '''
        if own_data:
            data = value.values
            data.flags.writeable = False
        else:
            data = immutable_filter(value.values)

        if own_index:
            index = value.index.values
            index.flags.writeable = False
        else:
            index = immutable_filter(value.index.values)

        # index is already managed, can own
        return cls(data, index=index)

    def __init__(self,
            values: SeriesInitializer,
            *,
            index: IndexInitializer=None,
            dtype: DtypeSpecifier=None,
            own_index: bool=False
            ) -> None:
        #-----------------------------------------------------------------------
        # values assignment

        values_constructor = None # if deferred

        # expose .values directly as it is immutable
        if not isinstance(values, np.ndarray):
            if isinstance(values, dict):
                # not sure if we should sort; not sure what to do if index is provided
                if index is not None:
                    raise Exception('cannot create Series from dictionary when index is defined')
                index = []
                def values_gen():
                    for k, v in _dict_to_sorted_items(values):
                        # populate index as side effect of iterating values
                        index.append(k)
                        yield v
                self.values = np.fromiter(values_gen(), dtype=dtype, count=len(values))

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
                # what to do here?
                raise Exception('type requested is not the type given')
            if values.shape == (): # handle special case of NP element
                def values_constructor(shape):
                    self.values = np.repeat(values, shape)
                    self.values.flags.writeable = False
            else:
                self.values = immutable_filter(values)

        #-----------------------------------------------------------------------
        # index assignment
        # NOTE: this generally must be done after values assignment, as from_items needs a values generator to be exhausted before looking to values

        if index is None or (hasattr(index, '__len__') and len(index) == 0):
            # create an integer index
            self._index = Index(range(len(self.values)), loc_is_iloc=True)
        elif own_index or (hasattr(index, 'STATIC') and index.STATIC):
            self._index = index
        else: # let index handle instantiation
            self._index = Index(index)

        shape = self._index.__len__()

        if values_constructor:
            values_constructor(shape) # updates self.values

        if len(self.values) != shape:
            raise Exception('values and index do not match length')

        #-----------------------------------------------------------------------
        # attributes

        self.loc = GetItem(self._extract_loc)
        self.iloc = GetItem(self._extract_iloc)

        self.mask = ExtractInterface(
                iloc=GetItem(self._extract_iloc_mask),
                loc=GetItem(self._extract_loc_mask),
                getitem=self._extract_loc_mask)

        self.masked_array = ExtractInterface(
                iloc=GetItem(self._extract_iloc_masked_array),
                loc=GetItem(self._extract_loc_masked_array),
                getitem=self._extract_loc_masked_array)

        self.assign = ExtractInterface(
                iloc=GetItem(self._extract_iloc_assign),
                loc=GetItem(self._extract_loc_assign),
                getitem=self._extract_loc_assign)


        self.iter_group = IterNode(
                container=self,
                function_items=self._axis_group_items,
                function_values=self._axis_group,
                yield_type=IterNodeType.VALUES
                )
        self.iter_group_items = IterNode(
                container=self,
                function_items=self._axis_group_items,
                function_values=self._axis_group,
                yield_type=IterNodeType.ITEMS
                )


        self.iter_element = IterNode(
                container=self,
                function_items=self._axis_element_items,
                function_values=self._axis_element,
                yield_type=IterNodeType.VALUES
                )
        self.iter_element_items = IterNode(
                container=self,
                function_items=self._axis_element_items,
                function_values=self._axis_element,
                yield_type=IterNodeType.ITEMS
                )

    #---------------------------------------------------------------------------
    # index manipulation

    def _reindex_other_like_iloc(self,
            value: 'Series',
            iloc_key: GetItemKeyType) -> 'Series':
        '''Given a value that is a Series, reindex it to the index components, drawn from this Series, that are specified by the iloc_key.
        '''
        return value.reindex(self._index._extract_iloc(iloc_key))


    def reindex(self,
            index: tp.Union[Index, tp.Sequence[tp.Any]],
            fill_value=np.nan) -> 'Series':
        '''
        Return a new Series based on the passed index.

        Args:
            fill_value: attempted to be used, but may be coerced by the dtype of this Series. `
        '''
        # TODO: implement `method` argument with bfill, ffill options

        if isinstance(index, (Index, IndexHierarchy)):
            # always use the Index constructor for safe reuse when possible
            index = index.__class__(index)
        else: # create the Index if not already an index, assume 1D
            index = Index(index)

        ic = IndexCorrespondence.from_correspondence(self.index, index)

        if ic.is_subset: # must have some common
            return self.__class__(self.values[ic.iloc_src],
                    index=index,
                    own_index=True)

        values = _full_for_fill(self.values.dtype, len(index), fill_value)

        # if some intersection of values
        if ic.has_common:
            values[ic.iloc_dst] = self.values[ic.iloc_src]

        # make immutable so a copy is not made
        values.flags.writeable = False
        return self.__class__(values,
                index=index,
                own_index=True)

    def relabel(self, mapper: CallableOrMapping) -> 'Series':
        '''
        Return a new Series based on a mapping (or callable) from old to new index values.
        '''
        return self.__class__(self.values,
                index=self._index.relabel(mapper),
                own_index=True)


    def reindex_flat(self):
        '''
        Return a new Series, where a hierarhical index (if deifined) is replaced with a flat, one-dimension index of tuples.
        '''
        return self.__class__(self.values, index=self._index.flat())

    def reindex_add_level(self, level: tp.Hashable):
        '''
        Return a new Series, adding a new root level to the index.
        '''
        return self.__class__(self.values, index=self._index.add_level(level))

    def reindex_drop_level(self, count: int=1):
        '''
        Return a new Series, dropping one or more leaf levels from the index.
        '''
        return self.__class__(self.values, index=self._index.drop_level(count))



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
        sel = np.logical_not(_isna(self.values))
        if not np.any(sel):
            return self

        values = self.values[sel]
        values.flags.writeable = False
        return self.__class__(values, index=self._index.loc[sel])

    def fillna(self, value) -> 'Series':
        '''Return a new Series after replacing NaN or None values with the supplied value.
        '''
        sel = _isna(self.values)
        if not np.any(sel):
            return self

        if isinstance(value, np.ndarray):
            raise Exception('cannot assign an array to fillna')
        else:
            value_dtype = np.array(value).dtype

        assigned_dtype = _resolve_dtype(value_dtype, self.values.dtype)

        if self.values.dtype == assigned_dtype:
            assigned = self.values.copy()
        else:
            assigned = self.values.astype(assigned_dtype)

        assigned[sel] = value
        assigned.flags.writeable = False
        return self.__class__(assigned, index=self._index)

    #---------------------------------------------------------------------------
    # operators

    def _ufunc_unary_operator(self, operator: tp.Callable) -> 'Series':
        return self.__class__(operator(self.values), index=self._index, dtype=self.dtype)

    def _ufunc_binary_operator(self, *, operator: tp.Callable, other) -> 'Series':

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
        result.flags.writeable = False
        return self.__class__(result, index=index)


    def _ufunc_axis_skipna(self, *, axis, skipna, ufunc, ufunc_skipna, dtype=None):
        '''For a Series, all functions of this type reduce the single axis of the Series to 1d, so Index has no use here.

        Args:
            dtype: not used, part of signature for a commin interface
        '''
        # following pandas convention, we replace Nones with nans so that, if skipna is False, a None can cause a nan result
        return _ufunc_skipna_1d(
                array=self.values,
                skipna=skipna,
                ufunc=ufunc,
                ufunc_skipna=ufunc_skipna)

    #---------------------------------------------------------------------------
    def __len__(self) -> int:
        '''Length of values.
        '''
        return self.values.__len__()

    def display(self, config: DisplayConfig=None) -> Display:
        '''Return a Display of the Series.
        '''
        config = config or DisplayActive.get()

        d = self._index.display(config=config)
        d.append_display(Display.from_values(
                self.values,
                header='<' + self.__class__.__name__ + '>',
                config=config))
        return d

    def __repr__(self):
        return repr(self.display())

    #---------------------------------------------------------------------------
    # common attributes from the numpy array

    @property
    def mloc(self):
        return mloc(self.values)

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def size(self):
        return self.values.size

    @property
    def nbytes(self):
        return self.values.nbytes

    #---------------------------------------------------------------------------
    # extraction

    def _extract_iloc(self, key: GetItemKeyType) -> 'Series':
        # iterable selection should be handled by NP (but maybe not if a tuple)
        return self.__class__(
                self.values[key],
                index=self._index.iloc[key])

    def _extract_loc(self, key: GetItemKeyType) -> 'Series':
        '''
        Compatibility:
            Pandas supports taking in iterables of keys, where some keys are not found in the index; a Series is returned as if a reindex operation was performed. This is undesirable. Better instead is to use reindex()
        '''
        iloc_key = self._index.loc_to_iloc(key)
        values = self.values[iloc_key]

        if not isinstance(values, np.ndarray): # if we have a single element
            return values
        # this might create new index from iloc, and then createa another Index on Index __init__
        return self.__class__(values, index=self._index.iloc[iloc_key])

    def __getitem__(self, key: GetItemKeyType) -> 'Series':
        '''A Loc selection (by index labels).

        Compatibility:
            Pandas supports using both loc and iloc style selections with the __getitem__ interface on Series. This is undesirable, so here we only expose the loc interface (making the Series dictionary like, but unlike the Index, where __getitem__ is an iloc).
        '''
        return self._extract_loc(key)

    #---------------------------------------------------------------------------
    # utilites for alternate extraction: mask and assignment

    def _extract_iloc_mask(self, key: GetItemKeyType) -> 'Series':
        '''Produce a new boolean Series of the same shape, where the values selected via iloc selection are True.
        '''
        mask = np.full(self.values.shape, False, dtype=bool)
        mask[key] = True
        mask.flags.writeable = False
        # can pass self here as it is immutable (assuming index cannot change)
        return self.__class__(mask, index=self._index)

    def _extract_loc_mask(self, key: GetItemKeyType) -> 'Series':
        '''Produce a new boolean Series of the same shape, where the values selected via loc selection are True.
        '''
        iloc_key = self._index.loc_to_iloc(key)
        return self._extract_iloc_mask(key=iloc_key)


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
        return SeriesAssign(data=self, iloc_key=key)

    def _extract_loc_assign(self, key: GetItemKeyType) -> 'SeriesAssign':
        iloc_key = self._index.loc_to_iloc(key)
        return SeriesAssign(data=self, iloc_key=iloc_key)

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
            ascending: bool=True,
            kind: str=_DEFAULT_SORT_KIND) -> 'Series':
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
            ascending: bool=True,
            kind: str=_DEFAULT_SORT_KIND) -> 'Series':
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
        v, _ = _iterable_to_array(other)
        array = np.in1d(self.values, v)
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

    #---------------------------------------------------------------------------
    # utility function to numpy array

    def unique(self) -> np.ndarray:
        '''
        Return a NumPy array of unqiue values.
        '''
        return np.unique(self.values)

    #---------------------------------------------------------------------------
    # export

    def to_pairs(self) -> tp.Iterable[tp.Tuple[tp.Hashable, tp.Any]]:
        '''
        Return a tuple of tuples, where each inner tuple is a pair of index label, value.
        '''
        if isinstance(self._index, IndexHierarchy):
            index_values = list(_array2d_to_tuples(self._index.values))
        else:
            index_values = self._index.values

        return tuple(zip(index_values, self.values))

    def to_pandas(self):
        '''
        Return a Pandas Series.
        '''
        import pandas
        return pandas.Series(self.values.copy(),
                index=self._index.values.copy())




class SeriesAssign:
    __slots__ = ('data', 'iloc_key')

    def __init__(self, *,
            data: Series,
            iloc_key: GetItemKeyType
            ) -> None:
        self.data = data
        self.iloc_key = iloc_key

    def __call__(self, value):
        if isinstance(value, Series):
            value = self.data._reindex_other_like_iloc(value, self.iloc_key).values

        if isinstance(value, np.ndarray):
            value_dtype = value.dtype
        else:
            value_dtype = np.array(value).dtype
        dtype = _resolve_dtype(self.data.dtype, value_dtype)

        if dtype == self.data.dtype:
            array = self.data.values.copy()
        else:
            array = self.data.values.astype(dtype)

        array[self.iloc_key] = value
        array.flags.writeable = False
        return Series(array, index=self.data.index)



