import typing as tp
from collections.abc import KeysView
import operator as operator_mod
from itertools import zip_longest

from functools import reduce

import numpy as np

from static_frame.core.util import DEFAULT_SORT_KIND
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import EMPTY_TUPLE
from static_frame.core.util import EMPTY_SLICE
from static_frame.core.util import SLICE_ATTRS
from static_frame.core.util import SLICE_START_ATTR
from static_frame.core.util import SLICE_STOP_ATTR
from static_frame.core.util import BOOL_TYPES
from static_frame.core.util import KEY_ITERABLE_TYPES
from static_frame.core.util import EMPTY_ARRAY
from static_frame.core.util import DTYPE_DATETIME_KIND

from static_frame.core.util import GetItemKeyType
from static_frame.core.util import CallableOrMapping
from static_frame.core.util import DtypeSpecifier
# from static_frame.core.util import IndexSpecifier
from static_frame.core.util import IndexInitializer
from static_frame.core.util import DepthLevelSpecifier
# from static_frame.core.util import mloc
from static_frame.core.util import ufunc_axis_skipna
from static_frame.core.util import iterable_to_array_1d
from static_frame.core.util import isin

from static_frame.core.util import immutable_filter
from static_frame.core.util import name_filter
from static_frame.core.util import array_shift
from static_frame.core.util import array2d_to_tuples
from static_frame.core.util import slice_to_inclusive_slice

from static_frame.core.util import DTYPE_INT_DEFAULT

from static_frame.core.selector_node import InterfaceGetItem
from static_frame.core.selector_node import InterfaceSelection1D
from static_frame.core.util import union1d
from static_frame.core.util import intersect1d
from static_frame.core.util import setdiff1d
from static_frame.core.util import to_datetime64
from static_frame.core.util import INT_TYPES

from static_frame.core.util import resolve_dtype
from static_frame.core.container import ContainerOperand
from static_frame.core.container_util import matmul


from static_frame.core.doc_str import doc_inject
from static_frame.core.index_base import IndexBase
# from static_frame.core.iter_node import IterNode
from static_frame.core.iter_node import IterNodeDepthLevel
from static_frame.core.iter_node import IterNodeType
from static_frame.core.iter_node import IterNodeApplyType


from static_frame.core.display import DisplayConfig
from static_frame.core.display import DisplayActive
from static_frame.core.display import Display
from static_frame.core.display import DisplayHeader

from static_frame.core.exception import ErrorInitIndex
from static_frame.core.exception import LocEmpty
from static_frame.core.exception import LocInvalid


if tp.TYPE_CHECKING:
    import pandas #pylint: disable=W0611 #pragma: no cover


I = tp.TypeVar('I', bound=IndexBase)


class ILocMeta(type):

    def __getitem__(cls,
            key: GetItemKeyType
            ) -> tp.Iterable[GetItemKeyType]:
        return cls(key)

class ILoc(metaclass=ILocMeta):
    '''A wrapper for embedding ``iloc`` specificiations within a single axis argument of a ``loc`` selection.
    '''

    __slots__ = (
            'key',
            )

    def __init__(self, key: GetItemKeyType):
        self.key = key

class LocMap:

    @staticmethod
    def map_slice_args(
            label_to_pos: tp.Callable[[tp.Iterable[tp.Hashable]], int],
            key: slice,
            labels: tp.Optional[np.ndarray] = None,
            offset: tp.Optional[int] = 0
            ) -> tp.Iterator[int]:
        '''Given a slice ``key`` and a label-to-position mapping, yield each integer argument necessary to create a new iloc slice. If the ``key`` defines a region with no constituents, raise ``LocEmpty``

        Args:
            label_to_pos: callable into mapping (can be a get() method from a dictionary)
        '''
        offset_apply = not offset is None

        for field in SLICE_ATTRS:
            attr = getattr(key, field)
            if attr is None:
                yield None
            elif isinstance(attr, np.datetime64):
                # if a datetime, we assume that the labels are ordered;
                if attr.dtype == labels.dtype:
                    pos = label_to_pos(attr)
                    if pos is None:
                        # if same type, and that atter is not in labels, we fail, just as we do in then non-datetime64 case. Only when datetimes are given in a different unit are we "loose" about matching.
                        raise LocInvalid('Invalid loc given in a slice', attr, field)

                    if field == SLICE_STOP_ATTR:
                        pos += 1 # stop is inclusive

                elif field == SLICE_START_ATTR:
                    # convert to the type of the atrs; this should get the relevant start
                    pos = label_to_pos(attr.astype(labels.dtype))
                    if pos is None: # we did not find a start position
                        matches = np.flatnonzero(labels.astype(attr.dtype) == attr)
                        if len(matches):
                            pos = matches[0]
                        else:
                            raise LocEmpty()

                elif field == SLICE_STOP_ATTR:
                    # convert labels to the slice attr value, compare, then get last
                    # add one, as this is an inclusive stop
                    # pos = np.flatnonzero(labels.astype(attr.dtype) == attr)[-1] + 1
                    matches = np.flatnonzero(labels.astype(attr.dtype) == attr)
                    if len(matches):
                        pos = matches[-1] + 1
                    else:
                        raise LocEmpty()

                if offset_apply:
                    pos += offset

                yield pos

            else:
                pos = label_to_pos(attr)
                if pos is None:
                    # NOTE: could raise LocEmpty() to silently handle this
                    raise LocInvalid('Invalid loc given in a slice', attr, field)

                if offset_apply:
                    pos += offset

                if field is SLICE_STOP_ATTR:
                    # loc selections are inclusive, so iloc gets one more
                    yield pos + 1
                else:
                    yield pos

    @classmethod
    def loc_to_iloc(cls, *,
            label_to_pos: tp.Dict,
            labels: np.ndarray,
            positions: np.ndarray,
            key: GetItemKeyType,
            offset: tp.Optional[int] = None
            ) -> GetItemKeyType:
        '''
        Note: all SF objects (Series, Index) need to be converted to basic types before being passed as `key` to this function.

        Args:
            offset: in the contect of an IndexHierarchical, the iloc positions returned from this funcition need to be shifted.
        Returns:
            An integer mapped slice, or GetItemKey type that is based on integers, compatible with TypeBlocks
        '''
        offset_apply = not offset is None

        # ILoc is handled prior to this call, in the Index.loc_to_iloc method

        if isinstance(key, slice):
            if offset_apply and key == NULL_SLICE:
                # when offset is defined (even if it is zero), null slice is not sufficiently specific; need to convert to an explict slice relative to the offset
                return slice(offset, len(positions) + offset)
            try:
                return slice(*cls.map_slice_args(
                        label_to_pos.get,
                        key,
                        labels,
                        offset)
                        )
            except LocEmpty:
                return EMPTY_SLICE

        if isinstance(key, np.datetime64):
            # convert this to the target representation, do a Boolean selection
            if labels.dtype != key.dtype:
                key = labels.astype(key.dtype) == key
            # if not different type, keep it the same so as to do a direct, single element selection

        # handles only lists and arrays; break out comparisons to avoid multiple
        is_array = isinstance(key, np.ndarray)
        is_list = isinstance(key, list)

        # can be an iterable of labels (keys) or an iterable of Booleans
        if is_array or is_list:
            if is_array and key.dtype.kind == DTYPE_DATETIME_KIND:
                if labels.dtype != key.dtype:
                    labels_ref = labels.astype(key.dtype)
                    # let Boolean key hit next branch
                    key = reduce(operator_mod.or_,
                            (labels_ref == k for k in key))
                    # NOTE: may want to raise instead of support this
                    # raise NotImplementedError(f'selecting {labels.dtype} with {key.dtype} is not presently supported')

            if is_array and key.dtype == bool:
                if offset_apply:
                    return positions[key] + offset
                return positions[key]

            # map labels to integer positions
            # NOTE: we may miss the opportunity to get a reference from values when we have contiguous keys
            if offset_apply:
                return [label_to_pos[x] + offset for x in key]
            return [label_to_pos[x] for x in key]

        # if a single element (an integer, string, or date, we just get the integer out of the map
        if offset_apply:
            return label_to_pos[key] + offset
        return label_to_pos[key]


def immutable_index_filter(index: I) -> I:
    '''Return an immutable index. All index objects handle converting from mutable to immutable via the __init__ constructor; but need to use appropriate class between Index and IndexHierarchy.'''

    if index.STATIC:
        return index
    return index._IMMUTABLE_CONSTRUCTOR(index)


def mutable_immutable_index_filter(target_static: bool, index: I) -> I:
    if target_static:
        return immutable_index_filter(index)
    # target mutable
    if index.STATIC:
        return index._MUTABLE_CONSTRUCTOR(index)
    return index.__class__(index) # create new instance

#-------------------------------------------------------------------------------

class PositionsAllocator:

    _size: int = 0
    _array: np.ndarray = np.arange(_size, dtype=DTYPE_INT_DEFAULT)
    _array.flags.writeable = False

    @classmethod
    def get(cls, size: int) -> np.ndarray:
        if size > cls._size:
            cls._size = size * 2
            cls._array = np.arange(cls._size, dtype=DTYPE_INT_DEFAULT)
            cls._array.flags.writeable = False
        # slices of immutable arrays are immutable
        return cls._array[:size]

#-------------------------------------------------------------------------------
_INDEX_SLOTS = (
        '_map',
        '_labels',
        '_positions',
        '_recache',
        '_name'
        )

@doc_inject(selector='index_init')
class Index(IndexBase):
    '''A mapping of labels to positions, immutable and of fixed size. Used by default in :obj:`Series` and as index and columns in :obj:`Frame`. Base class of all 1D indices.

    {args}
    '''

    __slots__ = _INDEX_SLOTS

    # _IMMUTABLE_CONSTRUCTOR is None from IndexBase
    # _MUTABLE_CONSTRUCTOR will be set after IndexGO defined

    _UFUNC_UNION = union1d
    _UFUNC_INTERSECTION = intersect1d
    _UFUNC_DIFFERENCE = setdiff1d

    _DTYPE: tp.Optional[np.dtype] = None # for specialized indices requiring a typed labels

    # for compatability with IndexHierarchy, where this is implemented as a property method
    depth: int = 1

    _map: tp.Optional[tp.Dict[tp.Hashable, int]]
    _labels: np.ndarray
    _positions: np.ndarray
    _recache: bool
    _name: tp.Hashable

    #---------------------------------------------------------------------------
    # methods used in __init__ that are customized in dervied classes; there, we need to mutate instance state, this these are instance methods
    @staticmethod
    def _extract_labels(
            mapping: tp.Optional[tp.Dict[tp.Hashable, int]],
            labels: tp.Iterable[tp.Hashable],
            dtype: tp.Optional[np.dtype] = None
            ) -> np.ndarray:
        '''Derive labels, a cache of the mapping keys in a sequence type (either an ndarray or a list).

        If the labels passed at instantiation are an ndarray, they are used after immutable filtering. Otherwise, the mapping keys are used to create an ndarray.

        This method is overridden in the derived class.

        Args:
            mapping: Can be None if loc_is_iloc.
            labels: might be an expired Generator, but if it is an immutable ndarray, we can use it without a copy.
        '''
        # pre-fetching labels for faster get_item construction
        if isinstance(labels, np.ndarray):
            if dtype is not None and dtype != labels.dtype:
                raise RuntimeError('invalid label dtype for this Index')
            return immutable_filter(labels)

        if hasattr(labels, '__len__'): # not a generator, not an array
            # resolving the dtype is expensive, pass if possible
            if len(labels) == 0:
                labels = EMPTY_ARRAY
            else:
                labels, _ = iterable_to_array_1d(labels, dtype=dtype)
        else: # labels may be an expired generator, must use the mapping
            if len(mapping) == 0:
                labels = EMPTY_ARRAY
            else:
                labels, _ = iterable_to_array_1d(mapping, dtype=dtype)
        # all arrays are immutable
        # assert labels.flags.writeable == False
        return labels

    @staticmethod
    def _extract_positions(
            size: int,
            positions: tp.Optional[tp.Sequence[int]]):
        # positions is either None or an ndarray
        if isinstance(positions, np.ndarray):
            return immutable_filter(positions)
        return PositionsAllocator.get(size)

    @staticmethod
    def _get_map(
            labels: tp.Iterable[tp.Hashable],
            positions: tp.Optional[tp.Sequence[int]] = None
            ) -> tp.Dict[tp.Hashable, int]:
        '''
        Return a dictionary mapping index labels to integer positions.

        NOTE: this function is critical to Index performance.

        Args:
            lables: an Iterable of hashables; can be a generator.
        '''
        if positions is not None: # can zip both without new collection
            return dict(zip(labels, positions))
        if hasattr(labels, '__len__'):
            # unhashable 2D numpy arrays will raise
            return dict(zip(labels, range(len(labels))))
        # support labels as a generator
        return {v: k for k, v in enumerate(labels)}

    #---------------------------------------------------------------------------
    # constructors

    @classmethod
    def from_labels(cls,
            labels: tp.Iterable[tp.Sequence[tp.Hashable]],
            *,
            name: tp.Hashable = None
            ) -> 'Index':
        '''
        Construct an ``Index`` from an iterable of labels, where each label is a hashable. Provided for a compatible interface to ``IndexHierarchy``.
        '''
        return cls(labels=labels, name=name)

    #---------------------------------------------------------------------------
    def __init__(self,
            labels: IndexInitializer,
            *,
            loc_is_iloc: bool = False,
            name: tp.Optional[tp.Hashable] = None,
            dtype: DtypeSpecifier = None
            ) -> None:

        self._recache: bool = False
        self._map: tp.Dict[tp.Hashable, int] = None

        positions = None

        # resolve the targetted labels dtype, by lookin at the class attr _DTYPE and/or the passed dtype argument
        if dtype is None:
            dtype_extract = self._DTYPE # set in some specialized Index classes
        else: # passed dtype is not None
            if self._DTYPE is not None and dtype != self._DTYPE:
                # NOTE: should never get to this branch, as derived Index classes that set _DTYPE remove dtype from __init__
                raise ErrorInitIndex('invalid dtype argument for this Index', dtype, self._DTYPE) #pragma: no cover
            # self._DTYPE is None, passed dtype is not None, use dtype
            dtype_extract = dtype

        #-----------------------------------------------------------------------
        # handle all Index subclasses
        if issubclass(labels.__class__, IndexBase):
            if labels._recache:
                labels._update_array_cache()
            if name is None and labels.name is not None:
                name = labels.name # immutable, so no copy necessary
            if labels.depth == 1: # not an IndexHierarchy
                if labels.STATIC and self.STATIC: # can take the map
                    self._map = labels._map
                # get a reference to the immutable arrays, even if this is an IndexGO index, we can take the cached arrays, assuming they are up to date
                positions = labels._positions
                loc_is_iloc = labels._map is None
                labels = labels._labels
            else: # IndexHierarchy
                # will be a generator of tuples; already updated caches
                labels = array2d_to_tuples(labels._labels)
        elif isinstance(labels, ContainerOperand):
            # it is a Series or similar
            array = labels.values
            if array.ndim == 1:
                labels = array
            else:
                labels = array2d_to_tuples(array)
        # else: assume an iterable suitable for labels usage

        #-----------------------------------------------------------------------
        if self._DTYPE is not None:
            # do not need to check arrays, as will and checked to match dtype_extract in _extract_labels
            if not isinstance(labels, np.ndarray):
                # for now, assume that if _DTYPE is defined, we have a date
                labels = (to_datetime64(v, dtype_extract) for v in labels)
            else: # coerce to target type
                labels = labels.astype(dtype_extract)

        self._name = name if name is None else name_filter(name)

        if self._map is None: # if _map not shared from another Index
            if not loc_is_iloc:
                self._map = self._get_map(labels, positions)
                size = len(self._map)
            else: # must assume labels are unique
                size = len(labels)
                if positions is None:
                    positions = PositionsAllocator.get(size)
        else: # map shared from another Index
            size = len(self._map)

        # this might be NP array, or a list, depending on if static or grow only; if an array, dtype will be compared with passed dtype_extract
        self._labels = self._extract_labels(self._map, labels, dtype_extract)
        self._positions = self._extract_positions(size, positions)

        if self._DTYPE and self._labels.dtype != self._DTYPE:
            raise ErrorInitIndex('invalid label dtype for this Index', #pragma: no cover
                    self._labels.dtype, self._DTYPE)

        if self._map is not None and len(self._map) != len(self._labels):
            raise ErrorInitIndex(f'labels ({len(self._labels)}) have non-unique values ({len(self._map)})')


    #---------------------------------------------------------------------------
    def __setstate__(self, state):
        '''
        Ensure that reanimated NP arrays are set not writeable.
        '''
        for key, value in state[1].items():
            setattr(self, key, value)
        self._labels.flags.writeable = False

    #---------------------------------------------------------------------------
    # name interface

    def rename(self: I, name: tp.Hashable) -> I:
        '''
        Return a new Frame with an updated name attribute.
        '''
        if self._recache:
            self._update_array_cache()
        # let the constructor handle reuse
        return self.__class__(self, name=name)

    #---------------------------------------------------------------------------
    # interfaces

    @property
    def loc(self) -> InterfaceGetItem:
        return InterfaceGetItem(self._extract_loc)

    @property
    def iloc(self) -> InterfaceGetItem:
        return InterfaceGetItem(self._extract_iloc)

    # # on Index, getitem is an iloc selector; on Series, getitem is a loc selector; for this extraction interface, we do not implement a getitem level function (using iloc would be consistent), as it is better to be explicit between iloc loc

    def _iter_label(self, depth_level: DepthLevelSpecifier = 0):
        yield from self._labels

    def _iter_label_items(self, depth_level: DepthLevelSpecifier = 0):
        yield from zip(self._positions, self._labels)

    @property
    def iter_label(self) -> IterNodeDepthLevel:
        return IterNodeDepthLevel(
                container=self,
                function_items=self._iter_label_items,
                function_values=self._iter_label,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.INDEX_LABELS
                )


    @property
    def drop(self) -> InterfaceSelection1D:
        return InterfaceSelection1D(
            func_iloc=self._drop_iloc,
            func_loc=self._drop_loc,
            )


    @doc_inject(select='astype')
    def astype(self, dtype: DtypeSpecifier) -> 'Index':
        '''
        Return an Index with type determined by `dtype` argument. Note that for Index, this is a simple function, whereas for ``IndexHierarchy``, this is an interface exposing both a callable and a getitem interface.

        Args:
            {dtype}
        '''
        return self.__class__(
                self.values.astype(dtype),
                name=self._name
                )

    #---------------------------------------------------------------------------

    def _update_array_cache(self) -> None:
        '''Derived classes can use this to set stored arrays, self._labels and self._positions.
        '''

    #---------------------------------------------------------------------------

    def __len__(self) -> int:
        if self._recache:
            self._update_array_cache()
        return len(self._labels)

    @doc_inject()
    def display(self,
            config: tp.Optional[DisplayConfig] = None,
            ) -> Display:
        '''{doc}

        Args:
            {config}
        '''
        config = config or DisplayActive.get()

        if self._recache:
            self._update_array_cache()

        return Display.from_values(self.values,
                header=DisplayHeader(self.__class__, self._name),
                config=config,
                outermost=True,
                index_depth=0,
                header_depth=1
                )

    #---------------------------------------------------------------------------
    # core internal representation

    @property
    def values(self) -> np.ndarray:
        '''A 1D array of labels. Note that these are they labels, i.e., the keys of the mapping, not the values, i.e. where those keys point to.
        '''
        if self._recache:
            self._update_array_cache()
        return self._labels

    @property
    def positions(self) -> np.ndarray:
        '''Return the immutable positions array. This is needed by some clients, such as Series and Frame, to support Boolean usage in drop.
        '''
        if self._recache:
            self._update_array_cache()
        return self._positions


    def values_at_depth(self, depth_level: DepthLevelSpecifier = 0):
        '''
        Return an NP array for the `depth_level` specified.
        '''
        if depth_level != 0:
            raise RuntimeError('invalid depth_level', depth_level)
        return self.values

    @doc_inject()
    def label_widths_at_depth(self,
            depth_level: DepthLevelSpecifier = 0
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, int]]:
        '''{}'''
        if depth_level != 0:
            raise RuntimeError('invalid depth_level', depth_level)
        yield from zip_longest(self.values, EMPTY_TUPLE, fillvalue=1)


    #---------------------------------------------------------------------------

    def copy(self: I) -> I:
        '''
        Return a new Index.
        '''
        # this is not a complete deepcopy, as _labels here is an immutable np array (a new map will be created); if this is an IndexGO, we will pass the cached, immutable NP array
        if self._recache:
            self._update_array_cache()
        return self.__class__(labels=self, name=self._name)

    def relabel(self: I, mapper: CallableOrMapping) -> I:
        '''
        Return a new Index with labels replaced by the callable or mapping; order will be retained. If a mapping is used, the mapping need not map all origin keys.
        '''
        if not callable(mapper):
            # if a mapper, it must support both __getitem__ and __contains__
            getitem = getattr(mapper, '__getitem__')
            return self.__class__(
                    (getitem(x) if x in mapper else x for x in self._labels),
                    name=self._name
                    )

        return self.__class__(
                (mapper(x) for x in self._labels),
                name=self._name
                )

    #---------------------------------------------------------------------------
    # extraction and selection

    def loc_to_iloc(self,
            key: GetItemKeyType,
            offset: tp.Optional[int] = None,
            key_transform: tp.Optional[tp.Callable[[GetItemKeyType], GetItemKeyType]] = None
            ) -> GetItemKeyType:
        '''
        Note: Boolean Series are reindexed to this index, then passed on as all Boolean arrays.

        Args:
            offset: A default of None is critical to avoid large overhead in unnecessary application of offsets.
            key_transform: A function that transforms keys to specialized type; used by IndexDate indices.
        Returns:
            Return GetItemKey type that is based on integers, compatible with TypeBlocks
        '''
        from static_frame.core.series import Series

        if self._recache:
            self._update_array_cache()

        if isinstance(key, ILoc):
            return key.key
        elif isinstance(key, Index):
            # if an Index, we simply use the values of the index
            key = key.values
        elif isinstance(key, Series):
            if key.dtype == bool:
                if _requires_reindex(key.index, self):
                    key = key.reindex(self, fill_value=False).values
                else: # the index is equal
                    key = key.values
            else:
                key = key.values

        if self._map is None: # loc_is_iloc
            if isinstance(key, np.ndarray):
                if key.dtype == bool:
                    return key
                if key.dtype != DTYPE_INT_DEFAULT:
                    # if key is an np.array, it must be an int or bool type
                    # could use tolist(), but we expect all keys to be integers
                    return key.astype(DTYPE_INT_DEFAULT)
            elif isinstance(key, slice):
                key = slice_to_inclusive_slice(key)
            return key

        if key_transform:
            key = key_transform(key)

        return LocMap.loc_to_iloc(
                label_to_pos=self._map,
                labels=self._labels,
                positions=self._positions, # always an np.ndarray
                key=key,
                offset=offset
                )

    def _extract_iloc(self, key: GetItemKeyType) -> 'Index':
        '''Extract a new index given an iloc key
        '''
        if self._recache:
            self._update_array_cache()

        if key is None:
            labels = self._labels
        elif isinstance(key, slice):
            if key == NULL_SLICE:
                labels = self._labels
            else:
                # if labels is an np array, this will be a view; if a list, a copy
                labels = self._labels[key]
        elif isinstance(key, KEY_ITERABLE_TYPES):
            # we assume Booleans have been normalized to integers here
            # can select directly from _labels[key] if if key is a list
            labels = self._labels[key]
        else: # select a single label value
            return self._labels[key]

        return self.__class__(labels=labels, name=self._name)

    def _extract_loc(self, key: GetItemKeyType) -> 'Index':
        return self._extract_iloc(self.loc_to_iloc(key))

    def __getitem__(self: I, key: GetItemKeyType) -> I:
        '''Extract a new index given an iloc key.
        '''
        return self._extract_iloc(key)


    def _drop_iloc(self, key: GetItemKeyType) -> 'Index':
        '''Create a new index after removing the values specified by the loc key.
        '''
        if self._recache:
            self._update_array_cache()

        if key is None:
            labels = self._labels # already immutable
        elif isinstance(key, np.ndarray) and key.dtype == bool:
            # can use labels, as we already recached
            # use Boolean area to select indices from positions, as np.delete does not work with arrays
            labels = np.delete(self._labels, self._positions[key])
            labels.flags.writeable = False
        else:
            labels = np.delete(self._labels, key)
            labels.flags.writeable = False

        return self.__class__(labels, name=self._name)

    def _drop_loc(self, key: GetItemKeyType) -> 'Index':
        '''Create a new index after removing the values specified by the loc key.
        '''
        return self._drop_iloc(self.loc_to_iloc(key))


    #---------------------------------------------------------------------------
    # operators

    def _ufunc_unary_operator(self, operator: tp.Callable) -> np.ndarray:
        '''Always return an NP array.
        '''
        if self._recache:
            self._update_array_cache()

        array = operator(self._labels)
        array.flags.writeable = False
        return array

    def _ufunc_binary_operator(self, *, operator: tp.Callable, other) -> np.ndarray:
        '''
        Binary operators applied to an index always return an NP array. This deviates from Pandas, where some operations (multipling an int index by an int) result in a new Index, while other operations result in a np.array (using == on two Index).
        '''
        if self._recache:
            self._update_array_cache()

        if issubclass(other.__class__, Index):
            other = other.values # operate on labels to labels

        if operator.__name__ == 'matmul':
            return matmul(self._labels, other)
        elif operator.__name__ == 'rmatmul':
            return matmul(other, self._labels)

        result = operator(self._labels, other)

        if not isinstance(result, np.ndarray):
            # see Series._ufunc_binary_operator for notes on why
            if isinstance(result, BOOL_TYPES):
                result = np.full(len(self._labels), result)
            else:
                raise RuntimeError('unexpected branch from non-array result of operator application to array') #pragma: no cover

        result.flags.writeable = False
        return result


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

        Args:
            dtype: Not used in 1D application, but collected here to provide a uniform signature.
        '''
        if self._recache:
            self._update_array_cache()

        # do not need to pass on composabel here
        return ufunc_axis_skipna(
                array=self._labels,
                skipna=skipna,
                axis=0,
                ufunc=ufunc,
                ufunc_skipna=ufunc_skipna
                )

    # _ufunc_shape_skipna defined in IndexBase

    #---------------------------------------------------------------------------
    # dictionary-like interface

    # NOTE: we intentionally exclude keys(), items(), and get() from Index classes, as they return inconsistent result when thought of as a dictionary


    def __iter__(self) -> tp.Iterator[tp.Hashable]:
        '''Iterate over labels.
        '''
        if self._recache:
            self._update_array_cache()
        return tp.cast(tp.Iterator[tp.Hashable], self._labels.__iter__())

    def __reversed__(self) -> tp.Iterator[tp.Hashable]:
        '''
        Returns a reverse iterator on the index labels.
        '''
        if self._recache:
            self._update_array_cache()
        return reversed(self._labels)

    def __contains__(self, value) -> bool:
        '''Return True if value in the labels.
        '''
        if self._map is None: # loc_is_iloc
            if isinstance(value, INT_TYPES):
                return value >= 0 and value < len(self)
            return False
        return self._map.__contains__(value)


    #---------------------------------------------------------------------------
    # utility functions

    def sort(self,
            ascending: bool = True,
            kind: str = DEFAULT_SORT_KIND) -> 'Index':
        '''Return a new Index with the labels sorted.

        Args:
            kind: Sort algorithm passed to NumPy.
        '''
        # force usage of property for caching
        v = np.sort(self.values, kind=kind)
        if not ascending:
            v = v[::-1]

        v.flags.writeable = False
        return self.__class__(v, name=self._name)

    def isin(self, other: tp.Iterable[tp.Any]) -> np.ndarray:
        '''
        Return a Boolean array showing True where a label is found in other. If other is a multidimensional array, it is flattened.
        '''
        return isin(self.values, other, array_is_unique=True)

    def roll(self, shift: int) -> 'Index':
        '''Return an Index with values rotated forward and wrapped around (with a postive shift) or backward and wrapped around (with a negative shift).
        '''
        values = self.values # force usage of property for cache update
        if shift % len(values):
            values = array_shift(
                    array=values,
                    shift=shift,
                    axis=0,
                    wrap=True)
            values.flags.writeable = False
        return self.__class__(values, name=self._name)

    #---------------------------------------------------------------------------
    # export

    def to_series(self) -> 'Series':
        '''Return a Series with values from this Index's labels.
        '''
        # not sure if index should be self here
        from static_frame import Series
        return Series(self.values, index=None, name=self._name)

    def add_level(self, level: tp.Hashable) -> 'IndexHierarchy':
        '''Return an IndexHierarhcy with an added root level.
        '''
        from static_frame import IndexHierarchy
        return IndexHierarchy.from_tree({level: self.values})

    def to_pandas(self) -> 'pandas.Index':
        '''Return a Pandas Index.
        '''
        import pandas
        # must copy to remove immutability, decouple reference
        return pandas.Index(self.values.copy(),
                name=self._name)

#-------------------------------------------------------------------------------
_INDEX_GO_SLOTS = (
        '_map',
        '_labels',
        '_positions',
        '_recache',
        '_name',
        '_labels_mutable',
        '_labels_mutable_dtype',
        '_positions_mutable_count',
        )


class _IndexGOMixin:

    STATIC = False
    __slots__ = () # define in derived class

    _map: tp.Optional[tp.Dict[tp.Hashable, int]]
    _labels_mutable: tp.List[tp.Hashable]
    _labels_mutable_dtype: np.dtype
    _positions_mutable_count: int

    def _extract_labels(self,
            mapping: tp.Optional[tp.Dict[tp.Hashable, int]],
            labels: tp.Iterable[tp.Hashable],
            dtype: tp.Optional[np.dtype] = None
            ) -> np.ndarray:
        '''Called in Index.__init__(). This creates and populates mutable storage as a side effect of array derivation; this storage will be grown as needed.
        '''
        labels = Index._extract_labels(mapping, labels, dtype)
        self._labels_mutable = labels.tolist()
        if len(labels):
            self._labels_mutable_dtype = labels.dtype
        else: # avoid setting to float default when labels is empty
            self._labels_mutable_dtype = None
        return labels

    def _extract_positions(self,
            size: int,
            positions: tp.Optional[tp.Sequence[int]]
            ) -> tp.Iterable[tp.Any]:
        '''Called in Index.__init__(). This creates and populates mutable storage. This creates and populates mutable storage as a side effect of array derivation.
        '''
        positions = Index._extract_positions(size, positions)
        self._positions_mutable_count = size
        return positions

    def _update_array_cache(self):

        if self._labels_mutable_dtype is not None and len(self._labels):
            # only update if _labels_mutable_dtype has been set and _labels exist
            self._labels_mutable_dtype = resolve_dtype(
                    self._labels.dtype,
                    self._labels_mutable_dtype)

        self._labels = np.array(self._labels_mutable, dtype=self._labels_mutable_dtype)
        self._labels.flags.writeable = False
        self._positions = PositionsAllocator.get(self._positions_mutable_count)
        self._recache = False

    #---------------------------------------------------------------------------
    # grow only mutation

    def append(self, value: tp.Hashable) -> None:
        '''append a value
        '''
        if self.__contains__(value):
            raise KeyError(f'duplicate key append attempted: {value}')

        # if loc_is_iloc, we might need to initialzie map  if not an increment that keeps loc_is_iloc relationship; create map after updating self._labels_mutable
        initialize_map = False
        if self._map is None: # loc_is_iloc
            if not (isinstance(value, INT_TYPES)
                    and value == self._positions_mutable_count):
                initialize_map = True
        else:
            # the new value is the count
            self._map[value] = self._positions_mutable_count

        if self._labels_mutable_dtype is not None:
            self._labels_mutable_dtype = resolve_dtype(
                    np.array(value).dtype,
                    self._labels_mutable_dtype)
        else:
            self._labels_mutable_dtype = np.array(value).dtype

        self._labels_mutable.append(value)

        if initialize_map:
            self._map = self._get_map(labels=self._labels_mutable)

        self._positions_mutable_count += 1
        self._recache = True

    def extend(self, values: KEY_ITERABLE_TYPES):
        '''Append multiple values
        Args:
            values: can be a generator.
        '''
        for value in values:
            self.append(value)


@doc_inject(selector='index_init')
class IndexGO(_IndexGOMixin, Index):
    '''A mapping of labels to positions, immutable with grow-only size. Used as columns in :obj:`FrameGO`.

    {args}
    '''
    _IMMUTABLE_CONSTRUCTOR = Index

    __slots__ = _INDEX_GO_SLOTS


# update class attr on Index after class initialziation
Index._MUTABLE_CONSTRUCTOR = IndexGO



#-------------------------------------------------------------------------------

def _index_initializer_needs_init(value) -> bool:
    '''Determine if value is a non-empty index initializer. This could almost just be a truthy test, but ndarrays need to be handled in isolation. Generators should return True.
    '''
    if value is None:
        return False
    if isinstance(value, IndexBase):
        return False
    if isinstance(value, np.ndarray):
        return bool(len(value))
    return bool(value)


def _requires_reindex(left: Index, right: Index) -> bool:
    '''
    Given two Index objects, determine if we need to reindex
    '''
    if len(left) != len(right):
        return True
    # do not need a new Index object, so just compare arrays directly, which might return a single Boolean if the types are not compatible

    # NOTE: NP raises a warning here if we go to scalar value
    ne = left.values != right.values
    if isinstance(ne, np.ndarray):
        return ne.any() # if any not equal, require reindex
    # assume we have a bool
    return ne # if not equal, require reindex
