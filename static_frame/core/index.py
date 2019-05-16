import typing as tp
from collections import KeysView

import numpy as np

from static_frame.core.util import DEFAULT_SORT_KIND
from static_frame.core.util import NULL_SLICE

from static_frame.core.util import SLICE_ATTRS
from static_frame.core.util import SLICE_STOP_ATTR

from static_frame.core.util import KEY_ITERABLE_TYPES
from static_frame.core.util import EMPTY_ARRAY

from static_frame.core.util import GetItemKeyType
from static_frame.core.util import CallableOrMapping
from static_frame.core.util import DtypeSpecifier
# from static_frame.core.util import IndexSpecifier
from static_frame.core.util import IndexInitializer
from static_frame.core.util import DepthLevelSpecifier
# from static_frame.core.util import mloc
from static_frame.core.util import ufunc_skipna_1d
from static_frame.core.util import iterable_to_array
from static_frame.core.util import key_to_datetime_key

from static_frame.core.util import immutable_filter
from static_frame.core.util import name_filter
from static_frame.core.util import array_shift
from static_frame.core.util import array2d_to_tuples

from static_frame.core.util import DateInitializer
from static_frame.core.util import YearMonthInitializer
from static_frame.core.util import YearInitializer

from static_frame.core.util import _to_datetime64

from static_frame.core.util import _DT64_DAY
from static_frame.core.util import _DT64_MONTH
from static_frame.core.util import _DT64_YEAR
from static_frame.core.util import _DT64_S
from static_frame.core.util import _DT64_MS

from static_frame.core.util import _TD64_DAY
from static_frame.core.util import _TD64_MONTH
from static_frame.core.util import _TD64_YEAR
from static_frame.core.util import _TD64_S
from static_frame.core.util import _TD64_MS

from static_frame.core.doc_str import doc_inject


from static_frame.core.util import GetItem
from static_frame.core.util import InterfaceSelection1D

from static_frame.core.index_base import IndexBase
from static_frame.core.iter_node import IterNode
from static_frame.core.iter_node import IterNodeType
from static_frame.core.iter_node import IterNodeApplyType

from static_frame.core.operator_delegate import MetaOperatorDelegate

from static_frame.core.display import DisplayConfig
from static_frame.core.display import DisplayActive
from static_frame.core.display import Display
from static_frame.core.display import DisplayHeader


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
            label_to_pos: tp.Callable[[tp.Hashable], int],
            key: slice,
            offset: tp.Optional[int] = 0):
        '''Given a slice and a label-to-position mapping, yield each argument necessary to create a new slice.

        Args:
            label_to_pos: callable into mapping (can be a get() method from a dictionary)
        '''
        offset_apply = not offset is None

        for field in SLICE_ATTRS:
            attr = getattr(key, field)
            if attr is None:
                yield None
            else:
                pos = label_to_pos(attr)
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

        if isinstance(key, ILoc):
            return key.key

        if isinstance(key, slice):
            if offset_apply and key == NULL_SLICE:
                # when offset is defined (even if it is zero), null slice is not sufficiently specific; need to convert to an explict slice relative to the offset
                return slice(offset,
                        len(positions) + offset,
                        )
            return slice(*cls.map_slice_args(
                label_to_pos.get,
                key,
                offset)
                )

        if isinstance(key, np.datetime64):
            # convert this to the target representation, do a Boolean selection
            if labels.dtype != key.dtype:
                key = labels.astype(key.dtype) == key
            # if not different type, keep it the same so as to do a direct, single element selection

        # handles only lists and arrays
        if isinstance(key, KEY_ITERABLE_TYPES):
            # can be an iterable of labels (keys) or an iterable of Booleans
            # if len(key) == len(label_to_pos) and isinstance(key[0], (bool, np.bool_)):
            if isinstance(key, np.ndarray) and key.dtype == bool:
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


def immutable_index_filter(
        index: tp.Union['Index', 'IndexHierarchy']) -> tp.Union[
        'Index', 'IndexHierarchy']:
    '''Return an immutable index. All index objects handle converting from mutable to immutable via the __init__ constructor; but need to use appropriate class between Index and IndexHierarchy.'''

    if index.STATIC:
        return index
    return index._IMMUTABLE_CONSTRUCTOR(index)

#-------------------------------------------------------------------------------

@doc_inject(selector='index_init')
class Index(IndexBase,
        metaclass=MetaOperatorDelegate):
    '''A mapping of labels to positions, immutable and of fixed size. Used by default in :py:class:`Series` and as index and columns in :py:class:`Frame`.

    {args}
    '''

    __slots__ = (
            '_map',
            '_labels',
            '_positions',
            '_recache',
            '_loc_is_iloc',
            '_name'
            )

    _UFUNC_UNION = np.union1d
    _UFUNC_INTERSECTION = np.intersect1d

    _DTYPE = None # for specialized indices requiring a typed labels
    # for compatability with IndexHierarchy, where this is implemented as a property method
    depth = 1

    #---------------------------------------------------------------------------
    # methods used in __init__ that are customized in dervied classes; there, we need to mutate instance state, this these are instance methods
    @staticmethod
    def _extract_labels(
            mapping,
            labels,
            dtype=None) -> tp.Tuple[tp.Iterable[int], tp.Iterable[tp.Any]]:
        '''Derive labels, a cache of the mapping keys in a sequence type (either an ndarray or a list).

        If the labels passed at instantiation are an ndarray, they are used after immutable filtering. Otherwise, the mapping keys are used to create an ndarray.

        This method is overridden in the derived class.

        Args:
            labels: might be an expired Generator, but if it is an immutable ndarray, we can use it without a copy.
        '''
        # pre-fetching labels for faster get_item construction
        if isinstance(labels, np.ndarray): # if an np array can handle directly
            if dtype is not None and dtype != labels.dtype:
                raise RuntimeError('invalid label dtype for this Index')
            return immutable_filter(labels)

        if hasattr(labels, '__len__'): # not a generator, not an array
            if not len(labels):
                return EMPTY_ARRAY # already immutable

            if isinstance(labels[0], tuple):
                assert dtype is None or dtype == object
                array = np.empty(len(labels), object)
                array[:] = labels
                labels = array # rename
            else:
                labels = np.array(labels, dtype)
        else: # labels may be an expired generator
            # until all Python dictionaries are ordered, we cannot just take keys()
            # labels = np.array(tuple(mapping.keys()))
            # assume object type so as to not create a temporary list
            labels = np.empty(len(mapping), dtype=dtype if dtype else object)
            for k, v in mapping.items():
                labels[v] = k

        labels.flags.writeable = False
        return labels

    @staticmethod
    def _extract_positions(
            mapping,
            positions):
        # positions is either None or an ndarray
        if isinstance(positions, np.ndarray): # if an np array can handle directly
            return immutable_filter(positions)
        positions = np.arange(len(mapping))
        positions.flags.writeable = False
        return positions

    @staticmethod
    def _get_map(
            labels: tp.Iterable[tp.Hashable],
            positions=None
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
            labels: tp.Iterable[tp.Sequence[tp.Hashable]]) -> 'Index':
        '''
        Construct an ``Index`` from an iterable of labels, where each label is a hashable. Provided for a compatible interfave to ``IndexHierarchy``.
        '''
        return cls(labels=labels)

    #---------------------------------------------------------------------------
    def __init__(self,
            labels: IndexInitializer,
            *,
            loc_is_iloc: bool = False,
            name: tp.Hashable = None,
            dtype: DtypeSpecifier = None
            ) -> None:

        self._recache = False
        self._map = None
        positions = None

        # resolve the targetted labels dtype, by lookin at the class attr _DTYPE and/or the passed dtype argument
        if dtype is None:
            dtype_extract = self._DTYPE # set in specialized Index classes
        else: # passed dtype is not None
            if self._DTYPE is not None and dtype != self._DTYPE:
                raise RuntimeError('invalid dtype argument for this Index',
                        dtype, self._DTYPE)
            # self._DTYPE is None, passed dtype is not None, use dtype
            dtype_extract = dtype

        # handle all Index subclasses
        if issubclass(labels.__class__, IndexBase):
            if labels._recache:
                labels._update_array_cache()
            if name is None and labels.name is not None:
                name = labels.name # immutable, so no copy necessary

            if labels.depth == 1: # not an IndexHierarchy
                if labels.STATIC: # can take the map
                    self._map = labels._map
                # get a reference to the immutable arrays, even if this is an IndexGO index, we can take the cached arrays, assuming they are up to date
                positions = labels._positions
                loc_is_iloc = labels._loc_is_iloc
                labels = labels._labels
            else: # IndexHierarchy
                # will be a generator of tuples; already updated caches
                labels = array2d_to_tuples(labels._labels)
        elif hasattr(labels, 'values'):
            # it is a Series or similar
            array = labels.values
            if array.ndim == 1:
                labels = array
            else:
                labels = array2d_to_tuples(array)

        if self._DTYPE is not None:
            if not isinstance(labels, np.ndarray):
                # do not need to look further at labels from IndexBase
                # do not need to look at array, as will be typed and checked to match dtype_extract in _extract_labels
                # import ipdb; ipdb.set_trace()
                labels = (_to_datetime64(v, dtype_extract) for v in labels)
            else:
                # coerce to target type
                labels = labels.astype(dtype_extract)

        self._name = name if name is None else name_filter(name)

        if self._map is None:
            self._map = self._get_map(labels, positions)

        # this might be NP array, or a list, depending on if static or grow only; if an array, dtype will be compared with passed dtype_extract
        self._labels = self._extract_labels(self._map, labels, dtype_extract)
        self._positions = self._extract_positions(self._map, positions)

        if self._DTYPE and self._labels.dtype != self._DTYPE:
            raise RuntimeError('invalid label dtype for this Index',
                    self._labels.dtype, self._DTYPE)
        if len(self._map) != len(self._labels):
            raise KeyError('labels have non-unique values')

        # NOTE: automatic discovery is possible, but not yet implemented
        self._loc_is_iloc = loc_is_iloc

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

    def rename(self, name: tp.Hashable) -> 'Index':
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
    def loc(self) -> GetItem:
        return GetItem(self._extract_loc)

    @property
    def iloc(self) -> GetItem:
        return GetItem(self._extract_iloc)

    # # on Index, getitem is an iloc selector; on Series, getitem is a loc selector; for this extraction interface, we do not implement a getitem level function (using iloc would be consistent), as it is better to be explicit between iloc loc

    def _iter_label(self, depth_level: DepthLevelSpecifier = 0):
        yield from self._labels

    def _iter_label_items(self, depth_level: DepthLevelSpecifier = 0):
        yield from zip(self._positions, self._labels)

    @property
    def iter_label(self) -> IterNode:
        return IterNode(
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

    #---------------------------------------------------------------------------

    def _update_array_cache(self):
        '''Derived classes can use this to set stored arrays, self._labels and self._positions.
        '''
        pass

    #---------------------------------------------------------------------------

    def __len__(self) -> int:
        if self._recache:
            self._update_array_cache()
        return len(self._labels)

    def display(self,
            config: tp.Optional[DisplayConfig] = None,
            ) -> Display:

        config = config or DisplayActive.get()

        if self._recache:
            self._update_array_cache()

        return Display.from_values(self.values,
                header=DisplayHeader(self.__class__, self._name),
                config=config,
                outermost=True,
                index_depth=0,
                columns_depth=1
                )

    #---------------------------------------------------------------------------
    # core internal representation

    @property
    def values(self) -> np.ndarray:
        '''Return the immutable labels array
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

    #---------------------------------------------------------------------------

    def copy(self) -> 'Index':
        '''
        Return a new Index.
        '''
        # this is not a complete deepcopy, as _labels here is an immutable np array (a new map will be created); if this is an IndexGO, we will pass the cached, immutable NP array
        if self._recache:
            self._update_array_cache()
        return self.__class__(labels=self)

    def relabel(self, mapper: CallableOrMapping) -> 'Index':
        '''
        Return a new Index with labels replaced by the callable or mapping; order will be retained. If a mapping is used, the mapping need not map all origin keys.
        '''
        if not callable(mapper):
            # if a mapper, it must support both __getitem__ and __contains__
            getitem = getattr(mapper, '__getitem__')
            return self.__class__(getitem(x) if x in mapper else x for x in self._labels)

        return self.__class__(mapper(x) for x in self._labels)

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
            key_transform: A function that transforms keys to specialized type; used by Data indices.
        Returns:
            Return GetItemKey type that is based on integers, compatible with TypeBlocks
        '''
        from static_frame.core.series import Series

        if self._recache:
            self._update_array_cache()

        if isinstance(key, Index):
            # if an Index, we simply use the values of the index
            key = key.values

        if isinstance(key, Series):
            if key.dtype == bool:
                if _requires_reindex(key.index, self):
                    key = key.reindex(self, fill_value=False).values
                else: # the index is equal
                    key = key.values
            else:
                key = key.values

        if self._loc_is_iloc:
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

        return self.__class__(labels=labels)

    def _extract_loc(self, key: GetItemKeyType) -> 'Index':
        return self._extract_iloc(self.loc_to_iloc(key))

    def __getitem__(self, key: GetItemKeyType) -> 'Index':
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

        return self.__class__(labels)

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

        array = operator(self._labels, other)
        array.flags.writeable = False
        return array


    def _ufunc_axis_skipna(self, *,
            axis,
            skipna,
            ufunc,
            ufunc_skipna,
            dtype=None):
        '''Axis argument is required but is irrelevant.

        Args:
            dtype: Not used in 1D application, but collected here to provide a uniform signature.
        '''
        return ufunc_skipna_1d(
                array=self._labels,
                skipna=skipna,
                ufunc=ufunc,
                ufunc_skipna=ufunc_skipna)

    #---------------------------------------------------------------------------
    # dictionary-like interface

    def keys(self) -> KeysView:
        '''
        Iterator of index labels.
        '''
        return self._map.keys()

    def __iter__(self):
        '''Iterate over labels.
        '''
        if self._recache:
            self._update_array_cache()
        return self._labels.__iter__()

    def __contains__(self, value) -> bool:
        '''Return True if value in the labels.
        '''
        return self._map.__contains__(value)

    def items(self) -> tp.Generator[tp.Tuple[tp.Any, tp.Any], None, None]:
        '''Iterator of pairs of index label and value.
        '''
        return self._map.items()

    def get(self, key, default=None):
        '''
        Return the value found at the index key, else the default if the key is not found.
        '''
        return self._map.get(key, default)

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
        return self.__class__(v)

    def isin(self, other: tp.Iterable[tp.Any]) -> np.ndarray:
        '''Return a Boolean array showing True where a label is found in other. If other is a multidimensional array, it is flattened.
        '''
        if self._recache:
            self._update_array_cache()
        v, assume_unique = iterable_to_array(other)
        return np.in1d(self._labels, v, assume_unique=assume_unique)

    def roll(self, shift: int) -> 'Index':
        '''Return an Index with values rotated forward and wrapped around (with a postive shift) or backward and wrapped around (with a negative shift).
        '''
        values = self.values # force usage of property for cache update
        if shift % len(values):
            values = array_shift(values,
                    shift,
                    axis=0,
                    wrap=True)
            values.flags.writeable = False
        # NOTE: could recycle self._map if we know this is an immutable Index
        return self.__class__(values)

    #---------------------------------------------------------------------------
    # export

    def to_series(self):
        '''Return a Series with values from this Index's labels.
        '''
        # not sure if index should be self here
        from static_frame import Series
        return Series(self.values, index=None)

    def add_level(self, level: tp.Hashable) -> 'IndexHierarchy':
        '''Return an IndexHierarhcy with an added root level.
        '''
        from static_frame import IndexHierarchy
        return IndexHierarchy.from_tree({level: self.values})

    def to_pandas(self):
        '''Return a Pandas Index.
        '''
        import pandas
        # must copy to remove immutability, decouple reference
        return pandas.Index(self.values.copy(),
                name=self._name)

#-------------------------------------------------------------------------------
@doc_inject(selector='index_init')
class IndexGO(Index):
    '''A mapping of labels to positions, immutable with grow-only size. Used as columns in :py:class:`FrameGO`.

    {args}
    '''
    STATIC = False
    _IMMUTABLE_CONSTRUCTOR = Index

    __slots__ = (
            '_map',
            '_labels',
            '_positions',
            '_recache',
            '_loc_is_iloc',
            '_name',
            '_labels_mutable',
            '_positions_mutable_count',
            )

    def _extract_labels(self,
            mapping,
            labels,
            dtype) -> tp.Iterable[tp.Any]:
        '''Called in Index.__init__(). This creates and populates mutable storage as a side effect of array derivation.
        '''
        labels = Index._extract_labels(mapping, labels, dtype)
        self._labels_mutable = labels.tolist()
        return labels

    def _extract_positions(self, mapping, positions) -> tp.Iterable[tp.Any]:
        '''Called in Index.__init__(). This creates and populates mutable storage. This creates and populates mutable storage as a side effect of array derivation.
        '''
        positions = Index._extract_positions(mapping, positions)
        self._positions_mutable_count = len(positions)
        return positions


    def _update_array_cache(self):
        # this might fail if a sequence is given as a label
        self._labels = np.array(self._labels_mutable)
        self._labels.flags.writeable = False
        self._positions = np.arange(self._positions_mutable_count)
        self._positions.flags.writeable = False
        self._recache = False

    #---------------------------------------------------------------------------
    # grow only mutation

    def append(self, value):
        '''append a value
        '''
        if value in self._map:
            raise KeyError('duplicate key append attempted', value)
        # the new value is the count
        self._map[value] = self._positions_mutable_count
        self._labels_mutable.append(value)

        # check value before incrementing
        if self._loc_is_iloc:
            if isinstance(value, int) and value == self._positions_mutable_count:
                pass # an increment that keeps loc is iloc relationship
            else:
                self._loc_is_iloc = False

        self._positions_mutable_count += 1
        self._recache = True

    def extend(self, values: KEY_ITERABLE_TYPES):
        '''Append multiple values
        Args:
            values: can be a generator.
        '''
        for value in values:
            if value in self._map:
                raise KeyError('duplicate key append attempted', value)
            # might bet better performance by calling extend() on _positions and _labels
            self.append(value)

#-------------------------------------------------------------------------------
# Specialized index for dates

class _IndexDatetime(Index):

    STATIC = True
    _DTYPE = None # define in base class

    __slots__ = (
            '_map',
            '_labels',
            '_positions',
            '_recache',
            '_loc_is_iloc',
            '_name'
            )

    #---------------------------------------------------------------------------
    # dict like interface

    def __contains__(self, value) -> bool:
        '''Return True if value in the labels. Will only return True for an exact match to the type of dates stored within.
        '''
        return self._map.__contains__(_to_datetime64(value))

    #---------------------------------------------------------------------------
    # operators

    def _ufunc_binary_operator(self, *, operator: tp.Callable, other) -> np.ndarray:

        if self._recache:
            self._update_array_cache()

        if issubclass(other.__class__, Index):
            other = other.values # operate on labels to labels
        elif isinstance(other, str):
            # do not pass dtype, as want to coerce to this parsed type, not the type of sled
            other = _to_datetime64(other)

        if isinstance(other, np.datetime64):
            # convert labels to other's datetime64 type to enable matching on month, year, etc.
            array = operator(self._labels.astype(other.dtype), other)
        else:
            array = operator(self._labels, other)

        array.flags.writeable = False
        return array


    def loc_to_iloc(self, key: GetItemKeyType) -> GetItemKeyType:
        '''
        Specialized for IndexData indices to convert string data representations into np.datetime64 objects as appropriate.
        '''
        # not passing self.dtype to key_to_datetime_key so as to allow of translation to a foreign datetime for comparison
        return Index.loc_to_iloc(self,
                key=key,
                key_transform=key_to_datetime_key)

    #---------------------------------------------------------------------------
    def to_pandas(self):
        '''Return a Pandas Index.
        '''
        import pandas
        # do not need a copy as Pandas will coerce to datetime64
        return pandas.DatetimeIndex(self.values,
                name=self._name)




#-------------------------------------------------------------------------------
@doc_inject(selector='index_init')
class IndexYear(_IndexDatetime):
    '''A mapping of years (via NumPy datetime64[Y]) to positions, immutable and of fixed size.

    {args}
    '''
    STATIC = True
    _DTYPE = _DT64_YEAR

    __slots__ = (
            '_map',
            '_labels',
            '_positions',
            '_recache',
            '_loc_is_iloc',
            '_name',
            )


    @classmethod
    def from_date_range(cls,
            start: DateInitializer,
            stop: DateInitializer,
            step: int = 1):
        '''
        Get an IndexYearMonth instance over a range of dates, where start and stop is inclusive.
        '''
        labels = np.arange(
                _to_datetime64(start, _DT64_DAY),
                _to_datetime64(stop, _DT64_DAY).astype(_DT64_YEAR) + _TD64_YEAR,
                np.timedelta64(step, 'Y'),
                dtype=_DT64_YEAR)

        labels.flags.writeable = False
        return cls(labels)

    @classmethod
    def from_year_month_range(cls,
            start: YearMonthInitializer,
            stop: YearMonthInitializer,
            step: int = 1):
        '''
        Get an IndexYearMonth instance over a range of months, where start and end are inclusive.
        '''

        labels = np.arange(
                _to_datetime64(start, _DT64_MONTH),
                _to_datetime64(stop, _DT64_MONTH).astype(_DT64_YEAR) + _TD64_YEAR,
                np.timedelta64(step, 'Y'),
                dtype=_DT64_YEAR)
        labels.flags.writeable = False
        return cls(labels)


    @classmethod
    def from_year_range(cls,
            start: YearInitializer,
            stop: YearInitializer,
            step: int = 1
            ):
        '''
        Get an IndexDate instance over a range of years, where start and end are inclusive.
        '''
        labels = np.arange(
                _to_datetime64(start, _DT64_YEAR),
                _to_datetime64(stop, _DT64_YEAR) + _TD64_YEAR,
                step=np.timedelta64(step, 'Y'),
                )
        labels.flags.writeable = False
        return cls(labels)

    #---------------------------------------------------------------------------
    def to_pandas(self):
        '''Return a Pandas Index.
        '''
        raise NotImplementedError('Pandas does not support a year type, and it is ambiguous if a date proxy should be the first of the year or the last of the year.')

@doc_inject(selector='index_init')
class IndexYearMonth(_IndexDatetime):
    '''A mapping of year months (via NumPy datetime64[M]) to positions, immutable and of fixed size.

    {args}
    '''
    STATIC = True
    _DTYPE = _DT64_MONTH

    __slots__ = (
            '_map',
            '_labels',
            '_positions',
            '_recache',
            '_loc_is_iloc',
            '_name'
            )

    @classmethod
    def from_date_range(cls,
            start: DateInitializer,
            stop: DateInitializer,
            step: int = 1):
        '''
        Get an IndexYearMonth instance over a range of dates, where start and stop is inclusive.
        '''
        labels = np.arange(
                _to_datetime64(start, _DT64_DAY),
                _to_datetime64(stop, _DT64_DAY).astype(_DT64_MONTH) + _TD64_MONTH,
                np.timedelta64(step, 'M'),
                dtype=_DT64_MONTH)

        labels.flags.writeable = False
        return cls(labels)

    @classmethod
    def from_year_month_range(cls,
            start: YearMonthInitializer,
            stop: YearMonthInitializer,
            step: int = 1):
        '''
        Get an IndexYearMonth instance over a range of months, where start and end are inclusive.
        '''

        labels = np.arange(
                _to_datetime64(start, _DT64_MONTH),
                _to_datetime64(stop, _DT64_MONTH) + _TD64_MONTH,
                np.timedelta64(step, 'M'),
                dtype=_DT64_MONTH)
        labels.flags.writeable = False
        return cls(labels)


    @classmethod
    def from_year_range(cls,
            start: YearInitializer,
            stop: YearInitializer,
            step: int = 1
            ):
        '''
        Get an IndexYearMonth instance over a range of years, where start and end are inclusive.
        '''
        labels = np.arange(
                _to_datetime64(start, _DT64_YEAR),
                _to_datetime64(stop, _DT64_YEAR) + _TD64_YEAR,
                step=np.timedelta64(step, 'M'),
                dtype=_DT64_MONTH)
        labels.flags.writeable = False
        return cls(labels)

    #---------------------------------------------------------------------------
    def to_pandas(self):
        '''Return a Pandas Index.
        '''
        raise NotImplementedError('Pandas does not support a year month type, and it is amiguous if a date proxy should be the first of the month or the last of the month.')


@doc_inject(selector='index_init')
class IndexDate(_IndexDatetime):
    '''A mapping of dates (via NumPy datetime64[D]) to positions, immutable and of fixed size.

    {args}
    '''
    STATIC = True
    _DTYPE = _DT64_DAY

    __slots__ = (
            '_map',
            '_labels',
            '_positions',
            '_recache',
            '_loc_is_iloc',
            '_name'
            )

    @classmethod
    def from_date_range(cls,
            start: DateInitializer,
            stop: DateInitializer,
            step: int = 1):
        '''
        Get an IndexDate instance over a range of dates, where start and stop is inclusive.
        '''
        labels = np.arange(
                _to_datetime64(start, _DT64_DAY),
                _to_datetime64(stop, _DT64_DAY) + _TD64_DAY,
                np.timedelta64(step, 'D'))
        labels.flags.writeable = False
        return cls(labels)

    @classmethod
    def from_year_month_range(cls,
            start: YearMonthInitializer,
            stop: YearMonthInitializer,
            step: int = 1):
        '''
        Get an IndexDate instance over a range of months, where start and end are inclusive.
        '''
        labels = np.arange(
                _to_datetime64(start, _DT64_MONTH),
                _to_datetime64(stop, _DT64_MONTH) + _TD64_MONTH,
                step=np.timedelta64(step, 'D'),
                dtype=_DT64_DAY)
        labels.flags.writeable = False
        return cls(labels)

    @classmethod
    def from_year_range(cls,
            start: YearInitializer,
            stop: YearInitializer,
            step: int = 1
            ):
        '''
        Get an IndexDate instance over a range of years, where start and end are inclusive.
        '''
        labels = np.arange(
                _to_datetime64(start, _DT64_YEAR),
                _to_datetime64(stop, _DT64_YEAR) + _TD64_YEAR,
                step=np.timedelta64(step, 'D'),
                dtype=_DT64_DAY)
        labels.flags.writeable = False
        return cls(labels)


#-------------------------------------------------------------------------------
@doc_inject(selector='index_init')
class IndexSecond(_IndexDatetime):
    '''A mapping of time stamps at the resolution of seconds (via NumPy datetime64[s]) to positions, immutable and of fixed size.

    {args}
    '''
    STATIC = True
    _DTYPE = _DT64_S

    __slots__ = (
            '_map',
            '_labels',
            '_positions',
            '_recache',
            '_loc_is_iloc',
            '_name',
            )

@doc_inject(selector='index_init')
class IndexMillisecond(_IndexDatetime):
    '''A mapping of time stamps at the resolutoin of milliseconds (via NumPy datetime64[ms]) to positions, immutable and of fixed size.

    {args}
    '''
    STATIC = True
    _DTYPE = _DT64_MS

    __slots__ = (
            '_map',
            '_labels',
            '_positions',
            '_recache',
            '_loc_is_iloc',
            '_name',
            )



#-------------------------------------------------------------------------------

def _is_index_initializer(value) -> bool:
    '''Determine if value is a non-empty index initializer. This could almost just be a truthy test, but ndarrays need to be handled in isolation. Generators should return True.
    '''
    if value is None:
        return False
    if isinstance(value, Index):
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

