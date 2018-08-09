import typing as tp
import numpy as np

from static_frame.core.util import _DEFAULT_SORT_KIND
from static_frame.core.util import _NULL_SLICE

from static_frame.core.util import _KEY_ITERABLE_TYPES
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import CallableOrMapping
from static_frame.core.util import DtypeSpecifier
# from static_frame.core.util import IndexSpecifier
from static_frame.core.util import IndexInitializer

from static_frame.core.util import mloc
from static_frame.core.util import _ufunc_skipna_1d
from static_frame.core.util import _iterable_to_array

from static_frame.core.util import GetItem

from static_frame.core.operator_delegate import MetaOperatorDelegate

from static_frame.core.display import DisplayConfig
from static_frame.core.display import DisplayActive
from static_frame.core.display import Display

from static_frame.core.type_blocks import TypeBlocks


#-------------------------------------------------------------------------------
class LocMap:

    SLICE_STOP_ATTR = 'stop'
    SLICE_ATTRS = ('start', SLICE_STOP_ATTR, 'step')

    @classmethod
    def map_slice_args(cls, label_to_pos, key: slice):
        '''Given a slice and a label to position mapping, yield each argument necessary to create a new slice.

        Args:
            label_to_pos: mapping, no order dependency
        '''
        for field in cls.SLICE_ATTRS:
            attr = getattr(key, field)
            if attr is None:
                yield None
            else:
                if field == cls.SLICE_STOP_ATTR:
                    # loc selections are inclusive, so iloc gets one more
                    yield label_to_pos[attr] + 1
                else:
                    yield label_to_pos[attr]

    @classmethod
    def loc_to_iloc(cls,
            label_to_pos: tp.Dict,
            positions: np.ndarray,
            key: GetItemKeyType) -> GetItemKeyType:
        '''
        Returns:
            An integer mapped slice, or GetItemKey type that is based on integers, compatible with TypeBlocks
        '''

        if isinstance(key, slice):
            return slice(*cls.map_slice_args(label_to_pos, key))

        elif isinstance(key, _KEY_ITERABLE_TYPES):

            # can be an iterable of labels (keys) or an iterable of Booleans
            # if len(key) == len(label_to_pos) and isinstance(key[0], (bool, np.bool_)):
            if isinstance(key, np.ndarray) and key.dtype == bool:
                return positions[key]

            # map labels to integer positions
            # NOTE: we miss the opportunity to get a reference from values when we have contiguous keys
            return [label_to_pos[x] for x in key]

        # if a single element (an integer, string, or date, we just get the integer out of the map
        return label_to_pos[key]



#-------------------------------------------------------------------------------

class Index(metaclass=MetaOperatorDelegate):
    '''A mapping of labels to positions, immutable and of fixed size. Used in :py:class:`Series` and as index and columns in :py:class:`Frame`.

    Args:
        labels: Iterable of values to be used as the index.
        loc_is_iloc: Optimization for when a contiguous integer index is provided as labels. Generally only set by internal clients.
        dtype: Optional dytpe to be used for labels.
    '''
    STATIC = True

    __slots__ = (
            '_map',
            '_labels',
            '_positions',
            '_recache',
            '_loc_is_iloc',
            'loc',
            'iloc',
            )

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
            labels: might be an expired Generator, but if it is an immutable npdarry, we can use it without a copy
        '''
        # pre-fetching labels for faster get_item construction
        if isinstance(labels, np.ndarray): # if an np array can handle directly
            labels = TypeBlocks.immutable_filter(labels)
        elif hasattr(labels, '__len__'): # not a generator, not an array
            labels = np.array(labels, dtype)
            labels.flags.writeable = False
        else: # labels may be an expired generator
            # until all Python dictionaries are ordered, we cannot just take keys()
            # labels = np.array(tuple(mapping.keys()))
            # assume object type so as to not create a temporary list
            labels = np.empty(len(mapping),
                    dtype=dtype if dtype else object)
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
            return TypeBlocks.immutable_filter(positions)
        positions = np.arange(len(mapping))
        positions.flags.writeable = False
        return positions

    @staticmethod
    def _get_map(labels, positions=None) -> tp.Dict[tp.Any, int]:
        '''
        Return a dictionary mapping index labels to integer positions.

        NOTE: this function is critical to Index performance.
        '''
        if positions is not None:
            return dict(zip(labels, positions))
        if hasattr(labels, '__len__'):
            return dict(zip(labels, range(len(labels))))
        # support labels as a generator
        return {v: k for k, v in enumerate(labels)}


    #---------------------------------------------------------------------------
    def __init__(self,
            labels: IndexInitializer,
            loc_is_iloc: bool=False,
            dtype: DtypeSpecifier=None
            ) -> None:

        self._recache = False
        self._map = None

        positions = None
        if issubclass(labels.__class__, Index):
            # get a reference to the immutable arrays
            # even if this is an IndexGO index, we can take the cached arrays, assuming they are up to date
            if labels.STATIC:
                # can take the map
                self._map = labels._map

            if labels._recache:
                labels._update_array_cache()

            positions = labels._positions
            loc_is_iloc = labels._loc_is_iloc
            labels = labels._labels

        if self._map is None:
            self._map = self._get_map(labels, positions)

        # this might be NP array, or a list, depending on if static or grow only
        self._labels = self._extract_labels(self._map, labels, dtype)
        self._positions = self._extract_positions(self._map, positions)

        # NOTE:  automatic discovery is possible
        self._loc_is_iloc = loc_is_iloc

        if len(self._map) != len(self._labels):
            raise KeyError('labels have non-unique values')

        self.loc = GetItem(self._extract_loc)
        self.iloc = GetItem(self._extract_iloc)


    def _update_array_cache(self):
        '''Derived classes can use this to set stored arrays, self._labels and self._positions.
        '''
        pass

    def display(self, config: DisplayConfig=None) -> Display:
        config = config or DisplayActive.get()

        if self._recache:
            self._update_array_cache()

        return Display.from_values(self.values,
                header='<' + self.__class__.__name__ + '>',
                config=config)

    def __repr__(self) -> str:
        return repr(self.display())


    def loc_to_iloc(self, key: GetItemKeyType) -> GetItemKeyType:
        '''
        Returns:
            Return GetItemKey type that is based on integers, compatible with TypeBlocks
        '''
        from static_frame.core.series import Series

        if self._recache:
            self._update_array_cache()

        if isinstance(key, Series):
            key = key.values

        if self._loc_is_iloc:
            return key

        return LocMap.loc_to_iloc(self._map, self._positions, key)

    def __len__(self) -> int:
        if self._recache:
            self._update_array_cache()
        return len(self._labels)

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


    @property
    def values(self) -> np.ndarray:
        '''Return the immutable labels array
        '''
        if self._recache:
            self._update_array_cache()
        return self._labels

    @property
    def mloc(self):
        '''Memory location
        '''
        if self._recache:
            self._update_array_cache()
        return mloc(self._labels)

    def copy(self) -> 'Index':
        '''
        Return a new Index.
        '''
        # this is not a complete deepcopy, as _labels here is an immutable np array (a new map will be created); if this is an IndexGO, we will pass the cached, immutable NP array
        if self._recache:
            self._update_array_cache()
        # return self.__class__(labels=self._labels)
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
    # set operations

    def intersection(self, other) -> 'Index':
        if self._recache:
            self._update_array_cache()

        if isinstance(other, np.ndarray):
            opperand = other
        else: # assume we can get it from a .values attribute
            opperand = other.values

        return self.__class__(labels=np.intersect1d(self._labels, opperand))

    def union(self, other) -> 'Index':
        if self._recache:
            self._update_array_cache()

        if isinstance(other, np.ndarray):
            opperand = other
        else: # assume we can get it from a .values attribute
            opperand = other.values

        return self.__class__(labels=np.union1d(self._labels, opperand))

    #---------------------------------------------------------------------------
    # extraction and selection

    def _extract_iloc(self, key) -> 'Index':
        '''Extract a new index given an iloc key
        '''
        if self._recache:
            self._update_array_cache()

        if key is None:
            labels = self._labels
        elif isinstance(key, slice):
            if key == _NULL_SLICE:
                labels = self._labels
            else:
                # if labels is an np array, this will be a view; if a list, a copy
                labels = self._labels[key]
        elif isinstance(key, _KEY_ITERABLE_TYPES):
            # we assume Booleans have been normalized to integers here
            # can select directly from _labels[key] if if key is a list
            labels = self._labels[key]
        else: # select a single label value
            labels = (self._labels[key],)
        return self.__class__(labels=labels)

    def _extract_loc(self, key: GetItemKeyType) -> 'Index':
        return self._extract_iloc(self.loc_to_iloc(key))

    def __getitem__(self, key: GetItemKeyType) -> 'Index':
        '''Extract a new index given an iloc key.
        '''
        return self._extract_iloc(key)

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


    def _ufunc_axis_skipna(self, *, axis, skipna, ufunc, ufunc_skipna, dtype=None):
        '''Axis argument is required but is irrelevant.

        Args:
            dtype: Not used in 1D application, but collected here to provide a uniform signature.
        '''
        return _ufunc_skipna_1d(
                array=self._labels,
                skipna=skipna,
                ufunc=ufunc,
                ufunc_skipna=ufunc_skipna)

    #---------------------------------------------------------------------------
    # utility functions

    def sort(self,
            ascending: bool=True,
            kind: str=_DEFAULT_SORT_KIND) -> 'Index':
        '''Return a new Index with the labels sorted.

        Args:
            kind: Sort algorithm passed to NumPy.
        '''
        v = np.sort(self._labels, kind=kind)
        if not ascending:
            v = v[::-1]
        v.flags.writeable = False
        return __class__(v)

    def isin(self, other: tp.Iterable[tp.Any]) -> np.ndarray:
        '''Return a Boolean array showing True where a label is found in other. If other is a multidimensional array, it is flattened.
        '''
        if self._recache:
            self._update_array_cache()
        v, assume_unique = _iterable_to_array(other)
        return np.in1d(self._labels, v, assume_unique=assume_unique)


class IndexGO(Index):
    '''
    A mapping of labels to positions, immutable with grow-only size. Used as columns in :py:class:`FrameGO`. Initialization arguments are the same as for :py:class:`Index`.
    '''
    STATIC = False

    __slots__ = (
            '_map',
            '_labels_mutable',
            '_positions_mutable_count',
            '_labels',
            '_positions',
            '_recache',
            'iloc',
            )

    def _extract_labels(self,
            mapping,
            labels,
            dtype) -> tp.Iterable[tp.Any]:
        '''Called in Index.__init__(). This creates and populates mutable storage as a side effect of array derivation.
        '''
        labels = super(IndexGO, self)._extract_labels(mapping, labels, dtype)
        self._labels_mutable = labels.tolist()
        return labels

    def _extract_positions(self, mapping, positions) -> tp.Iterable[tp.Any]:
        '''Called in Index.__init__(). This creates and populates mutable storage. This creates and populates mutable storage as a side effect of array derivation.
        '''
        positions = super(IndexGO, self)._extract_positions(mapping, positions)
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
        '''Add a value
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


    def extend(self, values: _KEY_ITERABLE_TYPES):
        '''Add multiple values
        Args:
            values: can be a generator.
        '''
        for value in values:
            if value in self._map:
                raise KeyError('duplicate key append attempted', value)
            # might bet better performance by calling extend() on _positions and _labels
            self.append(value)

