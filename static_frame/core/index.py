import typing as tp
from collections.abc import KeysView
import operator as operator_mod
from itertools import zip_longest
from functools import reduce

import numpy as np

from automap import AutoMap
from automap import FrozenAutoMap


from static_frame.core.container import ContainerOperand
from static_frame.core.container_util import apply_binary_operator
from static_frame.core.container_util import matmul
from static_frame.core.container_util import key_from_container_key

from static_frame.core.display import Display
from static_frame.core.display import DisplayActive
from static_frame.core.display_config import DisplayConfig
from static_frame.core.display import DisplayHeader
from static_frame.core.doc_str import doc_inject
from static_frame.core.exception import ErrorInitIndex
from static_frame.core.exception import ErrorInitIndexNonUnique
from static_frame.core.exception import LocEmpty
from static_frame.core.exception import LocInvalid
from static_frame.core.index_base import IndexBase
from static_frame.core.node_dt import InterfaceDatetime
from static_frame.core.node_iter import IterNodeApplyType
from static_frame.core.node_iter import IterNodeDepthLevel
from static_frame.core.node_iter import IterNodeType
from static_frame.core.node_selector import InterfaceGetItem
from static_frame.core.node_selector import InterfaceSelectDuo
from static_frame.core.node_selector import TContainer
from static_frame.core.node_str import InterfaceString

from static_frame.core.util import array_shift
from static_frame.core.util import array2d_to_tuples
from static_frame.core.util import BOOL_TYPES
from static_frame.core.util import CallableOrMapping
from static_frame.core.util import dtype_from_element
from static_frame.core.util import DEFAULT_SORT_KIND
from static_frame.core.util import DepthLevelSpecifier
from static_frame.core.util import DTYPE_DATETIME_KIND
from static_frame.core.util import DTYPE_INT_DEFAULT
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import DtypeSpecifier
from static_frame.core.util import EMPTY_ARRAY
from static_frame.core.util import EMPTY_SLICE
from static_frame.core.util import EMPTY_TUPLE
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import immutable_filter
from static_frame.core.util import IndexInitializer
from static_frame.core.util import INT_TYPES
from static_frame.core.util import intersect1d
from static_frame.core.util import isin
from static_frame.core.util import isna_array
from static_frame.core.util import iterable_to_array_1d
from static_frame.core.util import KEY_ITERABLE_TYPES
from static_frame.core.util import KeyIterableTypes
from static_frame.core.util import KeyTransformType
from static_frame.core.util import mloc
from static_frame.core.util import NAME_DEFAULT
from static_frame.core.util import name_filter
from static_frame.core.util import NameType
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import resolve_dtype
from static_frame.core.util import setdiff1d
from static_frame.core.util import SLICE_ATTRS
from static_frame.core.util import SLICE_START_ATTR
from static_frame.core.util import SLICE_STEP_ATTR
from static_frame.core.util import SLICE_STOP_ATTR
from static_frame.core.util import slice_to_inclusive_slice
from static_frame.core.util import to_datetime64
from static_frame.core.util import UFunc
from static_frame.core.util import ufunc_axis_skipna
from static_frame.core.util import union1d

if tp.TYPE_CHECKING:
    import pandas #pylint: disable=W0611 #pragma: no cover
    from static_frame import Series #pylint: disable=W0611 #pragma: no cover
    from static_frame import IndexHierarchy #pylint: disable=W0611 #pragma: no cover
    from static_frame.core.index_auto import RelabelInput #pylint: disable=W0611 #pragma: no cover

I = tp.TypeVar('I', bound=IndexBase)


class ILocMeta(type):

    def __getitem__(cls,
            key: GetItemKeyType
            ) -> tp.Iterable[GetItemKeyType]:
        return cls(key) #type: ignore

class ILoc(metaclass=ILocMeta):
    '''A wrapper for embedding ``iloc`` specifications within a single axis argument of a ``loc`` selection.
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
            ) -> tp.Iterator[tp.Union[int, None]]:
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
                if attr.dtype == labels.dtype: #type: ignore
                    if field != SLICE_STEP_ATTR:
                        pos: int = label_to_pos(attr)
                        if pos is None:
                            # if same type, and that atter is not in labels, we fail, just as we do in then non-datetime64 case. Only when datetimes are given in a different unit are we "loose" about matching.
                            raise LocInvalid('Invalid loc given in a slice', attr, field)
                    else: # step
                        pos = attr # should be an integer

                    if field == SLICE_STOP_ATTR:
                        pos += 1 # stop is inclusive

                elif field == SLICE_START_ATTR:
                    # convert to the type of the atrs; this should get the relevant start
                    pos = label_to_pos(attr.astype(labels.dtype)) #type: ignore
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
                    matches = np.flatnonzero(labels.astype(attr.dtype) == attr) #type: ignore
                    if len(matches):
                        pos = matches[-1] + 1
                    else:
                        raise LocEmpty()

                elif field == SLICE_STEP_ATTR:
                    pos = attr

                if offset_apply and field != SLICE_STEP_ATTR:
                    pos += offset #type: ignore

                yield pos

            else:
                if field != SLICE_STEP_ATTR:
                    pos = label_to_pos(attr)
                    if pos is None:
                        # NOTE: could raise LocEmpty() to silently handle this
                        raise LocInvalid('Invalid loc given in a slice', attr, field)
                    if offset_apply:
                        pos += offset #type: ignore
                else: # step
                    pos = attr # should be an integer

                if field == SLICE_STOP_ATTR:
                    # loc selections are inclusive, so iloc gets one more
                    pos += 1

                yield pos

    @classmethod
    def loc_to_iloc(cls, *,
            label_to_pos: tp.Dict[tp.Hashable, int],
            labels: np.ndarray,
            positions: np.ndarray,
            key: GetItemKeyType,
            offset: tp.Optional[int] = None,
            partial_selection: bool = False,
            ) -> GetItemKeyType:
        '''
        Note: all SF objects (Series, Index) need to be converted to basic types before being passed as `key` to this function.

        Args:
            offset: in the contect of an IndexHierarchical, the iloc positions returned from this funcition need to be shifted.
            partial_selection: if True and key is an iterable of labels that includes lables not in the mapping, available matches will be returned rather than raising.
        Returns:
            An integer mapped slice, or GetItemKey type that is based on integers, compatible with TypeBlocks
        '''
        offset_apply = not offset is None

        # ILoc is handled prior to this call, in the Index.loc_to_iloc method

        if isinstance(key, slice):
            if offset_apply and key == NULL_SLICE:
                # when offset is defined (even if it is zero), null slice is not sufficiently specific; need to convert to an explict slice relative to the offset
                return slice(offset, len(positions) + offset) #type: ignore
            try:
                return slice(*cls.map_slice_args(
                        label_to_pos.get, #type: ignore
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

        is_array = isinstance(key, np.ndarray)
        is_list = isinstance(key, list)

        # can be an iterable of labels (keys) or an iterable of Booleans
        if is_array or is_list:
            if is_array and key.dtype.kind == DTYPE_DATETIME_KIND:
                if labels.dtype != key.dtype:
                    labels_ref = labels.astype(key.dtype)
                    # let Boolean key advance to next branch
                    key = reduce(operator_mod.or_, (labels_ref == k for k in key))

            if is_array and key.dtype == DTYPE_BOOL:
                if offset_apply:
                    return positions[key] + offset
                return positions[key]

            # map labels to integer positions, return a list of integer positions
            # NOTE: we may miss the opportunity to identify contiguous keys and extract a slice
            # NOTE: we do more branching here to optimize performance
            if partial_selection:
                if offset_apply:
                    return [label_to_pos[k] + offset for k in key if k in label_to_pos] #type: ignore
                return [label_to_pos[k] for k in key if k in label_to_pos]
            if offset_apply:
                return [label_to_pos[k] + offset for k in key] #type: ignore
            return [label_to_pos[k] for k in key]

        # if a single element (an integer, string, or date, we just get the integer out of the map
        if offset_apply:
            return label_to_pos[key] + offset #type: ignore
        return label_to_pos[key]


def immutable_index_filter(index: I) -> IndexBase:
    '''Return an immutable index. All index objects handle converting from mutable to immutable via the __init__ constructor; but need to use appropriate class between Index and IndexHierarchy.'''

    if index.STATIC:
        return index
    return index._IMMUTABLE_CONSTRUCTOR(index)


def mutable_immutable_index_filter(
        target_static: bool,
        index: I
        ) -> IndexBase:
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

class Index(IndexBase):
    '''A mapping of labels to positions, immutable and of fixed size. Used by default in :obj:`Series` and as index and columns in :obj:`Frame`. Base class of all 1D indices.'''

    __slots__ = _INDEX_SLOTS

    # _IMMUTABLE_CONSTRUCTOR is None from IndexBase
    # _MUTABLE_CONSTRUCTOR will be set after IndexGO defined

    _UFUNC_UNION = union1d
    _UFUNC_INTERSECTION = intersect1d
    _UFUNC_DIFFERENCE = setdiff1d

    _DTYPE: tp.Optional[np.dtype] = None # for specialized indices requiring a typed labels

    # for compatability with IndexHierarchy, where this is implemented as a property method
    depth: int = 1

    _map: tp.Optional[FrozenAutoMap]
    _labels: np.ndarray
    _positions: np.ndarray
    _recache: bool
    _name: NameType

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
                raise ErrorInitIndex('invalid label dtype for this Index')
            return immutable_filter(labels)

        # labels may be an expired generator, must use the mapping
        labels_src = labels if hasattr(labels, '__len__') else mapping

        if len(labels_src) == 0: #type: ignore
            if dtype is None:
                labels = EMPTY_ARRAY
            else:
                labels = np.empty(0, dtype=dtype)
                labels.flags.writeable = False #type: ignore
        else: # resolving the dtype is expensive, pass if possible
            labels, _ = iterable_to_array_1d(labels_src, dtype=dtype) #type: ignore

        return labels

    @staticmethod
    def _extract_positions(
            size: int,
            positions: tp.Optional[tp.Sequence[int]]
            ) -> np.ndarray:
        # positions is either None or an ndarray
        if isinstance(positions, np.ndarray):
            return immutable_filter(positions)
        return PositionsAllocator.get(size)

    #---------------------------------------------------------------------------
    # constructors

    @classmethod
    def from_labels(cls: tp.Type[I],
            labels: tp.Iterable[tp.Sequence[tp.Hashable]],
            *,
            name: NameType = None
            ) -> I:
        '''
        Construct an ``Index`` from an iterable of labels, where each label is a hashable. Provided for a compatible interface to ``IndexHierarchy``.
        '''
        return cls(labels, name=name)

    #---------------------------------------------------------------------------
    @doc_inject(selector='index_init')
    def __init__(self,
            labels: IndexInitializer,
            *,
            loc_is_iloc: bool = False,
            name: NameType = NAME_DEFAULT,
            dtype: DtypeSpecifier = None
            ) -> None:
        '''Initializer.

        {args}
        '''
        self._recache: bool = False
        self._map: tp.Optional[FrozenAutoMap] = None

        positions = None
        is_typed = self._DTYPE is not None # only True for datetime64 indices

        # resolve the targetted labels dtype, by lookin at the class attr _DTYPE and/or the passed dtype argument
        if dtype is None:
            dtype_extract = self._DTYPE # set in some specialized Index classes
        else: # passed dtype is not None
            if is_typed and dtype != self._DTYPE:
                # NOTE: should never get to this branch, as derived Index classes that set _DTYPE remove dtype from __init__
                raise ErrorInitIndex('invalid dtype argument for this Index', dtype, self._DTYPE) #pragma: no cover
            # self._DTYPE is None, passed dtype is not None, use dtype
            dtype_extract = dtype

        #-----------------------------------------------------------------------
        # handle all Index subclasses
        if isinstance(labels, IndexBase):
            if labels._recache:
                labels._update_array_cache()
            if name is NAME_DEFAULT:
                name = labels.name # immutable, so no copy necessary
            if isinstance(labels, Index): # not an IndexHierarchy
                if (labels.STATIC and self.STATIC and dtype is None):
                    if not is_typed or (is_typed and self._DTYPE == labels.dtype):
                        # can take the map if static and if types in the dict are the same as those in the labels (or to become the labels after conversion)
                        self._map = labels._map
                # get a reference to the immutable arrays, even if this is an IndexGO index, we can take the cached arrays, assuming they are up to date; for datetime64 indices, we might need to translate to a different type
                positions = labels._positions
                loc_is_iloc = labels._map is None
                labels = labels._labels
            else: # IndexHierarchy
                # will be a generator of tuples; already updated caches
                labels = array2d_to_tuples(labels.__iter__())
        elif isinstance(labels, ContainerOperand):
            # it is a Series or similar
            array = labels.values # NOTE: should we take values or keys here?
            if array.ndim == 1:
                labels = array
            else:
                labels = array2d_to_tuples(array)
        # else: assume an iterable suitable for labels usage

        #-----------------------------------------------------------------------
        if is_typed:
            # do not need to check arrays, as will and checked to match dtype_extract in _extract_labels
            if not isinstance(labels, np.ndarray):
                # for now, assume that if _DTYPE is defined, we have a date
                labels = (to_datetime64(v, dtype_extract) for v in labels)
            # coerce to target type
            elif labels.dtype != dtype_extract:
                labels = labels.astype(dtype_extract)
                labels.flags.writeable = False #type: ignore

        self._name = None if name is NAME_DEFAULT else name_filter(name)

        if self._map is None: # if _map not shared from another Index
            if not loc_is_iloc:
                try:
                    self._map = FrozenAutoMap(labels) if self.STATIC else AutoMap(labels)
                except ValueError: # Automap will raise ValueError of non-unique values are encountered
                    pass
                if self._map is None:
                    raise ErrorInitIndexNonUnique(
                            f'labels ({len(tuple(labels))}) have non-unique values ({len(set(labels))})'
                            )
                size = len(self._map)
            else: # must assume labels are unique
                # labels must not be a generator, but we assume that internal clients that provided loc_is_iloc will not give a generator
                size = len(labels) #type: ignore
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


    #---------------------------------------------------------------------------
    def __setstate__(self, state: tp.Tuple[None, tp.Dict[str, tp.Any]]) -> None:
        '''
        Ensure that reanimated NP arrays are set not writeable.
        '''
        for key, value in state[1].items():
            setattr(self, key, value)
        self._labels.flags.writeable = False

    #---------------------------------------------------------------------------
    # name interface

    def rename(self: I, name: NameType) -> I:
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
    def loc(self) -> InterfaceGetItem[TContainer]:
        return InterfaceGetItem(self._extract_loc) #type: ignore

    @property
    def iloc(self) -> InterfaceGetItem[TContainer]:
        return InterfaceGetItem(self._extract_iloc) #type: ignore

    # # on Index, getitem is an iloc selector; on Series, getitem is a loc selector; for this extraction interface, we do not implement a getitem level function (using iloc would be consistent), as it is better to be explicit between iloc loc

    def _iter_label(self,
            depth_level: DepthLevelSpecifier = 0
            ) -> tp.Iterator[tp.Hashable]:
        yield from self._labels

    def _iter_label_items(self,
            depth_level: DepthLevelSpecifier = 0
            ) -> tp.Iterator[tp.Tuple[int, tp.Hashable]]:
        yield from zip(self._positions, self._labels)

    @property
    def iter_label(self) -> IterNodeDepthLevel[tp.Any]:
        return IterNodeDepthLevel(
                container=self,
                function_items=self._iter_label_items,
                function_values=self._iter_label,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.INDEX_LABELS
                )


    #---------------------------------------------------------------------------
    # common attributes from the numpy array

    @property # type: ignore
    @doc_inject()
    def mloc(self) -> int:
        '''{doc_int}
        '''
        if self._recache:
            self._update_array_cache()
        return mloc(self._labels)

    @property
    def dtype(self) -> np.dtype:
        '''
        Return the dtype of the underlying NumPy array.

        Returns:
            :obj:`numpy.dtype`
        '''
        if self._recache:
            self._update_array_cache()
        return self._labels.dtype

    @property
    def shape(self) -> tp.Tuple[int, ...]:
        '''
        Return a tuple describing the shape of the underlying NumPy array.

        Returns:
            :obj:`tp.Tuple[int]`
        '''
        if self._recache:
            self._update_array_cache()
        return tp.cast(tp.Tuple[int, ...], self._labels.shape)

    @property
    def ndim(self) -> int:
        '''
        Return the number of dimensions.

        Returns:
            :obj:`int`
        '''
        if self._recache:
            self._update_array_cache()
        return tp.cast(int, self._labels.ndim)

    @property
    def size(self) -> int:
        '''
        Return the size of the underlying NumPy array.

        Returns:
            :obj:`int`
        '''
        if self._recache:
            self._update_array_cache()
        return tp.cast(int, self._labels.size)

    @property
    def nbytes(self) -> int:
        '''
        Return the total bytes of the underlying NumPy array.

        Returns:
            :obj:`int`
        '''
        if self._recache:
            self._update_array_cache()
        return tp.cast(int, self._labels.nbytes)

    # def __bool__(self) -> bool:
    #     '''
    #     True if this container has size.
    #     '''
    #     if self._recache:
    #         self._update_array_cache()
    #     return bool(self._labels.size)

    #---------------------------------------------------------------------------
    # set operations

    def _ufunc_set(self: I,
            func: tp.Callable[[np.ndarray, np.ndarray, bool], np.ndarray],
            other: tp.Union['IndexBase', tp.Iterable[tp.Hashable]]
            ) -> I:
        '''
        Utility function for preparing and collecting values for Indices to produce a new Index.
        '''
        if self._recache:
            self._update_array_cache()

        if self.equals(other, compare_dtype=True):
            # compare dtype as result should be resolved, even if values are the same
            if (func is self.__class__._UFUNC_INTERSECTION or
                    func is self.__class__._UFUNC_UNION):
                # NOTE: this will delegate name attr
                return self if self.STATIC else self.copy()
            elif func is self.__class__._UFUNC_DIFFERENCE:
                if self._DTYPE is None: #type: ignore
                    # an index with a variable dtype accepts a dtype argument
                    return self.__class__((), dtype=self.dtype) #type: ignore
                # if self._DTYPE is defined, the default constructor does not take a dtype argument
                return self.__class__(())

        if isinstance(other, np.ndarray):
            operand = other
            assume_unique = False
        elif isinstance(other, IndexBase):
            operand = other.values
            assume_unique = True # can always assume unique
        else:
            operand, assume_unique = iterable_to_array_1d(other)

        cls = self.__class__

        # using assume_unique will permit retaining order when operands are identical
        labels = func(self.values, operand, assume_unique=assume_unique) # type: ignore
        if id(labels) == id(self.values):
            # NOTE: favor using cls constructor here as it permits maximal sharing of static resources and the underlying dictionary
            return cls(self)

        return cls.from_labels(labels)


    #---------------------------------------------------------------------------
    def _drop_iloc(self, key: GetItemKeyType) -> 'Index':
        '''Create a new index after removing the values specified by the loc key.
        '''
        if self._recache:
            self._update_array_cache()

        if key is None:
            if self.STATIC: # immutable, no selection, can return self
                return self
            labels = self._labels # already immutable
        elif isinstance(key, np.ndarray) and key.dtype == bool:
            # can use labels, as we already recached
            # use Boolean area to select indices from positions, as np.delete does not work with arrays
            labels = np.delete(self._labels, self._positions[key], axis=0)
            labels.flags.writeable = False
        else:
            labels = np.delete(self._labels, key, axis=0)
            labels.flags.writeable = False

        # from labels will work with both Index and IndexHierarchy
        return self.__class__.from_labels(labels, name=self._name)

    def _drop_loc(self, key: GetItemKeyType) -> 'IndexBase':
        '''Create a new index after removing the values specified by the loc key.
        '''
        return self._drop_iloc(self.loc_to_iloc(key))


    @property
    def drop(self) -> InterfaceSelectDuo[TContainer]:
        return InterfaceSelectDuo( #type: ignore
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
        from static_frame.core.index_datetime import _dtype_to_index_cls
        array = self.values.astype(dtype)
        cls = _dtype_to_index_cls(self.STATIC, array.dtype)
        return cls(
                array,
                name=self._name
                )


    #---------------------------------------------------------------------------
    @property
    def via_str(self) -> InterfaceString[np.ndarray]:
        '''
        Interface for applying string methods to elements in this container.
        '''
        if self._recache:
            self._update_array_cache()

        blocks = (self._labels,)
        cls = Index if self.STATIC else IndexGO

        def blocks_to_container(blocks: tp.Iterator[np.ndarray]) -> np.ndarray:
            return next(blocks)

        return InterfaceString(
                blocks=blocks,
                blocks_to_container=blocks_to_container,
                )

    @property
    def via_dt(self) -> InterfaceDatetime[np.ndarray]:
        '''
        Interface for applying datetime properties and methods to elements in this container.
        '''
        if self._recache:
            self._update_array_cache()

        blocks = (self.values,)
        cls = Index if self.STATIC else IndexGO

        def blocks_to_container(blocks: tp.Iterator[np.ndarray]) -> np.ndarray:
            return next(blocks)

        return InterfaceDatetime(
                blocks=blocks,
                blocks_to_container=blocks_to_container,
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

        header: tp.Optional[DisplayHeader]

        if config.type_show:
            header = DisplayHeader(self.__class__, self._name)
            header_depth = 1
        else:
            header = None
            header_depth = 0

        return Display.from_values(self.values,
                header=header,
                config=config,
                outermost=True,
                index_depth=0,
                header_depth=header_depth,
                )

    #---------------------------------------------------------------------------
    # core internal representation

    @property #type: ignore
    @doc_inject(selector='values_1d', class_name='Index')
    def values(self) -> np.ndarray:
        '''
        {}
        '''
        if self._recache:
            self._update_array_cache()
        return self._labels

    @property
    def positions(self) -> np.ndarray:
        '''Return the immutable positions array.
        '''
        # This is needed by some clients, such as Series and Frame, to support Boolean usage in drop.
        if self._recache:
            self._update_array_cache()
        return self._positions

    @staticmethod
    def _depth_level_validate(depth_level: DepthLevelSpecifier) -> None:
        '''
        Handle all variety of depth_level specifications for a 1D index: only 0, -1, and lists of the same are valid.
        '''
        if not isinstance(depth_level, int):
            depth_level = tuple(depth_level)
            if len(depth_level) != 1:
                raise RuntimeError('invalid depth_level', depth_level)
            depth_level = depth_level[0]

        if depth_level > 0 or depth_level < -1:
            raise RuntimeError('invalid depth_level', depth_level)

    def values_at_depth(self,
            depth_level: DepthLevelSpecifier = 0
            ) -> np.ndarray:
        '''
        Return an NP array for the `depth_level` specified.
        '''
        self._depth_level_validate(depth_level)
        return self.values

    @doc_inject()
    def label_widths_at_depth(self,
            depth_level: DepthLevelSpecifier = 0
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, int]]:
        '''{}'''
        self._depth_level_validate(depth_level)
        yield from zip_longest(self.values, EMPTY_TUPLE, fillvalue=1)


    #---------------------------------------------------------------------------

    def copy(self: I) -> I:
        '''
        Return a new Index.
        '''
        # this is not a complete deepcopy, as _labels here is an immutable np array (a new map will be created); if this is an IndexGO, we will pass the cached, immutable NP array
        if self._recache:
            self._update_array_cache()

        return self.__class__(self, name=self._name)

    def relabel(self, mapper: 'RelabelInput') -> 'Index':
        '''
        Return a new Index with labels replaced by the callable or mapping; order will be retained. If a mapping is used, the mapping need not map all origin keys.
        '''
        if self._recache:
            self._update_array_cache()

        if not callable(mapper):
            # if a mapper, it must support both __getitem__ and __contains__
            getitem = getattr(mapper, '__getitem__')
            return self.__class__(
                    (getitem(x) if x in mapper else x for x in self._labels),
                    name=self._name
                    )

        return self.__class__(
                (mapper(x) for x in self._labels), #type: ignore
                name=self._name
                )

    #---------------------------------------------------------------------------
    # extraction and selection

    def loc_to_iloc(self,
            key: GetItemKeyType,
            offset: tp.Optional[int] = None,
            key_transform: KeyTransformType = None,
            partial_selection: bool = False,
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

        key = key_from_container_key(self, key)

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
                offset=offset,
                partial_selection=partial_selection,
                )

    def _extract_iloc(self,
            key: GetItemKeyType
            ) -> tp.Union['Index', tp.Hashable]:
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
            return self._labels[key] #type: ignore

        return self.__class__(labels=labels, name=self._name)

    def _extract_loc(self: I,
            key: GetItemKeyType
            ) -> tp.Union['Index', tp.Hashable]:
        return self._extract_iloc(self.loc_to_iloc(key))

    def __getitem__(self: I,
            key: GetItemKeyType
            ) -> tp.Union['Index', tp.Hashable]:
        '''Extract a new index given an iloc key.
        '''
        return self._extract_iloc(key)

    #---------------------------------------------------------------------------
    # operators

    def _ufunc_unary_operator(self,
            operator: UFunc
            ) -> np.ndarray:
        '''Always return an NP array.
        '''
        if self._recache:
            self._update_array_cache()

        array = operator(self._labels)
        array.flags.writeable = False
        return array

    def _ufunc_binary_operator(self, *,
            operator: UFunc,
            other: tp.Any
            ) -> np.ndarray:
        '''
        Binary operators applied to an index always return an NP array. This deviates from Pandas, where some operations (multipling an int index by an int) result in a new Index, while other operations result in a np.array (using == on two Index).
        '''
        if self._recache:
            self._update_array_cache()

        values = self._labels
        other_is_array = False

        if issubclass(other.__class__, Index):
            other = other.values # operate on labels to labels
            other_is_array = True
        elif isinstance(other, np.ndarray):
            other_is_array = True

        if operator.__name__ == 'matmul':
            return matmul(values, other)
        elif operator.__name__ == 'rmatmul':
            return matmul(other, values)

        return apply_binary_operator(
                values=values,
                other=other,
                other_is_array=other_is_array,
                operator=operator,
                )

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

    def __contains__(self, value: tp.Any) -> bool:
        '''Return True if value in the labels.
        '''
        if self._map is None: # loc_is_iloc
            if isinstance(value, INT_TYPES):
                return value >= 0 and value < len(self) #type: ignore
            return False
        return self._map.__contains__(value) #type: ignore


    #---------------------------------------------------------------------------
    # utility functions

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

        if compare_class and self.__class__ != other.__class__:
            return False
        elif not isinstance(other, Index):
            return False

        # defer updating cache
        if self._recache:
            self._update_array_cache()

        # same type from here
        if len(self) != len(other):
            return False
        if compare_name and self.name != other.name:
            return False
        if compare_dtype and self.dtype != other.dtype:
            return False

        eq = self.values == other.values

        # NOTE: will only be False, or an array
        if eq is False:
            return eq #type: ignore

        if skipna:
            isna_both = (isna_array(self.values, include_none=False)
                    & isna_array(other.values, include_none=False))
            eq[isna_both] = True

        if not eq.all(): # avoid returning a NumPy Bool
            return False
        return True

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

    @doc_inject(selector='fillna')
    def fillna(self, value: tp.Any) -> 'Index':
        '''Return an :obj:`Index` with replacing null (NaN or None) with the supplied value.

        Args:
            {value}
        '''
        values = self.values # force usage of property for cache update
        sel = isna_array(values)
        if not np.any(sel):
            return self if self.STATIC else self.copy()

        value_dtype = dtype_from_element(value)
        assignable_dtype = resolve_dtype(value_dtype, values.dtype)

        if values.dtype == assignable_dtype:
            assigned = values.copy()
        else:
            assigned = values.astype(assignable_dtype)

        assigned[sel] = value
        assigned.flags.writeable = False

        return self.__class__(assigned, name=self._name)


    #---------------------------------------------------------------------------
    # export

    def to_series(self) -> 'Series':
        '''Return a Series with values from this Index's labels.
        '''
        # NOTE: while we might re-use the index on the index returned from this Series, such an approach will not work with IndexHierarchy.to_frame, as we do not know if the index should be on the index or columns; thus, returning an unindexed Series is appropriate
        from static_frame import Series
        return Series(self.values, name=self._name)


    def level_add(self, level: tp.Hashable) -> 'IndexHierarchy':
        '''Return an IndexHierarchy with an added root level.
        '''
        from static_frame import IndexHierarchy
        from static_frame import IndexHierarchyGO

        cls = IndexHierarchy if self.STATIC else IndexHierarchyGO
        return cls.from_tree({level: self.values}, name=self._name)


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

    _map: tp.Optional[AutoMap]
    _labels_mutable: tp.List[tp.Hashable]
    _labels_mutable_dtype: np.dtype
    _positions_mutable_count: int
    _positions: np.ndarray
    _labels: np.ndarray


    def _extract_labels(self,
            mapping: tp.Optional[tp.Dict[tp.Hashable, int]],
            labels: np.ndarray,
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
            ) -> np.ndarray:
        '''Called in Index.__init__(). This creates and populates mutable storage. This creates and populates mutable storage as a side effect of array derivation.
        '''
        positions = Index._extract_positions(size, positions)
        self._positions_mutable_count = size
        return positions

    def _update_array_cache(self) -> None:

        if self._labels_mutable_dtype is not None and len(self._labels):
            # only update if _labels_mutable_dtype has been set and _labels exist
            self._labels_mutable_dtype = resolve_dtype(
                    self._labels.dtype,
                    self._labels_mutable_dtype)

        # NOTE: necessary to support creation from iterable of tuples
        self._labels, _ = iterable_to_array_1d(
                self._labels_mutable,
                dtype=self._labels_mutable_dtype)
        self._positions = PositionsAllocator.get(self._positions_mutable_count)
        self._recache = False

    #---------------------------------------------------------------------------
    # grow only mutation

    def append(self, value: tp.Hashable) -> None:
        '''append a value
        '''
        if self.__contains__(value): #type: ignore
            raise KeyError(f'duplicate key append attempted: {value}')

        # we might need to initialize map if not an increment that keeps loc_is_iloc relationship
        initialize_map = False
        if self._map is None: # loc_is_iloc
            if not (isinstance(value, INT_TYPES)
                    and value == self._positions_mutable_count):
                initialize_map = True
        else:
            self._map.add(value)

        if self._labels_mutable_dtype is not None:
            self._labels_mutable_dtype = resolve_dtype(
                    dtype_from_element(value),
                    self._labels_mutable_dtype)
        else:
            self._labels_mutable_dtype = dtype_from_element(value)

        self._labels_mutable.append(value)

        if initialize_map:
            self._map = AutoMap(self._labels_mutable)

        self._positions_mutable_count += 1
        self._recache = True

    def extend(self, values: KeyIterableTypes) -> None:
        '''Append multiple values
        Args:
            values: can be a generator.
        '''
        for value in values:
            self.append(value)


class IndexGO(_IndexGOMixin, Index):
    '''A mapping of labels to positions, immutable with grow-only size. Used as columns in :obj:`FrameGO`.
    '''

    _IMMUTABLE_CONSTRUCTOR = Index
    __slots__ = _INDEX_GO_SLOTS


# update class attr on Index after class initialziation
Index._MUTABLE_CONSTRUCTOR = IndexGO



#-------------------------------------------------------------------------------

def _index_initializer_needs_init(
        value: tp.Optional[IndexInitializer]
        ) -> bool:
    '''Determine if value is a non-empty index initializer. This could almost just be a truthy test, but ndarrays need to be handled in isolation. Generators should return True.
    '''
    if value is None:
        return False
    if isinstance(value, IndexBase):
        return False
    if isinstance(value, np.ndarray):
        return bool(len(value))
    return bool(value)

