from __future__ import annotations

from collections import Counter
from copy import deepcopy
from itertools import chain
from itertools import zip_longest

import numpy as np
import typing_extensions as tp
from arraykit import array_deepcopy
from arraykit import array_to_tuple_iter
from arraykit import immutable_filter
from arraykit import mloc
from arraykit import name_filter
from arraykit import resolve_dtype
from arraymap import AutoMap  # pylint: disable=E0611
from arraymap import FrozenAutoMap  # pylint: disable=E0611
from arraymap import NonUniqueError  # pylint: disable=E0611

from static_frame.core.container import ContainerOperand
from static_frame.core.container_util import apply_binary_operator
from static_frame.core.container_util import index_from_optional_constructor
from static_frame.core.container_util import iter_component_signature_bytes
from static_frame.core.container_util import key_from_container_key
from static_frame.core.container_util import matmul
from static_frame.core.container_util import sort_index_for_order
from static_frame.core.display import Display
from static_frame.core.display import DisplayActive
from static_frame.core.display import DisplayHeader
from static_frame.core.display_config import DisplayConfig
from static_frame.core.doc_str import doc_inject
from static_frame.core.doc_str import doc_update
from static_frame.core.exception import ErrorInitIndex
from static_frame.core.exception import ErrorInitIndexNonUnique
from static_frame.core.index_base import IndexBase
from static_frame.core.loc_map import LocMap
from static_frame.core.node_dt import InterfaceDatetime
from static_frame.core.node_iter import IterNodeApplyType
from static_frame.core.node_iter import IterNodeDepthLevel
from static_frame.core.node_re import InterfaceRe
from static_frame.core.node_selector import InterfaceSelectDuo
from static_frame.core.node_selector import InterGetItemLocReduces
from static_frame.core.node_selector import TVContainer_co
from static_frame.core.node_str import InterfaceString
from static_frame.core.node_values import InterfaceValues
from static_frame.core.style_config import StyleConfig
from static_frame.core.util import DEFAULT_SORT_KIND
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import DTYPE_DATETIME_KIND
from static_frame.core.util import DTYPE_INT_DEFAULT
from static_frame.core.util import DTYPE_NA_KINDS
from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import EMPTY_ARRAY
from static_frame.core.util import INT_TYPES
from static_frame.core.util import KEY_ITERABLE_TYPES
from static_frame.core.util import NAME_DEFAULT
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import IterNodeType
from static_frame.core.util import PositionsAllocator
from static_frame.core.util import TDepthLevel
from static_frame.core.util import TDtypeSpecifier
from static_frame.core.util import TILocSelector
from static_frame.core.util import TILocSelectorMany
from static_frame.core.util import TILocSelectorOne
from static_frame.core.util import TIndexCtor
from static_frame.core.util import TIndexCtorSpecifier
from static_frame.core.util import TIndexInitializer
from static_frame.core.util import TKeyIterable
from static_frame.core.util import TKeyTransform
from static_frame.core.util import TLabel
from static_frame.core.util import TLocSelector
from static_frame.core.util import TLocSelectorMany
from static_frame.core.util import TName
from static_frame.core.util import TUFunc
from static_frame.core.util import argsort_array
from static_frame.core.util import array_sample
from static_frame.core.util import array_shift
from static_frame.core.util import array_ufunc_axis_skipna
from static_frame.core.util import arrays_equal
from static_frame.core.util import concat_resolved
from static_frame.core.util import dtype_from_element
from static_frame.core.util import isfalsy_array
from static_frame.core.util import isin
from static_frame.core.util import isna_array
from static_frame.core.util import iterable_to_array_1d
from static_frame.core.util import key_to_str
from static_frame.core.util import pos_loc_slice_to_iloc_slice
from static_frame.core.util import to_datetime64
from static_frame.core.util import ufunc_unique1d_indexer
from static_frame.core.util import validate_dtype_specifier

if tp.TYPE_CHECKING:
    import pandas  # pragma: no cover

    from static_frame import IndexHierarchy  # pylint: disable=C0412 #pragma: no cover
    from static_frame import Series  # pragma: no cover
    from static_frame.core.index_auto import TRelabelInput  # pragma: no cover

    TNDArrayAny = np.ndarray[tp.Any, tp.Any] #pragma: no cover
    TDtypeAny = np.dtype[tp.Any] #pragma: no cover

I = tp.TypeVar('I', bound='Index[tp.Any]')


class ILocMeta(type):

    def __getitem__(cls,
            key: TLocSelector
            ) -> 'ILoc':
        return cls(key) #type: ignore


class ILoc(metaclass=ILocMeta):
    '''A wrapper for embedding ``iloc`` specifications within a single axis argument of a ``loc`` selection.
    '''

    STATIC = True
    __slots__ = (
            'key',
            )

    def __init__(self, key: TLocSelector):
        self.key = key

    def __repr__(self) -> str:
        if isinstance(self.key, tuple):
            return f'<ILoc[{",".join(map(key_to_str, self.key))}]>'
        return f'<ILoc[{key_to_str(self.key)}]>'


def immutable_index_filter(index: IndexBase) -> IndexBase:
    '''Return an immutable index. All index objects handle converting from mutable to immutable via the __init__ constructor; but need to use appropriate class between Index and IndexHierarchy.'''

    if index.STATIC:
        return index
    return index._IMMUTABLE_CONSTRUCTOR(index)


def mutable_immutable_index_filter(
        target_static: bool,
        index: IndexBase,
        ) -> IndexBase:
    if target_static:
        return immutable_index_filter(index)
    # target mutable
    if index.STATIC:
        return index._MUTABLE_CONSTRUCTOR(index)
    return index.__class__(index) # create new instance


#-------------------------------------------------------------------------------

class _ArgsortCache(tp.NamedTuple):
    arr: TNDArrayAny
    key: TNDArrayAny

    def __deepcopy__(self, memo: tp.Dict[int, tp.Any]) -> '_ArgsortCache':
        obj = self.__class__(
                array_deepcopy(self.arr, memo=memo),
                array_deepcopy(self.key, memo=memo),
                )

        memo[id(self)] = obj
        return obj

TVDtype = tp.TypeVar('TVDtype', bound=np.generic, default=tp.Any) # pylint: disable=E1123

class Index(IndexBase, tp.Generic[TVDtype]):
    '''A mapping of labels to positions, immutable and of fixed size. Used by default in :obj:`Series` and as index and columns in :obj:`Frame`. Base class of all 1D indices.'''

    __slots__ = (
        '_map',
        '_labels',
        '_positions',
        '_recache',
        '_name',
        '_argsort_cache',
        )

    # _IMMUTABLE_CONSTRUCTOR is None from IndexBase
    # _MUTABLE_CONSTRUCTOR will be set after IndexGO defined

    _DTYPE: tp.Optional[TDtypeAny] = None # for specialized indices requiring a typed labels

    # for compatability with IndexHierarchy, where this is implemented as a property method
    depth: int = 1
    _NDIM: int = 1

    _map: tp.Optional[FrozenAutoMap]
    _labels: TNDArrayAny
    _positions: TNDArrayAny
    _recache: bool
    _name: TName
    _argsort_cache: tp.Optional[_ArgsortCache]

    #---------------------------------------------------------------------------
    # methods used in __init__ that are customized in derived classes; there, we need to mutate instance state, this these are instance methods
    @staticmethod
    def _extract_labels(
            mapping: tp.Optional[tp.Dict[TLabel, int]],
            labels: tp.Iterable[TLabel],
            dtype: TDtypeSpecifier = None
            ) -> TNDArrayAny:
        '''Derive labels, a cache of the mapping keys in a sequence type (either an ndarray or a list).

        If the labels passed at instantiation are an ndarray, they are used after immutable filtering. Otherwise, the mapping keys are used to create an ndarray.

        This method is overridden in the derived class.

        Args:
            mapping: Can be None if loc_is_iloc.
            labels: might be an expired Generator, but if it is an immutable ndarray, we can use it without a copy.
        '''
        # pre-fetching labels for faster get_item construction
        if labels.__class__ is np.ndarray:
            if dtype is not None and dtype != labels.dtype: #type: ignore
                raise ErrorInitIndex('invalid label dtype for this Index')
            # NOTE: all labels arrays should be made immutable before this call
            return labels #type: ignore

        # labels may be an expired generator, must use the mapping
        labels_src = labels if hasattr(labels, '__len__') else mapping

        if len(labels_src) == 0: #type: ignore
            if dtype is None:
                labels = EMPTY_ARRAY
            else:
                labels = np.empty(0, dtype=dtype)
                labels.flags.writeable = False
        else: # resolving the dtype is expensive, pass if possible
            labels, _ = iterable_to_array_1d(labels_src, dtype=dtype) #type: ignore

        return labels

    @staticmethod
    def _extract_positions(
            size: int,
            positions: tp.Optional[TNDArrayAny]
            ) -> TNDArrayAny:
        # positions is either None or an ndarray
        if positions.__class__ is np.ndarray:
            return immutable_filter(positions) # type: ignore
        return PositionsAllocator.get(size)

    #---------------------------------------------------------------------------
    # constructors

    @classmethod
    def from_labels(cls: tp.Type[I],
            labels: tp.Iterable[tp.Sequence[TLabel]],
            *,
            name: TName = None
            ) -> I:
        '''
        Construct an ``Index`` from an iterable of labels, where each label is a hashable. Provided for a compatible interface to ``IndexHierarchy``.
        '''
        return cls(labels, name=name)


    @staticmethod
    def _error_init_index_non_unique(
            labels: tp.Iterable[tp.Any],
            ) -> ErrorInitIndexNonUnique:
        '''Return an exception configured with an informative message.
        '''
        msg = ''
        labels_counter = Counter(labels)
        if len(labels_counter) == 0: # generator consumed
            msg = 'Labels have non-unique values. Examples from iterators not are available.'
        else:
            labels_all = sum(labels_counter.values())
            labels_duplicated = [repr(p[0]) for p in labels_counter.most_common(10) if p[1] > 1]
            msg = f'Labels have {labels_all - len(labels_counter)} non-unique values, including {", ".join(labels_duplicated)}.'
        return ErrorInitIndexNonUnique(msg)

    #---------------------------------------------------------------------------
    def __init__(self,
            labels: TIndexInitializer,
            *,
            loc_is_iloc: bool = False,
            name: TName = NAME_DEFAULT,
            dtype: TDtypeSpecifier = None,
            ) -> None:
        '''Initializer.

        {args}
        '''
        self._recache: bool = False
        self._map: tp.Optional[FrozenAutoMap] = None
        self._argsort_cache: tp.Optional[_ArgsortCache] = None

        positions: TNDArrayAny | None = None
        is_typed = self._DTYPE is not None # only True for datetime64 indices

        # resolve the targetted labels dtype, by lookin at the class attr _DTYPE and/or the passed dtype argument
        if dtype is None:
            dtype_extract = self._DTYPE # set in some specialized Index sub-classes
        else: # passed dtype is not None
            if is_typed and dtype != self._DTYPE:
                # NOTE: should never get to this branch, as derived Index classes that set _DTYPE remove dtype from __init__
                raise ErrorInitIndex('invalid dtype argument for this Index', dtype, self._DTYPE) #pragma: no cover
            # self._DTYPE is None, passed dtype is not None, use dtype
            dtype_extract = dtype # type: ignore

        #-----------------------------------------------------------------------
        if labels.__class__ is np.ndarray:
            labels = immutable_filter(labels) # type: ignore
        elif isinstance(labels, IndexBase):
            # handle all Index subclasses
            if labels._recache:
                labels._update_array_cache()
            if name is NAME_DEFAULT:
                name = labels.name # immutable, so no copy necessary

            if labels.depth == 1: # not an IndexHierarchy
                if (labels.STATIC and self.STATIC and dtype is None):
                    if not is_typed or (is_typed and self._DTYPE == labels.dtype): # type: ignore
                        # can take the map if static and if types in the dict are the same as those in the labels (or to become the labels after conversion)
                        self._map = labels._map #type: ignore
                # get a reference to the immutable arrays, even if this is an IndexGO index, we can take the cached arrays, assuming they are up to date; for datetime64 indices, we might need to translate to a different type
                positions = labels._positions #type: ignore
                loc_is_iloc = labels._map is None #type: ignore
                labels = labels._labels # type: ignore
            else: # IndexHierarchy
                # will be a generator of tuples; already updated caches
                labels = labels.__iter__()

        elif isinstance(labels, ContainerOperand):
            # it is a Series or similar
            array = labels.values
            if array.ndim == 1:
                labels = array
            else:
                labels = array_to_tuple_iter(array)
        # else: assume an iterable suitable for labels usage, we will identify strings later

        #-----------------------------------------------------------------------
        if is_typed:
            # do not need to check arrays, as will and checked to match dtype_extract in _extract_labels
            if not labels.__class__ is np.ndarray:
                # if is_typed, _DTYPE is defined, we have a date
                labels = (to_datetime64(v, dtype_extract) for v in labels) #type: ignore
            # coerce to target type
            elif labels.dtype != dtype_extract: #type: ignore
                labels = labels.astype(dtype_extract) #type: ignore
                labels.flags.writeable = False

        self._name = None if name is NAME_DEFAULT else name_filter(name) # pyright: ignore

        size: int
        if self._map is None: # if _map not shared from another Index
            if not loc_is_iloc:
                if isinstance(labels, str):
                    # NOTE: this is necessary as otherwise a malformed Index will be created, whereby the _map will treat the string as an iterable of chars, while the labels will not and have a single string value. This is consisten as other elements (ints, Booleans) are rejected on instantiation of the AutoMap
                    raise ErrorInitIndex('Cannot create an Index from a single string; provide an iterable of strings.')
                try:
                    self._map = FrozenAutoMap(labels) if self.STATIC else AutoMap(labels)
                except NonUniqueError: # Automap will raise ValueError of non-unique values are encountered
                    raise self._error_init_index_non_unique(labels) from None
                # must take length after map as might be iterator
                size = len(self._map) # pyright: ignore
            else:
                # if loc_is_iloc, labels must be positions and we assume that internal clients that provided loc_is_iloc will not give a generator
                size = len(labels) #type: ignore
                if positions is None:
                    positions = labels # type: ignore
        else: # map shared from another Index
            size = len(self._map)

        # this might be NP array, or a list, depending on if static or grow only; if an array, dtype will be compared with passed dtype_extract
        self._labels: TNDArrayAny = self._extract_labels(self._map, labels, dtype_extract)
        self._positions = self._extract_positions(size, positions)

        if self._DTYPE and self._labels.dtype != self._DTYPE:
            raise ErrorInitIndex('Invalid label dtype for this Index.', #pragma: no cover
                    self._labels.dtype, self._DTYPE)

        # NOTE: to implement GH # 374; do this after final self._labels creation as user may pass a dtype argument
        if not is_typed and self._labels.dtype.kind == DTYPE_DATETIME_KIND:
            raise ErrorInitIndex(f'Cannot create an `Index` with a `datetime64` array (with `dtype` {self._labels.dtype} and including {self._labels[:10]}); use a subclass (e.g. `IndexDate`) directly or as a constructor argument.')

    #---------------------------------------------------------------------------

    def __setstate__(self, state: tp.Tuple[None, tp.Dict[str, tp.Any]]) -> None:
        '''
        Ensure that reanimated NP arrays are set not writeable.
        '''
        for key, value in state[1].items():
            setattr(self, key, value)
        self._labels.flags.writeable = False

    def __deepcopy__(self: I, memo: tp.Dict[int, tp.Any]) -> I:
        assert not self._recache # __deepcopy__ is implemented on derived GO class

        obj = self.__class__.__new__(self.__class__)
        obj._map = deepcopy(self._map, memo)
        obj._labels = array_deepcopy(self._labels, memo)
        obj._positions = PositionsAllocator.get(len(self._labels))
        obj._recache = False
        obj._name = self._name # should be hashable/immutable
        obj._argsort_cache = deepcopy(self._argsort_cache, memo)

        memo[id(self)] = obj
        return obj

    def _memory_label_component_pairs(self,
            ) -> tp.Iterable[tp.Tuple[str, tp.Any]]:
        return (('Name', self._name),
                ('Map', self._map),
                ('Labels', self._labels),
                ('Positions', self._positions),
                )

    def __copy__(self: I) -> I:
        '''
        Return shallow copy of this Index.
        '''
        if self._recache:
            self._update_array_cache()

        return self.__class__(self, name=self._name)

    def copy(self: I) -> I:
        '''
        Return shallow copy of this Index.
        '''
        return self.__copy__()

    #---------------------------------------------------------------------------
    # name interface

    def rename(self: I, name: TName) -> I:
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
    def loc(self) -> InterGetItemLocReduces[TVContainer_co, TVDtype]:
        return InterGetItemLocReduces(self._extract_loc)

    @property
    def iloc(self) -> InterGetItemLocReduces[TVContainer_co, TVDtype]:
        return InterGetItemLocReduces(self._extract_iloc) #type: ignore

    # # on Index, getitem is an iloc selector; on Series, getitem is a loc selector; for this extraction interface, we do not implement a getitem level function (using iloc would be consistent), as it is better to be explicit between iloc loc

    def _iter_label(self,
            depth_level: tp.Optional[TDepthLevel] = None
            ) -> tp.Iterator[TLabel]:
        yield from self._labels

    def _iter_label_items(self,
            depth_level: tp.Optional[TDepthLevel] = None
            ) -> tp.Iterator[tp.Tuple[int, TLabel]]:
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

    @property
    @doc_inject()
    def mloc(self) -> int:
        '''{doc_int}
        '''
        if self._recache:
            self._update_array_cache()
        return mloc(self._labels)

    @property
    def dtype(self) -> np.dtype[TVDtype]:
        '''
        Return the dtype of the underlying NumPy array.

        Returns:
            :obj:`numpy.dtype`
        '''
        if self._recache:
            self._update_array_cache()
        return self._labels.dtype #type: ignore

    @property
    def shape(self) -> tp.Tuple[int, ...]:
        '''
        Return a tuple describing the shape of the underlying NumPy array.

        Returns:
            :obj:`tp.Tuple[int]`
        '''
        if self._recache:
            self._update_array_cache()
        return self._labels.shape

    @property
    def ndim(self) -> int:
        '''
        Return the number of dimensions.

        Returns:
            :obj:`int`
        '''
        if self._recache:
            self._update_array_cache()
        return self._labels.ndim

    @property
    def size(self) -> int:
        '''
        Return the size of the underlying NumPy array.

        Returns:
            :obj:`int`
        '''
        if self._recache:
            self._update_array_cache()
        return self._labels.size

    @property
    def nbytes(self) -> int:
        '''
        Return the total bytes of the underlying NumPy array.

        Returns:
            :obj:`int`
        '''
        if self._recache:
            self._update_array_cache()
        return self._labels.nbytes

    #---------------------------------------------------------------------------
    def _drop_iloc(self, key: TILocSelector) -> tp.Self:
        '''Create a new index after removing the values specified by the iloc key.
        '''
        if self._recache:
            self._update_array_cache()

        if key is None:
            if self.STATIC: # immutable, no selection, can return self
                return self
            labels = self._labels # already immutable
        elif key.__class__ is np.ndarray and key.dtype == bool: #type: ignore
            # can use labels, as we already recached
            # use Boolean area to select indices from positions, as np.delete does not work with arrays
            labels = np.delete(self._labels, self._positions[key], axis=0)
            labels.flags.writeable = False
        else:
            labels = np.delete(self._labels, key, axis=0)
            labels.flags.writeable = False

        # from labels will work with both Index and IndexHierarchy
        return self.__class__.from_labels(labels, name=self._name)

    def _drop_loc(self, key: TLocSelector) -> tp.Self:
        '''Create a new index after removing the values specified by the loc key.
        '''
        return self._drop_iloc(self._loc_to_iloc(key))


    @property
    def drop(self) -> InterfaceSelectDuo[TVContainer_co]:
        return InterfaceSelectDuo( #type: ignore
            func_iloc=self._drop_iloc,
            func_loc=self._drop_loc,
            )


    @doc_inject(select='astype')
    def astype(self, dtype: TDtypeSpecifier) -> Index[tp.Any]:
        '''
        Return an Index with type determined by `dtype` argument. If a `datetime64` dtype is provided, the appropriate ``Index`` subclass will be returned. Note that for Index, this is a simple function, whereas for ``IndexHierarchy``, this is an interface exposing both a callable and a getitem interface.

        Args:
            {dtype}
        '''
        from static_frame.core.index_datetime import dtype_to_index_cls

        dtype = validate_dtype_specifier(dtype)

        array = self.values.astype(dtype)
        array.flags.writeable = False
        cls = dtype_to_index_cls(self.STATIC, array.dtype)
        return cls(
                array,
                name=self._name
                )


    #---------------------------------------------------------------------------

    @property
    def via_values(self) -> InterfaceValues[Index[tp.Any]]:
        '''
        Interface for applying functions to values (as arrays) in this container.
        '''
        if self._recache:
            self._update_array_cache()

        return InterfaceValues(self)

    @property
    def via_str(self) -> InterfaceString[TNDArrayAny]:
        '''
        Interface for applying string methods to elements in this container.
        '''
        if self._recache:
            self._update_array_cache()

        def blocks_to_container(blocks: tp.Iterator[TNDArrayAny]) -> TNDArrayAny:
            return next(blocks)

        return InterfaceString(
                blocks=(self._labels,),
                blocks_to_container=blocks_to_container,
                ndim=self._NDIM,
                labels=range(1)
                )

    @property
    def via_dt(self) -> InterfaceDatetime[TNDArrayAny]:
        '''
        Interface for applying datetime properties and methods to elements in this container.
        '''
        if self._recache:
            self._update_array_cache()

        def blocks_to_container(blocks: tp.Iterator[TNDArrayAny]) -> TNDArrayAny:
            return next(blocks)

        return InterfaceDatetime(
                blocks=(self.values,),
                blocks_to_container=blocks_to_container,
                )

    def via_re(self,
            pattern: str,
            flags: int = 0,
            ) -> InterfaceRe[TNDArrayAny]:
        '''
        Interface for applying regular expressions to elements in this container.
        '''
        if self._recache:
            self._update_array_cache()

        def blocks_to_container(blocks: tp.Iterator[TNDArrayAny]) -> TNDArrayAny:
            return next(blocks)

        return InterfaceRe(
                blocks=(self._labels,),
                blocks_to_container=blocks_to_container,
                pattern=pattern,
                flags=flags,
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
            *,
            style_config: tp.Optional[StyleConfig] = None,
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
                style_config=style_config,
                )

    #---------------------------------------------------------------------------
    # core internal representation

    @property
    @doc_inject(selector='values_1d', class_name='Index')
    def values(self) -> TNDArrayAny:
        '''
        {}
        '''
        if self._recache:
            self._update_array_cache()
        return self._labels

    @property
    def positions(self) -> TNDArrayAny:
        '''Return the immutable positions array.
        '''
        # This is needed by some clients, such as Series and Frame, to support Boolean usage in drop.
        if self._recache:
            self._update_array_cache()
        return self._positions

    def _get_argsort_cache(self: I) -> _ArgsortCache:
        '''
        Return a cached NT containing self.values sorted, along with the argsort key

        This utilizes a lazy instance cache attribute, since sorting is expensive,
        and this operation is typically called either never, or often.
        '''
        if self._recache:
            self._update_array_cache()

        if self._argsort_cache is None:
            self._argsort_cache = _ArgsortCache(*ufunc_unique1d_indexer(self.values))

        return self._argsort_cache

    def _index_iloc_map(self: I, other: I) -> TNDArrayAny:
        '''
        Return an array of index locations to map from this array to another

        Equivalent to: self.iter_label().apply(other._loc_to_iloc)
        '''
        if self.__len__() == 0 or other.__len__() == 0:
            return EMPTY_ARRAY

        if self.dtype == DTYPE_OBJECT or other.dtype == DTYPE_OBJECT:
            return self.iter_label().apply(other._loc_to_iloc, dtype=DTYPE_INT_DEFAULT)  #type: ignore [no-any-return]

        # Equivalent to: ufunc_unique1d_indexer(self.values)
        ar1, ar1_indexer = self._get_argsort_cache()
        ar2 = other.values

        aux = concat_resolved((ar1, ar2))
        aux_sort_indices = argsort_array(aux)
        aux = aux[aux_sort_indices]

        mask = aux[1:] == aux[:-1]

        indexer: TNDArrayAny = aux_sort_indices[1:][mask] - ar1.size

        # We want to return these indices to match ar1 before it was sorted
        try:
            indexer = indexer[ar1_indexer]
        except IndexError as e:
            # Display the first missing element
            raise KeyError(self.difference(other)[0]) from e

        indexer.flags.writeable = False
        return indexer

    @staticmethod
    def _depth_level_validate(depth_level: TDepthLevel) -> None:
        '''
        Handle all variety of depth_level specifications for a 1D index: only 0, -1, and lists of the same are valid.
        '''
        if not isinstance(depth_level, INT_TYPES):
            depth_level = list(depth_level)
            if len(depth_level) != 1:
                raise RuntimeError('invalid depth_level', depth_level)
            depth_level = depth_level[0]

        if depth_level > 0 or depth_level < -1:
            raise RuntimeError('invalid depth_level', depth_level)

    def values_at_depth(self,
            depth_level: TDepthLevel = 0
            ) -> TNDArrayAny:
        '''
        Return an NP array for the `depth_level` specified.
        '''
        self._depth_level_validate(depth_level)
        return self.values

    @doc_inject()
    def label_widths_at_depth(self,
            depth_level: TDepthLevel = 0
            ) -> tp.Iterator[tp.Tuple[TLabel, int]]:
        '''{}'''
        self._depth_level_validate(depth_level)
        yield from zip_longest(self.values, (), fillvalue=1)

    @property
    def index_types(self) -> Series[tp.Any, np.object_]:
        '''
        Return a Series of Index classes for each index depth.

        Returns:
            :obj:`Series`
        '''
        from static_frame.core.series import Series
        return Series((self.__class__,), index=(self._name,), dtype=DTYPE_OBJECT)


    #---------------------------------------------------------------------------

    def relabel(self, mapper: 'TRelabelInput') -> Index[tp.Any]:
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
                (mapper(x) for x in self._labels),
                name=self._name
                )

    #---------------------------------------------------------------------------
    # extraction and selection

    def _loc_to_iloc(self,
            key: TLocSelector,
            key_transform: TKeyTransform = None,
            partial_selection: bool = False,
            ) -> TILocSelector:
        '''
        Args:
            key_transform: A function that transforms keys to specialized type; used by IndexDate indices.
        Returns:
            Return GetItemKey type that is based on integers, compatible with TypeBlocks
        '''
        if key.__class__ is ILoc:
            return key.key # type: ignore

        key = key_from_container_key(self, key)

        if self._map is None: # loc_is_iloc
            if key.__class__ is np.ndarray:
                if key.dtype == DTYPE_BOOL: #type: ignore
                    return key # type: ignore
                if key.dtype != DTYPE_INT_DEFAULT: #type: ignore
                    # if key is an np.array, it must be an int or bool type
                    # could use tolist(), but we expect all keys to be integers
                    return key.astype(DTYPE_INT_DEFAULT) #type: ignore
            elif key.__class__ is slice:
                # might raise LocInvalid
                key = pos_loc_slice_to_iloc_slice(key, self.__len__()) # type: ignore
            return key # type: ignore

        if key_transform:
            key = key_transform(key)

        # PERF: isolate for usage of _positions
        if self._recache:
            self._update_array_cache()

        return LocMap.loc_to_iloc(
                label_to_pos=self._map,
                labels=self._labels,
                positions=self._positions, # always an np.ndarray
                key=key,
                partial_selection=partial_selection,
                )

    @tp.overload
    def loc_to_iloc(self, key: TLabel) -> TILocSelectorOne: ...

    @tp.overload
    def loc_to_iloc(self, key: TLocSelectorMany) -> TILocSelectorMany: ...

    def loc_to_iloc(self,
            key: TLocSelector,
            ) -> TILocSelector:
        '''Given a label (loc) style key (either a label, a list of labels, a slice, or a Boolean selection), return the index position (iloc) style key. Keys that are not found will raise a KeyError or a sf.LocInvalid error.

        Args:
            key: a label key.
        '''
        if self._map is None: # loc is iloc
            # NOTE: the specialization here is to use the key on the positions array and return iloc values, rather than just propagating the selection array. This also handles and re-raises better exceptions.

            if not key.__class__ is slice:
                if self._recache:
                    self._update_array_cache()

                key = key_from_container_key(self, key)
                is_array = key.__class__ is np.ndarray
                try:
                    # NOTE: this insures that the returned type will be DTYPE_INT_DEFAULT
                    result = self._positions[key] # type: ignore
                except IndexError as e:
                    # NP gives us: IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
                    if is_array and key.dtype == DTYPE_BOOL: #type: ignore
                        raise # loc selection on Boolean array selection returns IndexError
                    raise KeyError(key) from e

                return result # return position as array

            # might raise LocInvalid
            return pos_loc_slice_to_iloc_slice(key, self.__len__()) # type: ignore

        return self._loc_to_iloc(key)

    def _extract_iloc(self,
            key: TILocSelector,
            ) -> tp.Any:
        '''Extract a new index given an iloc key.
        '''
        if self._recache:
            self._update_array_cache()

        if key is None:
            labels = self._labels
            loc_is_iloc = self._map is None
        elif key.__class__ is slice:
            if key == NULL_SLICE:
                labels = self._labels
                loc_is_iloc = self._map is None
            else:
                # if labels is an np array, this will be a view; if a list, a copy
                labels = self._labels[key]
                labels.flags.writeable = False
                loc_is_iloc = False
        elif isinstance(key, KEY_ITERABLE_TYPES):
            # can select directly from _labels[key] if if key is a list, array, or Boolean array
            labels = self._labels[key]
            labels.flags.writeable = False
            loc_is_iloc = False
        else: # select a single label value
            return self._labels[key]

        return self.__class__(labels=labels,
                loc_is_iloc=loc_is_iloc,
                name=self._name,
                )

    def _extract_iloc_by_int(self,
            key: int | np.integer[tp.Any],
            ) -> tp.Any:
        '''Extract an element given an iloc integer key.
        '''
        if self._recache:
            self._update_array_cache()
        return self._labels[key]

    def _extract_loc(self: I,
            key: TLocSelector
            ) -> tp.Any:
        return self._extract_iloc(self._loc_to_iloc(key))

    @tp.overload
    def __getitem__(self, key: TILocSelectorOne) -> TVDtype: ...

    @tp.overload
    def __getitem__(self, key: TILocSelectorMany) -> tp.Self: ...

    def __getitem__(self,
            key: TILocSelector
            ) -> tp.Any:
        '''Extract a new index given an iloc key.
        '''
        return self._extract_iloc(key)

    #---------------------------------------------------------------------------
    # operators

    def _ufunc_unary_operator(self,
            operator: TUFunc
            ) -> TNDArrayAny:
        '''Always return an NP array.
        '''
        if self._recache:
            self._update_array_cache()

        array = operator(self._labels)
        array.flags.writeable = False
        return array

    def _ufunc_binary_operator(self, *,
            operator: TUFunc,
            other: tp.Any,
            fill_value: object = np.nan,
            ) -> TNDArrayAny:
        '''
        Binary operators applied to an index always return an NP array. This deviates from Pandas, where some operations (multiplying an int index by an int) result in a new Index, while other operations result in a np.array (using == on two Index).
        '''
        from static_frame.core.frame import Frame
        from static_frame.core.series import Series

        if self._recache:
            self._update_array_cache()

        if isinstance(other, (Series, Frame)):
            raise ValueError('cannot use labelled container as an operand.')

        values = self._labels
        other_is_array = False

        if issubclass(other.__class__, Index):
            other = other.values # operate on labels to labels
            other_is_array = True
        elif other.__class__ is np.ndarray:
            other_is_array = True

        if operator.__name__ == 'matmul':
            return matmul(values, other) # type: ignore
        elif operator.__name__ == 'rmatmul':
            return matmul(other, values) # type: ignore

        return apply_binary_operator(
                values=values,
                other=other,
                other_is_array=other_is_array,
                operator=operator,
                )

    def _ufunc_axis_skipna(self, *,
            axis: int,
            skipna: bool,
            ufunc: TUFunc,
            ufunc_skipna: TUFunc,
            composable: bool,
            dtypes: tp.Tuple[TDtypeAny, ...],
            size_one_unity: bool
            ) -> tp.Any:
        '''

        Args:
            dtype: Not used in 1D application, but collected here to provide a uniform signature.
        '''
        if self._recache:
            self._update_array_cache()

        # do not need to pass on composabel here
        return array_ufunc_axis_skipna(
                array=self._labels,
                skipna=skipna,
                axis=0,
                ufunc=ufunc,
                ufunc_skipna=ufunc_skipna
                )

    def _ufunc_shape_skipna(self, *,
            axis: int,
            skipna: bool,
            ufunc: TUFunc,
            ufunc_skipna: TUFunc,
            composable: bool,
            dtypes: tp.Tuple[TDtypeAny, ...],
            size_one_unity: bool
            ) -> tp.Any:
        '''
        As Index and IndexHierarchy return np.ndarray from such operations, _ufunc_shape_skipna and _ufunc_axis_skipna can be defined the same.

        Returns:
            immutable NumPy array.
        '''
        # NOTE: for 1D Index, can use axis for shape ufunc
        return self._ufunc_axis_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=ufunc,
                ufunc_skipna=ufunc_skipna,
                composable=composable, # shape on axis 1 is never composable
                dtypes=dtypes,
                size_one_unity=size_one_unity
                )

    #---------------------------------------------------------------------------
    # dictionary-like interface

    # NOTE: we intentionally exclude keys(), items(), and get() from Index classes, as they return inconsistent result when thought of as a dictionary


    def __iter__(self) -> tp.Iterator[TLabel]:
        '''Iterate over labels.
        '''
        if self._recache:
            self._update_array_cache()
        yield from self._labels.__iter__()

    def __reversed__(self) -> tp.Iterator[TLabel]:
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

    def unique(self,
            depth_level: TDepthLevel = 0,
            order_by_occurrence: bool = False,
            ) -> TNDArrayAny:
        '''
        Return a NumPy array of unique values.

        Args:
            depth_level: defaults to 0 for for a 1D Index.
            order_by_occurrence: for 1D indices, this argument is a no-op. Provided for compatibility with IndexHierarchy.

        Returns:
            :obj:`numpy.ndarray`
        '''
        self._depth_level_validate(depth_level)
        return self.values

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
        if self._map is None and other._map is None:
            return True # have same length must be same integer range and dtype
        if compare_dtype and self.dtype != other.dtype:
            return False
        return arrays_equal(self.values, other.values, skipna=skipna)


    @doc_inject(selector='sort')
    def sort(self,
            ascending: bool = True,
            kind: str = DEFAULT_SORT_KIND,
            key: tp.Optional[tp.Callable[
                    [Index[tp.Any]],
                    tp.Union[TNDArrayAny, Index[tp.Any]]
                    ]] = None,
            ) -> tp.Self:
        '''Return a new Index with the labels sorted.

        Args:
            {ascending}
            {kind}
            {key}
        '''
        order = sort_index_for_order(self, kind=kind, ascending=ascending, key=key) #type: ignore [arg-type]
        return self._extract_iloc(order) #type: ignore

    def isin(self, other: tp.Iterable[tp.Any]) -> TNDArrayAny:
        '''
        Return a Boolean array showing True where a label is found in other. If other is a multidimensional array, it is flattened.
        '''
        return isin(self.values, other, array_is_unique=True)

    def roll(self, shift: int) -> tp.Self:
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
    # na handling
    # falsy handling

    def _drop_missing(self,
            func: tp.Callable[[TNDArrayAny], TNDArrayAny],
            dtype_kind_targets: tp.Optional[tp.FrozenSet[str]],
            ) -> tp.Self:
        '''
        Args:
            func: TUFunc that returns True for missing values
        '''
        labels = self.values
        if dtype_kind_targets is not None and labels.dtype.kind not in dtype_kind_targets:
            return self if self.STATIC else self.copy()

        # get positions that we want to keep
        isna = func(labels)
        length = len(labels)
        count = isna.sum()

        if count == length: # all are NaN
            return self.__class__((), name=self.name)
        if count == 0: # None are nan
            return self if self.STATIC else self.copy()

        sel = np.logical_not(isna)
        values = labels[sel]
        values.flags.writeable = False

        return self.__class__(values,
                name=self._name,
                )

    def dropna(self) -> tp.Self:
        '''
        Return a new :obj:`Index` after removing values of NaN or None.
        '''
        return self._drop_missing(isna_array, DTYPE_NA_KINDS)

    def dropfalsy(self) -> tp.Self:
        '''
        Return a new :obj:`Index` after removing values of NaN or None.
        '''
        return self._drop_missing(isfalsy_array, None)

    #---------------------------------------------------------------------------

    def _fill_missing(self,
            func: tp.Callable[[TNDArrayAny], TNDArrayAny],
            value: tp.Any,
            ) -> Index[tp.Any]:
        values = self.values # force usage of property for cache update
        sel = func(values)
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

    @doc_inject(selector='fillna')
    def fillna(self, value: tp.Any) -> Index[tp.Any]:
        '''Return an :obj:`Index` with replacing null (NaN or None) with the supplied value.

        Args:
            {value}
        '''
        return self._fill_missing(isna_array, value)

    @doc_inject(selector='fillna')
    def fillfalsy(self, value: tp.Any) -> Index[tp.Any]:
        '''Return an :obj:`Index` with replacing falsy values with the supplied value.

        Args:
            {value}
        '''
        return self._fill_missing(isfalsy_array, value)

    #---------------------------------------------------------------------------
    def _sample_and_key(self,
            count: int = 1,
            *,
            seed: tp.Optional[int] = None,
            ) -> tp.Tuple[tp.Self, TNDArrayAny]:
        # NOTE: base class defines pubic method
        # force usage of property for cache update
        # sort positions to avoid uncomparable objects
        key = array_sample(self.positions, count=count, seed=seed, sort=True)

        values = self.values[key]
        values.flags.writeable = False
        return self.__class__(values, name=self._name), key


    @doc_inject(selector='searchsorted', label_type='iloc (integer)')
    def iloc_searchsorted(self,
            values: tp.Any,
            *,
            side_left: bool = True,
            ) -> TNDArrayAny:
        '''
        {doc}

        Args:
            {values}
            {side_left}
        '''
        if not isinstance(values, str) and hasattr(values, '__len__'):
            if not values.__class__ is np.ndarray:
                values, _ = iterable_to_array_1d(values)
        return np.searchsorted(self.values, # type: ignore
                values,
                'left' if side_left else 'right',
                )

    @doc_inject(selector='searchsorted', label_type='loc (label)')
    def loc_searchsorted(self,
            values: tp.Any,
            *,
            side_left: bool = True,
            fill_value: tp.Any = np.nan,
            ) -> tp.Union[TLabel, TNDArrayAny]:
        '''
        {doc}

        Args:
            {values}
            {side_left}
            {fill_value}
        '''
        sel = self.iloc_searchsorted(values, side_left=side_left)

        length = self.__len__()
        if sel.ndim == 0 and sel == length: # an element:
            return fill_value #type: ignore [no-any-return]

        mask = sel == length
        if not mask.any():
            return self.values[sel]

        post = np.empty(len(sel),
                dtype=resolve_dtype(self.dtype,
                dtype_from_element(fill_value))
                )
        sel[mask] = 0 # set out of range values to zero
        post[:] = self.values[sel]
        post[mask] = fill_value
        post.flags.writeable = False
        return post

    def level_add(self,
            level: TLabel,
            *,
            index_constructor: TIndexCtorSpecifier = None,
            ) -> 'IndexHierarchy':
        '''Return an IndexHierarchy with an added root level.

        Args:
            level: A hashable to used as the new root.
            *
            index_constructor
        '''
        from static_frame import Index
        from static_frame import IndexGO
        from static_frame import IndexHierarchy
        from static_frame import IndexHierarchyGO

        cls = IndexHierarchy if self.STATIC else IndexHierarchyGO
        cls_depth: tp.Type[Index[tp.Any]] = Index if self.STATIC else IndexGO

        idx_ctor: TIndexCtor
        if index_constructor is None:
            # cannot assume new depth is the same index subclass
            idx_ctor = cls_depth
        else:
            idx_ctor = index_constructor

        index_d1 = index_from_optional_constructor(
                (level,),
                default_constructor=Index,
                explicit_constructor=idx_ctor
                )
        index_d2 = index_from_optional_constructor(  # force copy is self is GO
                self,
                default_constructor=self.__class__,
                )

        indices: tp.List[Index] = [index_d1, index_d2]  # type: ignore

        indexers = np.array(
                [
                    np.zeros(self.__len__(), dtype=DTYPE_INT_DEFAULT),
                    self.positions
                ]
        )
        indexers.flags.writeable = False

        return cls(
                indices=indices,
                indexers=indexers,
                name=self._name,
                )

    #---------------------------------------------------------------------------
    # export

    def to_series(self) -> Series[Index[np.int64], TVDtype]:
        '''Return a Series with values from this Index's labels.
        '''
        # NOTE: while we might re-use the index on the index returned from this Series, such an approach will not work with IndexHierarchy.to_frame, as we do not know if the index should be on the index or columns; thus, returning an unindexed Series is appropriate
        from static_frame import Series
        return Series(self.values, name=self._name)

    def to_pandas(self) -> 'pandas.Index':
        '''Return a Pandas Index.
        '''
        import pandas

        # must copy to remove immutability, decouple reference
        if self._map is None:
            return pandas.RangeIndex(self.__len__(), name=self._name) # pyright: ignore
        return pandas.Index(self.values.copy(),
                name=self._name)


    def _to_signature_bytes(self,
            include_name: bool = True,
            include_class: bool = True,
            encoding: str = 'utf-8',
            ) -> bytes:
        if self.dtype == DTYPE_OBJECT:
            raise TypeError('Object dtypes do not have stable hashes')
        return b''.join(chain(
                iter_component_signature_bytes(self,
                        include_name=include_name,
                        include_class=include_class,
                        encoding=encoding),
                (self.values.tobytes(),),
                ))


doc_update(Index.__init__, selector='index_init')

#-------------------------------------------------------------------------------

class _IndexGOMixin:

    STATIC = False
    # NOTE: must define __slots__ in derived class or get TypeError: multiple bases have instance lay-out conflict
    __slots__ = ()

    _DTYPE: tp.Optional[TDtypeAny]
    _map: tp.Optional[AutoMap]
    _labels: TNDArrayAny
    _positions: TNDArrayAny
    _labels_mutable: tp.List[TLabel]
    _labels_mutable_dtype: tp.Optional[TDtypeAny]
    _positions_mutable_count: int
    _argsort_cache: tp.Optional[_ArgsortCache]

    #---------------------------------------------------------------------------
    def __deepcopy__(self: I, memo: tp.Dict[int, tp.Any]) -> I: #type: ignore
        if self._recache:
            self._update_array_cache()

        obj = self.__class__.__new__(self.__class__)
        obj._map = deepcopy(self._map, memo)
        obj._labels = array_deepcopy(self._labels, memo)
        obj._positions = PositionsAllocator.get(len(self._labels))
        obj._recache = False # pylint: disable=E0237
        obj._name = self._name # pylint: disable=E0237
        obj._labels_mutable = deepcopy(self._labels_mutable, memo) #type: ignore
        obj._labels_mutable_dtype = deepcopy(self._labels_mutable_dtype, memo) #type: ignore
        obj._positions_mutable_count = self._positions_mutable_count #type: ignore
        obj._argsort_cache = deepcopy(self._argsort_cache, memo)

        memo[id(self)] = obj
        return obj

    #---------------------------------------------------------------------------
    def _extract_labels(self,
            mapping: tp.Optional[tp.Dict[TLabel, int]],
            labels: TNDArrayAny,
            dtype: tp.Optional[TDtypeAny] = None
            ) -> TNDArrayAny:
        '''Called in Index.__init__(). This creates and populates mutable storage as a side effect of array derivation; this storage will be grown as needed.
        '''
        labels = Index._extract_labels(mapping, labels, dtype)
        self._labels_mutable = labels.tolist() # must get a fresh list
        if len(labels):
            self._labels_mutable_dtype = labels.dtype
        else: # avoid setting to float default when labels is empty
            self._labels_mutable_dtype = None
        return labels

    def _extract_positions(self,
            size: int,
            positions: tp.Optional[TNDArrayAny]
            ) -> TNDArrayAny:
        '''Called in Index.__init__(). This creates and populates mutable storage as a side effect of array derivation.
        '''
        pos = Index._extract_positions(size, positions)
        self._positions_mutable_count = size
        return pos

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
        self._recache = False # pylint: disable=E0237

        # clear cache
        self._argsort_cache = None

    #---------------------------------------------------------------------------
    # grow only mutation

    def append(self, value: TLabel) -> None:
        '''Append a value to this Index. Note: if the appended value not permitted by a specific Index subclass, this will raise and the caller will need to derive a new index type.
        '''
        if self.__contains__(value): #type: ignore
            raise KeyError(f'duplicate key append attempted: {value!r}')

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

        # NOTE: this is not possile at present as all Index subclasses set _DTYPE
        # if self._DTYPE is not None and self._labels_mutable_dtype != self._DTYPE:
        #     raise GrowOnlyInvalid()

        self._labels_mutable.append(value)

        if initialize_map:
            self._map = AutoMap(self._labels_mutable)

        self._positions_mutable_count += 1
        self._recache = True # pylint: disable=E0237

    def extend(self, values: TKeyIterable) -> None:
        '''Append multiple values
        Args:
            values: can be a generator.
        '''
        for value in values:
            self.append(value)


INDEX_GO_LEAF_SLOTS = (
        '_labels_mutable',
        '_labels_mutable_dtype',
        '_positions_mutable_count',
        )

class IndexGO(_IndexGOMixin, Index[TVDtype]):
    '''A mapping of labels to positions, immutable with grow-only size. Used as columns in :obj:`FrameGO`.
    '''

    _IMMUTABLE_CONSTRUCTOR = Index
    __slots__ = INDEX_GO_LEAF_SLOTS


# update class attr on Index after class initialziation
Index._MUTABLE_CONSTRUCTOR = IndexGO



#-------------------------------------------------------------------------------

def _index_initializer_needs_init(
        value: tp.Optional[TIndexInitializer]
        ) -> bool:
    '''Determine if value is a non-empty index initializer. This could almost just be a truthy test, but ndarrays need to be handled in isolation. Generators should return True.
    '''
    if value is None:
        return False
    if isinstance(value, IndexBase):
        return False
    if value.__class__ is np.ndarray:
        return bool(len(value)) #type: ignore
    return bool(value)

