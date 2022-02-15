import functools
import itertools
import typing as tp
from copy import deepcopy
from ast import literal_eval

import numpy as np
from arraykit import (
    name_filter,
)
from static_frame.core.container_util import (
    key_from_container_key,
    matmul,
    rehierarch_from_type_blocks,
    sort_index_for_order,
)
from static_frame.core.display import (
    Display,
    DisplayActive,
    DisplayHeader,
)
from static_frame.core.util import array_to_duplicated, arrays_equal
from static_frame.core.display_config import DisplayConfig
from static_frame.core.doc_str import doc_inject
from static_frame.core.exception import ErrorInitIndex, ErrorInitIndexNonUnique
from static_frame.core.hloc import HLoc
from static_frame.core.index import (
    ILoc,
    Index,
    IndexGO,
    immutable_index_filter,
)
from static_frame.core.index import mutable_immutable_index_filter
from static_frame.core.index_auto import RelabelInput
from static_frame.core.index_base import IndexBase
from static_frame.core.index_datetime import IndexDatetime
from static_frame.core.index_level import TreeNodeT
from static_frame.core.node_dt import InterfaceDatetime
from static_frame.core.node_iter import (
    IterNodeApplyType,
    IterNodeDepthLevel,
    IterNodeType,
)
from static_frame.core.node_re import InterfaceRe
from static_frame.core.node_selector import (
    InterfaceAsType,
    InterfaceGetItem,
    TContainer,
)
from static_frame.core.node_str import InterfaceString
from static_frame.core.node_transpose import InterfaceTranspose
from static_frame.core.style_config import StyleConfig
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.util import (
    CONTINUATION_TOKEN_INACTIVE,
    DEFAULT_SORT_KIND,
    DTYPE_BOOL,
    DTYPE_OBJECT,
    INT_TYPES,
    NAME_DEFAULT,
    NULL_SLICE,
    BoolOrBools,
    DepthLevelSpecifier,
    DtypeSpecifier,
    GetItemKeyType,
    IndexConstructor,
    IndexConstructors,
    IndexInitializer,
    NameType,
    PositionsAllocator,
    UFunc,
    array2d_to_array1d,
    array2d_to_tuples,
    array_sample,
    intersect2d,
    isin,
    isin_array,
    isna_array,
    iterable_to_array_1d,
    iterable_to_array_2d,
    key_to_datetime_key,
    setdiff2d,
    union2d,
    ufunc_unique,
    ufunc_unique1d_counts,
    ufunc_unique1d_indexer,
)

if tp.TYPE_CHECKING:
    from pandas import DataFrame #pylint: disable=W0611 #pragma: no cover
    from static_frame.core.frame import Frame #pylint: disable=W0611,C0412 #pragma: no cover
    from static_frame.core.frame import FrameGO #pylint: disable=W0611,C0412 #pragma: no cover
    from static_frame.core.series import Series #pylint: disable=W0611,C0412 #pragma: no cover

IH = tp.TypeVar('IH', bound='IndexHierarchy')


def build_indexers_from_product(lists: tp.List[tp.Sequence[tp.Any]]) -> tp.List[np.ndarray]:
    """
    Creates a list of indexer arrays for the product of a list of lists.

    Assumes the lists are unique.

    This is equivalent to: ``np.array(list(itertools.product(*lists)))``
    except it scales incredibly well.

    It observes that the indexers for a product will look like this:

    Example:

    >>> lists = [[1, 2, 3], [4, 5, 6]]
    >>> build_indexers_from_product(lists)
    [
        array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
        array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
    ]

    >>> lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> build_indexers_from_product(lists)
    [
        array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]),
        array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]),
        array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]),
    ]
    """

    padded_lengths = np.full(len(lists) + 2, 1, dtype=int)
    padded_lengths[1:-1] = tuple(map(len, lists))

    all_group_reps = np.cumprod(padded_lengths)[:-2]
    all_index_reps = np.cumprod(padded_lengths[::-1])[-3::-1]

    result = []

    for i, (group_reps, index_reps) in enumerate(zip(all_group_reps, all_index_reps)):
        subsection = np.hstack(
            tuple(
                map(
                    # Repeat each index (i.e. element) `index_reps` times
                    functools.partial(np.tile, reps=index_reps),
                    range(padded_lengths[i+1]),
                )
            )
        )

        # Repeat each section `index_reps` times
        indexer = np.tile(subsection, reps=group_reps)
        indexer.flags.writeable = False
        result.append(indexer)

    return result


def blocks_to_container(blocks: tp.Iterator[np.ndarray]) -> np.ndarray:
    return TypeBlocks.from_blocks(blocks).values


def _mask_to_slice_or_ilocs(mask: np.ndarray) -> tp.Union[slice, np.ndarray, int]:
    assert mask.dtype == DTYPE_BOOL

    valid_ilocs = PositionsAllocator.get(len(mask))[mask]

    if len(valid_ilocs) == 1:
        return valid_ilocs[0]

    if len(valid_ilocs) == len(mask):
        return NULL_SLICE

    steps = ufunc_unique(valid_ilocs[1:] - valid_ilocs[:-1])

    if len(steps) == 1:
        [step] = steps
        return slice(valid_ilocs[0], valid_ilocs[-1] + 1, None if step == 1 else step)

    return valid_ilocs


#-------------------------------------------------------------------------------
class IndexHierarchy(IndexBase):
    '''A hierarchy of :obj:`Index` objects, defined as a strict tree of uniform depth across all branches.'''

    __slots__ = (
            '_name',
            '_indices',
            '_indexers',
            '_blocks',
            '_values',
            '_recache',
            )
    _name: NameType
    _indices: tp.List[Index] # Of index objects
    _indexers: tp.List[np.ndarray] # integer arrays
    _blocks: TypeBlocks
    _values: np.ndarray

    # _IMMUTABLE_CONSTRUCTOR is None from IndexBase
    # _MUTABLE_CONSTRUCTOR will be defined after IndexHierarhcyGO defined

    _INDEX_CONSTRUCTOR = Index

    _UFUNC_UNION = union2d
    _UFUNC_INTERSECTION = intersect2d
    _UFUNC_DIFFERENCE = setdiff2d
    _NDIM: int = 2

    #---------------------------------------------------------------------------
    # constructors

    @classmethod
    def _build_index_constructors(cls, index_constructors: IndexConstructors, depth: int) -> IndexConstructors:
        if index_constructors is None:
            return [cls._INDEX_CONSTRUCTOR for _ in range(depth)]

        if callable(index_constructors): # support a single constrctor
            return [index_constructors for _ in range(depth)]

        index_constructors = tuple(index_constructors)

        if len(index_constructors) != depth:
            raise ErrorInitIndex("When providing index constructors, number of index constructors must equal depth of IndexHierarchy.")
        return index_constructors

    @classmethod
    def _build_name_from_indices(cls, indices: tp.List[Index]) -> tp.Optional[tp.Tuple[tp.Hashable, ...]]:
        # build name from index names, assuming they are all specified
        name: tp.Tuple[tp.Hashable, ...] = tuple(index.name for index in indices)
        if any(n is None for n in name):
            return None

        return name

    @classmethod
    def from_product(cls: tp.Type[IH],
            *levels: IndexInitializer,
            name: NameType = None,
            index_constructors: IndexConstructors = None,
            ) -> IH:
        '''
        Given groups of iterables, return an ``IndexHierarchy`` made of the product of a values in those groups, where the first group is the top-most hierarchy.

        Args:
            *levels: index initializers (or Index instances) for each level
            name:
            index_consructors:

        Returns:
            :obj:`static_frame.IndexHierarchy`

        '''
        if len(levels) == 1:
            raise ErrorInitIndex('Cannot create IndexHierarchy from only one level.')

        indices = [] # store in a list, where index is depth

        index_constructors = cls._build_index_constructors(index_constructors, depth=len(levels))

        for lvl, constructor in itertools.zip_longest(levels, index_constructors):
            if constructor is None:
                raise ErrorInitIndex(f'Levels and index_constructors must be the same length.')

            # we call the constructor on all lvl, even if it is already an Index
            # This will raise if any incoming levels are not unique
            if isinstance(lvl, Index):
                indices.append(immutable_index_filter(lvl))
            else:
                indices.append(constructor(lvl))

        if name is None:
            name = cls._build_name_from_indices(indices)

        indexers = build_indexers_from_product(indices)

        return cls(
            name=name,
            indices=indices,
            indexers=indexers,
        )

    @classmethod
    def _from_tree(cls, tree: TreeNodeT) -> tp.Iterator[tp.Tuple[tp.Hashable, ...]]:
        values: tp.List[tp.Tuple[tp.Hashable, ...]] = []
        for label, subtree in tree.items():
            if isinstance(subtree, dict):
                for row in cls._from_tree(subtree):
                    yield (label, *row)
            else:
                for row in subtree: # type: ignore
                    yield (label, row)

        yield from values

    @classmethod
    def from_tree(cls: tp.Type[IH],
            tree: TreeNodeT,
            *,
            name: NameType = None,
            index_constructors: IndexConstructors = None,
            ) -> IH:
        '''
        Convert into a ``IndexHierarchy`` a dictionary defining keys to either iterables or nested dictionaries of the same.

        Returns:
            :obj:`static_frame.IndexHierarchy`
        '''
        return cls.from_labels(
            labels=cls._from_tree(tree),
            name=name,
            index_constructors=index_constructors,
        )

    @classmethod
    def from_labels(cls: tp.Type[IH],
            labels: tp.Iterable[tp.Sequence[tp.Hashable]],
            *,
            name: NameType = None,
            reorder_for_hierarchy: bool = False,
            index_constructors: IndexConstructors = None,
            depth_reference: tp.Optional[DepthLevelSpecifier] = None,
            continuation_token: tp.Union[tp.Hashable, None] = CONTINUATION_TOKEN_INACTIVE
            ) -> IH:
        '''
        Construct an ``IndexHierarhcy`` from an iterable of labels, where each label is tuple defining the component labels for all hierarchies.

        Args:
            labels: an iterator or generator of tuples.
            *,
            name:
            index_constructors:
            depth_reference:
            continuation_token: a Hashable that will be used as a token to identify when a value in a label should use the previously encountered value at the same depth.

        Returns:
            :obj:`static_frame.IndexHierarchy`
        '''
        labels_iter = iter(labels)

        try:
            label_row = next(labels_iter)
        except StopIteration:
            labels_are_empty = True
        else:
            labels_are_empty = False

        if labels_are_empty:
            if isinstance(labels, np.ndarray) and labels.ndim == 2:
                # if this is a 2D array, we can get the depth
                depth = labels.shape[1]

                if depth == 0: # an empty 2D array can have 0 depth
                    pass # do not set depth_reference, assume it is set
                elif ((depth_reference is None and depth > 1)
                        or (depth_reference is not None and depth_reference == depth)):
                    depth_reference = depth
                else:
                    raise ErrorInitIndex(f'depth_reference provided {depth_reference} does not match depth of supplied array {depth}')

            if not isinstance(depth_reference, INT_TYPES):
                raise ErrorInitIndex('depth_reference must be an integer when labels are empty.')

            if depth_reference == 1:
                raise ErrorInitIndex('Cannot create IndexHierarchy from only one level.')

            return cls(
                indices=[cls._INDEX_CONSTRUCTOR(()) for _ in range(depth_reference)],
                indexers=[PositionsAllocator.get(0) for _ in range(depth_reference)],
                name=name
            )

        depth = len(label_row)

        if depth == 1:
            raise ErrorInitIndex('Cannot create IndexHierarchy from only one level.')

        index_constructors = cls._build_index_constructors(index_constructors, depth=depth)

        hash_maps: tp.List[tp.Dict[tp.Hashable, int]] = [{} for _ in range(depth)]
        indexers: tp.List[tp.List[int]] = [[] for _ in range(depth)]

        prev_row: tp.Sequence[tp.Hashable] = ()

        while True:
            for i_zip, (hash_map, indexer, val) in enumerate(zip(hash_maps, indexers, label_row)):
                if val == continuation_token:
                    if prev_row:
                        i: int = indexer[-1]
                        val = prev_row[i_zip]
                    else:
                        i = len(hash_map)
                        hash_map[val] = len(hash_map)
                elif val not in hash_map:
                    i = len(hash_map)
                    hash_map[val] = len(hash_map)
                else:
                    i = hash_map[val]

                indexer.append(i)

            prev_row = label_row
            try:
                label_row = next(labels_iter)
            except StopIteration:
                break

            if len(label_row) != depth:
                raise ErrorInitIndex("All labels must have the same depth.")

        indices = [
            constructor(hash_map)
            for constructor, hash_map
            in zip(index_constructors, hash_maps)
        ]

        if name is None:
            name = cls._build_name_from_indices(indices)

        for i in range(depth):
            indexers[i] = np.array(indexers[i], dtype=int)
            indexers[i].flags.writeable = False # type: ignore

        return cls(indices=indices, indexers=indexers, name=name)

    @classmethod
    def from_index_items(cls: tp.Type[IH],
            items: tp.Iterable[tp.Tuple[tp.Hashable, Index]],
            *,
            index_constructor: tp.Optional[IndexConstructor] = None,
            name: NameType = None,
            ) -> IH:
        '''
        Given an iterable of pairs of label, :obj:`Index`, produce an :obj:`IndexHierarchy` where the labels are depth 0, the indices are depth 1.

        Args:
            items: iterable of pairs of label, :obj:`Index`.
            index_constructor: Optionally provide index constructor for outermost index.
        '''
        [depth1_constructor, depth2_constructor] = cls._build_index_constructors(index_constructor, depth=2)

        depth_1_index = []
        depth_2_index: tp.Optional[IndexGO] = None
        indexers_1 = []
        repeats = []

        for label, index in items:
            index: IndexGO = depth2_constructor(index) # type: ignore
            index = mutable_immutable_index_filter(cls.STATIC, index) #type: ignore

            depth_1_index.append(label)

            if depth_2_index is None:
                # We will grow this in-place
                depth_2_index = mutable_immutable_index_filter(False, index) # type: ignore
                new_indexer = PositionsAllocator.get(len(depth_2_index))
            else:
                new_labels = index.difference(depth_2_index) # Retains order!

                if new_labels.size:
                    depth_2_index.extend(new_labels)

                new_indexer = index.iter_label().apply(depth_2_index._loc_to_iloc)

            indexers_1.append(new_indexer)
            repeats.append(len(index))

        if not depth_1_index:
            assert depth_2_index is None
            return cls(
                indices=[cls._INDEX_CONSTRUCTOR(()) for _ in range(2)],
                indexers=[PositionsAllocator.get(0) for _ in range(2)],
                name=name,
            )

        def _repeat(i_repeat_tuple: tp.Tuple[int, int]) -> np.ndarray:
            i, repeat = i_repeat_tuple
            return np.tile(i, reps=repeat)

        indexers = []
        indexers.append(np.hstack(tuple(map(_repeat, enumerate(repeats)))))
        indexers.append(np.hstack(indexers_1))

        indexers[0].flags.writeable = False
        indexers[1].flags.writeable = False

        return cls(
            indices=[depth1_constructor(depth_1_index), depth_2_index], # type: ignore
            indexers=indexers,
            name=name,
        )

    @classmethod
    def from_labels_delimited(cls: tp.Type[IH],
            labels: tp.Iterable[str],
            *,
            delimiter: str = ' ',
            name: NameType = None,
            index_constructors: IndexConstructors = None,
            ) -> IH:
        '''
        Construct an :obj:`IndexHierarchy` from an iterable of labels, where each label is string defining the component labels for all hierarchies using a string delimiter. All components after splitting the string by the delimited will be literal evaled to produce proper types; thus, strings must be quoted.

        Args:
            labels: an iterator or generator of tuples.

        Returns:
            :obj:`static_frame.IndexHierarchy`
        '''
        def to_label(label: str) -> tp.Tuple[tp.Hashable, ...]:

            start, stop = None, None
            if label[0] in ('[', '('):
                start = 1
            if label[-1] in (']', ')'):
                stop = -1

            if start is not None or stop is not None:
                label = label[start: stop]

            parts = label.split(delimiter)
            if len(parts) <= 1:
                raise RuntimeError(f'Could not not parse more than one label from delimited string: {label}')

            try:
                return tuple(literal_eval(p) for p in parts)
            except ValueError as e:
                raise ValueError('A label is malformed. This may be due to not quoting a string label') from e

        return cls.from_labels(
                (to_label(label) for label in labels),
                name=name,
                index_constructors=index_constructors
                )

    @classmethod
    def from_names(cls: tp.Type[IH], names: tp.Iterable[tp.Hashable]) -> IH:
        '''
        Construct a zero-length :obj:`IndexHierarchy` from an iterable of ``names``, where the length of ``names`` defines the zero-length depth.

        Args:
            names: Iterable of hashable names per depth.
        '''
        name = tuple(names)
        if len(name) == 0:
            raise ErrorInitIndex("names must be non-empty.")

        return cls(
            indices=[cls._INDEX_CONSTRUCTOR((), name=name) for name in names],
            indexers=[PositionsAllocator.get(0) for _ in names],
            name=name,
        )

    @classmethod
    def _from_type_blocks(cls: tp.Type[IH],
            blocks: TypeBlocks,
            *,
            name: NameType = None,
            index_constructors: IndexConstructors = None,
            own_blocks: bool = False,
            ) -> IH:
        '''
        Construct an :obj:`IndexHierarchy` from a :obj:`TypeBlocks` instance.

        Args:
            blocks: a TypeBlocks instance

        Returns:
            :obj:`IndexHierarchy`
        '''
        indices: tp.List[Index] = []
        indexers: tp.List[np.ndarray] = []

        if index_constructors is None:
            constructor_iter = (cls._INDEX_CONSTRUCTOR for _ in range(blocks.shape[1]))
        elif callable(index_constructors):
            constructor_iter = (index_constructors for _ in range(blocks.shape[1]))
        else:
            constructor_iter = index_constructors # type: ignore

        for i, (block, constructor) in enumerate(itertools.zip_longest(blocks, constructor_iter)):

            if block is None or constructor is None:
                raise ErrorInitIndex(f'Levels and index_constructors must be the same length.')

            unique_values, indexer = ufunc_unique1d_indexer(block.values)

            # we call the constructor on all lvl, even if it is already an Index
            try:
                indices.append(constructor(unique_values))
            except ValueError:
                raise ErrorInitIndex(f'Could construct {constructor.__name__} with values at depth {i}')

            indexer.flags.writeable = False
            indexers.append(indexer)

        if index_constructors is not None:
            # If defined, we may have changed columnar dtypes in IndexLevels, and cannot reuse blocks
            if tuple(blocks.dtypes) != tuple(index.dtype for index in indices):
                blocks = None #type: ignore
                own_blocks = False

        return cls(
            indices=indices,
            indexers=indexers,
            name=name,
            blocks=blocks,
            own_blocks=own_blocks,
        )

    # NOTE: could have a _from_fields (or similar) that takes a sequence of column iterables/arrays

    @staticmethod
    def _ensure_uniqueness(indexers: tp.List[np.ndarray], values: np.ndarray) -> None:
        duplicates = array_to_duplicated(np.array(indexers), axis=1, exclude_first=True, exclude_last=False)

        if any(duplicates):
            first_duplicate = values[np.argmax(duplicates)]
            msg = f'Labels have {sum(duplicates)} non-unique values, including {tuple(first_duplicate)}.'
            raise ErrorInitIndexNonUnique(msg)

    def _gen_blocks_from_self(self) -> TypeBlocks:
        def gen_blocks() -> tp.Iterator[np.ndarray]:
            for i, index in enumerate(self._indices):
                indexer = self._indexers[i]
                yield index.values[indexer]

        return TypeBlocks.from_blocks(gen_blocks())

    #---------------------------------------------------------------------------
    def __init__(self,
            indices: tp.Union["IndexHierarchy", tp.List[Index]],
            *,
            indexers: tp.List[np.ndarray] = (), # type: ignore
            name: NameType = NAME_DEFAULT,
            blocks: tp.Optional[TypeBlocks] = None,
            own_blocks: bool = False,
            ):
        '''
        Initializer.

        Args:
            indices: list of :obj:`Index` objects
            indexers: list of indexer arrays
            name: name of the IndexHierarchy
        '''
        # TODO: Really ugly hack. Better to create specialized constructor
        if isinstance(indices, IndexHierarchy):
            if indexers:
                raise ErrorInitIndex('indexers must not be provided when copying an IndexHierarchy')
            if blocks is not None:
                raise ErrorInitIndex('blocks must not be provided when copying an IndexHierarchy')

            self._indices = [mutable_immutable_index_filter(self.STATIC, idx) for idx in indices._indices]
            self._indexers = indices._indexers
            self._name = indices._name
            self._blocks = indices._blocks
            return

        if not all(isinstance(arr, np.ndarray) for arr in indexers):
            raise ErrorInitIndex("indexers must be numpy arrays.")

        if not all(not arr.flags.writeable for arr in indexers):
            raise ErrorInitIndex("indexers must be read-only.")

        if not all(isinstance(index, Index) for index in indices):
            raise ErrorInitIndex("indices must be Index's!")

        self._indices = [mutable_immutable_index_filter(self.STATIC, idx) for idx in indices]
        self._indexers = indexers
        self._name = None if name is NAME_DEFAULT else name_filter(name)

        if blocks is not None:
            if own_blocks:
                self._blocks = blocks
            else:
                self._blocks = blocks.copy()
        else:
            self._blocks = self._gen_blocks_from_self()

        self._values = self._blocks.values

        self._ensure_uniqueness(self._indexers, self.values)
        self._recache = False

    def _update_array_cache(self) -> None:
        pass

    #---------------------------------------------------------------------------
    def __deepcopy__(self: IH, memo: tp.Dict[int, tp.Any]) -> IH:
        obj = self.__new__(self.__class__)
        obj._indices = deepcopy(self._indices, memo)
        obj._indexers = deepcopy(self._indexers, memo)
        obj._blocks = deepcopy(self._blocks, memo)
        obj._name = self._name # should be hashable/immutable

        memo[id(self)] = obj
        return obj #type: ignore

    def __copy__(self: IH) -> IH:
        '''
        Return a shallow copy of this IndexHierarchy.
        '''
        blocks = self._blocks.copy()
        return self.__class__(
                indices=self._indices,
                indexers=self._indexers,
                name=self._name,
                blocks=blocks,
                own_blocks=True
                )

    def copy(self: IH) -> IH:
        '''
        Return a shallow copy of this IndexHierarchy.
        '''
        return self.__copy__()

    #---------------------------------------------------------------------------
    # name interface

    def rename(self: IH, name: NameType) -> IH:
        '''
        Return a new IndexHierarchy with an updated name attribute.
        '''
        if self.STATIC:
            indices = self._indices
            blocks = self._blocks
        else:
            indices = [idx.copy() for idx in self._indices]
            blocks = self._blocks.copy()

        return self.__class__(
                indices=indices,
                indexers=list(self._indexers),
                name=name,
                blocks=blocks,
                own_blocks=True,
                )

    #---------------------------------------------------------------------------
    # interfaces

    @property
    def loc(self) -> InterfaceGetItem["IndexHierarchy"]:
        return InterfaceGetItem(self._extract_loc) #type: ignore

    @property
    def iloc(self) -> InterfaceGetItem["IndexHierarchy"]:
        return InterfaceGetItem(self._extract_iloc) #type: ignore

    def _iter_label(self,
            depth_level: tp.Optional[DepthLevelSpecifier] = None,
            ) -> tp.Iterator[tp.Hashable]:

        if depth_level is None: # default to full labels
            depth_level = list(range(self.depth))

        if isinstance(depth_level, INT_TYPES):
            yield from self._blocks._extract_array(column_key=depth_level)
        else:
            yield from array2d_to_tuples(
                    self._blocks._extract_array(column_key=depth_level)
                    )

    def _iter_label_items(self,
            depth_level: tp.Optional[DepthLevelSpecifier] = None,
            ) -> tp.Iterator[tp.Tuple[int, tp.Hashable]]:
        '''This function is not directly called in iter_label or related routines, fulfills the expectations of the IterNodeDepthLevel interface.
        '''
        yield from enumerate(self._iter_label(depth_level=depth_level))

    @property
    def iter_label(self) -> IterNodeDepthLevel[tp.Any]:
        return IterNodeDepthLevel(
                container=self,
                function_items=self._iter_label_items,
                function_values=self._iter_label,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.INDEX_LABELS
                )

    # NOTE: Index implements drop property

    @property #type: ignore
    @doc_inject(select='astype')
    def astype(self) -> InterfaceAsType[TContainer]:
        '''
        Retype one or more depths. Can be used as as function to retype the entire ``IndexHierarchy``; alternatively, a ``__getitem__`` interface permits retyping selected depths.

        Args:
            {dtype}
        '''
        return InterfaceAsType(func_getitem=self._extract_getitem_astype) #type: ignore

    #---------------------------------------------------------------------------
    @property
    def via_str(self) -> InterfaceString[np.ndarray]:
        '''
        Interface for applying string methods to elements in this container.
        '''
        return InterfaceString(
                blocks=self._blocks._blocks,
                blocks_to_container=blocks_to_container,
                )

    @property
    def via_dt(self) -> InterfaceDatetime[np.ndarray]:
        '''
        Interface for applying datetime properties and methods to elements in this container.
        '''
        return InterfaceDatetime(
                blocks=self._blocks._blocks,
                blocks_to_container=blocks_to_container,
                )

    @property
    def via_T(self) -> InterfaceTranspose["IndexHierarchy"]:
        '''
        Interface for using binary operators with one-dimensional sequences, where the opperand is applied column-wise.
        '''
        return InterfaceTranspose(container=self)

    def via_re(self,
            pattern: str,
            flags: int = 0,
            ) -> InterfaceRe[np.ndarray]:
        '''
        Interface for applying regular expressions to elements in this container.
        '''
        return InterfaceRe(
                blocks=self._blocks._blocks,
                blocks_to_container=blocks_to_container,
                pattern=pattern,
                flags=flags,
                )

    #---------------------------------------------------------------------------

    @property # type: ignore
    @doc_inject()
    def mloc(self) -> int:
        '''{doc_int}
        '''
        return self._blocks.mloc #type: ignore

    @property
    def dtypes(self) -> 'Series':
        '''
        Return a Series of dytpes for each index depth.

        Returns:
            :obj:`static_frame.Series`
        '''
        from static_frame.core.series import Series

        if self._name and isinstance(self._name, tuple) and len(self._name) == self.depth:
            labels: NameType = self._name
        else:
            labels = None

        return Series(self._blocks.dtypes, index=labels)

    @property
    def shape(self) -> tp.Tuple[int, ...]:
        '''
        Return a tuple describing the shape of the underlying NumPy array.

        Returns:
            :obj:`tp.Tuple[int]`
        '''
        return self._blocks._shape

    @property
    def ndim(self) -> int:
        '''
        Return the number of dimensions.

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
        return self._blocks.size

    @property
    def nbytes(self) -> int:
        '''
        Return the total bytes of the underlying NumPy array.

        Returns:
            :obj:`int`
        '''
        return self._blocks.nbytes

    #---------------------------------------------------------------------------

    def __len__(self) -> int:
        return self._blocks.__len__()

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

        sub_display: tp.Optional[Display] = None

        header_sub: tp.Optional[str]
        header: tp.Optional[DisplayHeader]

        if config.type_show:
            header = DisplayHeader(self.__class__, self._name)
            header_depth = 1
            header_sub = '' # need spacer
        else:
            header = None
            header_depth = 0
            header_sub = None

        for col in self._blocks.axis_values(0):
            # as a slice this is far more efficient as no copy is made
            if sub_display is None: # the first
                sub_display = Display.from_values(
                        col,
                        header=header,
                        config=config,
                        outermost=True,
                        index_depth=0,
                        header_depth=header_depth,
                        style_config=style_config,
                        )
            else:
                sub_display.extend_iterable(col, header=header_sub)

        return sub_display #type: ignore

    #---------------------------------------------------------------------------
    # set operations

    def _ufunc_set(self,
            func: tp.Callable[[np.ndarray, np.ndarray, bool], np.ndarray],
            other: tp.Union['IndexBase', tp.Iterable[tp.Hashable]]
            ) -> 'IndexHierarchy':
        '''
        Utility function for preparing and collecting values for Indices to produce a new Index.
        '''
        # can compare equality without cache update
        if self.equals(other, compare_dtype=True):
            # compare dtype as result should be resolved, even if values are the same
            if (func is self.__class__._UFUNC_INTERSECTION or
                    func is self.__class__._UFUNC_UNION):
                # NOTE: this will delegate name attr
                return self if self.STATIC else self.copy()
            elif func is self.__class__._UFUNC_DIFFERENCE:
                # we will no longer have type associations per depth
                return self.__class__.from_labels((), depth_reference=self.depth)

        if isinstance(other, np.ndarray):
            operand = other
            assume_unique = False
        elif isinstance(other, IndexBase):
            operand = other.values
            assume_unique = True # can always assume unique
        else:
            operand = iterable_to_array_2d(other) #type: ignore
            assume_unique = False

        both_sized = len(operand) > 0 and len(self) > 0

        if operand.ndim != 2:
            raise ErrorInitIndex('operand in IndexHierarchy set operations must ndim of 2')
        if both_sized and self.shape[1] != operand.shape[1]:
            raise ErrorInitIndex('operands in IndexHierarchy set operations must have matching depth')

        cls = self.__class__

        # using assume_unique will permit retaining order when operands are identical
        labels = func(self.values, operand, assume_unique=assume_unique) # type: ignore

        # derive index_constructors for IndexHierarchy
        index_constructors: tp.Optional[tp.Sequence[tp.Type[IndexBase]]]

        if both_sized and isinstance(other, IndexHierarchy):
            index_constructors = []
            # depth, and length of index_types, must be equal
            for cls_self, cls_other in zip(self._index_constructors, other._index_constructors):
                if cls_self == cls_other:
                    index_constructors.append(cls_self)
                else:
                    index_constructors.append(Index)
        else:
            # if other is not an IndexHierarchy, do not try to propagate types
            index_constructors = None

        return cls.from_labels(labels,
                index_constructors=index_constructors,
                depth_reference=self.depth)

    #---------------------------------------------------------------------------

    @property
    def _index_constructors(self) -> tp.Tuple[tp.Type[Index], ...]:
        return tuple(index.__class__ for index in self._indices)

    def _drop_iloc(self, key: GetItemKeyType) -> 'IndexHierarchy':
        '''Create a new index after removing the values specified by the loc key.
        '''
        blocks = TypeBlocks.from_blocks(self._blocks._drop_blocks(row_key=key))

        return self.__class__._from_type_blocks(blocks,
                index_constructors=self._index_constructors,
                name=self._name,
                own_blocks=True
                )

    def _drop_loc(self, key: GetItemKeyType) -> 'IndexHierarchy':
        '''Create a new index after removing the values specified by the loc key.
        '''
        return self._drop_iloc(self._loc_to_iloc(key))

    #---------------------------------------------------------------------------

    @property #type: ignore
    @doc_inject(selector='values_2d', class_name='IndexHierarchy')
    def values(self) -> np.ndarray:
        '''
        {}
        '''
        return self._blocks.values

    @property
    def positions(self) -> np.ndarray:
        '''Return the immutable positions array.
        '''
        return PositionsAllocator.get(self.__len__())

    @property
    def depth(self) -> int: #type: ignore
        return len(self._indices)

    def values_at_depth(self,
            depth_level: DepthLevelSpecifier = 0
            ) -> np.ndarray:
        '''
        Return an NP array for the ``depth_level`` specified.

        Args:
            depth_level: a single depth level, or iterable depth of depth levels.
        '''
        sel: GetItemKeyType

        if isinstance(depth_level, INT_TYPES):
            sel = depth_level
        else:
            sel = list(depth_level)

        return self._blocks._extract_array(column_key=sel)

    # TODO: VERY SLOW!
    @doc_inject()
    def label_widths_at_depth(self,
            depth_level: DepthLevelSpecifier = 0
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, int]]:
        '''{}'''
        pos: tp.Optional[int] = None

        if depth_level is None:
            raise NotImplementedError('depth_level of None is not supported')

        if not isinstance(depth_level, INT_TYPES):
            sel = list(depth_level)
            if len(sel) == 1:
                pos = sel.pop()
        else: # is an int
            pos = depth_level

        if pos is None:
            raise NotImplementedError("selecting multiple depth levels is not yet implemented")

        def _extractor(arr: np.ndarray, pos: int) -> tp.Iterator[tp.Tuple[tp.Hashable, int]]:
            unique, widths = ufunc_unique1d_counts(arr)
            labels = self._indices[pos].values[unique]
            yield from zip(labels, widths)

        # i.e. depth_level is an int
        if pos == 0:
            arr = self._indexers[pos]
            yield from _extractor(arr, pos)
            return

        def gen()-> tp.Iterator[tp.Tuple[tp.Hashable, int]]:
            for outer_level_idxs in itertools.product(*map(range, map(len, self._indices[:pos]))):
                screen = np.full(self.__len__(), True, dtype=bool)

                for i, outer_level_idx in enumerate(outer_level_idxs):
                    screen &= (self._indexers[i] == outer_level_idx)

                arr = self._indexers[pos][screen] # type: ignore
                yield from _extractor(arr, pos)

        yield from gen()

    @property
    def index_types(self) -> 'Series':
        '''
        Return a Series of Index classes for each index depth.

        Returns:
            :obj:`Series`
        '''
        from static_frame.core.series import Series

        labels: NameType

        if self._name and isinstance(self._name, tuple) and len(self._name) == self.depth:
            labels = self._name
        else:
            labels = None

        # NOTE: consider caching index_types
        return Series(self._index_constructors, index=labels, dtype=DTYPE_OBJECT)

    #---------------------------------------------------------------------------
    def relabel(self, mapper: RelabelInput) -> 'IndexHierarchy':
        '''
        Return a new IndexHierarchy with labels replaced by the callable or mapping; order will be retained. If a mapping is used, the mapping should map tuple representation of labels, and need not map all origin keys.
        '''
        if not callable(mapper):
            # if a mapper, it must support both __getitem__ and __contains__
            getitem = getattr(mapper, 'get')

            def gen() -> tp.Iterator[tp.Tuple[tp.Hashable, ...]]:
                for array in self._blocks.axis_values(axis=1):
                    # as np.ndarray are not hashable, must tuplize
                    label = tuple(array)
                    yield getitem(label, label)

            return self.__class__.from_labels(gen(),
                    name=self._name,
                    index_constructors=self._index_constructors,
                    )

        return self.__class__.from_labels(
                (mapper(x) for x in self._blocks.axis_values(axis=1)), #type: ignore
                name=self._name,
                index_constructors=self._index_constructors,
                )

    def relabel_at_depth(self,
            mapper: RelabelInput,
            depth_level: DepthLevelSpecifier = 0
            ) -> "IndexHierarchy":
        '''
        Return a new :obj:`IndexHierarchy` after applying `mapper` to a level or each individual level specified by `depth_level`.

        `mapper` can be a callable, mapping, or iterable.
            - If a callable, it must accept a single value, and return a single value.
            - If a mapping, it must map a single value to a single value.
            - If a iterable, it must be the same length as `self`.

        This call:

        >>> index.relabel_at_depth(mapper, depth_level=[0, 1, 2])

        is equivalent to:

        >>> for level in [0, 1, 2]:
        >>>     index = index.relabel_at_depth(mapper, depth_level=level)

        albeit more efficient.
        '''
        if isinstance(depth_level, INT_TYPES):
            depth_level = [depth_level]
            target_depths = set(depth_level)
        else:
            depth_level = sorted(depth_level)
            target_depths = set(depth_level)

            if len(target_depths) != len(depth_level):
                raise ValueError('depth_levels must be unique')

            if not depth_level:
                raise ValueError('depth_level must be non-empty')

        if any(level < 0 or level >= self.depth for level in depth_level):
            raise ValueError(f'Invalid depth level found. Valid levels: [0-{self.depth - 1}]')

        is_callable = callable(mapper)

        # Special handling for full replacements
        if not is_callable and not hasattr(mapper, 'get'):
            values, _ = iterable_to_array_1d(mapper, count=len(self))

            if len(values) != len(self):
                raise ValueError('Iterable must provide a value for each label')

            def gen() -> tp.Iterator[np.ndarray]:
                for depth_idx in range(self.depth):
                    if depth_idx in target_depths:
                        yield values
                    else:
                        yield self._blocks._extract_array_column(depth_idx)

            return self.__class__._from_type_blocks(
                    TypeBlocks.from_blocks(gen()),
                    name=self._name,
                    index_constructors=self._index_constructors,
                    own_blocks=True
                )

        mapper_func = mapper if is_callable else mapper.__getitem__ # type: ignore

        def get_new_label(label: tp.Hashable) -> tp.Hashable:
            if is_callable or label in mapper: # type: ignore
                return mapper_func(label) # type: ignore
            return label

        new_indices = list(self._indices)
        new_indexers = list(self._indexers)

        for level in depth_level:
            index = self._indices[level]

            new_index: tp.Dict[tp.Hashable, int] = {}
            index_remap: tp.Dict[tp.Hashable, tp.Hashable] = {}

            for label_idx, label in enumerate(index.values):
                new_label = get_new_label(label)

                if new_label not in new_index:
                    new_index[new_label] = len(new_index)
                else:
                    index_remap[label_idx] = new_index[new_label]

            new_indices[level] = index.__class__(new_index)

            indexer = np.array([index_remap.get(i, i) for i in self._indexers[level]])
            indexer.flags.writeable = False
            new_indexers[level] = indexer

        return self.__class__(
            indices=new_indices,
            indexers=new_indexers,
            name=self._name,
        )

    def rehierarch(self: IH,
            depth_map: tp.Sequence[int]
            ) -> IH:
        '''
        Return a new :obj:`IndexHierarchy` that conforms to the new depth assignments given be `depth_map`.
        '''
        index, _ = rehierarch_from_type_blocks(
                labels=self._blocks,
                index_cls=self.__class__,
                index_constructors=self._index_constructors,
                depth_map=depth_map,
                )
        return index #type: ignore

    def _get_outer_index_labels_in_order_they_appear(self) -> tp.Sequence[tp.Hashable]:
        """
        Index could be [A, B, C]
        Indexers could be [2, 0, 0, 2, 1]

        This function return [C, A, B] # shoutout to my initials
        """
        # get the outer level, or just the unique frame labels needed
        labels = self.values_at_depth(0)
        label_indexes = sorted(np.unique(labels, return_index=True)[1])
        return labels[label_indexes] # type: ignore

    #---------------------------------------------------------------------------

    def _process_key_at_depth(self, depth: int, key: GetItemKeyType) -> tp.Union[slice, np.ndarray]:
        if depth >= self.depth:
            raise RuntimeError(f'Invalid depth level for key={key} depth={depth}')

        key_at_depth = key[depth] # type: ignore

        # Key is already a mask!
        if isinstance(key_at_depth, np.ndarray) and key_at_depth.dtype == DTYPE_BOOL:
            return key_at_depth

        index_at_depth = self._indices[depth]
        indexer_at_depth = self._indexers[depth]

        if isinstance(key_at_depth, slice):
            if key_at_depth.start is not None:
                start: tp.Optional[int] = index_at_depth.loc_to_iloc(key_at_depth.start) # type: ignore
            else:
                start = 0

            if key_at_depth.step is not None and not isinstance(key_at_depth.step, INT_TYPES):
                raise NotImplementedError(f"step must be an integer. What does this even mean? {key_at_depth}")

            if key_at_depth.stop is not None:
                stop: tp.Optional[int] = index_at_depth.loc_to_iloc(key_at_depth.stop) + 1 # type: ignore
            else:
                stop = len(indexer_at_depth)

            return isin_array(
                    array=indexer_at_depth,
                    array_is_unique=False,
                    other=np.arange(start, stop, key_at_depth.step),
                    other_is_unique=True
                    )

        key_iloc = index_at_depth.loc_to_iloc(key_at_depth)

        if hasattr(key_iloc, "__len__"):
            if isinstance(key, np.ndarray):
                return isin_array(
                        array=indexer_at_depth,
                        array_is_unique=False,
                        other=key_iloc,
                        other_is_unique=True
                        )
            return isin(indexer_at_depth, key_iloc)

        return indexer_at_depth == key_iloc

    def _loc_to_iloc(self, key: tp.Union[GetItemKeyType, HLoc]) -> GetItemKeyType:
        '''
        Given iterable (or instance) of GetItemKeyType, determine the equivalent iloc key.

        When possible, prefer slices.
        '''
        if isinstance(key, ILoc):
            return key.key

        if isinstance(key, IndexHierarchy):

            if not key.depth == self.depth:
                raise KeyError(f"Key must have the same depth as the index. {key}")

            # TODO: Explore optimizations here
            return [self._loc_to_iloc(label) for label in key.iter_label()]

        if isinstance(key, np.ndarray) and key.dtype == DTYPE_BOOL:
            # TODO: Can I just return key?
            return self.positions[key]

        if isinstance(key, slice):
            if key == NULL_SLICE:
                return key

            # reuse - slice_to_inclusive_slice and/or LocMap.map_slice_args
            if key.start is not None:
                start: tp.Optional[int] = self._loc_to_iloc(key.start) # type: ignore
            else:
                start = None

            if key.step is not None and not isinstance(key.step, INT_TYPES):
                raise ValueError(f"step must be an integer. What does this even mean? {key}")

            if key.stop is not None:
                stop: tp.Optional[int] = self._loc_to_iloc(key.stop) + 1 # type: ignore
            else:
                stop = None

            return slice(start, stop, key.step)

        if isinstance(key, list):
            return [self._loc_to_iloc(k) for k in key]

        if isinstance(key, np.ndarray) and key.dtype != DTYPE_BOOL and key.ndim == 2:
            return [self._loc_to_iloc(k) for k in key]

        if key.__class__ is HLoc:
            # unpack any Series, Index, or ILoc into the context of this IndexHierarchy
            key = tuple(
                    HLoc(
                        tuple(
                            key_from_container_key(self, k, expand_iloc=True)
                            for k in key # type: ignore
                        )
                    )
                )
        else:
            # If the key is a series, key_from_container_key will invoke IndexCorrespondence
            # logic that eventually calls _loc_to_iloc on all the indices of that series.
            key = key_from_container_key(self, key)
            if isinstance(key, np.ndarray) and key.dtype == DTYPE_BOOL:
                return PositionsAllocator.get(len(key))[key]

            key = tuple(key)

        if any(isinstance(k, tuple) for k in key):
            return [self._loc_to_iloc(k) for k in key]

        meaningful_selections = {depth: not (isinstance(k, slice) and k == NULL_SLICE) for depth, k in enumerate(key)}

        can_return_element = all(
                meaningful and (isinstance(key[depth], str) or not hasattr(key[depth], "__len__"))
                for depth, meaningful
                in meaningful_selections.items()
                )

        meaningful_depths = sum(meaningful_selections.values())

        # Return a slice wherever possible
        if meaningful_depths == 1:

            depth = next(i for i, meaningful in meaningful_selections.items() if meaningful)
            mask = self._process_key_at_depth(depth=depth, key=key)

            if isinstance(mask, slice):
                return mask # Not actually a mask

        else:
            mask_2d = np.full(self.shape, True, dtype=bool)

            for depth, meaningful in meaningful_selections.items():
                if not meaningful:
                    continue

                result = self._process_key_at_depth(depth=depth, key=key)

                if isinstance(result, slice):
                    mask_2d[:, depth] = False
                    mask_2d[result, depth] = True
                else:
                    mask_2d[:, depth] = result

            mask = mask_2d.all(axis=1)
            del mask_2d

        result = PositionsAllocator.get(len(mask))[mask]

        # Even if there was one result, unless the HLoc specified all levels, we need to return a list
        if len(result) == 1 and can_return_element:
            return result[0]
        return result

    def loc_to_iloc(self,
            key: tp.Union[GetItemKeyType, HLoc]
            ) -> GetItemKeyType:
        '''Given a label (loc) style key (either a label, a list of labels, a slice, or a Boolean selection), return the index position (iloc) style key. Keys that are not found will raise a KeyError or a sf.LocInvalid error.

        Args:
            key: a label key.
        '''
        # NOTE: the public method is the same as the private method for IndexHierarchy, but not for Index
        return self._loc_to_iloc(key)

    def _extract_iloc(self,
            key: GetItemKeyType,
            ) -> tp.Union['IndexHierarchy', tp.Tuple[tp.Hashable]]:
        '''Extract a new index given an iloc key
        '''
        if isinstance(key, INT_TYPES):
            # return a tuple if selecting a single row
            # NOTE: if extracting a single row, should be able to get it from IndexLevel without forcing a complete recache
            # NOTE: Selecting a single row may force type coercion before values are added to the tuple; i.e., a datetime64 will go to datetime.date before going to the tuple
            return tuple(self._blocks._extract_array(row_key=key)) #type: ignore

        tb = self._blocks._extract(row_key=key)

        return self.__class__._from_type_blocks(tb,
                name=self._name,
                index_constructors=self._index_constructors,
                own_blocks=True,
                )

    def _extract_loc(self,
            key: GetItemKeyType
            ) -> tp.Union['IndexHierarchy', tp.Tuple[tp.Hashable]]:
        return self._extract_iloc(self._loc_to_iloc(key))

    def __getitem__(self, #pylint: disable=E0102
            key: GetItemKeyType
            ) -> tp.Union['IndexHierarchy', tp.Tuple[tp.Hashable]]:
        '''Extract a new index given an iloc key.
        '''
        return self._extract_iloc(key)

    #---------------------------------------------------------------------------

    def _extract_getitem_astype(self, key: GetItemKeyType) -> 'IndexHierarchyAsType':
        '''Given an iloc key (using integer positions for columns) return a configured IndexHierarchyAsType instance.
        '''
        # key is an iloc key
        if isinstance(key, tuple):
            raise KeyError('__getitem__ does not support multiple indexers')
        return IndexHierarchyAsType(self, key=key)

    #---------------------------------------------------------------------------
    # operators

    def _ufunc_unary_operator(self,
            operator: UFunc
            ) -> np.ndarray:
        '''Always return an NP array.
        '''
        values = self._blocks.values
        array = operator(values)
        array.flags.writeable = False
        return array

    def _ufunc_binary_operator(self, *,
            operator: UFunc,
            other: tp.Any,
            axis: int = 0,
            fill_value: object = np.nan,
            ) -> np.ndarray:
        '''
        Binary operators applied to an index always return an NP array. This deviates from Pandas, where some operations (multiplying an int index by an int) result in a new Index, while other operations result in a np.array (using == on two Index).
        '''
        from static_frame.core.series import Series
        from static_frame.core.frame import Frame

        if isinstance(other, (Series, Frame)):
            raise ValueError('cannot use labelled container as an operand.')

        if operator.__name__ == 'matmul':
            return matmul(self._blocks.values, other)
        elif operator.__name__ == 'rmatmul':
            return matmul(other, self._blocks.values)

        if isinstance(other, Index):
            other = other.values
        elif isinstance(other, IndexHierarchy):
            other = other._blocks

        tb = self._blocks._ufunc_binary_operator(
                operator=operator,
                other=other,
                axis=axis,
                )
        return tb.values

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
        Returns:
            immutable NumPy array.
        '''
        dtype = None if not dtypes else dtypes[0] # must be a tuple
        values = self._blocks.values

        if skipna:
            post = ufunc_skipna(values, axis=axis, dtype=dtype)
        else:
            post = ufunc(values, axis=axis, dtype=dtype)

        post.flags.writeable = False
        return post

    # _ufunc_shape_skipna defined in IndexBase

    #---------------------------------------------------------------------------
    # dictionary-like interface

    # NOTE: we intentionally exclude keys(), items(), and get() from Index classes, as they return inconsistent result when thought of as a dictionary

    def __iter__(self) -> tp.Iterator[tp.Tuple[tp.Hashable, ...]]:
        '''Iterate over labels.
        '''
        # Don't use .values, as that can coerce types
        yield from zip(*map(self.values_at_depth, range(self.depth)))

    def __reversed__(self) -> tp.Iterator[tp.Tuple[tp.Hashable, ...]]:
        '''
        Returns a reverse iterator on the index labels.
        '''
        for array in self._blocks.axis_values(1, reverse=True):
            yield tuple(array)

    def __contains__(self, value: tp.Tuple[tp.Hashable]) -> bool: # type: ignore
        '''Determine if a leaf loc is contained in this Index.
        '''
        # TODO: Can this be optimized, or is all the optimization already done in _loc_to_iloc?
        try:
            result = self._loc_to_iloc(value)
        except KeyError:
            return False

        if isinstance(result, np.ndarray):
            return bool(result.size)

        if isinstance(result, list):
            return bool(result)

        return True

    #---------------------------------------------------------------------------
    # utility functions

    def unique(self,
            depth_level: DepthLevelSpecifier = 0
            ) -> np.ndarray:
        '''
        Return a NumPy array of unique values.

        Args:
            depth_level: Specify a single depth or multiple depths in an iterable.

        Returns:
            :obj:`numpy.ndarray`
        '''
        pos: tp.Optional[int] = None
        if not isinstance(depth_level, INT_TYPES):
            sel = list(depth_level)
            if len(sel) == 1:
                pos = sel.pop()
        else: # is an int
            pos = depth_level

        if pos is not None: # i.e. a single level
            return self._indices[pos].values

        return ufunc_unique(array2d_to_array1d(self.values_at_depth(sel)))

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
        # NOTE: do not need to udpate array cache, as can compare elements in levels
        if id(other) == id(self):
            return True

        if compare_class and self.__class__ != other.__class__:
            return False

        if not isinstance(other, IndexHierarchy):
            return False

        # same type from here
        if self.shape != other.shape:
            return False

        if compare_name and self.name != other.name:
            return False

        if compare_dtype and not self.dtypes.equals(other.dtypes):
            return False

        if compare_class:
            for self_index, other_index in zip(self._indices, other._indices):
                if self_index.__class__ != other_index.__class__:
                    return False

        for i in range(self.depth):
            if not arrays_equal(self.values_at_depth(i), other.values_at_depth(i), skipna=skipna):
                return False

        return True

    @doc_inject(selector='sort')
    def sort(self: IH,
            *,
            ascending: BoolOrBools = True,
            kind: str = DEFAULT_SORT_KIND,
            key: tp.Optional[tp.Callable[['IndexHierarchy'], tp.Union[np.ndarray, 'IndexHierarchy']]] = None,
            ) -> IH:
        '''Return a new Index with the labels sorted.

        Args:
            {ascendings}
            {kind}
            {key}
        '''
        order = sort_index_for_order(self, kind=kind, ascending=ascending, key=key) #type: ignore [arg-type]

        blocks = self._blocks._extract(row_key=order)

        return self.__class__._from_type_blocks(blocks,
                index_constructors=self._index_constructors,
                name=self._name,
                own_blocks=True
                )

    def isin(self, other: tp.Iterable[tp.Iterable[tp.Hashable]]) -> np.ndarray:
        '''
        Return a Boolean array showing True where one or more of the passed in iterable of labels is found in the index.
        '''
        matches = []
        for seq in other:
            if not hasattr(seq, '__iter__'):
                raise RuntimeError('must provide one or more iterables within an iterable')
            # Coerce to hashable type
            as_tuple = tuple(seq)
            if len(as_tuple) == self.depth:
                # can pre-filter if iterable matches to length
                matches.append(as_tuple)

        if not matches:
            return np.full(self.__len__(), False, dtype=bool)

        return isin(self.flat().values, matches)

    def roll(self, shift: int) -> 'IndexHierarchy':
        '''Return an :obj:`IndexHierarchy` with values rotated forward and wrapped around (with a positive shift) or backward and wrapped around (with a negative shift).
        '''
        blocks = TypeBlocks.from_blocks(
                self._blocks._shift_blocks(row_shift=shift, wrap=True)
                )

        return self.__class__._from_type_blocks(blocks,
                index_constructors=self._index_constructors,
                name=self._name,
                own_blocks=True
                )

    @doc_inject(selector='fillna')
    def fillna(self, value: tp.Any) -> 'IndexHierarchy':
        '''Return an :obj:`IndexHierarchy` after replacing NA (NaN or None) with the supplied value.

        Args:
            {value}
        '''
        blocks = self._blocks.fill_missing_by_unit(value, None, func=isna_array)

        return self.__class__._from_type_blocks(blocks,
                index_constructors=self._index_constructors,
                name=self._name,
                own_blocks=True
                )

    def _sample_and_key(self,
            count: int = 1,
            *,
            seed: tp.Optional[int] = None,
            ) -> tp.Tuple['IndexHierarchy', np.ndarray]:

        # sort to ensure hierarchability
        key = array_sample(self.positions, count=count, seed=seed, sort=True)
        blocks = self._blocks._extract(row_key=key)

        container = self.__class__._from_type_blocks(blocks,
                index_constructors=self._index_constructors,
                name=self._name,
                own_blocks=True
                )
        return container, key

    @doc_inject(selector='searchsorted', label_type='iloc (integer)')
    def iloc_searchsorted(self,
            values: tp.Any,
            *,
            side_left: bool = True,
            ) -> tp.Union[tp.Hashable, tp.Iterable[tp.Hashable]]:
        '''
        {doc}

        Args:
            {values}
            {side_left}
        '''
        if isinstance(values, tuple):
            match_pre = [values] # normalize a multiple selection
            is_element = True
        elif isinstance(values, list):
            match_pre = values
            is_element = False
        else:
            raise NotImplementedError('A single label (as a tuple) or multiple labels (as a list) must be provided.')

        dt_pos = np.array([issubclass(idx_type, IndexDatetime)
                for idx_type in self._index_constructors])
        has_dt = dt_pos.any()

        values_for_match = np.empty(len(match_pre), dtype=object)

        for i, label in enumerate(match_pre):
            if has_dt:
                label = tuple(v if not dt_pos[j] else key_to_datetime_key(v)
                        for j, v in enumerate(label))
            values_for_match[i] = label

        post = self.flat().iloc_searchsorted(values_for_match, side_left=side_left)
        if is_element:
            return post[0] #type: ignore [no-any-return]
        return post #type: ignore [no-any-return]

    @doc_inject(selector='searchsorted', label_type='loc (label)')
    def loc_searchsorted(self,
            values: tp.Any,
            *,
            side_left: bool = True,
            fill_value: tp.Any = np.nan,
            ) -> tp.Union[tp.Hashable, tp.Iterable[tp.Hashable]]:
        '''
        {doc}

        Args:
            {values}
            {side_left}
            {fill_value}
        '''
        # will return an integer or an array of integers
        sel = self.iloc_searchsorted(values, side_left=side_left)

        length = self.__len__()
        if sel.ndim == 0 and sel == length: # an element:
            return fill_value #type: ignore [no-any-return]

        flat = self.flat().values
        mask = sel == length
        if not mask.any():
            return flat[sel] #type: ignore [no-any-return]

        post = np.empty(len(sel), dtype=object)
        sel[mask] = 0 # set out of range values to zero
        post[:] = flat[sel]
        post[mask] = fill_value
        post.flags.writeable = False
        return post #type: ignore [no-any-return]

    #---------------------------------------------------------------------------
    # export

    def _to_frame(self,
            constructor: tp.Type['Frame']
            ) -> 'Frame':

        return constructor(
                self._blocks.copy(),
                columns=None,
                index=None,
                own_data=True
                )

    def to_frame(self) -> 'Frame':
        '''
        Return :obj:`Frame` version of this :obj:`IndexHiearchy`.
        '''
        from static_frame import Frame
        return self._to_frame(Frame)

    def to_frame_go(self) -> 'FrameGO':
        '''
        Return a :obj:`FrameGO` version of this :obj:`IndexHierarchy`.
        '''
        from static_frame import FrameGO
        return self._to_frame(FrameGO) #type: ignore

    def to_pandas(self) -> 'DataFrame':
        '''Return a Pandas MultiIndex.
        '''
        import pandas

        # must copy to get a mutable array
        mi = pandas.MultiIndex(
                levels=[index.values.copy() for index in self._indices],
                codes=[arr.copy() for arr in self._indexers],
                )
        mi.name = self._name
        mi.names = self.names
        return mi

    def _build_tree_at_depth_from_mask(self, depth: int, mask: np.ndarray) -> tp.Union[TreeNodeT, Index]:

        if depth == self.depth - 1:
            values = self._indices[depth][self._indexers[depth][mask]]
            return self._indices[depth].__class__(values)

        tree: TreeNodeT = {}

        index_at_depth = self._indices[depth]
        indexer_at_depth = self._indexers[depth]

        for i in ufunc_unique(indexer_at_depth[mask]):
            tree[index_at_depth[i]] = self._build_tree_at_depth_from_mask(depth + 1, mask & (indexer_at_depth == i))

        return tree

    def to_tree(self) -> TreeNodeT:
        '''Returns the tree representation of an IndexHierarchy
        '''
        tree = self._build_tree_at_depth_from_mask(0, np.ones(len(self), dtype=bool))
        assert isinstance(tree, dict) # mypy
        return tree

    def flat(self) -> Index:
        '''Return a flat, one-dimensional index of tuples for each level.
        '''
        return self._INDEX_CONSTRUCTOR(self.__iter__(), name=self._name)

    def level_add(self: IH,
            level: tp.Hashable,
            *,
            index_constructor: IndexConstructor = None,
            ) -> IH:
        '''Return an IndexHierarchy with a new root (outer) level added.
        '''
        index_cls = self._INDEX_CONSTRUCTOR if index_constructor is None else index_constructor._MUTABLE_CONSTRUCTOR # type: ignore

        if self.STATIC:
            indices = [index_cls((level,)), *self._indices]
        else:
            indices = [index_cls((level,)), *(idx.copy() for idx in self._indices)]

        # Indexers are always immutable
        new_indexer = np.full(self.__len__(), 0, dtype=int)
        new_indexer.flags.writeable = False
        indexers = [new_indexer, *self._indexers]

        def gen_blocks() -> tp.Iterator[np.ndarray]:
            yield np.full(self.__len__(), indices[0].values)
            yield from self._blocks._blocks

        return self.__class__(
            indices=indices,
            indexers=indexers,
            name=self.name,
            blocks=TypeBlocks.from_blocks(gen_blocks()),
            own_blocks=True,
        )

    def level_drop(self,
            count: int = 1,
            ) -> tp.Union[Index, 'IndexHierarchy']:
        '''Return an IndexHierarchy with one or more leaf levels removed. This might change the size of the resulting index if the resulting levels are not unique.

        Args:
            count: A positive value is the number of depths to remove from the root (outer) side of the hierarchy; a negative value is the number of depths to remove from the leaf (inner) side of the hierarchy.
        '''
        # NOTE: this was implement with a bipolar ``count`` to specify what to drop, but it could have been implemented with a depth level specifier, supporting arbitrary removals. The approach taken here is likely faster as we reuse levels.
        if self._name_is_names():
            if count < 0:
                name = self._name[:count] #type: ignore
            elif count > 0:
                name = self._name[count:] #type: ignore
            if len(name) == 1:
                name = name[0]
        else:
            name = self._name

        if count < 0: # remove from inner
            if count <= (1 - self.depth):
                return self._index_constructors[-1](self._blocks.iloc[:,0].values.ravel(), name=name)

            return self.__class__(
                    indices=self._indices[:count],
                    indexers=self._indexers[:count],
                    name=name,
                    blocks=self._blocks[:count],
                    own_blocks=self.STATIC,
                    )

        elif count > 0: # remove from outer
            if count >= (self.depth - 1):
                return self._index_constructors[0](self._blocks.iloc[:,-1].values.ravel(), name=name)

            return self.__class__(
                    indices=self._indices[count:],
                    indexers=self._indexers[count:],
                    name=name,
                    blocks=self._blocks.iloc[:,count:],
                    own_blocks=self.STATIC,
                    )

        raise NotImplementedError('no handling for a 0 count drop level.')


class IndexHierarchyGO(IndexHierarchy):
    '''
    A hierarchy of :obj:`static_frame.Index` objects that permits mutation only in the addition of new hierarchies or labels.
    '''

    STATIC = False

    _IMMUTABLE_CONSTRUCTOR = IndexHierarchy
    _INDEX_CONSTRUCTOR = IndexGO

    __slots__ = (
            '_name',
            '_indices',
            '_indexers',
            '_blocks',
            '_values',
            )

    _indices: tp.List[IndexGO] # type: ignore

    def append(self, value: tp.Sequence[tp.Hashable]) -> None:
        '''
        Append a single label to this index.
        '''
        if value in self: # type: ignore
            raise ErrorInitIndexNonUnique(f"The label '{value}' is already in the index.")

        label_indexers = []

        for depth, label_at_depth in enumerate(value):
            if label_at_depth in self._indices[depth]:
                label_index = self._indices[depth]._loc_to_iloc(label_at_depth)
            else:
                label_index = len(self._indices[depth])
                self._indices[depth].append(label_at_depth)

            label_indexers.append(label_index)

        for i, label_index in enumerate(label_indexers):
            self._indexers[i] = np.append(self._indexers[i], label_index)
            self._indexers[i].flags.writeable = False

        # No need to ensure uniqueness! It's already been checked.
        self._blocks = self._gen_blocks_from_self()
        self._values = self._blocks.values

    def extend(self, other: IndexHierarchy) -> None:
        '''
        Extend this IndexHiearchy in-place
        '''
        for depth, (self_index, other_index) in enumerate(zip(self._indices, other._indices)):

            intersection = self_index.intersection(other_index)
            if not intersection.size:
                del intersection

                starting_len = len(self_index)

                # Easy case! We can simply append
                self_index.extend(other_index)

                new_indexer = np.hstack((self._indexers[depth], other._indexers[depth] + starting_len))
                new_indexer.flags.writeable = False
                self._indexers[depth] = new_indexer
                continue

            if len(intersection) == len(self_index) == len(other_index):
                del intersection

                if self_index.equals(other_index):
                    # Easy case! We just have to append the indexers; no change needed to the index

                    new_indexer = np.hstack((self._indexers[depth], other._indexers[depth]))
                    new_indexer.flags.writeable = False
                    self._indexers[depth] = new_indexer
                    continue

                # Same labels, but different order. We have to remap the indexers.
                indexer_remap = other_index.iter_label().apply(self_index._loc_to_iloc)

                remap_indexer = indexer_remap[other._indexers[depth]]
                new_indexer = np.hstack((self._indexers[depth], remap_indexer))
                new_indexer.flags.writeable = False
                self._indexers[depth] = new_indexer
                continue

            starting_len = len(self_index)

            self_index.extend(other_index[~other_index.isin(intersection)])

            def remap(k: tp.Hashable) -> int:
                if k in intersection:
                    return self_index._loc_to_iloc(k) # type: ignore
                return -1

            offset = starting_len - len(intersection)
            indexer_remap = other_index.iter_label().apply(remap)
            del intersection

            remap_indexer = indexer_remap[other._indexers[depth]]

            mask = remap_indexer == -1

            remap_indexer[mask] = (other._indexers[depth][mask] + offset)
            new_indexer = np.hstack((self._indexers[depth], remap_indexer))
            new_indexer.flags.writeable = False
            self._indexers[depth] = new_indexer

        # No need to ensure uniqueness! It's already been checked.
        self._blocks = self._gen_blocks_from_self()
        self._values = self._blocks.values
        self._ensure_uniqueness(self._indexers, self.values)

    def __copy__(self: IH) -> IH:
        '''
        Return a shallow copy of this IndexHierarchy.
        '''
        return self.__class__(
                indices=[index.copy() for index in self._indices],
                indexers=self._indexers,
                name=self._name,
                blocks=self._blocks.copy(),
                own_blocks=True,
                )


# update class attr on Index after class initialization
IndexHierarchy._MUTABLE_CONSTRUCTOR = IndexHierarchyGO


class IndexHierarchyAsType:

    __slots__ = ('container', 'key',)

    def __init__(self,
            container: IndexHierarchy,
            key: GetItemKeyType
            ) -> None:
        self.container = container
        self.key = key

    def __call__(self, dtype: DtypeSpecifier) -> IndexHierarchy:

        from static_frame.core.index_datetime import dtype_to_index_cls
        container = self.container

        # use TypeBlocks in both situations to avoid double casting
        blocks = TypeBlocks.from_blocks(
                container._blocks._astype_blocks(column_key=self.key, dtype=dtype)
                )

        # avoid coercion of datetime64 arrays that were not targetted in the selection
        index_constructors = container.index_types.values.copy()

        dtype_post = blocks.dtypes[self.key] # can select element or array
        if isinstance(dtype_post, np.dtype):
            index_constructors[self.key] = dtype_to_index_cls(
                    container.STATIC,
                    dtype_post)
        else: # assign iterable
            index_constructors[self.key] = [
                    dtype_to_index_cls(container.STATIC, dt)
                    for dt in dtype_post]

        return container.__class__._from_type_blocks(
                blocks,
                index_constructors=index_constructors,
                own_blocks=True
                )
