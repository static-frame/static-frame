from __future__ import annotations

import functools
import itertools
import typing as tp
from ast import literal_eval
from copy import deepcopy
from itertools import (
    chain,
    repeat,
    zip_longest,
)
from operator import attrgetter

import numpy as np
from arraykit import (
    name_filter,
    resolve_dtype,
)
from static_frame.core.array_go import ArrayGO
from static_frame.core.container_util import (
    index_from_optional_constructor,
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
from static_frame.core.display_config import DisplayConfig
from static_frame.core.doc_str import doc_inject
from static_frame.core.exception import ErrorInitIndex
from static_frame.core.hloc import HLoc
from static_frame.core.index import (
    ILoc,
    Index,
    IndexGO,
    mutable_immutable_index_filter,
)
from static_frame.core.index_auto import RelabelInput
from static_frame.core.index_base import IndexBase
from static_frame.core.index_datetime import IndexDatetime
from static_frame.core.index_level import (
    IndexLevel,
    IndexLevelGO,
    TreeNodeT,
)
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
    EMPTY_TUPLE,
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
    concat_resolved,
    intersect2d,
    isin,
    isna_array,
    iterable_to_array_1d,
    iterable_to_array_2d,
    key_to_datetime_key,
    setdiff2d,
    ufunc_unique,
    union2d,
)
from tqdm import tqdm

if tp.TYPE_CHECKING:
    from pandas import DataFrame #pylint: disable=W0611 #pragma: no cover
    from static_frame.core.frame import Frame #pylint: disable=W0611,C0412 #pragma: no cover
    from static_frame.core.frame import FrameGO #pylint: disable=W0611,C0412 #pragma: no cover
    from static_frame.core.series import Series #pylint: disable=W0611,C0412 #pragma: no cover

IH = tp.TypeVar('IH', bound='IndexHierarchy2')


def build_indexers_from_product(lists: tp.List[list]) -> tp.List[np.ndarray]:
    """
    Creates a list of indexer arrays for the product of a list of lists

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

    lengths = list(map(len, lists))

    padded = [1] + lengths + [1]
    all_group_reps = np.cumprod(padded)[:-2]
    all_index_reps = np.cumprod(padded[::-1])[-3::-1]

    result = []

    for i, (group_reps, index_reps) in enumerate(zip(all_group_reps, all_index_reps)):
        subsection = np.hstack(
            tuple(
                map(
                    # Repeat each index (i.e. element) `index_reps` times
                    functools.partial(np.tile, reps=index_reps),
                    range(lengths[i]),
                )
            )
        )

        # Repeat each section `index_reps` times
        result.append(np.tile(subsection, reps=group_reps))

    return result


def blocks_to_container(blocks: tp.Iterator[np.ndarray]) -> np.ndarray:
    return TypeBlocks.from_blocks(blocks).values

#-------------------------------------------------------------------------------
class IndexHierarchy2(IndexBase):
    '''A hierarchy of :obj:`Index` objects, defined as a strict tree of uniform depth across all branches.'''

    __slots__ = (
            '_name',
            '_indices',
            '_indexers',
            '__blocks',
            )
    _name: NameType
    _indices: ArrayGO # Of index objects
    _indexers: tp.List[np.ndarray] # integer arrays
    __blocks: tp.Optional[TypeBlocks]

    # _IMMUTABLE_CONSTRUCTOR is None from IndexBase
    # _MUTABLE_CONSTRUCTOR will be defined after IndexHierarhcyGO defined

    _INDEX_CONSTRUCTOR = Index
    _LEVEL_CONSTRUCTOR = IndexLevel

    _UFUNC_UNION = union2d
    _UFUNC_INTERSECTION = intersect2d
    _UFUNC_DIFFERENCE = setdiff2d
    _NDIM: int = 2

    #---------------------------------------------------------------------------
    # constructors

    @classmethod
    def from_product(cls: tp.Type[IH],
            *levels: IndexInitializer,
            name: NameType = None,
            index_constructors: tp.Optional[IndexConstructors] = None,
            ) -> IH:
        '''
        Given groups of iterables, return an ``IndexHierarchy2`` made of the product of a values in those groups, where the first group is the top-most hierarchy.

        Args:
            *levels: index initializers (or Index instances) for each level
            name:
            index_consructors:

        Returns:
            :obj:`static_frame.IndexHierarchy2`

        '''
        indices = [] # store in a list, where index is depth
        if index_constructors is None:
            for lvl in levels:
                if not isinstance(lvl, Index): # Index, not IndexBase
                    indices.append(cls._INDEX_CONSTRUCTOR(lvl))
                else:
                    indices.append(lvl)
        else:
            if callable(index_constructors): # support a single constrctor
                pair_iter = zip(levels, itertools.repeat(index_constructors))
            else:
                pair_iter = itertools.zip_longest(levels, index_constructors)

            for lvl, constructor in pair_iter:
                if constructor is None:
                    raise ErrorInitIndex(f'Levels and index_constructors must be the same length.')
                # we call the constructor on all lvl, even if it is already an Index
                indices.append(constructor(lvl))

        if len(indices) == 1:
            raise ErrorInitIndex('Cannot create IndexHierarchy2 from only one level.')

        # build name from index names, assuming they are all specified
        if name is None:
            name = tuple(index.name for index in indices)
            if any(n is None for n in name):
                name = None

        indexers = build_indexers_from_product(indices)

        return cls(
            name=name,
            indices=indices,
            indexers=indexers,
        )

    @classmethod
    def from_tree(cls: tp.Type[IH],
            tree: TreeNodeT,
            *,
            name: NameType = None,
            index_constructors: tp.Optional[IndexConstructors] = None,
            ) -> IH:
        '''
        Convert into a ``IndexHierarchy2`` a dictionary defining keys to either iterables or nested dictionaries of the same.

        Returns:
            :obj:`static_frame.IndexHierarchy2`
        '''
        raise NotImplementedError()

    @classmethod
    def from_labels(cls: tp.Type[IH],
            labels: tp.Iterable[tp.Sequence[tp.Hashable]],
            *,
            name: NameType = None,
            reorder_for_hierarchy: bool = False,
            index_constructors: tp.Optional[IndexConstructors] = None,
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
            :obj:`static_frame.IndexHierarchy2`
        '''
        # NOTE: This does nothing to enforce the sortedness of the labels!
        labels = iter(labels)
        label_row = next(labels)
        depth = len(label_row)

        if index_constructors is None:
            constructor_iter = (cls._INDEX_CONSTRUCTOR for _ in range(depth))
        elif callable(index_constructors):
            constructor_iter = (index_constructors for _ in range(depth))
        else:
            constructor_iter = tuple(index_constructors)
            if len(constructor_iter) != depth:
                raise ValueError("index_constructors must be the same length as the number of levels in the hierarchy.")

        hash_maps = [{} for _ in range(depth)]
        indexers = [[] for _ in range(depth)]

        prev_row = None

        while True:
            for hash_map, indexer, val in zip(hash_maps, indexers, label_row):
                if val is continuation_token:
                    if prev_row is None:
                        raise RuntimeError("continuation_token used without previous row.")
                    else:
                        i = indexer[-1]
                        val = prev_row[i]
                elif val not in hash_map:
                    i = len(hash_map)
                    hash_map[val] = len(hash_map)
                else:
                    i = hash_map[val]

                indexer.append(i)

            prev_row = label_row
            try:
                label_row = next(labels)
            except StopIteration:
                break

            if len(label_row) != depth:
                raise ErrorInitIndex("All labels must have the same depth.")

        return cls(
            indices=[constructor(hash_map) for constructor, hash_map in zip(constructor_iter, hash_maps)],
            indexers=list(map(np.array, indexers)),
            name=name,
        )

    @classmethod
    def from_index_items(cls: tp.Type[IH],
            items: tp.Iterable[tp.Tuple[tp.Hashable, Index]],
            *,
            index_constructor: tp.Optional[IndexConstructor] = None,
            name: NameType = None,
            ) -> IH:
        '''
        Given an iterable of pairs of label, :obj:`Index`, produce an :obj:`IndexHierarchy2` where the labels are depth 0, the indices are depth 1.

        Args:
            items: iterable of pairs of label, :obj:`Index`.
            index_constructor: Optionally provide index constructor for outermost index.
        '''
        raise NotImplementedError()

    @classmethod
    def from_labels_delimited(cls: tp.Type[IH],
            labels: tp.Iterable[str],
            *,
            delimiter: str = ' ',
            name: NameType = None,
            index_constructors: tp.Optional[IndexConstructors] = None,
            ) -> IH:
        '''
        Construct an :obj:`IndexHierarchy2` from an iterable of labels, where each label is string defining the component labels for all hierarchies using a string delimiter. All components after splitting the string by the delimited will be literal evaled to produce proper types; thus, strings must be quoted.

        Args:
            labels: an iterator or generator of tuples.

        Returns:
            :obj:`static_frame.IndexHierarchy2`
        '''
        raise NotImplementedError()

    @classmethod
    def from_names(cls: tp.Type[IH],
            names: tp.Iterable[tp.Hashable]
            ) -> IH:
        '''
        Construct a zero-length :obj:`IndexHierarchy2` from an iterable of ``names``, where the length of ``names`` defines the zero-length depth.

        Args:
            names: Iterable of hashable names per depth.
        '''
        name = tuple(names)
        return cls(
            indices=[cls._INDEX_CONSTRUCTOR((), name=name) for name in names],
            indexers=[np.array([], dtype=int) for _ in names],
            name=name,
        )

    @classmethod
    def _from_type_blocks(cls: tp.Type[IH],
            blocks: TypeBlocks,
            *,
            name: NameType = None,
            index_constructors: tp.Optional[IndexConstructors] = None,
            own_blocks: bool = False,
            ) -> IH:
        '''
        Construct an :obj:`IndexHierarchy2` from a :obj:`TypeBlocks` instance.

        Args:
            blocks: a TypeBlocks instance

        Returns:
            :obj:`IndexHierarchy2`
        '''
        indices: tp.List[Index] = []
        indexers: tp.List[np.ndarray] = []

        if index_constructors is None:
            constructor_iter = (cls._INDEX_CONSTRUCTOR for _ in range(blocks.shape[1]))
        elif callable(index_constructors):
            constructor_iter = (index_constructors for _ in range(blocks.shape[1]))
        else:
            constructor_iter = index_constructors

        for block, constructor in itertools.zip_longest(blocks, constructor_iter):

            if block is None or constructor is None:
                raise ErrorInitIndex(f'Levels and index_constructors must be the same length.')

            unique_values, indexer = np.unique(block.values.ravel(), return_inverse=True)

            # we call the constructor on all lvl, even if it is already an Index
            indices.append(constructor(unique_values))
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
            _blocks=blocks,
            _own_blocks=own_blocks,
        )

    # NOTE: could have a _from_fields (or similar) that takes a sequence of column iterables/arrays

    #---------------------------------------------------------------------------
    def __init__(self,
            indices: ArrayGO,
            indexers: tp.List[np.ndarray],
            *,
            name: NameType = NAME_DEFAULT,
            _blocks: tp.Optional[TypeBlocks] = None,
            _own_blocks: bool = False,
            ):
        '''
        Initializer.

        Args:
            indices: list of :obj:`Index` objects
            indexers: list of indexer arrays
            name: name of the IndexHierarchy2
        '''
        self._indices = indices
        self._indexers = indexers
        self._name = None if name is NAME_DEFAULT else name_filter(name)

        if _blocks is not None and not _own_blocks:
            self.__blocks = _blocks.copy()
        else:
            self.__blocks = _blocks

    #---------------------------------------------------------------------------
    def __deepcopy__(self: IH, memo: tp.Dict[int, tp.Any]) -> IH:
        obj = self.__new__(self.__class__)
        obj._indices = deepcopy(self._indices, memo)
        obj._indexers = deepcopy(self._indexers, memo)
        obj.__blocks = deepcopy(self.__blocks, memo)
        obj._name = self._name # should be hashable/immutable

        memo[id(self)] = obj
        return obj #type: ignore

    def __copy__(self: IH) -> IH:
        '''
        Return a shallow copy of this IndexHierarchy2.
        '''
        blocks = self._blocks.copy()
        return self.__class__(
                indices=self._indices,
                indexers=self._indexers,
                name=self._name,
                _blocks=blocks,
                _own_blocks=True
                )

    def copy(self: IH) -> IH:
        '''
        Return a shallow copy of this IndexHierarchy2.
        '''
        return self.__copy__()

    #---------------------------------------------------------------------------
    # name interface

    def rename(self: IH, name: NameType) -> IH:
        '''
        Return a new IndexHierarchy2 with an updated name attribute.
        '''
        raise NotImplementedError()

    #---------------------------------------------------------------------------
    # interfaces

    @property
    def loc(self) -> InterfaceGetItem[IH]:
        return InterfaceGetItem(self._extract_loc) #type: ignore

    @property
    def iloc(self) -> InterfaceGetItem[IH]:
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
        Retype one or more depths. Can be used as as function to retype the entire ``IndexHierarchy2``; alternatively, a ``__getitem__`` interface permits retyping selected depths.

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
    def via_T(self) -> InterfaceTranspose[IH]:
        '''
        Interface for using binary operators with one-dimensional sequences, where the opperand is applied column-wise.
        '''
        return InterfaceTranspose(
                container=self,
                )

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
            ) -> 'IndexHierarchy2':
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
            raise ErrorInitIndex('operand in IndexHierarchy2 set operations must ndim of 2')
        if both_sized and self.shape[1] != operand.shape[1]:
            raise ErrorInitIndex('operands in IndexHierarchy2 set operations must have matching depth')

        cls = self.__class__

        # using assume_unique will permit retaining order when operands are identical
        labels = func(self.values, operand, assume_unique=assume_unique) # type: ignore

        # derive index_constructors for IndexHierarchy2
        index_constructors: tp.Optional[tp.Sequence[tp.Type[IndexBase]]]

        if both_sized and isinstance(other, IndexHierarchy2):
            index_constructors = []
            # depth, and length of index_types, must be equal
            for cls_self, cls_other in zip(
                    self._index_constructors,
                    other._levels.index_types()):
                if cls_self == cls_other:
                    index_constructors.append(cls_self)
                else:
                    index_constructors.append(Index)
        else:
            # if other is not an IndexHierarchy2, do not try to propagate types
            index_constructors = None

        return cls.from_labels(labels,
                index_constructors=index_constructors,
                depth_reference=self.depth)

    #---------------------------------------------------------------------------

    @property
    def _index_constructors(self) -> tp.Tuple[tp.Type[Index]]:
        return tuple(index.__class__ for index in self._indices)

    def _drop_iloc(self, key: GetItemKeyType) -> 'IndexHierarchy2':
        '''Create a new index after removing the values specified by the loc key.
        '''
        blocks = TypeBlocks.from_blocks(self._blocks._drop_blocks(row_key=key))

        return self.__class__._from_type_blocks(blocks,
                index_constructors=self._index_constructors,
                name=self._name,
                own_blocks=True
                )

    def _drop_loc(self, key: GetItemKeyType) -> 'IndexHierarchy2':
        '''Create a new index after removing the values specified by the loc key.
        '''
        return self._drop_iloc(self._loc_to_iloc(key))

    #---------------------------------------------------------------------------

    @property
    def _blocks(self) -> TypeBlocks:
        if self.__blocks is None:
            def gen_blocks() -> tp.Iterator[np.ndarray]:
                for i, index in enumerate(self._indices):
                    indexer = self._indexers[i]
                    yield np.take(index.values, indexer)

            self.__blocks = TypeBlocks.from_blocks(gen_blocks())

        return self.__blocks

    @property #type: ignore
    @doc_inject(selector='values_2d', class_name='IndexHierarchy2')
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

    @doc_inject()
    def label_widths_at_depth(self,
            depth_level: DepthLevelSpecifier = 0
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, int]]:
        '''{}'''
        pos: tp.Optional[int] = None
        if not isinstance(depth_level, INT_TYPES):
            sel = list(depth_level)
            if len(sel) == 1:
                pos = sel.pop()
        else: # is an int
            pos = depth_level

        if pos is not None:  # i.e. depth_level is an int

            unique, widths = np.unique(self._indexers[pos], return_counts=True)
            labels = np.take(self._indices[pos], unique)

            result = np.empty(len(unique), dtype=object)
            result[:] = list(zip(labels, widths))
            result.flags.writeable = False
            return result

        raise NotImplementedError("selecting multiple depth levels is not yet implemented")

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
    def relabel(self, mapper: RelabelInput) -> 'IndexHierarchy2':
        '''
        Return a new IndexHierarchy2 with labels replaced by the callable or mapping; order will be retained. If a mapping is used, the mapping should map tuple representation of labels, and need not map all origin keys.
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
            ) -> "IndexHierarchy2":
        '''
        Return a new :obj:`IndexHierarchy2` after applying `mapper` to a level or each individual level specified by `depth_level`.

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
        raise NotImplementedError()

    def rehierarch(self: IH,
            depth_map: tp.Sequence[int]
            ) -> IH:
        '''
        Return a new :obj:`IndexHierarchy2` that conforms to the new depth assignments given be `depth_map`.
        '''
        index, _ = rehierarch_from_type_blocks(
                labels=self._blocks,
                index_cls=self.__class__,
                index_constructors=self._index_constructors,
                depth_map=depth_map,
                )
        return index #type: ignore

    #---------------------------------------------------------------------------

    def _loc_to_iloc(self,
            key: tp.Union[GetItemKeyType, HLoc]
            ) -> GetItemKeyType:
        '''
        Given iterable of GetItemKeyTypes, apply to each level of levels.
        '''
        if isinstance(key, ILoc):
            return key.key

        if isinstance(key, IndexHierarchy2):
            # default iteration of IH is as tuple
            raise NotImplementedError()
            #return [self._levels.leaf_loc_to_iloc(k) for k in key]

        if isinstance(key, np.ndarray) and key.dtype == DTYPE_BOOL:
            return self.positions[key]

        if isinstance(key, HLoc):
            # unpack any Series, Index, or ILoc into the context of this IndexHierarchy
            key = HLoc(tuple(
                    key_from_container_key(self, k, expand_iloc=True)
                    for k in key))
        else:
            key = key_from_container_key(self, key)

        raise NotImplementedError()
        #return self._levels.loc_to_iloc(key)

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
            ) -> tp.Union['IndexHierarchy2', tp.Tuple[tp.Hashable]]:
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
            ) -> tp.Union['IndexHierarchy2', tp.Tuple[tp.Hashable]]:
        return self._extract_iloc(self._loc_to_iloc(key))

    def __getitem__(self, #pylint: disable=E0102
            key: GetItemKeyType
            ) -> tp.Union['IndexHierarchy2', tp.Tuple[tp.Hashable]]:
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
        elif isinstance(other, IndexHierarchy2):
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

    def __contains__(self, #type: ignore
            value: tp.Tuple[tp.Hashable]
            ) -> bool:
        '''Determine if a leaf loc is contained in this Index.
        '''
        raise NotImplementedError()

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

        return np.unique(array2d_to_array1d(self.values_at_depth(sel)))

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

        if not isinstance(other, IndexHierarchy2):
            return False

        # same type from here
        if self.shape != other.shape:
            return False

        if compare_name and self.name != other.name:
            return False

        raise NotImplementedError()

    @doc_inject(selector='sort')
    def sort(self: IH,
            *,
            ascending: BoolOrBools = True,
            kind: str = DEFAULT_SORT_KIND,
            key: tp.Optional[tp.Callable[['IndexHierarchy2'], tp.Union[np.ndarray, 'IndexHierarchy2']]] = None,
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

    def roll(self, shift: int) -> 'IndexHierarchy2':
        '''Return an :obj:`IndexHierarchy2` with values rotated forward and wrapped around (with a positive shift) or backward and wrapped around (with a negative shift).
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
    def fillna(self, value: tp.Any) -> 'IndexHierarchy2':
        '''Return an :obj:`IndexHierarchy2` after replacing NA (NaN or None) with the supplied value.

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
            ) -> tp.Tuple['IndexHierarchy2', np.ndarray]:

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
        Return a :obj:`FrameGO` version of this :obj:`IndexHierarchy2`.
        '''
        from static_frame import FrameGO
        return self._to_frame(FrameGO) #type: ignore

    def to_pandas(self) -> 'DataFrame':
        '''Return a Pandas MultiIndex.
        '''
        import pandas

        # must copy to get a mutable array
        arrays = tuple(a.copy() for a in self._blocks.axis_values(axis=0))
        mi = pandas.MultiIndex.from_arrays(arrays)

        mi.name = self._name
        mi.names = self.names
        return mi

    def to_tree(self) -> TreeNodeT:
        '''Returns the tree representation of an IndexHierarchy2
        '''
        raise NotImplementedError()

    def flat(self) -> IndexBase:
        '''Return a flat, one-dimensional index of tuples for each level.
        '''
        return self._INDEX_CONSTRUCTOR(self.__iter__(), name=self._name)

    def level_add(self: IH,
            level: tp.Hashable,
            *,
            index_constructor: IndexConstructor = None,
            ) -> IH:
        '''Return an IndexHierarchy2 with a new root (outer) level added.
        '''
        raise NotImplementedError()

    def level_drop(self,
            count: int = 1,
            ) -> tp.Union[Index, 'IndexHierarchy2']:
        '''Return an IndexHierarchy2 with one or more leaf levels removed. This might change the size of the resulting index if the resulting levels are not unique.

        Args:
            count: A positive value is the number of depths to remove from the root (outer) side of the hierarchy; a negative value is the number of depths to remove from the leaf (inner) side of the hierarchy.
        '''
        raise NotImplementedError()


class IndexHierarchy2GO(IndexHierarchy2):
    '''
    A hierarchy of :obj:`static_frame.Index` objects that permits mutation only in the addition of new hierarchies or labels.
    '''

    STATIC = False

    _IMMUTABLE_CONSTRUCTOR = IndexHierarchy2
    _LEVEL_CONSTRUCTOR = IndexLevelGO
    _INDEX_CONSTRUCTOR = IndexGO

    __slots__ = (
            '_levels', # IndexLevel
            '_blocks',
            '_recache',
            '_name'
            )

    _levels: IndexLevelGO

    def append(self, value: tp.Sequence[tp.Hashable]) -> None:
        '''
        Append a single label to this index.
        '''
        raise NotImplementedError()

    def extend(self, other: IndexHierarchy2) -> None:
        '''
        Extend this IndexHiearchy in-place
        '''
        raise NotImplementedError()

    def __copy__(self: IH) -> IH:
        '''
        Return a shallow copy of this IndexHierarchy2.
        '''
        raise NotImplementedError()

# update class attr on Index after class initialization
IndexHierarchy2._MUTABLE_CONSTRUCTOR = IndexHierarchy2GO


class IndexHierarchyAsType:

    __slots__ = ('container', 'key',)

    def __init__(self,
            container: IndexHierarchy2,
            key: GetItemKeyType
            ) -> None:
        self.container = container
        self.key = key

    def __call__(self, dtype: DtypeSpecifier) -> IndexHierarchy2:

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
