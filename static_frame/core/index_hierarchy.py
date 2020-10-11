import typing as tp
from itertools import chain
from ast import literal_eval

import numpy as np


from static_frame.core.array_go import ArrayGO
from static_frame.core.container_util import apply_binary_operator
from static_frame.core.container_util import index_from_optional_constructor
from static_frame.core.container_util import matmul
from static_frame.core.container_util import rehierarch_from_type_blocks
from static_frame.core.container_util import key_from_container_key

from static_frame.core.display import Display
from static_frame.core.display import DisplayActive
from static_frame.core.display_config import DisplayConfig
from static_frame.core.display import DisplayHeader
from static_frame.core.doc_str import doc_inject

from static_frame.core.exception import ErrorInitIndex
from static_frame.core.hloc import HLoc
from static_frame.core.index import ILoc

from static_frame.core.index import Index
from static_frame.core.index import IndexGO
from static_frame.core.index import PositionsAllocator
from static_frame.core.index import mutable_immutable_index_filter
from static_frame.core.index_base import IndexBase
from static_frame.core.index_level import IndexLevel
from static_frame.core.index_level import IndexLevelGO
from static_frame.core.index_auto import RelabelInput

from static_frame.core.node_dt import InterfaceDatetime
from static_frame.core.node_iter import IterNodeApplyType
from static_frame.core.node_iter import IterNodeDepthLevel
from static_frame.core.node_iter import IterNodeType
from static_frame.core.node_selector import InterfaceAsType
from static_frame.core.node_selector import InterfaceGetItem
from static_frame.core.node_selector import TContainer
from static_frame.core.node_str import InterfaceString

from static_frame.core.type_blocks import TypeBlocks

from static_frame.core.util import DEFAULT_SORT_KIND
from static_frame.core.util import DepthLevelSpecifier
from static_frame.core.util import DtypeSpecifier
from static_frame.core.util import EMPTY_TUPLE

from static_frame.core.util import GetItemKeyType
from static_frame.core.util import IndexConstructor
from static_frame.core.util import IndexConstructors
from static_frame.core.util import IndexInitializer
from static_frame.core.util import INT_TYPES
from static_frame.core.util import intersect2d
from static_frame.core.util import isin
from static_frame.core.util import NAME_DEFAULT
from static_frame.core.util import name_filter
from static_frame.core.util import NameType
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import setdiff2d
from static_frame.core.util import UFunc
from static_frame.core.util import union2d
from static_frame.core.util import array2d_to_tuples
from static_frame.core.util import iterable_to_array_2d

if tp.TYPE_CHECKING:
    from pandas import DataFrame #pylint: disable=W0611 #pragma: no cover
    from static_frame.core.frame import Frame #pylint: disable=W0611 #pragma: no cover
    from static_frame.core.frame import FrameGO #pylint: disable=W0611 #pragma: no cover
    from static_frame.core.series import Series #pylint: disable=W0611 #pragma: no cover

IH = tp.TypeVar('IH', bound='IndexHierarchy')

CONTINUATION_TOKEN_INACTIVE = object()

#-------------------------------------------------------------------------------
class IndexHierarchy(IndexBase):
    '''A hierarchy of :obj:`Index` objects, defined as a strict tree of uniform depth across all branches.'''

    __slots__ = (
            '_levels',
            '_blocks',
            '_recache',
            '_name',
            )
    _levels: IndexLevel
    _blocks: TypeBlocks # should be tp.Optional[TypeBlocks] but many typing changes required
    _recache: bool
    _name: NameType

    # Temporary type overrides, until indices are generic.
    # __getitem__: tp.Callable[['IndexHierarchy', tp.Hashable], tp.Tuple[tp.Hashable, ...]]

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
            name: NameType = None
            ) -> IH:
        '''
        Given groups of iterables, return an ``IndexHierarchy`` made of the product of a values in those groups, where the first group is the top-most hierarchy.

        Returns:
            :obj:`static_frame.IndexHierarchy`

        '''
        indices = [] # store in a list, where index is depth
        for lvl in levels:
            if not isinstance(lvl, Index): # Index, not IndexBase
                lvl = cls._INDEX_CONSTRUCTOR(lvl)
            indices.append(lvl)

        if len(indices) == 1:
            raise RuntimeError('only one level given')

        # build name from index names, assuming they are all specified
        if name is None:
            name = tuple(index.name for index in indices)
            if any(n is None for n in name):
                name = None

        targets_previous = None

        # need to walk up from bottom to top
        # get depth pairs and iterate over those
        depth = len(indices) - 1
        while depth > 0:
            index = indices[depth]
            index_up = indices[depth - 1]
            # for each label in the next-up index, we need a reference to this index with an offset of that index (or level)
            targets = np.empty(len(index_up), dtype=object)

            offset = 0
            for idx, _ in enumerate(index_up):
                # this level does not have targets, only an index (as a leaf)
                level = cls._LEVEL_CONSTRUCTOR(index=index,
                        offset=offset,
                        targets=targets_previous)

                targets[idx] = level
                offset += len(level)
            targets_previous = ArrayGO(targets, own_iterable=True)
            depth -= 1

        level = cls._LEVEL_CONSTRUCTOR(index=index_up, targets=targets_previous)
        return cls(level, name=name)

    @classmethod
    def from_tree(cls: tp.Type[IH],
            tree: tp.Any,
            *,
            name: NameType = None
            ) -> IH:
        '''
        Convert into a ``IndexHierarchy`` a dictionary defining keys to either iterables or nested dictionaries of the same.

        Returns:
            :obj:`static_frame.IndexHierarchy`
        '''
        return cls(cls._LEVEL_CONSTRUCTOR.from_tree(tree), name=name)

    @classmethod
    def from_labels(cls: tp.Type[IH],
            labels: tp.Iterable[tp.Sequence[tp.Hashable]],
            *,
            name: NameType = None,
            reorder_for_hierarchy: bool = False,
            index_constructors: tp.Optional[IndexConstructors] = None,
            depth_reference: tp.Optional[int] = None,
            continuation_token: tp.Union[tp.Hashable, None] = CONTINUATION_TOKEN_INACTIVE
            ) -> IH:
        '''
        Construct an ``IndexHierarhcy`` from an iterable of labels, where each label is tuple defining the component labels for all hierarchies.

        Args:
            labels: an iterator or generator of tuples.
            reorder_for_hierarchy: reorder the labels to produce a hierarchable Index, assuming hierarchability is possible.
            continuation_token: a Hashable that will be used as a token to identify when a value in a label should use the previously encountered value at the same depth.

        Returns:
            :obj:`static_frame.IndexHierarchy`
        '''
        if reorder_for_hierarchy:
            if continuation_token != CONTINUATION_TOKEN_INACTIVE:
                raise RuntimeError('continuation_token not supported when reorder_for_hiearchy')
            # use from_records to ensure approprate columnar types
            from static_frame import Frame
            index_labels = Frame.from_records(labels)._blocks
            # this will reorder and create the index using this smae method, passed as cls.from_labels
            index, _ = rehierarch_from_type_blocks(
                    labels=index_labels,
                    depth_map=range(index_labels.shape[1]), # keep order
                    index_cls=cls,
                    index_constructors=index_constructors,
                    name=name,
                    )
            return index #type: ignore

        labels_iter = iter(labels)
        try:
            first = next(labels_iter)
            labels_empty = False
        except StopIteration:
            labels_empty = True

        if labels_empty:
            # if iterable is empty, must discover depth
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

            levels = cls._LEVEL_CONSTRUCTOR(
                    cls._INDEX_CONSTRUCTOR(EMPTY_TUPLE),
                    depth_reference=depth_reference)
            return cls(levels=levels, name=name)

        depth = len(first)
        # minimum permitted depth is 2
        if depth < 2:
            raise ErrorInitIndex('Cannot create an IndexHierarchy from only one level.')
        if index_constructors is not None and len(index_constructors) != depth:
            raise ErrorInitIndex('If providing index constructors, number of index constructors must equal depth of IndexHierarchy.')

        depth_max = depth - 1
        depth_pre_max = depth - 2

        token = object()
        observed_last = [token for _ in range(depth)]

        tree: tp.Any = dict() # order assumed and necessary
        # put first back in front
        for label in chain((first,), labels_iter):
            if len(label) != depth:
                raise ErrorInitIndex(f'Inconsistent label depth: expected {depth}, got {len(label)}')

            current = tree # NOTE: over the life of this loop, current can be a dict or a list
            # each label is an iterable
            for d, v in enumerate(label):
                if continuation_token is not CONTINUATION_TOKEN_INACTIVE:
                    if v == continuation_token:
                        # might check that observed_last[d] != token
                        v = observed_last[d]

                # shared implementation with from_labels -----------------------
                if d < depth_pre_max:
                    if v not in current:
                        current[v] = dict() # order necessary
                    else: # can only fetch this node (and not create a new node) if this is the sequential predecessor
                        if v != observed_last[d]:
                            raise ErrorInitIndex(f'invalid tree-form for IndexHierarchy: {v} in {label} cannot follow {observed_last[d]} when {v} has already been defined.')
                    current = current[v]
                    observed_last[d] = v
                elif d < depth_max:
                    if v not in current:
                        current[v] = list()
                    else: # cannot just fetch this list if it is not the predecessor
                        if v != observed_last[d]:
                            raise ErrorInitIndex(f'invalid tree-form for IndexHierarchy: {v} in {label} cannot follow {observed_last[d]} when {v} has already been defined.')
                    current = current[v]
                    observed_last[d] = v
                elif d == depth_max:
                    # if there are redundancies here they will be caught in index creation
                    current.append(v)
                else: # NOTE: cannot get here with length check above
                    raise ErrorInitIndex('label exceeded expected depth', label) #pragma: no cover
                # shared implementation with _from_type_blocks -----------------

        levels = cls._LEVEL_CONSTRUCTOR.from_tree(
                tree,
                index_constructors=index_constructors
                )
        return cls(levels=levels, name=name)

    @classmethod
    def from_index_items(cls: tp.Type[IH],
            items: tp.Iterable[tp.Tuple[tp.Hashable, Index]],
            *,
            index_constructor: tp.Optional[IndexConstructor] = None,
            ) -> IH:
        '''
        Given an iterable of pairs of label, :obj:`Index`, produce an :obj:`IndexHierarchy` where the labels are depth 0, the indices are depth 1.

        Args:
            items: iterable of pairs of label, :obj:`Index`.
            index_constructor: Optionally provide index constructor for outermost index.
        '''
        labels = []
        index_levels = []

        offset = 0
        for label, index in items:
            labels.append(label)
            index = mutable_immutable_index_filter(cls.STATIC, index) #type: ignore
            index_levels.append(cls._LEVEL_CONSTRUCTOR(
                    index,
                    offset=offset,
                    own_index=True)
            )
            offset += len(index)

        targets = ArrayGO(np.array(index_levels, dtype=object), own_iterable=True)

        index_outer = index_from_optional_constructor(labels,
                    default_constructor=cls._INDEX_CONSTRUCTOR,
                    explicit_constructor=index_constructor)
        levels = cls._LEVEL_CONSTRUCTOR(
                index=index_outer, #type: ignore
                targets=targets,
                own_index=True,
                depth_reference=2, # depth always 2
                )
        # import ipdb; ipdb.set_trace()
        return cls(levels)

    @classmethod
    def from_labels_delimited(cls: tp.Type[IH],
            labels: tp.Iterable[str],
            *,
            delimiter: str = ' ',
            name: NameType = None,
            index_constructors: tp.Optional[IndexConstructors] = None,
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

            return tuple(literal_eval(p) for p in parts)

        return cls.from_labels(
                (to_label(label) for label in labels),
                name=name,
                index_constructors=index_constructors
                )

    @classmethod
    def from_names(cls: tp.Type[IH],
            names: tp.Iterable[tp.Hashable]
            ) -> IH:
        '''
        Construct a zero-length :obj:`IndexHierarchy` from an iterable of ``names``, where the length of ``names`` defines the zero-length depth.

        Args:
            names: Iterable of hashable names per depth.
        '''
        name = tuple(names)
        depth = len(name)
        levels = cls._LEVEL_CONSTRUCTOR.from_depth(depth)
        return cls(levels, name=name)


    @classmethod
    def _from_type_blocks(cls: tp.Type[IH],
            blocks: TypeBlocks,
            *,
            name: NameType = None,
            index_constructors: tp.Optional[IndexConstructors] = None,
            own_blocks: bool = False,
            ) -> IH:
        '''
        Construct an :obj:`IndexHierarchy` from a :obj:`TypeBlocks` instance.

        Args:
            blocks: a TypeBlocks instance

        Returns:
            :obj:`IndexHierarchy`
        '''

        depth = blocks.shape[1]

        # minimum permitted depth is 2
        if depth < 2:
            raise ErrorInitIndex('cannot create an IndexHierarchy from only one level.')
        if index_constructors is not None and len(index_constructors) != depth:
            raise ErrorInitIndex('if providing index constructors, number of index constructors must equal depth of IndexHierarchy.')

        depth_max = depth - 1
        depth_pre_max = depth - 2

        token = object()
        observed_last = [token for _ in range(depth)]

        tree: tp.Any = dict() # order assumed and necessary

        idx_row_last = -1
        for (idx_row, d), v in blocks.element_items():
            if idx_row_last != idx_row:
                # for each row, we re-set current to the outermost reference
                current = tree
                idx_row_last = idx_row

            # shared implementation with from_labels ---------------------------
            if d < depth_pre_max:
                if v not in current:
                    current[v] = dict() # order necessary
                else: # can only fetch this node (and not create a new node) if this is the sequential predecessor
                    if v != observed_last[d]:
                        raise ErrorInitIndex(f'invalid tree-form for IndexHierarchy: {v} cannot follow {observed_last[d]} when {v} has already been defined.')
                current = current[v]
                observed_last[d] = v
            elif d < depth_max: # premax means inner values are a list
                if v not in current:
                    current[v] = list()
                else: # cannot just fetch this list if it is not the predecessor
                    if v != observed_last[d]:
                        raise ErrorInitIndex(f'invalid tree-form for IndexHierarchy: {v} cannot follow {observed_last[d]} when {v} has already been defined.')
                current = current[v]
                observed_last[d] = v
            elif d == depth_max:
                # if there are redundancies here they will be caught in index creation
                current.append(v)
            else:
                # cannot happen with TypeBlocks
                raise ErrorInitIndex('label exceeded expected depth', v) #pragma: no cover
            # shared implementation with from_labels ---------------------------

        levels = cls._LEVEL_CONSTRUCTOR.from_tree(
                tree,
                index_constructors=index_constructors,
                depth_reference=depth,
                )

        if index_constructors is not None:
            # If defined, we may have changed columnar dtypes in IndexLevels, and cannot reuse blocks
            if tuple(blocks.dtypes) != tuple(levels.dtype_per_depth()):
                blocks = None #type: ignore
                own_blocks = False

        return cls(levels=levels, name=name, blocks=blocks, own_blocks=own_blocks)


    #---------------------------------------------------------------------------
    def __init__(self,
            levels: tp.Union[IndexLevel, 'IndexHierarchy'],
            *,
            name: NameType = NAME_DEFAULT,
            blocks: tp.Optional[TypeBlocks] = None,
            own_blocks: bool = False,
            ):
        '''
        Initializer.

        Args:
            levels: :obj:`IndexLevels` instance, or, optionally, an :obj`IndexHierarchy` to be used to construct a new :obj`IndexHierarchy`.
        '''

        self._blocks = None #type: ignore

        if isinstance(levels, IndexHierarchy):
            if not blocks is None:
                raise ErrorInitIndex('cannot provide blocks when initializing with IndexHierarchy')
            index_level = levels._levels
            # if cache is updated, can get blocks
            if not levels._recache:
                self._blocks = levels._blocks.copy()
            # transfer name if not given as arg
            if name is NAME_DEFAULT:
                name = levels.name

        elif isinstance(levels, IndexLevel):
            index_level = levels
            if not blocks is None:
                self._blocks = blocks if own_blocks else blocks.copy()

        else:
            raise NotImplementedError(f'no handling for creation from {levels}')

        if self.STATIC and index_level.STATIC:
            self._levels = index_level
        else: # must deepcopy IndexLevels if not IndexHierarchy not static
            self._levels = index_level.to_index_level(
                    cls=self._LEVEL_CONSTRUCTOR
                    )

        if self._levels.depth <= 1:
            raise ErrorInitIndex(f'invalid depth ({self._levels.depth}) for IndexLevels composed in IndexHierarchy')

        self._recache = self._blocks is None
        self._name = None if name is NAME_DEFAULT else name_filter(name)

    #---------------------------------------------------------------------------
    # name interface

    def rename(self: IH, name: NameType) -> IH:
        '''
        Return a new Frame with an updated name attribute.
        '''
        # do not need to recache
        # let the constructor handle reuse
        return self.__class__(self, name=name)

    #---------------------------------------------------------------------------
    # interfaces

    @property
    def loc(self) -> InterfaceGetItem['IndexHierarchy']:
        return InterfaceGetItem(self._extract_loc) #type: ignore

    @property
    def iloc(self) -> InterfaceGetItem['IndexHierarchy']:
        return InterfaceGetItem(self._extract_iloc) #type: ignore


    def _iter_label(self,
            depth_level: DepthLevelSpecifier = 0,
            ) -> tp.Iterator[tp.Hashable]:

        # if no type blocks, use a levels
        if self._recache:
            if isinstance(depth_level, int):
                yield from self._levels.labels_at_depth(depth_level=depth_level)
            else:
                yield from zip(
                        *(self._levels.labels_at_depth(depth_level=d) for d in depth_level)
                        )
        else:
            if isinstance(depth_level, int):
                yield from self._blocks._extract_array(column_key=depth_level)
            else:
                yield from array2d_to_tuples(
                        self._blocks._extract_array(column_key=depth_level)
                        )

    def _iter_label_items(self,
            depth_level: DepthLevelSpecifier = 0,
            ) -> tp.Iterator[tp.Tuple[int, tp.Hashable]]:
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
        if self._recache:
            self._update_array_cache()

        def blocks_to_container(blocks: tp.Iterator[np.ndarray]) -> np.ndarray:
            return TypeBlocks.from_blocks(blocks).values

        return InterfaceString(
                blocks=self._blocks._blocks,
                blocks_to_container=blocks_to_container,
                )

    @property
    def via_dt(self) -> InterfaceDatetime[np.ndarray]:
        '''
        Interface for applying datetime properties and methods to elements in this container.
        '''
        if self._recache:
            self._update_array_cache()

        def blocks_to_container(blocks: tp.Iterator[np.ndarray]) -> np.ndarray:
            return TypeBlocks.from_blocks(blocks).values

        return InterfaceDatetime(
                blocks=self._blocks._blocks,
                blocks_to_container=blocks_to_container,
                )


    #---------------------------------------------------------------------------

    def _update_array_cache(self) -> None:
        self._blocks = self._levels.to_type_blocks()
        self._recache = False

    #---------------------------------------------------------------------------

    @property # type: ignore
    @doc_inject()
    def mloc(self) -> int:
        '''{doc_int}
        '''
        if self._recache:
            self._update_array_cache()

        return self._blocks.mloc #type: ignore

    @property
    def dtypes(self) -> 'Series':
        '''
        Return a Series of dytpes for each index depth.

        Returns:
            :obj:`static_frame.Series`
        '''
        from static_frame.core.series import Series

        if self._recache:
            # might use self._levels.dtype_per_depth
            self._update_array_cache()

        labels: NameType

        if self._name and isinstance(self._name, tuple) and len(self._name) == self.depth:
            labels = self._name
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
        if self._recache:
            return self._levels.__len__(), self._levels.depth
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
        if self._recache:
            return self._levels.__len__() * self._levels.depth
        return self._blocks.size

    @property
    def nbytes(self) -> int:
        '''
        Return the total bytes of the underlying NumPy array.

        Returns:
            :obj:`int`
        '''
        if self._recache:
            self._update_array_cache()
        return self._blocks.nbytes

    # def __bool__(self) -> bool:
    #     '''
    #     True if this container has size.
    #     '''
    #     if self._recache:
    #         return bool(self._levels.__len__()) and bool(self._levels.depth)
    #     return bool(self._blocks.size)

    #---------------------------------------------------------------------------

    def __len__(self) -> int:
        if self._recache:
            # avoid full recache
            return self._levels.__len__()
        return self._blocks.__len__()

    @doc_inject()
    def display(self,
            config: tp.Optional[DisplayConfig] = None
            ) -> Display:
        '''{doc}

        Args:
            {config}
        '''
        config = config or DisplayActive.get()

        if self._recache:
            self._update_array_cache()

        sub_display = None

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
                        header_depth=header_depth)
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

        if self._recache:
            self._update_array_cache()

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
            for cls_self, cls_other in zip(
                    self._levels.index_types(),
                    other._levels.index_types()):
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
    def _drop_iloc(self, key: GetItemKeyType) -> 'IndexHierarchy':
        '''Create a new index after removing the values specified by the loc key.
        '''
        if self._recache:
            self._update_array_cache()

        blocks = TypeBlocks.from_blocks(self._blocks._drop_blocks(row_key=key))
        index_constructors = tuple(self._levels.index_types())

        return self.__class__._from_type_blocks(blocks,
                index_constructors=index_constructors,
                name=self._name,
                own_blocks=True
                )

    def _drop_loc(self, key: GetItemKeyType) -> 'IndexHierarchy':
        '''Create a new index after removing the values specified by the loc key.
        '''
        return self._drop_iloc(self.loc_to_iloc(key))

    #---------------------------------------------------------------------------

    @property #type: ignore
    @doc_inject(selector='values_2d', class_name='IndexHierarchy')
    def values(self) -> np.ndarray:
        '''
        {}
        '''
        if self._recache:
            self._update_array_cache()
        return self._blocks.values

    @property
    def positions(self) -> np.ndarray:
        '''Return the immutable positions array.
        '''
        return PositionsAllocator.get(self.__len__())

    @property
    def depth(self) -> int: #type: ignore
        if self._recache:
            # avoid full recache to get depth
            return self._levels.depth
        return self._blocks.shape[1]

    def values_at_depth(self,
            depth_level: DepthLevelSpecifier = 0
            ) -> np.ndarray:
        '''
        Return an NP array for the ``depth_level`` specified.

        Args:
            depth_level: a single depth level, or iterable depth of depth levels.
        '''
        if self._recache:
            self._update_array_cache()

        sel: GetItemKeyType

        if isinstance(depth_level, int):
            sel = depth_level
        else:
            sel = list(depth_level)
        return self._blocks._extract_array(column_key=sel)

    @doc_inject()
    def label_widths_at_depth(self,
            depth_level: DepthLevelSpecifier = 0
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, int]]:
        '''{}'''
        if isinstance(depth_level, int):
            sel = depth_level
        else:
            raise NotImplementedError('selection from iterables is not implemented')
        yield from self._levels.label_widths_at_depth(depth_level=depth_level)

    @property
    def index_types(self) -> 'Series':
        '''
        Return a Series of Index classes for each index depth.

        Returns:
            :obj:`static_frame.Series`
        '''
        from static_frame.core.series import Series

        labels: NameType

        if self._name and isinstance(self._name, tuple) and len(self._name) == self.depth:
            labels = self._name
        else:
            labels = None

        # NOTE: consider caching index_types
        return Series(self._levels.index_types(), index=labels)

    #---------------------------------------------------------------------------

    def copy(self: IH) -> IH:
        '''
        Return a new IndexHierarchy. This is not a deep copy.
        '''
        if self._recache:
            self._update_array_cache()

        blocks = self._blocks.copy()
        return self.__class__(
                levels=self._levels,
                name=self._name,
                blocks=blocks,
                own_blocks=True
                )

    def relabel(self, mapper: RelabelInput) -> 'IndexHierarchy':
        '''
        Return a new IndexHierarchy with labels replaced by the callable or mapping; order will be retained. If a mapping is used, the mapping should map tuple representation of labels, and need not map all origin keys.
        '''
        if self._recache:
            self._update_array_cache()

        index_constructors = tuple(self._levels.index_types())

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
                    index_constructors=index_constructors,
                    )

        return self.__class__.from_labels(
                (mapper(x) for x in self._blocks.axis_values(axis=1)), #type: ignore
                name=self._name,
                index_constructors=index_constructors,
                )

    def rehierarch(self: IH,
            depth_map: tp.Sequence[int]
            ) -> IH:
        '''
        Return a new `IndexHierarchy` that conforms to the new depth assignments given be `depth_map`.
        '''
        if self._recache:
            self._update_array_cache()

        index_constructors = tuple(self._levels.index_types())

        index, _ = rehierarch_from_type_blocks(
                labels=self._blocks,
                index_cls=self.__class__,
                index_constructors=index_constructors,
                depth_map=depth_map,
                )
        return index #type: ignore

    #---------------------------------------------------------------------------

    def loc_to_iloc(self,
            key: tp.Union[GetItemKeyType, HLoc]
            ) -> GetItemKeyType:
        '''
        Given iterable of GetItemKeyTypes, apply to each level of levels.
        '''
        from static_frame.core.series import Series

        if isinstance(key, ILoc):
            return key.key
        elif isinstance(key, IndexHierarchy):
            # default iteration of IH is as tuple
            return [self._levels.leaf_loc_to_iloc(k) for k in key]

        key = key_from_container_key(self, key)

        if isinstance(key, HLoc):
            # unpack any Series, Index, or ILoc into the context of this IndexHierarchy
            key = HLoc(tuple(
                    key_from_container_key(self, k, expand_iloc=True)
                    for k in key))

        return self._levels.loc_to_iloc(key)

    def _extract_iloc(self,
            key: GetItemKeyType,
            ) -> tp.Union['IndexHierarchy', tp.Tuple[tp.Hashable]]:
        '''Extract a new index given an iloc key
        '''
        if self._recache:
            self._update_array_cache()

        if isinstance(key, INT_TYPES):
            # return a tuple if selecting a single row
            # NOTE: if extracting a single row, should be able to get it from IndexLevel without forcing a complete recache
            # NOTE: Selecting a single row may force type coercion before values are added to the tuple; i.e., a datetime64 will go to datetime.date before going to the tuple
            return tuple(self._blocks._extract_array(row_key=key)) #type: ignore

        index_constructors = tuple(self._levels.index_types())
        tb = self._blocks._extract(row_key=key)

        return self.__class__._from_type_blocks(tb,
                name=self._name,
                index_constructors=index_constructors,
                own_blocks=True,
                )

    def _extract_loc(self,
            key: GetItemKeyType
            ) -> tp.Union['IndexHierarchy', tp.Tuple[tp.Hashable]]:
        return self._extract_iloc(self.loc_to_iloc(key))

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
        if self._recache:
            self._update_array_cache()

        values = self._blocks.values
        array = operator(values)
        array.flags.writeable = False
        return array

    def _ufunc_binary_operator(self, *,
            operator: UFunc,
            other: tp.Any,
            ) -> np.ndarray:
        '''
        Binary operators applied to an index always return an NP array. This deviates from Pandas, where some operations (multipling an int index by an int) result in a new Index, while other operations result in a np.array (using == on two Index).
        '''
        if self._recache:
            self._update_array_cache()

        # NOTE: might use TypeBlocks._ufunc_binary_operator
        values = self._blocks.values

        other_is_array = False
        if isinstance(other, Index):
            # if this is a 1D index, must rotate labels before using an operator
            other = other.values.reshape((len(other), 1)) # operate on labels to labels
            other_is_array = True
        elif isinstance(other, IndexHierarchy):
            # already 2D
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
        Returns:
            immutable NumPy array.
        '''
        if self._recache:
            self._update_array_cache()

        dtype = None if not dtypes else dtypes[0]
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
        # NOTE: by iterating from levels, we avoid type casting to a row
        yield from self._levels.__iter__()

    def __reversed__(self) -> tp.Iterator[tp.Tuple[tp.Hashable, ...]]:
        '''
        Returns a reverse iterator on the index labels.
        '''
        if self._recache:
            self._update_array_cache()
        for array in self._blocks.axis_values(1, reverse=True):
            yield tuple(array)

    def __contains__(self, #type: ignore
            value: tp.Tuple[tp.Hashable]
            ) -> bool:
        '''Determine if a leaf loc is contained in this Index.
        '''
        # levels only, no need to recache as this is what has been mutated
        return self._levels.__contains__(value)
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
        # NOTE: do not need to udpate array cache, as can compare elemetns in levels
        if id(other) == id(self):
            return True

        if compare_class and self.__class__ != other.__class__:
            return False
        elif not isinstance(other, IndexHierarchy):
            return False

        # same type from here
        if self.shape != other.shape:
            return False
        if compare_name and self.name != other.name:
            return False

        return self._levels.equals(other._levels, #type: ignore
                compare_name=compare_name,
                compare_dtype=compare_dtype,
                compare_class=compare_class,
                skipna=skipna,
                )


    def sort(self: IH,
            ascending: bool = True,
            kind: str = DEFAULT_SORT_KIND
            ) -> IH:
        '''Return a new Index with the labels sorted.

        Args:
            kind: Sort algorithm passed to NumPy.
        '''
        if self._recache:
            self._update_array_cache()

        v = self._blocks.values
        order = np.lexsort([v[:, i] for i in range(v.shape[1]-1, -1, -1)])

        if not ascending:
            order = order[::-1]

        blocks = self._blocks._extract(row_key=order)
        index_constructors = tuple(self._levels.index_types())

        return self.__class__._from_type_blocks(blocks,
                index_constructors=index_constructors,
                name=self._name,
                own_blocks=True
                )

    def isin(self, other: tp.Iterable[tp.Iterable[tp.Hashable]]) -> np.ndarray:
        '''
        Return a Boolean array showing True where one or more of the passed in iterable of labels is found in the index.
        '''
        if self._recache:
            self._update_array_cache()

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
        if self._recache:
            self._update_array_cache()

        blocks = TypeBlocks.from_blocks(
                self._blocks._shift_blocks(row_shift=shift, wrap=True)
                )
        index_constructors = tuple(self._levels.index_types())

        return self.__class__._from_type_blocks(blocks,
                index_constructors=index_constructors,
                name=self._name,
                own_blocks=True
                )

    @doc_inject(selector='fillna')
    def fillna(self, value: tp.Any) -> 'IndexHierarchy':
        '''Return an :obj:`IndexHierarchy` after replacing null (NaN or None) with the supplied value.

        Args:
            {value}
        '''
        if self._recache:
            self._update_array_cache()

        blocks = self._blocks.fillna(value, None)
        index_constructors = tuple(self._levels.index_types())

        return self.__class__._from_type_blocks(blocks,
                index_constructors=index_constructors,
                name=self._name,
                own_blocks=True
                )


    #---------------------------------------------------------------------------
    # export

    def _to_frame(self,
            constructor: tp.Type['Frame']
            ) -> 'Frame':

        if self._recache:
            self._update_array_cache()

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

        if self._recache:
            self._update_array_cache()

        # must copy to get a mutable array
        arrays = tuple(a.copy() for a in self._blocks.axis_values(axis=0))
        mi = pandas.MultiIndex.from_arrays(arrays)

        mi.name = self._name
        mi.names = self.names
        return mi

    def flat(self) -> IndexBase:
        '''Return a flat, one-dimensional index of tuples for each level.
        '''
        return self._INDEX_CONSTRUCTOR(self.__iter__())

    def level_add(self: IH, level: tp.Hashable) -> IH:
        '''Return an IndexHierarchy with a new root (outer) level added.
        '''
        if self.STATIC: # can reuse levels
            levels_src = self._levels
        else:
            levels_src = self._levels.to_index_level()

        levels = self._LEVEL_CONSTRUCTOR(
                index=self._INDEX_CONSTRUCTOR((level,)),
                targets=ArrayGO([levels_src], own_iterable=True),
                offset=0,
                own_index=True
                )
        # can transfrom TypeBlocks appropriately and pass to constructor
        if not self._recache: # if we have TypeBlocks
            array = np.full(self.__len__(), level)
            array.flags.writeable = False
            blocks = TypeBlocks.from_blocks(chain((array,), self._blocks._blocks))
            return self.__class__(levels,
                    name=self._name,
                    blocks=blocks,
                    own_blocks=True)

        return self.__class__(levels, name=self._name)

    def level_drop(self, count: int = 1) -> tp.Union[Index, 'IndexHierarchy']:
        '''Return an IndexHierarchy with one or more leaf levels removed. This might change the size of the resulting index if the resulting levels are not unique.

        Args:
            count: A positive value is the number of depths to remove from the root (outer) side of the hierarhcy; a negative values is the number of depths to remove from the leaf (inner) side of the hierarchy.
        '''
        if count < 0: # remove from inner
            levels = self._levels.to_index_level()
            for _ in range(abs(count)):
                levels_stack = [levels]
                while levels_stack:
                    level = levels_stack.pop()
                    # check to see if children of this target are leaves
                    if level.targets[0].targets is None: #type: ignore
                        level.targets = None
                    else:
                        levels_stack.extend(level.targets) #type: ignore
                if levels.targets is None:  # if no targets, at the root
                    break
            if levels.targets is None: # fall back to 1D index
                return levels.index

            # if we have TypeBlocks and levels is the same length
            if not self._recache and levels.__len__() == self.__len__():
                blocks = self._blocks.iloc[NULL_SLICE, :count]
                return self.__class__(levels,
                        name=self._name,
                        blocks=blocks,
                        own_blocks=True
                        )
            return self.__class__(levels, name=self._name)

        elif count > 0: # remove from outer
            levels = self._levels.to_index_level()
            for _ in range(count):
                targets = []
                labels = []
                for target in levels.targets: #type: ignore
                    labels.extend(target.index)
                    if target.targets is not None:
                        targets.extend(target.targets)
                index = levels.index.__class__(labels)
                if not targets:
                    return index
                levels = levels.__class__(
                        index=index,
                        targets=ArrayGO(targets, own_iterable=True))

            # if we have TypeBlocks and levels is the same length
            if not self._recache and levels.__len__() == self.__len__():
                blocks = self._blocks.iloc[NULL_SLICE, count:]
                return self.__class__(levels,
                        name=self._name,
                        blocks=blocks,
                        own_blocks=True
                        )
            return self.__class__(levels, name=self._name)

        raise NotImplementedError('no handling for a 0 count drop level.')


class IndexHierarchyGO(IndexHierarchy):
    '''
    A hierarchy of :obj:`static_frame.Index` objects that permits mutation only in the addition of new hierarchies or labels.
    '''

    STATIC = False

    _IMMUTABLE_CONSTRUCTOR = IndexHierarchy
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
        self._levels.append(value)
        self._recache = True

    def extend(self, other: IndexHierarchy) -> None:
        '''
        Extend this IndexHiearchy in-place
        '''
        self._levels.extend(other._levels)
        self._recache = True

    def copy(self: IH) -> IH:
        '''
        Return a new IndexHierarchy. This is not a deep copy.
        '''
        if self._recache:
            self._update_array_cache()

        blocks = self._blocks.copy()
        return self.__class__(
                levels=self._levels.to_index_level(),
                name=self._name,
                blocks=blocks,
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

        from static_frame.core.index_datetime import _dtype_to_index_cls
        container = self.container

        if container._recache:
            container._update_array_cache()

        # use TypeBlocks in both situations to avoid double casting
        blocks = TypeBlocks.from_blocks(
                container._blocks._astype_blocks(column_key=self.key, dtype=dtype)
                )

        # avoid coercion of datetime64 arrays that were not targetted in the selection
        index_constructors = container.index_types.values.copy()

        dtype_post = blocks.dtypes[self.key] # can select element or array
        if isinstance(dtype_post, np.dtype):
            index_constructors[self.key] = _dtype_to_index_cls(
                    container.STATIC,
                    dtype_post)
        else: # assign iterable
            index_constructors[self.key] = [
                    _dtype_to_index_cls(container.STATIC, dt)
                    for dt in dtype_post]

        return container.__class__._from_type_blocks(
                blocks,
                index_constructors=index_constructors,
                own_blocks=True
                )

