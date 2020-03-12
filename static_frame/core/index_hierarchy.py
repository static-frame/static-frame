
import typing as tp
from collections.abc import KeysView
from itertools import chain
from ast import literal_eval


import numpy as np

from static_frame.core.util import DEFAULT_SORT_KIND

from static_frame.core.index_base import IndexBase
from static_frame.core.index import Index
from static_frame.core.index import IndexGO
from static_frame.core.index import _requires_reindex
from static_frame.core.index import mutable_immutable_index_filter

from static_frame.core.util import IndexConstructor
from static_frame.core.util import IndexConstructors
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import intersect2d
from static_frame.core.util import union2d
from static_frame.core.util import setdiff2d
from static_frame.core.util import array2d_to_tuples
from static_frame.core.util import name_filter
from static_frame.core.util import isin
from static_frame.core.util import iterable_to_array_2d


from static_frame.core.selector_node import InterfaceGetItem
from static_frame.core.selector_node import InterfaceAsType


from static_frame.core.util import KEY_ITERABLE_TYPES
from static_frame.core.util import CallableOrMapping
from static_frame.core.util import DepthLevelSpecifier
from static_frame.core.util import array_shift


from static_frame.core.container_util import matmul
from static_frame.core.container_util import index_from_optional_constructor
from static_frame.core.container_util import rehierarch_and_map

from static_frame.core.array_go import ArrayGO

from static_frame.core.display import DisplayConfig
from static_frame.core.display import DisplayActive
from static_frame.core.display import Display
from static_frame.core.display import DisplayHeader

from static_frame.core.iter_node import IterNodeType
from static_frame.core.iter_node import IterNodeDepthLevel
from static_frame.core.iter_node import IterNodeApplyType

from static_frame.core.hloc import HLoc

from static_frame.core.index_level import IndexLevel

from static_frame.core.index_level import IndexLevelGO
from static_frame.core.exception import ErrorInitIndex
from static_frame.core.doc_str import doc_inject

if tp.TYPE_CHECKING:

    from pandas import DataFrame #pylint: disable=W0611 #pragma: no cover
    from static_frame.core.frame import Frame #pylint: disable=W0611 #pragma: no cover


IH = tp.TypeVar('IH', bound='IndexHierarchy')

CONTINUATION_TOKEN_INACTIVE = object()

#-------------------------------------------------------------------------------
class IndexHierarchy(IndexBase):
    '''
    A hierarchy of :obj:`static_frame.Index` objects, defined as strict tree of uniform depth across all branches.
    '''
    __slots__ = (
            '_levels',
            '_labels',
            '_depth',
            '_recache',
            '_name'
            )
    _levels: IndexLevel
    _lables: np.ndarray
    _depth: int
    _keys: KeysView
    _recache: bool
    _name: tp.Hashable

    # Temporary type overrides, until indices are generic.
    __getitem__: tp.Callable[['IndexHierarchy', tp.Hashable], tp.Tuple[tp.Hashable, ...]]
    # __iter__: tp.Callable[['IndexHierarchy'], tp.Iterator[tp.Tuple[tp.Hashable, ...]]]
    # __reversed__: tp.Callable[['IndexHierarchy'], tp.Iterator[tp.Tuple[tp.Hashable, ...]]]

    # _IMMUTABLE_CONSTRUCTOR is None from IndexBase
    # _MUTABLE_CONSTRUCTOR will be defined after IndexHierarhcyGO defined

    _INDEX_CONSTRUCTOR = Index
    _LEVEL_CONSTRUCTOR = IndexLevel

    _UFUNC_UNION = union2d
    _UFUNC_INTERSECTION = intersect2d
    _UFUNC_DIFFERENCE = setdiff2d

    #---------------------------------------------------------------------------
    # constructors

    @classmethod
    def from_product(cls: tp.Type[IH],
            *levels,
            name: tp.Hashable = None
            ) -> IH: # tp.Iterable[tp.Hashable]
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
    def _tree_to_index_level(cls,
            tree,
            index_constructors: tp.Optional[IndexConstructors] = None
            ) -> IndexLevel:
        # tree: tp.Dict[tp.Hashable, tp.Union[Sequence[tp.Hashable], tp.Dict]]

        def get_index(labels, depth: int):
            if index_constructors:
                explicit_constructor = index_constructors[depth]
            else:
                explicit_constructor = None

            return index_from_optional_constructor(labels,
                    default_constructor=cls._INDEX_CONSTRUCTOR,
                    explicit_constructor=explicit_constructor)

        def get_level(level_data, offset=0, depth=0):

            if isinstance(level_data, dict):
                level_labels = []
                targets = np.empty(len(level_data), dtype=object)
                offset_local = 0

                # ordered key, value pairs, where the key is the label, the value is a list or dictionary; enmerate for insertion pre-allocated object array
                for idx, (k, v) in enumerate(level_data.items()):
                    level_labels.append(k)
                    level = get_level(v, offset=offset_local, depth=depth + 1)
                    targets[idx] = level
                    offset_local += len(level) # for lower level offsetting

                index = get_index(level_labels, depth=depth)
                targets = ArrayGO(targets, own_iterable=True)

            else: # an iterable, terminal node, no offsets needed
                index = get_index(level_data, depth=depth)
                targets = None

            return cls._LEVEL_CONSTRUCTOR(
                    index=index,
                    offset=offset,
                    targets=targets,
                    )

        return get_level(tree)


    @classmethod
    def from_tree(cls: tp.Type[IH],
            tree,
            *,
            name: tp.Hashable = None
            ) -> IH:
        '''
        Convert into a ``IndexHierarchy`` a dictionary defining keys to either iterables or nested dictionaries of the same.

        Returns:
            :obj:`static_frame.IndexHierarchy`
        '''
        return cls(cls._tree_to_index_level(tree), name=name)


    @classmethod
    def from_labels(cls: tp.Type[IH],
            labels: tp.Iterable[tp.Sequence[tp.Hashable]],
            *,
            name: tp.Hashable = None,
            reorder_for_hierarchy: bool = False,
            index_constructors: tp.Optional[IndexConstructors] = None,
            continuation_token: tp.Union[tp.Hashable, None] = CONTINUATION_TOKEN_INACTIVE
            ) -> IH:
        '''
        Construct an ``IndexHierarhcy`` from an iterable of labels, where each label is tuple defining the component labels for all hierarchies.

        Args:
            labels: an iterator or generator of tuples.
            reorder_for_hierarchy: reorder the labels to produce a hierarchible Index, assuming hierarchability is possible.
            continuation_token: a Hashable that will be used as a token to identify when a value in a label should use the previously encountered value at the same depth.

        Returns:
            :obj:`static_frame.IndexHierarchy`
        '''
        if reorder_for_hierarchy:
            if continuation_token != CONTINUATION_TOKEN_INACTIVE:
                raise RuntimeError('continuation_token not supported when reorder_for_hiearchy')
            # we need a single numpy array to use rehierarch_and_map
            index_labels = iterable_to_array_2d(labels)
            # this will reorder and create the index using this smae method, passed as cls.from_labels
            index, _ = rehierarch_and_map(
                    labels=index_labels,
                    depth_map=range(index_labels.shape[1]), # keep order
                    index_constructor=cls.from_labels,
                    index_constructors=index_constructors,
                    name=name,
                    )
            return index

        labels_iter = iter(labels)
        try:
            first = next(labels_iter)
        except StopIteration:
            # if iterable is empty, return empty index
            return cls(levels=cls._LEVEL_CONSTRUCTOR(
                    cls._INDEX_CONSTRUCTOR(())
                    ), name=name)

        depth = len(first)
        # minimum permitted depth is 2
        if depth < 2:
            raise ErrorInitIndex('cannot create an IndexHierarchy from only one level.')
        if index_constructors and len(index_constructors) != depth:
            raise ErrorInitIndex('if providing index constructors, number of index constructors must equal depth of IndexHierarchy.')

        depth_max = depth - 1
        depth_pre_max = depth - 2

        token = object()
        observed_last = [token for _ in range(depth)]

        tree = dict() # order assumed and necessary
        # put first back in front
        for label in chain((first,), labels_iter):
            current = tree # NOTE: over the life of this loop, current can be a dict or a list
            # each label is an iterable
            for d, v in enumerate(label):
                # print('d', d, 'v', v, 'depth_pre_max', depth_pre_max, 'depth_max', depth_max)
                if continuation_token is not CONTINUATION_TOKEN_INACTIVE:
                    if v == continuation_token:
                        # might check that observed_last[d] != token
                        v = observed_last[d]
                if d < depth_pre_max:
                    if v not in current:
                        current[v] = dict() # order necessary
                    else:
                        # can only fetch this node (and not create a new node) if this is the sequential predecessor
                        if v != observed_last[d]:
                            raise ErrorInitIndex('invalid tree-form for IndexHierarchy: {} in {} cannot follow {} when {} has already been defined.'.format(
                                    v,
                                    label,
                                    observed_last[d],
                                    v))
                    current = current[v]
                    observed_last[d] = v
                elif d < depth_max:
                    if v not in current:
                        current[v] = list()
                    else:
                        # cannot just fetch this list if it is not the predecessor
                        if v != observed_last[d]:
                            raise ErrorInitIndex('invalid tree-form for IndexHierarchy: {} in {} cannot follow {} when {} has already been defined.'.format(
                                    v,
                                    label,
                                    observed_last[d],
                                    v))
                    current = current[v]
                    observed_last[d] = v
                elif d == depth_max: # at depth max
                    # if there are redundancies here they will be caught in index creation
                    current.append(v)
                else:
                    raise ErrorInitIndex('label exceeded expected depth', label)

        return cls(levels=cls._tree_to_index_level(
                tree,
                index_constructors=index_constructors
                ), name=name)


# NOTE: this alternative implementation works, but is shown to be slower than the implementation used above
    # @classmethod
    # def from_labels(cls,
    #         labels: tp.Iterable[tp.Sequence[tp.Hashable]]) -> 'IndexHierarchy':
    #     '''
    #     From an iterable of labels, each constituting the components of each label, construct an index hierarcy.

    #     Args:
    #         labels: an iterator or generator of tuples.
    #     '''
    #     labels_iter = iter(labels)
    #     first = next(labels_iter)

    #     # minimum permitted depth is 2
    #     depth = len(first)
    #     assert depth >= 2
    #     depth_max = depth - 1

    #     pending_labels = [[] for _ in range(depth)]
    #     pending_targets = [[] for _ in range(depth)]
    #     previous_label = [None] * depth
    #     previous_level_length = [0] * depth

    #     active_position = [-1] * depth
    #     change = [False] * depth

    #     # but first back in front
    #     for offset, label in enumerate(chain((first,), labels_iter)):
    #         depth_label_pairs = list(enumerate(label))

    #         # iterate once through label to build change record; we do this here as, when iterating inner to outer below, we need to observe parent change
    #         for d, v in depth_label_pairs:
    #             assert v is not None # none is used as initial sentinalhs
    #             change[d] = previous_label[d] != v
    #             previous_label[d] = v

    #         # iterate again in reverse order, max depth first
    #         for d, v in reversed(depth_label_pairs):
    #             depth_parent = d - 1
    #             # from 0 to parent_depth, inclusive, gets all levels higher than this one
    #             if depth_parent >= 0:
    #                 is_change_parent = any(change[d_sub] for d_sub in range(depth_parent + 1))
    #             else:
    #                 is_change_parent = False

    #             if is_change_parent:
    #                 if offset > 0:
    #                     index = cls._INDEX_CONSTRUCTOR(pending_labels[d])
    #                     pending_labels[d] = [] #.clear()

    #                     if d == depth_max:
    #                         targets = None
    #                     else:
    #                         targets = np.array(pending_targets[d])
    #                         pending_targets[d] = [] #.clear()

    #                     level = cls._LEVEL_CONSTRUCTOR(
    #                             index=index,
    #                             targets=targets,
    #                             offset=previous_level_length[d])

    #                     previous_level_length[d] = len(level)
    #                     # after setting an upper level, all lower levels go to zero
    #                     for d_sub in range(d + 1, depth):
    #                         previous_level_length[d_sub] = 0

    #                     pending_targets[depth_parent].append(level)
    #                     # print('adding to pending targets', level, offset, len(level))

    #             if change[d] or is_change_parent:
    #                 # only update if changed, or parnet hc
    #                 pending_labels[d].append(v)

    #         # print(offset, label, change)

    #     # always one left to handle for all depths
    #     for d, v in reversed(depth_label_pairs):
    #         depth_parent = d - 1

    #         index = cls._INDEX_CONSTRUCTOR(pending_labels[d])
    #         if d == depth_max:
    #             targets = None
    #         else:
    #             targets = np.array(pending_targets[d])
    #             # pending_targets[d] = [] #.clear()

    #         level = cls._LEVEL_CONSTRUCTOR(
    #                 index=index,
    #                 targets=targets,
    #                 offset=previous_level_length[d])
    #         # assign it to pending unless we are at the top-most
    #         if depth_parent >= 0:
    #             pending_targets[depth_parent].append(level)

    #     # import ipdb; ipdb.set_trace()
    #     return cls(levels=level)


    @classmethod
    def from_index_items(cls: tp.Type[IH],
            items: tp.Iterable[tp.Tuple[tp.Hashable, Index]],
            *,
            index_constructor: tp.Optional[IndexConstructor] = None
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

            index = mutable_immutable_index_filter(cls.STATIC, index)
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

        return cls(cls._LEVEL_CONSTRUCTOR(
                index=index_outer,
                targets=targets,
                own_index=True
                ))


    @classmethod
    def from_labels_delimited(cls: tp.Type[IH],
            labels: tp.Iterable[str],
            *,
            delimiter: str = ' ',
            name: tp.Hashable = None,
            index_constructors: tp.Optional[IndexConstructors] = None,
            ) -> IH:
        '''
        Construct an ``IndexHierarhcy`` from an iterable of labels, where each label is string defining the component labels for all hierarchies using a string delimiter. All components after splitting the string by the delimited will be literal evaled to produce proper types; thus, strings must be quoted.

        Args:
            labels: an iterator or generator of tuples.

        Returns:
            :obj:`static_frame.IndexHierarchy`
        '''
        def trim_outer(label: str) -> str:
            start, stop = 0, len(label)
            if label[0] in ('[', '('):
                start = 1
            if label[-1] in (']', ')'):
                stop = -1
            return label[start: stop]

        labels = (tuple(literal_eval(x)
                for x in trim_outer(label).split(delimiter))
                for label in labels
                )
        return cls.from_labels(labels,
                name=name,
                index_constructors=index_constructors
                )

    #---------------------------------------------------------------------------
    def __init__(self,
            levels: tp.Union[IndexLevel, 'IndexHierarchy'],
            *,
            name: tp.Hashable = None
            ):
        '''
        Args:
            levels: IndexLevels instance, or, optionally, an IndexHierarchy to be used to construct a new IndexHierarchy.
            labels: a client can optionally provide the labels used to construct the levels, as an optional optimization in forming the IndexHierarchy.
        '''

        if issubclass(levels.__class__, IndexHierarchy):
            # handle construction from another IndexHierarchy
            if self.STATIC and levels.STATIC:
                self._levels = levels._levels
            else:
                # must deepcopy labels if not static; passing level constructor ensures we get a mutable if the parent is mutable
                self._levels = levels._levels.to_index_level(
                        cls=self._LEVEL_CONSTRUCTOR
                        )

            self._labels = levels.values
            self._depth = levels.depth
            self._recache = False

            if name is None and levels.name is not None:
                name = levels.name

        elif isinstance(levels, IndexLevel):
            # always assume ownership of passed in IndexLevel
            self._levels = levels
            # vlaues derived from levels are deferred
            self._labels = None
            self._depth = None
            self._recache = True

        else:
            raise NotImplementedError(f'no handling for creation from {levels}')

        self._name = name if name is None else name_filter(name)


    #---------------------------------------------------------------------------
    def __setstate__(self, state):
        '''
        Ensure that reanimated NP arrays are set not writeable.
        '''
        for key, value in state[1].items():
            setattr(self, key, value)
        if self._labels is not None:
            # might not yet have been created
            self._labels.flags.writeable = False

    #---------------------------------------------------------------------------
    # name interface

    def rename(self: IH, name: tp.Hashable) -> IH:
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


    def _iter_label(self, depth_level: int = 0):
        yield from self._levels.iter(depth_level=depth_level)

    def _iter_label_items(self, depth_level: int = 0):
        yield from enumerate(self._levels.iter(depth_level=depth_level))

    @property
    def iter_label(self) -> IterNodeDepthLevel:
        return IterNodeDepthLevel(
                container=self,
                function_items=self._iter_label_items,
                function_values=self._iter_label,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.INDEX_LABELS
                )

    # NOTE: Index implements drop property

    @property
    @doc_inject(select='astype')
    def astype(self) -> InterfaceAsType:
        '''
        Retype one or more depths. Can be used as as function to retype the entire ``IndexHierarchy``; alternatively, a ``__getitem__`` interface permits retyping selected depths.

        Args:
            {dtype}
        '''
        return InterfaceAsType(func_getitem=self._extract_getitem_astype)


    #---------------------------------------------------------------------------

    def _update_array_cache(self):
        # extract all features from self._levels
        self._depth = next(self._levels.depths())
        self._labels = self._levels.get_labels()
        self._recache = False

    #---------------------------------------------------------------------------

    def __len__(self) -> int:
        if self._recache:
            # faster to just get from levels instead of recaching
            return self._levels.__len__()
        return len(self._labels)

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

        # render display rows just of columns
        # sub_config_type = DisplayConfig(**config.to_dict(type_show=False))
        # sub_config_no_type = DisplayConfig(**config.to_dict(type_show=False))
        sub_config = config
        sub_display = None

        for d in range(self._depth):
            # as a slice this is far more efficient as no copy is made
            col = self._labels[:, d]
            # repeats = col == np.roll(col, 1)
            # repeats[0] = False
            # col[repeats] = '.' # TODO: spacer may not be best
            if sub_display is None: # the first
                sub_display = Display.from_values(
                        col,
                        header=DisplayHeader(self.__class__, self._name),
                        config=sub_config,
                        outermost=True,
                        index_depth=0,
                        header_depth=1)
            else:
                sub_display.extend_iterable(col, header='')

        return sub_display


    #---------------------------------------------------------------------------

    @property
    def values(self) -> np.ndarray:
        '''An 2D array of labels. Note: type coercion might be necessary.
        '''
        if self._recache:
            self._update_array_cache()
        return self._labels

    @property
    def depth(self) -> int:
        if self._recache:
            return next(self._levels.depths())
            # self._update_array_cache()
        return self._depth

    def values_at_depth(self,
            depth_level: DepthLevelSpecifier = 0
            ) -> np.ndarray:
        '''
        Return an NP array for the ``depth_level`` specified.

        Args:
            depth_level: a single depth level, or iterable depth of depth levels.
        '''
        if isinstance(depth_level, int):
            sel = depth_level
        else:
            sel = list(depth_level)
        # NOTE: thes values could have different types if we concatenate the values from each of the composed arrays, but outer layers would have to be multiplied
        return self.values[:, sel]


    @doc_inject()
    def label_widths_at_depth(self,
            depth_level: DepthLevelSpecifier = 0
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, int]]:
        '''{}'''
        if isinstance(depth_level, int):
            sel = depth_level
        else:
            raise NotImplementedError('selection from iterables is not implemented')
            # sel = list(depth_level)

        yield from self._levels.label_widths_at_depth(depth_level=depth_level)



    @property
    def dtypes(self) -> 'Series':
        '''
        Return a Series of dytpes for each index depth.

        Returns:
            :obj:`static_frame.Series`
        '''
        from static_frame.core.series import Series

        if self._name and len(self._name) == self.depth:
            labels = self._name
        else:
            labels = None
        return Series(self._levels.dtypes(), index=labels)


    @property
    def index_types(self) -> 'Series':
        '''
        Return a Series of Index classes for each index depth.

        Returns:
            :obj:`static_frame.Series`
        '''
        from static_frame.core.series import Series
        if self._name and len(self._name) == self.depth:
            labels = self._name
        else:
            labels = None
        return Series(self._levels.index_types(), index=labels)

    #---------------------------------------------------------------------------

    def copy(self: IH) -> IH:
        '''
        Return a new IndexHierarchy. This is not a deep copy.
        '''
        return self.__class__(levels=self._levels, name=self._name)


    def relabel(self, mapper: CallableOrMapping) -> 'IndexHierarchy':
        '''
        Return a new IndexHierarchy with labels replaced by the callable or mapping; order will be retained. If a mapping is used, the mapping should map tuple representation of labels, and need not map all origin keys.
        '''
        if self._recache:
            self._update_array_cache()

        if not callable(mapper):
            # if a mapper, it must support both __getitem__ and __contains__; as np.ndarray are not hashable, and self._labels is an np.ndarray, need to convert lookups to tuples
            getitem = getattr(mapper, '__getitem__')
            labels = (tuple(x) for x in self._labels)
            return self.__class__.from_labels(
                    (getitem(x) if x in mapper else x for x in labels),
                    name=self._name
                    )

        return self.__class__.from_labels(
                (mapper(x) for x in self._labels),
                name=self._name
                )

    def rehierarch(self,
            depth_map: tp.Iterable[int]
            ) -> 'IndexHierarchy':
        '''
        Return a new `IndexHierarchy` that conforms to the new depth assignments given be `depth_map`.
        '''
        index, _ = rehierarch_and_map(
                labels=self.values,
                index_constructor=self.__class__.from_labels,
                depth_map=depth_map,
                )
        return index

    #---------------------------------------------------------------------------

    def loc_to_iloc(self,
            key: tp.Union[GetItemKeyType, HLoc]
            ) -> GetItemKeyType:
        '''
        Given iterable of GetItemKeyTypes, apply to each level of levels.
        '''
        from static_frame.core.series import Series

        # NOTE: this implementation is different from Index.loc_to_iloc: here, we explicitly translate Series, Index, and IndexHierarchy before passing on to IndexLevels

        if isinstance(key, Index):
            # if an Index, we simply use the values of the index
            key = key.values

        if isinstance(key, IndexHierarchy):
            # default iteration of IH is as tuple
            return [self._levels.leaf_loc_to_iloc(k) for k in key]

        if isinstance(key, Series):
            if key.dtype == bool:
                # if a Boolean series, sort and reindex
                if _requires_reindex(key.index, self):
                    key = key.reindex(self, fill_value=False).values
                else: # the index is equal
                    key = key.values
            else:
                # For all other Series types, we simply assume that the values are to be used as keys in the IH. This ignores the index, but it does not seem useful to require the Series, used like this, to have a matching index value, as the index and values would need to be identical to have the desired selection.
                key = key.values

        # if an HLoc, will pass on to loc_to_iloc
        return self._levels.loc_to_iloc(key)

    def _extract_iloc(self, key) -> tp.Union['IndexHierarchy', tp.Tuple[tp.Hashable]]:
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
                labels = self._labels[key]
        elif isinstance(key, KEY_ITERABLE_TYPES):
            # we assume Booleans have been normalized to integers here
            # can select directly from _labels[key] if if key is a list
            labels = self._labels[key]
        else: # select a single label value: NOTE: convert array to tuple
            values = self._labels[key]
            if values.ndim == 1:
                return tuple(values)
            raise NotImplementedError(
                    'unhandled key type extracted a 2D array from labels') #pragma: no cover

        return self.__class__.from_labels(labels=labels, name=self._name)

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
        # key is an iloc key
        if isinstance(key, tuple):
            raise KeyError('__getitem__ does not support multiple indexers')
        # iloc_key = self.loc_to_iloc(key)
        return IndexHierarchyAsType(self, key=key)



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

        if isinstance(other, Index):
            # if this is a 1D index, must rotate labels before using an operator
            other = other.values.reshape((len(other), 1)) # operate on labels to labels
        elif isinstance(other, IndexHierarchy):
            # already 2D
            other = other.values # operate on labels to labels

        if operator.__name__ == 'matmul':
            return matmul(self._labels, other)
        elif operator.__name__ == 'rmatmul':
            return matmul(other, self._labels)

        array = operator(self._labels, other)
        array.flags.writeable = False
        return array


    def _ufunc_axis_skipna(self, *,
            axis,
            skipna,
            ufunc,
            ufunc_skipna,
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

        if skipna:
            post = ufunc_skipna(self._labels, axis=axis, dtype=dtype)
        else:
            post = ufunc(self._labels, axis=axis, dtype=dtype)

        post.flags.writeable = False
        return post


    # _ufunc_shape_skipna defined in IndexBase

    #---------------------------------------------------------------------------
    # dictionary-like interface

    # NOTE: we intentionally exclude keys(), items(), and get() from Index classes, as they return inconsistent result when thought of as a dictionary


    def __iter__(self) -> tp.Iterator[tp.Tuple[tp.Hashable, ...]]:
        '''Iterate over labels.
        '''
        if self._recache:
            self._update_array_cache()

        return tp.cast(tp.Iterator[tp.Hashable], array2d_to_tuples(self._labels.__iter__()))

    def __reversed__(self) -> tp.Iterator[tp.Tuple[tp.Hashable, ...]]:
        '''
        Returns a reverse iterator on the index labels.
        '''
        if self._recache:
            self._update_array_cache()
        return array2d_to_tuples(reversed(self._labels))


    def __contains__(self, value) -> bool:
        '''Determine if a leaf loc is contained in this Index.
        '''
        # levels only, no need to recache as this is what has been mutated
        return self._levels.__contains__(value)


    # def get(self, key: tp.Hashable, default: tp.Any = None) -> tp.Any:
    #     '''
    #     Return the value found at the index key, else the default if the key is not found.
    #     '''
    #     try:
    #         return self._levels.leaf_loc_to_iloc(key)
    #     except KeyError:
    #         return default

    #---------------------------------------------------------------------------
    # utility functions

    def sort(self,
            ascending: bool = True,
            kind: str = DEFAULT_SORT_KIND) -> 'Index':
        '''Return a new Index with the labels sorted.

        Args:
            kind: Sort algorithm passed to NumPy.
        '''
        if self._recache:
            self._update_array_cache()

        v = self._labels
        order = np.lexsort([v[:, i] for i in range(v.shape[1]-1, -1, -1)])

        if not ascending:
            order = order[::-1]

        values = v[order]
        values.flags.writeable = False
        return self.__class__.from_labels(values, name=self._name)


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
        '''Return an Index with values rotated forward and wrapped around (with a postive shift) or backward and wrapped around (with a negative shift).
        '''
        if self._recache:
            self._update_array_cache()

        values = self._labels

        if shift % len(values):
            values = array_shift(
                    array=values,
                    shift=shift,
                    axis=0,
                    wrap=True)
            values.flags.writeable = False
        return self.__class__.from_labels(values, name=self._name)



    #---------------------------------------------------------------------------
    # export

    def to_frame(self) -> 'Frame':
        '''
        Return the index as a Frame.
        '''
        from static_frame import Frame
        # NOTE: this should be done by column to preserve types per depth
        return Frame.from_records(self.values,
                columns=range(self._depth),
                index=None)

    def to_pandas(self) -> 'DataFrame':
        '''Return a Pandas MultiIndex.
        '''
        import pandas
        mi = pandas.MultiIndex.from_tuples(self.__iter__())
        mi.name = self._name
        mi.names = self.names
        return mi

    def flat(self) -> IndexBase:
        '''Return a flat, one-dimensional index of tuples for each level.
        '''
        return self._INDEX_CONSTRUCTOR(self.__iter__())

    def add_level(self, level: tp.Hashable):
        '''Return an IndexHierarchy with a new root level added.
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
        return self.__class__(levels, name=self._name)

    def drop_level(self, count: int = 1) -> tp.Union[Index, 'IndexHierarchy']:
        '''Return an IndexHierarchy with one or more leaf levels removed. This might change the size of the index if the resulting levels are not unique.
        '''

        if count < 0:
            levels = self._levels.to_index_level()
            for _ in range(abs(count)):
                levels_stack = [levels]
                while levels_stack:
                    level = levels_stack.pop()
                    # check to see if children of this target are leaves
                    if level.targets[0].targets is None:
                        level.targets = None
                    else:
                        levels_stack.extend(level.targets)
                if levels.targets is None:
                    # if our root level has no targets, we are at the root
                    break
            if levels.targets is None:
                # fall back to 1D index
                return levels.index
            return self.__class__(levels, name=self._name)

        elif count > 0:
            level = self._levels.to_index_level()
            for _ in range(count):
                # NOTE: do not need this check as we look ahead, below
                # if level.targets is None:
                #     return level.index
                targets = []
                labels = []
                for target in level.targets:
                    labels.extend(target.index)
                    if target.targets is not None:
                        targets.extend(target.targets)
                index = level.index.__class__(labels)
                if not targets:
                    return index
                level = level.__class__(index=index, targets=targets)
            return self.__class__(level, name=self._name)
        else:
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
            '_labels',
            '_depth',
            '_keys',
            '_length',
            '_recache',
            '_name'
            )

    # @classmethod
    # def from_pandas(cls, value) -> 'IndexHierarchyGO':
    #     '''
    #     Given a Pandas index, return the appropriate IndexBase derived class.
    #     '''
    #     return IndexBase.from_pandas(value, is_static=False)

    def append(self, value: tuple):
        '''
        Append a single label to this index.
        '''
        self._levels.append(value)
        self._recache = True

    def extend(self, other: IndexHierarchy):
        '''
        Extend this IndexHiearchy in-place
        '''
        self._levels.extend(other._levels)
        self._recache = True

    def copy(self: IH) -> IH:
        '''
        Return a new IndexHierarchy. This is not a deep copy.
        '''
        return self.__class__(
                levels=self._levels.to_index_level(),
                name=self._name
                )


# update class attr on Index after class initialziation
IndexHierarchy._MUTABLE_CONSTRUCTOR = IndexHierarchyGO



class IndexHierarchyAsType:

    __slots__ = ('container', 'key',)

    def __init__(self,
            container: IndexHierarchy,
            key: GetItemKeyType
            ) -> None:
        self.container = container
        self.key = key

    def __call__(self, dtype) -> 'IndexHierarchy':

        if self.key == NULL_SLICE:
            labels = self.container.values.astype(dtype)
            return self.container.__class__.from_labels(labels)

        def gen():
            for row in self.container.values:
                row = row.copy() # remove immutable reference
                # convert each column once per row; this is not optimal
                row[self.key] = row[self.key].astype(dtype)
                yield row

        return self.container.__class__.from_labels(gen())



