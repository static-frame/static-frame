
import typing as tp
from collections import OrderedDict
from collections import KeysView
from collections import deque
from itertools import chain

import numpy as np

from static_frame.core.util import DEFAULT_SORT_KIND
from static_frame.core.util import KEY_MULTIPLE_TYPES

from static_frame.core.index_base import IndexBase
from static_frame.core.index import Index
from static_frame.core.index import IndexGO
from static_frame.core.index import ILoc
from static_frame.core.index import _requires_reindex

from static_frame.core.util import GetItemKeyType
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import INT_TYPES
from static_frame.core.util import intersect2d
from static_frame.core.util import union2d
from static_frame.core.util import resolve_dtype_iter
from static_frame.core.util import array2d_to_tuples
from static_frame.core.util import ufunc_skipna_1d
from static_frame.core.util import name_filter

from static_frame.core.util import GetItem
from static_frame.core.util import KEY_ITERABLE_TYPES
from static_frame.core.util import CallableOrMapping
from static_frame.core.util import DepthLevelSpecifier

from static_frame.core.operator_delegate import MetaOperatorDelegate
from static_frame.core.array_go import ArrayGO

from static_frame.core.display import DisplayConfig
from static_frame.core.display import DisplayActive
from static_frame.core.display import Display
from static_frame.core.display import DisplayHeader

from static_frame.core.iter_node import IterNodeType
from static_frame.core.iter_node import IterNode
from static_frame.core.iter_node import IterNodeApplyType

from static_frame.core.hloc import HLoc

from static_frame.core.index_level import IndexLevel

from static_frame.core.index_level import IndexLevelGO


#-------------------------------------------------------------------------------
class IndexHierarchy(IndexBase,
        metaclass=MetaOperatorDelegate):
    '''
    A hierarchy of :py:class:`static_frame.Index` objects, defined as strict tree of uniform depth across all branches.
    '''
    __slots__ = (
            '_levels', # IndexLevel
            '_labels',
            '_depth',
            '_keys',
            '_length',
            '_recache',
            '_name'
            )

    # _IMMUTABLE_CONSTRUCTOR = None
    _INDEX_CONSTRUCTOR = Index
    _UFUNC_UNION = union2d
    _UFUNC_INTERSECTION = intersect2d

    _LEVEL_CONSTRUCTOR = IndexLevel


    #---------------------------------------------------------------------------
    # constructors

    @classmethod
    def from_product(cls,
            *levels,
            name: tp.Hashable = None
            ) -> 'IndexHierarchy': # tp.Iterable[tp.Hashable]
        '''
        Given groups of iterables, return an ``IndexHierarchy`` made of the product of a values in those groups, where the first group is the top-most hierarchy.

        Returns:
            :py:class:`static_frame.IndexHierarchy`

        '''
        indices = [] # store in a list, where index is depth
        for lvl in levels:
            if not isinstance(lvl, Index):
                lvl = cls._INDEX_CONSTRUCTOR(lvl)
            indices.append(lvl)
        if len(indices) == 1:
            raise NotImplementedError('only one level given')

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
    def _tree_to_index_level(cls, tree) -> IndexLevel:
        # tree: tp.Dict[tp.Hashable, tp.Union[Sequence[tp.Hashable], tp.Dict]]

        def get_level(level_data, offset=0):

            if isinstance(level_data, dict):
                level_labels = []
                targets = np.empty(len(level_data), dtype=object)
                offset_local = 0
                for idx, (k, v) in enumerate(level_data.items()):
                    level_labels.append(k)
                    level = get_level(v, offset=offset_local)
                    targets[idx] = level
                    offset_local += len(level)
                index = cls._INDEX_CONSTRUCTOR(level_labels)
                targets = ArrayGO(targets, own_iterable=True)
            else: # an iterable, terminal node, no offsets needed
                targets = None
                index = cls._INDEX_CONSTRUCTOR(level_data)

            return cls._LEVEL_CONSTRUCTOR(index=index, offset=offset, targets=targets)

        return get_level(tree)


    @classmethod
    def from_tree(cls,
            tree,
            *,
            name: tp.Hashable = None
            ) -> 'IndexHierarchy':
        '''
        Convert into a ``IndexHierarchy`` a dictionary defining keys to either iterables or nested dictionaries of the same.

        Returns:
            :py:class:`static_frame.IndexHierarchy`
        '''
        return cls(cls._tree_to_index_level(tree), name=name)


    @classmethod
    def from_labels(cls,
            labels: tp.Iterable[tp.Sequence[tp.Hashable]],
            *,
            name: tp.Hashable = None
            ) -> 'IndexHierarchy':
        '''
        Construct an ``IndexHierarhcy`` from an iterable of labels, where each label is tuple defining the component labels for all hierarchies.

        Args:
            labels: an iterator or generator of tuples.

        Returns:
            :py:class:`static_frame.IndexHierarchy`
        '''
        labels_iter = iter(labels)
        first = next(labels_iter)

        # minimum permitted depth is 2
        if len(first) < 2:
            raise RuntimeError('cannot create an IndexHierarhcy from only one level.')

        depth_max = len(first) - 1
        depth_pre_max = len(first) - 2

        token = object()
        observed_last = [token for _ in range(len(first))]

        tree = OrderedDict()
        # put first back in front
        for label in chain((first,), labels_iter):
            current = tree
            # each label is an iterable
            for d, v in enumerate(label):
                # print('d', d, 'v', v, 'depth_pre_max', depth_pre_max, 'depth_max', depth_max)
                if d < depth_pre_max:
                    if v not in current:
                        current[v] = OrderedDict()
                    else:
                        # can only fetch this node (and not create a new node) if this is the sequential predecessor
                        if v != observed_last[d]:
                            raise RuntimeError('invalid tree-form for IndexHierarchy: {} in {} cannot follow {} when {} has already been defined'.format(
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
                            raise RuntimeError('invalid tree-form for IndexHierarchy: {} in {} cannot follow {} when {} has already been defined.'.format(
                                    v,
                                    label,
                                    observed_last[d],
                                    v))
                    current = current[v]
                    observed_last[d] = v
                elif d == depth_max: # at depth max
                    # if there are redundancies her they will be caught in index creation
                    current.append(v)
                else:
                    raise RuntimeError('label exceeded expected depth', label)

        return cls(levels=cls._tree_to_index_level(tree), name=name)


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
            if levels.STATIC:
                self._levels = levels._levels
            else: # must deepcopy labels if not static
                self._levels = levels._levels.to_index_level()

            self._labels = levels.values
            self._depth = levels.depth
            self._keys = levels.keys() # immutable keys view can be shared
            self._length = self._labels.__len__() #levels.__len__()
            self._recache = False

            if name is None and levels.name is not None:
                name = levels.name

        elif isinstance(levels, IndexLevel):
            self._levels = levels
            # vlaues derived from levels are deferred
            self._labels = None
            self._depth = None
            self._keys = None
            self._length = None
            self._recache = True

        else:
            raise NotImplementedError('no handling for creation from', levels)

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


    def _iter_label(self, depth_level: int = 0):
        yield from self._levels.iter(depth_level=depth_level)

    def _iter_label_items(self, depth_level: int = 0):
        yield from enumerate(self._levels.iter(depth_level=depth_level))

    @property
    def iter_label(self) -> IterNode:
        return IterNode(
                container=self,
                function_items=self._iter_label_items,
                function_values=self._iter_label,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.INDEX_LABELS
                )

    #---------------------------------------------------------------------------

    def _update_array_cache(self):
        # extract all features from self._levels
        self._depth = next(self._levels.depths())
        # store both NP array of labels, as well as KeysView of hashable tuples
        self._labels = self._levels.get_labels()
        # note: this does not retain order in 3.5
        self._keys = KeysView._from_iterable(array2d_to_tuples(self._labels))
        # if we get labels, faster to get that length
        self._length = len(self._labels) #self._levels.__len__()
        self._recache = False

    #---------------------------------------------------------------------------

    def __len__(self) -> int:
        if self._recache:
            # faster to just get from levels
            return self._levels.__len__()
        return self._length

    def display(self,
            config: tp.Optional[DisplayConfig] = None
            ) -> Display:
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
                        columns_depth=1)
            else:
                sub_display.extend_iterable(col, header='')

        return sub_display


    #---------------------------------------------------------------------------

    @property
    def values(self) -> np.ndarray:
        if self._recache:
            self._update_array_cache()
        return self._labels

    @property
    def depth(self):
        if self._recache:
            return next(self._levels.depths())
            # self._update_array_cache()
        return self._depth

    def values_at_depth(self, depth_level: DepthLevelSpecifier = 0):
        '''
        Return an NP array for the `depth_level` specified.
        '''
        if isinstance(depth_level, int):
            sel = depth_level
        else:
            sel = list(depth_level)
        return self.values[:, sel]


    #---------------------------------------------------------------------------

    def copy(self) -> 'IndexHierarchy':
        '''
        Return a new IndexHierarchy. This is not a deep copy.
        '''
        return self.__class__(levels=self._levels)


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
            return self.__class__.from_labels(getitem(x) if x in mapper else x for x in labels)

        return self.__class__.from_labels(mapper(x) for x in self._labels)


    #---------------------------------------------------------------------------

    def loc_to_iloc(self,
            key: tp.Union[GetItemKeyType, HLoc]
            ) -> GetItemKeyType:
        '''
        Given iterable of GetItemKeyTypes, apply to each level of levels.
        '''
        from static_frame.core.series import Series

        if isinstance(key, Index):
            # if an Index, we simply use the values of the index
            key = key.values

        if isinstance(key, IndexHierarchy):
            return [self._levels.leaf_loc_to_iloc(tuple(k)) for k in key.values]

        if isinstance(key, Series):
            if key.dtype == bool:
                if _requires_reindex(key.index, self):
                    key = key.reindex(self, fill_value=False).values
                else: # the index is equal
                    key = key.values
            else:
                key = key.values

        # if an HLoc, will pass on to loc_to_iloc
        return self._levels.loc_to_iloc(key)

    def _extract_iloc(self, key) -> 'IndexHierarchy':
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
        else: # select a single label value
            values = self._labels[key]
            if values.ndim == 1:
                return tuple(values)
            labels = (values,)

        return self.__class__.from_labels(labels=labels)

    def _extract_loc(self, key: GetItemKeyType) -> 'IndexHierarchy':
        return self._extract_iloc(self.loc_to_iloc(key))

    def __getitem__(self, key: GetItemKeyType) -> 'IndexHierarchy':
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
        if self._recache:
            self._update_array_cache()
        return self._keys

    def __iter__(self):
        '''Iterate over labels.
        '''
        if self._recache:
            self._update_array_cache()
        # TODO: replace with iter of self._keys on 3.6
        return self._labels.__iter__()

    def __contains__(self, value) -> bool:
        '''Determine if a leaf loc is contained in this Index.
        '''
        # levels only, no need to recache as this is what has been mutated
        return self._levels.__contains__(value)

    def get(self, key, default=None):
        '''
        Return the value found at the index key, else the default if the key is not found.
        '''
        try:
            return self._levels.leaf_loc_to_iloc(key)
        except KeyError:
            return default

    #---------------------------------------------------------------------------
    # utility functions

    def sort(self,
            ascending: bool = True,
            kind: str = DEFAULT_SORT_KIND) -> 'Index':
        '''Return a new Index with the labels sorted.

        Args:
            kind: Sort algorithm passed to NumPy.
        '''
        raise NotImplementedError()

    def isin(self, other: tp.Iterable[tp.Any]) -> np.ndarray:
        '''Return a Boolean array showing True where a label is found in other. If other is a multidimensional array, it is flattened.
        '''
        raise NotImplementedError()

    def roll(self, shift: int) -> 'Index':
        '''Return an Index with values rotated forward and wrapped around (with a postive shift) or backward and wrapped around (with a negative shift).
        '''
        raise NotImplementedError()

    #---------------------------------------------------------------------------
    # export

    def to_frame(self):
        '''
        Return the index as a Frame.
        '''
        from static_frame import Frame
        return Frame.from_records(self.__iter__(),
                columns=range(self._depth),
                index=None)

    def to_pandas(self):
        '''Return a Pandas MultiIndex.
        '''
        # NOTE: cannot set name attribute via from_tuples
        import pandas
        return pandas.MultiIndex.from_tuples(list(map(tuple, self.__iter__())))

    def flat(self):
        '''Return a flat, one-dimensional index of tuples for each level.
        '''
        return self._INDEX_CONSTRUCTOR(array2d_to_tuples(self.__iter__()))

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
                offset=0
                )
        return self.__class__(levels)

    def drop_level(self, count: int = 1) -> tp.Union[Index, 'IndexHieararchy']:
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
            return self.__class__(levels)

        elif count > 0:
            level = self._levels.to_index_level()
            for _ in range(count):
                if level.targets is None:
                    # we should already have a copy
                    return level.index
                else:
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
            return self.__class__(level)
        else:
            raise NotImplementedError('no handling for a 0 count drop level.')



class IndexHierarchyGO(IndexHierarchy):

    '''
    A hierarchy of :py:class:`static_frame.Index` objects that permits mutation only in the addition of new hierarchies or labels.
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

    @classmethod
    def from_pandas(cls, value) -> 'IndexHierarchyGO':
        '''
        Given a Pandas index, return the appropriate IndexBase derived class.
        '''
        return IndexBase.from_pandas(value, is_go=True)


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


    def copy(self) -> 'IndexHierarchy':
        '''
        Return a new IndexHierarchy. This is not a deep copy.
        '''
        return self.__class__(levels=self._levels.to_index_level())
