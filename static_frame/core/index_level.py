

import typing as tp
from collections import deque
from itertools import zip_longest

import numpy as np


from static_frame.core.array_go import ArrayGO
from static_frame.core.container_util import index_from_optional_constructor
from static_frame.core.doc_str import doc_inject
from static_frame.core.exception import ErrorInitIndexLevel
from static_frame.core.hloc import HLoc
from static_frame.core.index import ILoc
from static_frame.core.index import Index
from static_frame.core.index import IndexGO
from static_frame.core.index import LocMap
from static_frame.core.index import mutable_immutable_index_filter
from static_frame.core.index_base import IndexBase
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import GetItemKeyTypeCompound
from static_frame.core.util import IndexConstructor
from static_frame.core.util import IndexConstructors
from static_frame.core.util import IndexInitializer
from static_frame.core.util import INT_TYPES
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import KEY_ITERABLE_TYPES
from static_frame.core.util import KEY_MULTIPLE_TYPES
from static_frame.core.util import resolve_dtype_iter
from static_frame.core.util import EMPTY_TUPLE


# if tp.TYPE_CHECKING:
#     from static_frame.core.type_blocks import TypeBlocks #pylint: disable=W0611 #pragma: no cover

INDEX_LEVEL_SLOTS = (
            'index',
            'targets',
            'offset',
            '_depth',
            '_length',
            )



class IndexLevel:
    '''
    A nestable representation of an Index, where labels in that index optionally point to other Index objects.
    '''
    __slots__ = INDEX_LEVEL_SLOTS
    index: Index
    targets: tp.Optional[ArrayGO]
    offset: int
    _depth: tp.Optional[int]
    _length: tp.Optional[int]

    STATIC: bool = True
    _INDEX_CONSTRUCTOR = Index

    @classmethod
    def from_level_data(cls,
            level_data: tp.Any,
            get_index: tp.Callable[[IndexInitializer, int], IndexBase],
            offset: int = 0,
            depth: int = 0,
            ) -> 'IndexLevel':
        '''
        Recursive function used in ``from_tree`` constructor.
        '''
        if isinstance(level_data, dict):
            level_labels = []
            targets = np.empty(len(level_data), dtype=object)
            offset_local = 0

            # ordered key, value pairs, where the key is the label, the value is a list or dictionary; enmerate for insertion pre-allocated object array
            for idx, (k, v) in enumerate(level_data.items()):
                level_labels.append(k)
                level = cls.from_level_data(v,
                        get_index,
                        offset=offset_local,
                        depth=depth + 1)
                targets[idx] = level
                offset_local += len(level) # for lower level offsetting

            index = get_index(level_labels, depth)
            targets = ArrayGO(targets, own_iterable=True)

        else: # an iterable, terminal node, no offsets needed
            index = get_index(level_data, depth)
            targets = None

        return cls(
                index=index, #type: ignore
                offset=offset,
                targets=targets,
                )

    @classmethod
    def from_tree(cls,
            tree: tp.Any, # recursively defined
            index_constructors: tp.Optional[IndexConstructors] = None,
            depth_reference: tp.Optional[int] = None,
            ) -> 'IndexLevel':
        '''
        Convert a tree structure to an IndexLevel instance. As a tree structure is a dictionary of keys to either a sequence of hashables or a dict of other keys, there is no way to represent a zero length, non-zero depth structure.
        '''
        # tree: tp.Dict[tp.Hashable, tp.Union[Sequence[tp.Hashable], tp.Dict]]

        def get_index(labels: IndexInitializer, depth: int) -> Index:
            explicit_constructor: tp.Optional[IndexConstructor]

            if index_constructors is not None:
                explicit_constructor = index_constructors[depth]
            else:
                explicit_constructor = None

            return index_from_optional_constructor(labels, #type: ignore
                    default_constructor=cls._INDEX_CONSTRUCTOR,
                    explicit_constructor=explicit_constructor)

        if len(tree) == 0:
            return cls(get_index((), 0), depth_reference=depth_reference)
        # NOTE: code check that returned object has depth equal depth_reference
        return cls.from_level_data(tree, get_index)


    @classmethod
    def from_depth(cls, depth: int) -> 'IndexLevel':
        '''
        Create zero-legnth IndexLevel from depth.
        '''
        return cls(cls._INDEX_CONSTRUCTOR(EMPTY_TUPLE),
                own_index=True,
                depth_reference=depth,
                )

    def __init__(self,
            index: Index,
            targets: tp.Optional[ArrayGO] = None,
            offset: int = 0,
            own_index: bool = False,
            depth_reference: tp.Optional[int] = None,
            ):
        '''
        Args:
            index: a 1D Index defining outer-most labels to integers in the `targets` ArrayGO.
            offset: integer offset for this level.
            targets: None, or an ArrayGO of IndexLevel objects
            own_index: Boolean to determine whether the Index can be owned by this IndexLevel; if False, a static index will be reused if appropriate for this IndexLevel class.
            depth_reference: for zero length Levels, provide the depth if it is greater than 1.
        '''
        if not isinstance(index, Index) or index.depth > 1:
            raise ErrorInitIndexLevel('cannot create an IndexLevel from a higher-dimensional Index.')

        # NOTE: indices that contain tuples will take additional work to support; we are not at this time checking for them, though values_at_depth will fail

        if own_index:
            self.index = index
        else:
            self.index = mutable_immutable_index_filter(self.STATIC, index) #type: ignore

        self.targets = targets
        self.offset = offset

        # NOTE: once _depth is set, is ever re-evaluated over the life of the instance, even if it is an IndexLevelGO
        if len(index) > 0:
            self._depth = 1 if self.targets is None else None
            self._length = None
        else:
            if depth_reference is None:
                raise ErrorInitIndexLevel('zero length index requires specification of depth_reference')
            self._depth = depth_reference
            self._length = 0

    #---------------------------------------------------------------------------
    def depths(self) -> tp.Iterator[int]:
        '''
        Get the depth of all leaves (which should all be the same). Mostly for integrity validation.
        '''
        # NOTE: as this uses a list instead of deque, the depths given will not be in the order of the actual leaves
        if self.targets is None or not len(self.targets):
            # NOTE: requires being set in __init__
            yield self._depth #type: ignore
        else:
            levels = [(self, 0)]
            while levels:
                level, depth = levels.pop()
                if level.targets is None: # terminus
                    yield depth + 1
                else:
                    next_depth = depth + 1
                    levels.extend([(lvl, next_depth) for lvl in level.targets])

    def _get_depth(self) -> int:
        '''
        Called once over the life of an instance to set self._depth; this is not re-evaluated over the life of the instance.
        '''
        if not len(self.index):
            raise AssertionError('zero-length indices should have depth set through depth_reference')

        if self.targets is None: # this may not need to be here
            return 1

        # if we need to recurse to the max depth
        level, depth = self, 1
        while True:
            if level.targets is None: # terminus
                return depth
            level, depth = level.targets[0], depth + 1

    @property
    def depth(self) -> int:
        if self._depth is None:
            self._depth = self._get_depth()
        return self._depth

    def _get_length(self) -> int:
        # NOTE: unlike depth, length can change in an IndexLevelGO and should be re-evaluated.
        if self.targets is None or not len(self.targets):
            return self.index.__len__()

        count = 0
        levels = [self]
        while levels:
            level = levels.pop()
            if level.targets is None: # terminus
                count += level.index.__len__()
            else:
                levels.extend(level.targets)
        return count

    def __len__(self) -> int:
        '''
        The length is the sum of all leaves
        '''
        # NOTE: in IndexLevelGO, setting length to None is how length is reset after mutation.
        if self._length is None:
            self._length = self._get_length()
        return self._length

    #---------------------------------------------------------------------------
    def label_widths_at_depth(self,
            depth_level: int = 0
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, int]]:
        '''
        Generator of pairs of label, width, for all labels found at a specified level.
        '''
        # given a: 1, 2, b: 1, 2, return ('a', 2), ('b', 2)

        def get_widths(index: Index,
                targets: tp.Optional[ArrayGO]
                ) -> tp.Iterator[tp.Tuple[tp.Hashable, int]]:
            if targets is None:
                for label in index:
                    yield (label, 1)
            else: # observe the offsets of the next
                transversed = 0
                for i, (label, level_next) in enumerate(
                        zip_longest(index, targets[1:], fillvalue=None)
                        ):
                    if level_next is not None:
                        # print(label, level_next.offset, transversed)
                        # if the next offset is zero, we are moving to a component that is under a fresh hierarchy
                        if level_next.offset > 0:
                            delta = level_next.offset - transversed
                        else:
                            delta = len(targets[i])
                        yield label, delta
                        # get only the incremental addition for this label
                        transversed += delta
                    else:
                        # we cannot use offset; must to more expensive length of component Levels
                        yield label, len(targets[i])

        levels = deque(((self, 0),))
        while levels:
            level, depth = levels.popleft()
            if depth == depth_level:
                yield from get_widths(level.index, level.targets)
                continue # do not need to descend
            if level.targets is not None: # terminus
                next_depth = depth + 1
                levels.extend([(lvl, next_depth) for lvl in level.targets])


    def labels_at_depth(self,
            depth_level: int = 0
            ) -> tp.Iterator[np.ndarray]:
        '''
        Generator of arrays found at a depth level.
        '''
        levels = deque(((self, 0),))
        while levels:
            level, depth = levels.popleft()
            if depth == depth_level:
                yield level.index.values
                continue # do not need to descend
            if level.targets is not None: # terminus
                next_depth = depth + 1
                levels.extend([(lvl, next_depth) for lvl in level.targets])


    def label_nodes_at_depth(self, depth_level: int) -> tp.Iterator[tp.Hashable]:
        '''Given a depth position, iterate over label nodes at that depth. Only nodes will be provided, which for outer depths may not be of length equal to the entire index.
        '''
        if depth_level == 0:
            yield from self.index
        else:
            levels = deque(((self, 0),))
            while levels:
                level, depth = levels.popleft()
                if depth == depth_level:
                    yield from level.index
                    continue # do not need to descend
                if level.targets is not None: # terminus
                    next_depth = depth + 1
                    levels.extend([(lvl, next_depth) for lvl in level.targets])


    # TODO: consider a different name; was dtypes()
    def dtypes_iter(self) -> tp.Iterator[np.dtype]:
        '''Return an iterator of all dtypes from every depth level.'''
        if self.targets is None or not len(self.targets):
            yield self.index.values.dtype
        else:
            levels = [self]
            while levels:
                level = levels.pop()
                yield level.index.values.dtype
                if level.targets is not None: # not terminus
                    levels.extend(level.targets)


    def dtypes_at_depth(self, depth_level: int) -> tp.Iterator[np.dtype]:
        '''
        Return all dtypes found on a depth.
        '''
        if not self.index.__len__():
            if depth_level in range(self.depth):
                yield self.index.dtype
            else:
                raise RuntimeError(f'invalid depth: {depth_level}')
        else:
            levels = deque(((self, 0),))
            while levels:
                level, depth = levels.popleft()
                if depth == depth_level:
                    yield level.index.dtype
                    continue # do not need to descend
                if level.targets is not None: # terminus
                    next_depth = depth + 1
                    levels.extend([(lvl, next_depth) for lvl in level.targets])

    def dtype_per_depth(self) -> tp.Iterator[np.dtype]:
        '''Return a tuple of resolved dtypes, one from each depth level.'''
        depth_count = self.depth
        for d in range(depth_count):
            yield resolve_dtype_iter(self.dtypes_at_depth(d))

    # consider renaming index_types_per_depth
    def index_types(self) -> tp.Iterator[np.dtype]:
        '''Return an iterator of representative Index classes, one from each depth level.'''
        if not self.index.__len__():
            yield from (self.index.__class__ for _ in range(self.depth))
        elif self.targets is None:
            yield self.index.__class__
        else:
            levels = [self]
            while levels:
                level = levels.pop()
                yield level.index.__class__
                if level.targets is not None: # not terminus
                    levels.append(level.targets[0])
                else:
                    break

    #---------------------------------------------------------------------------
    def __contains__(self, key: tp.Iterable[tp.Hashable]) -> bool:
        '''Given an iterable of single-element level keys (a leaf loc), return a bool.
        '''
        if not hasattr(key, '__iter__') or isinstance(key, str):
            return False

        node = self
        for k in key:
            if not node.index.__contains__(k):
                return False

            if node.targets is not None:
                node = node.targets[node.index.loc_to_iloc(k)]
                continue

            node.index.loc_to_iloc(k)
            return True # if above does not raise

        return False

    def leaf_loc_to_iloc(self,
            key: tp.Union[tp.Iterable[tp.Hashable], tp.Type[ILoc], tp.Type[HLoc]]
            ) -> int:
        '''Given an iterable of single-element level keys (a leaf loc), return the iloc value.

        Note that key components (level selectors) cannot be slices, lists, or np.ndarray.
        '''
        if isinstance(key, ILoc):
            return key.key

        node = self
        pos = 0
        key_depth_max = len(key) - 1 #type: ignore

        # NOTE: rather than a for/enumerate, this could use a while loop on an iter() and explicitly look at next() results to determine if the key matches
        for key_depth, k in enumerate(key): #type: ignore
            if isinstance(k, KEY_MULTIPLE_TYPES):
                raise RuntimeError(f'slices cannot be used in a leaf selection into an IndexHierarchy; try HLoc[{key}].')
            if node.targets is not None:
                node = node.targets[node.index.loc_to_iloc(k)]
                pos += node.offset
            else: # targets is None, meaning we are at max depth
                # k returns an integer
                offset = node.index.loc_to_iloc(k)
                assert isinstance(offset, INT_TYPES) # enforces leaf loc
                if key_depth == key_depth_max:
                    return pos + offset
                break # return exception below if key_depth not max depth

        raise KeyError(f'Invalid key length {key_depth_max + 1}; must be length {self.depth}.')

    def loc_to_iloc(self, key: GetItemKeyTypeCompound) -> GetItemKeyType:
        '''
        This is the low-level loc_to_iloc, analagous to LocMap.loc_to_iloc as used by Index. As such, the key at this point should not be a Series or Index object.

        If key is an np.ndarray, a Boolean array will be passed through; otherwise, it will be treated as an iterable of values to be passed to leaf_loc_to_iloc.
        '''
        from static_frame.core.series import Series

        if isinstance(key, slice):
            # given a top-level definition of a slice (and if that slice results in a single value), we can get a value range
            return slice(*LocMap.map_slice_args(self.leaf_loc_to_iloc, key))

        if isinstance(key, KEY_ITERABLE_TYPES): # iterables of leaf-locs
            if isinstance(key, np.ndarray) and key.dtype == bool:
                return key # keep as Boolean
            return [self.leaf_loc_to_iloc(x) for x in key]

        if not isinstance(key, HLoc): # assume a leaf loc tuple
            if not isinstance(key, tuple):
                raise KeyError(f'{key} cannot be used for loc selection from IndexHierarchy; try HLoc')
            return self.leaf_loc_to_iloc(key)

        # HLoc following: collect all ilocs for all leaf indices matching HLoc patterns
        ilocs = []
        levels = deque(((self, 0, 0),)) # order matters

        while levels:
            level, depth, offset = levels.popleft()

            depth_key = key[depth]
            # NOTE: depth_key should not be Series or Index at this point; IndexHierarchy is responsible for unpacking / reindexing prior to this call
            next_offset = offset + level.offset

            if isinstance(depth_key, np.ndarray) and depth_key.dtype == DTYPE_BOOL:
                # NOTE: use length of level, not length of index, as need to observe all leafs covered at this node.
                depth_key = depth_key[next_offset: next_offset + len(level)]
                if len(depth_key) > len(level.index):
                    # given leaf-Boolean, determine what upper nodes to select
                    depth_key = level.values_at_depth(0)[depth_key]
                    if len(depth_key) > 1:
                        # NOTE: must strip repeated labels, but cannot us np.unique as must retain order
                        depth_key = list(dict.fromkeys(depth_key).keys())

            # print(level, depth, offset, depth_key, next_offset)
            if level.targets is None:
                try:
                    # NOTE: as a selection list might be given within the HLoc, it will be tested accross many indices, and should support a partial matching
                    ilocs.append(level.index.loc_to_iloc(
                            depth_key,
                            offset=next_offset,
                            partial_selection=True,
                            ))
                except KeyError:
                    pass
            else: # when not at a leaf, we are selecting level_targets to descend withing
                try: # NOTE: no offset necessary as not a leaf selection
                    iloc = level.index.loc_to_iloc(depth_key, partial_selection=True)
                except KeyError:
                    pass
                else:
                    level_targets = level.targets[iloc] # get one or more IndexLevel objects
                    next_depth = depth + 1
                    # if not an ndarray, iloc has extracted a single IndexLevel
                    if isinstance(level_targets, IndexLevel):
                        levels.append((level_targets, next_depth, next_offset))
                    else:
                        levels.extend([(lvl, next_depth, next_offset)
                                for lvl in level_targets])

        iloc_count = len(ilocs)
        if iloc_count == 0:
            raise KeyError('no matching keys across all levels')

        if iloc_count == 1 and not key.has_key_multiple():
            return ilocs[0] # drop to a single iloc selection

        # NOTE: might be able to combine contiguous ilocs into a single slice
        iloc_flat: tp.List[GetItemKeyType] = [] # combine into one flat iloc
        length = self.__len__()
        for part in ilocs:
            if isinstance(part, slice):
                iloc_flat.extend(range(*part.indices(length)))
            elif isinstance(part, INT_TYPES):
                iloc_flat.append(part)
            else: # assume it is an iterable
                iloc_flat.extend(part) #type: ignore
        return iloc_flat

    #---------------------------------------------------------------------------
    @property
    def values(self) -> np.ndarray:
        '''
        Return an immutable NumPy 2D array of all labels found in this IndexLevels instance. This may coerce types.
        '''
        depth_count = self.depth
        shape = self.__len__(), depth_count

        # need to get a compatible dtype for all dtypes
        dtype = resolve_dtype_iter(self.dtypes_iter())
        labels = np.empty(shape, dtype=dtype)

        row_count = 0
        levels = deque(((self, 0, None),)) # order matters

        while levels:
            level, depth, row_previous = levels.popleft()

            if level.targets is None:
                rows = len(level.index.values)
                row_slice = slice(row_count, row_count + rows)
                labels[row_slice, :] = row_previous
                labels[row_slice, depth] = level.index.values
                row_count += rows

            else: # target is iterable np.ndaarray
                depth_next = depth + 1
                for label, level_target in zip(level.index.values, level.targets):
                    if row_previous is None:
                        # shown to be faster to allocate entire row width
                        row = np.empty(depth_count, dtype=dtype)
                    else:
                        row = row_previous.copy()
                    row[depth] = label
                    levels.append((level_target, depth_next, row))

        labels.flags.writeable = False
        return labels

    def __iter__(self) -> tp.Iterator[tp.Tuple[tp.Hashable, ...]]:
        # NOTE: this implementation shown to be faster than a recursive purely recursive implementation.
        depth_count = self.depth
        levels = deque(((self, 0, None),)) # order matters

        while levels:
            level, depth, row_previous = levels.popleft()

            if level.targets is None:
                for v in level.index.values:
                    row_previous[depth] = v #type: ignore
                    yield tuple(row_previous) #type: ignore
            else: # target is iterable np.ndaarray
                depth_next = depth + 1
                for label, level_target in zip(level.index.values, level.targets):
                    if row_previous is None:
                        # shown to be faster to allocate entire row width
                        row = [None] * depth_count
                    else:
                        row = row_previous.copy()
                    row[depth] = label
                    levels.append((level_target, depth_next, row)) #type: ignore


    def values_at_depth(self,
            depth_level: int
            ) -> np.ndarray:
        '''
        For the given depth, return a correctly typed immutable array of length equal to the number of rows in the cosolidate values presentation.
        '''
        depth_count = self.depth
        dtype = tuple(self.dtype_per_depth())[depth_level]

        length = self.__len__()
        # pre allocate array to ensure we use a resovled type
        array = np.empty(length, dtype=dtype)

        if length == 0:
            array.flags.writeable = False
            return array

        if depth_level == depth_count - 1:
            # at maximal depth, can concat underlying arrays
            np.concatenate(
                    tuple(self.labels_at_depth(depth_level)),
                    out=array
                    )
        else:
            def gen() -> tp.Iterator[np.ndarray]:
                for value, size in self.label_widths_at_depth(
                        depth_level=depth_level):
                    if dtype.kind == 'O' and isinstance(value, tuple):
                        # this appears to the only way to do this:
                        part = np.empty(size, dtype=dtype)
                        for i in range(size):
                            part[i] = value
                        yield part
                    else:
                        yield np.full(size, value, dtype=dtype)

            np.concatenate(tuple(gen()), out=array)

            #NOTE: This alternative form produced a unicode error only on Windows up ot NP 1.17 for some tests
            # start = 0
            # for value, size in self.label_widths_at_depth(depth_level):
            #     end = start + size
            #     array[start: end] = value
            #     start = end

        array.flags.writeable = False
        return array

    #---------------------------------------------------------------------------
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
        elif not isinstance(other, IndexLevel):
            return False

        # same type from here
        if self.__len__() != other.__len__():
            return False
        if self.depth != other.depth:
            return False

        kwargs = dict(
                compare_name=compare_name,
                compare_dtype=compare_dtype,
                compare_class=compare_class,
                skipna=skipna,
                )

        if ((self.targets is None or len(self.targets) == 0)
                and (other.targets is None or len(other.targets) == 0)):
            return self.index.equals(other.index, **kwargs) #type: ignore

        # same length and depth, can traverse trees
        # can store tuple of object ids to note those that have already been examined.
        equal_pairs = set()

        levels_self = [self]
        levels_other = [other]
        while levels_self and levels_other:
            level_self = levels_self.pop()
            level_other = levels_other.pop()

            pair = (id(level_self.index), id(level_other.index))
            pair_found = pair in equal_pairs

            if not pair_found and not level_self.index.equals(level_other.index, **kwargs):
                return False

            if not pair_found: # but we know it is equal
                equal_pairs.add(pair)

            if level_self.targets is not None and level_other.targets is not None: # not terminus
                levels_self.extend(level_self.targets)
                levels_other.extend(level_other.targets)
            if level_self.targets is None and level_other.targets is None: # terminus
                continue
            if level_self.targets is None or level_other.targets is None:
                # at least one is at a terminus
                return False

        if not levels_self and not levels_other:
            return True # both exhausted
        return False #pragma: no cover

    #---------------------------------------------------------------------------
    # exporters

    def to_index_level(self,
            offset: tp.Optional[int] = 0,
            cls: tp.Optional[tp.Type['IndexLevel']] = None,
            ) -> 'IndexLevel':
        '''
        A deepcopy with optional adjustments, such as a different offset and possibly a different class. The supplied class will be used to construct the IndexLevel instance (as well as internal indices), permitting the production of an IndexLevelGO.

        Args:
            offset: optionally provide a new offset for the copy. This is not applied recursively
        '''
        cls = cls if cls else self.__class__

        index = mutable_immutable_index_filter(cls.STATIC, self.index)

        if self.targets is not None:
            targets: tp.Optional[ArrayGO] = ArrayGO(
                [t.to_index_level(offset=None, cls=cls) for t in self.targets],
                own_iterable=True)
        else:
            targets = None

        offset = self.offset if offset is None else offset
        return cls(index=index, #type: ignore
                targets=targets,
                offset=offset,
                depth_reference=self.depth,
                )


    def to_type_blocks(self) -> TypeBlocks:
        '''
        Provide a correctly typed TypeBlocks representation.
        '''
        try:
            depth_count = self.depth
        except StopIteration:
            # assume we have no depth or length
            return TypeBlocks.from_zero_size_shape()

        return TypeBlocks.from_blocks(
                self.values_at_depth(d) for d in range(depth_count)
                )

#-------------------------------------------------------------------------------
class IndexLevelGO(IndexLevel):
    '''Grow only variant of IndexLevel
    '''
    __slots__ = INDEX_LEVEL_SLOTS
    index: IndexGO
    targets: tp.Optional[np.ndarray]
    offset: int
    _depth: tp.Optional[int]
    _length: tp.Optional[int]

    STATIC: bool = False
    _INDEX_CONSTRUCTOR = IndexGO

    #---------------------------------------------------------------------------
    # grow only mutation
    # depth cannot change over the life of IndexLevel

    def extend(self, level: IndexLevel) -> None:
        '''Extend this IndexLevel with another IndexLevel, assuming that it has compatible depth and Index types.
        '''
        depth = self.depth

        if level.targets is None:
            raise RuntimeError('found IndexLevel with None as targets')
        if depth != level.depth:
            raise RuntimeError('level for extension does not have necessary levels.')
        if tuple(self.index_types()) != tuple(level.index_types()):
            raise RuntimeError('level for extension does not have corresponding types.')

        # this will raise for duplicates
        self.index.extend(level.index.values)

        def target_gen() -> tp.Iterator[GetItemKeyType]:
            offset_prior = self.__len__()
            for t in level.targets: #type: ignore
                # only need to update offsets at this level, as lower levels are relative to this
                target = t.to_index_level(offset_prior, cls=self.__class__)
                offset_prior += len(target)
                yield target

        if self.targets is None:
            raise RuntimeError('found IndexLevel with None as targets')

        self.targets.extend(target_gen())

        # defer calculation be setting _length to None
        self._length = None

    def append(self, key: tp.Sequence[tp.Hashable]) -> None:
        '''Add a single, full-depth leaf loc.
        '''
        # find fist depth that does not contain key
        depth_count = self.depth

        if len(key) != depth_count:
            raise RuntimeError('appending key {} of insufficent depth {}'.format(
                        key, depth_count))

        if not self.index.__len__():
            # where we have zero length, create new root index and targets from the key alone
            depth_max = depth_count - 1
            level_previous = None

            for depth in range(depth_max, -1, -1):
                k = key[depth]
                # NOTE: we do not want to take index_types, as it based on notional index from zero length structure
                # index_constructor = index_types[depth]
                index = self._INDEX_CONSTRUCTOR((k,))
                if depth == depth_max:
                    targets = None
                else:
                    targets = ArrayGO([level_previous,], own_iterable=True)

                if depth != 0:
                    level_previous = IndexLevelGO(index, targets)
                else:
                    self.index = index
                    self.targets = targets
            self._length = None
            return

        # NOTE: does not use index_types when starting from a zero-length IndexLevels
        index_types = tuple(self.index_types())

        depth_not_found = -1
        edge_nodes = np.empty(depth_count, dtype=object)
        node = self

        for depth, k in enumerate(key):
            edge_nodes[depth] = node
            # only set on first encounter in descent
            if depth_not_found == -1 and not node.index.__contains__(k):
                depth_not_found = depth
            if node.targets is not None:
                node = node.targets[-1]

        if depth_not_found == -1:
            raise RuntimeError('unable to set depth_not_found') #pragma: no cover

        level_previous = None

        # iterate from the innermost depth out
        for depth in range(depth_count - 1, depth_not_found - 1, -1):
            node = edge_nodes[depth]
            k = key[depth]

            if depth == depth_not_found:
                # when at the the depth not found, we always update the index
                node.index.append(k)
                # if we have targets, must update them
                if node.targets is not None:
                    assert level_previous is not None
                    level_previous.offset = node.__len__()
                    node.targets.append(level_previous)
            else: # depth not found is higher up
                # NOTE: do not need to use index_from_optional_constructor, as no explicit constructor is being supplied, and we can expect that the existing types must be valid
                index_constructor = index_types[depth]
                if node.targets is None:
                    # we are at the max depth; will need to create a LevelGO to append in the next level
                    level_previous = IndexLevelGO(
                            index=index_constructor((k,)),
                            offset=0,
                            targets=None
                            )
                else:
                    targets = ArrayGO([level_previous,], own_iterable=True)
                    level_previous = IndexLevelGO(
                            index=index_constructor((k,)),
                            offset=0,
                            targets=targets
                            )
        # defer calculation be setting _length to None for all edge levels
        for node in edge_nodes:
            node._length = None




