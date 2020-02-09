

import typing as tp
from collections import deque
from itertools import zip_longest

import numpy as np

from static_frame.core.hloc import HLoc
from static_frame.core.index import Index
from static_frame.core.index import ILoc
from static_frame.core.index import IndexGO
from static_frame.core.array_go import ArrayGO

from static_frame.core.util import KEY_MULTIPLE_TYPES
from static_frame.core.util import KEY_ITERABLE_TYPES
from static_frame.core.util import INT_TYPES
from static_frame.core.util import GetItemKeyType

from static_frame.core.util import resolve_dtype_iter
from static_frame.core.util import GetItemKeyTypeCompound

# from static_frame.core.index_base import IndexBase

from static_frame.core.index import LocMap
from static_frame.core.index import mutable_immutable_index_filter
from static_frame.core.exception import ErrorInitIndexLevel

class IndexLevel:
    '''
    A nestable representation of an Index, where labels in that index optionally point to other Index objects.
    '''
    __slots__ = (
            'index',
            'targets',
            'offset'
            )
    index: Index
    targets: tp.Optional[ArrayGO]
    offset: int

    STATIC: bool = True
    # _INDEX_CONSTRUCTOR = Index

    def __init__(self,
            index: Index,
            targets: tp.Optional[ArrayGO] = None, # np.ndarray[IndexLevel]
            offset: int = 0,
            own_index: bool = False
            ):
        '''
        Args:
            index: a 1D Index defining outer-most labels to integers in the `targets` ArrayGO.
            offset: integer offset for this level.
            targets: None, or an ArrayGO of IndexLevel objects
            own_index: Boolean to determine whether the Index can be owned by this IndexLevel; if False, a static index will be reused if appropriate for this IndexLevel class.
        '''
        # if self.STATIC != index.STATIC:
        #     raise ErrorInitIndexLevel('incorrect static state in supplied index')
        if not isinstance(index, Index) or index.depth > 1:
            # all derived Index should be depth == 1
            raise ErrorInitIndexLevel('cannot create an IndexLevel from a higher-dimensional Index.')

        if own_index:
            self.index = index
        else:
            self.index = mutable_immutable_index_filter(self.STATIC, index)

        self.targets = targets
        self.offset = offset

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
        # index = cls._INDEX_CONSTRUCTOR(self.index)

        if self.targets is not None:
            targets: tp.Optional[ArrayGO] = ArrayGO(
                [t.to_index_level(offset=None, cls=cls) for t in self.targets],
                own_iterable=True)
        else:
            targets = None

        offset = self.offset if offset is None else offset
        return cls(index=index, targets=targets, offset=offset)

    def __len__(self) -> int:
        '''
        The length is the sum of all leaves
        '''
        if self.targets is None:
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
                        yield label, level_next.offset - transversed
                        transversed += level_next.offset
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



    def depths(self) -> tp.Iterator[int]:
        # NOTE: as this uses a list instead of deque, the depths given will not be in the order of the actual leaves
        if self.targets is None:
            yield 1
        else:
            levels = [(self, 0)]
            while levels:
                level, depth = levels.pop()
                if level.targets is None: # terminus
                    yield depth + 1
                else:
                    next_depth = depth + 1
                    levels.extend([(lvl, next_depth) for lvl in level.targets])

    def dtypes_all(self) -> tp.Iterator[np.dtype]:
        '''Return an iterator of all dtypes from every depth level.'''
        if self.targets is None:
            yield self.index.values.dtype
        else:
            levels = [self]
            while levels:
                level = levels.pop()
                yield level.index.values.dtype
                if level.targets is not None: # not terminus
                    levels.extend(level.targets)

    def dtypes(self) -> tp.Iterator[np.dtype]:
        '''Return an iterator of reprsentative dtypes, one from each depth level.'''
        if self.targets is None:
            yield self.index.values.dtype
        else:
            levels = [self]
            while levels:
                level = levels.pop()
                yield level.index.values.dtype
                if level.targets is not None: # not terminus
                    levels.append(level.targets[0])
                else:
                    break

    def index_types(self) -> tp.Iterator[np.dtype]:
        '''Return an iterator of reprsentative Index classes, one from each depth level.'''
        if self.targets is None:
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


    def __contains__(self, key: tp.Iterable[tp.Hashable]) -> bool:
        '''Given an iterable of single-element level keys (a leaf loc), return a bool.
        '''
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

    def iter(self, depth_level: int) -> tp.Iterator[tp.Hashable]:
        '''Given a depth position, return labels at that depth.
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

        raise KeyError(f'Invalid key length {key_depth_max + 1}; must be length {next(self.depths())}.')

    def loc_to_iloc(self, key: GetItemKeyTypeCompound) -> GetItemKeyType:
        '''
        This is the low-level loc_to_iloc, analagous to LocMap.loc_to_iloc as used by Index. As such, the key at this point should not be a Series or Index object.

        If key is an np.ndarray, a Boolean array will be passed through; otherwise, it will be treated as an iterable of values to be passed to leaf_loc_to_iloc.
        '''
        if isinstance(key, slice):
            # given a top-level definition of a slice (and if that slice results in a single value), we can get a value range
            return slice(*LocMap.map_slice_args(self.leaf_loc_to_iloc, key))

        # this should not match tuples that are leaf-locs
        if isinstance(key, KEY_ITERABLE_TYPES):
            if isinstance(key, np.ndarray) and key.dtype == bool:
                return key # keep as Boolean
            return [self.leaf_loc_to_iloc(x) for x in key]

        if not isinstance(key, HLoc):
            # assume it is a leaf loc tuple
            if not isinstance(key, tuple):
                raise KeyError(f'{key} cannot be used for loc selection from IndexHierarchy; try HLoc')
            return self.leaf_loc_to_iloc(key)

        # everything after this is an HLoc

        # collect all ilocs for all leaf indices matching HLoc patterns
        ilocs = []
        levels = deque(((self, 0, 0),)) # order matters

        while levels:
            level, depth, offset = levels.popleft()
            depth_key = key[depth]
            next_offset = offset + level.offset

            # print(level, depth, offset, depth_key, next_offset)

            if level.targets is None:
                try:
                    ilocs.append(level.index.loc_to_iloc(depth_key, offset=next_offset))
                except KeyError:
                    pass
            else: # target is iterable np.ndaarray
                try:
                    iloc = level.index.loc_to_iloc(depth_key) # no offset
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
            # import ipdb; ipdb.set_trace()
            raise KeyError('no matching keys across all levels')

        if iloc_count == 1 and not key.has_key_multiple():
            # drop to a single iloc selection
            return ilocs[0]

        # NOTE: might be able to combine contiguous ilocs into a single slice
        iloc_flat: tp.List[GetItemKeyType] = [] # combine into one flat iloc
        length = self.__len__()
        for part in ilocs:
            if isinstance(part, slice):
                iloc_flat.extend(range(*part.indices(length)))
            # just look for ints
            elif isinstance(part, INT_TYPES):
                iloc_flat.append(part)
            else: # assume it is an iterable
                assert part is not None
                iloc_flat.extend(part)
        return iloc_flat

    def get_labels(self) -> np.ndarray:
        '''
        Return an immutable NumPy 2D array of all labels found in this IndexLevels instance.
        '''
        # assume uniform depths
        depth_count = next(self.depths())
        shape = self.__len__(), depth_count

        dtype = resolve_dtype_iter(self.dtypes_all())
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


    # def values_at_depth(self,
    #         depth_level: int
    #         ) -> np.ndarray:
    #     # NOTE: not yet accepting a depth_level as an iterable of ints
    #     # NOTE: this only concatenates found values in stored indicies, meaning that the length of the array will differ based on the depth level
          # NOTE: use label_widths_at_depth() to build arrays not at max depth


    #     if depth_level == 0:
    #         return self.index.values
    #     else:
    #         def gen():
    #             levels = deque(((self, 0),))
    #             while levels:
    #                 level, depth = levels.popleft()
    #                 if depth == depth_level:
    #                     yield level.index.values
    #                     continue # do not need to descend
    #                 if level.targets is not None: # terminus
    #                     next_depth = depth + 1
    #                     levels.extend([(lvl, next_depth) for lvl in level.targets])

    #         return np.concatenate(tuple(gen()))



class IndexLevelGO(IndexLevel):
    '''Grow only variant of IndexLevel
    '''
    __slots__ = (
            'index',
            'targets',
            'offset'
            )
    index: IndexGO
    targets: tp.Optional[np.ndarray]
    offset: int

    # _INDEX_CONSTRUCTOR = IndexGO
    STATIC: bool = False

    #---------------------------------------------------------------------------
    # grow only mutation

    def extend(self, level: IndexLevel) -> None:
        '''Extend this IndexLevel with another IndexLevel, assuming that it has compatible depth and Index types.
        '''

        depth = next(self.depths())

        if level.targets is None:
            raise RuntimeError('found IndexLevel with None as targets')
        if depth != next(level.depths()):
            raise RuntimeError('level for extension does not have necessary levels.')
        if tuple(self.index_types()) != tuple(level.index_types()):
            raise RuntimeError('level for extension does not have corresponding types.')

        # this will raise for duplicates
        self.index.extend(level.index.values)

        def target_gen() -> tp.Iterator[GetItemKeyType]:
            offset_prior = self.__len__()
            for t in level.targets: #type: ignore # should be able to figure out level.targets is not None
                # only need to update offsets at this level, as lower levels are relative to this
                target = t.to_index_level(offset_prior, cls=self.__class__)
                offset_prior += len(target)
                yield target

        if self.targets is None:
            raise RuntimeError('found IndexLevel with None as targets')

        self.targets.extend(target_gen())

    def append(self, key: tp.Sequence[tp.Hashable]) -> None:
        '''Add a single, full-depth leaf loc.
        '''
        # find fist depth that does not contain key
        depth_count = next(self.depths())
        index_types = tuple(self.index_types())

        if len(key) != depth_count:
            raise RuntimeError('appending key {} of insufficent depth {}'.format(
                        key, depth_count))

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
            # print('key', k, 'current edge index', node.index.values)

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
                    # targets = np.empty(1, dtype=object)
                    targets = ArrayGO([level_previous,], own_iterable=True)
                    level_previous = IndexLevelGO(
                            index=index_constructor((k,)),
                            offset=0,
                            targets=targets
                            )
