

import typing as tp
from collections import deque

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
from static_frame.core.index import LocMap
from static_frame.core.util import resolve_dtype_iter




class IndexLevel:
    '''
    A nestable representation of an Index, where labels in that index optionally point to other Index objects.
    '''
    __slots__ = (
            'index',
            'targets',
            'offset'
            )

    def __init__(self,
            index: Index,
            targets: tp.Optional[ArrayGO] = None, # np.ndarray[IndexLevel]
            offset: int = 0
            ):
        '''
        Args:
            offset: integer offset for this level.
            targets: np.ndarray of Indices; np.array supports fancy indexing for iloc compatible usage.
        '''
        self.index = index
        self.targets = targets
        self.offset = offset

    def to_index_level(self,
            offset: tp.Optional[int] = 0,
            cls: tp.Type['IndexLevel'] = None,
            ) -> 'IndexLevel':
        '''
        A deepcopy with optional adjustments, such as a different offset and possibly a different class.

        Args:
            offset: optionally provide a new offset for the copy. This is not applied recursively
        '''
        index = self.index.copy()

        if self.targets is not None:
            targets = ArrayGO(
                [t.to_index_level(offset=None, cls=cls) for t in self.targets],
                own_iterable=True)
        else:
            targets = None

        offset = self.offset if offset is None else offset
        cls = cls if cls else self.__class__
        return cls(index=index, targets=targets, offset=offset)

    def __len__(self):
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

    def depths(self) -> tp.Generator[int, None, None]:
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

    def dtypes(self) -> tp.Generator[int, None, None]:
        # NOTE: as this uses a list instead of deque, the depths given will not be in the order of the actual leaves
        if self.targets is None:
            yield self.index.values.dtype
        else:
            levels = [self]
            while levels:
                level = levels.pop()
                # use pulbic interface, as this might be an IndexGO
                yield level.index.values.dtype
                if level.targets is not None: # not terminus
                    levels.extend(level.targets)

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
            else: # targets is None, meaning we are done
                node.index.loc_to_iloc(k)
                return True # if above does not raise

    def iter(self, depth_level: int) -> tp.Generator[tp.Hashable, None, None]:
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


    def leaf_loc_to_iloc(self, key: tp.Iterable[tp.Hashable]) -> int:
        '''Given an iterable of single-element level keys (a leaf loc), return the iloc value.
        '''
        if isinstance(key, ILoc):
            return key.key

        node = self
        pos = 0
        for k in key:
            if isinstance(k, KEY_MULTIPLE_TYPES):
                raise RuntimeError('slices cannot be used in a leaf selection into an IndexHierarchy; try HLoc[{}].'.format(key))
            if node.targets is not None:
                node = node.targets[node.index.loc_to_iloc(k)]
                pos += node.offset
            else: # targets is None, meaning we are done
                # assume that k returns an integer
                return pos + node.index.loc_to_iloc(k)


    def loc_to_iloc(self, key: GetItemKeyType) -> GetItemKeyType:
        '''
        This is the low-level loc_to_iloc, analagous to LocMap.loc_to_iloc as used by Index. As such, the key at this point should not be a Series or Index object.
        '''
        if isinstance(key, slice):
            # given a top-level definition of a slice (and if that slice results in a single value), we can get a value range
            return slice(*LocMap.map_slice_args(self.leaf_loc_to_iloc, key))

        # this should not match tuples that are leaf-locs
        if isinstance(key, KEY_ITERABLE_TYPES):
            if isinstance(key, np.ndarray) and key.dtype == bool:
                return key # keep as Boolean?
            return [self.leaf_loc_to_iloc(x) for x in key]

        # elif isinstance(key, IndexHierarchy):
        #     # values will give an iterable if rows, where rows are iloc selectors
        #     return [self.leaf_loc_to_iloc(tuple(x)) for x in key.values]

        if not isinstance(key, HLoc):
            # assume it is a leaf loc tuple
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
            # import ipdb; ipdb.set_trace()

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
            raise KeyError('no matching keys across all levels')

        if iloc_count == 1 and not key.has_key_multiple():
            # drop to a single iloc selection
            return ilocs[0]

        # NOTE: might be able to combine contiguous ilocs into a single slice
        iloc = [] # combine into one flat iloc
        length = self.__len__()
        for part in ilocs:
            if isinstance(part, slice):
                iloc.extend(range(*part.indices(length)))
            # just look for ints
            elif isinstance(part, INT_TYPES):
                iloc.append(part)
            else: # assume it is an iterable
                iloc.extend(part)
        return iloc

    def get_labels(self) -> np.ndarray:
        '''
        Return an immutable NumPy 2D array of all labels found in this IndexLevels instance.
        '''
        # assume uniform depths
        depth_count = next(self.depths())
        shape = self.__len__(), depth_count
        dtype = resolve_dtype_iter(self.dtypes())
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

class IndexLevelGO(IndexLevel):
    '''Grow only variant of IndexLevel
    '''
    __slots__ = (
            'index',
            'targets',
            'offset'
            )

    def __init__(self,
            index: IndexGO,
            targets: tp.Optional[np.ndarray] = None,
            offset: int = 0
            ):
        assert isinstance(index, IndexGO)
        # assume that we must copy this index as it is mutable; possibly add an own_index option if this can be optimized
        index = index.copy()
        IndexLevel.__init__(self, index=index, targets=targets, offset=offset)

    #---------------------------------------------------------------------------
    # grow only mutation

    def extend(self, level: IndexLevel):
        # assert isinstance(level, IndexLevelGO)

        depth = next(self.depths())
        if depth != next(level.depths()):
            raise Exception('level for extension does not have necessary levels.')

        # this will raise for duplicates
        self.index.extend(level.index.values)

        def target_gen():
            offset_prior = self.__len__()
            for t in level.targets:
                # only need to update offsets at this level, as lower levels are relative to this
                target = t.to_index_level(offset_prior, cls=self.__class__)
                offset_prior += len(target)
                yield target

        self.targets.extend(target_gen())


    def append(self, key: tuple):
        '''Add a single, full-depth leaf loc.
        '''
        # find fist depth that does not contain key
        depth_count = next(self.depths())

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

        assert depth_not_found != -1
        level_previous = None

        for depth in range(depth_count - 1, depth_not_found - 1, -1):
            node = edge_nodes[depth]
            k = key[depth]
            # print('key', k, 'current edge index', node.index.values)

            if depth == depth_not_found:
                # when at the the depth not found, we always update the index
                node.index.append(k)

                # if we have targets, must update them
                if node.targets is not None:
                    level_previous.offset = node.__len__()
                    node.targets.append(level_previous)

            else: # depth not found is higher up
                if node.targets is None:
                    # we are at the max depth; will need to create a LevelGO to append in th next level
                    level_previous = IndexLevelGO(
                            index=IndexGO((k,)),
                            offset=0,
                            targets=None
                            )
                else:
                    # targets = np.empty(1, dtype=object)
                    targets = ArrayGO([level_previous,], own_iterable=True)
                    level_previous = IndexLevelGO(
                            index=IndexGO((k,)),
                            offset=0,
                            targets=targets
                            )




