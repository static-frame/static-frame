
import typing as tp
from collections import OrderedDict
from collections import deque
from itertools import chain

import numpy as np

from static_frame.core.util import GetItemKeyType
from static_frame.core.util import _NULL_SLICE
from static_frame.core.util import SLICE_ATTRS
from static_frame.core.util import SLICE_STOP_ATTR

from static_frame.core.util import GetItem
from static_frame.core.util import _KEY_ITERABLE_TYPES
from static_frame.core.util import immutable_filter


from static_frame.core.display import DisplayConfig
from static_frame.core.display import DisplayActive
from static_frame.core.display import Display

from static_frame import Index



class HLocMeta(type):

    def __getitem__(self,
            key: GetItemKeyType
            ) -> tp.Iterable[GetItemKeyType]:
        if not isinstance(key, tuple):
            key = (key,)
        return self(key)

class HLoc(metaclass=HLocMeta):
    '''A container of hiearchical keys, that implements NULL slices or all lower dimensions that are not defined by construction.
    '''

    __slots__ = (
            'key',
            )

    def __init__(self, key: tp.Sequence[GetItemKeyType]):
        self.key = key

    def __iter__(self):
        return self.key.__iter__()

    def __getitem__(self, key: int):
        '''
        Each key reprsents a hierarchical level; if a key is not specified, the default should be to return the null slice.
        '''
        if key >= len(self.key):
            return _NULL_SLICE
        return self.key.__getitem__(key)

#-------------------------------------------------------------------------------
class IndexLevel:

    __slots__ = (
            'index',
            'targets',
            'offset'
            )

    def __init__(self,
            index: Index,
            targets: tp.Optional[np.ndarray]=None, # np.ndarray[IndexLevel]
            offset: int=0
            ):
        '''
        Args:
            offset: integer offset for this level.
            targets: np.ndarray of Indices; np.array supports fancy indexing for iloc compatible usage.
        '''
        if targets is not None:
            assert isinstance(targets, np.ndarray)
            try:
                assert len(targets) == len(index)
            except:
                raise Exception('targets must equal length of index')

        self.index = index
        self.targets = targets
        self.offset = offset

    def __len__(self):
        '''
        The length is the sum of all leaves
        '''
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
        # NOTE: as this uses a list instead of deque, the depths given will not be in the order of the leaves
        levels = [(self, 0)]
        while levels:
            level, depth = levels.pop()
            if level.targets is None: # terminus
                yield depth + 1
            else:
                next_depth = depth + 1
                levels.extend([(lvl, next_depth) for lvl in level.targets])

    def __contains__(self, key: tp.Iterable[tp.Hashable]) -> bool:
        '''Given an iterable of single-element level keys (a leaf loc), return a bool.
        '''
        node = self
        for k in key:
            if not node.index.__contains__(k):
                return False
            if node.targets is not None:
                node = node.targets[node.index.loc_to_iloc(k)]
            else: # targets is None, meaning we are done
                node.index.loc_to_iloc(k)
                return True # if above does not raise



    def leaf_loc_to_iloc(self, key: tp.Iterable[tp.Hashable]) -> int:
        '''Given an iterable of single-element level keys (a leaf loc), return the iloc value.
        '''
        node = self
        pos = 0
        for k in key:
            if node.targets is not None:
                node = node.targets[node.index.loc_to_iloc(k)]
                pos += node.offset
            else: # targets is None, meaning we are done
                # assume that k returns an integert
                return pos + node.index.loc_to_iloc(k)

    # def leaf_iloc_to_label(self, key: tp.Iterable[int]) -> tp.Hashable:
    #     '''
    #     Given leaf iloc key, return the terminal label
    #     '''
    #     node = self
    #     pos = 0
    #     for k in key:
    #         if node.targets is not None:
    #             node = node.targets[k]
    #             pos += node.offset
    #         else: # targets is None, meaning we are done
    #             # assume that k returns an integert
    #             return node.index.values[k - pos]

    def loc_to_iloc(self, key) -> GetItemKeyType:
        # TODO: should this be a generator?
        # NOTE: this is similar to Index.loc_to_iloc

        if isinstance(key, slice):
            # given a top-level definition of a slice (and if that slice results in a single value), we can get a value range
            # NOTE: this similar to LocMap.map_slice_args; can they be consolidated
            slice_args = []
            for field in SLICE_ATTRS:
                attr = getattr(key, field)
                if attr is None:
                    slice_args.append(attr)
                else:
                    pos = self.leaf_loc_to_iloc(attr)
                    if field is SLICE_STOP_ATTR:
                        slice_args.append(pos + 1)
                    else:
                        slice_args.append(pos)
            return slice(*slice_args)

        elif isinstance(key, _KEY_ITERABLE_TYPES):
            if isinstance(key, np.ndarray) and key.dtype == bool:
                return key # keep as Boolean?
            return [self.leaf_loc_to_iloc(x) for x in key]

        if not isinstance(key, HLoc):
            # assume it is a leaf loc tuple
            return self.leaf_loc_to_iloc(key)

        # collect all ilocs for all leaf indices matching HLoc patterns
        ilocs = []
        levels = deque() # order matters
        levels.append((self, 0, 0))
        while levels:
            level, depth, offset = levels.popleft()
            depth_key = key[depth]
            next_offset = offset + level.offset

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
        if len(ilocs) == 1:
            return ilocs[0]

        # TODO: might be able to combine contiguous ilocs into a single slice
        iloc = [] # combine into one flat iloc
        length = self.__len__()
        for part in ilocs:
            if isinstance(part, slice):
                iloc.extend(range(*part.indices(length)))
            # just look for ints
            elif isinstance(part, int):
                iloc.append(part)
            else: # assume it is an iterable
                iloc.extend(part)
        return iloc


    # def yield_depth(self, depth_target):
    #     '''
    #     Yield values for a single depth.
    #     '''

    #     levels = deque() # order matters
    #     levels.append((self, 0, None))
    #     while levels:
    #         level, depth, value_target = levels.popleft()

    #         if level.targets is None:
    #             for label in level.index.values:
    #                 if depth_target == depth:
    #                     yield label
    #                 else:
    #                     yield value_target

    #         else: # target is iterable np.ndaarray
    #             for label, level_target in zip(level.index.values, level.targets):
    #                 if depth_target == depth:
    #                     value_target = label
    #                 levels.append((level_target, depth+1, value_target))


    def get_labels_in_place(self):
        depth_count = next(self.depths())
        shape = self.__len__(), depth_count
        labels = np.empty(shape, dtype=object)
        row_count = 0

        levels = deque() # order matters
        levels.append((self, 0, None))

        while levels:
            level, depth, row_previous = levels.popleft()

            if level.targets is None:
                for label in level.index.values:
                    labels[row_count, :] = row_previous
                    labels[row_count, depth] = label
                    row_count += 1

            else: # target is iterable np.ndaarray
                depth_next = depth + 1
                for label, level_target in zip(level.index.values, level.targets):
                    if row_previous is None:
                        # shown to be faster to allocate entire row width
                        row = np.empty(depth_count, object)
                    else:
                        row = row_previous.copy()
                    row[depth] = label
                    levels.append((level_target, depth_next, row))

        return labels


    def get_labels_tuple_concat(self):

        depth_count = next(self.depths())
        shape = self.__len__(), depth_count
        labels = np.empty(shape, dtype=object)
        for idx, row in enumerate(_yield_tuples(self)):
            labels[idx, :] = row
        labels.flags.writeable = False
        return labels


    get_labels = get_labels_in_place
    # get_labels = get_labels_tuple_concat

        # # store template of a row

        # row_template = np.empty(depth_count, object)
        # # store next iloc for each depth
        # d_target_idx = np.empty(depth_count, int)
        # d_level = np.empty(depth_count, object)


        # # load initial values
        # level = self
        # for depth in range(depth_count):
        #     d_target_idx[depth] = 0
        #     d_level[depth] = level
        #     if level.targets is not None:
        #         level = level.targets[0]

        # for row_count in range(shape[0]):

        #     for depth in range(depth_count - 1, -1, -1):
        #         print(row_count, depth)

        #         if d_target_idx[depth] == len(d_level[depth].index.values):
        #             # need to go to parent to get the next target
        #             d_target_idx[depth] = 0

        #             # TODO: need to walk up depths from depth to 0
        #             # need to get next index at this level
        #             depth_parent = depth - 1
        #             if depth_parent < 0: #
        #                 import ipdb; ipdb.set_trace()
        #             level_parent = d_level[depth_parent]
        #             d_target_idx[depth_parent] = d_target_idx[depth_parent] + 1
        #             d_level[depth] = level_parent.targets[d_target_idx[depth_parent]]
        #             print('updating level', depth)

        #         # update row_template
        #         print('d_level', d_level)
        #         print('d_target_idx', d_target_idx)

        #         # for each depth, update row_template as necessary
        #         row_template[depth] = d_level[depth].index.values[d_target_idx[depth]]
        #         # update d index values and idx


        #         if d_level[depth].targets is None:
        #             # terminus, need to update epth
        #             d_target_idx[depth] += 1

        #     print('row_template', row_template)
        #     labels[row_count, :] = row_template



        # levels = deque() # order matters
        # levels.append((self, 0))
        # while levels:
        #     level, depth = levels.popleft()
        #     print(level, depth, row_count, row_template)

        #     if level.targets is None:
        #         # at at a termins, we right one row for each index label
        #         for label in level.index.values:
        #             row_template[depth] = label
        #             print(row_template)
        #             labels[row_count, :] = row_template
        #             row_count += 1
        #     else:
        #         for label, level_target in zip(level.index.values, level.targets):
        #             row_template[depth] = label
        #             levels.append((level_target, depth + 1))




#-------------------------------------------------------------------------------
# recursive functions for processing IndexLevels

# def _sum_length(level: IndexLevel, count=0):
#     if level.targets is None:
#         return count + level.index.__len__()
#     else:
#         for level in level.targets:
#             count += _sum_length(level)
#         return count

# def _yield_depths(level: IndexLevel, depth=0):
#     if level.targets is None:
#         yield depth + 1
#     else:
#         for level in level.targets:
#             yield from _yield_depths(level, depth + 1)

# def _yield_iloc(level: IndexLevel,
#         keys: HLoc,
#         depth: int,
#         offset: int):
#     '''
#     Generate iloc values given index level and iterator of keys.
#     '''
#     key = keys[depth]
#     if level.targets is None:
#         try:
#             yield level.index.loc_to_iloc(key, offset=offset + level.offset)
#         except KeyError:
#             pass
#     else:
#         # key may not be in this index; need to continue generator
#         try:
#             iloc = level.index.loc_to_iloc(key) # no offset
#         except KeyError:
#             iloc = None
#         # else: # can use else
#         if iloc is not None:
#             level_targets = level.targets[iloc] # get one or more IndexLevel objects
#             if not isinstance(level_targets, np.ndarray):
#                 # if not an ndarray, iloc has extracted as single IndexLevel
#                 yield from _yield_iloc(level_targets,
#                         keys,
#                         depth + 1,
#                         offset=offset + level.offset)
#             else: # target is iterable np.ndaarray
#                 for level_target in level_targets:
#                     yield from _yield_iloc(level_target,
#                             keys,
#                             depth + 1,
#                             offset=offset + level.offset)

def _yield_tuples(
        level: IndexLevel,
        parent_label=()):
    '''
    Generate tuples of all leaf-level labels of the index.
    '''
    # TODO: if we know the depth we can allocate tuples and assign with indices
    if level.targets is None:
        for label in level.index.values:
            yield parent_label + (label,)
    else:
        for label, level_target in zip(level.index.values, level.targets):
            yield from _yield_tuples(level_target, parent_label + (label,))


#-------------------------------------------------------------------------------
class IndexHierarchy:

    __slots__ = (
            '_levels',
            '_depth',
            '_length',
            '_labels',
            'loc',
            'iloc',
            )

    STATIC = True

    #---------------------------------------------------------------------------
    # constructors

    @classmethod
    def from_product(cls, *levels) -> 'IndexHierarchy': # tp.Iterable[tp.Hashable]
        indices = [] # store in a list, where index is depth
        for lvl in levels:
            if not isinstance(lvl, Index):
                lvl = Index(lvl)
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
                level = IndexLevel(index=index,
                        offset=offset,
                        targets=targets_previous)
                # print(level, index.values, offset, targets_previous)

                targets[idx] = level
                offset += len(level)

            targets_previous = targets
            depth -= 1

        level = IndexLevel(index=index_up, targets=targets_previous)
        return cls(level)


    @classmethod
    def from_tree(cls,
            tree,
            *,
            labels: tp.Optional[np.ndarray]=None) -> 'IndexHierarchy':
        '''
        Convert a dictionary of either iterables or dictionaries into a IndexHierarchy.

        Args:
            labels: a client can optionally provide the labels used to construct the tree, as an optional optimization in forming the IndexHierarchy.
        '''
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

                index = Index(level_labels)

            else: # an iterable, terminal node, no offsets needed
                targets = None
                index = Index(level_data)

            return IndexLevel(index=index, offset=offset, targets=targets)

        return cls(get_level(tree), labels=labels)


    @classmethod
    def from_labels(cls,
            labels: tp.Iterable[tp.Sequence[tp.Hashable]]) -> 'IndexHierarchy':
        '''
        From an iterable of labels, each constituting the components of each label, construct an index hierarcy.

        Args:
            labels: an iterator or generator of tuples.
        '''
        labels_iter = iter(labels)
        first = next(labels_iter)

        # minimum permitted depth is 2
        assert len(first) >= 2
        depth_max = len(first) - 1
        depth_pre_max = len(first) - 2

        tree = OrderedDict()

        # but first back in front
        for label in chain((first,), labels_iter):
            current = tree
            # each label is an iterablen
            for d, v in enumerate(label):
                if d < depth_pre_max:
                    if v not in current:
                        current[v] = OrderedDict()
                    current = current[v]
                elif d < depth_max:
                    if v not in current:
                        current[v] = list()
                    current = current[v]
                else: # at depth max
                    current.append(v)

        # benefit of passing labels only if labels is an np.array
        labels = None if not isinstance(labels, np.ndarray) else labels
        return cls.from_tree(tree, labels=labels)


    #---------------------------------------------------------------------------
    def __init__(self,
            levels: IndexLevel,
            *,
            labels: tp.Optional[np.ndarray]=None
            ):
        '''
        Args:
            labels: optional reference to np.ndarray to be used as labels, instead of generating from passed IndexLevel objects.
        '''

        self._levels = levels

        depths = set(self._levels.depths())
        assert len(depths) == 1
        self._depth = depths.pop()
        self._length = self._levels.__len__()


        if labels is None:
            # TODO: defer label construction until needed
            labels = self._levels.get_labels()
        else:
            assert labels.shape == (self._length, self._depth)
            # can keep without copying if already immutable
            labels = immutable_filter(labels)

        self._labels = labels

        self.loc = GetItem(self._extract_loc)
        self.iloc = GetItem(self._extract_iloc)


    def display(self, config: DisplayConfig=None) -> Display:
        config = config or DisplayActive.get()

        # render display rows just of columns
        sub_config = DisplayConfig(**config.to_dict(type_show=False))
        sub_display = None

        for d in range(self._depth):
            # as a slice this is far more efficient as no copy is made
            col = self._labels[:, d]

            # repeats = col == np.roll(col, 1)
            # repeats[0] = False
            # col[repeats] = '.' # TODO: spacer may not be best

            if sub_display is None:
                sub_display = Display.from_values(col,
                        header='',
                        config=sub_config)
            else:
                sub_display.append_iterable(col, header='')


        header = '<' + self.__class__.__name__ + '>'
        return Display.from_values(
                sub_display.to_rows()[1:-1], # truncate unused header
                header = header,
                config=config
                )

    def __repr__(self) -> str:
        return repr(self.display())

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        return self._labels.__iter__()

    def __contains__(self, value) -> bool:
        return self._levels.__contains__(value)

    @property
    def values(self) -> np.ndarray:
        return self._labels

    @property
    def mloc(self):
        return mloc(self._labels)

    def copy(self) -> 'IndexHierarchy':
        '''
        Return a new IndexHierarchy.
        '''
        return self.__class__(levels=self._levels, labels=self._labels)


    #---------------------------------------------------------------------------

    def loc_to_iloc(self,
            key: tp.Union[GetItemKeyType, HLoc]
            ) -> GetItemKeyType:
        '''
        Given iterable of GetItemKeyTypes, apply to each level of levels.

        '''
        return self._levels.loc_to_iloc(key)


    def _extract_iloc(self, key) -> 'IndexHierarchy':
        '''Extract a new index given an iloc key
        '''

        if key is None:
            labels = self._labels
        elif isinstance(key, slice):
            if key == _NULL_SLICE:
                labels = self._labels
            else:
                labels = self._labels[key]
        elif isinstance(key, _KEY_ITERABLE_TYPES):
            # we assume Booleans have been normalized to integers here
            # can select directly from _labels[key] if if key is a list
            labels = self._labels[key]
        else: # select a single label value
            labels = (self._labels[key],)

        return self.__class__.from_labels(labels=labels)

    def _extract_loc(self, key: GetItemKeyType) -> 'IndexHierarchy':
        return self._extract_iloc(self.loc_to_iloc(key))

    def __getitem__(self, key: GetItemKeyType) -> 'IndexHierarchy':
        '''Extract a new index given an iloc key.
        '''
        return self._extract_iloc(key)

    #---------------------------------------------------------------------------
    # export

    def to_frame(self):
        from static_frame import Frame

        # just need one, assume all the same
        depth = next(self._levels.depths())
        return Frame.from_records(self.__iter__(),
                columns=range(depth),
                index=None)