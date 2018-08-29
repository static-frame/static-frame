
import typing as tp
from collections import OrderedDict
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
        return _sum_length(self)

    def depths(self):
        return _yield_depths(self)

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


    def loc_to_iloc(self, key) -> GetItemKeyType:

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

        raise NotImplementedError()

#-------------------------------------------------------------------------------
# recursive functions for processing IndexLevels

def _sum_length(level: IndexLevel, count=0):
    if level.targets is None:
        return count + level.index.__len__()
    else:
        for level in level.targets:
            count += _sum_length(level)
        return count

def _yield_depths(level: IndexLevel, depth=0):
    if level.targets is None:
        yield depth + 1
    else:
        for level in level.targets:
            yield from _yield_depths(level, depth + 1)

def _yield_iloc(level: IndexLevel,
        keys: HLoc,
        depth: int,
        offset: int):
    '''
    Generate iloc values given index level and iterator of keys.
    '''
    key = keys[depth]
    if level.targets is None:
        try:
            yield level.index.loc_to_iloc(key, offset=offset + level.offset)
        except KeyError:
            pass
    else:
        # key may not be in this index; need to continue generator
        try:
            iloc = level.index.loc_to_iloc(key) # no offset
        except KeyError:
            iloc = None
        # else: # can use else
        if iloc is not None:
            level_targets = level.targets[iloc] # get one or more IndexLevel objects
            if not isinstance(level_targets, np.ndarray):
                # if not an ndarray, iloc has extracted as single IndexLevel
                yield from _yield_iloc(level_targets,
                        keys,
                        depth + 1,
                        offset=offset + level.offset)
            else: # target is iterable np.ndaarray
                for level_target in level_targets:
                    yield from _yield_iloc(level_target,
                            keys,
                            depth + 1,
                            offset=offset + level.offset)

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

        labels_shape = (self._length, self._depth)

        if labels is None:
            labels = np.empty(labels_shape, dtype='object')
            for idx, row in enumerate(_yield_tuples(self._levels)):
                labels[idx, :] = row
            labels.flags.writeable = False
        else:
            assert labels.shape == labels_shape
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
        return self._labels.__len__()

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
        if isinstance(key, slice):
            return self._levels.loc_to_iloc(key)
        elif isinstance(key, _KEY_ITERABLE_TYPES):
            return self._levels.loc_to_iloc(key)

        iloc_sources = []
        for part in _yield_iloc(self._levels, key, 0, 0):
            iloc_sources.append(part)

        # if only one slice is obtained, best to return that
        if len(iloc_sources) == 1:
            return iloc_sources[0]

        iloc = []
        for part in iloc_sources:
            if isinstance(part, slice):
                iloc.extend(range(*part.indices(self.__len__())))
            # shuould just look for ints
            elif isinstance(part, int):
                iloc.append(part)
            else:
                # assume it is an iterable
                iloc.extend(part)
        return iloc

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