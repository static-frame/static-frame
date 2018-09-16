
import typing as tp
from collections import OrderedDict
from collections import deque
from itertools import chain

import numpy as np

from static_frame.core.util import GetItemKeyType
from static_frame.core.util import _NULL_SLICE
from static_frame.core.util import SLICE_ATTRS
from static_frame.core.util import SLICE_STOP_ATTR
from static_frame.core.util import _INT_TYPES
from static_frame.core.util import _intersect2d
from static_frame.core.util import _union2d
from static_frame.core.util import _resolve_dtype_iter


from static_frame.core.util import GetItem
from static_frame.core.util import _KEY_ITERABLE_TYPES
from static_frame.core.util import immutable_filter
from static_frame.core.util import CallableOrMapping

from static_frame.core.operator_delegate import MetaOperatorDelegate

from static_frame.core.display import DisplayConfig
from static_frame.core.display import DisplayActive
from static_frame.core.display import Display

from static_frame import Index
from static_frame import IndexGO



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

    def to_index_level(self,
            offset: tp.Optional[int]=0,
            cls: tp.Type['IndexLevel']=None,
            ) -> 'IndexLevel':
        '''
        Not a copy, but a clone, made with a different offset and possibly a different class.
        Args:
            offset: optionally provide a new offset for the copy. This is not applied recursively
        '''
        index = self.index.copy()

        if self.targets is not None:
            targets = np.empty(len(self.targets), dtype=object)
            for idx, t in enumerate(self.targets):
                # offset of None retains existing offset
                targets[idx] = t.to_index_level(offset=None, cls=cls)
        else:
            targets = None

        offset = self.offset if offset is None else offset
        cls = cls if cls else self.__class__
        return cls(index=index, targets=targets, offset=offset)


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
        # NOTE: as this uses a list instead of deque, the depths given will not be in the order of the actual leaves
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

        # this should not match tuples that are leaf-locs
        elif isinstance(key, _KEY_ITERABLE_TYPES):
            if isinstance(key, np.ndarray) and key.dtype == bool:
                return key # keep as Boolean?
            return [self.leaf_loc_to_iloc(x) for x in key]

        elif not isinstance(key, HLoc):
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

        if len(ilocs) == 0:
            raise KeyError('no matching keys')
        elif len(ilocs) == 1:
            return ilocs[0]

        # TODO: might be able to combine contiguous ilocs into a single slice
        iloc = [] # combine into one flat iloc
        length = self.__len__()
        for part in ilocs:
            if isinstance(part, slice):
                iloc.extend(range(*part.indices(length)))
            # just look for ints
            elif isinstance(part, _INT_TYPES):
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
        dtype = _resolve_dtype_iter(self.dtypes())
        labels = np.empty(shape, dtype=dtype)
        row_count = 0

        levels = deque() # order matters
        levels.append((self, 0, None))

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
            targets: tp.Optional[np.ndarray]=None, # np.ndarray[IndexLevel]
            offset: int=0
            ):
        assert isinstance(index, IndexGO)
        # assume that we must copy this index as it is mutable; possibly add an own_index obtion if hthis can be optimized
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
        offset_prior = self.__len__()
        count_prior = len(self.targets)

        # allocate new targets array
        targets = np.empty(count_prior + len(level.targets), dtype=object)
        # can assign these in without copying, as they are owned by this instance
        targets[:count_prior] = self.targets

        # targets are other IndexLevel instances that may be GO or not
        for idx, t in enumerate(level.targets, start=count_prior):
            # only need to update offsets at this level, as lower levels are relative to this
            target = t.to_index_level(offset_prior, cls=self.__class__)
            targets[idx] = target
            offset_prior += len(target)

        # TODO: handle validation of incomplete depths, duplicate values
        self.targets = targets


    def append(self, key: tuple):
        '''Add a single, full-depth leaf loc.
        '''
        # find fist depth that does not contain key
        depth_count = next(self.depths())
        # import ipdb; ipdb.set_trace()
        assert len(key) == depth_count

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
                    # TODO: possibly defer target appending
                    target_count = len(node.targets)
                    targets = np.empty(target_count + 1, dtype=object)
                    targets[:target_count] = node.targets

                    level_previous.offset = node.__len__()
                    targets[target_count] = level_previous
                    node.targets = targets

            else: # depth not found is higher up
                if node.targets is None:
                    # we are at the max depth; will need to create a LevelGO to append in th next level
                    level_previous = IndexLevelGO(
                            index=IndexGO((k,)),
                            offset=0,
                            targets=None
                            )
                else:
                    targets = np.empty(1, dtype=object)
                    targets[0] = level_previous
                    level_previous = IndexLevelGO(
                            index=IndexGO((k,)),
                            offset=0,
                            targets=targets
                            )




#-------------------------------------------------------------------------------
class IndexHierarchy(metaclass=MetaOperatorDelegate):

    __slots__ = (
            '_levels', # IndexLevel
            '_labels',
            '_depth',
            '_length',
            '_recache',
            'loc',
            'iloc',
            )

    STATIC = True
    _LEVEL_CONSTRUCTOR = IndexLevel
    _INDEX_CONSTRUCTOR = Index

    #---------------------------------------------------------------------------
    # constructors

    @classmethod
    def from_product(cls, *levels) -> 'IndexHierarchy': # tp.Iterable[tp.Hashable]
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
                # print(level, index.values, offset, targets_previous)

                targets[idx] = level
                offset += len(level)

            targets_previous = targets
            depth -= 1

        level = cls._LEVEL_CONSTRUCTOR(index=index_up, targets=targets_previous)
        return cls(level)

    @classmethod
    def from_tree(cls,
            tree
            # *,
            # labels: tp.Optional[np.ndarray]=None
            ) -> 'IndexHierarchy':
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

                index = cls._INDEX_CONSTRUCTOR(level_labels)

            else: # an iterable, terminal node, no offsets needed
                targets = None
                index = cls._INDEX_CONSTRUCTOR(level_data)

            return cls._LEVEL_CONSTRUCTOR(index=index, offset=offset, targets=targets)

        return cls(get_level(tree))

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
                        # TODO: can be key-only OD
                        current[v] = list()
                    current = current[v]
                elif d == depth_max: # at depth max
                    # TODO: not checking that v not in current
                    current.append(v)
                else:
                    raise Exception('label exceeded expected depth', label)

        return cls.from_tree(tree)


    #---------------------------------------------------------------------------
    def __init__(self,
            levels: IndexLevel
            ):
        '''
        Args:
            labels: optional reference to np.ndarray to be used as labels, instead of generating from passed IndexLevel objects.
        '''

        self._levels = levels

        # vlaues derived from levels are deferred
        self._labels = None
        self._depth = None
        self._length = None
        self._recache = True

        self.loc = GetItem(self._extract_loc)
        self.iloc = GetItem(self._extract_iloc)


    def _update_array_cache(self):
        # extract all features from self._levels
        depths = set(self._levels.depths())
        assert len(depths) == 1
        self._depth = depths.pop()
        self._length = self._levels.__len__()
        self._labels = self._levels.get_labels()
        self._recache = False


    def display(self, config: DisplayConfig=None) -> Display:
        config = config or DisplayActive.get()

        if self._recache:
            self._update_array_cache()

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
        if self._recache:
            self._update_array_cache()
        return self._length

    def __iter__(self):
        if self._recache:
            self._update_array_cache()
        return self._labels.__iter__()

    def __contains__(self, value) -> bool:
        '''Determine if a leaf loc is contained in this Index.
        '''
        # levels only, no need to recache
        return self._levels.__contains__(value)

    @property
    def values(self) -> np.ndarray:
        if self._recache:
            self._update_array_cache()
        return self._labels

    @property
    def mloc(self):
        if self._recache:
            self._update_array_cache()
        return mloc(self._labels)

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
    # set operations

    def intersection(self, other) -> 'IndexHierarchy':
        if self._recache:
            self._update_array_cache()

        if isinstance(other, np.ndarray):
            opperand = other
        else: # assume we can get it from a .values attribute
            opperand = other.values

        return self.__class__.from_labels(_intersect2d(self._labels, opperand))

    def union(self, other) -> 'IndexHierarchy':
        if self._recache:
            self._update_array_cache()

        if isinstance(other, np.ndarray):
            opperand = other
        else: # assume we can get it from a .values attribute
            opperand = other.values

        return self.__class__.from_labels(_union2d(self._labels, opperand))

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
        if self._recache:
            self._update_array_cache()

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
        return _ufunc_skipna_1d(
                array=self._labels,
                skipna=skipna,
                ufunc=ufunc,
                ufunc_skipna=ufunc_skipna)

    #---------------------------------------------------------------------------
    # export

    def to_frame(self):
        from static_frame import Frame
        return Frame.from_records(self.__iter__(),
                columns=range(self._depth),
                index=None)



class IndexHierarchyGO(IndexHierarchy):

    STATIC = False
    _LEVEL_CONSTRUCTOR = IndexLevelGO
    _INDEX_CONSTRUCTOR = IndexGO

    __slots__ = (
            '_levels', # IndexLevel
            '_labels',
            '_depth',
            '_length',
            '_recache',
            'loc',
            'iloc',
            )

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


        # length = self._length + other._levels.__len__()

        # labels = np.empty((length, self._depth), dtype=object)
        # labels[0:self._length, :] = self._labels
        # labels[self._length:, :] = other._labels
        # labels.flags.writeable = False

        # self._length = length
        # self._labels = labels