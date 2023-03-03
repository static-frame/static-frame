import itertools
import operator
import typing as tp
from ast import literal_eval
from copy import deepcopy
from functools import partial
from itertools import chain

import numpy as np
from arraykit import first_true_1d
from arraykit import get_new_indexers_and_screen
from arraykit import name_filter

from static_frame.core.container_util import constructor_from_optional_constructor
from static_frame.core.container_util import get_col_dtype_factory
from static_frame.core.container_util import index_from_optional_constructor
from static_frame.core.container_util import iter_component_signature_bytes
from static_frame.core.container_util import key_from_container_key
from static_frame.core.container_util import matmul
from static_frame.core.container_util import rehierarch_from_type_blocks
from static_frame.core.container_util import sort_index_for_order
from static_frame.core.display import Display
from static_frame.core.display import DisplayActive
from static_frame.core.display import DisplayHeader
from static_frame.core.display_config import DisplayConfig
from static_frame.core.doc_str import doc_inject
from static_frame.core.exception import ErrorInitIndex
from static_frame.core.hloc import HLoc
from static_frame.core.index import ILoc
from static_frame.core.index import Index
from static_frame.core.index import IndexGO
from static_frame.core.index import immutable_index_filter
from static_frame.core.index import mutable_immutable_index_filter
from static_frame.core.index_auto import RelabelInput
from static_frame.core.index_base import IndexBase
# from static_frame.core.index_auto import IndexAutoConstructorFactory
from static_frame.core.index_datetime import IndexDatetime
from static_frame.core.index_datetime import IndexNanosecond
from static_frame.core.loc_map import HierarchicalLocMap
from static_frame.core.loc_map import LocMap
from static_frame.core.node_dt import InterfaceDatetime
from static_frame.core.node_iter import IterNodeApplyType
from static_frame.core.node_iter import IterNodeDepthLevel
from static_frame.core.node_iter import IterNodeType
from static_frame.core.node_re import InterfaceRe
from static_frame.core.node_selector import InterfaceAsType
from static_frame.core.node_selector import InterfaceGetItem
from static_frame.core.node_selector import TContainer
from static_frame.core.node_str import InterfaceString
from static_frame.core.node_transpose import InterfaceTranspose
from static_frame.core.node_values import InterfaceValues
from static_frame.core.style_config import StyleConfig
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.util import CONTINUATION_TOKEN_INACTIVE
from static_frame.core.util import DEFAULT_SORT_KIND
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import DTYPE_INT_DEFAULT
from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import DTYPE_UINT_DEFAULT
from static_frame.core.util import EMPTY_ARRAY_INT
from static_frame.core.util import INT_TYPES
from static_frame.core.util import KEY_MULTIPLE_TYPES
from static_frame.core.util import NAME_DEFAULT
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import BoolOrBools
from static_frame.core.util import DepthLevelSpecifier
from static_frame.core.util import DtypesSpecifier
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import IndexConstructor
from static_frame.core.util import IndexConstructors
from static_frame.core.util import IndexInitializer
from static_frame.core.util import NameType
from static_frame.core.util import PositionsAllocator
from static_frame.core.util import UFunc
from static_frame.core.util import array2d_to_array1d
from static_frame.core.util import array_deepcopy
from static_frame.core.util import array_sample
from static_frame.core.util import blocks_to_array_2d
from static_frame.core.util import is_dtype_specifier
from static_frame.core.util import is_neither_slice_nor_mask
from static_frame.core.util import isfalsy_array
from static_frame.core.util import isin
from static_frame.core.util import isna_array
from static_frame.core.util import iterable_to_array_1d
from static_frame.core.util import key_to_datetime_key
from static_frame.core.util import run_length_1d
from static_frame.core.util import ufunc_unique
from static_frame.core.util import ufunc_unique1d_indexer
from static_frame.core.util import ufunc_unique1d_positions
from static_frame.core.util import validate_depth_selection
from static_frame.core.util import view_2d_as_1d

if tp.TYPE_CHECKING:
    import pandas  # pragma: no cover
    from pandas import DataFrame  # pylint: disable=W0611 # pragma: no cover

    from static_frame.core.frame import Frame  # pylint: disable=W0611,C0412 # pragma: no cover
    from static_frame.core.frame import FrameGO  # pylint: disable=W0611,C0412 # pragma: no cover
    from static_frame.core.series import Series  # pylint: disable=W0611,C0412 # pragma: no cover

IH = tp.TypeVar('IH', bound='IndexHierarchy')
IHGO = tp.TypeVar('IHGO', bound='IndexHierarchyGO')
IHAsType = tp.TypeVar('IHAsType', bound='IndexHierarchyAsType')

SingleLabelType = tp.Tuple[tp.Hashable, ...]
TreeNodeT = tp.Dict[tp.Hashable, tp.Union[tp.Sequence[tp.Hashable], 'TreeNodeT']]

_NBYTES_GETTER = operator.attrgetter('nbytes')

CompoundLabelType = tp.Tuple[tp.Union[slice, tp.Hashable, tp.List[tp.Hashable]], ...]
LocKeyType = tp.Union[
    'IndexHierarchy',
    HLoc,
    ILoc,
    CompoundLabelType,
    np.ndarray,
    tp.List[CompoundLabelType],
    slice,
]
IntegerLocType = tp.Union[int, np.ndarray, tp.List[int], slice]
ExtractionType = tp.Union['IndexHierarchy', SingleLabelType]

HashableToIntMapsT = tp.List[tp.Dict[tp.Hashable, int]]
GrowableIndexersT = tp.List[tp.List[int]]


def build_indexers_from_product(list_lengths: tp.Sequence[int]) -> np.ndarray:
    '''
    Creates a 2D indexer array given a sequence of `list_lengths`

    This is equivalent to: ``np.array(list(itertools.product(*(map(range, list_lengths))))).T``
    except it scales incredibly well.

    It observes that the indexers for a product will look like this:

    Example:

    lengths: (3, 3, 3)
    result:
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2] # 0
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1] # 1
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2] # 2

    We can think of each depth level as repeating two parts: elements & groups.

    For depth 0, each element is repeated 6x, each group 1x
    For depth 1, each element is repeated 3x, each group 3x
    For depth 2, each element is repeated 1x, each group 6x

    We can take advantage of this clear pattern by using cumulative sums to
    determine what those repetitions are, and then apply them.
    '''
    padded_lengths = np.full(len(list_lengths) + 2, 1, dtype=DTYPE_INT_DEFAULT)
    padded_lengths[1:-1] = list_lengths

    all_group_reps = np.cumprod(padded_lengths)[:-2]
    all_index_reps = np.cumprod(padded_lengths[::-1])[-3::-1]

    # Impl borrowed from pandas/core/reshape/util.py:cartesian_product
    result = np.array(
        [
            np.tile(
                    np.repeat(PositionsAllocator.get(list_length), repeats=all_index_reps[i]),
                    reps=np.product(all_group_reps[i])
                    )
            for i, list_length
            in enumerate(list_lengths)
        ],
        dtype=DTYPE_INT_DEFAULT,
    )
    result.flags.writeable = False
    return result


# 71% of from_arrays_small
# 83% of from_arrays_large (83% ufunc_unique1d_indexer)
def construct_indices_and_indexers_from_column_arrays(
        *,
        column_iter: tp.Iterable[np.ndarray],
        index_constructors_iter: tp.Iterable[IndexConstructor],
        ) -> tp.Tuple[tp.List[Index], np.ndarray]:
    indices: tp.List[Index] = []
    indexers: tp.List[np.ndarray] = []

    for column, constructor in zip(column_iter, index_constructors_iter):
        # Alternative approach that retains order
        # positions, indexer = ufunc_unique1d_positions(column)
        # unsorted_unique = column[np.sort(positions)]
        # indexer_remap = ufunc_unique1d_indexer(unsorted_unique)[1]
        # indexer = indexer_remap[indexer]
        # unique_values = unsorted_unique

        unique_values, indexer = ufunc_unique1d_indexer(column)

        # we call the constructor on all lvl, even if it is already an Index
        indices.append(constructor(unique_values))
        indexers.append(indexer)

    indexers = np.array(indexers, dtype=DTYPE_INT_DEFAULT)
    indexers.flags.writeable = False # type: ignore

    return indices, indexers


class PendingRow:
    '''
    Encapsulates a new label row that has yet to be inserted into a IndexHierarchy.
    '''
    __slots__ = ('row',)
    def __init__(self, row: SingleLabelType) -> None:
        self.row = row

    def __len__(self) -> int:
        '''Each row is a single label in an IndexHierarchy'''
        return 1

    def __iter__(self) -> tp.Iterator[tp.Hashable]:
        yield from self.row


# ------------------------------------------------------------------------------

class IndexHierarchy(IndexBase):
    '''
    A hierarchy of :obj:`Index` objects.
    '''

    __slots__ = (
            '_indices',
            '_indexers',
            '_name',
            '_blocks',
            '_recache',
            '_values',
            '_map',
            '_index_types',
            '_pending_extensions',
            )

    _indices: tp.List[Index] # Of index objects
    _indexers: np.ndarray # 2D - integer arrays
    _name: NameType
    _blocks: TypeBlocks
    _recache: bool
    _values: tp.Optional[np.ndarray] # Used to cache the property `values`
    _map: HierarchicalLocMap
    _index_types: tp.Optional['Series'] # Used to cache the property `index_types`
    _pending_extensions: tp.Optional[tp.List[tp.Union[SingleLabelType, 'IndexHierarchy']]]

    # _IMMUTABLE_CONSTRUCTOR is None from IndexBase
    # _MUTABLE_CONSTRUCTOR will be defined after IndexHierarhcyGO defined

    _INDEX_CONSTRUCTOR: IndexConstructor = Index
    _NDIM: int = 2

    # --------------------------------------------------------------------------
    # constructors

    @classmethod
    def _build_index_constructors(cls: tp.Type[IH],
            *,
            index_constructors: IndexConstructors,
            depth: int,
            ) -> tp.Iterator[IndexConstructor]:
        '''
        Returns an iterable of `depth` number of index constructors based on user-provided ``index_constructors``.

        Args:
            dtype_per_depth: Optionally provide a dtype per depth to be used with ``IndexAutoConstructorFactory``.
        '''
        if index_constructors is None:
            yield from (cls._INDEX_CONSTRUCTOR for _ in range(depth))

        elif callable(index_constructors): # support a single constructor
            ctr = constructor_from_optional_constructor(
                    default_constructor=cls._INDEX_CONSTRUCTOR,
                    explicit_constructor=index_constructors
                    )
            yield from (ctr for _ in range(depth))
        else:
            index_constructors = tuple(index_constructors)
            if len(index_constructors) != depth:
                raise ErrorInitIndex(
                    f'When providing multiple index constructors, their number ({len(index_constructors)}) must equal the depth of the IndexHierarchy ({depth}).'
                    )
            for ctr in index_constructors:
                yield constructor_from_optional_constructor(
                        default_constructor=cls._INDEX_CONSTRUCTOR,
                        explicit_constructor=ctr
                        )

    @staticmethod
    def _build_name_from_indices(
            indices: tp.List[Index],
            ) -> tp.Optional[SingleLabelType]:
        '''
        Builds the IndexHierarchy name from the names of `indices`. If one is not specified, the name is None
        '''
        name: SingleLabelType = tuple(index.name for index in indices)
        if any(n is None for n in name):
            return None
        return name


    @classmethod
    def from_pandas(cls: tp.Type[IH],
            value: 'pandas.MultiIndex',
            ) -> IH:
        '''
        Given a Pandas index, return the appropriate IndexBase derived class.
        '''
        import pandas
        if not isinstance(value, pandas.MultiIndex):
            raise ErrorInitIndex(f'from_pandas must be called with a Pandas MultiIndex object, not: {type(value)}')


        if value.has_duplicates:
            raise ErrorInitIndex(f'cannot create IndexHierarchy from a MultiIndex with duplicates: {value}')

        # Remove bloated labels
        value = value.remove_unused_levels()

        # iterating over a hierarchical index will iterate over labels
        name: tp.Optional[tp.Tuple[tp.Hashable, ...]] = tuple(value.names)

        # if not assigned Pandas returns None for all components, which will raise issue if trying to unset this index.
        if all(n is None for n in name): #type: ignore
            name = None

        def build_index(pd_idx: pandas.Index) -> Index:
            # NOTE: Newer versions of pandas will not require Python date objects to live inside
            # a DatetimeIndex. Instead, it will be a regular Index with dtype=object.
            # Only numpy datetime objects are put into a DatetimeIndex.
            if isinstance(pd_idx, pandas.DatetimeIndex):
                constructor: tp.Type[Index] = IndexNanosecond
            else:
                constructor = Index

            if cls.STATIC:
                return constructor(pd_idx, name=pd_idx.name)
            return tp.cast(Index, constructor._MUTABLE_CONSTRUCTOR(pd_idx))

        indices: tp.List[Index] = []
        indexers: np.ndarray = np.empty((value.nlevels, len(value)), dtype=DTYPE_INT_DEFAULT)

        for i, (levels, codes) in enumerate(zip(value.levels, value.codes)):
            indexers[i] = codes
            indices.append(build_index(levels))

        indexers.flags.writeable = False

        return cls(
                indices=indices,
                indexers=indexers,
                name=name,
                )

    @classmethod
    def from_product(cls: tp.Type[IH],
            *levels: IndexInitializer,
            name: NameType = None,
            index_constructors: IndexConstructors = None,
            ) -> IH:
        '''
        Given groups of iterables, return an ``IndexHierarchy`` made of the product of a values in those groups, where the first group is the top-most hierarchy.

        Args:
            *levels: index initializers (or Index instances) for each level
            name:
            index_consructors:

        Returns:
            :obj:`static_frame.IndexHierarchy`

        '''
        if len(levels) == 1:
            raise ErrorInitIndex('Cannot create IndexHierarchy from only one level.')

        indices = [] # store in a list, where index is depth

        index_constructors_iter = cls._build_index_constructors(
                index_constructors=index_constructors,
                depth=len(levels),
                )

        for lvl, constructor in zip(levels, index_constructors_iter):
            if isinstance(lvl, Index):
                indices.append(immutable_index_filter(lvl))
            else:
                indices.append(constructor(lvl))

        if name is None:
            name = cls._build_name_from_indices(indices)

        indexers = build_indexers_from_product(list(map(len, indices)))

        return cls(
                name=name,
                indices=indices,
                indexers=indexers,
                )

    @classmethod
    def _from_tree(cls: tp.Type[IH],
            tree: TreeNodeT,
            ) -> tp.Iterator[SingleLabelType]:
        '''
        Yields all the labels provided by a `tree`
        '''
        for label, subtree in tree.items():
            if isinstance(subtree, dict):
                for treerow in cls._from_tree(subtree):
                    yield (label, *treerow)
            else:
                for row in subtree:
                    yield (label, row)

    @classmethod
    def from_tree(cls: tp.Type[IH],
            tree: TreeNodeT,
            *,
            name: NameType = None,
            index_constructors: IndexConstructors = None,
            ) -> IH:
        '''
        Convert into a ``IndexHierarchy`` a dictionary defining keys to either iterables or nested dictionaries of the same.

        Returns:
            :obj:`static_frame.IndexHierarchy`
        '''
        return cls.from_labels(
                labels=cls._from_tree(tree),
                name=name,
                index_constructors=index_constructors,
                )

    @classmethod
    def _from_empty(cls: tp.Type[IH],
            empty_labels: tp.Iterable[tp.Sequence[tp.Hashable]],
            *,
            name: NameType = None,
            depth_reference: tp.Optional[int] = None,
            index_constructors: IndexConstructors = None,
            ) -> IH:
        '''
        Construct an IndexHierarchy from an iterable of empty labels.
        '''
        if empty_labels.__class__ is np.ndarray and empty_labels.ndim == 2: # type: ignore
            # if this is a 2D array, we can get the depth
            depth = empty_labels.shape[1] # type: ignore

            if depth == 0: # an empty 2D array can have 0 depth
                pass # do not set depth_reference, assume it is set
            elif (depth_reference is None and depth > 1) or (
                depth_reference is not None and depth_reference == depth
            ):
                depth_reference = depth
            else:
                raise ErrorInitIndex(
                    f'depth_reference provided {depth_reference} does not match depth of supplied array {depth}'
                )

        if not isinstance(depth_reference, INT_TYPES):
            raise ErrorInitIndex(
                'depth_reference must be an integer when labels are empty.'
            )

        if depth_reference == 1:
            raise ErrorInitIndex('Cannot create IndexHierarchy from only one level.')

        indexers = np.array([EMPTY_ARRAY_INT for _ in range(depth_reference)],
                dtype=DTYPE_INT_DEFAULT)
        indexers.flags.writeable = False

        index_constructors_iter = cls._build_index_constructors(
                index_constructors=index_constructors,
                depth=depth_reference,
                )
        indices = [ctr(()) for ctr in index_constructors_iter]

        if name is None:
            name = cls._build_name_from_indices(indices)

        return cls(
                indices=indices,
                indexers=indexers,
                name=name,
                )

    @classmethod
    def from_values_per_depth(cls: tp.Type[IH],
            values: tp.Union[np.ndarray, tp.Sequence[tp.Iterable[tp.Hashable    ]]],
            *,
            name: NameType = None,
            depth_reference: tp.Optional[int] = None,
            index_constructors: IndexConstructors = None,
            ) -> IH:
        '''
        Construct an :obj:`IndexHierarchy` from a 2D NumPy array, or a collection of 1D arrays per depth.

        Very similar implementation to :meth:`_from_type_blocks`, but avoids creating TypeBlocks instance.

        Returns:
            :obj:`IndexHierarchy`
        '''
        if values.__class__ is np.ndarray:
            size, depth = values.shape # type: ignore
            column_iter = values.T # type: ignore
            arrays = values
        elif not len(values):
            size = 0
            depth = depth_reference
        else:
            arrays = []
            size = -1
            for column in values:
                if column.__class__ is np.ndarray:
                    arrays.append(column)
                else:
                    a, _ = iterable_to_array_1d(column)
                    arrays.append(a)

                if size == -1:
                    size = len(arrays[-1])
                elif size != len(arrays[-1]):
                    raise ErrorInitIndex('per depth iterables must be the same length')

            # NOTE: we are not checking that they are all 1D
            depth = len(arrays)
            column_iter = arrays

        if not size:
            if depth is None:
                raise RuntimeError('depth_reference must be specified for empty values')
            return cls._from_empty((), name=name, depth_reference=depth)

        index_constructors_iter = cls._build_index_constructors(
                index_constructors=index_constructors,
                depth=depth,
                )

        indices, indexers = construct_indices_and_indexers_from_column_arrays(
                column_iter=column_iter,
                index_constructors_iter=index_constructors_iter,
                )

        # NOTE: some index_constructors will change the dtype of the final array
        if index_constructors is None:
            blocks = TypeBlocks.from_blocks(arrays)
            own_blocks = True
        else:
            blocks = None
            own_blocks = False

        if name is None:
            name = cls._build_name_from_indices(indices)


        return cls(
                indices=indices,
                indexers=indexers,
                name=name,
                blocks=blocks,
                own_blocks=own_blocks,
                )

    @classmethod
    def from_labels(cls: tp.Type[IH],
            labels: tp.Iterable[tp.Sequence[tp.Hashable]],
            *,
            name: NameType = None,
            reorder_for_hierarchy: bool = False,
            index_constructors: IndexConstructors = None,
            depth_reference: tp.Optional[int] = None,
            continuation_token: tp.Union[tp.Hashable, None] = CONTINUATION_TOKEN_INACTIVE,
            ) -> IH:
        '''
        Construct an ``IndexHierarchy`` from an iterable of labels, where each label is tuple defining the component labels for all hierarchies.

        Args:
            labels: an iterator or generator of tuples.
            *,
            name:
            reorder_for_hierarchy: an optional argument that will ensure the resulting index is arranged in a tree-like structure.
            index_constructors:
            depth_reference:
            continuation_token: a Hashable that will be used as a token to identify when a value in a label should use the previously encountered value at the same depth.

        Returns:
            :obj:`IndexHierarchy`
        '''
        labels_iter = iter(labels)

        try:
            label_row = next(labels_iter)
        except StopIteration:
            labels_are_empty = True
        else:
            labels_are_empty = False

        if labels_are_empty:
            return cls._from_empty(labels, name=name, depth_reference=depth_reference)

        depth = len(label_row)
        if depth == 1:
            raise ErrorInitIndex('Cannot create IndexHierarchy from only one level.')

        # A mapping for each depth level, of label to index
        hash_maps: HashableToIntMapsT = [{} for _ in range(depth)]
        indexers: GrowableIndexersT = [[] for _ in range(depth)]

        prev_row: tp.Sequence[tp.Hashable] = ()

        while True:
            for hash_map, indexer, val in zip(hash_maps, indexers, label_row):
                # The equality check is heavy, so we short circuit when possible on an `is` check
                if (
                    continuation_token is not CONTINUATION_TOKEN_INACTIVE
                    and val == continuation_token
                ):
                    if prev_row:
                        i = indexer[-1] # Repeat the last observed index
                    else:
                        # This is the first row!
                        i = 0
                        hash_map[val] = 0
                elif val not in hash_map:
                    i = len(hash_map)
                    hash_map[val] = i
                else:
                    i = hash_map[val]

                indexer.append(i)

            prev_row = label_row
            try:
                label_row = next(labels_iter)
            except StopIteration:
                break

            if len(label_row) != depth:
                raise ErrorInitIndex('All labels must have the same depth.')

        # Convert to numpy array
        indexers = np.array(indexers, dtype=DTYPE_INT_DEFAULT)

        if reorder_for_hierarchy:
            # The innermost level (i.e. [:-1]) is irrelavant to lexsorting
            # We sort lexsort from right to left (i.e. [::-1])
            sort_order = np.lexsort(indexers[:-1][::-1])
            indexers = indexers[:, sort_order] # type: ignore

        indexers.flags.writeable = False # type: ignore

        index_constructors_iter = cls._build_index_constructors(
                index_constructors=index_constructors,
                depth=depth,
                )
        indices = [constructor(hash_map)
                for constructor, hash_map in zip(index_constructors_iter, hash_maps)
                ]

        if name is None:
            name = cls._build_name_from_indices(indices)

        return cls(
                indices=indices,
                indexers=indexers,
                name=name,
                )

    @classmethod
    def _from_index_items_1d(cls: tp.Type[IH],
            items: tp.Iterable[tp.Tuple[tp.Hashable, Index]],
            *,
            index_constructor: tp.Optional[IndexConstructor] = None,
            name: NameType = None,
            ) -> IH:
        labels: tp.List[tp.Hashable] = []
        index_inner: tp.Optional[IndexGO] = None  # We will grow this in-place
        indexers_inner: tp.List[np.ndarray] = []

        # Contains the len of the index for each label. Used to generate the outermost indexer
        repeats: tp.List[int] = []

        for label, index in items:
            if index.depth != 1:
                raise ErrorInitIndex("All indices must have the same shape.")

            labels.append(label)

            if index_inner is None:
                # This is the first index. Convert to IndexGO
                index_inner = tp.cast(IndexGO, mutable_immutable_index_filter(False, index))
                new_indexer = PositionsAllocator.get(len(index_inner))
            else:
                new_labels = index.difference(index_inner) # Retains order!

                if new_labels.size:
                    index_inner.extend(new_labels.values)

                new_indexer = index._index_iloc_map(index_inner)

            indexers_inner.append(new_indexer)
            repeats.append(len(index))

        index_inner = mutable_immutable_index_filter(cls.STATIC, index_inner) # type: ignore

        indexers: np.ndarray = np.array(
                [
                    np.hstack([np.repeat(val, repeats=reps) for val, reps in enumerate(repeats)]),
                    np.hstack(indexers_inner),
                ],
                dtype=DTYPE_INT_DEFAULT,
        )
        indexers.flags.writeable = False

        index_outer = index_from_optional_constructor(
                labels,
                default_constructor=cls._INDEX_CONSTRUCTOR,
                explicit_constructor=index_constructor,
                )

        return cls(
                indices=[index_outer, index_inner], # type: ignore
                indexers=indexers,
                name=name,
                )

    @classmethod
    def from_index_items(cls: tp.Type[IH],
            items: tp.Iterable[tp.Tuple[tp.Hashable, IndexBase]],
            *,
            index_constructor: IndexConstructor = None,
            name: NameType = None,
            ) -> IH:
        '''
        Given an iterable of pairs of label, :obj:`IndexBase`, produce an :obj:`IndexHierarchy` where the labels are depth 0, the indices are depth 1. While the provided :obj:`IndexBase` can be `Index` or `IndexHierarchy`, across all pairs all depths must be the same.

        Args:
            items: iterable of pairs of label, :obj:`IndexBase`.
            index_constructor: Optionally provide index constructor for outermost index.
        '''
        items = iter(items)
        try:
            label, index = next(items)
        except StopIteration:
            return cls._from_empty((), name=name, depth_reference=2)

        if index.depth == 1:
            return cls._from_index_items_1d(
                    itertools.chain([(label, index)], items), # type: ignore
                    index_constructor=index_constructor,
                    name=name,
                    )

        assert isinstance(index, IndexHierarchy) # mypy

        depth = index.depth
        labels: tp.List[tp.Hashable] = [label]
        repeats: tp.List[int] = [len(index)]
        existing_index_constructors: tp.List[IndexConstructor] = list(index._index_constructors)

        blocks = [index._blocks]
        for label, index in items:
            if index.depth != depth:
                raise ErrorInitIndex("All indices must have the same shape.")

            labels.append(label)
            repeats.append(len(index))
            blocks.append(index._blocks)  # type: ignore

            # If the IndexConstructor differs for any level, downcast to the
            # default constructor.
            for i, constructor in enumerate(index._index_constructors):  # type: ignore
                if constructor != existing_index_constructors[i]:
                    existing_index_constructors[i] = cls._INDEX_CONSTRUCTOR

        outer_level, _ = iterable_to_array_1d(
            itertools.chain.from_iterable(
                (
                    itertools.repeat(val, times=reps)
                    for val, reps in zip(labels, repeats)
                )
            ),
            count=sum(repeats),
        )
        outer_level.flags.writeable = False
        assert len(outer_level) == sum(map(len, blocks)) # sanity check

        def gen_blocks() -> tp.Iterable[np.ndarray]:
            yield outer_level
            yield from TypeBlocks.vstack_blocks_to_blocks(blocks)

        tb = TypeBlocks.from_blocks(gen_blocks())

        _, depth = tb.shape

        def gen_columns() -> tp.Iterator[np.ndarray]:
            for i in range(depth):
                yield tb._extract_array_column(i)

        if index_constructor is None:
            index_constructor = cls._INDEX_CONSTRUCTOR

        index_constructors_iter = cls._build_index_constructors(
                index_constructors=(index_constructor, *existing_index_constructors),
                depth=depth,
                )

        indices, indexers = construct_indices_and_indexers_from_column_arrays(
                column_iter=gen_columns(),
                index_constructors_iter=index_constructors_iter,
                )

        return cls(indices=indices, indexers=indexers, name=name, blocks=tb, own_blocks=True)

    @classmethod
    def from_labels_delimited(cls: tp.Type[IH],
            labels: tp.Iterable[str],
            *,
            delimiter: str = ' ',
            name: NameType = None,
            index_constructors: IndexConstructors = None,
            ) -> IH:
        '''
        Construct an :obj:`IndexHierarchy` from an iterable of labels, where each label is string defining the component labels for all hierarchies using a string delimiter. All components after splitting the string by the delimited will be literal evaled to produce proper types; thus, strings must be quoted.

        Args:
            labels: an iterator or generator of tuples.

        Returns:
            :obj:`static_frame.IndexHierarchy`
        '''
        def to_label(label: str) -> SingleLabelType:

            start, stop = None, None
            if label[0] in ('[', '('):
                start = 1
            if label[-1] in (']', ')'):
                stop = -1

            if start is not None or stop is not None:
                label = label[start: stop]

            parts = label.split(delimiter)
            if len(parts) <= 1:
                raise RuntimeError(
                    f'Could not not parse more than one label from delimited string: {label}')
            try:
                return tuple(literal_eval(p) for p in parts)
            except ValueError as e:
                raise ValueError(
                    'A label is malformed. This may be due to not quoting a string label'
                ) from e

        return cls.from_labels(
                (to_label(label) for label in labels),
                name=name,
                index_constructors=index_constructors
                )

    @classmethod
    def from_names(cls: tp.Type[IH],
            names: tp.Iterable[tp.Hashable],
            ) -> IH:
        '''
        Construct a zero-length :obj:`IndexHierarchy` from an iterable of ``names``, where the length of ``names`` defines the zero-length depth.

        Args:
            names: Iterable of hashable names per depth.
        '''
        # NOTE: this might take dtypes and/or IndexConstructors.
        name = tuple(names)
        if len(name) == 0:
            raise ErrorInitIndex('names must be non-empty.')

        return cls._from_empty((), name=name, depth_reference=len(name))

    @classmethod
    def _from_type_blocks(cls: tp.Type[IH],
            blocks: TypeBlocks,
            *,
            name: NameType = None,
            index_constructors: IndexConstructors = None,
            own_blocks: bool = False,
            name_interleave: bool = False,
            ) -> IH:
        '''
        Construct an :obj:`IndexHierarchy` from a :obj:`TypeBlocks` instance.

        Args:
            blocks: a TypeBlocks
            name_interleave: if True, merge names via index_constructors and the name argument.

        Returns:
            :obj:`IndexHierarchy`
        '''
        size, depth = blocks.shape
        if depth == 1:
            raise ErrorInitIndex('blocks must have at least two dimensions.')

        def gen_columns() -> tp.Iterator[np.ndarray]:
            for i in range(blocks.shape[1]):
                yield blocks._extract_array_column(i).reshape(size)

        index_constructors_iter = cls._build_index_constructors(
                index_constructors=index_constructors,
                depth=blocks.shape[1],
                )

        indices, indexers = construct_indices_and_indexers_from_column_arrays(
                column_iter=gen_columns(),
                index_constructors_iter=index_constructors_iter,
                )

        if not name_interleave and name is None:
            name = cls._build_name_from_indices(indices)
            # else, use passed name
        elif name_interleave:
            # NOTE: we always expect name to be a tuple when name_priorty is False as this pathway is exclusively from Frame.set_index_hierarchy()
            assert isinstance(name, tuple) and len(name) == len(indices)
            def gen() -> tp.Iterator[tp.Hashable]:
                for index, n in zip(indices, name): #type: ignore
                    if index.name is not None:
                        yield index.name
                    else:
                        yield n
            name = tuple(gen())

        init_blocks: tp.Optional[TypeBlocks] = blocks

        if index_constructors is not None:
            # If defined, we may have changed columnar dtypes in IndexLevels, and cannot reuse blocks
            if tuple(blocks.dtypes) != tuple(index.dtype for index in indices):
                init_blocks = None
                own_blocks = False

        return cls(
                indices=indices,
                indexers=indexers,
                name=name,
                blocks=init_blocks,
                own_blocks=own_blocks,
                )

    # --------------------------------------------------------------------------
    def _to_type_blocks(self: IH) -> TypeBlocks:
        '''
        Create a :obj:`TypeBlocks` instance from values respresented by `self._indices` and `self._indexers`.
        '''
        def gen_blocks() -> tp.Iterator[np.ndarray]:
            for index, indexer in zip(self._indices, self._indexers):
                array = index.values[indexer]
                array.flags.writeable = False
                yield array

        return TypeBlocks.from_blocks(gen_blocks())

    # --------------------------------------------------------------------------
    def __init__(self: IH,
            indices: tp.Union[IH, tp.List[Index]],
            *,
            indexers: np.ndarray = EMPTY_ARRAY_INT,
            name: NameType = NAME_DEFAULT,
            blocks: tp.Optional[TypeBlocks] = None,
            own_blocks: bool = False,
            ) -> None:
        '''
        Initializer.

        Args:
            indices: list of :obj:`Index` objects
            indexers: a 2D indexer array
            name: name of the IndexHierarchy
            blocks:
            own_blocks:
        '''
        self._recache = False
        self._index_types = None
        self._pending_extensions = None

        if isinstance(indices, IndexHierarchy):
            if indexers is not EMPTY_ARRAY_INT:
                raise ErrorInitIndex(
                    'indexers must not be provided when copying an IndexHierarchy'
                )
            if blocks is not None:
                raise ErrorInitIndex(
                    'blocks must not be provided when copying an IndexHierarchy'
                )

            if indices._recache:
                indices._update_array_cache()

            self._indices = [
                mutable_immutable_index_filter(self.STATIC, index)
                for index in indices._indices
                ]

            self._indexers = indices._indexers
            self._name = name if name is not NAME_DEFAULT else indices._name
            self._blocks = indices._blocks.copy()
            self._values = indices._values
            self._map = indices._map
            self._recache = False
            return

        if not (indexers.__class__ is np.ndarray and not indexers.flags.writeable):
            raise ErrorInitIndex('indexers must be a read-only numpy array.')

        if not all(isinstance(index, Index) for index in indices):
            raise ErrorInitIndex("indices must all Index's!")

        if len(indices) <= 1:
            raise ErrorInitIndex('Index Hierarchies must have at least two levels!')

        self._indices = [
            mutable_immutable_index_filter(self.STATIC, index) # type: ignore
            for index in indices
            ]
        self._indexers = indexers
        self._name = None if name is NAME_DEFAULT else name_filter(name)

        if blocks is None:
            self._blocks = self._to_type_blocks()
        elif own_blocks:
            self._blocks = blocks
        else:
            self._blocks = blocks.copy()

        self._values = None
        self._map = HierarchicalLocMap(indices=self._indices, indexers=self._indexers)

    def _update_array_cache(self: IH) -> None:
        # This MUST be set before entering this context
        assert self._pending_extensions is not None

        new_indexers = np.empty((self.depth, self.__len__()),
                dtype=DTYPE_INT_DEFAULT)
        current_size = len(self._blocks)

        for depth, indexer in enumerate(self._indexers):
            new_indexers[depth, :current_size] = indexer

        self._indexers = EMPTY_ARRAY_INT # Remove reference to old indexers

        offset = current_size
        # For all these extensions, we have already update self._indices - we now need to map indexers
        for pending in self._pending_extensions: # pylint: disable = E1133
            if pending.__class__ is PendingRow: # type: ignore
                for depth, label_at_depth in enumerate(pending):
                    label_index = self._indices[depth]._loc_to_iloc(label_at_depth)
                    new_indexers[depth, offset] = label_index

                offset += 1
            else:
                group_size = len(pending)

                for depth, (self_index, other_index) in enumerate(
                        zip(self._indices, pending._indices) # type: ignore
                    ):
                    remapped_indexers_unordered = other_index._index_iloc_map(self_index)
                    remapped_indexers_ordered = remapped_indexers_unordered[
                        pending._indexers[depth] # type: ignore
                    ]

                    new_indexers[depth, offset: offset + group_size] = remapped_indexers_ordered

                offset += group_size

        new_indexers.flags.writeable = False

        self._pending_extensions.clear()
        self._indexers = new_indexers
        self._blocks = self._to_type_blocks()
        self._values = None
        self._map = HierarchicalLocMap(indices=self._indices, indexers=self._indexers)
        self._recache = False

    # --------------------------------------------------------------------------

    def __setstate__(self, state: tp.Tuple[None, tp.Dict[str, tp.Any]]) -> None:
        '''
        Ensure that reanimated NP arrays are set not writeable.
        '''
        for key, value in state[1].items():
            setattr(self, key, value)

        self._indexers.flags.writeable = False
        if self._values is not None:
            self._values.flags.writeable = False

    def __deepcopy__(self: IH,
            memo: tp.Dict[int, tp.Any],
            ) -> IH:
        '''
        Return a deep copy of this IndexHierarchy.
        '''
        if self._recache:
            self._update_array_cache()

        obj: IH = self.__class__.__new__(self.__class__)
        obj._indices = deepcopy(self._indices, memo)
        obj._indexers = array_deepcopy(self._indexers, memo)
        obj._blocks = self._blocks.__deepcopy__(memo)
        obj._values = None
        obj._name = self._name # should be hashable/immutable
        obj._recache = False
        obj._index_types = deepcopy(self._index_types, memo)
        obj._pending_extensions = [] # this must be an empty list after recache
        obj._map = self._map.__deepcopy__(memo)

        memo[id(self)] = obj
        return obj

    def __copy__(self: IH) -> IH:
        '''
        Return a shallow copy of this IndexHierarchy.
        '''
        if self._recache:
            self._update_array_cache()

        return self.__class__(self)

    def copy(self: IH) -> IH:
        '''
        Return a shallow copy of this IndexHierarchy.
        '''
        return self.__copy__()

    def _memory_label_component_pairs(self,
            ) -> tp.Iterable[tp.Tuple[str, tp.Any]]:
        return (('Name', self._name),
                ('Indices', self._indices),
                ('Indexers', self._indexers),
                ('Blocks', self._blocks),
                ('Values', self._values),
                )

    # --------------------------------------------------------------------------
    # name interface

    def rename(self: IH,
            name: NameType,
            ) -> IH:
        '''
        Return a new IndexHierarchy with an updated name attribute.
        '''
        return self.__class__(self, name=name)

    # --------------------------------------------------------------------------
    # interfaces

    @property
    def loc(self: IH) -> InterfaceGetItem['IndexHierarchy']:
        return InterfaceGetItem(self._extract_loc)

    @property
    def iloc(self: IH) -> InterfaceGetItem['IndexHierarchy']:
        return InterfaceGetItem(self._extract_iloc)

    def _iter_label(self: IH,
            depth_level: tp.Optional[DepthLevelSpecifier] = None,
            ) -> tp.Iterator[tp.Hashable]:
        '''
        Iterate over labels at a given depth level.

        For multiple depth levels, the iterator will yield tuples of labels.
        '''
        if depth_level is None: # default to full labels
            depth_level = list(range(self.depth))
        else:
            validate_depth_selection(depth_level)

        if isinstance(depth_level, INT_TYPES):
            yield from self.values_at_depth(depth_level)
        else:
            yield from zip(*map(self.values_at_depth, depth_level))

    def _iter_label_items(self: IH,
            depth_level: tp.Optional[DepthLevelSpecifier] = None,
            ) -> tp.Iterator[tp.Tuple[int, tp.Hashable]]:
        '''
        This function is not directly called in iter_label or related routines, fulfills the expectations of the IterNodeDepthLevel interface.
        '''
        yield from enumerate(self._iter_label(depth_level=depth_level))

    @property
    def iter_label(self: IH) -> IterNodeDepthLevel[tp.Any]:
        return IterNodeDepthLevel(
                container=self,
                function_items=self._iter_label_items,
                function_values=self._iter_label,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.INDEX_LABELS
                )

    @property
    @doc_inject(select='astype')
    def astype(self: IH) -> InterfaceAsType[TContainer]:
        '''
        Retype one or more depths. Can be used as as function to retype the entire ``IndexHierarchy``; alternatively, a ``__getitem__`` interface permits retyping selected depths.

        Args:
            {dtype}
        '''
        return InterfaceAsType(func_getitem=self._extract_getitem_astype)

    # --------------------------------------------------------------------------
    @property
    def via_values(self) -> InterfaceValues[np.ndarray]:
        '''
        Interface for applying functions to values (as arrays) in this container.
        '''
        if self._recache:
            self._update_array_cache()

        return InterfaceValues(self)

    @property
    def via_str(self: IH) -> InterfaceString[np.ndarray]:
        '''
        Interface for applying string methods to elements in this container.
        '''
        if self._recache:
            self._update_array_cache()

        return InterfaceString(
                blocks=self._blocks._blocks,
                blocks_to_container=partial(blocks_to_array_2d, shape=self._blocks.shape),
                ndim=self._NDIM,
                labels=range(self.depth)
                )

    @property
    def via_dt(self: IH) -> InterfaceDatetime[np.ndarray]:
        '''
        Interface for applying datetime properties and methods to elements in this container.
        '''
        if self._recache:
            self._update_array_cache()

        return InterfaceDatetime(
                blocks=self._blocks._blocks,
                blocks_to_container=partial(blocks_to_array_2d, shape=self._blocks.shape),
                )

    @property
    def via_T(self: IH) -> InterfaceTranspose['IndexHierarchy']:
        '''
        Interface for using binary operators with one-dimensional sequences, where the opperand is applied column-wise.
        '''
        return InterfaceTranspose(container=self)

    def via_re(self: IH,
            pattern: str,
            flags: int = 0,
            ) -> InterfaceRe[np.ndarray]:
        '''
        Interface for applying regular expressions to elements in this container.
        '''
        if self._recache:
            self._update_array_cache()

        return InterfaceRe(
                blocks=self._blocks._blocks,
                blocks_to_container=partial(blocks_to_array_2d, shape=self._blocks.shape),
                pattern=pattern,
                flags=flags,
                )

    # --------------------------------------------------------------------------
    @property
    @doc_inject()
    def mloc(self: IH) -> np.ndarray:
        '''
        {doc_int}
        '''
        if self._recache:
            self._update_array_cache()

        return self._blocks.mloc

    @property
    def dtypes(self: IH) -> 'Series':
        '''
        Return a Series of dytpes for each index depth.

        Returns:
            :obj:`static_frame.Series`
        '''
        from static_frame.core.series import Series

        if (
            self._name
            and isinstance(self._name, tuple)
            and len(self._name) == self.depth
        ):
            labels: NameType = self._name
        else:
            labels = None

        return Series((index.dtype for index in self._indices), index=labels)

    @property
    def shape(self: IH) -> tp.Tuple[int, ...]:
        '''
        Return a tuple describing the shape of the underlying NumPy array.

        Returns:
            :obj:`tp.Tuple[int]`
        '''
        if self._recache:
            return self.__len__(), self.depth

        return self._blocks._shape

    @property
    def ndim(self: IH) -> int:
        '''
        Return the number of dimensions.

        Returns:
            :obj:`int`
        '''
        return self._NDIM

    @property
    def size(self: IH) -> int:
        '''
        Return the size of the underlying NumPy array.

        Returns:
            :obj:`int`
        '''
        if self._recache:
            self._update_array_cache()

        return self._blocks.size

    @property
    def nbytes(self: IH) -> int:
        '''
        Return the total bytes of the underlying NumPy array.

        Returns:
            :obj:`int`
        '''
        if self._recache:
            self._update_array_cache()

        total: int = sum(map(_NBYTES_GETTER, self._indices))
        total += sum(map(_NBYTES_GETTER, self._indexers))
        total += self._blocks.nbytes
        total += self._map.nbytes
        return total

    # --------------------------------------------------------------------------
    def __len__(self: IH) -> int:
        if self._recache:
            size = self._blocks.__len__()
            size += sum(map(len, self._pending_extensions))
            return size

        return self._blocks.__len__()

    @doc_inject()
    def display(self: IH,
            config: tp.Optional[DisplayConfig] = None,
            *,
            style_config: tp.Optional[StyleConfig] = None,
            ) -> Display:
        '''
        {doc}

        Args:
            {config}
        '''
        if self._recache:
            self._update_array_cache()

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

        return sub_display # type: ignore

    # --------------------------------------------------------------------------
    @property
    def _index_constructors(self: IH) -> tp.Iterator[tp.Type[Index]]:
        '''
        Yields the index constructors for each depth.
        '''
        yield from (index.__class__ for index in self._indices)

    def _drop_iloc(self: IH,
            key: GetItemKeyType,
            ) -> IH:
        '''
        Create a new index after removing the values specified by the iloc key.
        '''
        if self._recache:
            self._update_array_cache()

        blocks = TypeBlocks.from_blocks(self._blocks._drop_blocks(row_key=key))

        return self.__class__._from_type_blocks(
                blocks=blocks,
                index_constructors=self._index_constructors,
                name=self._name,
                own_blocks=True,
                )

    def _drop_loc(self: IH,
            key: GetItemKeyType,
            ) -> IH:
        '''
        Create a new index after removing the values specified by the loc key.
        '''
        return self._drop_iloc(self._loc_to_iloc(key))

    # --------------------------------------------------------------------------

    @property
    @doc_inject(selector='values_2d', class_name='IndexHierarchy')
    def values(self: IH) -> np.ndarray:
        '''
        {}
        '''
        if self._recache:
            self._update_array_cache()

        if self._values is None:
            self._values = self._blocks.values

        return self._values

    @property
    def positions(self: IH) -> np.ndarray:
        '''
        Return the immutable positions array.
        '''
        return PositionsAllocator.get(self.__len__())

    @property
    def depth(self: IH) -> int: # type: ignore
        '''
        Return the depth of the index hierarchy.
        '''
        return len(self._indices)

    def values_at_depth(self: IH,
            depth_level: DepthLevelSpecifier = 0,
            ) -> np.ndarray:
        '''
        Return an NP array for the ``depth_level`` specified.

        Args:
            depth_level: a single depth level, or iterable depth of depth levels.
        '''
        if self._recache:
            self._update_array_cache()

        validate_depth_selection(depth_level)

        if isinstance(depth_level, INT_TYPES):
            return self._blocks._extract_array_column(depth_level)
        return self._blocks._extract_array(column_key=list(depth_level))

    def index_at_depth(self: IH,
            depth_level: DepthLevelSpecifier = 0,
            ) -> tp.Union[Index, tp.Tuple[Index, ...]]:
        '''
        Return an index, or a tuple of indices for the ``depth_level`` specified.

        Args:
            depth_level: a single depth level, or iterable depth of depth levels.
        '''
        if self._recache:
            self._update_array_cache()

        validate_depth_selection(depth_level)

        if isinstance(depth_level, INT_TYPES):
            return self._indices[depth_level]

        return tuple(map(self._indices.__getitem__, depth_level))

    def indexer_at_depth(self: IH,
            depth_level: DepthLevelSpecifier = 0,
            ) -> np.ndarray:
        '''
        Return the indexers for the ``depth_level`` specified.
        Array will 2D if multiple depths are selected.

        Args:
            depth_level: a single depth level, or iterable depth of depth levels.
        '''
        if self._recache:
            self._update_array_cache()

        validate_depth_selection(depth_level)

        return self._indexers[depth_level]


    @doc_inject()
    def label_widths_at_depth(self: IH,
            depth_level: DepthLevelSpecifier = 0
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, int]]:
        '''
        {}
        '''
        if depth_level is None:
            raise NotImplementedError('depth_level of None is not supported')

        validate_depth_selection(depth_level)

        pos: tp.Optional[int] = None
        if not isinstance(depth_level, INT_TYPES):
            sel = list(depth_level)
            if len(sel) == 1:
                pos = sel.pop()
        else: # is an int
            pos = depth_level

        if pos is None:
            raise NotImplementedError(
                'selecting multiple depth levels is not yet implemented'
                )

        if self._recache:
            self._update_array_cache()
        indexer = self._indexers[pos]
        index = self._indices[pos]

        ilocs, widths = run_length_1d(indexer)

        yield from zip(map(index._extract_iloc_by_int, ilocs), widths)



    @property
    def index_types(self: IH) -> 'Series':
        '''
        Return a Series of Index classes for each index depth.

        Returns:
            :obj:`Series`
        '''
        if self._index_types is None:
            # Prefer to use `@functools.cached_property`, but that is not supported
            from static_frame.core.series import Series

            labels: NameType

            if (self._name
                    and isinstance(self._name, tuple)
                    and len(self._name) == self.depth
                    ):
                labels = self._name
            else:
                labels = None

            self._index_types = Series(
                    self._index_constructors,
                    index=labels,
                    dtype=DTYPE_OBJECT,
                    )

        return self._index_types

    # --------------------------------------------------------------------------
    def relabel(self: IH,
            mapper: RelabelInput,
            ) -> IH:
        '''
        Return a new IndexHierarchy with labels replaced by the callable or mapping; order will be retained. If a mapping is used, the mapping should map tuple representation of labels, and need not map all origin keys.
        '''
        if self._recache:
            self._update_array_cache()

        if not callable(mapper):
            # if a mapper, it must support both __getitem__ and __contains__
            getitem = getattr(mapper, 'get')

            def gen() -> tp.Iterator[SingleLabelType]:
                for array in self._blocks.axis_values(axis=1):
                    # as np.ndarray are not hashable, must tuplize
                    label = tuple(array)
                    yield getitem(label, label)

            return self.__class__.from_labels(
                    labels=gen(),
                    name=self._name,
                    index_constructors=self._index_constructors,
                    )

        return self.__class__.from_labels(
                (mapper(x) for x in self._blocks.axis_values(axis=1)),
                name=self._name,
                index_constructors=self._index_constructors,
                )

    def relabel_at_depth(self: IH,
            mapper: RelabelInput,
            depth_level: DepthLevelSpecifier = 0,
            ) -> IH:
        '''
        Return a new :obj:`IndexHierarchy` after applying `mapper` to a level or each individual level specified by `depth_level`.

        `mapper` can be a callable, mapping, or iterable.
            - If a callable, it must accept a single hashable, and return a single hashable.
            - If a mapping, it must map a single hashable to a single hashable.
            - If a iterable, it must be the same length as `self`.

        This call:

        >>> index.relabel_at_depth(mapper, depth_level=[0, 1, 2])

        is equivalent to:

        >>> for level in [0, 1, 2]:
        >>>     index = index.relabel_at_depth(mapper, depth_level=level)

        albeit more efficient.
        '''
        validate_depth_selection(depth_level)

        if isinstance(depth_level, INT_TYPES):
            depth_level = [depth_level]
            target_depths = set(depth_level)
        else:
            depth_level = sorted(depth_level)
            target_depths = set(depth_level)

            if len(target_depths) != len(depth_level):
                raise ValueError('depth_levels must be unique')

            if not depth_level:
                raise ValueError('depth_level must be non-empty')

        if any(level < 0 or level >= self.depth for level in depth_level):
            raise ValueError(
                f'Invalid depth level found. Valid levels: [0-{self.depth - 1}]'
            )

        if self._recache:
            self._update_array_cache()

        is_callable = callable(mapper)

        # Special handling for full replacements
        if not is_callable and not hasattr(mapper, 'get'):
            values, _ = iterable_to_array_1d(mapper, count=self.__len__())

            if len(values) != self.__len__():
                raise ValueError('Iterable must provide a value for each label')

            def gen() -> tp.Iterator[np.ndarray]:
                for depth_idx in range(self.depth):
                    if depth_idx in target_depths:
                        yield values
                    else:
                        yield self._blocks._extract_array_column(depth_idx)

            return self.__class__._from_type_blocks(
                    TypeBlocks.from_blocks(gen()),
                    name=self._name,
                    index_constructors=self._index_constructors,
                    own_blocks=True,
                    )

        mapper_func = mapper if is_callable else mapper.__getitem__ # type: ignore

        def get_new_label(label: tp.Hashable) -> tp.Hashable:
            if is_callable or label in mapper: # type: ignore
                return mapper_func(label) # type: ignore
            return label

        # depth_level might not contain all depths, so we will re-use our
        # indices/indexers for all those cases.
        new_indices = list(self._indices)
        new_indexers = self._indexers.copy()

        for level in depth_level:
            index = self._indices[level]

            new_index: tp.Dict[tp.Hashable, int] = {}
            index_remap: tp.Dict[tp.Hashable, tp.Hashable] = {}

            for label_idx, label in enumerate(index.values):
                new_label = get_new_label(label)

                if new_label not in new_index:
                    new_index[new_label] = len(new_index)
                else:
                    index_remap[label_idx] = new_index[new_label]

            new_indices[level] = index.__class__(new_index)

            indexer = np.array([index_remap.get(i, i) for i in self._indexers[level]], dtype=DTYPE_INT_DEFAULT)
            new_indexers[level] = indexer

        new_indexers.flags.writeable = False

        return self.__class__(
                indices=new_indices,
                indexers=new_indexers,
                name=self._name,
                )

    def rehierarch(self: IH,
            depth_map: tp.Sequence[int],
            *,
            index_constructors: IndexConstructors = None,
            ) -> IH:
        '''
        Return a new :obj:`IndexHierarchy` that conforms to the new depth assignments given be `depth_map`.
        '''
        if self._recache:
            self._update_array_cache()

        rehierarched_blocks, _ = rehierarch_from_type_blocks(
                labels=self._blocks,
                depth_map=depth_map,
                )

        if index_constructors is None:
            # transform the existing index constructors correspondingly
            index_constructors = self.index_types.values[list(depth_map)]

        return self.__class__._from_type_blocks(
            blocks=rehierarched_blocks,
            index_constructors=index_constructors,
            own_blocks=True,
            )

    # --------------------------------------------------------------------------

    def _build_mask_for_key_at_depth(self: IH,
            depth: int,
            key: tp.Union[np.ndarray, CompoundLabelType],
            available: tp.Optional[np.ndarray],
            ) -> np.ndarray:
        '''
        Determines the indexer mask for `key` at `depth`.

        Args:
            available: Optional Boolean array denoting with True the subset of region to search for start / end positions of slice values. We only take slices within regions previously selected by already-processed depths. If None, some optimizations are available for working with slices.
        '''
        # This private internal method assumes recache has already been checked for!

        key_at_depth = key[depth]

        # Key is already a mask!
        if key_at_depth.__class__ is np.ndarray and key_at_depth.dtype == DTYPE_BOOL: # type: ignore
            return key_at_depth

        index_at_depth = self._indices[depth]
        indexer_at_depth = self._indexers[depth]

        if isinstance(key_at_depth, slice):
            if available is None:
                multi_depth = False
            else:
                multi_depth = True
                unmatchable = ~available

            if key_at_depth.start is not None:
                matched = indexer_at_depth == index_at_depth.loc_to_iloc(key_at_depth.start)
                if multi_depth:
                    matched[unmatchable] = False # set all regions unavailable to slice to False
                start = first_true_1d(matched, forward=True)
            else:
                start = 0

            if key_at_depth.step is not None and not isinstance(
                    key_at_depth.step, INT_TYPES
                    ):
                raise TypeError(
                    f'slice step must be an integer, not {type(key_at_depth.step)}'
                    )

            if key_at_depth.stop is not None:
                # get the last stop value observed
                matched = indexer_at_depth == index_at_depth.loc_to_iloc(key_at_depth.stop)
                if multi_depth:
                    matched[unmatchable] = False
                stop = first_true_1d(matched, forward=False)
                stop += 1
            else:
                stop = len(indexer_at_depth)

            target = np.arange(start, stop, key_at_depth.step)
            post = np.full(len(indexer_at_depth), False)
            post[target] = True
            return post

        key_iloc = index_at_depth.loc_to_iloc(key_at_depth)

        if hasattr(key_iloc, '__len__'):
            # Cases where the key is a list of labels
            return isin(indexer_at_depth, key_iloc)

        return indexer_at_depth == key_iloc

    def _loc_to_iloc_index_hierarchy(self: IH,
            key: IH,
            ) -> tp.List[int]:
        '''
        Returns the boolean mask for a key that is an IndexHierarchy.

        For small keys, this approach is outperformed by the naive:

            [self._loc_to_iloc(label) for label in key.iter_label()]

        However, this approach quickly outperforms list comprehension as the key gets larger
        '''
        # This private internal method assumes recache has already been checked for!
        if not key.depth == self.depth:
            raise KeyError(f'Key must have the same depth as the index. {key}')

        if key._recache:
            key._update_array_cache()

        remapped_indexers: tp.List[np.ndaray] = []
        for key_index, self_index, key_indexer in zip(
                key._indices,
                self._indices,
                key._indexers,
                ):
            indexer_remap = key_index._index_iloc_map(self_index)
            remapped_indexers.append(indexer_remap[key_indexer])

        remapped_indexers = np.array(remapped_indexers, dtype=DTYPE_UINT_DEFAULT).T

        try:
            return self._map.indexers_to_iloc(remapped_indexers)
        except KeyError:
            # Display the first missing element
            raise KeyError(key.difference(self)[0]) from None

    def _loc_per_depth_to_iloc(self: IH,
            key: tp.Union[np.ndarray, CompoundLabelType],
            ) -> tp.Union[int, np.ndarray]:
        '''
        Return the indexer for a given key. Key is assumed to not be compound (i.e. HLoc, list of keys, etc)

        Will return a single integer for single, non-HLoc keys. Otherwise, returns a boolean mask.
        '''
        # This private internal method assumes recache has already been checked for!
        # We consider the NULL_SLICE to not be 'meaningful', as it requires no filtering
        meaningful_depths = [
                depth for depth, k in enumerate(key)
                if not (k.__class__ is slice and k == NULL_SLICE)
                ]

        if len(meaningful_depths) == 1:
            # Prefer to avoid construction of a 2D mask
            mask = self._build_mask_for_key_at_depth(
                    depth=meaningful_depths[0],
                    key=key,
                    available=None,
                    )
        else:
            if (len(meaningful_depths) == self.depth and all(map(is_neither_slice_nor_mask, key))):
                try:
                    return self._map.loc_to_iloc(key, self._indices)
                except KeyError:
                    raise KeyError(key) from None

            mask = np.full(self._indexers.shape[1], True, dtype=DTYPE_BOOL)

            for depth in meaningful_depths:
                mask &= self._build_mask_for_key_at_depth(
                        depth=depth,
                        key=key,
                        available=mask,
                        )

        return self.positions[mask]

    def _loc_to_iloc(self: IH,
            key: LocKeyType,
            ) -> IntegerLocType:
        '''
        Given iterable (or instance) of GetItemKeyType, determine the equivalent iloc key.

        When possible, prefer slice or single elements
        '''
        if key.__class__ is ILoc:
            return key.key # type: ignore

        if self._recache:
            self._update_array_cache()

        if isinstance(key, IndexHierarchy):
            return self._loc_to_iloc_index_hierarchy(key)

        if key.__class__ is np.ndarray and key.dtype == DTYPE_BOOL: # type: ignore
            return self.positions[key]

        if isinstance(key, slice):
            return slice(*LocMap.map_slice_args(self._loc_to_iloc, key))

        if isinstance(key, list):
            return [self._loc_to_iloc(k) for k in key]

        if key.__class__ is np.ndarray and key.ndim == 2: # type: ignore
            if key.dtype != DTYPE_OBJECT: # type: ignore
                return np.intersect1d(
                        view_2d_as_1d(self.values),
                        view_2d_as_1d(key),
                        assume_unique=False,
                        return_indices=True,
                        )[1]
            return [self._loc_to_iloc(k) for k in key] # type: ignore

        if key.__class__ is HLoc:
            # unpack any Series, Index, or ILoc into the context of this IndexHierarchy
            key = tuple(
                    key_from_container_key(self, k, True) for k in key # type: ignore
                    )
            if len(key) > self.depth:
                raise RuntimeError(
                    f'Too many depths specified for {key}. Expected: {self.depth}'
                    )
        else:
            # If the key is a series, key_from_container_key will invoke IndexCorrespondence
            # logic that eventually calls _loc_to_iloc on all the indices of that series.
            sanitized_key = key_from_container_key(self, key)

            if key is sanitized_key:
                # This is always either a tuple, or a 1D numpy array
                if len(key) != self.depth:
                    raise RuntimeError(
                        f'Invalid key length for {key}; must be length {self.depth}.'
                    )

                if any(isinstance(subkey, KEY_MULTIPLE_TYPES) for subkey in key): # type: ignore
                    raise RuntimeError(
                        f'slices cannot be used in a leaf selection into an IndexHierarchy; try HLoc[{key}].'
                    )
            else:
                key = sanitized_key
                if key.__class__ is np.ndarray and key.dtype == DTYPE_BOOL: # type: ignore
                    # When the key is a series with boolean values
                    return self.positions[key]

                key = tuple(key)

                for subkey in key:
                    if len(subkey) != self.depth:
                        raise RuntimeError(
                            f'Invalid key length for {subkey}; must be length {self.depth}.'
                        )
        if all(isinstance(k, tuple) for k in key): # type: ignore
            # We can occasionally receive a sequence of tuples
            return [self._loc_to_iloc(k) for k in key] # type: ignore

        return self._loc_per_depth_to_iloc(key)

    def loc_to_iloc(self: IH,
            key: LocKeyType,
            ) -> IntegerLocType:
        '''
        Given a label (loc) style key (either a label, a list of labels, a slice, an HLoc object, or a Boolean selection), return the index position (iloc) style key. Keys that are not found will raise a KeyError or a sf.LocInvalid error.

        Args:
            key: a label key.
        '''
        # NOTE: the public method is the same as the private method for IndexHierarchy, but not for Index
        return self._loc_to_iloc(key)

    def _extract_iloc(self: IH,
            key: IntegerLocType,
            ) -> ExtractionType:
        '''
        Extract a new index given an iloc key
        '''
        if self._recache:
            self._update_array_cache()

        if key is None:
            return self if self.STATIC else self.__deepcopy__({})

        if isinstance(key, INT_TYPES):
            # return a tuple if selecting a single row
            return tuple(self._blocks.iter_row_elements(key))

        tb = self._blocks._extract(row_key=key)
        if len(tb) == 0:
            return self.__class__._from_empty((),
                    name=self._name,
                    depth_reference=tb.shape[1],
                    index_constructors=self._index_constructors,
                    )

        new_indices: tp.List[Index] = []
        new_indexers: np.ndarray = np.empty((self.depth, len(tb)), dtype=DTYPE_INT_DEFAULT)

        for i, (index, indexer) in enumerate(zip(self._indices, self._indexers)):
            selection = indexer[key]
            if len(index) > len(selection):
                unique_indexes, new_indexer = ufunc_unique1d_indexer(selection)
            else:
                unique_indexes, new_indexer = get_new_indexers_and_screen(selection, index.positions)

            new_indices.append(index._extract_iloc(unique_indexes))
            new_indexers[i] = new_indexer

        new_indexers.flags.writeable = False

        return self.__class__(
                indices=new_indices,
                indexers=new_indexers,
                name=self._name,
                blocks=tb,
                own_blocks=True,
                )

    def _extract_iloc_by_int(self,
            key: int,
            ) -> tp.Tuple[tp.Hashable, ...]:
        '''Extract a single row as a tuple (without coercion) given an iloc integer key. This interface is overwhelmingly for compatibility with Index.
        '''
        if self._recache:
            self._update_array_cache()
        return tuple(self._blocks.iter_row_elements(key))

    def _extract_loc(self: IH,
            key: LocKeyType,
            ) -> ExtractionType:
        '''
        Extract a new index given an loc key
        '''
        return self._extract_iloc(self._loc_to_iloc(key))

    def __getitem__(self: IH,
            key: IntegerLocType,
            ) -> ExtractionType:
        '''
        Extract a new index given a key.
        '''
        return self._extract_iloc(key)

    # --------------------------------------------------------------------------

    def _extract_getitem_astype(self: IH,
            key: GetItemKeyType,
            ) -> 'IndexHierarchyAsType':
        '''
        Given an iloc key (using integer positions for columns) return a configured IndexHierarchyAsType instance.
        '''
        # key is an iloc key
        if isinstance(key, tuple):
            raise KeyError('__getitem__ does not support multiple indexers')
        validate_depth_selection(key)
        return IndexHierarchyAsType(self, key)

    # --------------------------------------------------------------------------
    # operators

    def _ufunc_unary_operator(self: IH,
            operator: UFunc,
            ) -> np.ndarray:
        '''
        Always return an NP array.
        '''
        array = operator(self.values)
        array.flags.writeable = False
        return array

    def _ufunc_binary_operator(self: IH,
            *,
            operator: UFunc,
            other: tp.Any,
            axis: int = 0,
            fill_value: object = np.nan,
            ) -> np.ndarray:
        '''
        Binary operators applied to an index always return an NP array. This deviates from Pandas, where some operations (multiplying an int index by an int) result in a new Index, while other operations result in a np.array (using == on two Index).
        '''
        from static_frame.core.frame import Frame
        from static_frame.core.series import Series

        if isinstance(other, (Series, Frame)):
            raise ValueError('cannot use labelled container as an operand.')

        if operator.__name__ == 'matmul':
            return matmul(self.values, other)
        elif operator.__name__ == 'rmatmul':
            return matmul(other, self.values)

        if self._recache:
            self._update_array_cache()

        if isinstance(other, Index):
            other = other.values
        elif isinstance(other, IndexHierarchy):
            if other._recache:
                other._update_array_cache()
            other = other._blocks

        tb = self._blocks._ufunc_binary_operator(
                    operator=operator,
                    other=other,
                    axis=axis,
                    )
        return tb.values

    def _ufunc_axis_skipna(self: IH,
            *,
            axis: int,
            skipna: bool,
            ufunc: UFunc,
            ufunc_skipna: UFunc,
            composable: bool,
            dtypes: tp.Tuple[np.dtype, ...],
            size_one_unity: bool,
            ) -> np.ndarray:
        '''
        Returns:
            immutable NumPy array.
        '''
        if self._recache:
            self._update_array_cache()

        if ufunc is np.max or ufunc is np.min:
            # max and min are treated per label; thus we do a lexical sort for axis 0
            if axis == 1:
                raise NotImplementedError(f'{ufunc} for {self.__class__.__name__} is not defined for axis {axis}.')

            # as we will be doing a lexicasl sort, must drop any label with a missing value
            order = sort_index_for_order(
                    self.dropna(condition=np.any) if skipna else self,
                    kind=DEFAULT_SORT_KIND,
                    ascending=True,
                    key=None,
                    )
            # if skipna, drop rows with any NaNs
            blocks = self._blocks._extract(row_key=order)
            # NOTE: this could return a tuple rather than an array
            return blocks._extract_array(row_key=(-1 if ufunc is np.max else 0))

        # NOTE: as min and max are by label, it is awkward that statistical functions are calculated as Frames, per depth level
        raise NotImplementedError(f'{ufunc} for {self.__class__.__name__} is not defined; convert to `Frame`.')

        # if not ufunc_is_statistical(ufunc):
        # return self._blocks.ufunc_axis_skipna(
        #         skipna=skipna,
        #         axis=axis,
        #         ufunc=ufunc,
        #         ufunc_skipna=ufunc_skipna,
        #         composable=composable,
        #         dtypes=dtypes,
        #         size_one_unity=size_one_unity,
        #         )

    def _ufunc_shape_skipna(self, *,
            axis: int,
            skipna: bool,
            ufunc: UFunc,
            ufunc_skipna: UFunc,
            composable: bool,
            dtypes: tp.Tuple[np.dtype, ...],
            size_one_unity: bool
            ) -> np.ndarray:
        '''
        As Index and IndexHierarchy return np.ndarray from such operations, _ufunc_shape_skipna and _ufunc_axis_skipna can be defined the same.

        Returns:
            immutable NumPy array.
        '''
        raise NotImplementedError(f'{ufunc} for {self.__class__.__name__} is not defined; convert to `Frame`.')

        # if self._recache:
        #     self._update_array_cache()

        # dtype = None if not dtypes else dtypes[0] # only a tuple
        # if skipna:
        #     post = ufunc_skipna(self.values, axis=axis, dtype=dtype)
        # else:
        #     post = ufunc(self.values, axis=axis, dtype=dtype)
        # post.flags.writeable = False
        # return post

    # --------------------------------------------------------------------------
    # dictionary-like interface

    # NOTE: we intentionally exclude keys(), items(), and get() from Index classes, as they return inconsistent result when thought of as a dictionary

    def __iter__(self: IH) -> tp.Iterator[SingleLabelType]:
        '''
        Iterate over labels.
        '''
        if self._recache:
            self._update_array_cache()
        # Don't use .values, as that can coerce types
        yield from self._blocks.iter_row_tuples(None)

    def __reversed__(self: IH) -> tp.Iterator[SingleLabelType]:
        '''
        Returns a reverse iterator on the index labels.
        '''
        if self._recache:
            self._update_array_cache()

        for array in self._blocks.axis_values(1, reverse=True):
            yield tuple(array)

    def __contains__(self: IH, # type: ignore
            value: SingleLabelType,
            ) -> bool:
        '''
        Determine if a label `value` is contained in this Index.
        '''
        try:
            result = self._loc_to_iloc(value)
        except KeyError:
            return False

        if hasattr(result, '__len__'):
            return bool(len(result))

        return True

    # --------------------------------------------------------------------------
    # utility functions

    def unique(self: IH,
            depth_level: DepthLevelSpecifier = 0,
            order_by_occurrence: bool = False,
            ) -> np.ndarray:
        '''
        Return a NumPy array of unique values.

        Args:
            depth_level: Specify a single depth or multiple depths in an iterable.
            order_by_occurrence: if True, values are ordered by when they first appear

        Returns:
            :obj:`numpy.ndarray`
        '''
        validate_depth_selection(depth_level)

        pos: tp.Optional[int] = None
        if not isinstance(depth_level, INT_TYPES):
            sel = list(depth_level)
            if len(sel) == 1:
                pos = sel.pop()
        else: # is an int
            pos = depth_level

        if pos is not None: # i.e. a single level
            if order_by_occurrence:
                # Index could be [A, B, C]
                # Indexers could be [2, 0, 0, 2, 1]
                # This function return [C, A, B] -- shoutout to my initials
                # get the outer level, or just the unique frame labels needed
                labels = self.values_at_depth(pos)
                label_indexes = ufunc_unique1d_positions(labels)[0]
                label_indexes.sort()
                return labels[label_indexes]

            return self._indices[pos].values

        if order_by_occurrence:
            raise NotImplementedError('order_by_occurrence not implemented for multiple depth levels.')

        return ufunc_unique(array2d_to_array1d(self.values_at_depth(sel)))

    @doc_inject()
    def equals(self: IH,
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

        if not isinstance(other, IndexHierarchy):
            return False

        # same type, depth from here
        if self.shape != other.shape:
            return False

        if compare_name and self.name != other.name:
            return False

        if compare_dtype and not self.dtypes.equals(other.dtypes):
            return False

        if compare_class:
            for self_index, other_index in zip(self._indices, other._indices):
                if self_index.__class__ != other_index.__class__:
                    return False

        if self._recache:
            self._update_array_cache()
        if other._recache:
            other._update_array_cache()

        return self._blocks.equals(other._blocks, # type: ignore
                compare_dtype=compare_dtype,
                compare_class=compare_class,
                skipna=skipna,
                )

    @doc_inject(selector='sort')
    def sort(self: IH,
            *,
            ascending: BoolOrBools = True,
            kind: str = DEFAULT_SORT_KIND,
            key: tp.Optional[tp.Callable[[IH], tp.Union[np.ndarray, IH]]] = None,
            ) -> IH:
        '''
        Return a new Index with the labels sorted.

        Args:
            {ascendings}
            {kind}
            {key}
        '''
        if self._recache:
            self._update_array_cache()

        order = sort_index_for_order(self, kind=kind, ascending=ascending, key=key)

        blocks = self._blocks._extract(row_key=order)
        indexers = self._indexers[:, order]
        indexers.flags.writeable = False

        return self.__class__(
                indices=self._indices, # will be copied with mutable_immutable_index_filter
                indexers=indexers,
                name=self._name,
                blocks=blocks,
                own_blocks=True,
                )

    def isin(self: IH,
            other: tp.Iterable[tp.Iterable[tp.Hashable]],
            ) -> np.ndarray:
        '''
        Return a Boolean array showing True where one or more of the passed in iterable of labels is found in the index.
        '''
        matches = []
        for seq in other:
            if not hasattr(seq, '__iter__'):
                raise RuntimeError(
                    'must provide one or more iterables within an iterable'
                )
            # Coerce to hashable type
            as_tuple = tuple(seq)
            if len(as_tuple) == self.depth:
                # can pre-filter if iterable matches to length
                matches.append(as_tuple)

        if not matches:
            return np.full(self.__len__(), False, dtype=bool)

        return isin(self.flat().values, matches)

    def roll(self: IH,
            shift: int,
            ) -> IH:
        '''
        Return an :obj:`IndexHierarchy` with values rotated forward and wrapped around (with a positive shift) or backward and wrapped around (with a negative shift).
        '''
        if self._recache:
            self._update_array_cache()

        blocks = TypeBlocks.from_blocks(
                self._blocks._shift_blocks_fill_by_element(row_shift=shift, wrap=True)
                )

        return self.__class__._from_type_blocks(
                blocks=blocks,
                index_constructors=self._index_constructors,
                name=self._name,
                own_blocks=True,
                )

    # --------------------------------------------------------------------------
    # utility functions

    def union(self: IH, *others: tp.Union[IH, tp.Iterable[tp.Hashable]]) -> IH:
        from static_frame.core.index_hierarchy_set_utils import index_hierarchy_union

        if all(isinstance(other, IndexHierarchy) for other in others):
            return index_hierarchy_union(self, *others) # type: ignore

        return IndexBase.union(self, *others)

    def intersection(self: IH, *others: tp.Union[IH, tp.Iterable[tp.Hashable]]) -> IH:
        from static_frame.core.index_hierarchy_set_utils import index_hierarchy_intersection

        if all(isinstance(other, IndexHierarchy) for other in others):
            return index_hierarchy_intersection(self, *others) # type: ignore

        return IndexBase.intersection(self, *others)

    def difference(self: IH, *others: tp.Union[IH, tp.Iterable[tp.Hashable]]) -> IH:
        from static_frame.core.index_hierarchy_set_utils import index_hierarchy_difference

        if all(isinstance(other, IndexHierarchy) for other in others):
            return index_hierarchy_difference(self, *others) # type: ignore

        return IndexBase.difference(self, *others)

    #---------------------------------------------------------------------------

    def _drop_missing(self,
            func: tp.Callable[[np.ndarray], np.ndarray],
            condition: tp.Callable[[np.ndarray], bool],
            ) -> IH:
        '''
        Return a new obj:`IndexHierarchy` after removing rows (axis 0) or columns (axis 1) where any or all values are NA (NaN or None). The condition is determined by  a NumPy ufunc that process the Boolean array returned by ``isna()``; the default is ``np.all``.

        Args:
            axis:
            condition:
        '''
        # returns Boolean areas that define axis to keep
        row_key, _ = self._blocks.drop_missing_to_keep_locations(
                axis=0, # always labels (rows) for IH
                condition=condition,
                func=func,
                )
        if self.STATIC and row_key.all(): #type: ignore
            return self #type: ignore

        return self._drop_iloc(~row_key) #type: ignore

    def dropna(self, *,
            condition: tp.Callable[[np.ndarray], bool] = np.all,
            ) -> IH:
        '''
        Return a new obj:`IndexHierarchy` after removing labels where any or all values are NA (NaN or None). The condition is determined by a NumPy ufunc that process the Boolean array returned by ``isna()``; the default is ``np.all``.

        Args:
            *,
            condition:
        '''
        return self._drop_missing(isna_array, condition)

    def dropfalsy(self, *,
            condition: tp.Callable[[np.ndarray], bool] = np.all,
            ) -> IH:
        '''
        Return a new obj:`IndexHierarchy` after removing labels where any or all values are falsy. The condition is determined by a NumPy ufunc that process the Boolean array returned by ``isna()``; the default is ``np.all``.

        Args:
            *,
            condition:
        '''
        return self._drop_missing(isfalsy_array, condition)

    #---------------------------------------------------------------------------

    @doc_inject(selector='fillna')
    def fillna(self: IH,
            value: tp.Any,
            ) -> IH:
        '''
        Return an :obj:`IndexHierarchy` after replacing NA (NaN or None) with the supplied value.

        Args:
            {value}
        '''
        if self._recache:
            self._update_array_cache()

        blocks = self._blocks.fill_missing_by_unit(value, None, func=isna_array)

        return self.__class__._from_type_blocks(
                blocks=blocks,
                index_constructors=self._index_constructors,
                name=self._name,
                own_blocks=True,
                )

    @doc_inject(selector='fillna')
    def fillfalsy(self: IH,
            value: tp.Any,
            ) -> IH:
        '''
        Return an :obj:`IndexHierarchy` after replacing falsy values with the supplied value.

        Args:
            {value}
        '''
        if self._recache:
            self._update_array_cache()

        blocks = self._blocks.fill_missing_by_unit(value, None, func=isfalsy_array)

        return self.__class__._from_type_blocks(
                blocks=blocks,
                index_constructors=self._index_constructors,
                name=self._name,
                own_blocks=True,
                )

    #---------------------------------------------------------------------------

    def _sample_and_key(self: IH,
            count: int = 1,
            *,
            seed: tp.Optional[int] = None,
            ) -> tp.Tuple[IH, np.ndarray]:
        '''
        Selects a deterministically random sample from this IndexHierarchy.

        Returns:
            The sampled IndexHierarchy
            An integer array of sampled iloc values
        '''
        if self._recache:
            self._update_array_cache()

        key = array_sample(self.positions, count=count, seed=seed, sort=True)
        blocks = self._blocks._extract(row_key=key)

        container = self.__class__._from_type_blocks(
                blocks=blocks,
                index_constructors=self._index_constructors,
                name=self._name,
                own_blocks=True,
                )
        return container, key

    @doc_inject(selector='searchsorted', label_type='iloc (integer)')
    def iloc_searchsorted(self: IH,
            values: tp.Any,
            *,
            side_left: bool = True,
            ) -> tp.Union[tp.Hashable, tp.Sequence[tp.Hashable]]:
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
            raise NotImplementedError(
                'A single label (as a tuple) or multiple labels (as a list) must be provided.'
            )

        dt_pos = np.fromiter(
            (
                issubclass(idx_type, IndexDatetime)
                for idx_type in self._index_constructors
            ),
            count=self.depth,
            dtype=DTYPE_BOOL,
        )
        has_dt = dt_pos.any()

        values_for_match = np.empty(len(match_pre), dtype=object)

        for i, label in enumerate(match_pre):
            if has_dt:
                label = tuple(
                    v if not dt_pos[j] else key_to_datetime_key(v)
                    for j, v in enumerate(label)
                )
            values_for_match[i] = label

        post: np.ndarray = self.flat().iloc_searchsorted(values_for_match, side_left=side_left)
        if is_element:
            return tp.cast(tp.Hashable, post[0])
        return tp.cast(tp.Sequence[tp.Hashable], post)

    @doc_inject(selector='searchsorted', label_type='loc (label)')
    def loc_searchsorted(self: IH,
            values: tp.Any,
            *,
            side_left: bool = True,
            fill_value: tp.Any = np.nan,
            ) -> tp.Union[tp.Hashable, tp.Iterable[tp.Hashable], tp.Any]:
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
            return fill_value

        flat = self.flat().values
        mask = sel == length
        if not mask.any():
            return flat[sel]

        post = np.empty(len(sel), dtype=object)
        sel[mask] = 0 # set out of range values to zero
        post[:] = flat[sel]
        post[mask] = fill_value
        post.flags.writeable = False
        return post

    # --------------------------------------------------------------------------
    # export

    def _to_frame(self: IH,
            constructor: tp.Type['Frame'],
            ) -> 'Frame':
        if self._recache:
            self._update_array_cache()

        return constructor(
                self._blocks.copy(),
                columns=None,
                index=None,
                own_data=True,
                )

    def to_frame(self: IH) -> 'Frame':
        '''
        Return :obj:`Frame` version of this :obj:`IndexHierarchy`.
        '''
        from static_frame import Frame
        return self._to_frame(Frame)

    def to_frame_go(self: IH) -> 'FrameGO':
        '''
        Return a :obj:`FrameGO` version of this :obj:`IndexHierarchy`.
        '''
        from static_frame import FrameGO
        return tp.cast(FrameGO, self._to_frame(FrameGO))

    def _to_signature_bytes(self,
            include_name: bool = True,
            include_class: bool = True,
            encoding: str = 'utf-8',
            ) -> bytes:

        v = (self.values_at_depth(i).tobytes() for i in range(self.depth))
        return b''.join(chain(
                iter_component_signature_bytes(self,
                        include_name=include_name,
                        include_class=include_class,
                        encoding=encoding),
                v,
                ))

    # --------------------------------------------------------------------------
    def to_pandas(self: IH) -> 'DataFrame':
        '''
        Return a Pandas MultiIndex.
        '''
        import pandas

        if self._recache:
            self._update_array_cache()

        # must copy to get a mutable array
        mi = pandas.MultiIndex(
                levels=[index.values.copy() for index in self._indices],
                codes=[arr.copy() for arr in self._indexers],
                )
        mi.name = self._name
        mi.names = self.names
        return mi

    # --------------------------------------------------------------------------

    def _build_tree_at_depth_from_mask(self: IH,
            depth: int,
            mask: np.ndarray,
            ) -> tp.Union[TreeNodeT, Index]:
        '''
        Recursively build a tree of :obj:`TreeNodeT` at `depth` given `mask`
        '''
        # This private internal method assumes recache has already been checked for!

        index_at_depth = self._indices[depth]
        indexer_at_depth = self._indexers[depth]

        if depth == self.depth - 1:
            values = index_at_depth[indexer_at_depth[mask]]
            return index_at_depth.__class__(values)

        tree: TreeNodeT = {}

        for i in ufunc_unique(indexer_at_depth[mask]):
            tree[index_at_depth[i]] = self._build_tree_at_depth_from_mask(
                    depth=depth + 1,
                    mask=mask & (indexer_at_depth == i),
                    )

        return tree

    def to_tree(self: IH) -> TreeNodeT:
        '''
        Returns the tree representation of an IndexHierarchy
        '''
        if self._recache:
            self._update_array_cache()

        tree = self._build_tree_at_depth_from_mask(
                depth=0,
                mask=np.ones(self.__len__(), dtype=bool),
                )
        return tree # type: ignore

    def flat(self: IH) -> Index:
        '''
        Return a flat, one-dimensional index of tuples for each level.
        '''
        return self._INDEX_CONSTRUCTOR(self.__iter__(), name=self._name) # type: ignore

    def level_add(self: IH,
            level: tp.Hashable,
            *,
            index_constructor: IndexConstructor = None,
            ) -> IH:
        '''
        Return an IndexHierarchy with a new root (outer) level added.
        '''
        if self._recache:
            self._update_array_cache()

        index_cls = self._INDEX_CONSTRUCTOR if index_constructor is None else index_constructor._MUTABLE_CONSTRUCTOR # type: ignore

        if self.STATIC:
            indices = [index_cls((level,)), *self._indices]
        else:
            indices = [index_cls((level,)), *(idx.copy() for idx in self._indices)]

        indexers = np.array(
                [
                    np.zeros(self.__len__(), dtype=DTYPE_INT_DEFAULT),
                    *self._indexers
                ],
                dtype=DTYPE_INT_DEFAULT,
        )
        indexers.flags.writeable = False

        def gen_blocks() -> tp.Iterator[np.ndarray]:
            # First index only has one value. Extract from array (instead of using `level`) since the constructor might have modified its type
            yield np.full(self.__len__(), indices[0][0])
            yield from self._blocks._blocks

        return self.__class__(
                indices=indices,
                indexers=indexers,
                name=self.name,
                blocks=TypeBlocks.from_blocks(gen_blocks()),
                own_blocks=True,
                )

    def level_drop(self: IH,
            count: int = 1,
            ) -> tp.Union[Index, IH]:
        '''
        Return an IndexHierarchy with one or more leaf levels removed.

        Args:
            count: A positive value is the number of depths to remove from the root (outer) side of the hierarchy; a negative value is the number of depths to remove from the leaf (inner) side of the hierarchy.
        '''
        if self._recache:
            self._update_array_cache()

        # NOTE: this was implement with a bipolar ``count`` to specify what to drop, but it could have been implemented with a depth level specifier, supporting arbitrary removals. The approach taken here is likely faster as we reuse levels.
        if self._name_is_names():
            if count < 0:
                name: NameType = self._name[:count] # type: ignore
            elif count > 0:
                name = self._name[count:] # type: ignore
            if len(name) == 1:
                name = name[0] # type: ignore
        else:
            name = self._name

        if count < 0:
            if count <= (1 - self.depth):
                # When the removal equals or exceeds the depth of the hierarchy, we just return the outermost index
                # NOTE: We can't copy the index directly, since the index order might not match.
                return self._indices[0].__class__(
                        self._blocks.iloc[:,0].values.reshape(self.__len__()),
                        name=name,
                        )

            # Remove from inner
            return self.__class__(
                    indices=self._indices[:count],
                    indexers=self._indexers[:count],
                    name=name,
                    blocks=self._blocks[:count],
                    own_blocks=self.STATIC,
                    )

        elif count > 0:
            if count >= (self.depth - 1):
                # When the removal equals or exceeds the depth of the hierarchy, we just return the innermost index
                # NOTE: We can't copy the index directly, since the index order might not match.
                return self._indices[-1].__class__(
                        self._blocks.iloc[:,-1].values.reshape(self.__len__()),
                        name=name,
                        )

            # Remove from outer
            return self.__class__(
                    indices=self._indices[count:],
                    indexers=self._indexers[count:],
                    name=name,
                    blocks=self._blocks.iloc[:,count:],
                    own_blocks=self.STATIC,
                    )

        raise NotImplementedError('no handling for a 0 count drop level.')


class IndexHierarchyGO(IndexHierarchy):
    '''
    A hierarchy of :obj:`static_frame.Index` objects that permits mutation only in the addition of new hierarchies or labels.
    '''

    STATIC = False

    _IMMUTABLE_CONSTRUCTOR = IndexHierarchy
    _INDEX_CONSTRUCTOR = IndexGO

    _indices: tp.List[IndexGO] # type: ignore

    def append(self: IHGO,
            value: tp.Sequence[tp.Hashable],
            ) -> None:
        '''
        Append a single label to this IndexHierarchyGO in-place
        '''
        # We do not check whether nor the key exists, as that is too expensive.
        # Instead, we delay failure until _recache
        if len(value) != self.depth:
            raise RuntimeError(
                f'key length ({len(value)}) does not match hierarchy depth ({self.depth})'
            )

        for depth, label_at_depth in enumerate(value):
            if label_at_depth not in self._indices[depth]:
                self._indices[depth].append(label_at_depth)

        if self._pending_extensions is None:
            self._pending_extensions = []

        self._pending_extensions.append(PendingRow(value))
        self._recache = True

    # 2/3 index.difference
    # 1/3 index.extend
    def extend(self: IHGO,
            other: IndexHierarchy,
            ) -> None:
        '''
        Extend this IndexHierarchyGO in-place
        '''
        # We do not check whether nor the key exists, as that is too expensive.
        # Instead, we delay failure until _recache

        for self_index, other_index in zip(self._indices, other._indices):
            # This will force a recache -> but that's okay.
            difference = other_index.difference(self_index)
            if difference.size:
                self_index.extend(difference)

        if self._pending_extensions is None:
            self._pending_extensions = []

        self._pending_extensions.append(other)
        self._recache = True


# update class attr on Index after class initialization
IndexHierarchy._MUTABLE_CONSTRUCTOR = IndexHierarchyGO


class IndexHierarchyAsType:

    __slots__ = (
            'container',
            'depth_key',
            )

    container: IndexHierarchy
    depth_key: GetItemKeyType

    def __init__(self: IHAsType,
            container: IndexHierarchy,
            depth_key: GetItemKeyType
            ) -> None:
        '''
        Args:
            depth_key: must be normalized iloc key via validate_depth_selection()
        '''
        self.container = container
        self.depth_key = depth_key

    def __call__(self: IHAsType,
            dtypes: DtypesSpecifier,
            *,
            consolidate_blocks: bool = False,
            ) -> IndexHierarchy:
        '''
        Entrypoint to `astype` the container
        '''
        from static_frame.core.index_datetime import dtype_to_index_cls
        container = self.container

        if container._recache:
            container._update_array_cache()

        if self.depth_key.__class__ is slice and self.depth_key == NULL_SLICE:
            dtype_factory = get_col_dtype_factory(dtypes, range(self.container.depth))
            gen = self.container._blocks._astype_blocks_from_dtypes(dtype_factory)
        else:
            if not is_dtype_specifier(dtypes):
                raise RuntimeError('must supply a single dtype specifier if using a depth selection other than the NULL slice')
            gen = self.container._blocks._astype_blocks(self.depth_key, dtypes)

        if consolidate_blocks:
            gen = TypeBlocks.consolidate_blocks(gen)

        blocks = TypeBlocks.from_blocks(gen, shape_reference=self.container.shape)

        # update index_constructors based on dtype
        index_constructors = container.index_types.values.copy()
        dtype_post = blocks.dtypes[self.depth_key] # can select element or array
        if isinstance(dtype_post, np.dtype): # if an element
            index_constructors[self.depth_key] = dtype_to_index_cls(
                    container.STATIC,
                    dtype_post,
                    )
        else: # dtype_post is an iterable of values of same size dpeth_key selection
            index_constructors[self.depth_key] = [
                dtype_to_index_cls(container.STATIC, dt) for dt in dtype_post
            ]

        return container.__class__._from_type_blocks(
                blocks=blocks,
                index_constructors=index_constructors,
                own_blocks=True,
                name=self.container._name,
                )
