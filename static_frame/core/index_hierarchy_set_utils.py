from __future__ import annotations

from functools import partial

import numpy as np
import typing_extensions as tp

from static_frame.core.container_util import index_many_to_one
from static_frame.core.exception import ErrorInitIndex
from static_frame.core.index import Index
from static_frame.core.index import mutable_immutable_index_filter
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.loc_map import HierarchicalLocMap
from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import DTYPE_UINT_DEFAULT
from static_frame.core.util import ManyToOneType
from static_frame.core.util import TIndexCtorSpecifier
from static_frame.core.util import TLabel
from static_frame.core.util import intersect1d
from static_frame.core.util import setdiff1d
from static_frame.core.util import ufunc_unique1d
from static_frame.core.util import ufunc_unique1d_indexer

if tp.TYPE_CHECKING:
    TNDArrayAny = np.ndarray[tp.Any, tp.Any] #pragma: no cover
    TDtypeAny = np.dtype[tp.Any] #pragma: no cover

class ValidationResult(tp.NamedTuple):
    indices: tp.List[IndexHierarchy]
    depth: int
    any_dropped: bool
    any_shallow_copies: bool
    name: TLabel
    index_constructors: tp.List[TIndexCtorSpecifier]


def _validate_and_process_indices(
        indices: tp.Tuple[IndexHierarchy, ...],
        ) -> ValidationResult:
    '''
    Common sanitization for IndexHierarchy operations.
    1. Remove empty or cloned indices
    2. Ensure all indices have same depth
    3. Use the first index's name, if all indices have same name
    4. Use the first index's index_constructors, if all indices have same index_constructors

    This will also invoke recache on all indices due to the `.size` call
    '''
    any_dropped = False
    any_shallow_copies = False

    name: TLabel = indices[0].name
    index_constructors = list(indices[0]._index_constructors)

    unique_signatures: tp.Set[TLabel] = set()
    unique_non_empty_indices: tp.List[IndexHierarchy] = []

    depth: tp.Optional[int] = None
    for idx in indices:
        if name is not None and idx.name != name:
            name = None

        if index_constructors:
            for ctor, other_ctor in zip(idx._index_constructors, index_constructors):
                if other_ctor != ctor:
                    index_constructors = []
                    break

        # Drop empty indices
        if not idx.size:
            any_dropped = True
            continue

        if depth is None:
            depth = idx.depth
        elif depth != idx.depth:
            raise ErrorInitIndex('All indices must have same depth')

        signature = tuple(idx._blocks.iter_block_signatures())

        if signature not in unique_signatures:
            unique_non_empty_indices.append(idx)
            unique_signatures.add(signature)
        else:
            any_shallow_copies = True

    assert depth is not None # mypy

    return ValidationResult(
            indices=unique_non_empty_indices,
            depth=depth,
            any_dropped=any_dropped,
            any_shallow_copies=any_shallow_copies,
            name=name,
            index_constructors=index_constructors or [Index] * depth, # type: ignore
            )


def get_encoding_invariants(indices: tp.List[Index[tp.Any]]
        ) -> tp.Tuple[TNDArrayAny, TDtypeAny]:
    # Our encoding scheme requires that we know the number of unique elements
    # for each union depth
    # `num_unique_elements_per_depth` is used as a bit union for the encodings
    bit_offset_encoders, encoding_can_overflow = HierarchicalLocMap.build_offsets_and_overflow(
        num_unique_elements_per_depth=list(map(len, indices)),
    )
    encoding_dtype = DTYPE_OBJECT if encoding_can_overflow else DTYPE_UINT_DEFAULT
    return bit_offset_encoders, encoding_dtype


def get_empty(
        index_constructors: tp.List[TIndexCtorSpecifier],
        name: TLabel,
        ) -> IndexHierarchy:
    return IndexHierarchy._from_empty(
            (),
            depth_reference=len(index_constructors),
            index_constructors=index_constructors,
            name=name,
            )


def build_union_indices(
        indices: tp.Sequence[IndexHierarchy],
        index_constructors: tp.List[TIndexCtorSpecifier],
        depth: int,
        ) -> tp.List[Index[tp.Any]]:
    union_indices: tp.List[Index[tp.Any]] = []

    for i in range(depth):
        union = index_many_to_one(
                (idx._indices[i] for idx in indices),
                cls_default=index_constructors[i], # type: ignore
                many_to_one_type=ManyToOneType.UNION,
                )
        union_indices.append(union) # type: ignore

    return union_indices


def _get_encodings(
        ih: IndexHierarchy,
        *,
        union_indices: tp.List[Index[tp.Any]],
        depth: int,
        bit_offset_encoders: TNDArrayAny,
        encoding_dtype: TDtypeAny,
        ) -> TNDArrayAny:
    '''Encode `ih` based on the union indices'''
    remapped_indexers: tp.List[TNDArrayAny] = []

    union_idx: Index[tp.Any]
    idx: Index[tp.Any]
    indexer: TNDArrayAny
    depth_level = list(range(depth))

    for union_idx, idx, indexer in zip(
            union_indices,
            ih.index_at_depth(depth_level),
            ih.indexer_at_depth(depth_level)
            ):
        # 2. For each depth, for each index, remap the indexers to the shared base.
        indexer_remap_key = idx._index_iloc_map(union_idx)
        remapped_indexers.append(indexer_remap_key[indexer])

    return HierarchicalLocMap.encode(
            np.array(remapped_indexers, dtype=encoding_dtype).T,
            bit_offset_encoders,
            )


def _remove_union_bloat(
        indices: tp.List[Index[tp.Any]],
        indexers: tp.List[TNDArrayAny] | TNDArrayAny,
        ) -> tp.Tuple[tp.List[Index[tp.Any]], TNDArrayAny]:
    # There is potentially a LOT of leftover bloat from all the unions. Clean up.
    final_indices: tp.List[Index[tp.Any]] = []
    final_indexers: tp.List[TNDArrayAny] = []

    index: Index[tp.Any]
    indexer: TNDArrayAny

    for index, indexer in zip(indices, indexers):
        unique, new_indexer = ufunc_unique1d_indexer(indexer)

        if len(unique) == len(index):
            final_indices.append(index)
            final_indexers.append(indexer)
        else:
            final_indices.append(index._extract_iloc(unique))
            final_indexers.append(new_indexer)

    final_indexers_arr = np.array(final_indexers, dtype=DTYPE_UINT_DEFAULT)
    final_indexers_arr.flags.writeable = False

    return final_indices, final_indexers_arr


def index_hierarchy_intersection(*indices: IndexHierarchy) -> IndexHierarchy:
    '''
    Equivalent to:

        >>> result = indices[0]
        >>> for index in indices[1:]:
        >>>     result = result.intersection(index)

    Algorithm:

        1. Determine the union of the depth-level indices for each index.
        2. For each index, remap `indexers_at_depth` using the shared union base.
        3. Convert the 2-D indexers to 1-D encodings.
        4. Find the iterative intersection for each encoding.
            a. If the intersection is ever empty, we can stop!
        5. Convert the intersection encodings back to 2-D indexers.
        6. Remove any bloat from the union indexers.
        7. Return a new IndexHierarchy using the union_indices and union_indexers.

    Note:
        The result is only guaranteed to be sorted if the union equals the first index.
        In every other case, it will most likely NOT be sorted.
    '''
    lhs = indices[0]

    if not lhs.size:
        # If the first index is empty, the intersection will also be empty
        return mutable_immutable_index_filter(lhs.STATIC, lhs) # type: ignore

    args = _validate_and_process_indices(indices)
    del indices
    filtered_indices = args.indices

    if args.any_dropped:
        # If any index was empty, the intersection will also be empty
        return get_empty(args.index_constructors, args.name)

    # 1. Find union_indices
    union_indices = build_union_indices(
            filtered_indices,
            args.index_constructors,
            args.depth,
            )

    # 2-3. Remap indexers and convert to encodings
    bit_offset_encoders, encoding_dtype = get_encoding_invariants(union_indices)

    get_encodings = partial(
            _get_encodings,
            union_indices=union_indices,
            depth=args.depth,
            bit_offset_encoders=bit_offset_encoders,
            encoding_dtype=encoding_dtype,
            )

    # For any two indices being compared, we will have `fewer comparisons and a `higher likelihood of
    # creating a `smaller index, if the second index is *as small as it can be*.
    #
    # We have a greatest chance of getting a `smaller intermediate intersection if we start from the
    # smallest to the largest index. This will, in turn, mean every subsequent intersection routine
    # will be comparing against a `smaller set of labels. And, as we get smaller, our likelihood of
    # becoming empty increases.
    #
    # This is why we sort the indices in reverse order, so we can pop the smallest first.
    filtered_indices = sorted(filtered_indices, key=len, reverse=True)

    # Choose the smallest
    first_ih = filtered_indices.pop()

    intersection_encodings = get_encodings(first_ih)

    while filtered_indices:
        next_encodings = get_encodings(filtered_indices.pop())

        # 4. Find the iterative intersection for each encodings.
        intersection_encodings = intersect1d(intersection_encodings, next_encodings)

        if not intersection_encodings.size:
            # 4.a. If the intermediate intersection is ever empty, the end result must be empty
            return get_empty(args.index_constructors, args.name)

    if len(intersection_encodings) == len(lhs):
        # In intersections, nothing can be added. If the size didn't change, then it means
        # nothing was removed, which means the union is the same as the first index
        return mutable_immutable_index_filter(lhs.STATIC, lhs) # type: ignore

    # 5. Convert the intersection encodings back to 2-D indexers
    intersection_indexers = HierarchicalLocMap.unpack_encoding(
            encoded_arr=intersection_encodings,
            bit_offset_encoders=bit_offset_encoders,
            encoding_can_overflow=encoding_dtype is DTYPE_OBJECT,
            )

    # 6. Remove any bloat from the union indexers.
    final_indices, final_indexers = _remove_union_bloat(union_indices, intersection_indexers)

    return IndexHierarchy(
        indices=final_indices,
        indexers=final_indexers,
        name=args.name,
    )


def index_hierarchy_difference(*indices: IndexHierarchy) -> IndexHierarchy:
    '''
    Equivalent to:

        >>> result = indices[0]
        >>> for index in indices[1:]:
        >>>     result = result.differece(index)

    Algorithm:

        1. Determine the union of the depth-level indices for each index.
        2. For each index, remap `indexers_at_depth` using the shared union base.
        3. Convert the 2-D indexers to 1-D encodings.
        4. Find the iterative difference for each encoding.
            a. If the difference is ever empty, we can stop!
        5. Convert the difference encodings back to 2-D indexers.
        6. Remove any bloat from the union indexers.
        7. Return a new IndexHierarchy using the union_indices and union_indexers.

    Note:
        The result is only guaranteed to be sorted if the union equals the first index.
        In every other case, it will most likely NOT be sorted.
    '''
    lhs = indices[0]

    if not lhs.size:
        # If the first index is empty, the intersection will also be empty
        return mutable_immutable_index_filter(lhs.STATIC, lhs) # type: ignore

    args = _validate_and_process_indices(indices)
    del indices
    filtered_indices = args.indices

    if args.any_shallow_copies:
        # The presence of any duplicates always means an empty result
        return get_empty(args.index_constructors, args.name)

    if len(filtered_indices) == 1:
        # All the other indices were empty!
        return mutable_immutable_index_filter(lhs.STATIC, lhs) # type: ignore

    # 1. Find union_indices
    union_indices = build_union_indices(
            filtered_indices,
            args.index_constructors,
            args.depth,
            )

    # 2-3. Remap indexers and convert to encodings
    bit_offset_encoders, encoding_dtype = get_encoding_invariants(union_indices)

    get_encodings = partial(
            _get_encodings,
            union_indices=union_indices,
            depth=args.depth,
            bit_offset_encoders=bit_offset_encoders,
            encoding_dtype=encoding_dtype,
            )

    # For any two indices being diffed, we will have `more comparisons and a `higher likelihood of
    # finding a match, if the second index is *as large as it can be*.
    #
    # We have a greatest chance of getting a `smaller intermediate difference if we start from the
    # largest to the smallest index. This will, in turn, mean every subsequent difference routine
    # will be comparing against the most labels it can, increasing our likelihood of becoming empty.
    #
    # This is why we sort the indices in order, so we can pop the largest first.
    filtered_indices = sorted(filtered_indices[1:], key=len) # (We will pop)

    difference_encodings = get_encodings(lhs)

    while filtered_indices:
        next_encodings = get_encodings(filtered_indices.pop())

        # 4. Find the iterative difference for each encoding.
        difference_encodings = setdiff1d(difference_encodings, next_encodings)

        if not difference_encodings.size:
            # 4.a. If the intermediate difference is ever empty, the end result must be empty
            return get_empty(args.index_constructors, args.name)

    if len(difference_encodings) == len(lhs):
        # In differences, nothing can be added. If the size didn't change, then it means
        # nothing was removed, which means the difference is the same as the first index
        return mutable_immutable_index_filter(lhs.STATIC, lhs) # type: ignore

    # 5. Convert the difference encodings back to 2-D indexers
    difference_indexers = HierarchicalLocMap.unpack_encoding(
            encoded_arr=difference_encodings,
            bit_offset_encoders=bit_offset_encoders,
            encoding_can_overflow=encoding_dtype is DTYPE_OBJECT,
            )

    # 6. Remove any bloat from the union indexers.
    final_indices, final_indexers = _remove_union_bloat(union_indices, difference_indexers)

    return IndexHierarchy(
        indices=final_indices,
        indexers=final_indexers,
        name=args.name,
    )


def index_hierarchy_union(*indices: IndexHierarchy) -> IndexHierarchy:
    '''
    Equivalent to:

        >>> result = indices[0]
        >>> for index in indices[1:]:
        >>>     result = result.union(index)

    Algorithm:

        1. Determine the union of the depth-level indices for each index.
        2. For each index, remap `indexers_at_depth` using the shared union base.
        3. Convert the 2-D indexers to 1-D encodings.
        4. Build up the union of the encodings.
        5. Convert the union encodings back to 2-D indexers.
        6. Return a new IndexHierarchy using the union_indices and union_indexers.

    Note:
        The result is only guaranteed to be sorted if the union equals the first index.
        In every other case, it will most likely NOT be sorted.
    '''
    lhs = indices[0]
    args = _validate_and_process_indices(indices)
    del indices
    filtered_indices = args.indices

    # 1. Find union_indices
    union_indices = build_union_indices(
            filtered_indices,
            args.index_constructors,
            args.depth,
            )

    # 2-3. Remap indexers and convert to encodings
    bit_offset_encoders, encoding_dtype = get_encoding_invariants(union_indices)

    get_encodings = partial(
            _get_encodings,
            union_indices=union_indices,
            depth=args.depth,
            bit_offset_encoders=bit_offset_encoders,
            encoding_dtype=encoding_dtype,
            )

    union_encodings = list(map(get_encodings, filtered_indices))
    del filtered_indices

    # 4. Build up the union of the encodings (i.e., whatever encodings are unique)
    union_array = ufunc_unique1d(np.hstack(union_encodings))

    if len(union_array) == len(lhs):
        # In unions, nothing can be dropped. If the size didn't change, then it means
        # nothing was added, which means the union is the same as the first index
        return mutable_immutable_index_filter(lhs.STATIC, lhs) # type: ignore

    # 5. Convert the union encodings back to 2-D indexers
    union_indexers = HierarchicalLocMap.unpack_encoding(
            encoded_arr=union_array,
            bit_offset_encoders=bit_offset_encoders,
            encoding_can_overflow=encoding_dtype is DTYPE_OBJECT,
            )

    return IndexHierarchy(
        indices=union_indices,
        indexers=union_indexers,
        name=args.name,
    )
