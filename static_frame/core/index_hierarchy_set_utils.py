from functools import partial
import typing as tp

import numpy as np
from static_frame.core.exception import ErrorInitIndex
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.loc_map import HierarchicalLocMap
from static_frame.core.index import Index
from static_frame.core.container_util import index_many_to_one
from static_frame.core.util import DtypeSpecifier
from static_frame.core.util import IndexConstructor
from static_frame.core.util import ManyToOneType
from static_frame.core.util import ufunc_unique1d
from static_frame.core.util import ufunc_unique1d_indexer
from static_frame.core.util import setdiff1d
from static_frame.core.util import intersect1d


class ValidationResult(tp.NamedTuple):
    indices: tp.List[IndexHierarchy]
    depth: int
    any_dropped: bool
    any_shallow_copies: bool
    name: tp.Hashable
    index_constructors: tp.List[IndexConstructor]


def _get_shallow_copy_key(ih: IndexHierarchy) -> tp.Tuple[int, ...]:
    # calling id(...) on numpy arrays is not always reliable, so we might not
    # get the optimization we hope for
    # Maybe look at numpy's view object instead
    return tuple(map(id, ih._blocks._blocks))


def _validate_and_process_indices(
        indices: tp.Tuple[IndexHierarchy],
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

    name: tp.Hashable = indices[0].name
    index_constructors = list(indices[0]._index_constructors)

    unique_indices: tp.Dict[tp.Tuple[int, ...], IndexHierarchy] = {}

    depth: tp.Optional[int] = None
    for idx in indices:
        if name is not None and idx.name != name:
            name = None

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

        key = _get_shallow_copy_key(idx)

        if key not in unique_indices:
            unique_indices[key] = idx
        else:
            any_shallow_copies = True

    assert depth is not None # mypy

    return ValidationResult(
            indices=list(unique_indices.values()),
            depth=depth,
            any_dropped=any_dropped,
            any_shallow_copies=any_shallow_copies,
            name=name,
            index_constructors=index_constructors or [Index] * depth,
            )


def get_encoding_invariants(indices: tp.List[Index]) -> tp.Tuple[np.ndarray, DtypeSpecifier]:
    # Our encoding scheme requires that we know the number of unique elements
    # for each union depth
    # `num_unique_elements_per_depth` is used as a bit union for the encodings
    bit_offset_encoders, encoding_can_overflow = HierarchicalLocMap.build_offsets_and_overflow(
        num_unique_elements_per_depth=list(map(len, indices)),
    )
    encoding_dtype = object if encoding_can_overflow else np.uint64
    return bit_offset_encoders, encoding_dtype


def return_specific(ih: IndexHierarchy) -> IndexHierarchy:
    return ih if ih.STATIC else ih.__deepcopy__({})


def return_empty(
        index_constructors: tp.List[IndexConstructor],
        name: tp.Hashable,
        ) -> IndexHierarchy:
    return IndexHierarchy._from_empty(
            (),
            depth_reference=len(index_constructors),
            index_constructors=index_constructors,
            name=name,
            )


def build_union_indices(
        indices: tp.Sequence[IndexHierarchy],
        index_constructors: tp.List[IndexConstructor],
        depth: int,
        ) -> tp.List[Index]:
    union_indices: tp.List[Index] = []

    for i in range(depth):
        union = index_many_to_one(
                (idx.index_at_depth(i) for idx in indices),
                cls_default=index_constructors[i],
                many_to_one_type=ManyToOneType.UNION,
                )
        union_indices.append(union)

    return union_indices


def _get_encodings(
        ih: IndexHierarchy,
        *,
        union_indices: tp.List[Index],
        depth: int,
        bit_offset_encoders: np.ndarray,
        encoding_dtype: DtypeSpecifier,
        ) -> np.ndarray:
    '''Encode `ih` based on the union indices'''
    remapped_indexers: tp.List[np.ndarray] = []

    union_idx: Index
    idx: Index
    indexer: np.ndarray

    for ( # type: ignore
        union_idx,
        idx,
        indexer
    ) in zip(
        union_indices,
        ih.index_at_depth(list(range(depth))),
        ih.indexer_at_depth(list(range(depth)))
    ):
        # 2. For each depth, for each index, remap the indexers to the shared base.
        indexer_remap_key = idx._index_iloc_map(union_idx)
        remapped_indexers.append(indexer_remap_key[indexer])

    return HierarchicalLocMap.encode(
            np.array(remapped_indexers, dtype=encoding_dtype).T,
            bit_offset_encoders,
            )


def _remove_union_bloat(
        indices: tp.List[Index],
        indexers: tp.List[np.ndarray],
        ) -> tp.Tuple[tp.List[Index], np.ndarray]:
    # There is potentially a LOT of leftover bloat from all the unions. Clean up.
    final_indices: tp.List[Index] = []
    final_indexers: tp.List[np.ndarray] = []

    for index, indexers in zip(indices, indexers):
        unique, new_indexers = ufunc_unique1d_indexer(indexers)

        if len(unique) == len(index):
            final_indices.append(index)
            final_indexers.append(indexers)
        else:
            final_indices.append(index._extract_iloc(unique))
            final_indexers.append(new_indexers)

    final_indexers_arr = np.array(final_indexers, dtype=np.uint64)
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
        return return_specific(lhs)

    result = _validate_and_process_indices(indices)
    del indices
    filtered_indices = result.indices

    if result.any_dropped:
        return return_empty(result.index_constructors, result.name)

    # 1. Find union_indices
    union_indices: tp.List[Index] = build_union_indices(
            filtered_indices,
            result.index_constructors,
            result.depth,
            )

    # 2-3. Remap indexers and convert to encodings
    bit_offset_encoders, encoding_dtype = get_encoding_invariants(union_indices)

    get_encodings = partial(
            _get_encodings,
            union_indices=union_indices,
            depth=result.depth,
            bit_offset_encoders=bit_offset_encoders,
            encoding_dtype=encoding_dtype,
            )

    # Start with the smallest index to minimize the number of remappings
    filtered_indices = sorted(filtered_indices, key=lambda x: x.size, reverse=True)

    # Choose the smallest
    first_ih = filtered_indices.pop()

    intersection_encodings = get_encodings(first_ih)

    while filtered_indices:
        next_encodings = get_encodings(filtered_indices.pop())

        # 4. Find the iterative intersection for each encodings.
        intersection_encodings = intersect1d(intersection_encodings, next_encodings)

        if not intersection_encodings.size:
            # 4.a. If the intersection is ever empty, we can stop!
            return return_empty(result.index_constructors, result.name)

    if len(intersection_encodings) == len(lhs):
        # In intersections, nothing can be added. If the size didn't change, then it means
        # nothing was removed, which means the union is the same as the first index
        return return_specific(lhs)

    # 5. Convert the intersection encodings back to 2-D indexers
    intersection_indexers = HierarchicalLocMap.unpack_encoding(
            intersection_encodings, bit_offset_encoders
            )

    # 6. Remove any bloat from the union indexers.
    final_indices, final_indexers = _remove_union_bloat(union_indices, intersection_indexers)

    return IndexHierarchy(
        indices=final_indices,
        indexers=final_indexers,
        name=result.name,
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
        return return_specific(lhs)

    result = _validate_and_process_indices(indices)
    del indices
    filtered_indices = result.indices

    if result.any_shallow_copies:
        # The presence of any duplicates always means an empty result: `S - S === {}``
        return return_empty(result.index_constructors, result.name)

    if len(filtered_indices) == 1 and filtered_indices[0] is lhs:
        # All the other indices were empty!
        return return_specific(lhs)

    # 1. Find union_indices
    union_indices: tp.List[Index] = build_union_indices(
            filtered_indices,
            result.index_constructors,
            result.depth,
            )

    # 2-3. Remap indexers and convert to encodings
    bit_offset_encoders, encoding_dtype = get_encoding_invariants(union_indices)

    get_encodings = partial(
            _get_encodings,
            union_indices=union_indices,
            depth=result.depth,
            bit_offset_encoders=bit_offset_encoders,
            encoding_dtype=encoding_dtype,
            )

    # Order the rest largest to smallest reversed (we will pop)
    filtered_indices = sorted(filtered_indices[1:], key=len)

    difference_encodings = get_encodings(lhs)

    while filtered_indices:
        next_encodings = get_encodings(filtered_indices.pop())

        # 4. Find the iterative difference for each encoding.
        difference_encodings = setdiff1d(difference_encodings, next_encodings)

        if not difference_encodings.size:
            # 4.a. If the difference is ever empty, we can stop!
            return return_empty(result.index_constructors, result.name)

    if len(difference_encodings) == len(lhs):
        # In differences, nothing can be added. If the size didn't change, then it means
        # nothing was removed, which means the union is the same as the first index
        return return_specific(lhs)

    # 5. Convert the difference encodings back to 2-D indexers
    difference_indexers = HierarchicalLocMap.unpack_encoding(
            difference_encodings, bit_offset_encoders
            )

    # 6. Remove any bloat from the union indexers.
    final_indices, final_indexers = _remove_union_bloat(union_indices, difference_indexers)

    return IndexHierarchy(
        indices=final_indices,
        indexers=final_indexers,
        name=result.name,
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
    result = _validate_and_process_indices(indices)
    del indices
    filtered_indices = result.indices

    # 1. Find union_indices
    union_indices: tp.List[Index] = build_union_indices(
            filtered_indices,
            result.index_constructors,
            result.depth,
            )

    # 2-3. Remap indexers and convert to encodings
    bit_offset_encoders, encoding_dtype = get_encoding_invariants(union_indices)

    get_encodings = partial(
            _get_encodings,
            union_indices=union_indices,
            depth=result.depth,
            bit_offset_encoders=bit_offset_encoders,
            encoding_dtype=encoding_dtype,
            )

    union_encodings: tp.List[np.ndarray] = list(map(get_encodings, filtered_indices))
    del filtered_indices

    # 4. Build up the union of the encodings (i.e., whatever encodings are unique)
    union_encodings = ufunc_unique1d(np.hstack(union_encodings))

    if len(union_encodings) == len(lhs):
        # In unions, nothing can be dropped. If the size didn't change, then it means
        # nothing was added, which means the union is the same as the first index
        return return_specific(lhs)

    # 5. Convert the union encodings back to 2-D indexers
    union_indexers = HierarchicalLocMap.unpack_encoding(
            union_encodings, bit_offset_encoders
            )

    return IndexHierarchy(
        indices=union_indices,
        indexers=union_indexers,
        name=result.name,
    )
