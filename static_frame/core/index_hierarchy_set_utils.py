import typing as tp

import numpy as np
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.loc_map import HierarchicalLocMap
from static_frame.core.index import Index
from static_frame.core.util import ufunc_unique1d
from static_frame.core.util import ufunc_unique1d_indexer


def _validate_and_drop_empty(
        indices: tp.Tuple[IndexHierarchy],
        ) -> tp.Tuple[tp.List[IndexHierarchy], int, bool]:
    '''
    Common sanitization for IndexHierarchy operations.

    This will also invoke recache on all indices due to the `.size` call
    '''
    starting_len = len(indices)

    depth = None
    filtered: tp.List[IndexHierarchy] = []
    for idx in indices:
        # Drop empty indices
        if not idx.size:
            continue

        filtered.append(idx)

        if depth is None:
            depth = idx.depth
        elif depth != idx.depth:
            raise RuntimeError('All indices must have same depth')

    return indices, depth, starting_len != len(indices)


def index_hierarchy_intersection(*indices: IndexHierarchy) -> IndexHierarchy:
    '''
    Equivalent to:

        >>> result = indices[0]
        >>> for index in indices[1:]:
        >>>     result = result.intersection(index)

    Algorithm:

        1. Determine the union of all indices at each depth.
        2. For each depth, for each index, remap the indexers to the shared base.
        3. Now, we can start working with encodings.
        4. Start iterating through, building up the progressive intersection.
        5. If the intersection is ever empty, we can stop.
        6. If we finish with values left over, we now need to clean up.
            This is because the encodings might be mapping to values from the union
            index that have been dropped

    Note:
        The result is NOT guaranteed to be sorted. It most likely will not be.
    '''
    result_name = indices[0].name
    result_index_constructors = indices[0]._index_constructors
    indices, depth, any_dropped = _validate_and_drop_empty(indices)

    def return_empty() -> IndexHierarchy:
        return IndexHierarchy._from_empty(
                (),
                depth_reference=depth,
                index_constructors=result_index_constructors,
                name=result_name,
                )

    if any_dropped:
        return return_empty()

    # 1. Determine the union of all indices at each depth.
    union_indices: tp.List[Index] = list(indices[0].index_at_depth(list(range(depth))))

    for i in range(1, len(indices)):
        union_indices = [
            union.union(idx)
            for union, idx in zip(
                    union_indices,
                    indices[i].index_at_depth(list(range(depth)))
                    )
        ]

    # Our encoding scheme requires that we know the number of unique elements
    # for each union depth
    # `num_unique_elements_per_depth` is used as a bit union for the encodings
    bit_offset_encoders, encoding_can_overflow = HierarchicalLocMap.build_offsets_and_overflow(
        num_unique_elements_per_depth=list(map(len, union_indices)),
    )
    encoding_dtype = object if encoding_can_overflow else np.uint64

    # Start with the smallest index to minimize the number of remappings
    indices = sorted(indices, key=lambda x: x.size, reverse=True)

    def get_encodings(ih: IndexHierarchy) -> np.ndarray:
        '''Encode `ih` based on the union indices'''
        remapped_indexers: tp.List[np.ndarray] = []

        for (
            union_idx,
            idx,
            indexer
        ) in zip(
            union_indices,
            ih.index_at_depth(list(range(depth))),
            ih.indexer_at_depth(list(range(depth)))
        ):
            # 2. For each depth, for each index, remap the indexers to the shared base.
            remapped_rhs = idx._index_iloc_map(union_idx)
            remapped_indexers.append(remapped_rhs[indexer])

        return HierarchicalLocMap.encode(
                np.array(remapped_indexers, dtype=encoding_dtype).T,
                bit_offset_encoders,
                )

    # Choose the smallest
    first_ih = indices.pop()

    intersection_encodings = get_encodings(first_ih)

    while indices:
        next_encodings = get_encodings(indices.pop())

        intersection_encodings = np.intersect1d(intersection_encodings, next_encodings)
        if not intersection_encodings.size:
            return return_empty()

    # Now, unpack the union encodings into their corresponding indexers
    intersection_indexers = HierarchicalLocMap.unpack_encoding(
            intersection_encodings, bit_offset_encoders
            )

    # There is potentially a LOT of leftover bloat from all the unions. Clean up.
    final_indices: tp.List[Index] = []
    final_indexers: tp.List[np.ndarray] = []

    for index, indexers in zip(union_indices, intersection_indexers):
        unique, new_indexers = ufunc_unique1d_indexer(indexers)

        if len(unique) == len(index):
            final_indices.append(index)
            final_indexers.append(indexers)
        else:
            final_indices.append(index._extract_iloc(unique))
            final_indexers.append(new_indexers)

    final_indexers = np.array(final_indexers, dtype=np.uint64)
    final_indexers.flags.writeable = False

    return IndexHierarchy(
        indices=final_indices,
        indexers=final_indexers,
        name=result_name,
    )


def index_hierarchy_difference(*indices: IndexHierarchy) -> IndexHierarchy:
    '''
    Equivalent to:

        >>> for ih in rhs:
        >>>     lhs -= ih

    Note:
        The result is not guaranteed to be sorted.
    '''
    # This call will call recache
    result_name = indices[0].name
    result_index_constructors = indices[0]._index_constructors
    indices, depth, _ = _validate_and_drop_empty(indices)

    def return_empty() -> IndexHierarchy:
        return IndexHierarchy._from_empty(
                (),
                depth_reference=depth,
                index_constructors=result_index_constructors,
                name=result_name,
                )

    lhs, rhs = indices

    if lhs.equals(rhs):
        return return_empty()

    elif not lhs.size or not rhs.size:
        # If either are empty, the difference will be the same as lhs
        return lhs if lhs.STATIC else lhs.__deepcopy__({})

    remapped_indexers_lhs: tp.List[np.ndarray] = []
    remapped_indexers_rhs: tp.List[np.ndarray] = []

    union_indices: tp.List[Index] = []

    for (
        idx_lhs,
        idx_rhs,
        indexer_lhs,
        indexer_rhs
    ) in zip(
        lhs.index_at_depth(list(range(lhs.depth))),
        rhs.index_at_depth(list(range(rhs.depth))),
        lhs.indexer_at_depth(list(range(lhs.depth))),
        rhs.indexer_at_depth(list(range(rhs.depth)))
    ):
        # Determine the union of both indices, to ensure that both indexers can
        # map to a shared base
        union = idx_lhs.union(idx_rhs)
        union_indices.append(union)

        # Build up the mappings for both indices to the shared base
        remapped_lhs = idx_lhs._index_iloc_map(union)
        remapped_rhs = idx_rhs._index_iloc_map(union)

        # Apply that mapping to the both indexers
        remapped_indexers_lhs.append(remapped_lhs[indexer_lhs])
        remapped_indexers_rhs.append(remapped_rhs[indexer_rhs])

    # Our encoding scheme requires that we know the number of unique elements
    # for each union depth
    # `num_unique_elements_per_depth` is used as a bit union for the encodings
    bit_offset_encoders, encoding_can_overflow = HierarchicalLocMap.build_offsets_and_overflow(
        num_unique_elements_per_depth=list(map(len, union_indices)),
    )
    encoding_dtype = object if encoding_can_overflow else np.uint64

    remapped_indexers_lhs = np.array(remapped_indexers_lhs, dtype=encoding_dtype)
    remapped_indexers_rhs = np.array(remapped_indexers_rhs, dtype=encoding_dtype)

    lhs_encodings = HierarchicalLocMap.encode(remapped_indexers_lhs.T, bit_offset_encoders)
    rhs_encodings = HierarchicalLocMap.encode(remapped_indexers_rhs.T, bit_offset_encoders)

    # Now, simply filter by which encodings only appear in lhs
    unique_to_lhs = np.in1d(lhs_encodings, rhs_encodings, invert=True)

    # Now, extract the true union block.
    blocks = lhs._blocks._extract(unique_to_lhs)

    return IndexHierarchy._from_type_blocks(
        blocks,
        name=result_name,
        own_blocks=True,
        index_constructors=result_index_constructors,
    )


def index_hierarchy_union(*indices: IndexHierarchy) -> IndexHierarchy:
    '''
    Equivalent to:

        >>> result = indices[0]
        >>> for index in indices[1:]:
        >>>     result = result.union(index)

    Note:
        The result is NOT guaranteed to be sorted. It most likely will not be.
    '''
    # This call will call recache
    result_name = indices[0].name
    (lhs, *others), depth, _ = _validate_and_drop_empty(indices)
    del indices

    remapped_indexers_lhs: tp.List[np.ndarray] = []
    remapped_indexers_others: tp.List[tp.List[np.ndarray]] = [[] for _ in others]

    union_indices: tp.List[Index] = []

    for i in range(depth):
        idx_lhs = lhs.index_at_depth(i)
        indexer_lhs = lhs.indexer_at_depth(i)

        # Determine the union of both indices, to ensure that both indexers can
        # map to a shared base
        union = idx_lhs.union(*(rhs.index_at_depth(i) for rhs in others))
        union_indices.append(union)

        # Build up the mappings for both indices to the shared base
        remapped_lhs = idx_lhs._index_iloc_map(union)

        # Apply that mapping to the both indexers
        remapped_indexers_lhs.append(remapped_lhs[indexer_lhs])

        for j, rhs in enumerate(others):
            idx_rhs = rhs.index_at_depth(i)
            indexer_rhs = rhs.indexer_at_depth(i)

            remapped_rhs = idx_rhs._index_iloc_map(union)
            remapped_indexers_others[j].append(remapped_rhs[indexer_rhs])

    # Our encoding scheme requires that we know the number of unique elements
    # for each union depth
    # `num_unique_elements_per_depth` is used as a bit union for the encodings
    bit_offset_encoders, encoding_can_overflow = HierarchicalLocMap.build_offsets_and_overflow(
        num_unique_elements_per_depth=list(map(len, union_indices)),
    )
    encoding_dtype = object if encoding_can_overflow else np.uint64

    remapped_indexers_lhs = np.array(remapped_indexers_lhs, dtype=encoding_dtype)
    remapped_indexers_others = [
        np.array(remapped_indexers_rhs, dtype=encoding_dtype) for remapped_indexers_rhs in remapped_indexers_others
    ]

    lhs_encodings = HierarchicalLocMap.encode(remapped_indexers_lhs.T, bit_offset_encoders)
    others_encodings = [
        HierarchicalLocMap.encode(remapped_indexers_rhs.T, bit_offset_encoders)
        for remapped_indexers_rhs in remapped_indexers_others
    ]

    # Given all encodings, determine which are unique (i.e. the union!)
    union_encodings = ufunc_unique1d(
            np.hstack((lhs_encodings, *others_encodings))
            )

    # Now, unpack the union encodings into their corresponding indexers
    union_indexers = HierarchicalLocMap.unpack_encoding(
            union_encodings, bit_offset_encoders
            )

    return IndexHierarchy(
        indices=union_indices,
        indexers=union_indexers,
        name=result_name,
    )
