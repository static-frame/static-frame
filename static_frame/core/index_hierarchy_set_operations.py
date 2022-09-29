import typing as tp

import numpy as np
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.loc_map import HierarchicalLocMap
from static_frame.core.index import Index
from static_frame.core.util import ufunc_unique1d
from functools import reduce


class IndexHierarchySetResult(tp.NamedTuple):
    lhs_encodings: np.ndarray # 1-D
    others_encodings: tp.List[np.ndarray] # 1-D
    bit_offset_encoders: np.ndarray # 1-D
    union_indices: tp.List[Index]


class IndexHierarchySetResult_Old(tp.NamedTuple):
    lhs_encodings: np.ndarray # 1-D
    rhs_encodings: np.ndarray # 1-D
    remapped_indexers_lhs: np.ndarray # 2-D
    remapped_indexers_rhs: np.ndarray # 2-D
    union_indices: tp.List[Index]
    encoding_can_overflow: bool




def _index_hierarchy_set_many(
        lhs: IndexHierarchy,
        *others: IndexHierarchy,
    ) -> IndexHierarchySetResult:
    # Recache is automatically called when calling `.index_at_depth(...)`

    if any(rhs.depth != lhs.depth for rhs in others):
        raise RuntimeError('All indices must have same depth')

    remapped_indexers_lhs: tp.List[np.ndarray] = []
    remapped_indexers_others: tp.List[tp.List[np.ndarray]] = [[] for _ in others]

    union_indices: tp.List[Index] = []

    for i in range(lhs.depth):
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

    def _encode_indexers(indexers: np.ndarray) -> np.ndarray:
        # See HierarchicalLocMap docs for more details on the encoding scheme
        return np.bitwise_or.reduce(indexers.T << bit_offset_encoders, axis=1)

    lhs_encodings = _encode_indexers(remapped_indexers_lhs)
    others_encodings = [
        _encode_indexers(remapped_indexers_rhs) for remapped_indexers_rhs in remapped_indexers_others
    ]

    return IndexHierarchySetResult(
        lhs_encodings=lhs_encodings,
        others_encodings=others_encodings,
        bit_offset_encoders=bit_offset_encoders,
        union_indices=union_indices,
    )


def _index_hierarchy_set(
        lhs: IndexHierarchy,
        rhs: IndexHierarchy,
    ) -> IndexHierarchySetResult_Old:
    '''
    Shared logic for all set operations (union, intersection, difference)

    They all need each of these steps to be completed, but they require
    different amounts of intermediate work to be returned
    '''

    if lhs._recache:
        lhs._update_array_cache()

    if rhs._recache:
        rhs._update_array_cache()

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

    # See HierarchicalLocMap docs for more details on the encoding scheme
    lhs_encodings = np.bitwise_or.reduce(
        remapped_indexers_lhs.T << bit_offset_encoders, axis=1
    )
    rhs_encodings = np.bitwise_or.reduce(
        remapped_indexers_rhs.T << bit_offset_encoders, axis=1
    )

    return IndexHierarchySetResult_Old(
        lhs_encodings=lhs_encodings,
        rhs_encodings=rhs_encodings,
        remapped_indexers_lhs=remapped_indexers_lhs,
        remapped_indexers_rhs=remapped_indexers_rhs,
        union_indices=union_indices,
        encoding_can_overflow=encoding_can_overflow,
    )


def index_hierarchy_intersection(
        lhs: IndexHierarchy,
        *rhs: IndexHierarchy,
    ) -> IndexHierarchy:
    '''
    Equivalent to:

        >>> for ih in rhs:
        >>>     lhs &= ih

    Note:
        The result is not guaranteed to be sorted.
    '''

    if len(rhs) != 1:
        raise NotImplementedError('only one rhs supported')

    [rhs] = rhs

    if lhs.equals(rhs):
        return lhs if lhs.STATIC else lhs.__deepcopy__({})
    elif not lhs.size or not rhs.size:
        # If either are empty, the intersection will also be empty
        return IndexHierarchy._from_empty(
                (),
                depth_reference=lhs.depth,
                index_constructors=lhs._index_constructors,
                )

    result = _index_hierarchy_set(lhs, rhs)

    # Since the encoding utilized the union, we can safely ask for the
    # intersection between the mappings
    mask = np.in1d(result.lhs_encodings, result.rhs_encodings)

    # Now, extract the true union block.
    blocks = lhs._blocks._extract(mask)

    return IndexHierarchy._from_type_blocks(
        blocks,
        name=lhs.name,
        own_blocks=True,
        index_constructors=lhs._index_constructors,
    )


def index_hierarchy_difference(
        lhs: IndexHierarchy,
        *rhs: IndexHierarchy,
    ) -> IndexHierarchy:
    '''
    Equivalent to:

        >>> for ih in rhs:
        >>>     lhs -= ih

    Note:
        The result is not guaranteed to be sorted.
    '''

    if len(rhs) != 1:
        raise NotImplementedError('only one rhs supported')

    [rhs] = rhs

    if lhs.equals(rhs):
        return IndexHierarchy._from_empty(
                (),
                depth_reference=lhs.depth,
                index_constructors=lhs._index_constructors,
                )
    elif not lhs.size or not rhs.size:
        # If either are empty, the difference will be the same as lhs
        return lhs if lhs.STATIC else lhs.__deepcopy__({})

    result = _index_hierarchy_set(lhs, rhs)

    # Now, simply filter by which encodings only appear in lhs
    unique_to_lhs = np.in1d(result.lhs_encodings, result.rhs_encodings, invert=True)

    # Now, extract the true union block.
    blocks = lhs._blocks._extract(unique_to_lhs)

    return IndexHierarchy._from_type_blocks(
        blocks,
        name=lhs.name,
        own_blocks=True,
        index_constructors=lhs._index_constructors,
    )


def index_hierarchy_union(
        lhs: IndexHierarchy,
        *others: IndexHierarchy,
    ) -> IndexHierarchy:
    '''
    Equivalent to:

        >>> for ih in rhs:
        >>>     lhs |= ih

    Note:
        The result is NOT guaranteed to be sorted. It most likely will not be.
    '''
    # Drop empty indices
    result = _index_hierarchy_set_many(lhs, *(rhs for rhs in others if rhs.size))

    # Given all encodings, determine which are unique (i.e. the union!)
    union_encodings = ufunc_unique1d(
            np.hstack((result.lhs_encodings, *result.others_encodings))
            )

    # Now, unpack the union encodings into their corresponding indexers
    union_indexers = HierarchicalLocMap.unpack_encoding(
            union_encodings, result.bit_offset_encoders
            )

    return IndexHierarchy(
        indices=result.union_indices,
        indexers=union_indexers,
        name=lhs.name,
    )
