import typing as tp

import numpy as np
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.loc_map import HierarchicalLocMap
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.index import Index


class IndexHierarchySetResult(tp.NamedTuple):
    lhs_encodings: np.ndarray # 1-D
    rhs_encodings: np.ndarray # 1-D
    remapped_indexers_lhs: np.ndarray # 2-D
    remapped_indexers_rhs: np.ndarray # 2-D
    union_indices: tp.List[Index]
    encoding_can_overflow: bool


def _index_hierarchy_set(
        lhs: IndexHierarchy, rhs: IndexHierarchy
        ) -> IndexHierarchySetResult:
    '''
    Shared logic between union and intersection.
    They both need all of these steps completed, but they both require different
    amounts of work to be returned
    '''
    remapped_indexers_lhs: tp.List[np.ndarray] = []
    remapped_indexers_rhs: tp.List[np.ndarray] = []
    num_unique_elements_per_depth: tp.List[int] = []

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
        num_unique_elements_per_depth.append(len(union))

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
        num_unique_elements_per_depth=num_unique_elements_per_depth
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

    return IndexHierarchySetResult(
        lhs_encodings=lhs_encodings,
        rhs_encodings=rhs_encodings,
        remapped_indexers_lhs=remapped_indexers_lhs,
        remapped_indexers_rhs=remapped_indexers_rhs,
        union_indices=union_indices,
        encoding_can_overflow=encoding_can_overflow,
    )


def index_hierarchy_intersection(
        lhs: IndexHierarchy, rhs: IndexHierarchy
        ) -> IndexHierarchy:
    '''
    lhs & rhs # (where lhs and rhs are IndexHierarchies)

    Note:
        The result is not guaranteed to be sorted.
    '''
    result = _index_hierarchy_set(lhs, rhs)

    # Since the encoding utilitzed the union, we can safely ask for the
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


def index_hierarchy_union(
    lhs: IndexHierarchy,
    rhs: IndexHierarchy,
) -> IndexHierarchy:

    result = _index_hierarchy_set(lhs, rhs)

    # Since the encoding utilitzed the union, we can safely ask for the
    # intersection between the mappings
    unique_to_b = np.in1d(result.rhs_encodings, result.lhs_encodings, invert=True)

    # We can now build up our new IndexHierarchy from each component.
    # Basically, we choose everything from lhs, and then concat the unique
    # elements from rhs.
    indexers = np.hstack(
        (result.remapped_indexers_lhs, result.remapped_indexers_rhs.T[unique_to_b].T)
    )
    if result.encoding_can_overflow:
        indexers = indexers.astype(np.uint64)
    indexers.flags.writeable = False

    lhs_blocks = lhs._blocks
    rhs_blocks = rhs._blocks._extract(unique_to_b)
    blocks = TypeBlocks.from_blocks(
        TypeBlocks.vstack_blocks_to_blocks((lhs_blocks, rhs_blocks))
    )

    return IndexHierarchy(
        indices=result.union_indices,
        indexers=indexers,
        name=lhs.name,
        blocks=blocks,
        own_blocks=True,
    )
