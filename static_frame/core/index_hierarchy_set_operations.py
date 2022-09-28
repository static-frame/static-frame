import typing as tp

import numpy as np
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.loc_map import HierarchicalLocMap
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.index import Index
from static_frame.core.util import DtypeSpecifier



def index_hierarchy_intersection(
        lhs: IndexHierarchy, rhs: IndexHierarchy
        ) -> IndexHierarchy:
    '''
    lhs & rhs # (where lhs and rhs are IndexHierarchies)

    Note:
        The result is not guaranteed to be sorted.
    '''
    remapped_indexers_lhs: tp.List[np.ndarray] = []
    remapped_indexers_rhs: tp.List[np.ndarray] = []
    num_unique_elements_per_depth: tp.List[int] = []

    for (
        idx_lhs,
        idx_rhs,
        indexer_lhs,
        indexer_rhs
    ) in zip(
        lhs.index_at_depth(range(lhs.depth)),
        rhs.index_at_depth(range(rhs.depth)),
        lhs.indexer_at_depth(range(lhs.depth)),
        rhs.indexer_at_depth(range(rhs.depth))
    ):
        # Determine the union of both indices, to ensure that both indexers can
        # map to a shared base
        union = idx_lhs.union(idx_rhs)
        num_unique_elements_per_depth.append(len(union))

        # Map the indices to the shared base
        remapped_lhs = idx_lhs._index_iloc_map(union)
        remapped_rhs = idx_rhs._index_iloc_map(union)

        # Use that base to encode both indexers
        remapped_indexers_lhs.append(remapped_lhs[indexer_lhs])
        remapped_indexers_rhs.append(remapped_rhs[indexer_rhs])

    # Our encoding scheme requires that we know the number of unique elements
    # for each union depth
    # `num_unique_elements_per_depth` is used as a bit union for the encodings
    bit_offset_encoders, encoding_can_overflow = HierarchicalLocMap.build_offsets_and_overflow(
        num_unique_elements_per_depth=num_unique_elements_per_depth
    )
    encoding_dtype = object if encoding_can_overflow else np.uint64

    def encode_ih_to_union_encodings(arrs: tp.List[np.ndarray]) -> np.ndarray:
        # See HierarchicalLocMap docs for more details on the encoding scheme
        arr = np.array(arrs, dtype=encoding_dtype).T
        arr <<= bit_offset_encoders
        return np.bitwise_or.reduce(arr, axis=1)

    lhs_encodings = encode_ih_to_union_encodings(remapped_indexers_lhs)
    rhs_encodings = encode_ih_to_union_encodings(remapped_indexers_rhs)

    # Since the encoding utilitzed the union, we can safely ask for the
    # intersection between the mappings
    mask = np.in1d(lhs_encodings, rhs_encodings)

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
    # ~0%
    remapped_indexers_lhs: tp.List[np.ndarray] = []
    remapped_indexers_rhs: tp.List[np.ndarray] = []
    num_unique_elements_per_depth: tp.List[int] = []

    union_indices: tp.List[Index] = []

    for idx_lhs, idx_rhs, indexer_lhs, indexer_rhs in zip(
        lhs._indices, rhs._indices, lhs._indexers, rhs._indexers
    ):
        union = idx_lhs.union(idx_rhs)
        union_indices.append(union)
        num_unique_elements_per_depth.append(len(union))

        remapped_a = idx_lhs._index_iloc_map(union)
        remapped_b = idx_rhs._index_iloc_map(union)

        remapped_indexers_lhs.append(remapped_a[indexer_lhs])
        remapped_indexers_rhs.append(remapped_b[indexer_rhs])

    (
        bit_offset_encoders,
        encoding_can_overflow,
    ) = HierarchicalLocMap.build_offsets_and_overflow(
        num_unique_elements_per_depth=num_unique_elements_per_depth
    )

    encoding_dtype = object if encoding_can_overflow else np.uint64

    remapped_indexers_lhs = np.array(remapped_indexers_lhs, dtype=encoding_dtype)
    remapped_indexers_rhs = np.array(remapped_indexers_rhs, dtype=encoding_dtype)

    # ~3%
    lhs_encodings = np.bitwise_or.reduce(
        remapped_indexers_lhs.T << bit_offset_encoders, axis=1
    )
    rhs_encodings = np.bitwise_or.reduce(
        remapped_indexers_rhs.T << bit_offset_encoders, axis=1
    )

    # ~46%
    unique_to_b = np.in1d(rhs_encodings, lhs_encodings, invert=True)

    indexers = np.hstack(
        (remapped_indexers_lhs, remapped_indexers_rhs.T[unique_to_b].T)
    )
    if encoding_can_overflow:
        indexers = indexers.astype(np.uint64)
    indexers.flags.writeable = False

    # ~2%
    lhs_blocks = lhs._blocks
    rhs_blocks = rhs._blocks._extract(unique_to_b)
    blocks = TypeBlocks.from_blocks(
        TypeBlocks.vstack_blocks_to_blocks((lhs_blocks, rhs_blocks))
    )

    # ~47%
    return IndexHierarchy(
        indices=union_indices,
        indexers=indexers,
        name=lhs.name,
        blocks=blocks,
        own_blocks=True,
    )
