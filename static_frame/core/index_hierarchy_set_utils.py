import typing as tp

import numpy as np
from static_frame.core.exception import ErrorInitIndex
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.loc_map import HierarchicalLocMap
from static_frame.core.index import Index
from static_frame.core.container_util import index_many_to_one
from static_frame.core.util import DtypeSpecifier, IndexConstructor, ManyToOneType, ufunc_unique1d
from static_frame.core.util import ufunc_unique1d_indexer


class ValidationResult(tp.NamedTuple):
    indices: tp.List[IndexHierarchy]
    depth: int
    any_dropped: bool
    name: tp.Hashable
    index_constructors: tp.List[IndexConstructor]


def _validate_and_drop_empty(
        indices: tp.Tuple[IndexHierarchy],
        ) -> ValidationResult:
    '''
    Common sanitization for IndexHierarchy operations.

    This will also invoke recache on all indices due to the `.size` call
    '''
    any_dropped = False

    name: tp.Hashable = indices[0].name
    index_constructors = list(indices[0]._index_constructors)

    depth = None
    filtered: tp.List[IndexHierarchy] = []
    for idx in indices:
        if idx.name != name:
            name = None

        if index_constructors:
            for i, ctor in enumerate(idx._index_constructors):
                if ctor != index_constructors[i]:
                    index_constructors = []
                    break

        # Drop empty indices
        if not idx.size:
            any_dropped = True
            continue

        filtered.append(idx)

        if depth is None:
            depth = idx.depth
        elif depth != idx.depth:
            raise ErrorInitIndex('All indices must have same depth')

    if not index_constructors:
        index_constructors = [Index] * depth

    return ValidationResult(
            indices=filtered,
            depth=depth,
            any_dropped=any_dropped,
            name=name,
            index_constructors=index_constructors,
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
    lhs = indices[0]

    if not lhs.size:
        return return_specific(lhs)

    indices, depth, any_dropped, name, index_constructors = _validate_and_drop_empty(indices)

    def return_empty() -> IndexHierarchy:
        return IndexHierarchy._from_empty(
                (),
                depth_reference=depth,
                index_constructors=index_constructors,
                name=name,
                )

    if any_dropped:
        return return_empty()

    # 1. Determine the union of all indices at each depth.
    union_indices: tp.List[Index] = []

    for i in range(depth):
        union = index_many_to_one(
                (idx.index_at_depth(i) for idx in indices),
                cls_default=index_constructors[i],
                many_to_one_type=ManyToOneType.UNION,
                )
        union_indices.append(union)

    bit_offset_encoders, encoding_dtype = get_encoding_invariants(union_indices)

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

        # TODO: Use util.intersect1d?
        intersection_encodings = np.intersect1d(intersection_encodings, next_encodings)
        if not intersection_encodings.size:
            return return_empty()

    if len(intersection_encodings) == len(lhs):
        return return_specific(lhs)

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
        name=name,
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
    lhs = indices[0]

    if not lhs.size:
        return return_specific(lhs)

    indices, depth, _, name, index_constructors = _validate_and_drop_empty(indices)

    def return_empty() -> IndexHierarchy:
        return IndexHierarchy._from_empty(
                (),
                depth_reference=depth,
                index_constructors=index_constructors,
                name=name,
                )

    if len(indices) == 1 and indices[0] is lhs:
        return return_specific(lhs)

    # 1. Determine the union of all indices at each depth.
    union_indices: tp.List[Index] = []

    for i in range(depth):
        union = index_many_to_one(
                (idx.index_at_depth(i) for idx in indices),
                cls_default=index_constructors[i],
                many_to_one_type=ManyToOneType.UNION,
                )
        union_indices.append(union)

    # Our encoding scheme requires that we know the number of unique elements
    # for each union depth
    # `num_unique_elements_per_depth` is used as a bit union for the encodings
    bit_offset_encoders, encoding_can_overflow = HierarchicalLocMap.build_offsets_and_overflow(
        num_unique_elements_per_depth=list(map(len, union_indices)),
    )
    encoding_dtype = object if encoding_can_overflow else np.uint64

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

    # Order the rest largest to smallest reversed (we will pop)
    indices = sorted(indices[1:], key=lambda x: x.size)

    difference_encodings = get_encodings(lhs)

    while indices:
        next_encodings = get_encodings(indices.pop())

        difference_encodings = np.setdiff1d(difference_encodings, next_encodings)
        if not difference_encodings.size:
            return return_empty()

    if len(difference_encodings) == len(lhs):
        return return_specific(lhs)

    # Now, unpack the union encodings into their corresponding indexers
    difference_indexers = HierarchicalLocMap.unpack_encoding(
            difference_encodings, bit_offset_encoders
            )

    # There is potentially a LOT of leftover bloat from all the unions. Clean up.
    final_indices: tp.List[Index] = []
    final_indexers: tp.List[np.ndarray] = []

    for index, indexers in zip(union_indices, difference_indexers):
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
        name=name,
    )

# TODO: Is the general case too much overhead for the case of 2 indices?
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
    (lhs, *others), depth, _, name, index_constructors = _validate_and_drop_empty(indices)
    del indices

    remapped_indexers_lhs: tp.List[np.ndarray] = []
    remapped_indexers_others: tp.List[tp.List[np.ndarray]] = [[] for _ in others]

    union_indices: tp.List[Index] = []

    for i in range(depth):
        idx_lhs = lhs.index_at_depth(i)
        indexer_lhs = lhs.indexer_at_depth(i)

        # Determine the union of both indices, to ensure that both indexers can
        # map to a shared base
        union = index_many_to_one(
                (idx_lhs, *(rhs.index_at_depth(i) for rhs in others)),
                cls_default=index_constructors[i],
                many_to_one_type=ManyToOneType.UNION,
                )
        union_indices.append(union)

        # Build up the mappings for both indices to the shared base
        # TODO: Better name how-to-remap
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

    if len(union_encodings) == len(lhs):
        return return_specific(lhs)

    # Now, unpack the union encodings into their corresponding indexers
    union_indexers = HierarchicalLocMap.unpack_encoding(
            union_encodings, bit_offset_encoders
            )

    return IndexHierarchy(
        indices=union_indices,
        indexers=union_indexers,
        name=name,
    )
