from __future__ import annotations

import numpy as np
import typing_extensions as tp
from arraykit import FrozenAutoMap, NonUniqueError, TriMap, factorize, nonzero_1d

# from static_frame.core.container_util import FILL_VALUE_AUTO_DEFAULT
from static_frame.core.container_util import (
    arrays_from_index_frame,
    get_col_fill_value_factory,
    index_from_optional_constructor,
    index_many_concat,
)
from static_frame.core.index import Index
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.util import (
    DTYPE_BOOL,
    DTYPE_INT_DEFAULT,
    DTYPE_OBJECT,
    EMPTY_ARRAY_INT,
    INT64_MAX,
    NULL_SLICE,
    Join,
    TDepthLevel,
    TLabel,
    TLocSelector,
    TNDArray1DBool,
    TNDArrayAny,
    WarningsSilent,
    dtype_from_element,
    isna_array,
)

if tp.TYPE_CHECKING:
    from static_frame.core.generic_aliases import TFrameAny
    from static_frame.core.index_base import IndexBase

# -------------------------------------------------------------------------------


def _join_trimap_target_one(
    src_target: TNDArrayAny,
    dst_target: TNDArrayAny,
    join_type: Join,
) -> TriMap:
    """
    A TriMap constructor and mapper when target is only one column. Returns a mapped TriMap instance.
    """
    dst_count = len(dst_target)
    tm = TriMap(len(src_target), dst_count)

    # Fast path: a unique dst key lets us replace the O(n_src * n_dst) elementwise scan
    # with a single vectorized hash lookup (FrozenAutoMap.get_all_fill) and a bulk C
    # registration (register_many_from_one) -- an O(n_src + n_dst) hash join. Non-unique
    # dst raises NonUniqueError and falls through to the general (many-match) loop below.
    try:
        dst_map: tp.Optional[FrozenAutoMap] = FrozenAutoMap(dst_target)
    except NonUniqueError:
        dst_map = None
    if dst_map is not None:
        # dst_pos[i] is the matched dst iloc for src row i, or -1 if unmatched
        dst_pos = dst_map.get_all_fill(src_target)
        # LEFT/RIGHT/OUTER keep every src row (unmatched -> fill); INNER drops unmatched
        # src, so only take the fast path when every src row matched.
        if join_type is not Join.INNER or bool((dst_pos != -1).all()):
            tm.register_many_from_one(dst_pos)
            if join_type is Join.OUTER:
                tm.register_unmatched_dst()
            return tm
        # INNER with unmatched src rows: fall through to the general loop below

    # NOTE: explored using an LRU wrapper on this cache and it only degraded performance; not sure the memory befit is worth the performance cost.
    src_element_to_matched_idx = dict()
    with WarningsSilent():
        for src_i, src_element in enumerate(src_target):
            if src_element not in src_element_to_matched_idx:
                # TODO: recycle the matched Boolean array
                matched = src_element == dst_target

                # NOTE: remove when min numpy is 1.25
                if matched is False:
                    matched_idx = EMPTY_ARRAY_INT
                    matched_len = 0
                else:  # convert Booleans to integer positions
                    matched_idx = nonzero_1d(matched)
                    matched_len = len(matched_idx)

                src_element_to_matched_idx[src_element] = (matched_idx, matched_len)
            else:
                matched_idx, matched_len = src_element_to_matched_idx[src_element]

            if matched_len == 0:
                if join_type is not Join.INNER:
                    tm.register_one(src_i, -1)
            elif matched_len == 1:
                tm.register_one(src_i, matched_idx[0])
            else:  # one source value to many positions
                tm.register_many(src_i, matched_idx)

    if join_type is Join.OUTER:
        tm.register_unmatched_dst()
    return tm


def _encode_join_keys_int(
    src_target: list[TNDArrayAny],
    dst_target: list[TNDArrayAny],
) -> None | tp.Tuple[TNDArrayAny, TNDArrayAny]:
    """
    Encode a multi-column join key as a single int64 per row by jointly factorizing each
    depth column (over src and dst together, so equal values share a code) and combining
    the per-column codes in mixed radix. Returns ``(dst_key, src_key)`` int64 arrays, or
    ``None`` if the combined cardinality would overflow int64 (caller falls back).

    Rows holding a NaN/NaT in any key column are given a unique negative sentinel code so
    they never match, preserving the elementwise ``==`` semantics (``nan != nan``) of the
    original loop; ``None`` is *not* masked, as ``None == None`` matches under that loop.
    """
    n_dst = len(dst_target[0])
    total = n_dst + len(src_target[0])
    key = np.zeros(total, dtype=DTYPE_INT_DEFAULT)
    invalid = np.zeros(total, dtype=DTYPE_BOOL)

    radix = 1  # Python int: multiply without wraparound to detect overflow
    for cd, cs in zip(dst_target, src_target):
        cat = np.concatenate((cd, cs))
        uniques, codes = factorize(cat)
        k = len(uniques)
        if radix * k > INT64_MAX:
            return None

        key = key * k + codes.astype(DTYPE_INT_DEFAULT)
        radix *= k
        # only NaN/NaT never match (nan != nan); None is a matchable key
        na = isna_array(cat, include_none=False)
        if na.any():
            invalid |= na

    if invalid.any():
        # unique negatives cannot collide with the non-negative valid codes, nor with
        # each other, so any NaN/NaT-bearing row matches nothing
        key[invalid] = -1 - np.arange(invalid.sum(), dtype=DTYPE_INT_DEFAULT)

    return key[:n_dst], key[n_dst:]


def _match_pairs_from_keys(
    dst_key: TNDArrayAny,
    src_key: TNDArrayAny,
    join_type: Join,
) -> tp.Tuple[TNDArrayAny, TNDArrayAny]:
    """
    Given int64 keys where equal values indicate a match, produce ``(src_pairs, dst_pairs)``
    for ``TriMap.register_pairs`` fully vectorized. Pairs are emitted in src order, and each
    src's dst matches in ascending dst-index order (matching the loop's ``nonzero_1d`` order).
    A ``-1`` dst marks an unmatched src (kept for LEFT/OUTER, dropped for INNER).
    """
    n_src = len(src_key)
    is_inner = join_type is Join.INNER
    # group dst positions by key; a stable sort keeps each group's dst indices ascending
    order = np.argsort(dst_key, kind='stable')
    uniq, starts, counts = np.unique(
        dst_key[order], return_index=True, return_counts=True
    )
    if len(uniq) == 0:  # no dst rows to match against
        if is_inner:
            return np.empty(0, dtype=DTYPE_INT_DEFAULT), np.empty(
                0, dtype=DTYPE_INT_DEFAULT
            )
        return np.arange(n_src, dtype=DTYPE_INT_DEFAULT), np.full(
            n_src, -1, dtype=DTYPE_INT_DEFAULT
        )

    pos = np.searchsorted(uniq, src_key)
    in_range = pos < len(uniq)
    pos_clip = np.where(in_range, pos, 0)

    matched = in_range & (uniq[pos_clip] == src_key)
    src_n = np.where(matched, counts[pos_clip], 0).astype(DTYPE_INT_DEFAULT)
    grp_start = np.where(matched, starts[pos_clip], 0).astype(DTYPE_INT_DEFAULT)

    eff = src_n if is_inner else np.maximum(src_n, 1)
    src_pairs = np.repeat(np.arange(n_src, dtype=DTYPE_INT_DEFAULT), eff)

    # position within each src's run of output rows
    within = np.arange(len(src_pairs), dtype=DTYPE_INT_DEFAULT) - np.repeat(
        np.cumsum(eff) - eff, eff
    )
    # clamp within to the group size so an unmatched LEFT row (eff==1, src_n==0) is safe
    gather = np.repeat(grp_start, eff) + np.minimum(
        within, np.repeat(np.maximum(src_n, 1) - 1, eff)
    )

    dst_pairs = np.where(np.repeat(src_n == 0, eff), -1, order[gather]).astype(
        DTYPE_INT_DEFAULT
    )
    return np.ascontiguousarray(src_pairs), np.ascontiguousarray(dst_pairs)


def _join_trimap_target_many_loop(
    src_target: list[TNDArrayAny],
    dst_target: list[TNDArrayAny],
    join_type: Join,
    target_depth: int,
) -> TriMap:
    """
    Fallback multi-column matcher: an O(n_unique_src * n_dst) elementwise scan, used only
    when the int64 key encoding would overflow or a key dtype cannot be factorized.
    """
    src_element_to_matched_idx = dict()
    tm = TriMap(len(src_target[0]), len(dst_target[0]))
    matched_per_depth = np.empty((len(dst_target[0]), target_depth), dtype=DTYPE_BOOL)

    with WarningsSilent():
        # by iterating elements and comparing one depth at time, we avoid forcing any type conversions
        for src_i, src_elements in enumerate(zip(*src_target)):
            if src_elements not in src_element_to_matched_idx:
                for d, e in enumerate(src_elements):
                    matched_per_depth[NULL_SLICE, d] = e == dst_target[d]
                matched: TNDArray1DBool = matched_per_depth.all(axis=1)  # type: ignore
                matched_idx = nonzero_1d(matched)
                matched_len = len(matched_idx)

                src_element_to_matched_idx[src_elements] = (matched_idx, matched_len)
            else:
                matched_idx, matched_len = src_element_to_matched_idx[src_elements]

            if matched_len == 0:
                if join_type is not Join.INNER:
                    tm.register_one(src_i, -1)
            elif matched_len == 1:
                tm.register_one(src_i, matched_idx[0])
            else:  # one source value to many positions
                tm.register_many(src_i, matched_idx)

    if join_type is Join.OUTER:
        tm.register_unmatched_dst()

    return tm


def _join_trimap_target_many(
    src_target: list[TNDArrayAny],
    dst_target: list[TNDArrayAny],
    join_type: Join,
    target_depth: int,
) -> TriMap:
    """
    A TriMap constructor and mapper when target is more than one column. Encodes the
    multi-column key to a single int64 and performs an O(n) vectorized hash join, falling
    back to the elementwise-scan loop only on int64 overflow. Returns a mapped TriMap.
    """
    try:
        encoded = _encode_join_keys_int(src_target, dst_target)
    except (TypeError, ValueError):
        # a key dtype that cannot be factorized -> use the general loop
        encoded = None

    if encoded is None:
        return _join_trimap_target_many_loop(
            src_target, dst_target, join_type, target_depth
        )

    dst_key, src_key = encoded
    tm = TriMap(len(src_target[0]), len(dst_target[0]))
    src_pairs, dst_pairs = _match_pairs_from_keys(dst_key, src_key, join_type)
    tm.register_pairs(src_pairs, dst_pairs)
    if join_type is Join.OUTER:
        tm.register_unmatched_dst()
    return tm


def join(
    frame: TFrameAny,
    other: TFrameAny,  # support a named Series as a 1D frame?
    *,
    join_type: Join,  # intersect, left, right, union,
    left_depth_level: tp.Optional[TDepthLevel] = None,
    left_columns: TLocSelector = None,
    right_depth_level: tp.Optional[TDepthLevel] = None,
    right_columns: TLocSelector = None,
    left_template: str = '{}',
    right_template: str = '{}',
    fill_value: tp.Any = np.nan,
    include_index: bool = False,
    merge: bool = False,
    merge_labels: tp.Sequence[TLabel] | None = None,
) -> TFrameAny:
    from static_frame.core.frame import Frame

    # -----------------------------------------------------------------------
    # find matches
    if not isinstance(join_type, Join):
        raise NotImplementedError(f'`join_type` must be one of {tuple(Join)}')

    if left_depth_level is None and left_columns is None:
        raise RuntimeError(
            'Must specify one or both of left_depth_level and left_columns.'
        )
    if right_depth_level is None and right_columns is None:
        raise RuntimeError(
            'Must specify one or both of right_depth_level and right_columns.'
        )

    # reduce the targets to 2D arrays; possible coercion in some cases, but seems inevitable as we will be doing row-wise comparisons
    left_target: TNDArrayAny | list[TNDArrayAny]
    right_target: TNDArrayAny | list[TNDArrayAny]

    left_target, left_target_fields = arrays_from_index_frame(
        frame, left_depth_level, left_columns
    )
    right_target, right_target_fields = arrays_from_index_frame(
        other, right_depth_level, right_columns
    )

    if (target_depth := len(left_target)) != len(right_target):
        raise RuntimeError('left and right selections must be the same width.')

    if merge:
        # when merging, we drop target columns if defined for easier array processing
        if left_columns is not None:
            frame = frame.drop[left_columns]
        if right_columns is not None:
            other = other.drop[right_columns]

    if target_depth == 1:
        left_target = left_target[0]
        right_target = right_target[0]

    # Find matching pairs. Get iloc of left to iloc of right.
    left_frame = frame
    right_frame = other
    left_index = frame.index
    right_index = other.index
    left_columns = frame.columns
    right_columns = other.columns

    if join_type == Join.RIGHT:
        src_target = right_target
        dst_target = left_target
    else:
        src_target = left_target
        dst_target = right_target

    if target_depth == 1:
        tm = _join_trimap_target_one(src_target, dst_target, join_type)  # type: ignore
    else:
        tm = _join_trimap_target_many(src_target, dst_target, join_type, target_depth)  # type: ignore
    tm.finalize()

    # ---------------------------------------------------------------------------
    # prepare final columns

    default_ctr = frame._COLUMNS_CONSTRUCTOR
    if left_template != '{}':
        left_columns = default_ctr(left_columns.via_str.format(left_template))
    if right_template != '{}':
        right_columns = default_ctr(right_columns.via_str.format(right_template))

    if merge:
        if merge_labels is not None:
            # because we want this value to be like like selections given for the targets, we want an element to be acceptable for Index construction
            if not hasattr(merge_labels, '__iter__') or isinstance(
                merge_labels, (str, tuple)
            ):
                merge_labels = [merge_labels]
            merge_columns = index_from_optional_constructor(
                merge_labels, default_constructor=default_ctr
            )
            if len(merge_columns) != target_depth:
                raise RuntimeError(
                    'merge labels must be the same width as left and right selections.'
                )
        elif join_type is Join.RIGHT:
            merge_columns = default_ctr(right_target_fields)
        else:
            merge_columns = default_ctr(left_target_fields)
        final_columns = index_many_concat(
            (merge_columns, left_columns, right_columns), default_ctr
        )
    else:
        final_columns = index_many_concat((left_columns, right_columns), default_ctr)

    # we must use post template column names as there might be name conflicts
    get_col_fill_value = get_col_fill_value_factory(fill_value, columns=final_columns)

    # ---------------------------------------------------------------------------
    arrays = []
    # if we have matched all in src, we do not need fill values
    if join_type is Join.RIGHT:
        src_no_fill = tm.dst_no_fill
        dst_no_fill = tm.src_no_fill
        map_src_no_fill = tm.map_dst_no_fill  # swap sides
        map_src_fill = tm.map_dst_fill
        map_dst_no_fill = tm.map_src_no_fill
        map_dst_fill = tm.map_src_fill
    else:
        src_no_fill = tm.src_no_fill
        dst_no_fill = tm.dst_no_fill
        map_src_no_fill = tm.map_src_no_fill
        map_src_fill = tm.map_src_fill
        map_dst_no_fill = tm.map_dst_no_fill
        map_dst_fill = tm.map_dst_fill

    col_idx = 0
    if merge:  # src, dst labels will be correct for left/right orientation
        if target_depth == 1:
            arrays.append(tm.map_merge(src_target, dst_target))  # type: ignore
        else:
            for src, dst in zip(src_target, dst_target):
                arrays.append(tm.map_merge(src, dst))

    if src_no_fill():
        for proto in left_frame._blocks.axis_values():
            arrays.append(map_src_no_fill(proto))
            col_idx += 1
    else:
        for proto in left_frame._blocks.axis_values():
            fv = get_col_fill_value(col_idx, proto.dtype)
            fv_dtype = dtype_from_element(fv)
            arrays.append(map_src_fill(proto, fv, fv_dtype))
            col_idx += 1

    if dst_no_fill():
        for proto in right_frame._blocks.axis_values():
            arrays.append(map_dst_no_fill(proto))
            col_idx += 1
    else:
        for proto in right_frame._blocks.axis_values():
            fv = get_col_fill_value(col_idx, proto.dtype)
            fv_dtype = dtype_from_element(fv)
            arrays.append(map_dst_fill(proto, fv, fv_dtype))
            col_idx += 1

    # ---------------------------------------------------------------------------
    final_index: tp.Optional[IndexBase]
    if include_index:
        # NOTE: we are not yet accomdating a merge of index values into a non-tuple index, even when is_many is True
        own_index = True
        if join_type is not Join.OUTER and not tm.is_many():
            if join_type is Join.INNER:
                # NOTE: this could also be right; we could have an inner left and an inner right
                # NOTE: this will not preserve an IndexHierarchy
                final_index = Index(map_src_no_fill(left_index.values))
            elif join_type is Join.LEFT:
                final_index = left_index
            elif join_type is Join.RIGHT:
                final_index = right_index
        else:
            # NOTE: the fill value might need to be varied if left/right index already has None
            left_index_values = (
                left_index.values if left_index.depth == 1 else left_index.flat().values  # type: ignore
            )
            right_index_values = (
                right_index.values
                if right_index.depth == 1
                else right_index.flat().values  # type: ignore
            )

            # NOTE: not sure if src/dst arrangement is correct for right join
            if src_no_fill():
                labels_left = map_src_no_fill(left_index_values)
            else:
                labels_left = map_src_fill(left_index_values, None, DTYPE_OBJECT)
            if dst_no_fill():
                labels_right = map_dst_no_fill(right_index_values)
            else:
                labels_right = map_dst_fill(right_index_values, None, DTYPE_OBJECT)

            final_index = Index(zip(labels_left, labels_right), dtype=DTYPE_OBJECT)
    else:
        own_index = False
        final_index = None

    return frame.__class__(
        TypeBlocks.from_blocks(arrays),
        columns=final_columns,
        index=final_index,
        own_data=True,
        own_index=own_index,
        own_columns=True,
    )
