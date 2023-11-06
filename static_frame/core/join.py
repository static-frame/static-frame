from __future__ import annotations

from itertools import chain
from itertools import product

import numpy as np
import typing_extensions as tp
from arraykit import resolve_dtype

# from static_frame.core.container_util import FILL_VALUE_AUTO_DEFAULT
from static_frame.core.container_util import arrays_from_index_frame
from static_frame.core.container_util import is_fill_value_factory_initializer
from static_frame.core.exception import InvalidFillValue
from static_frame.core.index import Index
from static_frame.core.index_auto import IndexAutoFactory
from static_frame.core.type_blocks import TypeBlocks
# from static_frame.core.util import NULL_SLICE
from static_frame.core.util import Join
from static_frame.core.util import Pair
from static_frame.core.util import PairLeft
from static_frame.core.util import PairRight
from static_frame.core.util import TDepthLevel
from static_frame.core.util import TLabel
from static_frame.core.util import TLocSelector
from static_frame.core.util import WarningsSilent
from static_frame.core.util import array2d_to_tuples
from static_frame.core.util import dtype_from_element

if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.frame import FrameGO  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.generic_aliases import TFrameAny  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.generic_aliases import TFrameGOAny  # pylint: disable=W0611 #pragma: no cover


def join(frame: TFrameAny,
        other: TFrameAny, # support a named Series as a 1D frame?
        *,
        join_type: Join, # intersect, left, right, union,
        left_depth_level: tp.Optional[TDepthLevel] = None,
        left_columns: TLocSelector = None,
        right_depth_level: tp.Optional[TDepthLevel] = None,
        right_columns: TLocSelector = None,
        left_template: str = '{}',
        right_template: str = '{}',
        fill_value: tp.Any = np.nan,
        include_index: bool = False,
        ) -> TFrameAny:

    from static_frame.core.frame import FrameGO

    # NOTE: pre 1.0 these were optional parameters; now, we always return the data without an index; in the future, we might add back parameters to control how and if an index is returned
    # composite_index: bool = True,
    composite_index_fill_value: TLabel = None

    if is_fill_value_factory_initializer(fill_value):
        raise InvalidFillValue(fill_value, 'join')

    left_index = frame._index
    right_index = other._index
    fill_value_dtype = dtype_from_element(fill_value)

    #-----------------------------------------------------------------------
    # find matches

    if left_depth_level is None and left_columns is None:
        raise RuntimeError('Must specify one or both of left_depth_level and left_columns.')
    if right_depth_level is None and right_columns is None:
        raise RuntimeError('Must specify one or both of right_depth_level and right_columns.')

    # reduce the targets to 2D arrays; possible coercion in some cases, but seems inevitable as we will be doing row-wise comparisons
    target_left = TypeBlocks.from_blocks(
            arrays_from_index_frame(frame, left_depth_level, left_columns)).values
    target_right = TypeBlocks.from_blocks(
            arrays_from_index_frame(other, right_depth_level, right_columns)).values

    if target_left.shape[1] != target_right.shape[1]:
        raise RuntimeError('left and right selections must be the same width.')

    # Find matching pairs. Get iloc of left to iloc of right.

    is_many = False # one to many or many to many

    map_iloc: tp.Dict[int, np.ndarray[tp.Any, np.dtype[np.int_]]] = {}
    seen = set() # this stores

    # NOTE: this could be optimized by always iterating over the shorter target

    for idx_left, row_left in enumerate(target_left):
        # Get 1D vector showing matches along right's full heigh
        with WarningsSilent():
            matched = row_left == target_right
        if matched is False:
            continue
        matched = matched.all(axis=1)
        if not matched.any():
            continue
        # convert Booleans to integer positions
        matched_idx = np.flatnonzero(matched)
        if not is_many: # if user did not select composite index
            if len(matched_idx) > 1:
                is_many = True
            elif len(matched_idx) == 1:
                if matched_idx[0] in seen:
                    is_many = True
                seen.add(matched_idx[0])
        # build up a dictionary of left ilocs to an integer array of right matches
        # note that if row_left is the same as a previous row_left, we duplicate the matched_idx
        map_iloc[idx_left] = matched_idx

    #-----------------------------------------------------------------------
    # store collections of matches, derive final index

    left_loc_set: tp.Set[TLabel] = set() # all left loc labels that match
    right_loc_set: tp.Set[TLabel] = set() # all right loc labels that match
    many_loc: tp.List[Pair] = []

    cifv = composite_index_fill_value

    # NOTE: doing selection and using iteration (from set, and with zip, below) reduces chances for type coercion in IndexHierarchy
    left_loc = left_index[list(map_iloc.keys())]

    # iter over idx_left, matched_idx in right, left loc labels
    for (k, v), left_loc_element in zip(map_iloc.items(), left_loc):
        left_loc_set.add(left_loc_element)

        right_loc_part = right_index.values[v]
        if right_loc_part.ndim == 2:
            right_loc_set.update(array2d_to_tuples(right_loc_part))
        else:
            right_loc_set.update(right_loc_part) # iter 1D array

        if is_many:
            many_loc.extend(Pair(p) for p in product((left_loc_element,), right_loc_part))

    #-----------------------------------------------------------------------
    # get final_index; if is_many is True, many_loc (and Pair instances) will be used
    final_index: Index[tp.Any]

    if join_type is Join.INNER:
        if is_many:
            final_index = Index(many_loc)
        else: # just those matched from the left, which are also on right
            final_index = Index(left_loc)
    elif join_type is Join.LEFT:
        if is_many:
            extend = (PairLeft((x, cifv))
                    for x in left_index if x not in left_loc_set)
            # What if we are extending an index that already has a tuple
            final_index = Index(chain(many_loc, extend))
        else:
            final_index = left_index #type: ignore
    elif join_type is Join.RIGHT:
        if is_many:
            extend = (PairRight((cifv, x))
                    for x in right_index if x not in right_loc_set)
            final_index = Index(chain(many_loc, extend))
        else:
            final_index = right_index #type: ignore
    elif join_type is Join.OUTER:
        extend_left = (PairLeft((x, cifv))
                for x in left_index if x not in left_loc_set)
        extend_right = (PairRight((cifv, x))
                for x in right_index if x not in right_loc_set)
        if is_many:
            # must revese the many_loc so as to preserent right id first
            final_index = Index(chain(many_loc, extend_left, extend_right))
        else:
            final_index = left_index.union(right_index) #type: ignore
    else:
        raise NotImplementedError(f'index source must be one of {tuple(Join)}')

    #-----------------------------------------------------------------------
    # construct final frame
    final: TFrameGOAny
    if not is_many:
        final = FrameGO(index=final_index)
        left_column_labels = (left_template.format(c) for c in frame.columns)
        final.extend(frame.relabel(columns=left_column_labels), fill_value=fill_value)
        # build up a Series for each new column
        for idx_col, col in enumerate(other.columns):
            values = []
            for loc in final_index:
                # what if loc is in both left and rihgt?
                if loc in left_index and left_index._loc_to_iloc(loc) in map_iloc:
                    iloc = map_iloc[left_index._loc_to_iloc(loc)] #type: ignore
                    assert len(iloc) == 1 # not is_many, so all have to be length 1
                    values.append(other._extract_iloc((iloc[0], idx_col)))
                elif loc in right_index:
                    values.append(other._extract_loc((loc, col)))
                else:
                    values.append(fill_value)
            final[right_template.format(col)] = values

        if include_index:
            return final.to_frame()
        return final.to_frame().relabel(IndexAutoFactory)

    # From here, is_many is True
    row_key: tp.List[int]  = []
    final_index_left = []
    for p in final_index:
        if p.__class__ is Pair: # in both
            iloc = left_index._loc_to_iloc(p[0]) #type: ignore
            row_key.append(iloc) #type: ignore
            final_index_left.append(p)
        elif p.__class__ is PairLeft:
            row_key.append(left_index._loc_to_iloc(p[0])) #type: ignore
            final_index_left.append(p)

    # extract potentially repeated rows
    tb = frame._blocks._extract(row_key=row_key)
    left_column_labels = (left_template.format(c) for c in frame.columns)

    final = FrameGO(tb,
            index=Index(final_index_left),
            columns=left_column_labels,
            own_data=True)

    # only do this if we have PairRight above
    if len(final_index_left) < len(final_index):
        final = final.reindex(final_index, fill_value=fill_value)

    # populate from right columns
    # NOTE: find optimized path to avoid final_index iteration per column in all scenarios

    other_dtypes = other.dtypes.values
    for idx_col, col in enumerate(other.columns):
        array = np.empty(len(final_index), dtype=other_dtypes[idx_col])
        resolved_dtype = resolve_dtype(array.dtype, fill_value_dtype)

        for i, pair in enumerate(final_index):
            # NOTE: we used to support pair being something other than a Pair subclass (which would append fill_value to array), but it appears that if is_many is True, each value in final_index will be a Pair instance
            # assert isinstance(pair, Pair)
            if pair.__class__ is PairRight: # get from right
                array[i] = other._extract(right_index._loc_to_iloc(pair[1]), idx_col) #type: ignore
            elif pair.__class__ is PairLeft:
                # get from left, but we do not have col, so fill value
                if resolved_dtype != array.dtype:
                    array = array.astype(resolved_dtype)
                array[i] = fill_value
            else:
                array[i] = other._extract(right_index._loc_to_iloc(pair[1]), idx_col) # type: ignore

        array.flags.writeable = False
        final[right_template.format(col)] = array

    if include_index:
        return final.to_frame()
    return final.to_frame().relabel(IndexAutoFactory)





# def join_sort(left: 'Frame',
#         right: 'Frame', # support a named Series as a 1D frame?
#         *,
#         # join_type: Join, # intersect, left, right, union,
#         left_depth_level: tp.Optional[TDepthLevel] = None,
#         left_columns: TLocSelector = None,
#         right_depth_level: tp.Optional[TDepthLevel] = None,
#         right_columns: TLocSelector = None,
#         left_template: str = '{}',
#         right_template: str = '{}',
#         fill_value: tp.Any = np.nan,
#         composite_index: bool = True,
#         composite_index_fill_value: TLabel = None,
#         ) -> 'Frame':

#     from static_frame.core.frame import Frame

#     if is_fill_value_factory_initializer(fill_value):
#         raise InvalidFillValue(fill_value, 'join')

#     left_index = left.index
#     right_index = right.index

#     # NOTE: creating a new TypeBlocks instance as we might be combining index and normal columns; keep TypeBlocks for sorting;
#     # NOTE: might optimize for single-arrays, operating only an array
#     l_targets = TypeBlocks.from_blocks(
#             arrays_from_index_frame(left, left_depth_level, left_columns))
#     r_targets = TypeBlocks.from_blocks(
#             arrays_from_index_frame(right, right_depth_level, right_columns))

#     if l_targets.shape[1] != r_targets.shape[1]:
#         raise RuntimeError('left and right selections must be the same width.')
#     target_depth = l_targets.shape[1]

#     l_target_sorted, l_order = l_targets.sort(1, key=NULL_SLICE)
#     r_target_sorted, r_order = r_targets.sort(1, key=NULL_SLICE)
#     l_blocks_sorted = left._blocks._extract(l_order)
#     r_blocks_sorted = right._blocks._extract(r_order)

#     get_col_fill_value = lambda i, dtype: FILL_VALUE_AUTO_DEFAULT[dtype]
#     l_shifted = TypeBlocks.from_blocks(
#             l_target_sorted._shift_blocks_fill_by_callable(
#             row_shift=1,
#             column_shift=0,
#             wrap=False,
#             get_col_fill_value=get_col_fill_value
#             ))

#     r_shifted = TypeBlocks.from_blocks(
#             r_target_sorted._shift_blocks_fill_by_callable(
#             row_shift=1,
#             column_shift=0,
#             wrap=False,
#             get_col_fill_value=get_col_fill_value
#             ))

#     # ignore first comparison because it was filled
#     l_tt = (l_target_sorted != l_shifted).values.any(axis=1)
#     l_tt[0] = False
#     l_transitions = np.flatnonzero(l_tt)

#     r_tt = (r_target_sorted != r_shifted).values.any(axis=1)
#     r_tt[0] = False
#     r_transitions = np.flatnonzero(r_tt)

#     # not sure if we need this...
#     # l_unique = set((0,))
#     # l_unique.update(l_transitions)
#     def get_slices(
#             transitions: np.ndarray,
#             targets: TypeBlocks
#             ) -> tp.Tuple[tp.List[tp.Tuple[tp.Tuple[tp.Any, ...], slice]], bool]:
#         slices = [] # maybe a dictionary?
#         multi = False
#         start = 0
#         for t in transitions:
#             slc = slice(start, t)
#             if multi is False and t - start > 1:
#                 multi = True
#             slices.append((tuple(targets.iter_row_elements(start)), slc))
#             # can determine if any is more than 1
#             start = t

#         slc = slice(start, len(targets))
#         if multi is False and t - start > 1:
#             multi = True
#         slices.append((tuple(targets.iter_row_elements(start)), slc))

#         return slices, multi

#     # NOTE: while getting slices we can find out if either left/right has a target with more than one row; this indicates a many situation
#     l_slices, l_multi = get_slices(l_transitions, l_target_sorted)
#     r_slices, r_multi = get_slices(r_transitions, r_target_sorted)
#     r_slices_iter = iter(r_slices)

#     for l_target, l_slice in l_slices:
#         if target_depth == 1:
#             target = l_target[0]
#         else:
#             raise NotImplementedError()

#         l_sel = l_blocks_sorted._extract(l_slice)

#         match = False
#         for r_target, r_slice in r_slices_iter:
#             if l_target == r_target:
#                 match = True
#                 break

#         if match:
#             # for every col on the left, match one or more on the right
#             pass

#         # import ipdb; ipdb.set_trace()


#     return Frame() # temp



