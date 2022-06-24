import typing as tp
from itertools import chain
from itertools import product

import numpy as np

from static_frame.core.util import GetItemKeyType
from static_frame.core.util import DepthLevelSpecifier
from static_frame.core.util import Join
from static_frame.core.container_util import arrays_from_index_frame
from static_frame.core.container_util import is_fill_value_factory_initializer
from static_frame.core.util import Pair
from static_frame.core.util import PairLeft
from static_frame.core.util import PairRight
from static_frame.core.exception import InvalidFillValue
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.util import WarningsSilent
from static_frame.core.index import Index


if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame

def join(frame: 'Frame',
        other: 'Frame', # support a named Series as a 1D frame?
        *,
        join_type: Join, # intersect, left, right, union,
        left_depth_level: tp.Optional[DepthLevelSpecifier] = None,
        left_columns: GetItemKeyType = None,
        right_depth_level: tp.Optional[DepthLevelSpecifier] = None,
        right_columns: GetItemKeyType = None,
        left_template: str = '{}',
        right_template: str = '{}',
        fill_value: tp.Any = np.nan,
        composite_index: bool = True,
        composite_index_fill_value: tp.Hashable = None,
        ) -> 'Frame':

    from static_frame.core.frame import FrameGO

    if is_fill_value_factory_initializer(fill_value):
        raise InvalidFillValue(fill_value, 'join')

    left_index = frame.index
    right_index = other.index

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
    # If composite_index is True, is_many is True, if False, need to check if it is possible to not havea composite index.
    is_many = composite_index # one to many or many to many

    map_iloc = {}
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

    if not composite_index and is_many:
        raise RuntimeError('A composite index is required in this join.')

    #-----------------------------------------------------------------------
    # store collections of matches, derive final index

    left_loc_set = set() # all left loc labels that match
    right_loc_set = set() # all right loc labels that match
    many_loc = []

    cifv = composite_index_fill_value

    # NOTE: doing selection and using iteration (from set, and with zip, below) reduces chances for type coercion in IndexHierarchy
    left_loc = left_index[list(map_iloc.keys())]

    # iter over idx_left, matched_idx in right, left loc labels
    for (k, v), left_loc_element in zip(map_iloc.items(), left_loc):
        left_loc_set.add(left_loc_element)

        right_loc_part = right_index.values[v]
        right_loc_set.update(right_loc_part)

        if is_many:
            many_loc.extend(Pair(p) for p in product((left_loc_element,), right_loc_part))

    #-----------------------------------------------------------------------
    # get final_index; if is_many is True, many_loc (and Pair instances) will be used

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
            final_index = left_index
    elif join_type is Join.RIGHT:
        if is_many:
            extend = (PairRight((cifv, x))
                    for x in right_index if x not in right_loc_set)
            final_index = Index(chain(many_loc, extend))
        else:
            final_index = right_index
    elif join_type is Join.OUTER:
        extend_left = (PairLeft((x, cifv))
                for x in left_index if x not in left_loc_set)
        extend_right = (PairRight((cifv, x))
                for x in right_index if x not in right_loc_set)
        if is_many:
            # must revese the many_loc so as to preserent right id first
            final_index = Index(chain(many_loc, extend_left, extend_right))
        else:
            final_index = left_index.union(right_index)
    else:
        raise NotImplementedError(f'index source must be one of {tuple(Join)}')

    #-----------------------------------------------------------------------
    # construct final frame

    if not is_many:
        final = FrameGO(index=final_index)
        left_columns = (left_template.format(c) for c in frame.columns)
        final.extend(frame.relabel(columns=left_columns), fill_value=fill_value)
        # build up a Series for each new column
        for idx_col, col in enumerate(other.columns):
            values = []
            for loc in final_index:
                # what if loc is in both left and rihgt?
                if loc in left_index and left_index._loc_to_iloc(loc) in map_iloc:
                    iloc = map_iloc[left_index._loc_to_iloc(loc)]
                    assert len(iloc) == 1 # not is_many, so all have to be length 1
                    values.append(other.iloc[iloc[0], idx_col])
                elif loc in right_index:
                    values.append(other.loc[loc, col])
                else:
                    values.append(fill_value)
            final[right_template.format(col)] = values
        return final.to_frame()

    # From here, is_many is True
    row_key = []
    final_index_left = []
    for p in final_index:
        if p.__class__ is Pair: # in both
            iloc = left_index._loc_to_iloc(p[0])
            row_key.append(iloc)
            final_index_left.append(p)
        elif p.__class__ is PairLeft:
            row_key.append(left_index._loc_to_iloc(p[0]))
            final_index_left.append(p)

    # extract potentially repeated rows
    tb = frame._blocks._extract(row_key=row_key)
    left_columns = (left_template.format(c) for c in frame.columns)

    final = FrameGO(tb,
            index=Index(final_index_left),
            columns=left_columns,
            own_data=True)

    # only do this if we have PairRight above
    if len(final_index_left) < len(final_index):
        final = final.reindex(final_index, fill_value=fill_value)

    # populate from right columns
    # NOTE: find optimized path to avoid final_index iteration per column in all scenarios

    for idx_col, col in enumerate(other.columns):
        values = []
        for pair in final_index:
            # NOTE: we used to support pair being something other than a Pair subclass (which would append fill_value to values), but it appears that if is_many is True, each value in final_index will be a Pair instance
            # assert isinstance(pair, Pair)
            loc_left, loc_right = pair
            if pair.__class__ is PairRight: # get from right
                values.append(other.loc[loc_right, col])
            elif pair.__class__ is PairLeft:
                # get from left, but we do not have col, so fill value
                values.append(fill_value)
            else: # is this case needed?
                values.append(other.loc[loc_right, col])

        final[right_template.format(col)] = values
    return final.to_frame()
