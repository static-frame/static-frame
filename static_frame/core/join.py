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
from static_frame.core.util import EMPTY_ARRAY_INT
from static_frame.core.util import Join
from static_frame.core.util import Pair
from static_frame.core.util import PairLeft
from static_frame.core.util import PairRight
from static_frame.core.util import TDepthLevel
from static_frame.core.util import TLabel
from static_frame.core.util import TLocSelector
from static_frame.core.util import WarningsSilent
from static_frame.core.util import dtype_from_element
from static_frame.core.util import DTYPE_OBJECT

TNDArrayAny = np.ndarray[tp.Any, tp.Any]
TNDArrayInt = np.ndarray[tp.Any, np.dtype[np.int64]]
TDtypeAny = np.dtype[tp.Any]

if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.frame import FrameGO  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.generic_aliases import TFrameAny  # pylint: disable=W0611 #pragma: no cover


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

    from static_frame.core.frame import Frame
    from static_frame.core.frame import FrameGO

    cifv: TLabel = None

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

    if (target_width := target_left.shape[1]) != target_right.shape[1]:
        raise RuntimeError('left and right selections must be the same width.')


    is_many = False # one to many or many to many

    # Find matching pairs. Get iloc of left to iloc of right.
    map_iloc: tp.Dict[int, np.ndarray[tp.Any, np.dtype[np.int_]]] = {}
    seen = set()

    with WarningsSilent():
        for idx_left, row_left in enumerate(target_left):
            # Get 1D vector showing matches along right's full heigh
            matched = row_left == target_right
            if matched is False:
                continue

            if target_width == 1: # matched can be reshaped
                matched = matched.reshape(len(matched))
            else:
                matched = matched.all(axis=1)
            # convert Booleans to integer positions, unpack tuple to one element
            matched_idx, = np.nonzero(matched)

            if not len(matched_idx):
                continue

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

    right_index_ndim = right_index.ndim
    # NOTE: doing selection and using iteration (from set, and with zip, below) instead of .values reduces chances for type coercion in IndexHierarchy
    left_loc_mapped = left_index[list(map_iloc.keys())]

    if is_many:
        right_loc_set = set() # all right loc labels that match
        many_loc: tp.List[Pair] = []

        if right_index_ndim != 1 and right_index._recache:
            right_index._update_array_cache()
            right_index_blocks = right_index._blocks # type: ignore

        # iter over idx_left, matched_idx in right, left loc labels
        for v, left_loc_element in zip(map_iloc.values(), left_loc_mapped):
            # NOTE: v is a 1D array that might have 1 or more integers, depending on correspondence
            if len(v) == 1:
                label = right_index[v[0]]
                right_loc_set.add(label)
                many_loc.append(Pair((left_loc_element, label)))
            else:
                if right_index_ndim == 1:
                    right_loc_part = right_index.values[v]
                else: # already called `_update_array_cache`
                    right_loc_part = list(right_index_blocks._extract(v).iter_row_tuples())

                right_loc_set.update(right_loc_part)
                many_loc.extend(Pair(p) for p in product((left_loc_element,), right_loc_part))

    #-----------------------------------------------------------------------
    # get final_index; if is_many is True, many_loc (and Pair instances) will be used
    final_index: Index[tp.Any]

    if join_type is Join.INNER:
        if is_many:
            final_index = Index(many_loc)
        else: # just those matched from the left, which are also on right
            final_index = Index(left_loc_mapped)
    elif join_type is Join.LEFT:
        if is_many:
            extend = (PairLeft((x, cifv))
                    for x in left_index if x not in left_loc_mapped)
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
        if is_many:
            extend_left = (PairLeft((x, cifv))
                    for x in left_index if x not in left_loc_mapped)
            extend_right = (PairRight((cifv, x))
                    for x in right_index if x not in right_loc_set)
            # must revese the many_loc so as to preserent right id first
            final_index = Index(chain(many_loc, extend_left, extend_right))
        else:
            final_index = left_index.union(right_index) #type: ignore
    else:
        raise NotImplementedError(f'index source must be one of {tuple(Join)}')

    final_len = len(final_index)

    #-----------------------------------------------------------------------
    # construct final frame
    final: TFrameAny
    if not is_many:
        final_column_labels = chain(
                (left_template.format(c) for c in frame.columns),
                (right_template.format(c) for c in other.columns)
                )
        blocks = frame.reindex(final_index, fill_value=fill_value)._blocks # steal this reference and mutate it

        if final_len <= len(map_iloc):
            # extend from `other``, only re-ordering by `map_iloc`
            blocks.extend(other._blocks._extract([v[0] for v in map_iloc.values()]))
            final = Frame(blocks,
                    columns=final_column_labels,
                    index=final_index,
                    own_data=True,
                    own_index=True,
                    )
            return final if include_index else final.relabel(IndexAutoFactory)

        # get iloc selections to assign values in other to destination in final array
        assign_to: tp.List[int] = []
        assign_from: tp.List[int] = []
        for idx_row, loc in enumerate(final_index):
            if loc in left_index:
                if loc in left_loc_mapped: # means this loc is in map_iloc
                    left_iloc = left_index._loc_to_iloc(loc)
                    assign_to.append(idx_row)
                    # `map_iloc` values are arrays of one value
                    assign_from.append(map_iloc[left_iloc][0]) # type: ignore
            else: # in right_index
                assign_to.append(idx_row)
                assign_from.append(right_index._loc_to_iloc(loc)) # type: ignore

        def gen() -> tp.Iterator[TNDArrayAny]:
            for col in other._blocks.axis_values():
                # as `final_index` > `map_iloc`, we will use `fill_value`
                resolved_dtype = resolve_dtype(col.dtype, fill_value_dtype)
                array = np.full(final_len, fill_value, dtype=resolved_dtype)
                array[assign_to] = col[assign_from]
                array.flags.writeable = False
                yield array

        blocks.extend(gen())
        final = Frame(blocks,
                columns=final_column_labels,
                index=final_index,
                own_data=True,
                own_index=True,
                )
        return final if include_index else final.relabel(IndexAutoFactory)

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
            own_data=True,
            own_index=True,
            )

    # only do this if we have PairRight above
    if len(final_index_left) < final_len:
        final = final.reindex(final_index, fill_value=fill_value)

    # populate from right columns
    # NOTE: find optimized path to avoid final_index iteration per column in all scenarios

    other_dtypes = other.dtypes.values
    for idx_col, col in enumerate(other.columns):
        array = np.empty(final_len, dtype=other_dtypes[idx_col])
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

#-------------------------------------------------------------------------------

class TriMap:
    '''Store mappings from a `src` array to the a final array, and from a `dst` array to a final array. Partition 1-to-1 mappings from 1-to-many and many-to-many mappings.
    '''
    __slots__ = (
            '_src_match',
            '_dst_match',

            '_src_one_to',
            '_src_one_from',
            '_dst_one_to',
            '_dst_one_from',

            '_src_many_from',
            '_src_many_to',
            '_dst_many_from',
            '_dst_many_to',

            '_i',
            '_is_many',
            )

    def __init__(self, src_len: int, dst_len: int) -> None:

        self._src_match = np.full(src_len, False)
        self._dst_match = np.full(dst_len, False)

        self._src_one_from: tp.List[int] = []
        self._src_one_to: tp.List[int] = []

        self._dst_one_from: tp.List[int] = []
        self._dst_one_to: tp.List[int] = []

        # many could be a dictionary but need arrays as keys
        self._src_many_from: tp.List[int | TNDArrayInt] = []
        self._src_many_to: tp.List[slice] = []

        self._dst_many_from: tp.List[int | TNDArrayInt] = []
        self._dst_many_to: tp.List[slice] = []

        self._i = 0 # position in the final
        self._is_many = False

    def register_one(self, src_from: int, dst_from: int) -> None:
        '''Register a source position `src_from` and automatically register the destination position.
        '''
        if src_matched := src_from >= 0:
            self._src_one_from.append(src_from)
            self._src_one_to.append(self._i)

        if dst_matched := dst_from >= 0:
            self._dst_one_from.append(dst_from)
            self._dst_one_to.append(self._i)

        if src_matched and dst_matched:
            # if we have seen this value before in src
            if not self._is_many and self._src_match[src_from]:
                self._is_many = True

            self._src_match[src_from] = True
            self._dst_match[dst_from] = True

        self._i += 1

    def register_many(self,
            src_from: int | TNDArrayInt,
            dst_from: TNDArrayInt,
            ) -> None:
        '''Register a source position `src_from` and automatically register the destination positions based on `dst_from`. Length of `dst_from` should always be greater than 1.
        '''
        increment = len(dst_from)
        s = slice(self._i, self._i + increment)

        self._src_many_from.append(src_from)
        self._src_many_to.append(s)

        self._dst_many_from.append(dst_from)
        self._dst_many_to.append(s)

        self._src_match[src_from] = True
        self._dst_match[dst_from] = True

        self._i += increment
        self._is_many = True

    #---------------------------------------------------------------------------
    def unmatched_src(self) -> bool:
        return self._src_match.sum() < len(self._src_match)

    def unmatched_dst(self) -> bool:
        return self._dst_match.sum() < len(self._dst_match)

    def src_no_fill(self) -> bool:
        return self._src_match.sum() == self._i

    def dst_no_fill(self) -> bool:
        return self._dst_match.sum() == self._i

    def unmatched_dst_indices(self) -> TNDArrayInt:
        idx, = np.nonzero(~self._dst_match)
        return idx

    def __len__(self) -> int:
        return self._i

    def is_many(self) -> bool:
        return self._is_many

    #---------------------------------------------------------------------------

    def _transfer_from_src(self,
            array_from: TNDArrayAny,
            array_to: TNDArrayAny,
            ) -> TNDArrayAny:
        # NOTE: array_from, array_to here might be any type, invluding object types
        array_to[self._src_one_to] = array_from[self._src_one_from]

        # if many_from, many_to are empty, this is a no-op
        for assign_from, assign_to in zip(self._src_many_from, self._src_many_to):
            array_to[assign_to] = array_from[assign_from]

        array_to.flags.writeable = False
        return array_to

    def _transfer_from_dst(self,
            array_from: TNDArrayAny,
            array_to: TNDArrayAny,
            ) -> TNDArrayAny:
        # NOTE: array_from, array_to here might be any type, including object types
        array_to[self._dst_one_to] = array_from[self._dst_one_from]

        # if many_from, many_to are empty, this is a no-op
        for assign_from, assign_to in zip(self._dst_many_from, self._dst_many_to):
            array_to[assign_to] = array_from[assign_from]

        array_to.flags.writeable = False
        return array_to

    def map_src_no_fill(self,
            array_from: TNDArrayAny,
            ) -> TNDArrayAny:
        '''Apply all mappings from `array_from` to `array_to`.
        '''
        array_to = np.empty(self._i, dtype=array_from.dtype)
        return self._transfer_from_src(array_from, array_to)

    def map_src_fill(self,
            array_from: TNDArrayAny,
            fill_value: tp.Any,
            fill_value_dtype: TDtypeAny,
            ) -> TNDArrayAny:
        '''Apply all mappings from `array_from` to `array_to`.
        '''
        resolved_dtype = resolve_dtype(array_from.dtype, fill_value_dtype)
        array_to = np.full(self._i, fill_value, dtype=resolved_dtype)
        return self._transfer_from_src(array_from, array_to)

    def map_dst_no_fill(self,
            array_from: TNDArrayAny,
            ) -> TNDArrayAny:
        '''Apply all mappings from `array_from` to `array_to`.
        '''
        array_to = np.empty(self._i, dtype=array_from.dtype)
        return self._transfer_from_dst(array_from, array_to)

    def map_dst_fill(self,
            array_from: TNDArrayAny,
            fill_value: tp.Any,
            fill_value_dtype: TDtypeAny,
            ) -> TNDArrayAny:
        '''Apply all mappings from `array_from` to `array_to`.
        '''
        resolved_dtype = resolve_dtype(array_from.dtype, fill_value_dtype)
        array_to = np.full(self._i, fill_value, dtype=resolved_dtype)
        return self._transfer_from_dst(array_from, array_to)



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

    from static_frame.core.frame import Frame

    # cifv: TLabel = None

    if is_fill_value_factory_initializer(fill_value):
        raise InvalidFillValue(fill_value, 'join')

    # TODO: support column-based fill values
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

    if (target_width := target_left.shape[1]) != target_right.shape[1]:
        raise RuntimeError('left and right selections must be the same width.')

    if target_width == 1: # reshape into 1D arrays
        target_left = target_left.reshape(len(target_left))
        target_right = target_right.reshape(len(target_right))

    # Find matching pairs. Get iloc of left to iloc of right.
    src_frame = frame
    dst_frame = other
    src_index = frame.index
    dst_index = other.index
    src_target = target_left
    dst_target = target_right

    src_element_to_matched_idx = dict() # make this an LRU
    tm = TriMap(len(src_target), len(dst_target))

    with WarningsSilent():
        for src_i, src_element in enumerate(src_target):
            # Get 1D vector showing matches along right's full heigh; this is expensive and can be cached, keyed by `src_element`. If src_element is an array (when target_width > 1) we cannot cache unless we convert that array into a tuple.
            if src_element not in src_element_to_matched_idx:
                matched = src_element == dst_target
                if matched is False:
                    matched_idx = EMPTY_ARRAY_INT
                    matched_len = 0
                else:
                    if target_width > 1: # matched is 2d
                        matched = matched.all(axis=1)
                    assert matched.ndim == 1
                    # convert Booleans to integer positions, unpack tuple to one element
                    matched_idx, = np.nonzero(matched)
                    matched_len = len(matched_idx)

                src_element_to_matched_idx[src_element] = (matched_idx, matched_len)
            else:
                matched_idx, matched_len = src_element_to_matched_idx[src_element]

            if matched_len == 0:
                if join_type is not Join.INNER:
                    tm.register_one(src_i, -1)
            elif matched_len == 1:
                tm.register_one(src_i, matched_idx[0])
            else: # one source value to many positions
                tm.register_many(src_i, matched_idx)

    # no matching or caching needed
    if join_type is Join.OUTER and tm.unmatched_dst():
        for dst_i in tm.unmatched_dst_indices():
            tm.register_one(-1, dst_i)

    #---------------------------------------------------------------------------
    arrays = []
    # if we have matched all in src, we do not need fill values
    if tm.src_no_fill():
        for proto in src_frame._blocks.axis_values():
            arrays.append(tm.map_src_no_fill(proto))
    else:
        for proto in src_frame._blocks.axis_values():
            arrays.append(tm.map_src_fill(proto, fill_value, fill_value_dtype))

    if tm.dst_no_fill():
        for proto in dst_frame._blocks.axis_values():
            arrays.append(tm.map_dst_no_fill(proto))
    else:
        for proto in dst_frame._blocks.axis_values():
            arrays.append(tm.map_dst_fill(proto, fill_value, fill_value_dtype))

    final_column_labels = chain(
            (left_template.format(c) for c in frame.columns),
            (right_template.format(c) for c in other.columns)
            )

    #---------------------------------------------------------------------------
    if include_index:
        own_index = True
        if join_type is not Join.OUTER and not tm.is_many():
            if join_type is Join.INNER:
                final_index = Index(tm.map_src_fill(src_index, None, DTYPE_OBJECT))
            elif join_type is Join.LEFT:
                final_index = src_index
            elif join_type is Join.RIGHT:
                final_index = dst_index
        # NOTE: the other scenario when we can have a non-tuple index is if only one value us coming from each side; this is probably not common
        else:
            # NOTE: will need to flatten hierarchical indices to tuples
            # the fill value can be varied
            labels_src = tm.map_src_fill(src_index, None, DTYPE_OBJECT)
            labels_dst = tm.map_dst_fill(dst_index, None, DTYPE_OBJECT)
            final_index = Index(zip(labels_src, labels_dst))
    else:
        own_index = False
        final_index = None

    return Frame(TypeBlocks.from_blocks(arrays),
            columns=final_column_labels,
            index=final_index,
            own_data=True,
            own_index=own_index,
            )
