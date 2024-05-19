from __future__ import annotations

from itertools import chain

import numpy as np
import typing_extensions as tp
from arraykit import resolve_dtype

# from static_frame.core.container_util import FILL_VALUE_AUTO_DEFAULT
from static_frame.core.container_util import arrays_from_index_frame
from static_frame.core.container_util import get_col_fill_value_factory
from static_frame.core.index import Index
from static_frame.core.index_base import IndexBase
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import EMPTY_ARRAY_INT
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import Join
from static_frame.core.util import TDepthLevel
from static_frame.core.util import TLocSelector
from static_frame.core.util import WarningsSilent
from static_frame.core.util import dtype_from_element

TNDArrayAny = np.ndarray[tp.Any, tp.Any]
TNDArrayInt = np.ndarray[tp.Any, np.dtype[np.int64]]
TDtypeAny = np.dtype[tp.Any]

if tp.TYPE_CHECKING:
    from static_frame.core.generic_aliases import TFrameAny  # pylint: disable=W0611 #pragma: no cover

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

            '_len',
            '_is_many',
            '_src_connected',
            '_dst_connected',
            )

    def __init__(self, src_len: int, dst_len: int) -> None:
        self._len = 0 # position in the final
        self._is_many = False
        self._src_connected = 0
        self._dst_connected = 0

        self._src_match = np.full(src_len, False)
        self._dst_match = np.full(dst_len, False)

        self._src_one_from: tp.List[int] = []
        self._src_one_to: tp.List[int] = []

        self._dst_one_from: tp.List[int] = []
        self._dst_one_to: tp.List[int] = []

        self._src_many_from: tp.List[int] = []
        self._src_many_to: tp.List[slice] = []

        self._dst_many_from: tp.List[TNDArrayInt] = []
        self._dst_many_to: tp.List[slice] = []


    def register_one(self, src_from: int, dst_from: int) -> None:
        '''Register a source position `src_from` and automatically register the destination position.
        '''
        if src_matched := src_from >= 0:
            self._src_one_from.append(src_from)
            self._src_one_to.append(self._len)
            self._src_connected += 1

        if dst_matched := dst_from >= 0:
            self._dst_one_from.append(dst_from)
            self._dst_one_to.append(self._len)
            self._dst_connected += 1

        if src_matched and dst_matched:
            # if we have seen this value before in src
            if not self._is_many and (self._src_match[src_from] or self._dst_match[dst_from]):
                self._is_many = True
            self._src_match[src_from] = True
            self._dst_match[dst_from] = True

        self._len += 1

    def register_unmapped_dst(self) -> None:
        if self._dst_match.sum() < len(self._dst_match):
            idx, = np.nonzero(~self._dst_match)
            for dst_i in idx:
                self.register_one(-1, dst_i)

    def register_many(self,
            src_from: int,
            dst_from: TNDArrayInt,
            ) -> None:
        '''Register a source position `src_from` and automatically register the destination positions based on `dst_from`. Length of `dst_from` should always be greater than 1.
        '''
        # assert isinstance(src_from, int)
        # assert not isinstance(dst_from, int)

        increment = len(dst_from)
        s = slice(self._len, self._len + increment)

        self._src_many_from.append(src_from)
        self._src_many_to.append(s)

        self._dst_many_from.append(dst_from)
        self._dst_many_to.append(s)

        self._src_match[src_from] = True
        self._dst_match[dst_from] = True

        self._len += increment
        self._is_many = True
        self._src_connected += increment
        self._dst_connected += increment

    #---------------------------------------------------------------------------
    # after registration is complete, these metrics can be used; they might all be calculated and stored with a finalize() method?

    # def unmatched_src(self) -> bool:
    #     return self._src_match.sum() < len(self._src_match)

    # def unmatched_dst(self) -> bool:
    #     return self._dst_match.sum() < len(self._dst_match) # type: ignore

    def src_no_fill(self) -> bool:
        return self._src_connected == self._len

    def dst_no_fill(self) -> bool:
        return self._dst_connected == self._len

    def is_many(self) -> bool:
        return self._is_many

    #---------------------------------------------------------------------------

    def _transfer_from_src(self,
            array_from: TNDArrayAny,
            array_to: TNDArrayAny,
            ) -> TNDArrayAny:
        # NOTE: array_from, array_to here might be any type, including object types
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
        array_to = np.empty(self._len, dtype=array_from.dtype)
        return self._transfer_from_src(array_from, array_to)

    def map_src_fill(self,
            array_from: TNDArrayAny,
            fill_value: tp.Any,
            fill_value_dtype: TDtypeAny,
            ) -> TNDArrayAny:
        '''Apply all mappings from `array_from` to `array_to`.
        '''
        resolved_dtype = resolve_dtype(array_from.dtype, fill_value_dtype)
        array_to = np.full(self._len, fill_value, dtype=resolved_dtype)
        return self._transfer_from_src(array_from, array_to)

    def map_dst_no_fill(self,
            array_from: TNDArrayAny,
            ) -> TNDArrayAny:
        '''Apply all mappings from `array_from` to `array_to`.
        '''
        array_to = np.empty(self._len, dtype=array_from.dtype)
        return self._transfer_from_dst(array_from, array_to)

    def map_dst_fill(self,
            array_from: TNDArrayAny,
            fill_value: tp.Any,
            fill_value_dtype: TDtypeAny,
            ) -> TNDArrayAny:
        '''Apply all mappings from `array_from` to `array_to`.
        '''
        resolved_dtype = resolve_dtype(array_from.dtype, fill_value_dtype)
        array_to = np.full(self._len, fill_value, dtype=resolved_dtype)
        return self._transfer_from_dst(array_from, array_to)

#-------------------------------------------------------------------------------

def _join_trimap_target_one(
        src_target: TNDArrayAny,
        dst_target: TNDArrayAny,
        join_type: Join,
        ) -> TriMap:

    src_element_to_matched_idx = dict() # make this an LRU
    tm = TriMap(len(src_target), len(dst_target))

    with WarningsSilent():
        for src_i, src_element in enumerate(src_target):
            if src_element not in src_element_to_matched_idx:
                # TODO: recycle the matched Boolean array
                matched = src_element == dst_target
                if matched is False:
                    matched_idx = EMPTY_ARRAY_INT
                    matched_len = 0
                else: # convert Booleans to integer positions
                    matched_idx, = np.nonzero(matched) # unpack
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

    # if join_type is Join.OUTER and tm.unmatched_dst():
    #     for dst_i in tm.unmatched_dst_indices():
    #         tm.register_one(-1, dst_i)
    if join_type is Join.OUTER:
        tm.register_unmapped_dst()
    return tm

def _join_trimap_target_many(
        src_target: list[TNDArrayAny],
        dst_target: list[TNDArrayAny],
        join_type: Join,
        target_depth: int,
        ) -> TriMap:

    src_element_to_matched_idx = dict() # make this an LRU
    tm = TriMap(len(src_target[0]), len(dst_target[0]))
    matched_per_depth = np.empty((len(dst_target[0]), target_depth), dtype=DTYPE_BOOL)

    with WarningsSilent():
        # by iterating elements and comparing one depth at time, we avoid forcing any type conversions
        for src_i, src_elements in enumerate(zip(*src_target)):
            if src_elements not in src_element_to_matched_idx:
                for d, e in enumerate(src_elements):
                    matched_per_depth[NULL_SLICE, d] = e == dst_target[d]
                matched = matched_per_depth.all(axis=1)
                matched_idx, = np.nonzero(matched) # unpack
                matched_len = len(matched_idx)

                src_element_to_matched_idx[src_elements] = (matched_idx, matched_len)
            else:
                matched_idx, matched_len = src_element_to_matched_idx[src_elements]

            if matched_len == 0:
                if join_type is not Join.INNER:
                    tm.register_one(src_i, -1)
            elif matched_len == 1:
                tm.register_one(src_i, matched_idx[0])
            else: # one source value to many positions
                tm.register_many(src_i, matched_idx)

    # if join_type is Join.OUTER and tm.unmatched_dst():
    #     for dst_i in tm.unmatched_dst_indices():
    #         tm.register_one(-1, dst_i)
    if join_type is Join.OUTER:
        tm.register_unmapped_dst()

    return tm


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
    #-----------------------------------------------------------------------
    # find matches
    if not isinstance(join_type, Join):
        raise NotImplementedError(f'`join_type` must be one of {tuple(Join)}')

    if left_depth_level is None and left_columns is None:
        raise RuntimeError('Must specify one or both of left_depth_level and left_columns.')
    if right_depth_level is None and right_columns is None:
        raise RuntimeError('Must specify one or both of right_depth_level and right_columns.')

    # reduce the targets to 2D arrays; possible coercion in some cases, but seems inevitable as we will be doing row-wise comparisons
    left_target: TNDArrayAny | list[TNDArrayAny] = list(
            arrays_from_index_frame(frame, left_depth_level, left_columns))
    right_target: TNDArrayAny | list[TNDArrayAny] = list(
            arrays_from_index_frame(other, right_depth_level, right_columns))

    if (target_depth := len(left_target)) != len(right_target):
        raise RuntimeError('left and right selections must be the same width.')

    if target_depth == 1: # reshape into 1D arrays
        left_target = left_target[0]
        right_target = right_target[0]

    # Find matching pairs. Get iloc of left to iloc of right.
    left_frame = frame
    right_frame = other
    left_index = frame.index
    right_index = other.index

    if join_type == Join.RIGHT:
        src_target = right_target
        dst_target = left_target
    else:
        src_target = left_target
        dst_target = right_target

    if target_depth == 1:
        tm = _join_trimap_target_one(src_target, dst_target, join_type) # type: ignore
    else:
        tm = _join_trimap_target_many(src_target, dst_target, join_type, target_depth) # type: ignore

    #---------------------------------------------------------------------------

    final_columns = list(chain(
            (left_template.format(c) for c in frame.columns),
            (right_template.format(c) for c in other.columns)
            ))
    # we must use post template column names as there might be name conflicts
    get_col_fill_value = get_col_fill_value_factory(fill_value, columns=final_columns)

    arrays = []
    # if we have matched all in src, we do not need fill values
    if join_type is Join.RIGHT:
        src_no_fill = tm.dst_no_fill
        dst_no_fill = tm.src_no_fill
        map_src_no_fill = tm.map_dst_no_fill
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

    #---------------------------------------------------------------------------
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

            left_index_values = left_index.values if left_index.depth == 1 else left_index.flat().values # type: ignore
            right_index_values = right_index.values if right_index.depth == 1 else right_index.flat().values # type: ignore

            # NOTE: not sure if src/dst arrangement is correct for right join
            if src_no_fill():
                labels_left = map_src_no_fill(left_index_values)
            else:
                labels_left = map_src_fill(left_index_values, None, DTYPE_OBJECT)

            if dst_no_fill():
                labels_right = map_dst_no_fill(right_index_values)
            else:
                labels_right = map_dst_fill(right_index_values, None, DTYPE_OBJECT)

            final_index = Index(zip(labels_left, labels_right))
    else:
        own_index = False
        final_index = None

    return Frame(TypeBlocks.from_blocks(arrays),
            columns=final_columns,
            index=final_index,
            own_data=True,
            own_index=own_index,
            )
