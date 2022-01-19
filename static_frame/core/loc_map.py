import typing as tp
from functools import reduce

import numpy as np

from static_frame.core.exception import LocInvalid

from static_frame.core.util import SLICE_ATTRS
from static_frame.core.util import SLICE_START_ATTR
from static_frame.core.util import SLICE_STEP_ATTR
from static_frame.core.util import SLICE_STOP_ATTR
from static_frame.core.util import OPERATORS
from static_frame.core.util import DTYPE_OBJECTABLE_DT64_UNITS
from static_frame.core.util import EMPTY_SLICE
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import DTYPE_DATETIME_KIND
from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import NULL_SLICE

from static_frame.core.exception import LocEmpty

TypePos = tp.Optional[int]
LocEmptyInstance = LocEmpty()

class LocMap:

    @staticmethod
    def map_slice_args(
            label_to_pos: tp.Callable[[tp.Iterable[tp.Hashable]], int],
            key: slice,
            labels: tp.Optional[np.ndarray] = None,
            offset: tp.Optional[int] = 0
            ) -> tp.Iterator[tp.Union[int, None]]:
        '''Given a slice ``key`` and a label-to-position mapping, yield each integer argument necessary to create a new iloc slice. If the ``key`` defines a region with no constituents, raise ``LocEmpty``

        Args:
            label_to_pos: callable into mapping (can be a get() method from a dictionary)
        '''
        # NOTE: it is expected that NULL_SLICE is alreayd identified
        offset_apply = not offset is None
        labels_astype: tp.Optional[np.ndarray] = None

        for field in SLICE_ATTRS:
            attr = getattr(key, field)
            if attr is None:
                yield None

            elif attr.__class__ is np.datetime64:
                if field == SLICE_STEP_ATTR:
                    raise RuntimeError(f'Step cannot be {attr}')
                # if we match the same dt64 unit, simply use label_to_pos, increment stop
                if attr.dtype == labels.dtype: # type: ignore
                    pos: TypePos = label_to_pos(attr)
                    if pos is None:
                        # if same type, and that atter is not in labels, we fail, just as we do in then non-datetime64 case. Only when datetimes are given in a different unit are we "loose" about matching.
                        raise LocInvalid('Invalid loc given in a slice', attr, field)
                    if field == SLICE_STOP_ATTR:
                        pos += 1 # stop is inclusive
                elif field == SLICE_START_ATTR:
                    # NOTE: as an optimization only for the start attr, we can try to convert attr to labels unit and see if there is a match; this avoids astyping the entire labels array
                    pos: TypePos = label_to_pos(attr.astype(labels.dtype)) #type: ignore
                    if pos is None: # we did not find a start position
                        labels_astype = labels.astype(attr.dtype) #type: ignore
                        matches = np.flatnonzero(labels_astype == attr)
                        if len(matches):
                            pos = matches[0]
                        else:
                            raise LocEmptyInstance
                elif field == SLICE_STOP_ATTR:
                    # NOTE: we do not want to convert attr to labels dtype and take the match as we want to get the last of all possible matches of labels at the attr unit
                    # NOTE: try to re-use labels_astype if possible
                    if labels_astype is None or labels_astype.dtype != attr.dtype:
                        labels_astype = labels.astype(attr.dtype) #type: ignore
                    matches = np.flatnonzero(labels_astype == attr)
                    if len(matches):
                        pos = matches[-1] + 1
                    else:
                        raise LocEmptyInstance

                if offset_apply:
                    pos += offset #type: ignore
                yield pos

            else:
                if field != SLICE_STEP_ATTR:
                    pos = label_to_pos(attr)
                    if pos is None:
                        # NOTE: could raise LocEmpty() to silently handle this
                        raise LocInvalid('Invalid loc given in a slice', attr, field)
                    if offset_apply:
                        pos += offset #type: ignore
                else: # step
                    pos = attr # should be an integer
                if field == SLICE_STOP_ATTR:
                    # loc selections are inclusive, so iloc gets one more
                    pos += 1 #type: ignore
                yield pos

    @classmethod
    def loc_to_iloc(cls, *,
            label_to_pos: tp.Dict[tp.Hashable, int],
            labels: np.ndarray,
            positions: np.ndarray,
            key: GetItemKeyType,
            offset: tp.Optional[int] = None,
            partial_selection: bool = False,
            ) -> GetItemKeyType:
        '''
        Note: all SF objects (Series, Index) need to be converted to basic types before being passed as `key` to this function.

        Args:
            offset: in the context of an IndexHierarchical, the iloc positions returned from this funcition need to be shifted.
            partial_selection: if True and key is an iterable of labels that includes labels not in the mapping, available matches will be returned rather than raising.
        Returns:
            An integer mapped slice, or GetItemKey type that is based on integers, compatible with TypeBlocks
        '''
        # NOTE: ILoc is handled prior to this call, in the Index._loc_to_iloc method
        offset_apply = not offset is None

        if key.__class__ is slice:
            if key == NULL_SLICE:
                if offset_apply:
                    # when offset is defined (even if it is zero), null slice is not sufficiently specific; need to convert to an explicit slice relative to the offset
                    return slice(offset, len(positions) + offset) #type: ignore
                else:
                    return NULL_SLICE
            try:
                return slice(*cls.map_slice_args(
                        label_to_pos.get, #type: ignore
                        key,
                        labels,
                        offset)
                        )
            except LocEmpty:
                return EMPTY_SLICE

        labels_is_dt64 = labels.dtype.kind == DTYPE_DATETIME_KIND

        if key.__class__ is np.datetime64:
            # if we have a single dt64, convert this to the key's unit and do a Boolean selection if the key is a less-granular unit
            if (labels.dtype == DTYPE_OBJECT
                    and np.datetime_data(key.dtype)[0] in DTYPE_OBJECTABLE_DT64_UNITS): #type: ignore
                key = key.astype(DTYPE_OBJECT) #type: ignore
            elif labels_is_dt64 and key.dtype < labels.dtype: #type: ignore
                key = labels.astype(key.dtype) == key #type: ignore
            # if not different type, keep it the same so as to do a direct, single element selection

        is_array = key.__class__ is np.ndarray
        is_list = isinstance(key, list)

        # can be an iterable of labels (keys) or an iterable of Booleans
        if is_array or is_list:
            if is_array and key.dtype.kind == DTYPE_DATETIME_KIND: #type: ignore
                if (labels.dtype == DTYPE_OBJECT
                        and np.datetime_data(key.dtype)[0] in DTYPE_OBJECTABLE_DT64_UNITS): #type: ignore
                    # if key is dt64 and labels are object, then for objectable units we can convert key to object to permit matching in the AutoMap
                    # NOTE: tolist() is expected to be faster than astype object for smaller collections
                    key = key.tolist() #type: ignore
                    is_array = False
                    is_list = True
                elif labels_is_dt64 and key.dtype < labels.dtype: #type: ignore
                    # change the labels to the dt64 dtype, i.e., if the key is years, recast the labels as years, and do a Boolean selection of everything that matches each key
                    labels_ref = labels.astype(key.dtype) # type: ignore
                    # NOTE: this is only correct if both key and labels are dt64, and key is a less granular unit, as the order in the key and will not be used
                    # let Boolean key advance to next branch
                    key = reduce(OPERATORS['__or__'], (labels_ref == k for k in key)) # type: ignore

            if is_array and key.dtype == DTYPE_BOOL: #type: ignore
                if offset_apply:
                    return positions[key] + offset
                return positions[key]

            # map labels to integer positions, return a list of integer positions
            # NOTE: we may miss the opportunity to identify contiguous keys and extract a slice
            # NOTE: we do more branching here to optimize performance
            if partial_selection:
                if offset_apply:
                    return [label_to_pos[k] + offset for k in key if k in label_to_pos] #type: ignore
                return [label_to_pos[k] for k in key if k in label_to_pos] # type: ignore
            if offset_apply:
                return [label_to_pos[k] + offset for k in key] #type: ignore
            return [label_to_pos[k] for k in key] # type: ignore

        # if a single element (an integer, string, or date, we just get the integer out of the map
        if offset_apply:
            return label_to_pos[key] + offset #type: ignore
        return label_to_pos[key] #type: ignore

