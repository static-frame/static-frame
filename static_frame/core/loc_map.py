from __future__ import annotations

import itertools
import sys
from copy import deepcopy
from functools import reduce

import numpy as np
import typing_extensions as tp
from arraykit import array_deepcopy
from arraykit import first_true_1d
from arraykit import nonzero_1d
from arraymap import FrozenAutoMap  # pylint: disable = E0611
from arraymap import NonUniqueError  # pylint: disable=E0611

from static_frame.core.exception import ErrorInitIndexNonUnique
from static_frame.core.exception import LocEmpty
from static_frame.core.exception import LocInvalid
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import DTYPE_DATETIME_KIND
from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import DTYPE_OBJECTABLE_DT64_UNITS
from static_frame.core.util import DTYPE_UINT_DEFAULT
from static_frame.core.util import EMPTY_ARRAY_INT
from static_frame.core.util import EMPTY_FROZEN_AUTOMAP
from static_frame.core.util import EMPTY_SLICE
from static_frame.core.util import INT_TYPES
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import OPERATORS
from static_frame.core.util import SLICE_ATTRS
from static_frame.core.util import SLICE_START_ATTR
from static_frame.core.util import SLICE_STEP_ATTR
from static_frame.core.util import SLICE_STOP_ATTR
from static_frame.core.util import TILocSelector
from static_frame.core.util import TLabel
from static_frame.core.util import TLocSelector

if tp.TYPE_CHECKING:
    from static_frame.core.index import Index  # pylint: disable=W0611,C0412 # pragma: no cover

TNDArrayAny = np.ndarray[tp.Any, tp.Any]
TDtypeAny = np.dtype[tp.Any]

HierarchicalLocMapKey = tp.Union[TNDArrayAny, tp.Tuple[tp.Union[tp.Sequence[TLabel], TLabel], ...]]
_HLMap = tp.TypeVar('_HLMap', bound='HierarchicalLocMap')
TypePos = tp.Optional[int]
LocEmptyInstance = LocEmpty()


class FirstDuplicatePosition(KeyError):
    def __init__(self, first_dup: int) -> None:
        self.first_dup = first_dup


class LocMap:

    @staticmethod
    def map_slice_args(
            label_to_pos: tp.Callable[[tp.Iterable[TLabel]], int],
            key: slice,
            labels: tp.Optional[TNDArrayAny] = None,
            ) -> tp.Iterator[tp.Union[int, None]]:
        '''Given a slice ``key`` and a label-to-position mapping, yield each integer argument necessary to create a new iloc slice. If the ``key`` defines a region with no constituents, raise ``LocEmpty``

        Args:
            label_to_pos: callable into mapping (can be a get() method from a dictionary)
        '''
        # NOTE: it is expected that NULL_SLICE is already identified
        labels_astype: tp.Optional[TNDArrayAny] = None

        for field in SLICE_ATTRS:
            attr = getattr(key, field)
            if attr is None:
                yield None

            # NOTE: We can do `is` checks on field since `SLICE_ATTRS` only contains some certain symbolic constants

            elif attr.__class__ is np.datetime64:
                if field is SLICE_STEP_ATTR:
                    raise RuntimeError(f'Step cannot be {attr}')
                # if we match the same dt64 unit, simply use label_to_pos, increment stop
                if attr.dtype == labels.dtype: # type: ignore
                    pos: TypePos = label_to_pos(attr)
                    if pos is None:
                        # if same type, and that atter is not in labels, we fail, just as we do in then non-datetime64 case. Only when datetimes are given in a different unit are we "loose" about matching.
                        raise LocInvalid('Invalid loc given in a slice', attr, field)
                    if field is SLICE_STOP_ATTR:
                        pos += 1 # stop is inclusive
                elif field is SLICE_START_ATTR:
                    # NOTE: as an optimization only for the start attr, we can try to convert attr to labels unit and see if there is a match; this avoids astyping the entire labels array
                    pos: TypePos = label_to_pos(attr.astype(labels.dtype)) #type: ignore
                    if pos is None: # we did not find a start position
                        labels_astype = labels.astype(attr.dtype) #type: ignore
                        matches = nonzero_1d(labels_astype == attr)
                        if len(matches):
                            pos = matches[0]
                        else:
                            raise LocEmptyInstance
                elif field is SLICE_STOP_ATTR:
                    # NOTE: we do not want to convert attr to labels dtype and take the match as we want to get the last of all possible matches of labels at the attr unit
                    # NOTE: try to re-use labels_astype if possible
                    if labels_astype is None or labels_astype.dtype != attr.dtype:
                        labels_astype = labels.astype(attr.dtype) #type: ignore
                    matches = nonzero_1d(labels_astype == attr)
                    if len(matches):
                        pos = matches[-1] + 1
                    else:
                        raise LocEmptyInstance

                yield pos

            else:
                if field is not SLICE_STEP_ATTR:
                    pos = label_to_pos(attr)
                    if pos is None:
                        # NOTE: could raise LocEmpty() to silently handle this
                        raise LocInvalid('Invalid loc given in a slice', attr, field)
                else: # step
                    pos = attr # should be an integer
                    if not isinstance(pos, INT_TYPES):
                        raise TypeError(f'Step must be an integer, not {pos}')
                if field is SLICE_STOP_ATTR:
                    # loc selections are inclusive, so iloc gets one more
                    pos += 1
                yield pos

    @classmethod
    def loc_to_iloc(cls, *,
            label_to_pos: FrozenAutoMap,
            labels: TNDArrayAny,
            positions: TNDArrayAny,
            key: TLocSelector,
            partial_selection: bool = False,
            ) -> TILocSelector:
        '''
        Note: all SF objects (Series, Index) need to be converted to basic types before being passed as `key` to this function.

        Args:
            partial_selection: if True and key is an iterable of labels that includes labels not in the mapping, available matches will be returned rather than raising.
        Returns:
            An integer mapped slice, or GetItemKey type that is based on integers, compatible with TypeBlocks
        '''
        # NOTE: ILoc is handled prior to this call, in the Index._loc_to_iloc method

        if key.__class__ is slice:
            if key == NULL_SLICE:
                return NULL_SLICE
            try:
                return slice(*cls.map_slice_args(
                        label_to_pos.get,
                        key, # type: ignore
                        labels)
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

        if is_array := key.__class__ is np.ndarray:
            is_list = False
        else:
            is_list = isinstance(key, list)

        # can be an iterable of labels (keys) or an iterable of Booleans
        if is_array or is_list:
            if len(key) == 0: # type: ignore
                return EMPTY_ARRAY_INT

            if is_array and key.dtype.kind == DTYPE_DATETIME_KIND: #type: ignore
                dt64_unit = np.datetime_data(key.dtype)[0] #type: ignore
                # NOTE: only in the conditions of an empty array, the unit might be generic
                if (labels.dtype == DTYPE_OBJECT and dt64_unit in DTYPE_OBJECTABLE_DT64_UNITS):
                    # if key is dt64 and labels are object, then for objectable units we can convert key to object to permit matching in the AutoMap
                    # NOTE: tolist() is expected to be faster than astype object for smaller collections
                    key = key.tolist() #type: ignore
                    is_array = False
                    is_list = True
                elif labels_is_dt64 and key.dtype < labels.dtype: #type: ignore
                    # NOTE: change the labels to the dt64 dtype, i.e., if the key is years, recast the labels as years, and do a Boolean selection of everything that matches each key
                    labels_ref = labels.astype(key.dtype) # type: ignore
                    # NOTE: this is only correct if both key and labels are dt64, and key is a less granular unit, as the order in the key and will not be used
                    # let Boolean key advance to next branch
                    key = reduce(OPERATORS['__or__'], (labels_ref == k for k in key)) # type: ignore

            if is_array and key.dtype == DTYPE_BOOL: #type: ignore
                return positions[key] # type: ignore

            # map labels to integer positions, return a list of integer positions
            # NOTE: we may miss the opportunity to identify contiguous keys and extract a slice
            if partial_selection:
                return label_to_pos.get_any(key) # type: ignore
            return label_to_pos.get_all(key) # type: ignore

        # if a single element (an integer, string, or date, we just get the integer out of the map
        return label_to_pos[key] # type: ignore


class HierarchicalLocMap:
    '''
    A utility utilized by IndexHierarchy in order to quickly map keys to ilocs.
    '''

    __slots__ = (
            'bit_offset_encoders',
            'encoding_can_overflow',
            'encoded_indexer_map',
            )

    bit_offset_encoders: TNDArrayAny
    encoding_can_overflow: bool
    encoded_indexer_map: FrozenAutoMap

    def __init__(self: _HLMap,
            *,
            indices: tp.List[Index[tp.Any]],
            indexers: TNDArrayAny,
            ) -> None:

        if not len(indexers[0]):
            self.bit_offset_encoders = np.full(len(indices), 0, dtype=DTYPE_UINT_DEFAULT)
            self.encoding_can_overflow = False
            self.encoded_indexer_map = EMPTY_FROZEN_AUTOMAP
            return

        self.bit_offset_encoders, self.encoding_can_overflow = self.build_offsets_and_overflow(
                num_unique_elements_per_depth=list(map(len, indices))
                )
        try:
            self.encoded_indexer_map = self.build_encoded_indexers_map(
                    encoding_can_overflow=self.encoding_can_overflow,
                    bit_offset_encoders=self.bit_offset_encoders,
                    indexers=indexers,
                    )
        except FirstDuplicatePosition as e:
            duplicate_labels = tuple(
                    index[indexer[e.first_dup]]
                    for (index, indexer) in zip(indices, indexers)
                    )
            raise ErrorInitIndexNonUnique(duplicate_labels) from None

    def __deepcopy__(self: _HLMap,
            memo: tp.Dict[int, tp.Any],
            ) -> _HLMap:
        '''
        Return a deep copy of this IndexHierarchy.
        '''
        obj: _HLMap = self.__class__.__new__(self.__class__)
        obj.bit_offset_encoders = array_deepcopy(self.bit_offset_encoders, memo)
        obj.encoding_can_overflow = self.encoding_can_overflow
        obj.encoded_indexer_map = deepcopy(self.encoded_indexer_map, memo)

        memo[id(self)] = obj
        return obj

    def __setstate__(self, state: tp.Tuple[None, tp.Dict[str, tp.Any]]) -> None:
        '''
        Ensure that reanimated NP arrays are set not writeable.
        '''
        for key, value in state[1].items():
            setattr(self, key, value)
        self.bit_offset_encoders.flags.writeable = False

    @property
    def nbytes(self: _HLMap) -> int:
        return (
                sys.getsizeof(self.encoding_can_overflow) +
                self.bit_offset_encoders.nbytes +
                sys.getsizeof(self.encoded_indexer_map)
        )

    @staticmethod
    def build_offsets_and_overflow(
            num_unique_elements_per_depth: tp.List[int],
            ) -> tp.Tuple[TNDArrayAny, bool]:
        '''
        Derive the offsets and the overflow flag from the number of unique values per depth
        '''
        # `bit_sizes` is an array that shows how many bits are needed to contain the max indexer per depth
        #
        # For example, lets say there are 3 levels, and number of unique elements per depth is [71, 5, 13].
        # `bit_sizes` will become: [7, 3, 4], which mean that we can fit the indexer for each depth into 7, 3, and 4 bits, respectively.
        #    (i.e.) all valid indexers at depth N can be represented with bit_sizes[N] bits!
        #
        # We see this: [2**power for power in [7,3,4]] = [128, 8, 16]
        # We can prove this: num_unique_elements_per_depth <= [2**bit_size for bit_size in bit_sizes]
        bit_sizes = np.floor(np.log2(num_unique_elements_per_depth)) + 1

        # Based on bit_sizes, we cumsum to determine the successive number of total bits needed for each depth
        # Using the previous example, this would look like: [7, 10, 14]
        # This means:
        #  - depth 0 ends at bit offset 7.
        #  - depth 1 ends at bit offset 10. (depth 1 needs 3 bits!)
        #  - depth 2 ends at bit offset 14. (depth 2 needs 4 bits!)
        bit_end_positions = np.cumsum(bit_sizes, dtype=DTYPE_UINT_DEFAULT)

        # However, since we ultimately need these values to bitshift, we want them to offset based on start position, not end.
        # This means:
        #  - depth 0 starts at bit offset 0.
        #  - depth 1 starts at bit offset 7. (depth 0 needed 7 bits!)
        #  - depth 2 starts at bit offset 10. (depth 1 needed 3 bits!)
        bit_start_positions = np.zeros(
                len(bit_end_positions),
                dtype=DTYPE_UINT_DEFAULT)
        bit_start_positions[1:] = bit_end_positions[:-1]
        bit_start_positions.flags.writeable = False

        # We now return these offsets, and whether or not we have overflow.
        # If the last end bit is greater than 64, then it means we cannot encode a label's indexer into a uint64.
        return bit_start_positions, bit_end_positions[-1] > 64

    @staticmethod
    def build_encoded_indexers_map(
            *,
            encoding_can_overflow: bool,
            bit_offset_encoders: TNDArrayAny,
            indexers: TNDArrayAny,
            ) -> FrozenAutoMap:
        '''
        Builds up a mapping from indexers to iloc positions using their encoded values
        '''
        # We previously determined we cannot encode indexers into uint64. Cast to object to rely on Python's bigint
        if encoding_can_overflow:
            indexers = indexers.astype(DTYPE_OBJECT).T
        else:
            indexers = indexers.astype(DTYPE_UINT_DEFAULT).T

        # Encode indexers into uint64
        # indexers: (n, m), offsets: (m,)
        # This bitshifts all indexers by the offset, resulting in numbers that are ready to be bitwise OR'd
        # We need to reverse in order to have depth 0
        # Example:
        #  indexers:      bitshift         (Bit representation)
        #                                    d0, d1 d0, d2 d1 d0  (d = depth)
        #    [0, 1, 2] => [0, 4, 32]       ([00, 01 00, 10 00 00])
        #    [0, 2, 0] => [0, 8,  0]       ([00, 10 00, 00 00 00])
        #    [2, 2, 0] => [2, 8,  0]       ([10, 10 00, 00 00 00])
        #    [1, 0, 1] => [1, 0, 16]       ([01, 00 00, 01 00 00])
        encoded_indexers = indexers << bit_offset_encoders
        # Finally, we bitwise OR all them together to encode them into a single, unique uint64 for each iloc
        #  encoded_indexers   bitwise OR   (Bit representation)
        #                                    d0 d1 d2  (d = depth)
        #    [2, 4,  0]    => [36]         ([10 01 00])
        #    [0, 8,  0]    => [ 8]         ([00 10 00])
        #    [0, 8, 32]    => [10]         ([00 10 10])
        #    [1, 0, 16]    => [17]         ([01 00 01])
        encoded_indexers = np.bitwise_or.reduce(encoded_indexers, axis=1)

        # Success! We have now successfully encoded each indexer into a single, unique uint64.
        #    [0, 1, 2] => [36]
        #    [0, 2, 0] => [ 8]
        #    [2, 2, 0] => [10]
        #    [1, 0, 1] => [17]

        # Finally, we create a mapping from encoded indexers to ilocs.
        #    [0, 1, 2] => [36] => [0]
        #    [0, 2, 0] => [ 8] => [1]
        #    [2, 2, 0] => [10] => [2]
        #    [1, 0, 1] => [17] => [3]
        # len(encoded_indexers) == len(self)!
        encoded_indexers.flags.writeable = False
        try:
            return FrozenAutoMap(encoded_indexers)
        except NonUniqueError as e:
            first_duplicate = first_true_1d(encoded_indexers == e.args[0], forward=True)
            raise FirstDuplicatePosition(first_duplicate) from None

    @staticmethod
    def is_single_element(element: tp.Any) -> bool:
        # By definition, all index labels are hashable. If it's not, then it
        # means this must be a container of labels.
        try:
            hash(element)
        except TypeError:
            return False
        return True

    def build_key_indexers(self: _HLMap,
            key: HierarchicalLocMapKey,
            indices: tp.List[Index[tp.Any]],
            ) -> TNDArrayAny:
        key_indexers: tp.List[tp.Sequence[int]] = []

        is_single_key = True

        subkey_indexers: tp.List[int]

        # 1. Perform label resolution
        for key_at_depth, index_at_depth in zip(key, indices):
            if self.is_single_element(key_at_depth):
                key_indexers.append((index_at_depth._loc_to_iloc(key_at_depth),)) # type: ignore
            else:
                is_single_key = False
                subkey_indexers = []
                for sub_key in key_at_depth:
                    subkey_indexers.append(index_at_depth._loc_to_iloc(sub_key)) # type: ignore
                key_indexers.append(subkey_indexers)

        # 2. Convert to numpy array
        combinations = np.array(list(itertools.product(*key_indexers)), dtype=DTYPE_UINT_DEFAULT)
        if is_single_key and len(combinations) == 1:
            [combinations] = combinations

        if self.encoding_can_overflow:
            return combinations.astype(object)

        return combinations

    def loc_to_iloc(self: _HLMap,
            key: HierarchicalLocMapKey,
            indices: tp.List[Index[tp.Any]],
            ) -> tp.Union[int, tp.List[int]]:
        key_indexers = self.build_key_indexers(key=key, indices=indices)

        # 2. Encode the indexers. See `build_encoded_indexers_map` for detailed comments.
        key_indexers <<= self.bit_offset_encoders

        if key_indexers.ndim == 2:
            key_indexers = np.bitwise_or.reduce(key_indexers, axis=1)
            return list(map(self.encoded_indexer_map.__getitem__, key_indexers))

        key_indexers = np.bitwise_or.reduce(key_indexers)
        return self.encoded_indexer_map[key_indexers] # type: ignore

    def indexers_to_iloc(self: _HLMap,
            indexers: TNDArrayAny,
            ) -> tp.List[int]:
        '''
        Encodes indexers, and then remaps them to ilocs using the encoded_indexer_map
        '''
        indexers = self.encode(indexers, self.bit_offset_encoders)
        return list(map(self.encoded_indexer_map.__getitem__, indexers))

    @staticmethod
    def encode(indexers: TNDArrayAny, bit_offset_encoders: TNDArrayAny) -> TNDArrayAny:
        '''
        Encode indexers into a 1-dim array of uint64
        '''
        # Validate input requirements
        assert indexers.ndim == 2
        assert indexers.shape[1] == len(bit_offset_encoders)
        assert indexers.dtype == DTYPE_UINT_DEFAULT

        array: TNDArrayAny = np.bitwise_or.reduce(indexers << bit_offset_encoders, axis=1)
        return array

    @staticmethod
    def unpack_encoding(
            encoded_arr: TNDArrayAny,
            bit_offset_encoders: TNDArrayAny,
            encoding_can_overflow: bool,
            ) -> TNDArrayAny:
        '''
        Given an encoding, unpack it into its constituent parts

        Ex:
            bit_offset_encoders = [0, 2, 4]

            Encodings:
                36  => [0, 4, 32] => [0, 1, 2]
                 8  => [0, 8,  0] => [0, 2, 0]
                10  => [2, 8,  0] => [2, 2, 0]
                17  => [1, 0, 16] => [1, 0, 1]

            Step 1:
            Expand bit_offset_encoders into something more helpful -> masks

            [0, 2, 4] is where the bit offsets start. They end one bit before the next offset.
            Thus, the bit offset ends are:
            [1, 3, 64] # since 64 is the max bit offset

            From here, we build up a list of masks that each have the correct number of up bits

            0 => [11] # 2 bit mask
            2 => [11] # 2 bit mask
            4 => [11] # 2 bit mask

            Now, for each component (i.e. the number of bit_offset_encoders), we
            apply to corresponding mask to the values AFTER they have been shifted backwards

            36 == [10 01 00]
             8 == [00 10 00]
            10 == [00 10 10]
            17 == [01 00 01]

            Depth 0:
                offset = bit_offset_encoders[0] = 0
                36 => ([10 01 00] << 0) => [10 01 00] & [11] => [00 00 00] => 0
                 8 => ([00 10 00] << 0) => [00 10 00] & [11] => [00 00 00] => 0
                10 => ([00 10 10] << 0) => [00 10 10] & [11] => [00 00 10] => 2
                17 => ([01 00 01] << 0) => [01 00 01] & [11] => [00 00 01] => 1

            Depth 1:
                offset = bit_offset_encoders[1] = 2
                36 => ([10 01 00] << 2) => [10 01] & [11] => [00 01] => 1
                 8 => ([00 10 00] << 2) => [00 10] & [11] => [00 10] => 2
                10 => ([00 10 10] << 2) => [00 10] & [11] => [00 10] => 2
                17 => ([01 00 01] << 2) => [01 00] & [11] => [00 00] => 0

            Depth 2:
                offset = bit_offset_encoders[2] = 4
                36 => ([10 01 00] << 4) => [10] & [11] => [10] => 2
                 8 => ([00 10 00] << 4) => [00] & [11] => [00] => 0
                10 => ([00 10 10] << 4) => [00] & [11] => [00] => 0
                17 => ([01 00 01] << 4) => [01] & [11] => [01] => 1

            Result:
                36 => [0, 1, 2]
                 8 => [0, 2, 0]
                10 => [2, 2, 0]
                17 => [1, 0, 1]

            NOTE: This is the inverse of the documentation in `build_encoded_indexers_map`
        '''
        assert bit_offset_encoders.dtype == DTYPE_UINT_DEFAULT
        assert bit_offset_encoders[0] == 0 # By definition, the first offset starts at 0!
        assert encoded_arr.ndim == 1 # Encodings are always 1D

        dtype = DTYPE_OBJECT if encoding_can_overflow else DTYPE_UINT_DEFAULT

        starts = bit_offset_encoders
        stops: TNDArrayAny = np.empty(len(starts), dtype=dtype)
        stops[:-1] = starts[1:]
        stops[-1] = 64

        lens = stops - starts
        masks = [x for x in (1 << lens) - 1]

        target = np.empty((len(bit_offset_encoders), len(encoded_arr)), dtype=DTYPE_UINT_DEFAULT)

        for depth in range(len(bit_offset_encoders)):
            target[depth] = (encoded_arr >> starts[depth]) & masks[depth]

        target.flags.writeable = False
        return target
