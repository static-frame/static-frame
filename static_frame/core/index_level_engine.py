import itertools
import sys
import typing as tp
from copy import deepcopy

import numpy as np
from automap import FrozenAutoMap  # pylint: disable = E0611

from static_frame.core.exception import ErrorInitIndexNonUnique
from static_frame.core.index import Index
from static_frame.core.util import DTYPE_UINT_DEFAULT
from static_frame.core.util import array_deepcopy

KeyForEngine = tp.Union[np.ndarray, tp.Tuple[tp.Union[tp.Sequence[tp.Hashable], tp.Hashable], ...]]

_Engine = tp.TypeVar('_Engine', bound='IndexLevelEngine')


class IndexLevelEngine:
    '''
    A utility engine utilized by IndexHierarchy in order to quickly map keys to ilocs.
    '''

    __slots__ = (
            'bit_offset_encoders',
            'encoding_can_overflow',
            'encoded_indexer_map',
            )

    bit_offset_encoders: np.ndarray
    encoding_can_overflow: bool
    encoded_indexer_map: FrozenAutoMap

    def __init__(self: _Engine,
            *,
            indices: tp.List[Index],
            indexers: tp.List[np.ndarray],
            ) -> None:

        if not len(indexers[0]):
            self.bit_offset_encoders = np.full(len(indices), 0, dtype=DTYPE_UINT_DEFAULT)
            self.encoding_can_overflow = False
            self.encoded_indexer_map = FrozenAutoMap()
            return

        self.bit_offset_encoders, self.encoding_can_overflow = self.build_offsets_and_overflow(
                num_unique_elements_per_depth=list(map(len, indices))
                )
        self.encoded_indexer_map = self.build_encoded_indexers_map(indexers)

    def __deepcopy__(self: _Engine,
            memo: tp.Dict[int, tp.Any],
            ) -> _Engine:
        '''
        Return a deep copy of this IndexHierarchy.
        '''
        obj: _Engine = self.__new__(self.__class__)
        obj.bit_offset_encoders = array_deepcopy(self.bit_offset_encoders, memo)
        obj.encoding_can_overflow = self.encoding_can_overflow
        obj.encoded_indexer_map = deepcopy(self.encoded_indexer_map, memo)

        memo[id(self)] = obj
        return obj

    @property
    def nbytes(self: _Engine) -> int:
        return (
                sys.getsizeof(self.encoding_can_overflow) +
                tp.cast(int, self.bit_offset_encoders.nbytes) +
                sys.getsizeof(self.encoded_indexer_map)
        )

    @staticmethod
    def build_offsets_and_overflow(
            num_unique_elements_per_depth: tp.List[int],
            ) -> tp.Tuple[np.ndarray, bool]:
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
        bit_end_positions = np.cumsum(bit_sizes)

        # However, since we ultimately need these values to bitshift, we want them to offset based on start position, not end.
        # This means:
        #  - depth 0 starts at bit offset 0.
        #  - depth 1 starts at bit offset 7. (depth 0 needed 7 bits!)
        #  - depth 2 starts at bit offset 10. (depth 1 needed 3 bits!)
        bit_start_positions = np.concatenate(([0], bit_end_positions))[:-1].astype(DTYPE_UINT_DEFAULT)
        bit_start_positions.flags.writeable = False

        # We now return these offsets, and whether or not we have overflow.
        # If the last end bit is greater than 64, then it means we cannot encode a label's indexer into a uint64.
        return bit_start_positions, bit_end_positions[-1] > 64

    def build_encoded_indexers_map(self: _Engine,
            indexers: np.ndarray,
            ) -> FrozenAutoMap:
        '''
        Builds up a mapping from indexers to iloc positions using their encoded values
        '''
        # We previously determined we cannot encode indexers into uint64. Cast to object to rely on Python's bigint
        if self.encoding_can_overflow:
            indexers = indexers.astype(object).T
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
        encoded_indexers = indexers << self.bit_offset_encoders

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
        try:
            return FrozenAutoMap(encoded_indexers.tolist()) # Automap is faster with Python lists :(
        except ValueError as e:
            raise ErrorInitIndexNonUnique(*e.args) from None

    @staticmethod
    def is_single_element(element: tp.Hashable) -> bool:
        return not hasattr(element, '__len__') or isinstance(element, str)

    def build_key_indexers(self: _Engine,
            key: KeyForEngine,
            indices: tp.List[Index],
            ) -> np.ndarray:
        key_indexers: tp.List[tp.Sequence[int]] = []

        is_single_key = True

        # 1. Perform label resolution
        for key_at_depth, index_at_depth in zip(key, indices):
            if self.is_single_element(key_at_depth):
                key_indexers.append((index_at_depth._loc_to_iloc(key_at_depth),))
            else:
                is_single_key = False
                subkey_indexers: tp.List[int] = []
                for sub_key in key_at_depth:
                    subkey_indexers.append(index_at_depth._loc_to_iloc(sub_key))
                key_indexers.append(subkey_indexers)

        # 2. Convert to numpy array
        combinations = np.array(list(itertools.product(*key_indexers)), dtype=DTYPE_UINT_DEFAULT)
        if is_single_key and len(combinations) == 1:
            [combinations] = combinations

        if self.encoding_can_overflow:
            return combinations.astype(object)

        return combinations

    def loc_to_iloc(self: _Engine,
            key: KeyForEngine,
            indices: tp.List[Index],
            ) -> int:
        key_indexers = self.build_key_indexers(key=key, indices=indices)

        # 2. Encode the indexers. See `build_encoded_indexers_map` for detailed comments.
        key_indexers <<= self.bit_offset_encoders

        if key_indexers.ndim == 2:
            key_indexers = np.bitwise_or.reduce(key_indexers, axis=1)
            return list(map(self.encoded_indexer_map.__getitem__, key_indexers)) # type: ignore

        key_indexers = np.bitwise_or.reduce(key_indexers)
        return self.encoded_indexer_map[key_indexers] # type: ignore
