
import typing as tp
import numpy as np
from numpy import char as npc

from static_frame.core.util import EMPTY_TUPLE
from static_frame.core.util import DTYPE_STR_KIND
from static_frame.core.util import DTYPE_STR
from static_frame.core.util import UFunc

from static_frame.core.util import array_from_element_method


if tp.TYPE_CHECKING:

    from static_frame.core.frame import Frame  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index import Index  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchy  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.series import Series  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.type_blocks import TypeBlocks  #pylint: disable = W0611 #pragma: no cover

# only ContainerOperand subclasses
TContainer = tp.TypeVar('TContainer', 'Index', 'IndexHierarchy', 'Series', 'Frame', 'TypeBlocks')

BlocksType = tp.Iterable[np.ndarray]
ToContainerType = tp.Callable[[tp.Iterator[np.ndarray]], TContainer]


class InterfaceString(tp.Generic[TContainer]):

    # NOTE: based on https://numpy.org/doc/stable/reference/routines.char.html

    __slots__ = (
        '_blocks', # function that returns array of strings
        '_blocks_to_container', # partialed function that will return a new container
        )

    def __init__(self,
            blocks: BlocksType,
            blocks_to_container: ToContainerType[TContainer]
            ) -> None:
        self._blocks: BlocksType = blocks
        self._blocks_to_container: ToContainerType[TContainer] = blocks_to_container


    @staticmethod
    def _process_blocks(
            blocks: BlocksType,
            func: UFunc,
            args: tp.Tuple[tp.Any, ...] = EMPTY_TUPLE,
            astype_str: bool = True,
            ) -> tp.Iterator[np.ndarray]:

        for block in blocks:
            if astype_str and block.dtype not in DTYPE_STR_KIND:
                block = block.astype(DTYPE_STR)
            array = func(block, *args)
            array.flags.writeable = False
            yield array

    @staticmethod
    def _process_tuple_blocks(*,
            blocks: BlocksType,
            method_name: str,
            dtype: np.dtype,
            args: tp.Tuple[tp.Any, ...] = EMPTY_TUPLE,
            ) -> tp.Iterator[np.ndarray]:

        for block in blocks:
            if block.dtype not in DTYPE_STR_KIND:
                block = block.astype(DTYPE_STR)

            # resultant array is immutable
            array = array_from_element_method(
                    array=block,
                    method_name=method_name,
                    args=args,
                    dtype=dtype,
                    pre_insert=tuple,
                    )
            yield array

    #---------------------------------------------------------------------------
    def capitalize(self) -> TContainer:
        '''
        Return a container with only the first character of each element capitalized.
        '''
        # return self._blocks_to_container(npc.capitalize(self._blocks()))
        block_gen = self._process_blocks(self._blocks, npc.capitalize)
        return self._blocks_to_container(block_gen)

    def center(self,
            width: int,
            fillchar: str = ' '
            ) -> TContainer:
        '''
        Return a container with its elements centered in a string of length ``width``.
        '''
        block_gen = self._process_blocks(self._blocks, npc.center, (width, fillchar))
        return self._blocks_to_container(block_gen)

    def decode(self,
            encoding: tp.Optional[str] = None,
            errors: tp.Optional[str] = None,
            ) -> TContainer:
        '''
        Apply str.decode() to each element. Elements must be bytes.
        '''
        block_gen = self._process_blocks(
                blocks=self._blocks,
                func=npc.decode,
                args=(encoding, errors),
                astype_str=False, # needs to be bytes
                )
        return self._blocks_to_container(block_gen)

    def encode(self,
            encoding: tp.Optional[str] = None,
            errors: tp.Optional[str] = None,
            ) -> TContainer:
        '''
        Apply str.encode() to each element. Elements must be strings.
        '''
        block_gen = self._process_blocks(self._blocks, npc.encode, (encoding, errors))
        return self._blocks_to_container(block_gen)

    def ljust(self,
            width: int,
            fillchar: str = ' '
            ) -> TContainer:
        '''
        Return a container with its elements ljusted in a string of length ``width``.
        '''
        block_gen = self._process_blocks(self._blocks, npc.ljust, (width, fillchar))
        return self._blocks_to_container(block_gen)

    def replace(self,
            old: str,
            new: str,
            count: tp.Optional[int] = None,
            ) -> TContainer:
        '''
        Return a container with its elements replaced in a string of length ``width``.
        '''
        block_gen = self._process_blocks(self._blocks, npc.replace, (old, new, count))
        return self._blocks_to_container(block_gen)

    def rjust(self,
            width: int,
            fillchar: str = ' '
            ) -> TContainer:
        '''
        Return a container with its elements rjusted in a string of length ``width``.
        '''
        block_gen = self._process_blocks(self._blocks, npc.rjust, (width, fillchar))
        return self._blocks_to_container(block_gen)

    def rsplit(self,
            sep: str,
            maxsplit: int = -1,
            ) -> TContainer:
        '''
        For each element, return a tuple of the words in the string, using sep as the delimiter string.
        '''
        # NOTE: npc.rsplit gives an array of lists, so implement our own routine to get an array of tuples.
        block_gen = self._process_tuple_blocks(
                blocks=self._blocks,
                method_name='rsplit',
                args=(sep, maxsplit),
                dtype=object
                )
        return self._blocks_to_container(block_gen)

    def rstrip(self,
            chars: tp.Optional[str] = None,
            ) -> TContainer:
        '''
        For each element, return a copy with the trailing characters removed.
        '''
        block_gen = self._process_blocks(self._blocks, npc.rstrip, (chars,))
        return self._blocks_to_container(block_gen)

    def split(self,
            sep: str,
            maxsplit: int = -1,
            ) -> TContainer:
        '''
        For each element, return a tuple of the words in the string, using sep as the delimiter string.
        '''
        # NOTE: npc.split gives an array of lists, so implement our own routine to get an array of tuples.
        block_gen = self._process_tuple_blocks(
                blocks=self._blocks,
                method_name='split',
                args=(sep, maxsplit),
                dtype=object
                )
        return self._blocks_to_container(block_gen)

    def strip(self,
            chars: tp.Optional[str] = None,
            ) -> TContainer:
        '''
        For each element, return a copy with the leading and trailing characters removed.
        '''
        block_gen = self._process_blocks(self._blocks, npc.strip, (chars,))
        return self._blocks_to_container(block_gen)

    def swapcase(self) -> TContainer:
        '''
        Return a container with uppercase characters converted to lowercase and vice versa.
        '''
        block_gen = self._process_blocks(self._blocks, npc.swapcase)
        return self._blocks_to_container(block_gen)

    def title(self) -> TContainer:
        '''
        Return a container with uppercase characters converted to lowercase and vice versa.
        '''
        block_gen = self._process_blocks(self._blocks, npc.title)
        return self._blocks_to_container(block_gen)

    # translate: akward input

    def upper(self) -> TContainer:
        '''
        Return a container with uppercase characters converted to lowercase and vice versa.
        '''
        block_gen = self._process_blocks(self._blocks, npc.upper)
        return self._blocks_to_container(block_gen)

    def zfill(self,
            width: int,
            ) -> TContainer:
        '''
        Return the string left-filled with zeros.
        '''
        block_gen = self._process_blocks(self._blocks, npc.zfill, (width,))
        return self._blocks_to_container(block_gen)

    #---------------------------------------------------------------------------

    def count(self,
            sub: str,
            start: tp.Optional[int] = None,
            end: tp.Optional[int] = None
            ) -> TContainer:
        '''
        Returns a container with the number of non-overlapping occurrences of substring sub in the optional range ``start``, ``end``.
        '''
        block_gen = self._process_blocks(self._blocks, npc.count, (sub, start, end))
        return self._blocks_to_container(block_gen)

    def endswith(self,
            suffix: str,
            start: tp.Optional[int] = None,
            end: tp.Optional[int] = None
            ) -> TContainer:
        '''
        Returns a container with the number of non-overlapping occurrences of substring sub in the optional range ``start``, ``end``.
        '''
        block_gen = self._process_blocks(self._blocks, npc.endswith, (suffix, start, end))
        return self._blocks_to_container(block_gen)

    # find
    # index
    # isalpha
    # isalnum
    # isdecimal
    # isdigit
    # islower
    # isnumeric
    # isspace
    # istitle
    # isupper
    # rfind
    # rindex

    def startswith(self,
            prefix: str,
            start: tp.Optional[int] = None,
            end: tp.Optional[int] = None
            ) -> TContainer:
        '''
        Returns a container with the number of non-overlapping occurrences of substring sub in the optional range ``start``, ``end``.
        '''
        block_gen = self._process_blocks(self._blocks, npc.startswith, (prefix, start, end))
        return self._blocks_to_container(block_gen)

