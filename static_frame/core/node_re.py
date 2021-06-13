import typing as tp
import re
#from functools import partial

import numpy as np

from static_frame.core.node_selector import Interface
from static_frame.core.node_selector import TContainer
from static_frame.core.util import array_from_element_apply
from static_frame.core.util import DTYPE_STR
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import AnyCallable
from static_frame.core.util import DTYPE_STR_KINDS
from static_frame.core.util import DTYPE_OBJECT

if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index import Index  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchy  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.series import Series  #pylint: disable = W0611 #pragma: no cover


BlocksType = tp.Iterable[np.ndarray]
ToContainerType = tp.Callable[[tp.Iterator[np.ndarray]], TContainer]

class InterfaceRe(Interface[TContainer]):

    __slots__ = (
            '_blocks',
            '_blocks_to_container',
            '_pattern',
            )
    INTERFACE = (
            'search',
            'match',
            'fullmatch',
            'split',
            'findall',
            'sub',
            'subn',
            'groups', # property
            )


    def __init__(self,
            blocks: BlocksType,
            blocks_to_container: ToContainerType[TContainer],
            pattern: str,
            flags: int = 0,
            ) -> None:
        self._blocks: BlocksType = blocks
        self._blocks_to_container: ToContainerType[TContainer] = blocks_to_container
        self._pattern = re.compile(pattern, flags)

    @staticmethod
    def _process_blocks(*,
            blocks: BlocksType,
            func: AnyCallable,
            dtype: np.dtype,
            ) -> tp.Iterator[np.ndarray]:
        '''
        Element-wise processing of a methods on objects in a block
        '''
        for block in blocks:
            if block.dtype not in DTYPE_STR_KINDS:
                block = block.astype(DTYPE_STR)

            # resultant array is immutable
            array = array_from_element_apply(
                    array=block,
                    func=func,
                    dtype=dtype,
                    )
            yield array

    #---------------------------------------------------------------------------
    def search(self, pos: int = 0, endpos: tp.Optional[int] = None) -> TContainer:
        '''
        Scan through string looking for the first location where this regular expression produces a match, and return a corresponding match object. Return None if no position in the string matches the pattern; note that this is different from finding a zero-length match at some point in the string.

        The optional second parameter pos gives an index in the string where the search is to start; it defaults to 0.

        The optional parameter endpos limits how far the string will be searched; it will be as if the string is endpos characters long, so only the characters from pos to endpos - 1 will be searched for a match.
        '''
        args: tp.Tuple[int, ...]
        if endpos is not None:
            args = (pos, endpos)
        else:
            args = (pos,)

        func = lambda s: self._pattern.search(s, *args) is not None

        block_gen = self._process_blocks(
                blocks=self._blocks,
                func=func,
                dtype=DTYPE_BOOL,
                )
        return self._blocks_to_container(block_gen)

    def match(self, pos: int = 0, endpos: tp.Optional[int] = None) -> TContainer:
        '''
        If zero or more characters at the beginning of string match this regular expression, return a corresponding match object. Return None if the string does not match the pattern; note that this is different from a zero-length match.
        '''
        args: tp.Tuple[int, ...]
        if endpos is not None:
            args = (pos, endpos)
        else:
            args = (pos,)

        func = lambda s: self._pattern.match(s, *args) is not None

        block_gen = self._process_blocks(
                blocks=self._blocks,
                func=func,
                dtype=DTYPE_BOOL,
                )
        return self._blocks_to_container(block_gen)

    def fullmatch(self, pos: int = 0, endpos: tp.Optional[int] = None) -> TContainer:
        '''
        If the whole string matches this regular expression, return a corresponding match object. Return None if the string does not match the pattern; note that this is different from a zero-length match.
        '''
        args: tp.Tuple[int, ...]
        if endpos is not None:
            args = (pos, endpos)
        else:
            args = (pos,)

        func = lambda s: self._pattern.fullmatch(s, *args) is not None

        block_gen = self._process_blocks(
                blocks=self._blocks,
                func=func,
                dtype=DTYPE_BOOL,
                )
        return self._blocks_to_container(block_gen)

    def split(self, maxsplit: int = 0) -> TContainer:
        '''
        Split string by the occurrences of pattern. If capturing parentheses are used in pattern, then the text of all groups in the pattern are also returned as part of the resulting list. If maxsplit is nonzero, at most maxsplit splits occur, and the remainder of the string is returned as the final element of the list.
        '''
        func = lambda s: tuple(self._pattern.split(s, maxsplit=maxsplit))

        block_gen = self._process_blocks(
                blocks=self._blocks,
                func=func,
                dtype=DTYPE_OBJECT,
                )
        return self._blocks_to_container(block_gen)

    def findall(self, pos: int = 0, endpos: tp.Optional[int] = None) -> TContainer:
        '''
        Return all non-overlapping matches of pattern in string, as a list of strings. The string is scanned left-to-right, and matches are returned in the order found. If one or more groups are present in the pattern, return a list of groups; this will be a list of tuples if the pattern has more than one group. Empty matches are included in the result.
        '''
        args: tp.Tuple[int, ...]
        if endpos is not None:
            args = (pos, endpos)
        else:
            args = (pos,)

        func = lambda s: tuple(self._pattern.findall(s, *args))

        block_gen = self._process_blocks(
                blocks=self._blocks,
                func=func,
                dtype=DTYPE_OBJECT,
                )
        return self._blocks_to_container(block_gen)


    def sub(self, repl: str, count: int = 0) -> TContainer:
        '''
        Return the string obtained by replacing the leftmost non-overlapping occurrences of pattern in string by the replacement repl. If the pattern isn’t found, string is returned unchanged. repl can be a string or a function; if it is a string, any backslash escapes in it are processed.
        '''
        func = lambda s: self._pattern.sub(repl, s, count=count)

        block_gen = self._process_blocks(
                blocks=self._blocks,
                func=func,
                dtype=DTYPE_STR,
                )
        return self._blocks_to_container(block_gen)
