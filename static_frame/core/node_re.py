from __future__ import annotations

import re

import typing_extensions as tp

from static_frame.core.node_selector import Interface, InterfaceBatch, TVContainer_co
from static_frame.core.util import (
    DTYPE_BOOL,
    DTYPE_OBJECT,
    DTYPE_STR,
    DTYPE_STR_KINDS,
    TCallableAny,
    array_from_element_apply,
)

if tp.TYPE_CHECKING:
    import numpy as np  # pragma: no cover

    from static_frame.core.batch import Batch  # pragma: no cover
    from static_frame.core.frame import Frame  # #pragma: no cover
    from static_frame.core.index import Index  # #pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchy  # #pragma: no cover
    from static_frame.core.series import Series  # #pragma: no cover

    TNDArrayAny = np.ndarray[tp.Any, tp.Any]  # pragma: no cover
    TDtypeAny = np.dtype[tp.Any]  # pragma: no cover
    BlocksType = tp.Iterable[TNDArrayAny]  # pragma: no cover
    ToContainerType = tp.Callable[
        [tp.Iterator[TNDArrayAny]], TVContainer_co
    ]  # pragma: no cover


INTERFACE_RE = (
    'search',
    'match',
    'fullmatch',
    'split',
    'findall',
    'sub',
    'subn',
)


class InterfaceRe(Interface, tp.Generic[TVContainer_co]):
    __slots__ = (
        '_blocks',
        '_blocks_to_container',
        '_pattern',
    )
    _INTERFACE = INTERFACE_RE

    def __init__(
        self,
        blocks: BlocksType,
        blocks_to_container: ToContainerType[TVContainer_co],  # type: ignore[type-var]
        pattern: str,
        flags: int = 0,
    ) -> None:
        self._blocks: BlocksType = blocks
        self._blocks_to_container: ToContainerType[TVContainer_co] = blocks_to_container
        self._pattern = re.compile(pattern, flags)

    @staticmethod
    def _process_blocks(
        *,
        blocks: BlocksType,
        func: TCallableAny,
        dtype: TDtypeAny,
    ) -> tp.Iterator[TNDArrayAny]:
        """
        Element-wise processing of a methods on objects in a block
        """
        for block in blocks:
            if block.dtype.kind not in DTYPE_STR_KINDS:
                block = block.astype(DTYPE_STR)

            # resultant array is immutable
            array = array_from_element_apply(
                array=block,
                func=func,
                dtype=dtype,
            )
            yield array

    # ---------------------------------------------------------------------------
    def search(self, pos: int = 0, endpos: tp.Optional[int] = None) -> TVContainer_co:
        """
        Scan through string looking for the first location where this regular expression produces a match and return True, else False. Note that this is different from finding a zero-length match at some point in the string.

        Args:
            pos: Gives an index in the string where the search is to start; it defaults to 0.
            endpos: Limits how far the string will be searched; it will be as if the string is endpos characters long.
        """
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

    def match(self, pos: int = 0, endpos: tp.Optional[int] = None) -> TVContainer_co:
        """
        If zero or more characters at the beginning of string match this regular expression return True, else False. Note that this is different from a zero-length match.

        Args:
            pos: Gives an index in the string where the search is to start; it defaults to 0.
            endpos: Limits how far the string will be searched; it will be as if the string is endpos characters long.
        """
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

    def fullmatch(self, pos: int = 0, endpos: tp.Optional[int] = None) -> TVContainer_co:
        """
        If the whole string matches this regular expression, return True, else False. Note that this is different from a zero-length match.

        Args:
            pos: Gives an index in the string where the search is to start; it defaults to 0.
            endpos: Limits how far the string will be searched; it will be as if the string is endpos characters long.
        """
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

    def split(self, maxsplit: int = 0) -> TVContainer_co:
        """
        Split string by the occurrences of pattern. If capturing parentheses are used in pattern, then the text of all groups in the pattern are also returned as part of the resulting tuple.

        Args:
            maxsplit: If nonzero, at most maxsplit splits occur, and the remainder of the string is returned as the final element of the tuple.
        """
        func = lambda s: tuple(self._pattern.split(s, maxsplit=maxsplit))

        block_gen = self._process_blocks(
            blocks=self._blocks,
            func=func,
            dtype=DTYPE_OBJECT,
        )
        return self._blocks_to_container(block_gen)

    def findall(self, pos: int = 0, endpos: tp.Optional[int] = None) -> TVContainer_co:
        """
        Return all non-overlapping matches of pattern in string, as a tuple of strings. The string is scanned left-to-right, and matches are returned in the order found. If one or more groups are present in the pattern, return a tuple of groups; this will be a tuple of tuples if the pattern has more than one group. Empty matches are included in the result.

        Args:
            pos: Gives an index in the string where the search is to start; it defaults to 0.
            endpos: Limits how far the string will be searched; it will be as if the string is endpos characters long.
        """
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

    def sub(self, repl: str, count: int = 0) -> TVContainer_co:
        """
        Return the string obtained by replacing the leftmost non-overlapping occurrences of pattern in string by the replacement ``repl``. If the pattern is not found, the string is returned unchanged.

        Args:
            repl: A string or a function; if it is a string, any backslash escapes in it are processed.
            count: The optional argument count is the maximum number of pattern occurrences to be replaced; count must be a non-negative integer. If omitted or zero, all occurrences will be replaced.
        """
        func = lambda s: self._pattern.sub(repl, s, count=count)

        block_gen = self._process_blocks(
            blocks=self._blocks,
            func=func,
            dtype=DTYPE_STR,
        )
        return self._blocks_to_container(block_gen)

    def subn(self, repl: str, count: int = 0) -> TVContainer_co:
        """
        Perform the same operation as sub(), but return a tuple (new_string, number_of_subs_made).

        Args:
            repl: A string or a function; if it is a string, any backslash escapes in it are processed.
            count: The optional argument count is the maximum number of pattern occurrences to be replaced; count must be a non-negative integer. If omitted or zero, all occurrences will be replaced.
        """
        func = lambda s: self._pattern.subn(repl, s, count=count)

        block_gen = self._process_blocks(
            blocks=self._blocks,
            func=func,
            dtype=DTYPE_OBJECT,  # returns tuples
        )
        return self._blocks_to_container(block_gen)


# -------------------------------------------------------------------------------
class InterfaceBatchRe(InterfaceBatch):
    """Alternate re interface specialized for the :obj:`Batch`."""

    __slots__ = (
        '_batch_apply',
        '_pattern',
        '_flags',
    )
    _INTERFACE = INTERFACE_RE

    def __init__(
        self,
        batch_apply: tp.Callable[[TCallableAny], 'Batch'],
        pattern: str,
        flags: int = 0,
    ) -> None:
        self._batch_apply = batch_apply
        self._pattern = pattern
        self._flags = flags

    # ---------------------------------------------------------------------------
    def search(self, pos: int = 0, endpos: tp.Optional[int] = None) -> 'Batch':
        """
        Scan through string looking for the first location where this regular expression produces a match and return True, else False. Note that this is different from finding a zero-length match at some point in the string.

        Args:
            pos: Gives an index in the string where the search is to start; it defaults to 0.
            endpos: Limits how far the string will be searched; it will be as if the string is endpos characters long.
        """
        return self._batch_apply(
            lambda c: c.via_re(self._pattern, self._flags).search(pos, endpos)
        )

    def match(self, pos: int = 0, endpos: tp.Optional[int] = None) -> 'Batch':
        """
        If zero or more characters at the beginning of string match this regular expression return True, else False. Note that this is different from a zero-length match.

        Args:
            pos: Gives an index in the string where the search is to start; it defaults to 0.
            endpos: Limits how far the string will be searched; it will be as if the string is endpos characters long.
        """
        return self._batch_apply(
            lambda c: c.via_re(self._pattern, self._flags).match(pos, endpos)
        )

    def fullmatch(self, pos: int = 0, endpos: tp.Optional[int] = None) -> 'Batch':
        """
        If the whole string matches this regular expression, return True, else False. Note that this is different from a zero-length match.

        Args:
            pos: Gives an index in the string where the search is to start; it defaults to 0.
            endpos: Limits how far the string will be searched; it will be as if the string is endpos characters long.
        """
        return self._batch_apply(
            lambda c: c.via_re(self._pattern, self._flags).fullmatch(pos, endpos)
        )

    def split(self, maxsplit: int = 0) -> 'Batch':
        """
        Split string by the occurrences of pattern. If capturing parentheses are used in pattern, then the text of all groups in the pattern are also returned as part of the resulting tuple.

        Args:
            maxsplit: If nonzero, at most maxsplit splits occur, and the remainder of the string is returned as the final element of the tuple.
        """
        return self._batch_apply(
            lambda c: c.via_re(self._pattern, self._flags).split(maxsplit)
        )

    def findall(self, pos: int = 0, endpos: tp.Optional[int] = None) -> 'Batch':
        """
        Return all non-overlapping matches of pattern in string, as a tuple of strings. The string is scanned left-to-right, and matches are returned in the order found. If one or more groups are present in the pattern, return a tuple of groups; this will be a tuple of tuples if the pattern has more than one group. Empty matches are included in the result.

        Args:
            pos: Gives an index in the string where the search is to start; it defaults to 0.
            endpos: Limits how far the string will be searched; it will be as if the string is endpos characters long.
        """
        return self._batch_apply(
            lambda c: c.via_re(self._pattern, self._flags).findall(pos, endpos)
        )

    def sub(self, repl: str, count: int = 0) -> 'Batch':
        """
        Return the string obtained by replacing the leftmost non-overlapping occurrences of pattern in string by the replacement ``repl``. If the pattern is not found, the string is returned unchanged.

        Args:
            repl: A string or a function; if it is a string, any backslash escapes in it are processed.
            count: The optional argument count is the maximum number of pattern occurrences to be replaced; count must be a non-negative integer. If omitted or zero, all occurrences will be replaced.
        """
        return self._batch_apply(
            lambda c: c.via_re(self._pattern, self._flags).sub(repl, count)
        )

    def subn(self, repl: str, count: int = 0) -> 'Batch':
        """
        Perform the same operation as sub(), but return a tuple (new_string, number_of_subs_made).

        Args:
            repl: A string or a function; if it is a string, any backslash escapes in it are processed.
            count: The optional argument count is the maximum number of pattern occurrences to be replaced; count must be a non-negative integer. If omitted or zero, all occurrences will be replaced.
        """
        return self._batch_apply(
            lambda c: c.via_re(self._pattern, self._flags).subn(repl, count)
        )
