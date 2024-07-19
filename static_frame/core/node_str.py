from __future__ import annotations

from functools import reduce

import numpy as np
import typing_extensions as tp
from numpy import char as npc

from static_frame.core.container_util import get_col_format_factory
from static_frame.core.node_selector import Interface
from static_frame.core.node_selector import InterfaceBatch
from static_frame.core.node_selector import TVContainer_co
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import DTYPE_STR
from static_frame.core.util import DTYPE_STR_KINDS
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import OPERATORS
from static_frame.core.util import TCallableAny
from static_frame.core.util import TLabel
from static_frame.core.util import TLocSelector
from static_frame.core.util import TUFunc
from static_frame.core.util import array_from_element_apply
from static_frame.core.util import array_from_element_method

if tp.TYPE_CHECKING:
    from static_frame.core.batch import Batch  # pragma: no cover
    from static_frame.core.index_base import IndexBase  # pragma: no cover
    TNDArrayAny = np.ndarray[tp.Any, tp.Any] #pragma: no cover
    TDtypeAny = np.dtype[tp.Any] #pragma: no cover

    BlocksType = tp.Iterable[TNDArrayAny] #pragma: no cover
    ToContainerType = tp.Callable[[tp.Iterator[TNDArrayAny]], TVContainer_co] #pragma: no cover

INTERFACE_STR = (
        '__getitem__',
        'capitalize',
        'center',
        'contains',
        'count',
        'decode',
        'encode',
        'endswith',
        'find',
        'format',
        'index',
        'isalnum',
        'isalpha',
        'isdecimal',
        'isdigit',
        'islower',
        'isnumeric',
        'isspace',
        'istitle',
        'isupper',
        'ljust',
        'len',
        'lower',
        'lstrip',
        'partition',
        'replace',
        'rfind',
        'rindex',
        'rjust',
        'rpartition',
        'rsplit',
        'rstrip',
        'split',
        'startswith',
        'strip',
        'swapcase',
        'title',
        'upper',
        'zfill',
)


class InterfaceString(Interface, tp.Generic[TVContainer_co]):

    # NOTE: based on https://numpy.org/doc/stable/reference/routines.char.html

    __slots__ = (
            '_blocks',
            '_blocks_to_container',
            '_ndim',
            '_labels',
            )
    _INTERFACE = INTERFACE_STR

    def __init__(self,
            blocks: BlocksType,
            blocks_to_container: ToContainerType[TVContainer_co], # type: ignore[type-var]
            ndim: int,
            labels: tp.Sequence[TLabel] | IndexBase,
            ) -> None:
        self._blocks: BlocksType = blocks
        self._blocks_to_container: ToContainerType[TVContainer_co] = blocks_to_container
        self._ndim: int = ndim
        self._labels: tp.Sequence[TLabel] | IndexBase = labels

    #---------------------------------------------------------------------------

    @staticmethod
    def _process_blocks(
            blocks: BlocksType,
            func: TUFunc,
            args: tp.Tuple[tp.Any, ...] = (),
            astype_str: bool = True,
            ) -> tp.Iterator[TNDArrayAny]:
        '''
        Block-wise processing of blocks after optional string conversion. Non-string conversion is necessary for ``decode``.
        '''
        for block in blocks:
            if astype_str and block.dtype.kind not in DTYPE_STR_KINDS:
                block = block.astype(DTYPE_STR)
            array = func(block, *args)
            array.flags.writeable = False
            yield array

    @staticmethod
    def _process_tuple_blocks(*,
            blocks: BlocksType,
            method_name: str,
            dtype: TDtypeAny,
            args: tp.Tuple[tp.Any, ...] = (),
            ) -> tp.Iterator[TNDArrayAny]:
        '''
        Element-wise processing of a methods on objects in a block, with pre-insert conversion to a tuple.
        '''
        for block in blocks:
            if block.dtype.kind not in DTYPE_STR_KINDS:
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

    @staticmethod
    def _process_element_blocks(*,
            blocks: BlocksType,
            method_name: str,
            dtype: TDtypeAny,
            args: tp.Tuple[tp.Any, ...] = (),
            ) -> tp.Iterator[TNDArrayAny]:
        '''
        Element-wise processing of a methods on objects in a block, with pre-insert conversion to a tuple.
        '''
        for block in blocks:
            if block.dtype.kind not in DTYPE_STR_KINDS:
                block = block.astype(DTYPE_STR)

            # resultant array is immutable
            array = array_from_element_method(
                    array=block,
                    method_name=method_name,
                    args=args,
                    dtype=dtype,
                    )
            yield array


    #---------------------------------------------------------------------------
    def __getitem__(self,  key: TLocSelector) -> TVContainer_co:
        '''
        Return a container with the provided selection or slice of each element.
        '''
        block_gen = self._process_element_blocks(
                blocks=self._blocks,
                method_name='__getitem__',
                args=(key,),
                dtype=DTYPE_STR,
                )
        return self._blocks_to_container(block_gen)

    def capitalize(self) -> TVContainer_co:
        '''
        Return a container with only the first character of each element capitalized.
        '''
        # return self._blocks_to_container(npc.capitalize(self._blocks()))
        block_gen = self._process_blocks(self._blocks, npc.capitalize)
        return self._blocks_to_container(block_gen)

    def center(self,
            width: int,
            fillchar: str = ' '
            ) -> TVContainer_co:
        '''
        Return a container with its elements centered in a string of length ``width``.
        '''
        block_gen = self._process_blocks(self._blocks, npc.center, (width, fillchar))
        return self._blocks_to_container(block_gen)

    def contains(self,  item: str) -> TVContainer_co:
        '''
        Return a Boolean container showing True of item is a substring of elements.
        '''
        block_gen = self._process_element_blocks(
                blocks=self._blocks,
                method_name='__contains__',
                args=(item,),
                dtype=DTYPE_BOOL,
                )
        return self._blocks_to_container(block_gen)

    def count(self,
            sub: str,
            start: int = 0,
            end: tp.Optional[int] = None
            ) -> TVContainer_co:
        '''
        Returns a container with the number of non-overlapping occurrences of substring sub in the optional range ``start``, ``end``.
        '''
        block_gen = self._process_blocks(self._blocks, npc.count, (sub, start, end))
        return self._blocks_to_container(block_gen)

    def decode(self,
            encoding: tp.Optional[str] = None,
            errors: tp.Optional[str] = None,
            ) -> TVContainer_co:
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
            ) -> TVContainer_co:
        '''
        Apply str.encode() to each element. Elements must be strings.
        '''
        block_gen = self._process_blocks(self._blocks, npc.encode, (encoding, errors))
        return self._blocks_to_container(block_gen)

    def endswith(self,
            suffix: tp.Union[str, tp.Iterable[str]],
            start: int = 0,
            end: tp.Optional[int] = None
            ) -> TVContainer_co:
        '''
        Returns a container with the number of non-overlapping occurrences of substring ``suffix`` (or an iterable of suffixes) in the optional range ``start``, ``end``.
        '''

        if isinstance(suffix, str):
            block_iter = self._process_blocks(self._blocks, npc.endswith, (suffix, start, end))
            return self._blocks_to_container(block_iter)

        def block_gen() -> tp.Iterator[TNDArrayAny]:
            blocks_per_sub = (
                    self._process_blocks(self._blocks, npc.endswith, (sub, start, end))
                    for sub in suffix)
            func = OPERATORS['__or__']
            for block_layers in zip(*blocks_per_sub):
                array = reduce(func, block_layers)
                array.flags.writeable = False
                yield array

        return self._blocks_to_container(block_gen())

    def find(self,
            sub: str,
            start: int = 0,
            end: tp.Optional[int] = None
            ) -> TVContainer_co:
        '''
        For each element, return the lowest index in the string where substring ``sub`` is found.
        '''
        block_gen = self._process_blocks(self._blocks, npc.find, (sub, start, end))
        return self._blocks_to_container(block_gen)

    def format(self, format: str) -> TVContainer_co:
        '''
        For each element, return a string resulting from calling the string ``format`` argument's ``format`` method with the the element. Format strings (given within curly braces) can use Python's format mini language: https://docs.python.org/3/library/string.html#formatspec

        Args:
            format: A string, an iterable of strings, or a mapping of labels to strings. For 1D containers, an iterable of strings must be of length equal to the container; a mapping can use Index labels (for a Series) or positions (for an Index). For 2D containers, an iterable of strings must be of length equal to the columns (for a Frame) or the depth (for an Index Hierarchy); a mapping can use column labels (for a Frame) or depths (for an IndexHierarchy).
        '''

        format_factory = get_col_format_factory(format, self._labels)

        if self._ndim == 1:
            # apply the format per label in series
            def block_gen() -> tp.Iterator[TNDArrayAny]:
                post = []
                for i, v in enumerate(next(iter(self._blocks))):
                    func = format_factory(i).format
                    post.append(func(v))
                array = np.array(post, dtype=DTYPE_STR)
                array.flags.writeable = False
                yield array
        else:
            def block_gen() -> tp.Iterator[TNDArrayAny]:
                pos = 0
                for block in self._blocks:
                    if block.ndim == 1:
                        func = format_factory(pos).format
                        yield array_from_element_apply(block, func, DTYPE_STR)
                        pos += 1
                    else:
                        for i in range(block.shape[1]):
                            func = format_factory(pos).format
                            yield array_from_element_apply(
                                    block[NULL_SLICE, i],
                                    func,
                                    DTYPE_STR,
                                    )
                            pos += 1
        return self._blocks_to_container(block_gen())

    def index(self,
            sub: str,
            start: int = 0,
            end: tp.Optional[int] = None
            ) -> TVContainer_co:
        '''
        Like ``find``, but raises ``ValueError`` when the substring is not found.
        '''
        block_gen = self._process_blocks(self._blocks, npc.index, (sub, start, end))
        return self._blocks_to_container(block_gen)

    def isalnum(self) -> TVContainer_co:
        '''
        Returns true for each element if all characters in the string are alphanumeric and there is at least one character, false otherwise.
        '''
        block_gen = self._process_blocks(self._blocks, npc.isalnum)
        return self._blocks_to_container(block_gen)

    def isalpha(self) -> TVContainer_co:
        '''
        Returns true for each element if all characters in the string are alphabetic and there is at least one character, false otherwise.
        '''
        block_gen = self._process_blocks(self._blocks, npc.isalpha)
        return self._blocks_to_container(block_gen)

    def isdecimal(self) -> TVContainer_co:
        '''
        For each element, return True if there are only decimal characters in the element.
        '''
        block_gen = self._process_blocks(self._blocks, npc.isdecimal)
        return self._blocks_to_container(block_gen)

    def isdigit(self) -> TVContainer_co:
        '''
        Returns true for each element if all characters in the string are digits and there is at least one character, false otherwise.
        '''
        block_gen = self._process_blocks(self._blocks, npc.isdigit)
        return self._blocks_to_container(block_gen)

    def islower(self) -> TVContainer_co:
        '''
        Returns true for each element if all cased characters in the string are lowercase and there is at least one cased character, false otherwise.
        '''
        block_gen = self._process_blocks(self._blocks, npc.islower)
        return self._blocks_to_container(block_gen)

    def isnumeric(self) -> TVContainer_co:
        '''
        For each element in self, return True if there are only numeric characters in the element.
        '''
        block_gen = self._process_blocks(self._blocks, npc.isnumeric)
        return self._blocks_to_container(block_gen)

    def isspace(self) -> TVContainer_co:
        '''
        Returns true for each element if there are only whitespace characters in the string and there is at least one character, false otherwise.
        '''
        block_gen = self._process_blocks(self._blocks, npc.isspace)
        return self._blocks_to_container(block_gen)

    def istitle(self) -> TVContainer_co:
        '''
        Returns true for each element if the element is a titlecased string and there is at least one character, false otherwise.
        '''
        block_gen = self._process_blocks(self._blocks, npc.istitle)
        return self._blocks_to_container(block_gen)

    def isupper(self) -> TVContainer_co:
        '''
        Returns true for each element if all cased characters in the string are uppercase and there is at least one character, false otherwise.
        '''
        block_gen = self._process_blocks(self._blocks, npc.isupper)
        return self._blocks_to_container(block_gen)

    def len(self) -> TVContainer_co:
        '''
        Return the length of the string.
        '''
        block_gen = self._process_blocks(self._blocks, npc.str_len)
        return self._blocks_to_container(block_gen)

    def ljust(self,
            width: int,
            fillchar: str = ' '
            ) -> TVContainer_co:
        '''
        Return a container with its elements ljusted in a string of length ``width``.
        '''
        block_gen = self._process_blocks(self._blocks, npc.ljust, (width, fillchar))
        return self._blocks_to_container(block_gen)

    def lower(self) -> TVContainer_co:
        '''
        Return an array with the elements of self converted to lowercase.
        '''
        block_gen = self._process_blocks(self._blocks, npc.lower)
        return self._blocks_to_container(block_gen)

    def lstrip(self,
            chars: tp.Optional[str] = None,
            ) -> TVContainer_co:
        '''
        For each element, return a copy with the leading characters removed.
        '''
        block_gen = self._process_blocks(self._blocks, npc.lstrip, (chars,))
        return self._blocks_to_container(block_gen)

    def partition(self,
            sep: str,
            ) -> TVContainer_co:
        '''
        Partition each element around ``sep``.
        '''
        # NOTE: py str.partition gives a tuple.
        block_gen = self._process_tuple_blocks(
                blocks=self._blocks,
                method_name='partition',
                args=(sep,),
                dtype=DTYPE_OBJECT,
                )
        return self._blocks_to_container(block_gen)

    def replace(self,
            old: str,
            new: str,
            count: int = -1,
            ) -> TVContainer_co:
        '''
        Return a container with its elements replaced in a string of length ``width``.
        '''
        block_gen = self._process_blocks(self._blocks, npc.replace, (old, new, count))
        return self._blocks_to_container(block_gen)

    def rfind(self,
            sub: str,
            start: int = 0,
            end: tp.Optional[int] = None
            ) -> TVContainer_co:
        '''
        For each element, return the highest index in the string where substring ``sub`` is found, such that sub is contained within ``start``, ``end``.
        '''
        block_gen = self._process_blocks(self._blocks, npc.rfind, (sub, start, end))
        return self._blocks_to_container(block_gen)

    def rindex(self,
            sub: str,
            start: int = 0,
            end: tp.Optional[int] = None
            ) -> TVContainer_co:
        '''
        Like ``rfind``, but raises ``ValueError`` when the substring ``sub`` is not found.
        '''
        block_gen = self._process_blocks(self._blocks, npc.rindex, (sub, start, end))
        return self._blocks_to_container(block_gen)

    def rjust(self,
            width: int,
            fillchar: str = ' '
            ) -> TVContainer_co:
        '''
        Return a container with its elements rjusted in a string of length ``width``.
        '''
        block_gen = self._process_blocks(self._blocks, npc.rjust, (width, fillchar))
        return self._blocks_to_container(block_gen)

    def rpartition(self,
            sep: str,
            ) -> TVContainer_co:
        '''
        Partition (split) each element around the right-most separator.
        '''
        # NOTE: py str.rpartition gives a tuple.
        block_gen = self._process_tuple_blocks(
                blocks=self._blocks,
                method_name='rpartition',
                args=(sep,),
                dtype=DTYPE_OBJECT,
                )
        return self._blocks_to_container(block_gen)

    def rsplit(self,
            sep: str,
            maxsplit: int = -1,
            ) -> TVContainer_co:
        '''
        For each element, return a tuple of the words in the string, using sep as the delimiter string.
        '''
        # NOTE: npc.rsplit gives an array of lists, so implement our own routine to get an array of tuples.
        block_gen = self._process_tuple_blocks(
                blocks=self._blocks,
                method_name='rsplit',
                args=(sep, maxsplit),
                dtype=DTYPE_OBJECT,
                )
        return self._blocks_to_container(block_gen)

    def rstrip(self,
            chars: tp.Optional[str] = None,
            ) -> TVContainer_co:
        '''
        For each element, return a copy with the trailing characters removed.
        '''
        block_gen = self._process_blocks(self._blocks, npc.rstrip, (chars,))
        return self._blocks_to_container(block_gen)

    def split(self,
            sep: str,
            maxsplit: int = -1,
            ) -> TVContainer_co:
        '''
        For each element, return a tuple of the words in the string, using sep as the delimiter string.
        '''
        # NOTE: npc.split gives an array of lists, so implement our own routine to get an array of tuples.
        block_gen = self._process_tuple_blocks(
                blocks=self._blocks,
                method_name='split',
                args=(sep, maxsplit),
                dtype=DTYPE_OBJECT,
                )
        return self._blocks_to_container(block_gen)

    #splitlines: not likely useful

    def startswith(self,
            prefix: tp.Union[str, tp.Iterable[str]],
            start: int = 0,
            end: tp.Optional[int] = None
            ) -> TVContainer_co:
        '''
        Returns a container with the number of non-overlapping occurrences of substring `prefix` (or an iterable of prefixes) in the optional range ``start``, ``end``.
        '''
        if isinstance(prefix, str):
            block_iter = self._process_blocks(self._blocks, npc.startswith, (prefix, start, end))
            return self._blocks_to_container(block_iter)

        def block_gen() -> tp.Iterator[TNDArrayAny]:
            blocks_per_sub = (
                    self._process_blocks(self._blocks, npc.startswith, (sub, start, end))
                    for sub in prefix)
            func = OPERATORS['__or__']
            for block_layers in zip(*blocks_per_sub):
                array = reduce(func, block_layers)
                array.flags.writeable = False
                yield array

        return self._blocks_to_container(block_gen())

    def strip(self,
            chars: tp.Optional[str] = None,
            ) -> TVContainer_co:
        '''
        For each element, return a copy with the leading and trailing characters removed.
        '''
        block_gen = self._process_blocks(self._blocks, npc.strip, (chars,))
        return self._blocks_to_container(block_gen)

    def swapcase(self) -> TVContainer_co:
        '''
        Return a container with uppercase characters converted to lowercase and vice versa.
        '''
        block_gen = self._process_blocks(self._blocks, npc.swapcase)
        return self._blocks_to_container(block_gen)

    def title(self) -> TVContainer_co:
        '''
        Return a container with uppercase characters converted to lowercase and vice versa.
        '''
        block_gen = self._process_blocks(self._blocks, npc.title)
        return self._blocks_to_container(block_gen)

    # translate: akward input

    def upper(self) -> TVContainer_co:
        '''
        Return a container with uppercase characters converted to lowercase and vice versa.
        '''
        block_gen = self._process_blocks(self._blocks, npc.upper)
        return self._blocks_to_container(block_gen)

    def zfill(self,
            width: int,
            ) -> TVContainer_co:
        '''
        Return the string left-filled with zeros.
        '''
        block_gen = self._process_blocks(self._blocks, npc.zfill, (width,))
        return self._blocks_to_container(block_gen)



class InterfaceBatchString(InterfaceBatch):
    '''Alternate string interface specialized for the :obj:`Batch`.
    '''
    __slots__ = (
            '_batch_apply',
            )
    _INTERFACE = INTERFACE_STR

    def __init__(self,
            batch_apply: tp.Callable[[TCallableAny], 'Batch'],
            ) -> None:
        self._batch_apply = batch_apply

    #---------------------------------------------------------------------------
    def __getitem__(self,  key: TLocSelector) -> 'Batch':
        '''
        Return a container with the provided selection or slice of each element.
        '''
        return self._batch_apply(lambda c: c.via_str.__getitem__(key))


    def capitalize(self) -> 'Batch':
        '''
        Return a container with only the first character of each element capitalized.
        '''
        return self._batch_apply(lambda c: c.via_str.capitalize())

    def center(self,
            width: int,
            fillchar: str = ' '
            ) -> 'Batch':
        '''
        Return a container with its elements centered in a string of length ``width``.
        '''
        return self._batch_apply(lambda c: c.via_str.center(width, fillchar))

    def count(self,
            sub: str,
            start: int = 0,
            end: tp.Optional[int] = None
            ) -> 'Batch':
        '''
        Returns a container with the number of non-overlapping occurrences of substring sub in the optional range ``start``, ``end``.
        '''
        return self._batch_apply(lambda c: c.via_str.count(sub, start, end))

    def contains(self,
            item: str,
            ) -> 'Batch':
        '''
        Returns a container with the number of non-overlapping occurrences of substring sub in the optional range ``start``, ``end``.
        '''
        return self._batch_apply(lambda c: c.via_str.contains(item))

    def decode(self,
            encoding: tp.Optional[str] = None,
            errors: tp.Optional[str] = None,
            ) -> 'Batch':
        '''
        Apply str.decode() to each element. Elements must be bytes.
        '''
        return self._batch_apply(lambda c: c.via_str.decode(encoding, errors))

    def encode(self,
            encoding: tp.Optional[str] = None,
            errors: tp.Optional[str] = None,
            ) -> 'Batch':
        '''
        Apply str.encode() to each element. Elements must be strings.
        '''
        return self._batch_apply(lambda c: c.via_str.encode(encoding, errors))

    def endswith(self,
            suffix: tp.Union[str, tp.Iterable[str]],
            start: int = 0,
            end: tp.Optional[int] = None
            ) -> 'Batch':
        '''
        Returns a container with the number of non-overlapping occurrences of substring ``suffix`` (or an iterable of suffixes) in the optional range ``start``, ``end``.
        '''
        return self._batch_apply(lambda c: c.via_str.endswith(suffix, start, end))

    def find(self,
            sub: str,
            start: int = 0,
            end: tp.Optional[int] = None
            ) -> 'Batch':
        '''
        For each element, return the lowest index in the string where substring ``sub`` is found.
        '''
        return self._batch_apply(lambda c: c.via_str.find(sub, start, end))

    def format(self,
            format: str,
            ) -> 'Batch':
        '''
        For each element, return a string resulting from calling the `format` argument's `format` method with the values in this container.
        '''
        return self._batch_apply(lambda c: c.via_str.format(format))

    def index(self,
            sub: str,
            start: int = 0,
            end: tp.Optional[int] = None
            ) -> 'Batch':
        '''
        Like ``find``, but raises ``ValueError`` when the substring is not found.
        '''
        return self._batch_apply(lambda c: c.via_str.index(sub, start, end))

    def isalnum(self) -> 'Batch':
        '''
        Returns true for each element if all characters in the string are alphanumeric and there is at least one character, false otherwise.
        '''
        return self._batch_apply(lambda c: c.via_str.isalnum())

    def isalpha(self) -> 'Batch':
        '''
        Returns true for each element if all characters in the string are alphabetic and there is at least one character, false otherwise.
        '''
        return self._batch_apply(lambda c: c.via_str.isalpha())

    def isdecimal(self) -> 'Batch':
        '''
        For each element, return True if there are only decimal characters in the element.
        '''
        return self._batch_apply(lambda c: c.via_str.isdecimal())

    def isdigit(self) -> 'Batch':
        '''
        Returns true for each element if all characters in the string are digits and there is at least one character, false otherwise.
        '''
        return self._batch_apply(lambda c: c.via_str.isdigit())

    def islower(self) -> 'Batch':
        '''
        Returns true for each element if all cased characters in the string are lowercase and there is at least one cased character, false otherwise.
        '''
        return self._batch_apply(lambda c: c.via_str.islower())

    def isnumeric(self) -> 'Batch':
        '''
        For each element in self, return True if there are only numeric characters in the element.
        '''
        return self._batch_apply(lambda c: c.via_str.isnumeric())

    def isspace(self) -> 'Batch':
        '''
        Returns true for each element if there are only whitespace characters in the string and there is at least one character, false otherwise.
        '''
        return self._batch_apply(lambda c: c.via_str.isspace())

    def istitle(self) -> 'Batch':
        '''
        Returns true for each element if the element is a titlecased string and there is at least one character, false otherwise.
        '''
        return self._batch_apply(lambda c: c.via_str.istitle())

    def isupper(self) -> 'Batch':
        '''
        Returns true for each element if all cased characters in the string are uppercase and there is at least one character, false otherwise.
        '''
        return self._batch_apply(lambda c: c.via_str.isupper())

    def len(self) -> 'Batch':
        '''
        Return the length of the string.
        '''
        return self._batch_apply(lambda c: c.via_str.len())

    def ljust(self,
            width: int,
            fillchar: str = ' '
            ) -> 'Batch':
        '''
        Return a container with its elements ljusted in a string of length ``width``.
        '''
        return self._batch_apply(lambda c: c.via_str.ljust(width, fillchar))

    def lower(self) -> 'Batch':
        '''
        Return an array with the elements of self converted to lowercase.
        '''
        return self._batch_apply(lambda c: c.via_str.lower())

    def lstrip(self,
            chars: tp.Optional[str] = None,
            ) -> 'Batch':
        '''
        For each element, return a copy with the leading characters removed.
        '''
        return self._batch_apply(lambda c: c.via_str.lstrip(chars))

    def partition(self,
            sep: str,
            ) -> 'Batch':
        '''
        Partition each element around ``sep``.
        '''
        return self._batch_apply(lambda c: c.via_str.partition(sep))

    def replace(self,
            old: str,
            new: str,
            count: int = -1,
            ) -> 'Batch':
        '''
        Return a container with its elements replaced in a string of length ``width``.
        '''
        return self._batch_apply(lambda c: c.via_str.replace(old, new, count))

    def rfind(self,
            sub: str,
            start: int = 0,
            end: tp.Optional[int] = None
            ) -> 'Batch':
        '''
        For each element, return the highest index in the string where substring ``sub`` is found, such that sub is contained within ``start``, ``end``.
        '''
        return self._batch_apply(lambda c: c.via_str.rfind(sub, start, end))

    def rindex(self,
            sub: str,
            start: int = 0,
            end: tp.Optional[int] = None
            ) -> 'Batch':
        '''
        Like ``rfind``, but raises ``ValueError`` when the substring ``sub`` is not found.
        '''
        return self._batch_apply(lambda c: c.via_str.rindex(sub, start, end))

    def rjust(self,
            width: int,
            fillchar: str = ' '
            ) -> 'Batch':
        '''
        Return a container with its elements rjusted in a string of length ``width``.
        '''
        return self._batch_apply(lambda c: c.via_str.rjust(width, fillchar))

    def rpartition(self,
            sep: str,
            ) -> 'Batch':
        '''
        Partition (split) each element around the right-most separator.
        '''
        return self._batch_apply(lambda c: c.via_str.rpartition(sep))

    def rsplit(self,
            sep: str,
            maxsplit: int = -1,
            ) -> 'Batch':
        '''
        For each element, return a tuple of the words in the string, using sep as the delimiter string.
        '''
        return self._batch_apply(lambda c: c.via_str.rsplit(sep, maxsplit))

    def rstrip(self,
            chars: tp.Optional[str] = None,
            ) -> 'Batch':
        '''
        For each element, return a copy with the trailing characters removed.
        '''
        return self._batch_apply(lambda c: c.via_str.rstrip(chars))

    def split(self,
            sep: str,
            maxsplit: int = -1,
            ) -> 'Batch':
        '''
        For each element, return a tuple of the words in the string, using sep as the delimiter string.
        '''
        return self._batch_apply(lambda c: c.via_str.split(sep, maxsplit))

    #splitlines: not likely useful

    def startswith(self,
            prefix: tp.Union[str, tp.Iterable[str]],
            start: int = 0,
            end: tp.Optional[int] = None
            ) -> 'Batch':
        '''
        Returns a container with the number of non-overlapping occurrences of substring `prefix` (or an iterable of prefixes) in the optional range ``start``, ``end``.
        '''
        return self._batch_apply(lambda c: c.via_str.startswith(prefix, start, end))

    def strip(self,
            chars: tp.Optional[str] = None,
            ) -> 'Batch':
        '''
        For each element, return a copy with the leading and trailing characters removed.
        '''
        return self._batch_apply(lambda c: c.via_str.strip(chars))

    def swapcase(self) -> 'Batch':
        '''
        Return a container with uppercase characters converted to lowercase and vice versa.
        '''
        return self._batch_apply(lambda c: c.via_str.swapcase())

    def title(self) -> 'Batch':
        '''
        Return a container with uppercase characters converted to lowercase and vice versa.
        '''
        return self._batch_apply(lambda c: c.via_str.title())

    # translate: akward input

    def upper(self) -> 'Batch':
        '''
        Return a container with uppercase characters converted to lowercase and vice versa.
        '''
        return self._batch_apply(lambda c: c.via_str.upper())

    def zfill(self,
            width: int,
            ) -> 'Batch':
        '''
        Return the string left-filled with zeros.
        '''
        return self._batch_apply(lambda c: c.via_str.zfill(width))

