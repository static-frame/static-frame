
import typing as tp
import numpy as np
from numpy import char as npc #type: ignore

if tp.TYPE_CHECKING:

    from static_frame.core.frame import Frame  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index import Index  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchy  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.series import Series  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.type_blocks import TypeBlocks  #pylint: disable = W0611 #pragma: no cover

# only ContainerOperand subclasses
TContainer = tp.TypeVar('TContainer', 'Index', 'IndexHierarchy', 'Series', 'Frame', 'TypeBlocks')

ToArrayType = tp.Callable[[], np.ndarray]
ToContainerType = tp.Callable[[np.ndarray], TContainer]


class InterfaceStr(tp.Generic[TContainer]):

    # NOTE: based on https://numpy.org/doc/stable/reference/routines.char.html

    __slots__ = (
        '_func_to_array', # function that returns array of strings
        '_func_to_container', # partialed function that will return a new container
        )

    def __init__(self,
            func_to_array: ToArrayType,
            func_to_container: ToContainerType[TContainer]
            ) -> None:
        self._func_to_array: ToArrayType = func_to_array
        self._func_to_container: ToContainerType[TContainer] = func_to_container

    def capitalize(self) -> TContainer:
        '''
        Return a container with only the first character of each element capitalized.
        '''
        return self._func_to_container(npc.capitalize(self._func_to_array()))

    def center(self,
            width: int,
            fillchar: str = ' '
            ) -> TContainer:
        '''
        Return a container with its elements centered in a string of length ``width``.
        '''
        array = npc.center(
                self._func_to_array(),
                width=width,
                fillchar=fillchar,
                )
        array.flags.writeable = False
        return self._func_to_container(array)

    def decode(self,
            encoding: tp.Optional[str] = None,
            errors: tp.Optional[str] = None,
            ) -> TContainer:
        '''
        Apply str.decode() to each element. Elements must be bytes.
        '''
        array = npc.decode(
                self._func_to_array(),
                encoding=encoding,
                errors=errors,
                )
        array.flags.writeable = False
        return self._func_to_container(array)

    def encode(self,
            encoding: tp.Optional[str] = None,
            errors: tp.Optional[str] = None,
            ) -> TContainer:
        '''
        Apply str.encode() to each element. Elements must be strings.
        '''
        array = npc.encode(
                self._func_to_array(),
                encoding=encoding,
                errors=errors,
                )
        array.flags.writeable = False
        return self._func_to_container(array)

    # join: processes two arrays

    def ljust(self,
            width: int,
            fillchar: str = ' '
            ) -> TContainer:
        '''
        Return a container with its elements ljusted in a string of length ``width``.
        '''
        array = npc.ljust(
                self._func_to_array(),
                width=width,
                fillchar=fillchar,
                )
        array.flags.writeable = False
        return self._func_to_container(array)

    # partition: np returns a 2D array; could return a Series of tuples

    def replace(self,
            old: str,
            new: str,
            count: tp.Optional[int] = None,
            ) -> TContainer:
        '''
        Return a container with its elements replaced in a string of length ``width``.
        '''
        array = npc.replace(
                self._func_to_array(),
                old=old,
                new=new,
                count=count,
                )
        array.flags.writeable = False
        return self._func_to_container(array)

    def rjust(self,
            width: int,
            fillchar: str = ' '
            ) -> TContainer:
        '''
        Return a container with its elements rjusted in a string of length ``width``.
        '''
        array = npc.rjust(
                self._func_to_array(),
                width=width,
                fillchar=fillchar,
                )
        array.flags.writeable = False
        return self._func_to_container(array)

    # rpartition

    def rsplit(self,
            sep: str,
            maxsplit: int = -1,
            ) -> TContainer:
        '''
        For each element, return a tuple of the words in the string, using sep as the delimiter string.
        '''
        # NOTE: npc.rsplit gives an array of lists, so implement our own routine to get an array of tuples.

        # convert lists to tuples
        src = self._func_to_array()
        size = len(src)
        dst = np.empty(size, dtype=object)
        for idx in range(size):
            dst[idx] = tuple(src[idx].rsplit(sep, maxsplit))

        dst.flags.writeable = False
        return self._func_to_container(dst)

    def rstrip(self,
            chars: tp.Optional[str] = None,
            ) -> TContainer:
        '''
        For each element, return a copy with the trailing characters removed.
        '''
        array = npc.rstrip(
                self._func_to_array(),
                chars=chars,
                )
        array.flags.writeable = False
        return self._func_to_container(array)

    def split(self,
            sep: str,
            maxsplit: int = -1,
            ) -> TContainer:
        '''
        For each element, return a tuple of the words in the string, using sep as the delimiter string.
        '''
        # NOTE: npc.split gives an array of lists, so implement our own routine to get an array of tuples.

        # convert lists to tuples
        src = self._func_to_array()

        # if src.shape is 2D, will need a different implementation
        size = len(src)
        dst = np.empty(size, dtype=object)
        for idx in range(size):
            dst[idx] = tuple(src[idx].split(sep, maxsplit))

        dst.flags.writeable = False
        return self._func_to_container(dst)

    def strip(self,
            chars: tp.Optional[str] = None,
            ) -> TContainer:
        '''
        For each element, return a copy with the leading and trailing characters removed.
        '''
        array = npc.strip(
                self._func_to_array(),
                chars=chars,
                )
        array.flags.writeable = False
        return self._func_to_container(array)

    def swapcase(self) -> TContainer:
        '''
        Return a container with uppercase characters converted to lowercase and vice versa.
        '''
        array = npc.swapcase(self._func_to_array())
        array.flags.writeable = False
        return self._func_to_container(array)

    def title(self) -> TContainer:
        '''
        Return a container with uppercase characters converted to lowercase and vice versa.
        '''
        array = npc.title(self._func_to_array())
        array.flags.writeable = False
        return self._func_to_container(array)

    # translate: akward input

    def upper(self) -> TContainer:
        '''
        Return a container with uppercase characters converted to lowercase and vice versa.
        '''
        array = npc.upper(self._func_to_array())
        array.flags.writeable = False
        return self._func_to_container(array)

    def zfill(self,
            width: int,
            ) -> TContainer:
        '''
        Return the string left-filled with zeros.
        '''
        array = npc.zfill(
                self._func_to_array(),
                width=width,
                )
        array.flags.writeable = False
        return self._func_to_container(array)

    #---------------------------------------------------------------------------

    def count(self,
            sub: str,
            start: tp.Optional[int] = None,
            end: tp.Optional[int] = None
            ) -> TContainer:
        '''
        Returns a container with the number of non-overlapping occurrences of substring sub in the optional range ``start``, ``end``.
        '''
        array = npc.count(
                self._func_to_array(),
                sub=sub,
                start=start,
                end=end,
                )
        array.flags.writeable = False
        return self._func_to_container(array)

    def endswith(self,
            suffix: str,
            start: tp.Optional[int] = None,
            end: tp.Optional[int] = None
            ) -> TContainer:
        '''
        Returns a container with the number of non-overlapping occurrences of substring sub in the optional range ``start``, ``end``.
        '''
        array = npc.endswith(
                self._func_to_array(),
                suffix=suffix,
                start=start,
                end=end,
                )
        array.flags.writeable = False
        return self._func_to_container(array)

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
        array = npc.startswith(
                self._func_to_array(),
                prefix=prefix,
                start=start,
                end=end,
                )
        array.flags.writeable = False
        return self._func_to_container(array)