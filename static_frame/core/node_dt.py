
import typing as tp
from datetime import datetime
from datetime import date

import numpy as np

from static_frame.core.node_selector import Interface
from static_frame.core.node_selector import TContainer
from static_frame.core.util import array_from_element_attr
from static_frame.core.util import array_from_element_method
from static_frame.core.util import array_from_element_apply
from static_frame.core.util import DT64_AS
from static_frame.core.util import DT64_DAY
from static_frame.core.util import DT64_FS
from static_frame.core.util import DT64_H
from static_frame.core.util import DT64_M
from static_frame.core.util import DT64_MONTH
from static_frame.core.util import DT64_MS
from static_frame.core.util import DT64_NS
from static_frame.core.util import DT64_PS
from static_frame.core.util import DT64_S
from static_frame.core.util import DT64_US
from static_frame.core.util import DT64_YEAR
from static_frame.core.util import DTYPE_DATETIME_KIND
from static_frame.core.util import DTYPE_INT_DEFAULT
from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import DTYPE_STR
from static_frame.core.util import EMPTY_TUPLE
from static_frame.core.util import DTYPE_STR_KINDS

if tp.TYPE_CHECKING:

    from static_frame.core.frame import Frame  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index import Index  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchy  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.series import Series  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.type_blocks import TypeBlocks  #pylint: disable = W0611 #pragma: no cover

BlocksType = tp.Iterable[np.ndarray]
ToContainerType = tp.Callable[[tp.Iterator[np.ndarray]], TContainer]

#https://docs.python.org/3/library/datetime.html

class InterfaceDatetime(Interface[TContainer]):

    __slots__ = (
            '_blocks', # function that returns iterable of arrays
            '_blocks_to_container', # partialed function that will return a new container
            )
    INTERFACE = (
            'year',
            'month',
            'day',
            'weekday',
            'timetuple',
            'fromisoformat',
            'isoformat',
            'strftime',
            'strptime',
            'strpdate',
            )

    DT64_EXCLUDE_YEAR = (DT64_YEAR,)
    DT64_EXCLUDE_YEAR_MONTH = (DT64_YEAR, DT64_MONTH)
    DT64_TIME = frozenset((
            DT64_H,
            DT64_M,
            DT64_S,
            DT64_MS,
            DT64_US,
            DT64_NS,
            DT64_PS,
            DT64_FS,
            DT64_AS,
            ))

    DT64_EXCLUDE_YEAR_MONTH_SUB_MICRO = frozenset((
            DT64_YEAR,
            DT64_MONTH,
            DT64_NS,
            DT64_PS,
            DT64_FS,
            DT64_AS,
            ))

    def __init__(self,
            blocks: BlocksType,
            blocks_to_container: ToContainerType[TContainer]
            ) -> None:
        self._blocks: BlocksType = blocks
        self._blocks_to_container: ToContainerType[TContainer] = blocks_to_container

    @staticmethod
    def _validate_dtype_non_str(
            dtype: np.dtype,
            exclude: tp.Iterable[np.dtype] = EMPTY_TUPLE,
            ) -> None:
        '''
        Only support dtypes that are (or contain) datetime64 types. This is because most conversions from string can be done simply with astype().
        '''
        if ((dtype.kind == DTYPE_DATETIME_KIND
                or dtype == DTYPE_OBJECT)
                and dtype not in exclude
                ):
            return
        raise RuntimeError(f'invalid dtype ({dtype}) for date operation')


    @staticmethod
    def _validate_dtype_str(
            dtype: np.dtype,
            exclude: tp.Iterable[np.dtype] = EMPTY_TUPLE,
            ) -> None:
        '''
        Only support dtypes that are (or contain) strings.
        '''
        if ((dtype.kind in DTYPE_STR_KINDS
                or dtype == DTYPE_OBJECT)
                and dtype not in exclude
                ):
            return
        raise RuntimeError(f'invalid dtype ({dtype}) for operation on string types')

    #---------------------------------------------------------------------------

    @property
    def year(self) -> TContainer:
        'Return the year of each element.'

        def blocks() -> tp.Iterator[np.ndarray]:
            for block in self._blocks:
                self._validate_dtype_non_str(block.dtype)

                if block.dtype.kind == DTYPE_DATETIME_KIND:
                    array = block.astype(DT64_YEAR).astype(DTYPE_INT_DEFAULT) + 1970
                    array.flags.writeable = False
                else: # must be object type
                    array = array_from_element_attr(
                            array=block,
                            attr_name='year',
                            dtype=DTYPE_INT_DEFAULT)
                yield array

        return self._blocks_to_container(blocks())

    @property
    def month(self) -> TContainer:
        '''
        Return the month of each element, between 1 and 12 inclusive.
        '''

        def blocks() -> tp.Iterator[np.ndarray]:
            for block in self._blocks:
                self._validate_dtype_non_str(block.dtype, exclude=self.DT64_EXCLUDE_YEAR)

                if block.dtype.kind == DTYPE_DATETIME_KIND:
                    array = block.astype(DT64_MONTH).astype(DTYPE_INT_DEFAULT) % 12 + 1
                    array.flags.writeable = False
                else: # must be object type
                    array = array_from_element_attr(
                            array=block,
                            attr_name='month',
                            dtype=DTYPE_INT_DEFAULT)
                yield array

        return self._blocks_to_container(blocks())

    @property
    def day(self) -> TContainer:
        '''
        Return the day of each element, between 1 and the number of days in the given month of the given year.
        '''

        def blocks() -> tp.Iterator[np.ndarray]:
            for block in self._blocks:
                self._validate_dtype_non_str(block.dtype, exclude=self.DT64_EXCLUDE_YEAR_MONTH)

                if block.dtype.kind == DTYPE_DATETIME_KIND:
                    if block.dtype != DT64_DAY:
                        block = block.astype(DT64_DAY)
                    # subtract the first of the month, then shfit
                    array = (block - block.astype(DT64_MONTH)).astype(DTYPE_INT_DEFAULT) + 1
                    array.flags.writeable = False
                else: # must be object type
                    array = array_from_element_attr(
                            array=block,
                            attr_name='day',
                            dtype=DTYPE_INT_DEFAULT)

                yield array

        return self._blocks_to_container(blocks())



    #---------------------------------------------------------------------------

    # replace: akward to implement, as cannot provide None for the parameters that you do not want to set

    def weekday(self) -> TContainer:
        '''
        Return the day of the week as an integer, where Monday is 0 and Sunday is 6.
        '''

        def blocks() -> tp.Iterator[np.ndarray]:
            for block in self._blocks:
                self._validate_dtype_non_str(block.dtype, exclude=self.DT64_EXCLUDE_YEAR_MONTH)

                if block.dtype.kind == DTYPE_DATETIME_KIND:
                    if block.dtype != DT64_DAY: # go to day first, then object
                        block = block.astype(DT64_DAY)
                    block = block.astype(DTYPE_OBJECT)
                # all object arrays by this point

                # returns an immutable array
                array = array_from_element_method(
                        array=block,
                        method_name='weekday',
                        args=EMPTY_TUPLE,
                        dtype=DTYPE_INT_DEFAULT
                        )
                yield array

        return self._blocks_to_container(blocks())


    #---------------------------------------------------------------------------
    # time methods

    def timetuple(self) -> TContainer:
        '''
        Return a ``time.struct_time`` such as returned by time.localtime().
        '''

        def blocks() -> tp.Iterator[np.ndarray]:
            for block in self._blocks:

                # NOTE: nanosecond and lower will return integers; should exclude
                self._validate_dtype_non_str(block.dtype,
                        exclude=self.DT64_EXCLUDE_YEAR_MONTH_SUB_MICRO)

                if block.dtype.kind == DTYPE_DATETIME_KIND:
                    block = block.astype(DTYPE_OBJECT)
                # all object arrays by this point

                # returns an immutable array
                array = array_from_element_method(
                        array=block,
                        method_name='timetuple',
                        args=EMPTY_TUPLE,
                        dtype=DTYPE_OBJECT
                        )
                yield array

        return self._blocks_to_container(blocks())

    def isoformat(self, sep: str = 'T', timespec: str = 'auto') -> TContainer:
        '''
        Return a string representing the date in ISO 8601 format, YYYY-MM-DD.
        '''

        def blocks() -> tp.Iterator[np.ndarray]:
            for block in self._blocks:

                self._validate_dtype_non_str(block.dtype,
                        exclude=self.DT64_EXCLUDE_YEAR_MONTH_SUB_MICRO)

                args = EMPTY_TUPLE
                if block.dtype.kind == DTYPE_DATETIME_KIND:
                    if block.dtype in self.DT64_TIME:
                        # if we know this is a time type, we can pass args
                        args = (sep, timespec) #type: ignore
                    block = block.astype(DTYPE_OBJECT)

                # all object arrays by this point
                # NOTE: we cannot determine if an Object array has date or datetime objects with a full iteration, so we cannot be sure if we need to pass args or not.

                # returns an immutable array
                array = array_from_element_method(
                        array=block,
                        method_name='isoformat',
                        args=args,
                        dtype=DTYPE_STR,
                        )
                yield array

        return self._blocks_to_container(blocks())

    def fromisoformat(self) -> TContainer:
        '''
        Return a :obj:`datetime.date` object from an ISO 8601 format.
        '''

        def blocks() -> tp.Iterator[np.ndarray]:
            for block in self._blocks:
                # permit only string types, or objects types that contain strings
                self._validate_dtype_str(block.dtype)

                # NOTE: might use fromisoformat on date/datetime objects directly; assumed to be faster to go through datetime64 objects, fromisoformat is only available on python 3.7

                array_dt64 = block.astype(np.datetime64)
                if array_dt64.dtype in self.DT64_EXCLUDE_YEAR_MONTH_SUB_MICRO:
                    raise RuntimeError(f'invalid derived dtype ({array_dt64.dtype}) for iso format')
                array = array_dt64.astype(object)
                array.flags.writeable = False
                yield array

        return self._blocks_to_container(blocks())

    def strftime(self, format: str) -> TContainer:
        '''
        Return a string representing the date, controlled by an explicit ``format`` string.
        '''

        def blocks() -> tp.Iterator[np.ndarray]:
            for block in self._blocks:

                # NOTE: nanosecond and lower will return integers; should exclud
                self._validate_dtype_non_str(block.dtype,
                        exclude=self.DT64_EXCLUDE_YEAR_MONTH_SUB_MICRO)

                if block.dtype.kind == DTYPE_DATETIME_KIND:
                    block = block.astype(DTYPE_OBJECT)
                # all object arrays by this point

                # returns an immutable array
                array = array_from_element_method(
                        array=block,
                        method_name='strftime',
                        args=(format,),
                        dtype=DTYPE_STR,
                        )
                yield array

        return self._blocks_to_container(blocks())

    def strptime(self, format: str) -> TContainer:
        '''
        Return a Python datetime object from parsing a string defined with ``format``.
        '''
        def func(s: str) -> datetime:
            return datetime.strptime(s, format)

        def blocks() -> tp.Iterator[np.ndarray]:
            for block in self._blocks:
                # permit only string types, or objects types that contain strings
                self._validate_dtype_str(block.dtype)
                # returns an immutable array
                array = array_from_element_apply(
                        array=block,
                        func=func,
                        dtype=DTYPE_OBJECT,
                        )
                yield array

        return self._blocks_to_container(blocks())


    def strpdate(self, format: str) -> TContainer:
        '''
        Return a Python date object from parsing a string defined with ``format``.
        '''
        def func(s: str) -> date:
            return datetime.strptime(s, format).date()

        def blocks() -> tp.Iterator[np.ndarray]:
            for block in self._blocks:
                # permit only string types, or objects types that contain strings
                self._validate_dtype_str(block.dtype)
                # returns an immutable array
                array = array_from_element_apply(
                        array=block,
                        func=func,
                        dtype=DTYPE_OBJECT,
                        )
                yield array

        return self._blocks_to_container(blocks())





















