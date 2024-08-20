from __future__ import annotations

from datetime import date
from datetime import datetime

import numpy as np
import typing_extensions as tp
from arraykit import isna_element
from arraykit import resolve_dtype

from static_frame.core.node_selector import Interface
from static_frame.core.node_selector import InterfaceBatch
from static_frame.core.node_selector import TVContainer_co
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
from static_frame.core.util import DTYPE_STR_KINDS
from static_frame.core.util import DTYPE_YEAR_MONTH_STR
from static_frame.core.util import DTYPE_YEAR_QUARTER_STR
from static_frame.core.util import FILL_VALUE_DEFAULT
from static_frame.core.util import TCallableAny
from static_frame.core.util import TDtypeAny
from static_frame.core.util import TNDArrayAny
from static_frame.core.util import TNDArrayIntDefault
from static_frame.core.util import array_from_element_apply
from static_frame.core.util import array_from_element_attr
from static_frame.core.util import array_from_element_method
from static_frame.core.util import dtype_from_element
from static_frame.core.util import isna_array

if tp.TYPE_CHECKING:
    from static_frame.core.batch import Batch  # pragma: no cover
    from static_frame.core.frame import Frame  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.index import Index  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchy  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.series import Series  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.type_blocks import TypeBlocks  # pylint: disable=W0611 #pragma: no cover

    BlocksType = tp.Iterable[TNDArrayAny] #pragma: no cover
    ToContainerType = tp.Callable[[tp.Iterator[TNDArrayAny]], TVContainer_co] #pragma: no cover

INTERFACE_DT = (
        '__call__',
        'year',
        'year_month',
        'year_quarter',
        'month',
        'day',
        'hour',
        'minute',
        'second',
        'weekday',
        'quarter',
        'is_month_end',
        'is_month_start',
        'is_year_end',
        'is_year_start',
        'is_quarter_end',
        'is_quarter_start',
        'timetuple',
        'isoformat',
        'fromisoformat',
        'strftime',
        'strptime',
        'strpdate',
        )

class InterfaceDatetime(Interface, tp.Generic[TVContainer_co]):

    __slots__ = (
            '_blocks', # function that returns iterable of arrays
            '_blocks_to_container', # partialed function that will return a new container
            '_fill_value',
            '_fill_value_dtype',
            )
    _INTERFACE = INTERFACE_DT

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
            *,
            blocks: BlocksType,
            blocks_to_container: ToContainerType[TVContainer_co], #type: ignore[type-var]
            fill_value: tp.Any = FILL_VALUE_DEFAULT,
            ) -> None:
        self._blocks: BlocksType = blocks
        self._blocks_to_container: ToContainerType[TVContainer_co] = blocks_to_container
        self._fill_value: tp.Any = fill_value

        # only set attr if we will need to use the value
        if fill_value is not FILL_VALUE_DEFAULT:
            self._fill_value_dtype = dtype_from_element(fill_value)

        # self._fill_value_dtype: tp.Optional[TDtypeAny] = (None
        #         if fill_value is FILL_VALUE_DEFAULT
        #         else dtype_from_element(fill_value))

    def __call__(self,
            *,
            fill_value: tp.Any,
            ) -> 'InterfaceDatetime[TVContainer_co]':
        '''
        Args:
            fill_value: If NAT are encountered, use this value.
        '''
        return self.__class__(
                blocks=self._blocks,
                blocks_to_container=self._blocks_to_container,
                fill_value=fill_value,
                )

    @staticmethod
    def _validate_dtype_non_str(
            dtype: TDtypeAny,
            exclude: tp.Iterable[TDtypeAny] = (),
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
            dtype: TDtypeAny,
            exclude: tp.Iterable[TDtypeAny] = (),
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

    def _fill_missing_dt64(self,
            array_src: TNDArrayAny,
            array_dst: TNDArrayAny,
            ) -> TNDArrayAny:
        '''
        Args:
            array_src: The raw array, before any dytpe conversions; used to identify missing values.
            array_dst: The array post any conversions, to be filled with missing values.
        '''
        targets = isna_array(array_src)
        if targets.any():
            if self._fill_value is FILL_VALUE_DEFAULT:
                raise RuntimeError('Cannot convert NaT: provide a `fill_value` to `via_dt()`.')
            else:
                dt = resolve_dtype(array_dst.dtype, self._fill_value_dtype)
                if dt != array_dst.dtype:
                    array_dst = array_dst.astype(dt)
                array_dst[targets] = self._fill_value

        array_dst.flags.writeable = False
        return array_dst

    def _fill_missing_element_method(self,
            array: TNDArrayAny,
            *,
            method_name: str,
            args: tp.Tuple[tp.Any, ...],
            dtype: TDtypeAny,
            ) -> TNDArrayAny:
        if self._fill_value is FILL_VALUE_DEFAULT:
            if isna_array(array).any():
                raise RuntimeError('Cannot convert NaT: provide a `fill_value` to `via_dt()`.')

            array = array_from_element_method(
                    array=array,
                    method_name=method_name,
                    args=args,
                    dtype=dtype,
                    )
        else:
            dt = resolve_dtype(dtype, self._fill_value_dtype)
            if dtype.itemsize == 0 and dt.kind == dtype.kind:
                dt = dtype # set to unsized

            def func(e: tp.Any) -> tp.Any:
                if isna_element(e):
                    return self._fill_value
                return getattr(e, method_name)(*args)

            array = array_from_element_apply(
                    array=array,
                    func=func,
                    dtype=dt,
                    )
        return array

    def _fill_missing_element_attr(self,
            array: TNDArrayAny,
            *,
            attr_name: str,
            dtype: TDtypeAny,
            ) -> TNDArrayAny:
        if self._fill_value is FILL_VALUE_DEFAULT:
            if isna_array(array).any():
                raise RuntimeError('Cannot convert NaT: provide a `fill_value` to `via_dt()`.')

            array = array_from_element_attr(
                    array=array,
                    attr_name=attr_name,
                    dtype=dtype,
                    )
        else:
            dt = resolve_dtype(dtype, self._fill_value_dtype)
            assert dtype.itemsize != 0 # expect numeric scalar

            def func(e: tp.Any) -> tp.Any:
                if isna_element(e):
                    return self._fill_value
                return getattr(e, attr_name)

            array = array_from_element_apply(
                    array=array,
                    func=func,
                    dtype=dt,
                    )
        return array

    @staticmethod
    def _array_to_quarter_int(
            block: TNDArrayAny,
            ) -> TNDArrayIntDefault:
        # astype object dtypes to month
        if block.dtype != DT64_MONTH:
            b = block.astype(DT64_MONTH)
        else:
            b = block
        # months will start from 0
        bint = b.astype(DTYPE_INT_DEFAULT) % 12
        array = np.empty(block.shape, dtype=DTYPE_INT_DEFAULT)
        np.floor_divide(bint, 3, out=array)
        return array + 1

    #---------------------------------------------------------------------------
    # date, datetime attributes

    @property
    def year(self) -> TVContainer_co:
        'Return the year of each element.'

        def blocks() -> tp.Iterator[TNDArrayAny]:
            for block in self._blocks:
                self._validate_dtype_non_str(block.dtype)

                if block.dtype.kind == DTYPE_DATETIME_KIND:
                    array = block.astype(DT64_YEAR).astype(DTYPE_INT_DEFAULT) + 1970
                    array = self._fill_missing_dt64(block, array)
                else: # must be object type
                    array = self._fill_missing_element_attr(
                            array=block,
                            attr_name='year',
                            dtype=DTYPE_INT_DEFAULT)
                yield array

        return self._blocks_to_container(blocks())

    @property
    def month(self) -> TVContainer_co:
        '''
        Return the month of each element, between 1 and 12 inclusive.
        '''

        def blocks() -> tp.Iterator[TNDArrayAny]:
            for block in self._blocks:
                self._validate_dtype_non_str(block.dtype, exclude=self.DT64_EXCLUDE_YEAR)

                if block.dtype.kind == DTYPE_DATETIME_KIND:
                    array = block.astype(DT64_MONTH).astype(DTYPE_INT_DEFAULT) % 12 + 1
                    array = self._fill_missing_dt64(block, array)
                else: # must be object type
                    array = self._fill_missing_element_attr(
                            array=block,
                            attr_name='month',
                            dtype=DTYPE_INT_DEFAULT)
                yield array

        return self._blocks_to_container(blocks())

    @property
    def year_month(self) -> TVContainer_co:
        '''
        Return the year and month of each element as string formatted YYYY-MM.
        '''

        def blocks() -> tp.Iterator[TNDArrayAny]:
            for block in self._blocks:
                self._validate_dtype_non_str(block.dtype, exclude=self.DT64_EXCLUDE_YEAR)

                if block.dtype.kind == DTYPE_DATETIME_KIND:
                    array = block.astype(DT64_MONTH).astype(DTYPE_YEAR_MONTH_STR)
                    array = self._fill_missing_dt64(block, array)
                else:
                    array = self._fill_missing_element_method(block,
                            method_name='strftime',
                            args=('%Y-%m',),
                            dtype=DTYPE_YEAR_MONTH_STR,
                            )
                yield array

        return self._blocks_to_container(blocks())

    @property
    def year_quarter(self) -> TVContainer_co:
        '''
        Return the year and quarter of each element as a string formatted YYYY-QQ.
        '''
        def blocks() -> tp.Iterator[TNDArrayAny]:
            for block in self._blocks:
                self._validate_dtype_non_str(block.dtype, exclude=self.DT64_EXCLUDE_YEAR)
                array_year = block.astype(DT64_YEAR)
                array_quarter = self._array_to_quarter_int(block)

                # get full size and flat iter, then reshape if necessary
                array = np.empty(block.size, dtype=DTYPE_YEAR_QUARTER_STR)
                for i, (y, q) in enumerate(zip(array_year.flat, array_quarter.flat)):
                    array[i] = f'{y}-Q{q}'
                if block.ndim == 2:
                    array = array.reshape(block.shape)
                yield self._fill_missing_dt64(block, array)

        return self._blocks_to_container(blocks())

    @property
    def day(self) -> TVContainer_co:
        '''
        Return the day of each element, between 1 and the number of days in the given month of the given year.
        '''

        def blocks() -> tp.Iterator[TNDArrayAny]:
            for block in self._blocks:
                self._validate_dtype_non_str(block.dtype, exclude=self.DT64_EXCLUDE_YEAR_MONTH)

                if block.dtype.kind == DTYPE_DATETIME_KIND:
                    if block.dtype != DT64_DAY:
                        block = block.astype(DT64_DAY)
                    # subtract the first of the month, then shift
                    array = (block - block.astype(DT64_MONTH)).astype(DTYPE_INT_DEFAULT) + 1 # type: ignore
                    array = self._fill_missing_dt64(block, array)
                else: # must be object type
                    array = self._fill_missing_element_attr(
                            array=block,
                            attr_name='day',
                            dtype=DTYPE_INT_DEFAULT)
                yield array

        return self._blocks_to_container(blocks())

    #---------------------------------------------------------------------------
    # datetime attributes

    @property
    def hour(self) -> TVContainer_co:
        '''
        Return the hour of each element, between 0 and 24.
        '''
        def blocks() -> tp.Iterator[TNDArrayAny]:
            for block in self._blocks:
                # permit all dt64 types
                self._validate_dtype_non_str(block.dtype)

                if block.dtype.kind == DTYPE_DATETIME_KIND:
                    if block.dtype != DT64_H:
                        block = block.astype(DT64_H)
                    # subtract the first of the month, then shfit
                    array = block.astype(DTYPE_INT_DEFAULT) % 24
                    array = self._fill_missing_dt64(block, array)
                else: # must be object datetime type
                    array = self._fill_missing_element_attr(
                            array=block,
                            attr_name='hour',
                            dtype=DTYPE_INT_DEFAULT)
                yield array

        return self._blocks_to_container(blocks())

    @property
    def minute(self) -> TVContainer_co:
        '''
        Return the minute of each element, between 0 and 60.
        '''
        def blocks() -> tp.Iterator[TNDArrayAny]:
            for block in self._blocks:
                # permit all dt64 types
                self._validate_dtype_non_str(block.dtype)

                if block.dtype.kind == DTYPE_DATETIME_KIND:
                    if block.dtype != DT64_M:
                        block = block.astype(DT64_M)
                    array = block.astype(DTYPE_INT_DEFAULT) % 60
                    array = self._fill_missing_dt64(block, array)
                else: # must be object datetime type
                    array = self._fill_missing_element_attr(
                            array=block,
                            attr_name='minute',
                            dtype=DTYPE_INT_DEFAULT)
                yield array

        return self._blocks_to_container(blocks())

    @property
    def second(self) -> TVContainer_co:
        '''
        Return the second of each element, between 0 and 60.
        '''
        def blocks() -> tp.Iterator[TNDArrayAny]:
            for block in self._blocks:
                # permit all dt64 types
                self._validate_dtype_non_str(block.dtype)

                if block.dtype.kind == DTYPE_DATETIME_KIND:
                    if block.dtype != DT64_S:
                        block = block.astype(DT64_S)
                    # subtract the first of the month, then shfit
                    array = block.astype(DTYPE_INT_DEFAULT) % 60
                    array = self._fill_missing_dt64(block, array)
                else: # must be object datetime type
                    array = self._fill_missing_element_attr(
                            array=block,
                            attr_name='second',
                            dtype=DTYPE_INT_DEFAULT)
                yield array

        return self._blocks_to_container(blocks())


    #---------------------------------------------------------------------------

    # replace: awkward to implement, as cannot provide None for the parameters that you do not want to set

    def weekday(self) -> TVContainer_co:
        '''
        Return the day of the week as an integer, where Monday is 0 and Sunday is 6.
        '''
        def blocks() -> tp.Iterator[TNDArrayAny]:
            for block in self._blocks:
                self._validate_dtype_non_str(block.dtype, exclude=self.DT64_EXCLUDE_YEAR_MONTH)

                if block.dtype.kind == DTYPE_DATETIME_KIND:
                    if block.dtype != DT64_DAY: # go to day first, then object
                        block = block.astype(DT64_DAY)
                    # shift to set first Monday, then modulo
                    array = (block.astype(DTYPE_INT_DEFAULT) + 3) % 7
                    array = self._fill_missing_dt64(block, array)
                else:
                    # NOTE: might be faster to convert to datetime64 then do shift / modulo
                    array = self._fill_missing_element_method(
                            array=block,
                            method_name='weekday',
                            args=(),
                            dtype=DTYPE_INT_DEFAULT
                            )
                yield array

        return self._blocks_to_container(blocks())

    def quarter(self) -> TVContainer_co:
        '''
        Return the quarter of the year as an integer, where January through March is quarter 1.
        '''
        def blocks() -> tp.Iterator[TNDArrayAny]:
            for block in self._blocks:
                self._validate_dtype_non_str(block.dtype)
                array = self._array_to_quarter_int(block)
                yield self._fill_missing_dt64(block, array)

        return self._blocks_to_container(blocks())

    #---------------------------------------------------------------------------
    # boolean matches

    def is_month_end(self) -> TVContainer_co:
        '''Return Boolean indicators if the day is the month end.
        '''
        def blocks() -> tp.Iterator[TNDArrayAny]:
            for block in self._blocks:
                self._validate_dtype_non_str(block.dtype, exclude=self.DT64_EXCLUDE_YEAR_MONTH)

                # astype object dtypes to day too
                if block.dtype != DT64_DAY:
                    b = block.astype(DT64_DAY)
                else:
                    b = block
                # convert to month, shift to next, convert to day, slide back to eom
                array = b == ((b.astype(DT64_MONTH) + 1).astype(DT64_DAY) - 1)
                array = self._fill_missing_dt64(block, array)
                yield array

        return self._blocks_to_container(blocks())

    def is_month_start(self) -> TVContainer_co:
        '''Return Boolean indicators if the day is the month start.
        '''
        def blocks() -> tp.Iterator[TNDArrayAny]:
            for block in self._blocks:
                self._validate_dtype_non_str(block.dtype, exclude=self.DT64_EXCLUDE_YEAR_MONTH)

                # astype object dtypes to day too
                if block.dtype != DT64_DAY:
                    b = block.astype(DT64_DAY)
                else:
                    b = block
                array = b == b.astype(DT64_MONTH).astype(DT64_DAY)
                array = self._fill_missing_dt64(block, array)
                yield array

        return self._blocks_to_container(blocks())


    def is_year_end(self) -> TVContainer_co:
        '''Return Boolean indicators if the day is the year end.
        '''
        def blocks() -> tp.Iterator[TNDArrayAny]:
            for block in self._blocks:
                self._validate_dtype_non_str(block.dtype, exclude=self.DT64_EXCLUDE_YEAR_MONTH)

                # astype object dtypes to day too
                if block.dtype != DT64_DAY:
                    b = block.astype(DT64_DAY)
                else:
                    b = block
                # convert to year, shift to next, convert to day, slide back to eoy
                array = b == ((b.astype(DT64_YEAR) + 1).astype(DT64_DAY) - 1)
                array = self._fill_missing_dt64(block, array)
                yield array

        return self._blocks_to_container(blocks())

    def is_year_start(self) -> TVContainer_co:
        '''Return Boolean indicators if the day is the year start.
        '''
        def blocks() -> tp.Iterator[TNDArrayAny]:
            for block in self._blocks:
                self._validate_dtype_non_str(block.dtype, exclude=self.DT64_EXCLUDE_YEAR_MONTH)

                # astype object dtypes to day too
                if block.dtype != DT64_DAY:
                    b = block.astype(DT64_DAY)
                else:
                    b = block
                # convert to month, shift to next, convert to day, slide back to eom
                array = b == b.astype(DT64_YEAR).astype(DT64_DAY)
                array = self._fill_missing_dt64(block, array)
                yield array

        return self._blocks_to_container(blocks())


    def is_quarter_end(self) -> TVContainer_co:
        '''Return Boolean indicators if the day is the quarter end.
        '''
        def blocks() -> tp.Iterator[TNDArrayAny]:
            for block in self._blocks:
                self._validate_dtype_non_str(block.dtype, exclude=self.DT64_EXCLUDE_YEAR_MONTH)

                # astype object dtypes to day too
                if block.dtype != DT64_DAY:
                    b = block.astype(DT64_DAY)
                else:
                    b = block

                # convert to month, shift to next, convert to day, slide back to eom
                month = b.astype(DT64_MONTH)
                eom = (month + 1).astype(DT64_DAY) - 1
                # months starting from 0
                month_int = month.astype(DTYPE_INT_DEFAULT) % 12
                month_valid = ((month_int == 2)
                        | (month_int == 5)
                        | (month_int == 8)
                        | (month_int == 11)
                        )
                array = (b == eom) & month_valid
                array = self._fill_missing_dt64(block, array)
                yield array

        return self._blocks_to_container(blocks())

    def is_quarter_start(self) -> TVContainer_co:
        '''Return Boolean indicators if the day is the quarter start.
        '''
        def blocks() -> tp.Iterator[TNDArrayAny]:
            for block in self._blocks:
                self._validate_dtype_non_str(block.dtype, exclude=self.DT64_EXCLUDE_YEAR_MONTH)

                # astype object dtypes to day too
                if block.dtype != DT64_DAY:
                    b = block.astype(DT64_DAY)
                else:
                    b = block

                # convert to month, shift to next, convert to day, slide back to eom
                month = b.astype(DT64_MONTH)
                som = month.astype(DT64_DAY)
                # months starting from 0
                month_int = month.astype(DTYPE_INT_DEFAULT) % 12
                month_valid = ((month_int == 0)
                        | (month_int == 3)
                        | (month_int == 6)
                        | (month_int == 9)
                        )
                array = (b == som) & month_valid
                array = self._fill_missing_dt64(block, array)
                yield array

        return self._blocks_to_container(blocks())

    #---------------------------------------------------------------------------
    # time methods

    def timetuple(self) -> TVContainer_co:
        '''
        Return a ``time.struct_time`` such as returned by time.localtime().
        '''
        def blocks() -> tp.Iterator[TNDArrayAny]:
            for block in self._blocks:

                # NOTE: nanosecond and lower will return integers; should exclude
                self._validate_dtype_non_str(block.dtype,
                        exclude=self.DT64_EXCLUDE_YEAR_MONTH_SUB_MICRO)

                if block.dtype.kind == DTYPE_DATETIME_KIND:
                    block = block.astype(DTYPE_OBJECT)
                # all object arrays by this point
                array = self._fill_missing_element_method(
                        array=block,
                        method_name='timetuple',
                        args=(),
                        dtype=DTYPE_OBJECT
                        )
                yield array

        return self._blocks_to_container(blocks())

    def isoformat(self, sep: str = 'T', timespec: str = 'auto') -> TVContainer_co:
        '''
        Return a string representing the date in ISO 8601 format, YYYY-MM-DD.
        '''

        def blocks() -> tp.Iterator[TNDArrayAny]:
            for block in self._blocks:

                self._validate_dtype_non_str(block.dtype,
                        exclude=self.DT64_EXCLUDE_YEAR_MONTH_SUB_MICRO)

                args = ()
                if block.dtype.kind == DTYPE_DATETIME_KIND:
                    if block.dtype in self.DT64_TIME:
                        # if we know this is a time type, we can pass args
                        args = (sep, timespec) #type: ignore
                    block = block.astype(DTYPE_OBJECT)

                # all object arrays by this point
                # NOTE: we cannot determine if an Object array has date or datetime objects with a full iteration, so we cannot be sure if we need to pass args or not.
                array = self._fill_missing_element_method(
                        array=block,
                        method_name='isoformat',
                        args=args,
                        dtype=DTYPE_STR,
                        )
                yield array

        return self._blocks_to_container(blocks())

    def fromisoformat(self) -> TVContainer_co:
        '''
        Return a :obj:`datetime.date` object from an ISO 8601 format.
        '''

        def blocks() -> tp.Iterator[TNDArrayAny]:
            for block in self._blocks:
                # permit only string types, or objects types that contain strings
                self._validate_dtype_str(block.dtype)

                # NOTE: might use fromisoformat on date/datetime objects directly; assumed to be faster to go through datetime64 objects, fromisoformat is only available on python 3.7

                array_dt64 = block.astype(np.datetime64)
                if array_dt64.dtype in self.DT64_EXCLUDE_YEAR_MONTH_SUB_MICRO:
                    raise RuntimeError(f'invalid derived dtype ({array_dt64.dtype}) for iso format')
                array = array_dt64.astype(object)
                array = self._fill_missing_dt64(block, array)
                yield array

        return self._blocks_to_container(blocks())

    def strftime(self, format: str) -> TVContainer_co:
        '''
        Return a string representing the date, controlled by an explicit ``format`` string.
        '''

        def blocks() -> tp.Iterator[TNDArrayAny]:
            for block in self._blocks:

                # NOTE: nanosecond and lower will return integers; should exclud
                self._validate_dtype_non_str(block.dtype,
                        exclude=self.DT64_EXCLUDE_YEAR_MONTH_SUB_MICRO)

                if block.dtype.kind == DTYPE_DATETIME_KIND:
                    block = block.astype(DTYPE_OBJECT)
                # all object arrays by this point

                # returns an immutable array
                array = self._fill_missing_element_method(
                        array=block,
                        method_name='strftime',
                        args=(format,),
                        dtype=DTYPE_STR,
                        )
                yield array

        return self._blocks_to_container(blocks())

    def strptime(self, format: str) -> TVContainer_co:
        '''
        Return a Python datetime object from parsing a string defined with ``format``.
        '''
        def func(s: str) -> datetime:
            return datetime.strptime(s, format)

        def blocks() -> tp.Iterator[TNDArrayAny]:
            for block in self._blocks:
                # permit only string types, or objects types that contain strings
                # NOTE: no missing handling necessary
                self._validate_dtype_str(block.dtype)
                # returns an immutable array
                array = array_from_element_apply(
                        array=block,
                        func=func,
                        dtype=DTYPE_OBJECT,
                        )
                yield array

        return self._blocks_to_container(blocks())


    def strpdate(self, format: str) -> TVContainer_co:
        '''
        Return a Python date object from parsing a string defined with ``format``.
        '''
        def func(s: str) -> date:
            return datetime.strptime(s, format).date()

        def blocks() -> tp.Iterator[TNDArrayAny]:
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

#-------------------------------------------------------------------------------

class InterfaceBatchDatetime(InterfaceBatch):
    '''Alternate datetime interface specialized for the :obj:`Batch`.
    '''
    __slots__ = (
            '_batch_apply',
            '_fill_value',
            )
    _INTERFACE = INTERFACE_DT

    def __init__(self,
            batch_apply: tp.Callable[[TCallableAny], 'Batch'],
            *,
            fill_value: tp.Any = FILL_VALUE_DEFAULT,
            ) -> None:
        self._batch_apply = batch_apply
        self._fill_value = fill_value


    def __call__(self,
            *,
            fill_value: tp.Any,
            ) -> 'InterfaceBatchDatetime':
        '''
        Args:
            fill_value: If NAT are encountered, use this value.
        '''
        return self.__class__(self._batch_apply,
                fill_value=fill_value
                )

    #---------------------------------------------------------------------------
    # date, datetime attributes

    @property
    def year(self) -> 'Batch':
        'Return the year of each element.'
        return self._batch_apply(lambda c: c.via_dt(fill_value=self._fill_value).year)

    @property
    def month(self) -> 'Batch':
        '''
        Return the month of each element, between 1 and 12 inclusive.
        '''
        return self._batch_apply(lambda c: c.via_dt(fill_value=self._fill_value).month)

    @property
    def year_month(self) -> 'Batch':
        '''
        Return the year and month of each element as string formatted YYYY-MM.
        '''
        return self._batch_apply(lambda c: c.via_dt(fill_value=self._fill_value).year_month)

    @property
    def year_quarter(self) -> 'Batch':
        '''
        Return the year and quarter of each element as string formatted YYYY-QQ.
        '''
        return self._batch_apply(lambda c: c.via_dt(fill_value=self._fill_value).year_quarter)

    @property
    def day(self) -> 'Batch':
        '''
        Return the day of each element, between 1 and the number of days in the given month of the given year.
        '''
        return self._batch_apply(lambda c: c.via_dt(fill_value=self._fill_value).day)

    #---------------------------------------------------------------------------
    # datetime attributes

    @property
    def hour(self) -> 'Batch':
        '''
        Return the hour of each element, between 0 and 24.
        '''
        return self._batch_apply(lambda c: c.via_dt(fill_value=self._fill_value).hour)

    @property
    def minute(self) -> 'Batch':
        '''
        Return the minute of each element, between 0 and 60.
        '''
        return self._batch_apply(lambda c: c.via_dt(fill_value=self._fill_value).minute)

    @property
    def second(self) -> 'Batch':
        '''
        Return the second of each element, between 0 and 60.
        '''
        return self._batch_apply(lambda c: c.via_dt(fill_value=self._fill_value).second)

    #---------------------------------------------------------------------------

    # replace: akward to implement, as cannot provide None for the parameters that you do not want to set

    def weekday(self) -> 'Batch':
        '''
        Return the day of the week as an integer, where Monday is 0 and Sunday is 6.
        '''
        return self._batch_apply(lambda c: c.via_dt(fill_value=self._fill_value).weekday())

    def quarter(self) -> 'Batch':
        '''
        Return the quarter of the year as an integer, where January through March is quarter 1.
        '''
        return self._batch_apply(lambda c: c.via_dt(fill_value=self._fill_value).quarter())

    #---------------------------------------------------------------------------
    # boolean matches

    def is_month_end(self) -> 'Batch':
        '''Return Boolean indicators if the day is the month end.
        '''
        return self._batch_apply(lambda c: c.via_dt(fill_value=self._fill_value).is_month_end())

    def is_month_start(self) -> 'Batch':
        '''Return Boolean indicators if the day is the month start.
        '''
        return self._batch_apply(lambda c: c.via_dt(fill_value=self._fill_value).is_month_start())

    def is_year_end(self) -> 'Batch':
        '''Return Boolean indicators if the day is the year end.
        '''
        return self._batch_apply(lambda c: c.via_dt(fill_value=self._fill_value).is_year_end())

    def is_year_start(self) -> 'Batch':
        '''Return Boolean indicators if the day is the year start.
        '''
        return self._batch_apply(lambda c: c.via_dt(fill_value=self._fill_value).is_year_start())

    def is_quarter_end(self) -> 'Batch':
        '''Return Boolean indicators if the day is the quarter end.
        '''
        return self._batch_apply(lambda c: c.via_dt(fill_value=self._fill_value).is_quarter_end())

    def is_quarter_start(self) -> 'Batch':
        '''Return Boolean indicators if the day is the quarter start.
        '''
        return self._batch_apply(lambda c: c.via_dt(fill_value=self._fill_value).is_quarter_start())

    #---------------------------------------------------------------------------
    # time methods

    def timetuple(self) -> 'Batch':
        '''
        Return a ``time.struct_time`` such as returned by time.localtime().
        '''
        return self._batch_apply(lambda c: c.via_dt(fill_value=self._fill_value).timetuple())

    def isoformat(self, sep: str = 'T', timespec: str = 'auto') -> 'Batch':
        '''
        Return a string representing the date in ISO 8601 format, YYYY-MM-DD.
        '''
        return self._batch_apply(lambda c: c.via_dt(fill_value=self._fill_value).isoformat(sep, timespec))

    def fromisoformat(self) -> 'Batch':
        '''
        Return a :obj:`datetime.date` object from an ISO 8601 format.
        '''
        return self._batch_apply(lambda c: c.via_dt(fill_value=self._fill_value).fromisoformat())

    def strftime(self, format: str) -> 'Batch':
        '''
        Return a string representing the date, controlled by an explicit ``format`` string.
        '''
        return self._batch_apply(lambda c: c.via_dt(fill_value=self._fill_value).strftime(format))

    def strptime(self, format: str) -> 'Batch':
        '''
        Return a Python datetime object from parsing a string defined with ``format``.
        '''
        return self._batch_apply(lambda c: c.via_dt(fill_value=self._fill_value).strptime(format))

    def strpdate(self, format: str) -> 'Batch':
        '''
        Return a Python date object from parsing a string defined with ``format``.
        '''
        return self._batch_apply(lambda c: c.via_dt(fill_value=self._fill_value).strpdate(format))
















