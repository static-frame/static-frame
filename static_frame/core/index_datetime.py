import typing as tp
import datetime

import numpy as np

from automap import AutoMap  # pylint: disable = E0611


from static_frame.core.doc_str import doc_inject
from static_frame.core.index import _INDEX_GO_SLOTS
from static_frame.core.index import _INDEX_SLOTS
from static_frame.core.index import _IndexGOMixin
from static_frame.core.index import Index
from static_frame.core.index import IndexGO
from static_frame.core.util import DateInitializer
from static_frame.core.util import DT64_DAY
from static_frame.core.util import DT64_H
from static_frame.core.util import DT64_M
from static_frame.core.util import DT64_MONTH
from static_frame.core.util import DT64_MS
from static_frame.core.util import DT64_NS
from static_frame.core.util import DT64_S
from static_frame.core.util import DT64_US
from static_frame.core.util import DT64_YEAR
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import IndexInitializer
from static_frame.core.util import key_to_datetime_key
from static_frame.core.util import TD64_DAY
from static_frame.core.util import TD64_MONTH
from static_frame.core.util import TD64_YEAR
from static_frame.core.util import to_datetime64
from static_frame.core.util import to_timedelta64
from static_frame.core.util import YearInitializer
from static_frame.core.util import YearMonthInitializer
from static_frame.core.util import NameType
from static_frame.core.util import NAME_DEFAULT

if tp.TYPE_CHECKING:
    import pandas  #pylint: disable = W0611 #pragma: no cover

I = tp.TypeVar('I', bound='IndexDatetime')

#-------------------------------------------------------------------------------
# Specialized index for dates

class IndexDatetime(Index):
    '''
    Derivation of Index to support Datetime operations. Derived classes must define _DTYPE.
    '''

    STATIC = True
    _DTYPE = None # define in derived class
    __slots__ = _INDEX_SLOTS

    @doc_inject(selector='index_date_time_init')
    def __init__(self,
            labels: IndexInitializer,
            *,
            name: NameType = NAME_DEFAULT,
            ):
        '''Initializer.

        {args}
        '''
        # __init__ here leaves out the dtype argument, reducing the signature to arguments relevant for these derived classes
        Index.__init__(self, labels=labels, name=name)

    #---------------------------------------------------------------------------
    # dict like interface

    def __contains__(self, value: tp.Any) -> bool:
        '''Return True if value in the labels. Will only return True for an exact match to the type of dates stored within.
        '''
        return self._map.__contains__(to_datetime64(value)) #type: ignore

    #---------------------------------------------------------------------------
    # operators

    def _ufunc_binary_operator(self, *,
            operator: tp.Callable[..., tp.Any],
            other: object) -> np.ndarray:

        if self._recache:
            self._update_array_cache()

        if operator.__name__ == 'matmul' or operator.__name__ == 'rmatmul':
            raise NotImplementedError('matrix multiplication not supported')

        if isinstance(other, Index):
            other = other.values # operate on labels to labels
        elif isinstance(other, str):
            # do not pass dtype, as want to coerce to this parsed type, not the type of sled
            other = to_datetime64(other)

        if isinstance(other, np.datetime64):
            # convert labels to other's datetime64 type to enable matching on month, year, etc.
            array = operator(self._labels.astype(other.dtype), other)
        elif isinstance(other, datetime.timedelta):
            array = operator(self._labels, to_timedelta64(other))
        else:
            # np.timedelta64 should work fine here
            array = operator(self._labels, other)

        array.flags.writeable = False
        return array

    def _loc_to_iloc(self,  # type: ignore
            key: GetItemKeyType,
            *,
            offset: tp.Optional[int] = None,
            partial_selection: bool = False,
            ) -> GetItemKeyType:
        '''
        Specialized for IndexData indices to convert string data representations into np.datetime64 objects as appropriate.
        '''
        # not passing self.dtype to key_to_datetime_key so as to allow translation to a foreign datetime; slice comparison will be handled by map_slice_args
        return Index._loc_to_iloc(self,
                key=key,
                offset=offset,
                key_transform=key_to_datetime_key,
                partial_selection=partial_selection,
                )

    #---------------------------------------------------------------------------
    def to_pandas(self) -> 'pandas.DatetimeIndex':
        '''Return a Pandas Index.
        '''
        import pandas
        return pandas.DatetimeIndex(self.values.copy(),
                name=self._name)


    #---------------------------------------------------------------------------

    @doc_inject(selector='searchsorted', label_type='iloc (integer)')
    def iloc_searchsorted(self,
            values: tp.Any,
            *,
            side_left: bool = True,
            ) -> tp.Union[tp.Hashable, tp.Iterable[tp.Hashable]]:
        '''
        {doc}

        Args:
            {values}
            {side_left}
        '''
        # permit variable forms of date specification
        return Index.iloc_searchsorted(self, #type: ignore [no-any-return]
                key_to_datetime_key(values),
                side_left=side_left,
                )


#-------------------------------------------------------------------------------
class _IndexDatetimeGOMixin(_IndexGOMixin):

    _DTYPE: tp.Optional[np.dtype]
    _map: tp.Optional[AutoMap]
    __slots__ = () # define in derived class

    def append(self, value: tp.Hashable) -> None:
        '''Specialize for fixed-typed indices: convert `value` argument; do not need to resolve_dtype with each addition; self._map is never None
        '''
        value = to_datetime64(value, self._DTYPE)
        if self._map is not None:
            try:
                self._map.add(value)
            except ValueError:
                raise KeyError(f'duplicate key append attempted: {value}')
        self._labels_mutable.append(value)
        self._positions_mutable_count += 1 #pylint: disable=E0237
        self._recache = True #pylint: disable=E0237

#-------------------------------------------------------------------------------
class IndexYear(IndexDatetime):
    '''A mapping of years (NumPy :obj:`datetime64[Y]`) to positions, immutable and of fixed size.
    '''
    STATIC = True
    _DTYPE = DT64_YEAR
    __slots__ = _INDEX_SLOTS

    @classmethod
    def from_date_range(cls: tp.Type[I],
            start: DateInitializer,
            stop: DateInitializer,
            step: int = 1,
            *,
            name: tp.Optional[tp.Hashable] = None) -> I:
        '''
        Get an IndexYearMonth instance over a range of dates, where start and stop are inclusive.
        '''
        labels = np.arange(
                to_datetime64(start, DT64_DAY),
                to_datetime64(stop, DT64_DAY).astype(DT64_YEAR) + TD64_YEAR,
                np.timedelta64(step, 'Y'),
                dtype=DT64_YEAR)
        labels.flags.writeable = False
        return cls(labels, name=name)

    @classmethod
    def from_year_month_range(cls: tp.Type[I],
            start: YearMonthInitializer,
            stop: YearMonthInitializer,
            step: int = 1,
            *,
            name: tp.Optional[tp.Hashable] = None
            ) -> I:
        '''
        Get an IndexYearMonth instance over a range of months, where start and end are inclusive.
        '''

        labels = np.arange(
                to_datetime64(start, DT64_MONTH),
                to_datetime64(stop, DT64_MONTH).astype(DT64_YEAR) + TD64_YEAR,
                np.timedelta64(step, 'Y'),
                dtype=DT64_YEAR)
        labels.flags.writeable = False
        return cls(labels, name=name)


    @classmethod
    def from_year_range(cls: tp.Type[I],
            start: YearInitializer,
            stop: YearInitializer,
            step: int = 1,
            *,
            name: tp.Optional[tp.Hashable] = None
            ) -> I:
        '''
        Get an IndexDate instance over a range of years, where start and end are inclusive.
        '''
        labels = np.arange(
                to_datetime64(start, DT64_YEAR),
                to_datetime64(stop, DT64_YEAR) + TD64_YEAR,
                step=np.timedelta64(step, 'Y'),
                )
        labels.flags.writeable = False
        return cls(labels, name=name)

    #---------------------------------------------------------------------------
    def to_pandas(self) -> None:
        '''Return a Pandas Index.
        '''
        raise NotImplementedError('Pandas does not support a year type, and it is ambiguous if a date proxy should be the first of the year or the last of the year.')


class IndexYearGO(_IndexDatetimeGOMixin, IndexYear):

    _IMMUTABLE_CONSTRUCTOR = IndexYear
    __slots__ = _INDEX_GO_SLOTS

IndexYear._MUTABLE_CONSTRUCTOR = IndexYearGO

#-------------------------------------------------------------------------------
class IndexYearMonth(IndexDatetime):
    '''A mapping of year months (NumPy :obj:`datetime64[M]`) to positions, immutable and of fixed size.
    '''
    STATIC = True
    _DTYPE = DT64_MONTH
    __slots__ = _INDEX_SLOTS

    @classmethod
    def from_date_range(cls: tp.Type[I],
            start: DateInitializer,
            stop: DateInitializer,
            step: int = 1,
            *,
            name: tp.Optional[tp.Hashable] = None
            ) -> I:
        '''
        Get an IndexYearMonth instance over a range of dates, where start and stop is inclusive.
        '''
        labels = np.arange(
                to_datetime64(start, DT64_DAY),
                to_datetime64(stop, DT64_DAY).astype(DT64_MONTH) + TD64_MONTH,
                np.timedelta64(step, 'M'),
                dtype=DT64_MONTH)

        labels.flags.writeable = False
        return cls(labels, name=name)

    @classmethod
    def from_year_month_range(cls: tp.Type[I],
            start: YearMonthInitializer,
            stop: YearMonthInitializer,
            step: int = 1,
            *,
            name: tp.Optional[tp.Hashable] = None
            ) -> I:
        '''
        Get an IndexYearMonth instance over a range of months, where start and end are inclusive.
        '''

        labels = np.arange(
                to_datetime64(start, DT64_MONTH),
                to_datetime64(stop, DT64_MONTH) + TD64_MONTH,
                np.timedelta64(step, 'M'),
                dtype=DT64_MONTH)
        labels.flags.writeable = False
        return cls(labels, name=name)


    @classmethod
    def from_year_range(cls: tp.Type[I],
            start: YearInitializer,
            stop: YearInitializer,
            step: int = 1,
            *,
            name: tp.Optional[tp.Hashable] = None
            ) -> I:
        '''
        Get an IndexYearMonth instance over a range of years, where start and end are inclusive.
        '''
        labels = np.arange(
                to_datetime64(start, DT64_YEAR),
                to_datetime64(stop, DT64_YEAR) + TD64_YEAR,
                step=np.timedelta64(step, 'M'),
                dtype=DT64_MONTH)
        labels.flags.writeable = False
        return cls(labels, name=name)

    #---------------------------------------------------------------------------
    def to_pandas(self) -> None:
        '''Return a Pandas Index.
        '''
        raise NotImplementedError('Pandas does not support a year month type, and it is ambiguous if a date proxy should be the first of the month or the last of the month.')


class IndexYearMonthGO(_IndexDatetimeGOMixin, IndexYearMonth):

    _IMMUTABLE_CONSTRUCTOR = IndexYearMonth
    __slots__ = _INDEX_GO_SLOTS

IndexYearMonth._MUTABLE_CONSTRUCTOR = IndexYearMonthGO

#-------------------------------------------------------------------------------

class IndexDate(IndexDatetime):
    '''A mapping of dates (NumPy :obj:`datetime64[D]`) to positions, immutable and of fixed size.
    '''
    STATIC = True
    _DTYPE = DT64_DAY
    __slots__ = _INDEX_SLOTS

    @classmethod
    def from_date_range(cls: tp.Type[I],
            start: DateInitializer,
            stop: DateInitializer,
            step: int = 1,
            *,
            name: tp.Optional[tp.Hashable] = None
            ) -> I:
        '''
        Get an IndexDate instance over a range of dates, where start and stop is inclusive.
        '''
        labels = np.arange(
                to_datetime64(start, DT64_DAY),
                to_datetime64(stop, DT64_DAY) + TD64_DAY,
                np.timedelta64(step, 'D'))
        labels.flags.writeable = False
        return cls(labels, name=name)

    @classmethod
    def from_year_month_range(cls: tp.Type[I],
            start: YearMonthInitializer,
            stop: YearMonthInitializer,
            step: int = 1,
            *,
            name: tp.Optional[tp.Hashable] = None) -> I:
        '''
        Get an IndexDate instance over a range of months, where start and end are inclusive.
        '''
        labels = np.arange(
                to_datetime64(start, DT64_MONTH),
                to_datetime64(stop, DT64_MONTH) + TD64_MONTH,
                step=np.timedelta64(step, 'D'),
                dtype=DT64_DAY)
        labels.flags.writeable = False
        return cls(labels, name=name)

    @classmethod
    def from_year_range(cls: tp.Type[I],
            start: YearInitializer,
            stop: YearInitializer,
            step: int = 1,
            *,
            name: tp.Optional[tp.Hashable] = None
            ) -> I:
        '''
        Get an IndexDate instance over a range of years, where start and end are inclusive.
        '''
        labels = np.arange(
                to_datetime64(start, DT64_YEAR),
                to_datetime64(stop, DT64_YEAR) + TD64_YEAR,
                step=np.timedelta64(step, 'D'),
                dtype=DT64_DAY)
        labels.flags.writeable = False
        return cls(labels, name=name)


class IndexDateGO(_IndexDatetimeGOMixin, IndexDate):

    _IMMUTABLE_CONSTRUCTOR = IndexDate
    __slots__ = _INDEX_GO_SLOTS

IndexDate._MUTABLE_CONSTRUCTOR = IndexDateGO

#-------------------------------------------------------------------------------
class IndexHour(IndexDatetime):
    '''A mapping of hours (NumPy :obj:`datetime64[h]`) to positions, immutable and of fixed size.
    '''
    STATIC = True
    _DTYPE = DT64_H
    __slots__ = _INDEX_SLOTS

class IndexHourGO(_IndexDatetimeGOMixin, IndexHour):

    _IMMUTABLE_CONSTRUCTOR = IndexHour
    __slots__ = _INDEX_GO_SLOTS

IndexHour._MUTABLE_CONSTRUCTOR = IndexHourGO

#-------------------------------------------------------------------------------
class IndexMinute(IndexDatetime):
    '''A mapping of minutes (NumPy :obj:`datetime64[m]`) to positions, immutable and of fixed size.
    '''
    STATIC = True
    _DTYPE = DT64_M
    __slots__ = _INDEX_SLOTS

class IndexMinuteGO(_IndexDatetimeGOMixin, IndexMinute):

    _IMMUTABLE_CONSTRUCTOR = IndexMinute
    __slots__ = _INDEX_GO_SLOTS

IndexMinute._MUTABLE_CONSTRUCTOR = IndexMinuteGO

#-------------------------------------------------------------------------------
class IndexSecond(IndexDatetime):
    '''A mapping of seconds (NumPy :obj:`datetime64[s]`) to positions, immutable and of fixed size.
    '''
    STATIC = True
    _DTYPE = DT64_S
    __slots__ = _INDEX_SLOTS

class IndexSecondGO(_IndexDatetimeGOMixin, IndexSecond):

    _IMMUTABLE_CONSTRUCTOR = IndexSecond
    __slots__ = _INDEX_GO_SLOTS

IndexSecond._MUTABLE_CONSTRUCTOR = IndexSecondGO

#-------------------------------------------------------------------------------
class IndexMillisecond(IndexDatetime):
    '''A mapping of milliseconds (NumPy :obj:`datetime64[ms]`) to positions, immutable and of fixed size.
    '''
    STATIC = True
    _DTYPE = DT64_MS
    __slots__ = _INDEX_SLOTS

class IndexMillisecondGO(_IndexDatetimeGOMixin, IndexMillisecond):

    _IMMUTABLE_CONSTRUCTOR = IndexMillisecond
    __slots__ = _INDEX_GO_SLOTS

IndexMillisecond._MUTABLE_CONSTRUCTOR = IndexMillisecondGO

#-------------------------------------------------------------------------------
class IndexMicrosecond(IndexDatetime):
    '''A mapping of microseconds (NumPy :obj:`datetime64[us]`) to positions, immutable and of fixed size.
    '''
    STATIC = True
    _DTYPE = DT64_US
    __slots__ = _INDEX_SLOTS

class IndexMicrosecondGO(_IndexDatetimeGOMixin, IndexMicrosecond):

    _IMMUTABLE_CONSTRUCTOR = IndexMicrosecond
    __slots__ = _INDEX_GO_SLOTS

IndexMicrosecond._MUTABLE_CONSTRUCTOR = IndexMicrosecondGO

#-------------------------------------------------------------------------------
class IndexNanosecond(IndexDatetime):
    '''A mapping of nanoseconds (NumPy :obj:`datetime64[ns]`) to positions, immutable and of fixed size.
    '''
    STATIC = True
    _DTYPE = DT64_NS
    __slots__ = _INDEX_SLOTS

class IndexNanosecondGO(_IndexDatetimeGOMixin, IndexNanosecond):

    _IMMUTABLE_CONSTRUCTOR = IndexNanosecond
    __slots__ = _INDEX_GO_SLOTS

IndexNanosecond._MUTABLE_CONSTRUCTOR = IndexNanosecondGO



#-------------------------------------------------------------------------------
_DTYPE_TO_CLASS = {cls._DTYPE: cls for cls in (
        IndexYear,
        IndexYearMonth,
        IndexDate,
        IndexHour,
        IndexMinute,
        IndexSecond,
        IndexMillisecond,
        IndexMicrosecond,
        IndexNanosecond
        )}

def _dtype_to_index_cls(static: bool, dtype: np.dtype) -> tp.Type[Index]:
    '''
    Given an the class of the Index from which this is valled, as well as the dtype of the resultant array, return the appropriate Index class.
    '''

    resolved_static = _DTYPE_TO_CLASS.get(dtype)
    if resolved_static is not None:
        if static:
            return resolved_static
        return resolved_static._MUTABLE_CONSTRUCTOR #type: ignore
    # if origin is a dt64, and dtype is not a dt64, we can go to Index or IndexGO
    if static:
        return Index
    return IndexGO

