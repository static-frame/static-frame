from __future__ import annotations

import datetime
from functools import partial

import numpy as np
import typing_extensions as tp

from static_frame.core.doc_str import doc_inject, doc_update
from static_frame.core.exception import (
    GrowOnlyInvalid,
    InvalidDatetime64Initializer,
    LocInvalid,
)
from static_frame.core.index import INDEX_GO_LEAF_SLOTS, Index, IndexGO, _IndexGOMixin
from static_frame.core.util import (
    DT64_DAY,
    DT64_H,
    DT64_M,
    DT64_MONTH,
    DT64_MS,
    DT64_NS,
    DT64_S,
    DT64_US,
    DT64_YEAR,
    DTYPE_BOOL,
    NAME_DEFAULT,
    TD64_DAY,
    TD64_MONTH,
    TD64_YEAR,
    TDateInitializer,
    TDtypeDT64,
    TILocSelector,
    TIndexInitializer,
    TKeyTransform,
    TLabel,
    TLocSelector,
    TName,
    TNDArrayAny,
    TYearInitializer,
    TYearMonthInitializer,
    WarningsSilent,
    key_to_datetime_key,
    to_datetime64,
    to_timedelta64,
)

if tp.TYPE_CHECKING:
    import pandas  # pragma: no cover
    from arraykit import AutoMap  # pragma: no cover

key_to_datetime_key_year = partial(key_to_datetime_key, dtype=DT64_YEAR)

I = tp.TypeVar('I', bound='IndexDatetime')

# -------------------------------------------------------------------------------
# Specialized index for dates


class IndexDatetime(Index[np.datetime64]):
    """
    Derivation of Index to support Datetime operations. Derived classes must define _DTYPE.
    """

    STATIC = True
    _DTYPE: TDtypeDT64  # define in derived class
    __slots__ = ()

    def __init__(
        self,
        labels: TIndexInitializer,
        /,
        *,
        loc_is_iloc: bool = False,
        name: TName = NAME_DEFAULT,
    ) -> None:
        """Initializer.

        {args}
        """
        assert not loc_is_iloc
        # __init__ here leaves out the dtype argument, reducing the signature to arguments relevant for these derived classes
        Index.__init__(
            self,
            labels,
            name=name,
            loc_is_iloc=loc_is_iloc,
        )

    # ---------------------------------------------------------------------------
    # dict like interface

    def __contains__(
        self,
        value: tp.Any,
        /,
    ) -> bool:
        """Return True if value in the labels. Will only return True for an exact match to the type of dates stored within."""
        try:
            v = to_datetime64(value)
        except ValueError:  # cannot be converted to dt64
            return False
        return self._map.__contains__(v)  # type: ignore

    # ---------------------------------------------------------------------------
    # operators

    def _ufunc_binary_operator(
        self,
        *,
        operator: tp.Callable[..., tp.Any],
        other: tp.Any,
        fill_value: tp.Any = np.nan,
    ) -> TNDArrayAny:
        if self._recache:
            self._update_array_cache()

        if operator.__name__ == 'matmul' or operator.__name__ == 'rmatmul':
            raise NotImplementedError('matrix multiplication not supported')

        if isinstance(other, Index):
            other = other.values  # operate on labels to labels
            other_is_array = True
        elif isinstance(other, str):
            # do not pass dtype, as want to coerce to this parsed type, not the type of self
            other = to_datetime64(other)
            other_is_array = False
        elif other.__class__ is np.ndarray:
            other_is_array = True
        else:
            other_is_array = False

        result: TNDArrayAny
        if isinstance(other, np.datetime64):
            # convert labels to other's datetime64 type to enable matching on month, year, etc.
            result = operator(self._labels.astype(other.dtype), other)
        elif isinstance(other, datetime.timedelta):
            result = operator(self._labels, to_timedelta64(other))
        else:  # np.timedelta64 should work fine here
            with WarningsSilent():
                result = operator(self._labels, other)

        # NOTE: remove when min numpy is 1.25
        # NOTE: similar branching as in container_util.apply_binary_operator
        # NOTE: all string will have been converted to dt64, or raise ValueError; comparison to same sized iterables (list, tuple) will result in an array when they are the same size
        if result is False:  # will never be True
            if (  # type: ignore
                not other_is_array
                and hasattr(other, '__len__')
                and len(other) == len(self)  # pyright: ignore
            ):
                # NOTE: equality comparisons of an array to same sized iterable normally return an array, but with dt64 types they just return False
                result = np.full(self.shape, result, dtype=DTYPE_BOOL)
            elif other_is_array and other.size == 1:  # pyright: ignore
                # elements in arrays of 0 or more dimensions are acceptable; this is what NP does for arithmetic operators when the types are compatible
                result = np.full(self.shape, result, dtype=DTYPE_BOOL)
            else:
                raise ValueError('operands could not be broadcast together')
                # raise on unaligned shapes as is done for arithmetic operators

        result.flags.writeable = False
        return result

    def _loc_to_iloc(
        self,
        key: TLocSelector,
        key_transform: TKeyTransform = key_to_datetime_key,
        partial_selection: bool = False,
    ) -> TILocSelector:
        """
        Specialized for IndexData indices to convert string data representations into np.datetime64 objects as appropriate.
        """
        # not passing self.dtype to key_to_datetime_key so as to allow translation to a foreign datetime; slice comparison will be handled by map_slice_args
        return Index._loc_to_iloc(
            self,
            key=key,
            key_transform=key_transform,
            partial_selection=partial_selection,
        )

    # ---------------------------------------------------------------------------
    def to_pandas(self) -> 'pandas.DatetimeIndex':
        """Return a Pandas Index."""
        import pandas

        return pandas.DatetimeIndex(self.values.copy(), name=self._name)  # pyright: ignore

    # ---------------------------------------------------------------------------

    @doc_inject(selector='searchsorted', label_type='iloc (integer)')
    def iloc_searchsorted(
        self,
        values: tp.Any,
        /,
        *,
        side_left: bool = True,
    ) -> TNDArrayAny:
        """
        {doc}

        Args:
            {values}
            {side_left}
        """
        # permit variable forms of date specification
        return Index.iloc_searchsorted(
            self,
            key_to_datetime_key(values),
            side_left=side_left,
        )


doc_update(IndexDatetime.__init__, selector='index_date_time_init')


# -------------------------------------------------------------------------------
class _IndexDatetimeGOMixin(_IndexGOMixin):
    _DTYPE: TDtypeDT64
    _map: tp.Optional[AutoMap]
    __slots__ = ()  # define in derived class

    def append(
        self,
        value: TLabel,
        /,
    ) -> None:
        """Specialize for fixed-typed indices: convert `value` argument; do not need to resolve_dtype with each addition; self._map is never None"""
        try:
            value = to_datetime64(value, self._DTYPE)  # type: ignore
        except ValueError:
            raise GrowOnlyInvalid() from None

        if self._map is not None:  # if starting from an empty index
            try:
                self._map.add(value)
            except ValueError as e:
                raise KeyError(f'duplicate key append attempted: {value}') from e
        self._labels_mutable.append(value)
        self._positions_mutable_count += 1
        self._recache = True


# -------------------------------------------------------------------------------
class IndexYear(IndexDatetime):
    """A mapping of years (NumPy :obj:`datetime64[Y]`) to positions, immutable and of fixed size."""

    STATIC = True
    _DTYPE = DT64_YEAR
    __slots__ = ()

    @classmethod
    def from_date_range(
        cls: tp.Type[I],
        start: TDateInitializer,
        stop: TDateInitializer,
        step: int = 1,
        *,
        name: tp.Optional[TLabel] = None,
    ) -> I:
        """
        Get an IndexYearMonth instance over a range of dates, where start and stop are inclusive.
        """
        labels = np.arange(
            to_datetime64(start, DT64_DAY),
            to_datetime64(stop, DT64_DAY).astype(DT64_YEAR) + TD64_YEAR,
            np.timedelta64(step, 'Y'),
            dtype=DT64_YEAR,
        )
        labels.flags.writeable = False
        return cls(labels, name=name)

    @classmethod
    def from_year_month_range(
        cls: tp.Type[I],
        start: TYearMonthInitializer,
        stop: TYearMonthInitializer,
        step: int = 1,
        *,
        name: tp.Optional[TLabel] = None,
    ) -> I:
        """
        Get an IndexYearMonth instance over a range of months, where start and end are inclusive.
        """

        labels = np.arange(
            to_datetime64(start, DT64_MONTH),
            to_datetime64(stop, DT64_MONTH).astype(DT64_YEAR) + TD64_YEAR,
            np.timedelta64(step, 'Y'),
            dtype=DT64_YEAR,
        )
        labels.flags.writeable = False
        return cls(labels, name=name)

    @classmethod
    def from_year_range(
        cls: tp.Type[I],
        start: TYearInitializer,
        stop: TYearInitializer,
        step: int = 1,
        *,
        name: tp.Optional[TLabel] = None,
    ) -> I:
        """
        Get an IndexDate instance over a range of years, where start and end are inclusive.
        """
        labels: TNDArrayAny = np.arange(
            to_datetime64(start, DT64_YEAR),
            to_datetime64(stop, DT64_YEAR) + TD64_YEAR,
            step=np.timedelta64(step, 'Y'),
        )
        labels.flags.writeable = False
        return cls(labels, name=name)

    # ---------------------------------------------------------------------------
    # specializations to permit integers as years

    def __contains__(
        self,
        value: tp.Any,
        /,
    ) -> bool:
        """Return True if value in the labels. Will only return True for an exact match to the type of dates stored within."""
        try:
            return self._map.__contains__(to_datetime64(value, self._DTYPE))  # type: ignore
        except InvalidDatetime64Initializer:
            # if value is a dt64 and not of a the same unit as self._DTYPE, the initializations exception is raised, which means False
            return False

    def _loc_to_iloc(
        self,
        key: TLocSelector,
        key_transform: TKeyTransform = key_to_datetime_key_year,
        partial_selection: bool = False,
    ) -> TILocSelector:
        """
        Specialized for IndexData indices to convert string data representations into np.datetime64 objects as appropriate.
        """
        try:
            return Index._loc_to_iloc(
                self,
                key=key,
                key_transform=key_transform,
                partial_selection=partial_selection,
            )
        except InvalidDatetime64Initializer as e:
            raise LocInvalid(e.args[0]) from None

    # ---------------------------------------------------------------------------
    def to_pandas(self) -> None:
        """Return a Pandas Index."""
        raise NotImplementedError(
            'Pandas does not support a year type, and it is ambiguous if a date proxy should be the first of the year or the last of the year.'
        )


class IndexYearGO(_IndexDatetimeGOMixin, IndexYear):
    _IMMUTABLE_CONSTRUCTOR = IndexYear
    __slots__ = INDEX_GO_LEAF_SLOTS


IndexYear._MUTABLE_CONSTRUCTOR = IndexYearGO


# -------------------------------------------------------------------------------
class IndexYearMonth(IndexDatetime):
    """A mapping of year months (NumPy :obj:`datetime64[M]`) to positions, immutable and of fixed size."""

    STATIC = True
    _DTYPE = DT64_MONTH
    __slots__ = ()

    @classmethod
    def from_date_range(
        cls: tp.Type[I],
        start: TDateInitializer,
        stop: TDateInitializer,
        step: int = 1,
        *,
        name: tp.Optional[TLabel] = None,
    ) -> I:
        """
        Get an IndexYearMonth instance over a range of dates, where start and stop is inclusive.
        """
        labels = np.arange(
            to_datetime64(start, DT64_DAY),
            to_datetime64(stop, DT64_DAY).astype(DT64_MONTH) + TD64_MONTH,
            np.timedelta64(step, 'M'),
            dtype=DT64_MONTH,
        )

        labels.flags.writeable = False
        return cls(labels, name=name)

    @classmethod
    def from_year_month_range(
        cls: tp.Type[I],
        start: TYearMonthInitializer,
        stop: TYearMonthInitializer,
        step: int = 1,
        *,
        name: tp.Optional[TLabel] = None,
    ) -> I:
        """
        Get an IndexYearMonth instance over a range of months, where start and end are inclusive.
        """

        labels = np.arange(
            to_datetime64(start, DT64_MONTH),
            to_datetime64(stop, DT64_MONTH) + TD64_MONTH,
            np.timedelta64(step, 'M'),
            dtype=DT64_MONTH,
        )
        labels.flags.writeable = False
        return cls(labels, name=name)

    @classmethod
    def from_year_range(
        cls: tp.Type[I],
        start: TYearInitializer,
        stop: TYearInitializer,
        step: int = 1,
        *,
        name: tp.Optional[TLabel] = None,
    ) -> I:
        """
        Get an IndexYearMonth instance over a range of years, where start and end are inclusive.
        """
        labels = np.arange(
            to_datetime64(start, DT64_YEAR),
            to_datetime64(stop, DT64_YEAR) + TD64_YEAR,
            step=np.timedelta64(step, 'M'),
            dtype=DT64_MONTH,
        )
        labels.flags.writeable = False
        return cls(labels, name=name)

    # ---------------------------------------------------------------------------
    def to_pandas(self) -> None:
        """Return a Pandas Index."""
        raise NotImplementedError(
            'Pandas does not support a year month type, and it is ambiguous if a date proxy should be the first of the month or the last of the month.'
        )


class IndexYearMonthGO(_IndexDatetimeGOMixin, IndexYearMonth):
    _IMMUTABLE_CONSTRUCTOR = IndexYearMonth
    __slots__ = INDEX_GO_LEAF_SLOTS


IndexYearMonth._MUTABLE_CONSTRUCTOR = IndexYearMonthGO

# -------------------------------------------------------------------------------


class IndexDate(IndexDatetime):
    """A mapping of dates (NumPy :obj:`datetime64[D]`) to positions, immutable and of fixed size."""

    STATIC = True
    _DTYPE = DT64_DAY
    __slots__ = ()

    @classmethod
    def from_date_range(
        cls: tp.Type[I],
        start: TDateInitializer,
        stop: TDateInitializer,
        step: int = 1,
        *,
        name: tp.Optional[TLabel] = None,
    ) -> I:
        """
        Get an IndexDate instance over a range of dates, where start and stop is inclusive.
        """
        labels: TNDArrayAny = np.arange(
            to_datetime64(start, DT64_DAY),
            to_datetime64(stop, DT64_DAY) + TD64_DAY,
            np.timedelta64(step, 'D'),
        )
        labels.flags.writeable = False
        return cls(labels, name=name)

    @classmethod
    def from_year_month_range(
        cls: tp.Type[I],
        start: TYearMonthInitializer,
        stop: TYearMonthInitializer,
        step: int = 1,
        *,
        name: tp.Optional[TLabel] = None,
    ) -> I:
        """
        Get an IndexDate instance over a range of months, where start and end are inclusive.
        """
        labels = np.arange(
            to_datetime64(start, DT64_MONTH),
            to_datetime64(stop, DT64_MONTH) + TD64_MONTH,
            step=np.timedelta64(step, 'D'),
            dtype=DT64_DAY,
        )
        labels.flags.writeable = False
        return cls(labels, name=name)

    @classmethod
    def from_year_range(
        cls: tp.Type[I],
        start: TYearInitializer,
        stop: TYearInitializer,
        step: int = 1,
        *,
        name: tp.Optional[TLabel] = None,
    ) -> I:
        """
        Get an IndexDate instance over a range of years, where start and end are inclusive.
        """
        labels = np.arange(
            to_datetime64(start, DT64_YEAR),
            to_datetime64(stop, DT64_YEAR) + TD64_YEAR,
            step=np.timedelta64(step, 'D'),
            dtype=DT64_DAY,
        )
        labels.flags.writeable = False
        return cls(labels, name=name)


class IndexDateGO(_IndexDatetimeGOMixin, IndexDate):
    _IMMUTABLE_CONSTRUCTOR = IndexDate
    __slots__ = INDEX_GO_LEAF_SLOTS


IndexDate._MUTABLE_CONSTRUCTOR = IndexDateGO


# -------------------------------------------------------------------------------
class IndexHour(IndexDatetime):
    """A mapping of hours (NumPy :obj:`datetime64[h]`) to positions, immutable and of fixed size."""

    STATIC = True
    _DTYPE = DT64_H
    __slots__ = ()


class IndexHourGO(_IndexDatetimeGOMixin, IndexHour):
    _IMMUTABLE_CONSTRUCTOR = IndexHour
    __slots__ = INDEX_GO_LEAF_SLOTS


IndexHour._MUTABLE_CONSTRUCTOR = IndexHourGO


# -------------------------------------------------------------------------------
class IndexMinute(IndexDatetime):
    """A mapping of minutes (NumPy :obj:`datetime64[m]`) to positions, immutable and of fixed size."""

    STATIC = True
    _DTYPE = DT64_M
    __slots__ = ()


class IndexMinuteGO(_IndexDatetimeGOMixin, IndexMinute):
    _IMMUTABLE_CONSTRUCTOR = IndexMinute
    __slots__ = INDEX_GO_LEAF_SLOTS


IndexMinute._MUTABLE_CONSTRUCTOR = IndexMinuteGO


# -------------------------------------------------------------------------------
class IndexSecond(IndexDatetime):
    """A mapping of seconds (NumPy :obj:`datetime64[s]`) to positions, immutable and of fixed size."""

    STATIC = True
    _DTYPE = DT64_S
    __slots__ = ()


class IndexSecondGO(_IndexDatetimeGOMixin, IndexSecond):
    _IMMUTABLE_CONSTRUCTOR = IndexSecond
    __slots__ = INDEX_GO_LEAF_SLOTS


IndexSecond._MUTABLE_CONSTRUCTOR = IndexSecondGO


# -------------------------------------------------------------------------------
class IndexMillisecond(IndexDatetime):
    """A mapping of milliseconds (NumPy :obj:`datetime64[ms]`) to positions, immutable and of fixed size."""

    STATIC = True
    _DTYPE = DT64_MS
    __slots__ = ()


class IndexMillisecondGO(_IndexDatetimeGOMixin, IndexMillisecond):
    _IMMUTABLE_CONSTRUCTOR = IndexMillisecond
    __slots__ = INDEX_GO_LEAF_SLOTS


IndexMillisecond._MUTABLE_CONSTRUCTOR = IndexMillisecondGO


# -------------------------------------------------------------------------------
class IndexMicrosecond(IndexDatetime):
    """A mapping of microseconds (NumPy :obj:`datetime64[us]`) to positions, immutable and of fixed size."""

    STATIC = True
    _DTYPE = DT64_US
    __slots__ = ()


class IndexMicrosecondGO(_IndexDatetimeGOMixin, IndexMicrosecond):
    _IMMUTABLE_CONSTRUCTOR = IndexMicrosecond
    __slots__ = INDEX_GO_LEAF_SLOTS


IndexMicrosecond._MUTABLE_CONSTRUCTOR = IndexMicrosecondGO


# -------------------------------------------------------------------------------
class IndexNanosecond(IndexDatetime):
    """A mapping of nanoseconds (NumPy :obj:`datetime64[ns]`) to positions, immutable and of fixed size."""

    STATIC = True
    _DTYPE = DT64_NS
    __slots__ = ()


class IndexNanosecondGO(_IndexDatetimeGOMixin, IndexNanosecond):
    _IMMUTABLE_CONSTRUCTOR = IndexNanosecond
    __slots__ = INDEX_GO_LEAF_SLOTS


IndexNanosecond._MUTABLE_CONSTRUCTOR = IndexNanosecondGO


# -------------------------------------------------------------------------------
_DTYPE_TO_CLASS: tp.Dict[TDtypeDT64, tp.Type[Index[np.datetime64]]] = {
    cls._DTYPE: cls
    for cls in (
        IndexYear,
        IndexYearMonth,
        IndexDate,
        IndexHour,
        IndexMinute,
        IndexSecond,
        IndexMillisecond,
        IndexMicrosecond,
        IndexNanosecond,
    )
}


def dtype_to_index_cls(static: bool, dtype: TDtypeDT64) -> tp.Type[Index[tp.Any]]:
    """
    Given an the class of the Index from which this is valled, as well as the dtype of the resultant array, return the appropriate Index class.
    """

    resolved_static: tp.Type[Index[tp.Any]] | None = _DTYPE_TO_CLASS.get(dtype)
    if resolved_static is not None:
        if static:
            return resolved_static
        return resolved_static._MUTABLE_CONSTRUCTOR  # type: ignore
    # if origin is a dt64, and dtype is not a dt64, we can go to Index or IndexGO
    if static:
        return Index
    return IndexGO
