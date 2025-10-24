from __future__ import annotations

import csv
from collections.abc import Set, Sized
from copy import deepcopy
from functools import partial
from itertools import chain, product

import numpy as np
import typing_extensions as tp
from arraykit import (
    array_deepcopy,
    astype_array,
    delimited_to_arrays,
    first_true_1d,
    immutable_filter,
    mloc,
    name_filter,
    resolve_dtype,
)
from numpy.ma import MaskedArray

from static_frame.core.assign import Assign
from static_frame.core.container import ContainerOperand
from static_frame.core.container_util import (
    apply_binary_operator,
    axis_window_items,
    get_col_fill_value_factory,
    index_from_optional_constructor,
    index_many_concat,
    index_many_to_one,
    is_fill_value_factory_initializer,
    iter_component_signature_bytes,
    matmul,
    pandas_to_numpy,
    rehierarch_from_index_hierarchy,
    sort_index_from_params,
)
from static_frame.core.display import Display, DisplayActive, DisplayHeader
from static_frame.core.display_config import DisplayConfig, DisplayFormats
from static_frame.core.doc_str import doc_inject, doc_update
from static_frame.core.exception import (
    AxisInvalid,
    ErrorInitSeries,
    RelabelInvalid,
    immutable_type_error_factory,
)
from static_frame.core.index import Index
from static_frame.core.index_auto import (
    IndexAutoFactory,
    IndexDefaultConstructorFactory,
    TIndexAutoFactory,
    TIndexInitOrAuto,
    TRelabelInput,
)
from static_frame.core.index_base import IndexBase
from static_frame.core.index_correspondence import IndexCorrespondence, assign_via_ic

# from static_frame.core.index_correspondence import assign_via_mask
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.node_dt import InterfaceDatetime
from static_frame.core.node_fill_value import InterfaceFillValue
from static_frame.core.node_iter import (
    IterNodeApplyType,
    IterNodeDepthLevel,
    IterNodeGroup,
    IterNodeGroupOther,
    IterNodeNoArgMapable,
    IterNodeWindow,
)
from static_frame.core.node_re import InterfaceRe
from static_frame.core.node_selector import (
    InterfaceAssignTrio,
    InterfaceSelectTrio,
    InterGetItemILocReduces,
    InterGetItemLocReduces,
)
from static_frame.core.node_str import InterfaceString
from static_frame.core.node_values import InterfaceValues
from static_frame.core.rank import RankMethod, rank_1d
from static_frame.core.series_mapping import SeriesMapping
from static_frame.core.style_config import (
    STYLE_CONFIG_DEFAULT,
    StyleConfig,
    style_config_css_factory,
)
from static_frame.core.util import (
    BOOL_TYPES,
    DEFAULT_SORT_KIND,
    DTYPE_NA_KINDS,
    DTYPE_OBJECT,
    EMPTY_ARRAY,
    EMPTY_SLICE,
    FILL_VALUE_DEFAULT,
    FLOAT_TYPES,
    INT_TYPES,
    NAME_DEFAULT,
    NULL_SLICE,
    REVERSE_SLICE,
    STRING_TYPES,
    IterNodeType,
    ManyToOneType,
    SortStatus,
    TBoolOrBools,
    TCallableAny,
    TDepthLevel,
    TDtypeSpecifier,
    TILocSelector,
    TILocSelectorMany,
    TILocSelectorOne,
    TIndexCtorSpecifier,
    TIndexCtorSpecifiers,
    TIndexInitializer,
    TLabel,
    TLocSelector,
    TLocSelectorMany,
    TName,
    TNDArrayIntDefault,
    TPathSpecifierOrTextIO,
    TSeriesInitializer,
    TSortKinds,
    TUFunc,
    argmax_1d,
    argmin_1d,
    array_shift,
    array_to_duplicated,
    array_to_groups_and_locations,
    array_ufunc_axis_skipna,
    arrays_equal,
    binary_transition,
    concat_resolved,
    depth_level_from_specifier,
    dtype_from_element,
    dtype_kind_to_na,
    dtype_to_fill_value,
    full_for_fill,
    iloc_to_insertion_iloc,
    intersect1d,
    is_callable_or_mapping,
    isfalsy_array,
    isin,
    isna_array,
    iterable_to_array_1d,
    slices_from_targets,
    transition_slices_from_group,
    ufunc_unique1d,
    ufunc_unique_enumerated,
    validate_dtype_specifier,
    write_optional_file,
)

if tp.TYPE_CHECKING:
    import pandas

    from static_frame.core.generic_aliases import (
        TBusAny,
        TFrameAny,
        TFrameGOAny,
        TFrameHEAny,
    )

    TNDArrayAny = np.ndarray[tp.Any, tp.Any]
    TDtypeAny = np.dtype[tp.Any]
    FrameType = tp.TypeVar('FrameType', bound=TFrameAny)


# -------------------------------------------------------------------------------
TVDtype = tp.TypeVar('TVDtype', bound=np.generic, default=tp.Any)
TVIndex = tp.TypeVar('TVIndex', bound=IndexBase, default=tp.Any)


def _NA_VALUES_CTOR(count: int) -> None: ...


class Series(ContainerOperand, tp.Generic[TVIndex, TVDtype]):
    """A one-dimensional, ordered, labelled container, immutable and of fixed size."""

    __slots__ = (
        'values',
        '_index',
        '_name',
    )
    values: TNDArrayAny
    _index: IndexBase
    _NDIM: int = 1

    # ---------------------------------------------------------------------------
    @classmethod
    def from_element(
        cls,
        element: tp.Any,
        /,
        *,
        index: tp.Union[TIndexInitializer, IndexAutoFactory],
        dtype: TDtypeSpecifier = None,
        name: TName = None,
        index_constructor: tp.Optional[TIndexCtorSpecifier] = None,
        own_index: bool = False,
    ) -> tp.Self:
        """
        Create a :obj:`static_frame.Series` from a single element. The size of the resultant container will be determined by the ``index`` argument.

        Returns:
            :obj:`static_frame.Series`
        """
        if own_index:
            index_final = index
        else:
            index_final = index_from_optional_constructor(
                index, default_constructor=Index, explicit_constructor=index_constructor
            )

        length = len(index_final)  # type: ignore
        dtype = None if dtype is None else np.dtype(dtype)
        array = full_for_fill(
            dtype,
            length,
            element,
            resolve_fill_value_dtype=dtype is None,  # True means derive from fill value
        )
        array.flags.writeable = False
        return cls(
            array,
            index=index_final,
            name=name,
            own_index=True,
        )

    @classmethod
    def from_items(
        cls,
        pairs: tp.Iterable[tp.Tuple[TLabel, tp.Any]],
        /,
        *,
        dtype: TDtypeSpecifier = None,
        name: TName = None,
        index_constructor: tp.Optional[tp.Callable[..., IndexBase]] = None,
    ) -> tp.Self:
        """Series construction from an iterator or generator of pairs, where the first pair value is the index and the second is the value.

        Args:
            pairs: Iterable of pairs of index, value.
            dtype: dtype or valid dtype specifier.
            name:
            index_constructor:

        Returns:
            :obj:`static_frame.Series`
        """
        index = []

        def values() -> tp.Iterator[tp.Any]:
            for k, v in pairs:
                # populate index list as side effect of iterating values
                index.append(k)
                yield v

        return cls(
            values(),
            index=index,
            dtype=dtype,
            name=name,
            index_constructor=index_constructor,
        )

    @classmethod
    def from_delimited(
        cls,
        delimited: str,
        /,
        *,
        delimiter: str,
        index: tp.Optional[TIndexInitOrAuto] = None,
        dtype: TDtypeSpecifier = None,
        name: TName = None,
        index_constructor: tp.Optional[TIndexCtorSpecifier] = None,
        skip_initial_space: bool = False,
        quoting: int = csv.QUOTE_MINIMAL,
        quote_char: str = '"',
        quote_double: bool = True,
        escape_char: tp.Optional[str] = None,
        thousands_char: str = '',
        decimal_char: str = '.',
        own_index: bool = False,
    ) -> tp.Self:
        """Series construction from a delimited string.

        Args:
            dtype: if None, dtype will be inferred.
        """
        get_col_dtype = None if dtype is None else lambda x: dtype
        [array] = delimited_to_arrays(
            (delimited,),  # make into iterable of one string
            dtypes=get_col_dtype,
            delimiter=delimiter,
            quoting=quoting,
            quotechar=quote_char,
            doublequote=quote_double,
            escapechar=escape_char,
            thousandschar=thousands_char,
            decimalchar=decimal_char,
            skipinitialspace=skip_initial_space,
        )
        if own_index:
            index_final = index
        else:
            index = IndexAutoFactory(len(array)) if index is None else index
            index_final = index_from_optional_constructor(
                index, default_constructor=Index, explicit_constructor=index_constructor
            )
        return cls(
            array,
            index=index_final,
            name=name,
            own_index=True,
        )

    @classmethod
    def from_dict(
        cls,
        mapping: tp.Mapping[tp.Any, tp.Any],
        /,
        *,
        dtype: TDtypeSpecifier = None,
        name: TName = None,
        index_constructor: tp.Optional[tp.Callable[..., IndexBase]] = None,
    ) -> tp.Self:
        """Series construction from a dictionary, where the first pair value is the index and the second is the value.

        Args:
            mapping: a dictionary or similar mapping interface.
            dtype: dtype or valid dtype specifier.

        Returns:
            :obj:`Series`
        """
        return cls.from_items(
            mapping.items(), name=name, dtype=dtype, index_constructor=index_constructor
        )

    @classmethod
    def from_concat(
        cls,
        containers: tp.Iterable[tp.Union[TSeriesAny, TBusAny]],
        /,
        *,
        index: tp.Optional[TIndexInitOrAuto] = None,
        index_constructor: tp.Optional[TIndexCtorSpecifier] = None,
        name: TName = NAME_DEFAULT,
    ) -> tp.Self:
        """
        Concatenate multiple :obj:`Series` into a new :obj:`Series`.

        Args:
            containers: Iterable of ``Series`` from which values in the new ``Series`` are drawn.
            index: If None, the resultant index will be the concatenation of all indices (assuming they are unique in combination). If ``IndexAutoFactory``, the resultant index is a auto-incremented integer index. Otherwise, the value is used as a index initializer.
            index_constructor:
            name:

        Returns:
            :obj:`static_frame.Series`
        """
        array_values = []
        if index is None:
            indices = []

        name_first = NAME_DEFAULT
        name_aligned = True

        for c in containers:
            if name_first == NAME_DEFAULT:
                name_first = c.name
            elif name_first != c.name:
                name_aligned = False

            array_values.append(c.values)
            if index is None:
                indices.append(c.index)

        # End quickly if empty iterable
        if not array_values:
            return cls((), index=index, name=name)

        # returns immutable arrays
        values = concat_resolved(array_values)

        own_index = False
        if index is None:
            index = index_many_concat(
                indices,
                cls_default=Index,
                explicit_constructor=index_constructor,
            )
            own_index = True
        elif index is IndexAutoFactory:
            # set index arg to None to force IndexAutoFactory usage in creation
            index = None
        # else, index was supplied as an iterable, above

        if name == NAME_DEFAULT:
            # only derive if not explicitly set
            name = name_first if name_aligned else None

        return cls(
            values,
            index=index,
            name=name,
            index_constructor=index_constructor,
            own_index=own_index,
        )

    @classmethod
    def from_concat_items(
        cls,
        items: tp.Iterable[tp.Tuple[TLabel, TSeriesAny]],
        /,
        *,
        name: TName = None,
        index_constructor: tp.Optional[TIndexCtorSpecifier] = None,
    ) -> tp.Self:
        """
        Produce a :obj:`Series` with a hierarchical index from an iterable of pairs of labels, :obj:`Series`. The :obj:`IndexHierarchy` is formed from the provided labels and the :obj:`Index` if each :obj:`Series`.

        Args:
            items: Iterable of pairs of label, :obj:`Series`

        Returns:
            :obj:`static_frame.Series`
        """
        array_values = []

        if index_constructor is None or isinstance(
            index_constructor, IndexDefaultConstructorFactory
        ):
            # default index constructor expects delivery of Indices for greater efficiency
            def gen() -> tp.Iterator[tp.Tuple[TLabel, IndexBase]]:
                for label, series in items:
                    array_values.append(series.values)
                    yield label, series._index
        else:

            def gen() -> tp.Iterator[tp.Tuple[TLabel, IndexBase]]:
                for label, series in items:
                    array_values.append(series.values)
                    yield from product((label,), series._index)  # pyright: ignore

        values: TNDArrayAny
        try:
            # populates array_values as side
            ih = index_from_optional_constructor(
                gen(),
                default_constructor=IndexHierarchy.from_index_items,
                explicit_constructor=index_constructor,
            )
            # returns immutable array
            values = concat_resolved(array_values)
            own_index = True
        except StopIteration:
            # Default to empty when given an empty iterable
            ih = None
            values = EMPTY_ARRAY
            own_index = False

        return cls(values, index=ih, own_index=own_index, name=name)

    @classmethod
    def from_overlay(
        cls,
        containers: tp.Iterable[tp.Self],
        /,
        *,
        index: tp.Optional[TIndexInitializer] = None,
        union: bool = True,
        name: TName = None,
        func: tp.Callable[[TNDArrayAny], TNDArrayAny] = isna_array,
        fill_value: tp.Any = FILL_VALUE_DEFAULT,
    ) -> tp.Self:
        """Return a new :obj:`Series` made by overlaying containers, aligned values are filled with values from subsequent containers with left-to-right precedence. Values are filled based on a passed function that must return a Boolean array. By default, that function is `isna_array`, returning True for missing values (NaN and None).

        Args:
            containers: Iterable of :obj:`Series`.
            *
            index: An :obj:`Index` or :obj:`IndexHierarchy`, or index initializer, to be used as the index upon which all containers are aligned. :obj:`IndexAutoFactory` is not supported.
            union: If True, and no ``index`` argument is supplied, a union index from ``containers`` will be used; if False, the intersection index will be used.
            name:
            func:
            fill_value:
        """
        if not hasattr(containers, '__len__'):
            containers = tuple(containers)  # exhaust a generator

        if index is None:
            index = index_many_to_one(
                (c.index for c in containers),
                cls_default=Index,
                many_to_one_type=ManyToOneType.UNION
                if union
                else ManyToOneType.INTERSECT,
            )
        else:  # construct an index if not an index
            if not isinstance(index, IndexBase):
                index = Index(index)

        container_iter = iter(containers)
        container_first = next(container_iter)

        if container_first._index.equals(index):
            post = cls(container_first.values, index=index, own_index=True, name=name)
        else:
            # if the indices are not equal, we have to reindex, and we need to provide a fill_value that does minimal type corcion to the original
            if fill_value is FILL_VALUE_DEFAULT:
                fill_value = dtype_kind_to_na(container_first.dtype.kind)
            post = container_first.reindex(index, fill_value=fill_value).rename(name)

        for container in container_iter:
            filled = post._fill_missing(container, func)
            post = filled
        return post

    @classmethod
    @doc_inject()
    def from_pandas(
        cls,
        value: 'pandas.Series[tp.Any]',  # pyright: ignore
        /,
        *,
        index: TIndexInitOrAuto = None,
        index_constructor: TIndexCtorSpecifier = None,
        name: TName = NAME_DEFAULT,
        own_data: bool = False,
    ) -> tp.Self:
        """Given a Pandas Series, return a Series.

        Args:
            value: Pandas Series.
            *
            index_constructor:
            name:
            {own_data}

        Returns:
            :obj:`static_frame.Series`
        """
        import pandas

        if not isinstance(value, pandas.Series):
            raise ErrorInitSeries(
                f'from_pandas must be called with a Pandas Series object, not: {type(value)}'
            )

        data = pandas_to_numpy(value, own_data=own_data)

        name = name if name is not NAME_DEFAULT else value.name

        own_index = False
        if index is IndexAutoFactory:
            index = None
        elif index is not None:
            pass  # pass index into constructor
        elif isinstance(value.index, pandas.MultiIndex):
            index = IndexHierarchy.from_pandas(value.index)
            own_index = True
        else:  # if None
            index = Index.from_pandas(value.index)
            own_index = index_constructor is None

        return cls(
            data,
            index=index,
            index_constructor=index_constructor,
            own_index=own_index,
            name=name,
        )

    # ---------------------------------------------------------------------------
    def __init__(
        self,
        values: TSeriesInitializer,
        /,
        *,
        index: tp.Union[
            TIndexInitializer, IndexAutoFactory, TIndexAutoFactory, None
        ] = None,
        name: TName = NAME_DEFAULT,
        dtype: TDtypeSpecifier = None,
        index_constructor: tp.Optional[TIndexCtorSpecifier] = None,
        own_index: bool = False,
    ) -> None:
        """Initializer.

        Args:
            values: An iterable of values to be aligned with the supplied (or automatically generated) index.
            {index}
            name:
            dtype:
            index_constructor:
            {own_index}
        """

        if own_index and index is None:
            raise ErrorInitSeries('cannot own_index if no index is provided.')

        # -----------------------------------------------------------------------
        # values assignment

        values_constructor = _NA_VALUES_CTOR

        if values.__class__ is not np.ndarray:
            if isinstance(values, dict):
                raise ErrorInitSeries(
                    'use Series.from_dict to create a Series from a mapping.'
                )
            elif isinstance(values, Series):
                self.values = values.values  # take immutable array
                if dtype is not None and dtype != values.dtype:
                    raise ErrorInitSeries(
                        f'when supplying values via Series, the dtype argument is not required; if provided ({dtype}), it must agree with the dtype of the Series ({values.dtype})'
                    )
                if index is None and index_constructor is None:
                    # set up for direct assignment below; index is always immutable
                    index = values.index
                    own_index = True
                if name is NAME_DEFAULT:
                    name = values.name  # propagate Series.name
            elif hasattr(values, '__iter__') and not isinstance(values, STRING_TYPES):
                # returned array is already immutable
                self.values, _ = iterable_to_array_1d(values, dtype=dtype)
            else:  # it must be an element, or a string
                raise ErrorInitSeries(
                    'Use Series.from_element to create a Series from an element.'
                )

        else:  # is numpy array
            if dtype is not None and dtype != values.dtype:  # type: ignore
                raise ErrorInitSeries(
                    f'when supplying values via array, the dtype argument is not required; if provided ({dtype}), it must agree with the dtype of the array ({values.dtype})'  # type: ignore
                )

            if values.shape == ():  # type: ignore
                # handle special case of NP element
                def values_constructor(count: int) -> None:
                    self.values = np.repeat(values, count)  # type: ignore
                    self.values.flags.writeable = False
            else:
                self.values = immutable_filter(values)  # type: ignore

        self._name = None if name is NAME_DEFAULT else name_filter(name)  # pyright: ignore

        # -----------------------------------------------------------------------
        # index assignment
        self._index: IndexBase

        if own_index:
            self._index = index  # type: ignore
        elif index is None or index is IndexAutoFactory:
            # if a values constructor is defined, self.values is not yet defined, and no index is supplied, the resultant shape will be of length 1. (If an index is supplied, the shape might be larger than one if an array element was given
            if values_constructor is not _NA_VALUES_CTOR:
                value_count = 1
            else:
                value_count = len(self.values)
            self._index = IndexAutoFactory.from_optional_constructor(
                value_count,
                default_constructor=Index,
                explicit_constructor=index_constructor,
            )
        else:  # an iterable of labels, or an index subclass
            self._index = index_from_optional_constructor(
                index, default_constructor=Index, explicit_constructor=index_constructor
            )
        index_count = self._index.__len__()  # pyright: ignore

        if not self._index.STATIC:  # pyright: ignore
            raise ErrorInitSeries('non-static index cannot be assigned to Series')

        if values_constructor is not _NA_VALUES_CTOR:
            values_constructor(index_count)  # updates self.values
            # must update after calling values constructor
        value_count = len(self.values)

        # -----------------------------------------------------------------------
        # final evaluation

        if self.values.ndim != self._NDIM:
            raise ErrorInitSeries('dimensionality of final values not supported')
        if value_count != index_count:
            raise ErrorInitSeries(
                f'Index has incorrect size (got {index_count}, expected {value_count})'
            )

    # ---------------------------------------------------------------------------
    def __setstate__(self, state: tp.Any) -> None:
        """
        Ensure that reanimated NP arrays are set not writeable.
        """
        for key, value in state[1].items():
            setattr(self, key, value)
        self.values.flags.writeable = False

    def __deepcopy__(self, memo: tp.Dict[int, tp.Any]) -> tp.Self:
        obj = self.__class__.__new__(self.__class__)
        obj.values = array_deepcopy(self.values, memo)
        obj._index = deepcopy(self._index, memo)
        obj._name = self._name  # should be hashable/immutable

        memo[id(self)] = obj
        return obj

    def __copy__(self) -> tp.Self:
        """
        Return shallow copy of this Series.
        """
        return self

    def _memory_label_component_pairs(
        self,
    ) -> tp.Iterable[tp.Tuple[str, tp.Any]]:
        return (('Name', self._name), ('Index', self._index), ('Values', self.values))

    # ---------------------------------------------------------------------------
    def __reversed__(self) -> tp.Iterator[TLabel]:
        """
        Returns a reverse iterator on the series' index.

        Returns:
            :obj:`Index`
        """
        return reversed(self._index)

    # ---------------------------------------------------------------------------
    # name interface

    @property
    @doc_inject()
    def name(self) -> TName:
        """{}"""
        return self._name

    def rename(
        self,
        name: TName = NAME_DEFAULT,
        /,
        *,
        index: TName = NAME_DEFAULT,
    ) -> tp.Self:
        """
        Return a new Series with an updated name attribute.
        """
        name = self.name if name is NAME_DEFAULT else name
        i = self._index if index is NAME_DEFAULT else self._index.rename(index)

        return self.__class__(
            self.values,
            index=i,
            name=name,
        )

    # ---------------------------------------------------------------------------
    # interfaces

    @property
    def loc(self) -> InterGetItemLocReduces[TSeriesAny, TVDtype]:
        """
        Interface for label-based selection.
        """
        return InterGetItemLocReduces(self._extract_loc)  # type: ignore

    @property
    def iloc(self) -> InterGetItemILocReduces[TSeriesAny, TVDtype]:
        """
        Interface for position-based selection.
        """
        return InterGetItemILocReduces(self._extract_iloc)

    @property
    def drop(self) -> InterfaceSelectTrio[TSeriesAny]:
        """
        Interface for dropping elements from :obj:`static_frame.Series`. This alway returns a `Series`.
        """
        return InterfaceSelectTrio(  # type: ignore
            func_iloc=self._drop_iloc,
            func_loc=self._drop_loc,
            func_getitem=self._drop_loc,
        )

    @property
    def mask(self) -> InterfaceSelectTrio[TSeriesAny]:
        """
        Interface for extracting Boolean :obj:`static_frame.Series`.
        """
        return InterfaceSelectTrio(  # type: ignore
            func_iloc=self._extract_iloc_mask,
            func_loc=self._extract_loc_mask,
            func_getitem=self._extract_loc_mask,
        )

    @property
    def masked_array(self) -> InterfaceSelectTrio[TSeriesAny]:
        """
        Interface for extracting NumPy Masked Arrays.
        """
        return InterfaceSelectTrio(  # type: ignore
            func_iloc=self._extract_iloc_masked_array,
            func_loc=self._extract_loc_masked_array,
            func_getitem=self._extract_loc_masked_array,
        )

    @property
    def assign(self) -> InterfaceAssignTrio['SeriesAssign']:
        """
        Interface for doing assignment-like selection and replacement.
        """
        # NOTE: this is not a InterfaceAssignQuartet, like on Frame
        return InterfaceAssignTrio(  # type: ignore
            func_iloc=self._extract_iloc_assign,
            func_loc=self._extract_loc_assign,
            func_getitem=self._extract_loc_assign,
            delegate=SeriesAssign,
        )

    # ---------------------------------------------------------------------------
    @property
    def via_values(self) -> InterfaceValues[TSeriesAny]:
        """
        Interface for applying functions to values (as arrays) in this container.
        """
        return InterfaceValues(self)

    @property
    def via_str(self) -> InterfaceString[TSeriesAny]:
        """
        Interface for applying string methods to elements in this container.
        """

        def blocks_to_container(blocks: tp.Iterator[TNDArrayAny]) -> TSeriesAny:
            return self.__class__(
                next(blocks),  # assume only one
                index=self._index,
                name=self._name,
                own_index=True,
            )

        return InterfaceString(
            blocks=(self.values,),
            blocks_to_container=blocks_to_container,
            ndim=self._NDIM,
            labels=range(1),
        )

    @property
    def via_dt(self) -> InterfaceDatetime[TSeriesAny]:
        """
        Interface for applying datetime properties and methods to elements in this container.
        """

        def blocks_to_container(blocks: tp.Iterator[TNDArrayAny]) -> TSeriesAny:
            return self.__class__(
                next(blocks),  # assume only one
                index=self._index,
                name=self._name,
                own_index=True,
            )

        return InterfaceDatetime(
            blocks=(self.values,),
            blocks_to_container=blocks_to_container,
        )

    def via_fill_value(
        self,
        fill_value: object = np.nan,
        /,
    ) -> InterfaceFillValue[TSeriesAny]:
        """
        Interface for using binary operators and methods with a pre-defined fill value.
        """
        return InterfaceFillValue(
            container=self,
            fill_value=fill_value,
        )

    def via_re(
        self,
        pattern: str,
        flags: int = 0,
        /,
    ) -> InterfaceRe[TSeriesAny]:
        """
        Interface for applying regular expressions to elements in this container.
        """

        def blocks_to_container(blocks: tp.Iterator[TNDArrayAny]) -> TSeriesAny:
            return self.__class__(
                next(blocks),  # assume only one
                index=self._index,
                name=self._name,
                own_index=True,
            )

        return InterfaceRe(
            blocks=(self.values,),
            blocks_to_container=blocks_to_container,
            pattern=pattern,
            flags=flags,
        )

    @property
    def via_mapping(self) -> SeriesMapping[tp.Any, TVDtype]:
        """
        Return a wrapper around Series data that fully implements the Python Mapping interface.
        """
        # NOTE: cannot type the key from the Series as the component type is wrapped in an Index; in the case of IndexHierarchy, the key type is a object (labels are tuples)
        return SeriesMapping(self)  # type: ignore [arg-type]

    # ---------------------------------------------------------------------------
    @property
    def iter_group(self) -> IterNodeGroup[TSeriesAny]:
        """
        Iterator of :obj:`Series`, where each :obj:`Series` matches unique values.
        """
        return IterNodeGroup(
            container=self,
            function_items=partial(self._axis_group_items, group_source=self.values),
            function_values=partial(self._axis_group, group_source=self.values),
            yield_type=IterNodeType.VALUES,
            apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_VALUES,
        )

    @property
    def iter_group_items(self) -> IterNodeGroup[TSeriesAny]:
        return IterNodeGroup(
            container=self,
            function_items=partial(self._axis_group_items, group_source=self.values),
            function_values=partial(self._axis_group, group_source=self.values),
            yield_type=IterNodeType.ITEMS,
            apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_VALUES,
        )

    # ---------------------------------------------------------------------------
    @property
    def iter_group_array(self) -> IterNodeGroup[TSeriesAny]:
        """
        Iterator of :obj:`Series`, where each :obj:`Series` matches unique values.
        """
        return IterNodeGroup(
            container=self,
            function_items=partial(
                self._axis_group_items, as_array=True, group_source=self.values
            ),
            function_values=partial(
                self._axis_group, as_array=True, group_source=self.values
            ),
            yield_type=IterNodeType.VALUES,
            apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_VALUES,
        )

    @property
    def iter_group_array_items(self) -> IterNodeGroup[TSeriesAny]:
        return IterNodeGroup(
            container=self,
            function_items=partial(
                self._axis_group_items, group_source=self.values, as_array=True
            ),
            function_values=partial(
                self._axis_group, group_source=self.values, as_array=True
            ),
            yield_type=IterNodeType.ITEMS,
            apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_VALUES,
        )

    # ---------------------------------------------------------------------------
    @property
    def iter_group_labels(self) -> IterNodeDepthLevel[TSeriesAny]:
        return IterNodeDepthLevel(
            container=self,
            function_items=self._axis_group_labels_items,
            function_values=self._axis_group_labels,
            yield_type=IterNodeType.VALUES,
            apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_LABELS,
        )

    @property
    def iter_group_labels_items(self) -> IterNodeDepthLevel[TSeriesAny]:
        return IterNodeDepthLevel(
            container=self,
            function_items=self._axis_group_labels_items,
            function_values=self._axis_group_labels,
            yield_type=IterNodeType.ITEMS,
            apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_LABELS,
        )

    # ---------------------------------------------------------------------------
    @property
    def iter_group_labels_array(self) -> IterNodeDepthLevel[TSeriesAny]:
        return IterNodeDepthLevel(
            container=self,
            function_items=partial(self._axis_group_labels_items, as_array=True),
            function_values=partial(self._axis_group_labels, as_array=True),
            yield_type=IterNodeType.VALUES,
            apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_LABELS,
        )

    @property
    def iter_group_labels_array_items(self) -> IterNodeDepthLevel[TSeriesAny]:
        return IterNodeDepthLevel(
            container=self,
            function_items=partial(self._axis_group_labels_items, as_array=True),
            function_values=partial(self._axis_group_labels, as_array=True),
            yield_type=IterNodeType.ITEMS,
            apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_LABELS,
        )

    # ---------------------------------------------------------------------------
    @property
    def iter_group_other(
        self,
    ) -> IterNodeGroupOther[TSeriesAny]:
        """
        Iterator of :obj:`Series`, grouped by unique values found in the passed container.
        """
        return IterNodeGroupOther(
            container=self,
            function_items=self._axis_group_items,
            function_values=self._axis_group,
            yield_type=IterNodeType.VALUES,
            apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_VALUES,
        )

    @property
    def iter_group_other_items(
        self,
    ) -> IterNodeGroupOther[TSeriesAny]:
        """
        Iterator of pairs of label, :obj:`Series`, grouped by unique values found in the passed container.
        """
        return IterNodeGroupOther(
            container=self,
            function_items=self._axis_group_items,
            function_values=self._axis_group,
            yield_type=IterNodeType.ITEMS,
            apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_VALUES,
        )

    # ---------------------------------------------------------------------------
    @property
    def iter_group_other_array(self) -> IterNodeGroupOther[TSeriesAny]:
        return IterNodeGroupOther(
            container=self,
            function_items=partial(self._axis_group_items, as_array=True),
            function_values=partial(self._axis_group, as_array=True),
            yield_type=IterNodeType.VALUES,
            apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_VALUES,
        )

    @property
    def iter_group_other_array_items(self) -> IterNodeGroupOther[TSeriesAny]:
        return IterNodeGroupOther(
            container=self,
            function_items=partial(self._axis_group_items, as_array=True),
            function_values=partial(self._axis_group, as_array=True),
            yield_type=IterNodeType.ITEMS,
            apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_VALUES,
        )

    # ---------------------------------------------------------------------------
    @property
    def iter_element(self) -> IterNodeNoArgMapable[TSeriesAny]:
        """
        Iterator of elements.
        """
        return IterNodeNoArgMapable(
            container=self,
            function_items=self._axis_element_items,
            function_values=self._axis_element,
            yield_type=IterNodeType.VALUES,
            apply_type=IterNodeApplyType.SERIES_VALUES,
        )

    @property
    def iter_element_items(self) -> IterNodeNoArgMapable[TSeriesAny]:
        """
        Iterator of label, element pairs.
        """
        return IterNodeNoArgMapable(
            container=self,
            function_items=self._axis_element_items,
            function_values=self._axis_element,
            yield_type=IterNodeType.ITEMS,
            apply_type=IterNodeApplyType.SERIES_VALUES,
        )

    # ---------------------------------------------------------------------------
    @property
    def iter_window(self) -> IterNodeWindow[TSeriesAny]:
        function_values = partial(self._axis_window, as_array=False)
        function_items = partial(self._axis_window_items, as_array=False)
        return IterNodeWindow(
            container=self,
            function_values=function_values,
            function_items=function_items,
            yield_type=IterNodeType.VALUES,
            apply_type=IterNodeApplyType.SERIES_ITEMS,
        )

    @property
    def iter_window_items(self) -> IterNodeWindow[TSeriesAny]:
        function_values = partial(self._axis_window, as_array=False)
        function_items = partial(self._axis_window_items, as_array=False)
        return IterNodeWindow(
            container=self,
            function_values=function_values,
            function_items=function_items,
            yield_type=IterNodeType.ITEMS,
            apply_type=IterNodeApplyType.SERIES_ITEMS,
        )

    @property
    def iter_window_array(self) -> IterNodeWindow[TSeriesAny]:
        function_values = partial(self._axis_window, as_array=True)
        function_items = partial(self._axis_window_items, as_array=True)
        return IterNodeWindow(
            container=self,
            function_values=function_values,
            function_items=function_items,
            yield_type=IterNodeType.VALUES,
            apply_type=IterNodeApplyType.SERIES_ITEMS,
        )

    @property
    def iter_window_array_items(self) -> IterNodeWindow[TSeriesAny]:
        function_values = partial(self._axis_window, as_array=True)
        function_items = partial(self._axis_window_items, as_array=True)
        return IterNodeWindow(
            container=self,
            function_values=function_values,
            function_items=function_items,
            yield_type=IterNodeType.ITEMS,
            apply_type=IterNodeApplyType.SERIES_ITEMS,
        )

    # ---------------------------------------------------------------------------
    # index manipulation

    def _reindex_other_like_iloc(
        self,
        value: TSeriesAny,
        iloc_key: TILocSelector,
        fill_value: tp.Any = np.nan,
    ) -> TSeriesAny:
        """Given a value that is a Series, reindex that Series argument to the index components, drawn from this Series, that are specified by the iloc_key. This means that this returns a new Series that corresponds to the index of this Series based on the iloc selection."""
        iloc_many: TILocSelectorMany
        if isinstance(iloc_key, INT_TYPES):
            iloc_many = [iloc_key]  # type: ignore[list-item]
        else:
            iloc_many = iloc_key
        return value.reindex(self._index._extract_iloc(iloc_many), fill_value=fill_value)

    @doc_inject(selector='reindex', class_name='Series')
    def reindex(
        self,
        index: TIndexInitializer,
        *,
        fill_value: tp.Any = np.nan,
        own_index: bool = False,
        check_equals: bool = True,
    ) -> tp.Self:
        """
        {doc}

        Args:
            index: {index_initializer}
            columns: {index_initializer}
            {fill_value}
            {own_index}
        """
        index_owned: IndexBase
        if own_index:
            index_owned = index  # type: ignore
        else:
            index_owned = index_from_optional_constructor(
                index, default_constructor=Index
            )

        # NOTE: it is assumed that the equals comparison is faster than continuing with this method
        if check_equals and self._index.equals(index_owned):
            # if labels are equal (even if a different Index subclass), we can simply use the new Index
            return self.__class__(
                self.values, index=index_owned, own_index=True, name=self._name
            )

        ic = IndexCorrespondence.from_correspondence(self._index, index_owned)
        if not ic.size:
            # NOTE: take slice to ensure same type of index and array
            return self._extract_iloc(EMPTY_SLICE)

        if ic.is_subset:  # must have some common
            values = self.values[ic.iloc_src]
            values.flags.writeable = False
            return self.__class__(
                values, index=index_owned, own_index=True, name=self._name
            )

        values_src = self.values
        if is_fill_value_factory_initializer(fill_value):
            fv = get_col_fill_value_factory(fill_value, None)(0, values_src.dtype)
        else:
            fv = fill_value

        values = full_for_fill(values_src.dtype, len(index_owned), fv)
        assign_via_ic(ic, values_src, values)
        assert not values.flags.writeable

        return self.__class__(values, index=index_owned, own_index=True, name=self._name)

    @doc_inject(selector='relabel', class_name='Series')
    def relabel(
        self,
        index: tp.Optional[TRelabelInput],
        *,
        index_constructor: TIndexCtorSpecifier = None,
    ) -> tp.Self:
        """
        {doc}

        Args:
            index: {relabel_input_index}
        """
        own_index = False
        index_init: TIndexInitializer | None
        if index is IndexAutoFactory:
            index_init = None
        elif index is None:
            index_init = self._index
            own_index = index_constructor is None
        elif is_callable_or_mapping(index):
            index_init = self._index.relabel(index)
            own_index = index_constructor is None
        elif isinstance(index, Set):
            raise RelabelInvalid()
        else:
            index_init = index  # type: ignore

        return self.__class__(
            self.values,
            index=index_init,
            index_constructor=index_constructor,
            own_index=own_index,
            name=self._name,
        )

    @doc_inject(selector='relabel_flat', class_name='Series')
    def relabel_flat(self) -> tp.Self:
        """
        {doc}
        """
        if not isinstance(self._index, IndexHierarchy):
            raise RuntimeError('cannot flatten an Index that is not an IndexHierarchy')

        return self.__class__(
            self.values,
            index=self._index.flat(),
            name=self._name,
        )

    @doc_inject(selector='relabel_level_add', class_name='Series')
    def relabel_level_add(
        self,
        level: TLabel,
        /,
        *,
        index_constructor: TIndexCtorSpecifier = None,
    ) -> tp.Self:
        """
        {doc}

        Args:
            level: {level}
        """
        return self.__class__(
            self.values,
            index=self._index.level_add(level, index_constructor=index_constructor),
            name=self._name,
        )

    @doc_inject(selector='relabel_level_drop', class_name='Series')
    def relabel_level_drop(
        self,
        count: int = 1,
        /,
    ) -> tp.Self:
        """
        {doc}

        Args:
            count: {count}
        """
        if not isinstance(self._index, IndexHierarchy):
            raise RuntimeError(
                'cannot drop level of an Index that is not an IndexHierarchy'
            )

        return self.__class__(
            self.values,
            index=self._index.level_drop(count),
            name=self._name,
        )

    def rehierarch(
        self,
        depth_map: tp.Sequence[int],
        /,
        *,
        index_constructors: TIndexCtorSpecifiers = None,
    ) -> tp.Self:
        """
        Return a new :obj:`Series` with new a hierarchy based on the supplied ``depth_map``.
        """
        if self._index.depth == 1:
            raise RuntimeError('cannot rehierarch when there is no hierarchy')

        index, iloc_map = rehierarch_from_index_hierarchy(
            labels=self._index,  # type: ignore
            depth_map=depth_map,
            index_constructors=index_constructors,
            name=self._index.name,
        )
        values = self.values[iloc_map]
        values.flags.writeable = False
        return self.__class__(
            values,
            index=index,
            name=self._name,
        )

    # ---------------------------------------------------------------------------
    # na handling

    def isna(self) -> tp.Self:
        """
        Return a same-indexed, Boolean :obj:`Series` indicating which values are NaN or None.
        """
        values = isna_array(self.values)
        values.flags.writeable = False
        return self.__class__(values, index=self._index, own_index=True)

    def notna(self) -> tp.Self:
        """
        Return a same-indexed, Boolean :obj:`Series` indicating which values are NaN or None.
        """
        values = np.logical_not(isna_array(self.values))
        values.flags.writeable = False
        return self.__class__(values, index=self._index, own_index=True)

    def dropna(self) -> tp.Self:
        """
        Return a new :obj:`Series` after removing values of NaN or None.
        """
        if self.values.dtype.kind not in DTYPE_NA_KINDS:
            # return the same array in a new series
            return self.__class__(
                self.values, index=self._index, name=self._name, own_index=True
            )

        # get positions that we want to keep
        isna = isna_array(self.values)
        length = len(self.values)
        count = isna.sum()

        if count == length:  # all are NaN
            return self.__class__(
                (),
                name=self._name,
                index=self._index[[]],
                own_index=True,
            )
        if count == 0:  # None are nan
            return self

        sel = np.logical_not(isna)
        values = self.values[sel]
        values.flags.writeable = False

        return self.__class__(
            values,
            index=self._index._extract_iloc(
                sel
            ),  # PERF: use _extract_iloc as we have a Boolean array
            name=self._name,
            own_index=True,
        )

    # ---------------------------------------------------------------------------
    # falsy handling

    def isfalsy(self) -> tp.Self:
        """
        Return a same-indexed, Boolean :obj:`Series` indicating which values are falsy.
        """
        values = isfalsy_array(self.values)
        values.flags.writeable = False
        return self.__class__(values, index=self._index, own_index=True)

    def notfalsy(self) -> tp.Self:
        """
        Return a same-indexed, Boolean :obj:`Series` indicating which values are falsy.
        """
        values = np.logical_not(isfalsy_array(self.values))
        values.flags.writeable = False
        return self.__class__(values, index=self._index, own_index=True)

    def dropfalsy(self) -> tp.Self:
        """
        Return a new :obj:`Series` after removing values of falsy.
        """
        # get positions that we want to keep
        isfalsy = isfalsy_array(self.values)
        length = len(self.values)
        count = isfalsy.sum()

        if count == length:  # all are falsy
            return self.__class__((), name=self.name)
        if count == 0:  # None are falsy
            return self.__class__(
                self.values, index=self._index, name=self._name, own_index=True
            )

        sel = np.logical_not(isfalsy)
        values = self.values[sel]
        values.flags.writeable = False

        return self.__class__(
            values,
            index=self._index._extract_iloc(
                sel
            ),  # PERF: use _extract_iloc as we have a Boolean array
            name=self._name,
            own_index=True,
        )

    # ---------------------------------------------------------------------------
    # na filling

    def _fill_missing(
        self,
        value: tp.Any,  # an element or a Series
        func: tp.Callable[[TNDArrayAny], TNDArrayAny],
    ) -> tp.Self:
        """
        Args:
            func: A function that returns a same-shaped array of Booleans.

        """
        values = self.values
        sel = func(values)
        if not np.any(sel):
            return self

        if hasattr(value, '__iter__') and not isinstance(value, STRING_TYPES):
            if not isinstance(value, Series):
                raise RuntimeError(
                    'unlabeled iterables cannot be used for fillna: use a Series'
                )
            value_dtype = value.dtype
            # choose a fill value that will not force a type coercion
            fill_value = dtype_to_fill_value(value_dtype)
            # find targets that are NaN in self and have labels in value; otherwise, might fill values after reindexing, and end up filling a fill_value rather than keeping original (na) value
            labels_common = intersect1d(self.index.values[sel], value.index.values)
            sel = self.index.isin(labels_common)
            if not np.any(sel):  # avoid copying, retyping
                return self

            # must reindex to align ordering; just get array
            value = self._reindex_other_like_iloc(
                value, sel, fill_value=fill_value
            ).values
        else:
            value_dtype = dtype_from_element(value)

        assignable_dtype = resolve_dtype(value_dtype, values.dtype)

        # assigned = assign_via_mask(values, assignable_dtype, sel, value)
        if values.dtype == assignable_dtype:
            assigned = values.copy()
            assigned[sel] = value
        else:
            assigned = astype_array(values, assignable_dtype)
            assigned[sel] = value
        assigned.flags.writeable = False

        return self.__class__(
            assigned,
            index=self._index,
            name=self._name,
            own_index=True,
        )

    @doc_inject(selector='fillna')
    def fillna(
        self,
        value: tp.Any,  # an element or a Series
        /,
    ) -> tp.Self:
        """Return a new :obj:`Series` after replacing NA (NaN or None) with the supplied value. The ``value`` can be an element or :obj:`Series`.

        Args:
            {value}
        """
        return self._fill_missing(value, isna_array)

    @doc_inject(selector='fillna')
    def fillfalsy(
        self,
        value: tp.Any,  # an element or a Series
        /,
    ) -> tp.Self:
        """Return a new :obj:`Series` after replacing falsy values with the supplied value. The ``value`` can be an element or :obj:`Series`.

        Args:
            {value}
        """
        return self._fill_missing(value, isfalsy_array)

    # ---------------------------------------------------------------------------
    @staticmethod
    def _fill_missing_directional(
        array: TNDArrayAny,
        directional_forward: bool,
        func_target: TUFunc,
        limit: int = 0,
    ) -> TNDArrayAny:
        """Return a new :obj:`Series` after feeding forward the last non-null (NaN or None) observation across contiguous nulls.

        Args:
            count: Set the limit of nan values to be filled per nan region. A value of 0 is equivalent to no limit.
            func_target: the function to use to identify fill targets
        """
        # sel = isna_array(array)
        sel = func_target(array)
        if not np.any(sel):
            return array

        def slice_condition(target_slice: slice) -> bool:
            # NOTE: start is never None
            return sel[target_slice.start]  # type: ignore

        # type is already compatible, no need for check
        assigned = array.copy()
        target_index = binary_transition(sel)
        target_values = array[target_index]
        length = len(array)

        for target_slice, value in slices_from_targets(
            target_index=target_index,
            target_values=target_values,
            length=length,
            directional_forward=directional_forward,
            limit=limit,
            slice_condition=slice_condition,  # isna True in region
        ):
            assigned[target_slice] = value

        assigned.flags.writeable = False
        return assigned

    @doc_inject(selector='fillna')
    def fillna_forward(
        self,
        limit: int = 0,
        /,
    ) -> tp.Self:
        """Return a new :obj:`Series` after feeding forward the last non-null (NaN or None) observation across contiguous nulls.

        Args:
            {limit}
        """
        return self.__class__(
            self._fill_missing_directional(
                array=self.values,
                directional_forward=True,
                func_target=isna_array,
                limit=limit,
            ),
            index=self._index,
            name=self._name,
        )

    @doc_inject(selector='fillna')
    def fillna_backward(
        self,
        limit: int = 0,
        /,
    ) -> tp.Self:
        """Return a new :obj:`Series` after feeding backward the last non-null (NaN or None) observation across contiguous nulls.

        Args:
            {limit}
        """
        return self.__class__(
            self._fill_missing_directional(
                array=self.values,
                directional_forward=False,
                func_target=isna_array,
                limit=limit,
            ),
            index=self._index,
            name=self._name,
        )

    @doc_inject(selector='fillna')
    def fillfalsy_forward(
        self,
        limit: int = 0,
        /,
    ) -> tp.Self:
        """Return a new :obj:`Series` after feeding forward the last non-falsy observation across contiguous falsy values.

        Args:
            {limit}
        """
        return self.__class__(
            self._fill_missing_directional(
                array=self.values,
                directional_forward=True,
                func_target=isfalsy_array,
                limit=limit,
            ),
            index=self._index,
            name=self._name,
        )

    @doc_inject(selector='fillna')
    def fillfalsy_backward(
        self,
        limit: int = 0,
        /,
    ) -> tp.Self:
        """Return a new :obj:`Series` after feeding backward the last non-falsy observation across contiguous falsy values.

        Args:
            {limit}
        """
        return self.__class__(
            self._fill_missing_directional(
                array=self.values,
                directional_forward=False,
                func_target=isfalsy_array,
                limit=limit,
            ),
            index=self._index,
            name=self._name,
        )

    @staticmethod
    def _fill_missing_sided(
        array: TNDArrayAny,
        value: tp.Any,
        sided_leading: bool,
        func_target: TUFunc,
    ) -> TNDArrayAny:
        """
        Args:
            sided_leading: True sets the side to fill is the leading side; False sets the side to fill to the trailiing side.
        """
        sel = func_target(array)

        if not np.any(sel):
            return array

        sided_index = 0 if sided_leading else -1

        if not sel[sided_index]:
            # sided value is not null: nothing to do
            return array  # assume immutable

        if value.__class__ is np.ndarray:
            raise RuntimeError('cannot assign an array to fillna')

        assignable_dtype = resolve_dtype(dtype_from_element(value), array.dtype)

        if array.dtype == assignable_dtype:
            assigned = array.copy()
        else:
            assigned = astype_array(array, assignable_dtype)

        ft = first_true_1d(~sel, forward=sided_leading)
        if ft != -1:
            if sided_leading:
                sel_slice = slice(0, ft)
            else:  # trailing
                sel_slice = slice(ft + 1, None)
        else:
            sel_slice = NULL_SLICE

        assigned[sel_slice] = value
        assigned.flags.writeable = False
        return assigned

    @doc_inject(selector='fillna')
    def fillna_leading(
        self,
        value: tp.Any,
        /,
    ) -> tp.Self:
        """Return a new :obj:`Series` after filling leading (and only leading) null (NaN or None) with the supplied value.

        Args:
            {value}
        """
        return self.__class__(
            self._fill_missing_sided(
                array=self.values,
                value=value,
                func_target=isna_array,
                sided_leading=True,
            ),
            index=self._index,
            name=self._name,
        )

    @doc_inject(selector='fillna')
    def fillna_trailing(
        self,
        value: tp.Any,
        /,
    ) -> tp.Self:
        """Return a new :obj:`Series` after filling trailing (and only trailing) null (NaN or None) with the supplied value.

        Args:
            {value}
        """
        return self.__class__(
            self._fill_missing_sided(
                array=self.values,
                value=value,
                func_target=isna_array,
                sided_leading=False,
            ),
            index=self._index,
            name=self._name,
        )

    @doc_inject(selector='fillna')
    def fillfalsy_leading(
        self,
        value: tp.Any,
        /,
    ) -> tp.Self:
        """Return a new :obj:`Series` after filling leading (and only leading) falsy values with the supplied value.

        Args:
            {value}
        """
        return self.__class__(
            self._fill_missing_sided(
                array=self.values,
                value=value,
                func_target=isfalsy_array,
                sided_leading=True,
            ),
            index=self._index,
            name=self._name,
        )

    @doc_inject(selector='fillna')
    def fillfalsy_trailing(
        self,
        value: tp.Any,
        /,
    ) -> tp.Self:
        """Return a new :obj:`Series` after filling trailing (and only trailing) falsy values with the supplied value.

        Args:
            {value}
        """
        return self.__class__(
            self._fill_missing_sided(
                array=self.values,
                value=value,
                func_target=isfalsy_array,
                sided_leading=False,
            ),
            index=self._index,
            name=self._name,
        )

    # ---------------------------------------------------------------------------
    # operators

    def _ufunc_unary_operator(self, operator: TUFunc) -> tp.Self:
        """
        For unary operations, the `name` attribute propagates.
        """
        values = operator(self.values)
        return self.__class__(
            values,
            index=self._index,
            dtype=values.dtype,  # some operators might change the dtype
            name=self._name,
        )

    def _ufunc_binary_operator(
        self,
        *,
        operator: TUFunc,
        other: tp.Any,
        axis: int = 0,
        fill_value: tp.Any = np.nan,
    ) -> tp.Self:
        """
        For binary operations, the `name` attribute does not propagate unless other is a scalar.
        """
        # get both reverse and regular
        if operator.__name__ == 'matmul':
            return matmul(self, other)  # type: ignore
        elif operator.__name__ == 'rmatmul':
            return matmul(other, self)  # type: ignore

        values = self.values
        index = self._index
        other_is_array = False

        if isinstance(other, Series):
            name = None
            other_is_array = True
            if not self._index.equals(other._index):
                # if not equal, create a new Index by forming the union
                index = self._index.union(other._index)
                # now need to reindex the Series
                values = self.reindex(
                    index,
                    own_index=True,
                    check_equals=False,
                    fill_value=fill_value,
                ).values
                other = other.reindex(
                    index,
                    own_index=True,
                    check_equals=False,
                    fill_value=fill_value,
                ).values
            else:
                other = other.values
        elif other.__class__ is np.ndarray:
            name = None
            other_is_array = True
            if other.ndim > 1:
                raise NotImplementedError(
                    'Operator application to greater dimensionalities will result in an array with more than 1 dimension.'
                )
        elif other.__class__ is InterfaceFillValue:
            raise RuntimeError(
                'via_fill_value interfaces can only be used on the left-hand side of binary expressions.'
            )
        else:
            name = self._name

        result = apply_binary_operator(
            values=values,
            other=other,
            other_is_array=other_is_array,
            operator=operator,
        )
        return self.__class__(result, index=index, name=name)

    def _ufunc_axis_skipna(
        self,
        *,
        axis: int,
        skipna: bool,
        ufunc: TUFunc,
        ufunc_skipna: TUFunc,
        composable: bool,
        dtypes: tp.Tuple[TDtypeAny, ...],
        size_one_unity: bool,
    ) -> tp.Any:
        """
        For a Series, all functions of this type reduce the single axis of the Series to a single element, so Index has no use here.

        Args:
            dtype: not used, part of signature for a common interface
        """
        return array_ufunc_axis_skipna(
            array=self.values,
            skipna=skipna,
            axis=0,
            ufunc=ufunc,
            ufunc_skipna=ufunc_skipna,
        )

    def _ufunc_shape_skipna(
        self,
        *,
        axis: int,
        skipna: bool,
        ufunc: TUFunc,
        ufunc_skipna: TUFunc,
        composable: bool,
        dtypes: tp.Tuple[TDtypeAny, ...],
        size_one_unity: bool,
    ) -> tp.Self:
        """
        NumPy ufunc proccessors that retain the shape of the processed.

        Args:
            dtypes: not used, part of signature for a common interface
        """
        values = array_ufunc_axis_skipna(
            array=self.values,
            skipna=skipna,
            axis=0,
            ufunc=ufunc,
            ufunc_skipna=ufunc_skipna,
        )
        values.flags.writeable = False
        return self.__class__(values, index=self._index)

    # ---------------------------------------------------------------------------
    def __len__(self) -> int:
        """Length of values."""
        return self.values.__len__()

    def _display(
        self,
        config: DisplayConfig,
        *,
        display_cls: Display,
        style_config: tp.Optional[StyleConfig] = None,
    ) -> Display:
        """
        Private display interface to be shared by Bus and Series.
        """
        index_depth = self._index.depth if config.include_index else 0
        display_index = self._index.display(config)

        # When showing type we need 2: one for the Series type, the other for the index type.
        header_depth = 2 * config.type_show

        # create an empty display based on index display
        d = Display(
            [list() for _ in range(len(display_index))],
            config=config,
            outermost=True,
            index_depth=index_depth,
            header_depth=header_depth,
            style_config=style_config,
        )

        if config.include_index:
            d.extend_display(display_index)
            header_values = '' if config.type_show else None
        else:
            header_values = None

        d.extend_display(
            Display.from_values(self.values, header=header_values, config=config)
        )

        if config.type_show:
            d.insert_displays(display_cls.flatten())

        return d

    @doc_inject()
    def display(
        self,
        config: tp.Optional[DisplayConfig] = None,
        /,
        *,
        style_config: tp.Optional[StyleConfig] = None,
    ) -> Display:
        """{doc}

        Args:
            {config}
        """
        config = config or DisplayActive.get()
        display_cls = Display.from_values(
            (), header=DisplayHeader(self.__class__, self._name), config=config
        )
        return self._display(
            config,
            display_cls=display_cls,
            style_config=style_config,
        )

    # ---------------------------------------------------------------------------
    # common attributes from the numpy array

    @property
    @doc_inject()
    def mloc(self) -> int:
        """{doc_int}"""
        return mloc(self.values)

    @property
    def dtype(self) -> TDtypeAny:
        """
        Return the dtype of the underlying NumPy array.

        Returns:
            :obj:`numpy.dtype`
        """
        dt: TDtypeAny = self.values.dtype
        return dt

    @property
    def shape(self) -> tp.Tuple[int]:
        """
        Return a tuple describing the shape of the underlying NumPy array.

        Returns:
            :obj:`Tuple[int]`
        """
        return self.values.shape  # type: ignore

    @property
    def ndim(self) -> int:
        """
        Return the number of dimensions, which for a `Series` is always 1.

        Returns:
            :obj:`int`
        """
        return self._NDIM

    @property
    def size(self) -> int:
        """
        Return the size of the underlying NumPy array.

        Returns:
            :obj:`int`
        """
        return self.values.size

    @property
    def nbytes(self) -> int:
        """
        Return the total bytes of the underlying NumPy array.

        Returns:
            :obj:`int`
        """
        return self.values.nbytes

    # ---------------------------------------------------------------------------
    # extraction

    # def _extract_array(self, key: TLocSelector) -> TNDArrayAny:
    #     return self.values[key]

    @tp.overload
    def _extract_iloc(self, key: TILocSelectorMany) -> tp.Self: ...

    @tp.overload
    def _extract_iloc(self, key: TILocSelectorOne) -> tp.Any: ...

    def _extract_iloc(self, key: TILocSelector) -> tp.Any:
        values = self.values[key]  # let `IndexError` propagate
        if isinstance(key, INT_TYPES):  # if we have a single element
            return values

        return self.__class__(values, index=self._index.iloc[key], name=self._name)

    @tp.overload
    def _extract_loc(self, key: TLocSelectorMany) -> tp.Self: ...

    @tp.overload
    def _extract_loc(self, key: TLabel) -> tp.Any: ...

    def _extract_loc(self, key: TLocSelector) -> tp.Any:
        """
        Compatibility:
            Pandas supports taking in iterables of keys, where some keys are not found in the index; a Series is returned as if a reindex operation was performed. This is undesirable. Better instead is to use reindex()
        """
        iloc_key = self._index._loc_to_iloc(key)
        try:
            return self._extract_iloc(iloc_key)
        except IndexError:
            raise KeyError(key) from None

    @tp.overload
    def __getitem__(self, key: TLocSelectorMany) -> tp.Self: ...

    @tp.overload
    def __getitem__(self, key: TLabel) -> TVDtype: ...

    @doc_inject(selector='selector')
    def __getitem__(self, key: TLocSelector) -> tp.Any:
        """Selector of values by label.

        Args:
            key: {key_loc}

        Compatibility:
            Pandas supports using both loc and iloc style selections with the __getitem__ interface on Series. This is undesirable, so here we only expose the loc interface (making the Series dictionary like, but unlike the Index, where __getitem__ is an iloc).
        """
        return self._extract_loc(key)

    def __setitem__(self, key: TLabel, value: tp.Any) -> None:
        raise immutable_type_error_factory(self.__class__, '', key, value)

    # ---------------------------------------------------------------------------
    # utilities for alternate extraction: drop, mask and assignment

    def _drop_iloc(self, key: TILocSelector) -> tp.Self:
        if key.__class__ is np.ndarray and key.dtype == bool:  # type: ignore
            # use Boolean array to select indices from Index positions, as np.delete does not work with arrays
            values = np.delete(self.values, self._index.positions[key])
        else:
            values = np.delete(self.values, key)  # type: ignore
        values.flags.writeable = False

        index = self._index._drop_iloc(key)

        return self.__class__(values, index=index, name=self._name, own_index=True)

    def _drop_loc(self, key: TLocSelector) -> tp.Self:
        return self._drop_iloc(self._index._loc_to_iloc(key))

    # ---------------------------------------------------------------------------

    def _extract_iloc_mask(self, key: TILocSelector) -> tp.Self:
        """Produce a new boolean Series of the same shape, where the values selected via iloc selection are True. The `name` attribute is not propagated."""
        mask = np.full(self.values.shape, False, dtype=bool)
        mask[key] = True
        mask.flags.writeable = False
        return self.__class__(mask, index=self._index)

    def _extract_loc_mask(self, key: TLocSelector) -> tp.Self:
        """Produce a new boolean Series of the same shape, where the values selected via loc selection are True. The `name` attribute is not propagated."""
        iloc_key = self._index._loc_to_iloc(key)
        return self._extract_iloc_mask(key=iloc_key)

    # ---------------------------------------------------------------------------

    def _extract_iloc_masked_array(
        self, key: TILocSelector
    ) -> MaskedArray[tp.Any, tp.Any]:
        """Produce a new boolean Series of the same shape, where the values selected via iloc selection are True."""
        mask = self._extract_iloc_mask(key=key)
        return MaskedArray(data=self.values, mask=mask.values)  # type: ignore

    def _extract_loc_masked_array(self, key: TLocSelector) -> MaskedArray[tp.Any, tp.Any]:
        """Produce a new boolean Series of the same shape, where the values selected via loc selection are True."""
        iloc_key = self._index._loc_to_iloc(key)
        return self._extract_iloc_masked_array(key=iloc_key)

    # ---------------------------------------------------------------------------

    def _extract_iloc_assign(self, key: TILocSelector) -> 'SeriesAssign':
        return SeriesAssign(self, key)

    def _extract_loc_assign(self, key: TLocSelector) -> 'SeriesAssign':
        iloc_key = self._index._loc_to_iloc(key)
        return SeriesAssign(self, iloc_key)

    # ---------------------------------------------------------------------------
    # axis functions

    @tp.overload
    def _axis_group_items(
        self,
        *,
        axis: int,
        as_array: tp.Literal[True],
        group_source: TNDArrayAny,
    ) -> tp.Iterator[tp.Tuple[TLabel, TNDArrayAny]]: ...

    @tp.overload
    def _axis_group_items(
        self,
        *,
        axis: int,
        as_array: tp.Literal[False],
        group_source: TNDArrayAny,
    ) -> tp.Iterator[tp.Tuple[TLabel, Series[TVIndex, TVDtype]]]: ...

    @tp.overload
    def _axis_group_items(
        self,
        *,
        axis: int,
        as_array: bool,
        group_source: TNDArrayAny,
    ) -> tp.Iterator[tp.Tuple[TLabel, Series[TVIndex, TVDtype]]]: ...

    def _axis_group_items(
        self,
        *,
        axis: int = 0,
        as_array: tp.Literal[True, False] = False,
        group_source: TNDArrayAny,
    ) -> tp.Iterator[tp.Tuple[TLabel, Series[TVIndex, TVDtype] | TNDArrayAny]]:
        """
        Args:
            group_source: Array to use to discovery groups; can be self.values to grouping on contained values.
        """
        if axis != 0:
            raise AxisInvalid(f'invalid axis {axis}')
        # NOTE: this could be optimized with a sorting-based apporach when possible
        groups, locations = array_to_groups_and_locations(group_source)

        func = self.values.__getitem__ if as_array else self._extract_iloc

        for idx, g in enumerate(groups):
            selection = locations == idx
            yield g, func(selection)

    def _axis_group(
        self,
        *,
        axis: int = 0,
        as_array: tp.Literal[True, False] = False,
        group_source: TNDArrayAny,
    ) -> tp.Iterator[TSeriesAny]:
        yield from (
            x
            for _, x in self._axis_group_items(
                axis=axis,
                as_array=as_array,
                group_source=group_source,
            )
        )

    def _axis_element_items(
        self,
    ) -> tp.Iterator[tp.Tuple[TLabel, tp.Any]]:
        """Generator of index, value pairs, equivalent to Series.items(). Repeated to have a common signature as other axis functions."""
        yield from zip(self._index.__iter__(), self.values)

    def _axis_element(
        self,
    ) -> tp.Iterator[tp.Any]:
        yield from self.values

    def _axis_group_labels_items(
        self,
        depth_level: tp.Optional[TDepthLevel] = None,
        *,
        as_array: bool = False,
    ) -> tp.Iterator[tp.Tuple[TLabel, TNDArrayAny | Series[TVIndex, TVDtype]]]:
        if depth_level is None:
            depth_level = 0

        func = self.values.__getitem__ if as_array else self._extract_iloc

        if self._index._NDIM == 1:
            for idx, key in enumerate(self._index):
                yield key, func([idx])
            return

        values = self._index.values_at_depth(depth_level)
        group_to_tuple = values.ndim > 1

        if self._index._check_sort_status_at_depth(depth_level):  # type: ignore
            group_source = self._index._indexers[depth_level]  # type: ignore
            if group_to_tuple:
                group_source = group_source.T

            transition_slices, _ = transition_slices_from_group(group_source)

            for slc in transition_slices:
                group = values[slc.start]

                if group_to_tuple:
                    group = tuple(group)

                yield group, func(slc)  # pyright: ignore[reportReturnType]
        else:
            groups, locations = array_to_groups_and_locations(values)

            for idx, g in enumerate(groups):
                selection = locations == idx
                if group_to_tuple:
                    g = tuple(g)
                yield g, func(selection)

    def _axis_group_labels(
        self,
        depth_level: TDepthLevel = 0,
        *,
        as_array: bool = False,
    ) -> tp.Iterator[TNDArrayAny | Series[TVIndex, TVDtype]]:
        yield from (
            x
            for _, x in self._axis_group_labels_items(
                depth_level=depth_level,
                as_array=as_array,
            )
        )

    def _axis_window_items(
        self,
        *,
        size: int,
        axis: int = 0,
        step: int = 1,
        window_sized: bool = True,
        window_func: tp.Optional[TCallableAny] = None,
        window_valid: tp.Optional[TCallableAny] = None,
        label_shift: int = 0,
        label_missing_skips: bool = True,
        label_missing_raises: bool = False,
        start_shift: int = 0,
        size_increment: int = 0,
        as_array: bool = False,
    ) -> tp.Iterator[tp.Tuple[TLabel, TNDArrayAny | Series[TVIndex, TVDtype]]]:
        """Generator of index, processed-window pairs."""
        yield from axis_window_items(
            source=self,
            axis=axis,
            size=size,
            step=step,
            window_sized=window_sized,
            window_func=window_func,
            window_valid=window_valid,
            label_shift=label_shift,
            label_missing_skips=label_missing_skips,
            label_missing_raises=label_missing_raises,
            start_shift=start_shift,
            size_increment=size_increment,
            as_array=as_array,
            derive_label=True,
        )

    def _axis_window(
        self,
        *,
        size: int,
        axis: int = 0,
        step: int = 1,
        window_sized: bool = True,
        window_func: tp.Optional[TCallableAny] = None,
        window_valid: tp.Optional[TCallableAny] = None,
        label_shift: int = 0,
        label_missing_skips: bool = True,
        label_missing_raises: bool = False,
        start_shift: int = 0,
        size_increment: int = 0,
        as_array: bool = False,
    ) -> tp.Iterator[tp.Union[TNDArrayAny, TSeriesAny]]:
        yield from (
            x
            for _, x in axis_window_items(
                source=self,
                axis=axis,
                size=size,
                step=step,
                window_sized=window_sized,
                window_func=window_func,
                window_valid=window_valid,
                label_shift=label_shift,
                label_missing_skips=label_missing_skips,
                label_missing_raises=label_missing_raises,
                start_shift=start_shift,
                size_increment=size_increment,
                as_array=as_array,
                derive_label=False,
            )
        )

    # ---------------------------------------------------------------------------

    @property
    def index(self) -> IndexBase:
        """
        The index instance assigned to this container.

        Returns:
            :obj:`static_frame.Index`
        """
        return self._index

    # ---------------------------------------------------------------------------
    # dictionary-like interface

    def keys(self) -> IndexBase:
        """
        Iterator of index labels.

        Returns:
            :obj:`Iterator[TLabel]`
        """
        return self._index

    def __iter__(self) -> tp.Iterator[TLabel]:
        """
        Iterator of index labels, same as :obj:`static_frame.Series.keys`.

        Returns:
            :obj:`Iterator[Hashasble]`
        """
        return self._index.__iter__()

    def __contains__(
        self,
        value: TLabel,
        /,
    ) -> bool:
        """
        Inclusion of value in index labels.

        Returns:
            :obj:`bool`
        """
        return self._index.__contains__(value)

    def items(self) -> tp.Iterator[tp.Tuple[tp.Any, tp.Any]]:
        """Iterator of pairs of index label and value.

        Returns:
            :obj:`Iterator[Tuple[Hashable, Any]]`
        """
        return zip(self._index.__iter__(), self.values)

    def get(
        self,
        key: TLabel,
        default: tp.Any = None,
    ) -> tp.Any:
        """
        Return the value found at the index key, else the default if the key is not found.

        Returns:
            :obj:`Any`
        """
        if key not in self._index:
            return default
        return self.__getitem__(key)

    # ---------------------------------------------------------------------------
    # transformations resulting in the same dimensionality

    def _reverse(self, axis: int = 0) -> tp.Self:
        """
        Return a reversed copy of this container, with no data copied.
        """
        return self._extract_iloc(REVERSE_SLICE)

    def _apply_ordering(
        self,
        order: TNDArrayIntDefault,
        sort_status: SortStatus,
        axis: int = 0,
    ) -> tp.Self:
        """
        Return a copy of this container with the specified ordering applied along the index of axis
        """
        index = self._index[order]
        index._sort_status = sort_status
        values = self.values[order]
        values.flags.writeable = False

        return self.__class__(values, index=index, name=self._name, own_index=True)

    @doc_inject(selector='sort')
    def sort_index(
        self,
        *,
        ascending: TBoolOrBools = True,
        kind: TSortKinds = DEFAULT_SORT_KIND,
        key: tp.Optional[
            tp.Callable[[IndexBase], tp.Union[TNDArrayAny, IndexBase]]
        ] = None,
    ) -> tp.Self:
        """
        Return a new Series ordered by the sorted Index.

        Args:
            *
            {ascendings}
            {kind}
            {key}

        Returns:
            :obj:`Series`
        """
        return sort_index_from_params(
            self._index,
            ascending=ascending,
            key=key,
            kind=kind,
            container=self,
        )

    @doc_inject(selector='sort')
    def sort_values(
        self,
        *,
        ascending: bool = True,
        kind: TSortKinds = DEFAULT_SORT_KIND,
        key: tp.Optional[
            tp.Callable[[TSeriesAny], tp.Union[TNDArrayAny, TSeriesAny]]
        ] = None,
    ) -> tp.Self:
        """
        Return a new Series ordered by the sorted values.

        Args:
            *
            {ascending}
            {kind}
            {key}

        Returns:
            :obj:`Series`
        """
        if key:
            cfs = key(self)
            cfs_values = cfs if cfs.__class__ is np.ndarray else cfs.values  # type: ignore
        else:
            cfs_values = self.values

        asc_is_element = isinstance(ascending, BOOL_TYPES)
        if not asc_is_element:
            raise RuntimeError('Multiple ascending values not permitted.')

        # argsort lets us do the sort once and reuse the results
        order = np.argsort(cfs_values, kind=kind)
        if not ascending:
            order = order[::-1]

        index = self._index[order]

        values = self.values[order]
        values.flags.writeable = False

        return self.__class__(values, index=index, name=self._name, own_index=True)

    def isin(
        self,
        other: tp.Iterable[tp.Any],
        /,
    ) -> tp.Self:
        """
        Return a same-sized Boolean Series that shows if the same-positioned element is in the iterable passed to the function.

        Returns:
            :obj:`Series`
        """
        # returns an immutable array
        array = isin(self.values, other)
        return self.__class__(array, index=self._index, name=self._name)

    @doc_inject(class_name='Series')
    def clip(
        self,
        *,
        lower: tp.Optional[tp.Union[float, TSeriesAny]] = None,
        upper: tp.Optional[tp.Union[float, TSeriesAny]] = None,
    ) -> tp.Self:
        """{}

        Args:
            lower: value or ``Series`` to define the inclusive lower bound.
            upper: value or ``Series`` to define the inclusive upper bound.

        Returns:
            :obj:`Series`
        """
        args: tp.List[TNDArrayAny | float | None] = []
        for idx, arg in enumerate((lower, upper)):
            # arg = args[idx]
            if isinstance(arg, Series):
                # after reindexing, strip away the index
                # NOTE: using the bound forces going to a float type; this may not be the best approach
                bound = -np.inf if idx == 0 else np.inf
                args.append(arg.reindex(self.index).fillna(bound).values)
            elif hasattr(arg, '__iter__'):
                raise RuntimeError(
                    'only Series are supported as iterable lower/upper arguments'
                )
            else:
                args.append(arg)

        array = np.clip(self.values, *args)  # type: ignore
        array.flags.writeable = False
        return self.__class__(array, index=self._index, name=self._name)

    def transpose(self) -> tp.Self:
        """Transpose. For a 1D immutable container, this returns a reference to self.

        Returns:
            :obj:`Series`
        """
        return self

    @property
    def T(self) -> tp.Self:
        """Transpose. For a 1D immutable container, this returns a reference to self.

        Returns:
            :obj:`Series`
        """
        return self.transpose()

    @doc_inject(selector='duplicated')
    def duplicated(
        self,
        *,
        exclude_first: bool = False,
        exclude_last: bool = False,
    ) -> tp.Self:
        """
        Return a same-sized Boolean Series that shows True for all values that are duplicated.

        Args:
            {exclude_first}
            {exclude_last}

        Returns:
            :obj:`numpy.ndarray`
        """
        duplicates = array_to_duplicated(
            self.values, exclude_first=exclude_first, exclude_last=exclude_last
        )
        duplicates.flags.writeable = False
        return self.__class__(duplicates, index=self._index)

    @doc_inject(selector='duplicated')
    def drop_duplicated(
        self, *, exclude_first: bool = False, exclude_last: bool = False
    ) -> tp.Self:
        """
        Return a Series with duplicated values removed.

        Args:
            {exclude_first}
            {exclude_last}

        Returns:
            :obj:`Series`
        """
        duplicates = array_to_duplicated(
            self.values, exclude_first=exclude_first, exclude_last=exclude_last
        )
        keep = ~duplicates
        return self.__class__(self.values[keep], index=self._index[keep], name=self._name)

    @doc_inject(select='astype')
    def astype(
        self,
        dtype: TDtypeSpecifier,
        /,
    ) -> tp.Self:
        """
        Return a Series with type determined by `dtype` argument. Note that for Series, this is a simple function, whereas for ``Frame``, this is an interface exposing both a callable and a getitem interface.

        Args:
            {dtype}

        Returns:
            :obj:`Series`
        """
        dtype = validate_dtype_specifier(dtype)
        array = astype_array(self.values, dtype)
        array.flags.writeable = False
        return self.__class__(array, index=self._index, name=self._name)

    def __round__(
        self,
        decimals: int = 0,
        /,
    ) -> tp.Self:
        """
        Return a Series rounded to the given decimals. Negative decimals round to the left of the decimal point.

        Args:
            decimals: number of decimals to round to.

        Returns:
            :obj:`Series`
        """
        return self.__class__(
            np.round(self.values, decimals), index=self._index, name=self._name
        )

    def roll(
        self,
        shift: int,
        /,
        *,
        include_index: bool = False,
    ) -> tp.Self:
        """Return a Series with values rotated forward and wrapped around the index (with a positive shift) or backward and wrapped around the index (with a negative shift).

        Args:
            shift: Positive or negative integer shift.
            include_index: Determine if the Index is shifted with the underlying data.

        Returns:
            :obj:`Series`
        """
        if shift % len(self.values):
            values = array_shift(array=self.values, shift=shift, axis=0, wrap=True)
            values.flags.writeable = False
        else:
            values = self.values

        if include_index:
            index = self._index.roll(shift=shift)
            own_index = True
        else:
            index = self._index
            own_index = False

        return self.__class__(values, index=index, name=self._name, own_index=own_index)

    def shift(
        self,
        shift: int,
        /,
        *,
        fill_value: tp.Any = np.nan,
    ) -> tp.Self:
        """Return a `Series` with values shifted forward on the index (with a positive shift) or backward on the index (with a negative shift).

        Args:
            shift: Positive or negative integer shift.
            fill_value: Value to be used to fill data missing after the shift.

        Returns:
            :obj:`Series`
        """
        if is_fill_value_factory_initializer(fill_value):
            fv = get_col_fill_value_factory(fill_value, None)(0, self.values.dtype)
        else:
            fv = fill_value

        if shift:
            values = array_shift(
                array=self.values, shift=shift, axis=0, wrap=False, fill_value=fv
            )
            values.flags.writeable = False
        else:
            values = self.values

        return self.__class__(values, index=self._index, name=self._name)

    # ---------------------------------------------------------------------------
    # ranking transformations resulting in the same dimensionality

    def _rank(
        self,
        *,
        method: RankMethod,
        skipna: bool = True,
        ascending: bool = True,
        start: int = 0,
        fill_value: tp.Any = np.nan,
    ) -> tp.Self:
        if is_fill_value_factory_initializer(fill_value):
            fv = get_col_fill_value_factory(fill_value, None)(0, self.values.dtype)
        else:
            fv = fill_value

        if not skipna or self.dtype.kind not in DTYPE_NA_KINDS:
            rankable = self
        else:
            # only call dropna if necessary
            rankable = self.dropna()

        # returns an immutable array
        values = rank_1d(
            rankable.values,
            method=method,
            ascending=ascending,
            start=start,
        )

        if rankable is self or len(values) == len(self):
            return self.__class__(
                values,
                index=self.index,
                name=self._name,
                own_index=True,
            )

        post = self.__class__(
            values,
            index=rankable.index,
            name=self._name,
            own_index=True,
        )
        # this will preserve the name
        return post.reindex(
            self.index,
            fill_value=fv,
            check_equals=False,  # the index will never be equal
        )

    @doc_inject(selector='rank')
    def rank_ordinal(
        self,
        *,
        skipna: bool = True,
        ascending: bool = True,
        start: int = 0,
        fill_value: tp.Any = np.nan,
    ) -> tp.Self:
        """Rank values distinctly, where ties get distinct values that maintain their ordering, and ranks are contiguous unique integers.

        Args:
            {skipna}
            {ascending}
            {start}
            {fill_value}

        Returns:
            :obj:`Series`
        """
        return self._rank(
            method=RankMethod.ORDINAL,
            skipna=skipna,
            ascending=ascending,
            start=start,
            fill_value=fill_value,
        )

    @doc_inject(selector='rank')
    def rank_dense(
        self,
        *,
        skipna: bool = True,
        ascending: bool = True,
        start: int = 0,
        fill_value: tp.Any = np.nan,
    ) -> tp.Self:
        """Rank values as compactly as possible, where ties get the same value, and ranks are contiguous (potentially non-unique) integers.

        Args:
            {skipna}
            {ascending}
            {start}
            {fill_value}

        Returns:
            :obj:`Series`
        """
        return self._rank(
            method=RankMethod.DENSE,
            skipna=skipna,
            ascending=ascending,
            start=start,
            fill_value=fill_value,
        )

    @doc_inject(selector='rank')
    def rank_min(
        self,
        *,
        skipna: bool = True,
        ascending: bool = True,
        start: int = 0,
        fill_value: tp.Any = np.nan,
    ) -> tp.Self:
        """Rank values where tied values are assigned the minimum ordinal rank; ranks are potentially non-contiguous and non-unique integers.

        Args:
            {skipna}
            {ascending}
            {start}
            {fill_value}

        Returns:
            :obj:`Series`
        """
        return self._rank(
            method=RankMethod.MIN,
            skipna=skipna,
            ascending=ascending,
            start=start,
            fill_value=fill_value,
        )

    @doc_inject(selector='rank')
    def rank_max(
        self,
        *,
        skipna: bool = True,
        ascending: bool = True,
        start: int = 0,
        fill_value: tp.Any = np.nan,
    ) -> tp.Self:
        """Rank values where tied values are assigned the maximum ordinal rank; ranks are potentially non-contiguous and non-unique integers.

        Args:
            {skipna}
            {ascending}
            {start}
            {fill_value}

        Returns:
            :obj:`Series`
        """
        return self._rank(
            method=RankMethod.MAX,
            skipna=skipna,
            ascending=ascending,
            start=start,
            fill_value=fill_value,
        )

    @doc_inject(selector='rank')
    def rank_mean(
        self,
        *,
        skipna: bool = True,
        ascending: bool = True,
        start: int = 0,
        fill_value: tp.Any = np.nan,
    ) -> tp.Self:
        """Rank values where tied values are assigned the mean of the ordinal ranks; ranks are potentially non-contiguous and non-unique floats.

        Args:
            {skipna}
            {ascending}
            {start}
            {fill_value}

        Returns:
            :obj:`Series`
        """
        return self._rank(
            method=RankMethod.MEAN,
            skipna=skipna,
            ascending=ascending,
            start=start,
            fill_value=fill_value,
        )

    # ---------------------------------------------------------------------------
    # transformations resulting in changed dimensionality

    @doc_inject(selector='head', class_name='Series')
    def head(
        self,
        count: int = 5,
        /,
    ) -> TSeriesAny:
        """{doc}

        Args:
            {count}

        Returns:
            :obj:`Series`
        """
        return self.iloc[:count]

    @doc_inject(selector='tail', class_name='Series')
    def tail(
        self,
        count: int = 5,
        /,
    ) -> TSeriesAny:
        """{doc}s

        Args:
            {count}

        Returns:
            :obj:`Series`
        """
        return self.iloc[-count:]

    def count(
        self,
        *,
        skipna: bool = True,
        skipfalsy: bool = False,
        unique: bool = False,
        axis: int = 0,
    ) -> int:
        """
        Return the count of non-NA, non-falsy, and/or unique elements.

        Args:
            skipna: skip NA (NaN, None) values.
            skipfalsy: skip falsu values (0, '', False, None, NaN)
            unique: Count unique items after optionally applying ``skipna`` or ``skipfalsy`` removals.
        """
        # NOTE: axis arg for compat with Frame, is not used
        if not skipna and skipfalsy:
            raise RuntimeError('Cannot skipfalsy and not skipna.')

        values = self.values
        if not skipna and not skipfalsy and not unique:
            return len(values)

        valid: tp.Optional[TNDArrayAny] = None
        if skipfalsy:  # always includes skipna
            valid = ~isfalsy_array(values)
        elif skipna:  # NOTE: elif, as skipfalsy incldues skipna
            valid = ~isna_array(values)

        if unique and valid is None:
            return len(ufunc_unique1d(values))
        elif unique and valid is not None:  # valid is a Boolean array
            return len(ufunc_unique1d(values[valid]))
        elif not unique and valid is not None:
            return valid.sum()  # type: ignore [no-any-return]
        # not unique, valid is None, means no removals, handled above
        raise NotImplementedError()  # pragma: no cover

    @doc_inject(selector='sample')
    def sample(
        self,
        count: int = 1,
        /,
        *,
        seed: tp.Optional[int] = None,
    ) -> tp.Self:
        """{doc}

        Args:
            {count}
            {seed}
        """
        index, key = self._index._sample_and_key(count=count, seed=seed)
        values = self.values[key]
        values.flags.writeable = False
        return self.__class__(values, index=index, name=self._name)

    # ---------------------------------------------------------------------------

    @doc_inject(selector='argminmax')
    def loc_min(self, *, skipna: bool = True) -> TLabel:
        """
        Return the label corresponding to the minimum value found.

        Args:
            {skipna}

        Returns:
            TLabel
        """
        # if skipna is False and a NaN is returned, this will raise
        post = argmin_1d(self.values, skipna=skipna)
        if isinstance(post, FLOAT_TYPES):  # NaN was returned
            raise RuntimeError('cannot produce loc representation from NaN')
        return self.index[post]  # type: ignore

    @doc_inject(selector='argminmax')
    def iloc_min(
        self,
        *,
        skipna: bool = True,
    ) -> int:
        """
        Return the integer index corresponding to the minimum value found.

        Args:
            {skipna}

        Returns:
            int
        """
        return argmin_1d(self.values, skipna=skipna)  # type: ignore

    @doc_inject(selector='argminmax')
    def loc_max(
        self,
        *,
        skipna: bool = True,
    ) -> TLabel:
        """
        Return the label corresponding to the maximum value found.

        Args:
            {skipna}

        Returns:
            TLabel
        """
        post = argmax_1d(self.values, skipna=skipna)
        if isinstance(post, FLOAT_TYPES):  # NaN was returned
            raise RuntimeError('cannot produce loc representation from NaN')
        return self.index[post]  # type: ignore

    @doc_inject(selector='argminmax')
    def iloc_max(
        self,
        *,
        skipna: bool = True,
    ) -> int:
        """
        Return the integer index corresponding to the maximum value.

        Args:
            {skipna}

        Returns:
            int
        """
        return argmax_1d(self.values, skipna=skipna)  # type: ignore

    # ---------------------------------------------------------------------------

    def _label_not_missing(
        self,
        *,
        return_label: bool,
        forward: bool,
        fill_value: TLabel = np.nan,
        func: tp.Callable[[TNDArrayAny], TNDArrayAny],
    ) -> TLabel:
        """
        Return the label corresponding to the first not NA (None or nan) value found.

        Args:
            {skipna}

        Returns:
            TLabel
        """
        # if skipna is False and a NaN is returned, this will raise
        if not len(self.values):
            return fill_value
        target = ~func(self.values)
        pos = first_true_1d(target, forward=forward)
        if pos == -1:
            return fill_value
        if return_label:
            return self._index[pos]  # type: ignore
        return pos

    def iloc_notna_first(
        self,
        *,
        fill_value: int = -1,
    ) -> TLabel:
        """
        Return the position corresponding to the first not NA (None or nan) value found.

        Args:
            {fill_value}

        Returns:
            TLabel
        """
        return self._label_not_missing(
            return_label=False,
            forward=True,
            fill_value=fill_value,
            func=isna_array,
        )

    def iloc_notna_last(
        self,
        *,
        fill_value: int = -1,
    ) -> TLabel:
        """
        Return the position corresponding to the last not NA (None or nan) value found.

        Args:
            {fill_value}

        Returns:
            TLabel
        """
        return self._label_not_missing(
            return_label=False,
            forward=False,
            fill_value=fill_value,
            func=isna_array,
        )

    def loc_notna_first(
        self,
        *,
        fill_value: TLabel = np.nan,
    ) -> TLabel:
        """
        Return the label corresponding to the first not NA (None or nan) value found.

        Args:
            {fill_value}

        Returns:
            TLabel
        """
        return self._label_not_missing(
            return_label=True,
            forward=True,
            fill_value=fill_value,
            func=isna_array,
        )

    def loc_notna_last(
        self,
        *,
        fill_value: TLabel = -1,
    ) -> TLabel:
        """
        Return the label corresponding to the last not NA (None or nan) value found.

        Args:
            {fill_value}

        Returns:
            TLabel
        """
        return self._label_not_missing(
            return_label=True,
            forward=False,
            fill_value=fill_value,
            func=isna_array,
        )

    # ---------------------------------------------------------------------------
    def loc_notfalsy_first(
        self,
        *,
        fill_value: TLabel = np.nan,
    ) -> TLabel:
        """
        Return the label corresponding to the first non-falsy (including nan) value found.

        Args:
            {fill_value}

        Returns:
            TLabel
        """
        return self._label_not_missing(
            return_label=True,
            forward=True,
            fill_value=fill_value,
            func=isfalsy_array,
        )

    def iloc_notfalsy_first(
        self,
        *,
        fill_value: int = -1,
    ) -> TLabel:
        """
        Return the position corresponding to the first non-falsy (including nan) value found.

        Args:
            {fill_value}

        Returns:
            TLabel
        """
        return self._label_not_missing(
            return_label=False,
            forward=True,
            fill_value=fill_value,
            func=isfalsy_array,
        )

    def loc_notfalsy_last(
        self,
        *,
        fill_value: TLabel = np.nan,
    ) -> TLabel:
        """
        Return the label corresponding to the last non-falsy (including nan) value found.

        Args:
            {fill_value}

        Returns:
            TLabel
        """
        return self._label_not_missing(
            return_label=True,
            forward=False,
            fill_value=fill_value,
            func=isfalsy_array,
        )

    def iloc_notfalsy_last(
        self,
        *,
        fill_value: int = -1,
    ) -> TLabel:
        """
        Return the position corresponding to the last non-falsy (including nan) value found.

        Args:
            {fill_value}

        Returns:
            TLabel
        """
        return self._label_not_missing(
            return_label=False,
            forward=False,
            fill_value=fill_value,
            func=isfalsy_array,
        )

    # ---------------------------------------------------------------------------
    def cov(
        self,
        other: tp.Union[TSeriesAny, TNDArrayAny],
        /,
        *,
        ddof: int = 1,
    ) -> float:
        """
        Return the index-aligned covariance to the supplied :obj:`Series`.

        Args:
            ddof: Delta degrees of freedom, defaults to 1.
        """
        if isinstance(other, Series):
            other = other.loc[self._index].values
        # by convention, we return just the corner
        return np.cov(self.values, other, ddof=ddof)[0, -1]  # type: ignore [no-any-return]

    def corr(
        self,
        other: tp.Union[TSeriesAny, TNDArrayAny],
        /,
    ) -> float:
        """
        Return the index-aligned correlation to the supplied :obj:`Series`.

        Args:
            other: Series to be correlated with by selection on corresponding labels.
        """
        if isinstance(other, Series):
            other = other.loc[self._index].values
        # by convention, we return just the corner
        return np.corrcoef(self.values, other)[0, -1]  # type: ignore [no-any-return]

    # ---------------------------------------------------------------------------

    @doc_inject(selector='searchsorted', label_type='iloc (integer)')
    def iloc_searchsorted(
        self,
        values: tp.Any,
        /,
        *,
        side_left: bool = True,
    ) -> TNDArrayAny:  # might be 0 dim scalar
        """
        {doc}

        Args:
            {values}
            {side_left}
        """
        if not isinstance(values, STRING_TYPES) and hasattr(values, '__len__'):
            if values.__class__ is not np.ndarray:
                values, _ = iterable_to_array_1d(values)
        post: TNDArrayAny = np.searchsorted(
            self.values,  # pyright: ignore
            values,
            'left' if side_left else 'right',
        )
        return post

    @doc_inject(selector='searchsorted', label_type='loc (label)')
    def loc_searchsorted(
        self,
        values: tp.Any,
        /,
        *,
        side_left: bool = True,
        fill_value: tp.Any = np.nan,
    ) -> tp.Union[TLabel, TNDArrayAny]:
        """
        {doc}

        Args:
            {values}
            {side_left}
            {fill_value}
        """
        sel: TNDArrayAny = self.iloc_searchsorted(values, side_left=side_left)

        length = self.__len__()
        if sel.ndim == 0 and sel == length:  # an element:
            return fill_value  # type: ignore [no-any-return]

        # sel and mask might be zero-dimensional
        found: TNDArrayAny = sel != length
        if found.all():  # if all matches within series
            if self._index.ndim == 1:
                return self._index.values[sel]
            elif found.sum() == 1:
                return self._index._extract_iloc(sel)  # pyright: ignore

        if self._index.ndim == 1:
            post = np.full(
                len(sel),
                fill_value,
                dtype=resolve_dtype(
                    self._index.dtype,  # type: ignore
                    dtype_from_element(fill_value),
                ),
            )
            post[found] = self._index.values[sel[found]]
        else:
            # build object array of tuples
            post = np.full(len(sel), fill_value, dtype=object)
            for i, (j, assign) in enumerate(zip(sel, found)):
                if assign:
                    post[i] = self._index._extract_iloc(j)

        post.flags.writeable = False
        return post

    # ---------------------------------------------------------------------------
    def _insert(
        self,
        key: int | np.integer[tp.Any],  # iloc positions
        container: TSeriesAny,
        *,
        after: bool,
    ) -> tp.Self:
        if not isinstance(container, Series):
            raise NotImplementedError(f'No support for inserting with {type(container)}')

        if not len(container.index):  # must be empty data, empty index container
            return self

        # this filter is needed to handle possible invalid ILoc values passed through
        key = iloc_to_insertion_iloc(key, self.values.__len__()) + after

        dtype = resolve_dtype(self.values.dtype, container.dtype)
        values = np.empty(len(self) + len(container), dtype=dtype)
        key_end = key + len(container)

        values_prior = self.values

        values[:key] = values_prior[:key]
        values[key:key_end] = container.values
        values[key_end:] = values_prior[key:]
        values.flags.writeable = False

        labels_prior = self._index.values

        index = self._index.__class__.from_labels(
            chain(
                labels_prior[:key],
                container._index.__iter__(),
                labels_prior[key:],
            )
        )

        return self.__class__(
            values,
            index=index,
            name=self._name,
            own_index=True,
        )

    @doc_inject(selector='insert')
    def insert_before(
        self,
        key: TLabel,
        container: TSeriesAny,
        /,
    ) -> tp.Self:
        """
        Create a new :obj:`Series` by inserting a :obj:`Series` at the position before the label specified by ``key``.

        Args:
            {key_before}
            {container}

        Returns:
            :obj:`Series`
        """
        iloc_key = self._index._loc_to_iloc(key)
        if not isinstance(iloc_key, INT_TYPES):
            raise RuntimeError(f'Unsupported key type: {key!r}')
        return self._insert(iloc_key, container, after=False)

    @doc_inject(selector='insert')
    def insert_after(
        self,
        key: TLabel,  # iloc positions
        container: TSeriesAny,
        /,
    ) -> tp.Self:
        """
        Create a new :obj:`Series` by inserting a :obj:`Series` at the position after the label specified by ``key``.

        Args:
            {key_after}
            {container}

        Returns:
            :obj:`Series`
        """
        iloc_key = self._index._loc_to_iloc(key)
        if not isinstance(iloc_key, INT_TYPES):
            raise RuntimeError(f'Unsupported key type: {key!r}')
        return self._insert(iloc_key, container, after=True)

    # ---------------------------------------------------------------------------
    # utility function to numpy array or other types

    def unique(self) -> TNDArrayAny:
        """
        Return a NumPy array of unique values.

        Returns:
            :obj:`numpy.ndarray`
        """
        return ufunc_unique1d(self.values)

    @doc_inject()
    def unique_enumerated(
        self,
        *,
        retain_order: bool = False,
        func: tp.Optional[tp.Callable[[tp.Any], bool]] = None,
    ) -> tp.Tuple[TNDArrayAny, TNDArrayAny]:
        """
        {doc}
        {args}
        """
        return ufunc_unique_enumerated(
            self.values,
            retain_order=retain_order,
            func=func,
        )

    @doc_inject()
    def equals(
        self,
        other: tp.Any,
        /,
        *,
        compare_name: bool = False,
        compare_dtype: bool = False,
        compare_class: bool = False,
        skipna: bool = True,
    ) -> bool:
        """
        {doc}

        Args:
            {compare_name}
            {compare_dtype}
            {compare_class}
            {skipna}
        """
        if id(other) == id(self):
            return True

        # NOTE: there are presently no Series subclasses, but better to be consistent
        if compare_class and self.__class__ != other.__class__:
            return False
        elif not isinstance(other, Series):
            return False

        if len(self.values) != len(other.values):
            return False
        if compare_name and self._name != other._name:
            return False
        if compare_dtype and self.values.dtype != other.values.dtype:
            return False

        if not arrays_equal(self.values, other.values, skipna=skipna):
            return False

        return self._index.equals(
            other._index,
            compare_name=compare_name,
            compare_dtype=compare_dtype,
            compare_class=compare_class,
            skipna=skipna,
        )

    # ---------------------------------------------------------------------------
    # export

    def to_pairs(self) -> tp.Iterable[tp.Tuple[TLabel, tp.Any]]:
        """
        Return a tuple of tuples, where each inner tuple is a pair of index label, value.

        Returns:
            tp.Iterable[tp.Tuple[TLabel, tp.Any]]
        """
        index_values: tp.Iterable[TLabel]
        if isinstance(self._index, IndexHierarchy):
            index_values = self._index.__iter__()
        else:
            index_values = self._index.values

        return tuple(zip(index_values, self.values))

    def _to_frame(
        self,
        *,
        constructor: tp.Type[FrameType],
        axis: int = 1,
        index: TIndexInitOrAuto = None,
        index_constructor: TIndexCtorSpecifier = None,
        columns: TIndexInitOrAuto = None,
        columns_constructor: TIndexCtorSpecifier = None,
        name: TName = NAME_DEFAULT,
    ) -> FrameType:
        """
        Common function for creating :obj:`Frame` from :obj:`Series`.
        """
        from static_frame import TypeBlocks

        if axis == 1:
            # present as a column
            def block_gen() -> tp.Iterator[TNDArrayAny]:
                yield self.values

            if index is IndexAutoFactory:
                index = None
                own_index = False
            elif index is not None:
                own_index = False
            else:
                index = self._index
                own_index = index_constructor is None

            if columns is IndexAutoFactory:
                columns = None
            elif columns is not None:
                pass  # precedent
            elif self._name is None:
                columns = None
            else:
                columns = (self._name,)
            own_columns = False

        elif axis == 0:

            def block_gen() -> tp.Iterator[TNDArrayAny]:
                array = self.values
                yield array.reshape(1, array.shape[0])

            if index is IndexAutoFactory:
                index = None
            elif index is not None:
                pass
            elif self._name is None:
                index = None
            else:
                index = (self._name,)
            own_index = False

            # if column constuctor is static, we can own the static index
            if columns is IndexAutoFactory:
                columns = None
                own_columns = False
            elif columns is not None:
                own_columns = False
            else:
                columns = self._index
                own_columns = constructor._COLUMNS_CONSTRUCTOR.STATIC and (
                    columns_constructor is None
                )
        else:
            raise NotImplementedError(f'no handling for axis {axis}')

        return constructor(
            TypeBlocks.from_blocks(block_gen()),
            index=index,
            columns=columns,
            index_constructor=index_constructor,
            columns_constructor=columns_constructor,
            own_data=True,
            own_index=own_index,
            own_columns=own_columns,
            name=name if name is not NAME_DEFAULT else None,
        )

    def to_frame(
        self,
        *,
        axis: int = 1,
        index: TIndexInitOrAuto = None,
        index_constructor: TIndexCtorSpecifier = None,
        columns: TIndexInitOrAuto = None,
        columns_constructor: TIndexCtorSpecifier = None,
        name: TName = NAME_DEFAULT,
    ) -> TFrameAny:
        """
        Return a :obj:`Frame` view of this :obj:`Series`. As underlying data is immutable, this is a no-copy operation.

        Args:
            axis: Axis 1 (default) creates a single-column :obj:`Frame` with the same index: axis 0 creates a single-row :obj:`Frame` with the index as columns.
            *
            index_constructor:
            columns_constructor:
            name:

        Returns:
            :obj:`Frame`
        """
        from static_frame import Frame

        return self._to_frame(
            constructor=Frame,
            axis=axis,
            index=index,
            index_constructor=index_constructor,
            columns=columns,
            columns_constructor=columns_constructor,
            name=name,
        )

    def to_frame_go(
        self,
        *,
        axis: int = 1,
        index: TIndexInitOrAuto = None,
        index_constructor: TIndexCtorSpecifier = None,
        columns: TIndexInitOrAuto = None,
        columns_constructor: TIndexCtorSpecifier = None,
        name: TName = NAME_DEFAULT,
    ) -> TFrameGOAny:
        """
        Return :obj:`FrameGO` view of this :obj:`Series`. As underlying data is immutable, this is a no-copy operation.

        Args:
            axis:
            *
            index_constructor:
            columns_constructor:
        Returns:
            :obj:`FrameGO`
        """
        from static_frame import FrameGO

        return self._to_frame(
            constructor=FrameGO,
            axis=axis,
            index=index,
            index_constructor=index_constructor,
            columns=columns,
            columns_constructor=columns_constructor,
            name=name,
        )

    def to_frame_he(
        self,
        *,
        axis: int = 1,
        index: TIndexInitOrAuto = None,
        index_constructor: TIndexCtorSpecifier = None,
        columns: TIndexInitOrAuto = None,
        columns_constructor: TIndexCtorSpecifier = None,
        name: TName = NAME_DEFAULT,
    ) -> TFrameHEAny:
        """
        Return :obj:`FrameHE` view of this :obj:`Series`. As underlying data is immutable, this is a no-copy operation.

        Args:
            axis:
            *
            index_constructor:
            columns_constructor:
        Returns:
            :obj:`FrameHE`
        """
        from static_frame import FrameHE

        return self._to_frame(
            constructor=FrameHE,
            axis=axis,
            index=index,
            index_constructor=index_constructor,
            columns=columns,
            columns_constructor=columns_constructor,
            name=name,
        )

    def to_series_he(self) -> TSeriesHEAny:
        """
        Return a :obj:`SeriesHE` from this :obj:`Series`.
        """
        return SeriesHE(
            self.values,
            index=self._index,
            name=self._name,
            own_index=True,
        )

    def _to_signature_bytes(
        self,
        include_name: bool = True,
        include_class: bool = True,
        encoding: str = 'utf-8',
    ) -> bytes:
        if self.values.dtype == DTYPE_OBJECT:
            raise TypeError('Object dtypes do not have stable hashes')

        return b''.join(
            chain(
                iter_component_signature_bytes(
                    self,
                    include_name=include_name,
                    include_class=include_class,
                    encoding=encoding,
                ),
                (
                    self._index._to_signature_bytes(
                        include_name=include_name,
                        include_class=include_class,
                        encoding=encoding,
                    ),
                    self.values.tobytes(),
                ),
            )
        )

    # ---------------------------------------------------------------------------

    def to_pandas(self) -> 'pandas.Series[tp.Any]': # pyright: ignore
        """
        Return a Pandas Series.

        Returns:
            :obj:`pandas.Series`
        """
        import pandas

        return pandas.Series(
            self.values.copy(), index=self._index.to_pandas(), name=self._name
        )

    @doc_inject(class_name='Series')
    def to_html(
        self,
        config: tp.Optional[DisplayConfig] = None,
        /,
        *,
        style_config: tp.Optional[StyleConfig] = STYLE_CONFIG_DEFAULT,
    ) -> str:
        """
        {}
        """
        config = config or DisplayActive.get(type_show=False)
        config = config.to_display_config(
            display_format=DisplayFormats.HTML_TABLE,
        )
        style_config = style_config_css_factory(style_config, self)
        return repr(self.display(config, style_config=style_config))

    @doc_inject(class_name='Series')
    def to_html_datatables(
        self,
        fp: tp.Optional[TPathSpecifierOrTextIO] = None,
        /,
        *,
        show: bool = True,
        config: tp.Optional[DisplayConfig] = None,
    ) -> tp.Optional[str]:
        """
        {}
        """
        config = config or DisplayActive.get(type_show=False)
        config = config.to_display_config(
            display_format=DisplayFormats.HTML_DATATABLES,
        )
        content = repr(self.display(config))
        # path_filter applied in call
        fp = write_optional_file(content=content, fp=fp)

        if show:
            assert isinstance(fp, str)  # pragma: no cover
            import webbrowser  # pragma: no cover

            webbrowser.open_new_tab(fp)  # pragma: no cover
        return fp


doc_update(Series.__init__, selector='container_init', class_name='Series')


# -------------------------------------------------------------------------------
class SeriesAssign(Assign):
    __slots__ = ('container', 'key')

    _INTERFACE = (
        '__call__',
        'apply',
        'apply_element',
        'apply_element_items',
    )

    def __init__(
        self,
        container: TSeriesAny,
        key: TILocSelector,
    ) -> None:
        """
        Args:
            key: an iloc-style key.
        """
        self.container: TSeriesAny = container
        self.key: TILocSelector = key

    def __call__(
        self,
        value: tp.Any,  # any possible assignment type
        *,
        fill_value: tp.Any = np.nan,
    ) -> TSeriesAny:
        """
        Assign the ``value`` in the position specified by the selector. The `name` attribute is propagated to the returned container.

        Args:
            value:  Value to assign, which can be a :obj:`Series`, np.ndarray, or element.
            *.
            fill_value: If the ``value`` parameter has to be reindexed, this element will be used to fill newly created elements.
        """
        if isinstance(value, Series):
            value = self.container._reindex_other_like_iloc(
                value, self.key, fill_value=fill_value
            ).values

        if value.__class__ is np.ndarray:
            if len(value) == 0:
                return self.container
            value_dtype = value.dtype
        elif isinstance(value, tuple):
            value_dtype = DTYPE_OBJECT
        elif hasattr(value, '__iter__') and not isinstance(value, STRING_TYPES):
            value, _ = iterable_to_array_1d(value, count=len(value))
            if len(value) == 0:
                return self.container
            value_dtype = value.dtype
        else:  # strings, other elements
            value_dtype = dtype_from_element(value)

        dtype = resolve_dtype(self.container.dtype, value_dtype)

        # create or copy the array to return
        if dtype == self.container.dtype:
            array = self.container.values.copy()
        else:
            array = astype_array(self.container.values, dtype)

        array[self.key] = value
        array.flags.writeable = False

        return self.container.__class__(
            array, index=self.container._index, name=self.container._name
        )

    def apply(
        self,
        func: TCallableAny,
        *,
        fill_value: tp.Any = np.nan,
    ) -> TSeriesAny:
        """
        Provide a function to apply to the assignment target, and use that as the assignment value.

        Args:
            func: A function to apply to the assignment target.
            *.
            fill_value: If the function does not produce a container with a matching index, the element will be used to fill newly created elements.
        """
        value = func(self.container.iloc[self.key])
        return self.__call__(value, fill_value=fill_value)

    def apply_element(
        self,
        func: TCallableAny,
        *,
        dtype: TDtypeSpecifier = None,
        fill_value: tp.Any = np.nan,
    ) -> TSeriesAny:
        """
        Provide a function to apply to each element in the assignment target, and use that as the assignment value.

        Args:
            func: A function to apply to the assignment target.
            *
            fill_value: If the function does not produce a container with a matching index, the element will be used to fill newly created elements.
        """
        return self.apply(
            lambda c: c.iter_element().apply(func, dtype=dtype),
            fill_value=fill_value,
        )

    def apply_element_items(
        self,
        func: TCallableAny,
        *,
        dtype: TDtypeSpecifier = None,
        fill_value: tp.Any = np.nan,
    ) -> TSeriesAny:
        """
        Provide a function, taking pairs of label, element, to apply to each element in the assignment target, and use that as the assignment value.

        Args:
            func: A function, taking pairs of label, element, to apply to the assignment target.
            *
            fill_value: If the function does not produce a container with a matching index, the element will be used to fill newly created elements.
        """
        return self.apply(
            lambda c: c.iter_element_items().apply(func, dtype=dtype),
            fill_value=fill_value,
        )


# -------------------------------------------------------------------------------


class SeriesHE(Series[TVIndex, TVDtype]):
    """
    A hash/equals subclass of :obj:`Series`, permiting usage in a Python set, dictionary, or other contexts where a hashable container is needed. To support hashability, ``__eq__`` is implemented to return a Boolean rather than an Boolean :obj:`Series`.
    """

    __slots__ = ('_hash',)
    _hash: int

    def __eq__(self, other: tp.Any) -> bool:
        """
        Return True if other is a ``Series`` with the same labels, values, and name. Container class and underlying dtypes are not independently compared.
        """
        return self.equals(
            other,
            compare_name=True,
            compare_dtype=False,
            compare_class=False,
            skipna=True,
        )

    def __ne__(
        self,
        other: tp.Any,
        /,
    ) -> bool:
        """
        Return False if other is a ``Series`` with the same labels, values, and name. Container class and underlying dtypes are not independently compared.
        """
        return not self.__eq__(other)

    def __hash__(self) -> int:
        if not hasattr(self, '_hash'):
            self._hash = hash(tuple(self.index.values))
        return self._hash

    def to_series(self) -> TSeriesAny:
        """
        Return a ``Series`` from this ``SeriesHE``.
        """
        return Series(
            self.values,
            index=self._index,
            name=self._name,
            own_index=True,
        )

    # ---------------------------------------------------------------------------
    # interfaces are redefined to show type returned type

    @property
    def loc(self) -> InterGetItemLocReduces[TSeriesHEAny, TVDtype]:
        """
        Interface for label-based selection.
        """
        return InterGetItemLocReduces(self._extract_loc)  # type: ignore

    @property
    def iloc(self) -> InterGetItemILocReduces[TSeriesHEAny, TVDtype]:
        """
        Interface for position-based selection.
        """
        return InterGetItemILocReduces(self._extract_iloc)


TSeriesAny = Series[tp.Any, tp.Any]
TSeriesHEAny = SeriesHE[tp.Any, tp.Any]
