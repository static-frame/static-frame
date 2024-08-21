from __future__ import annotations

from collections import defaultdict
from functools import partial
from itertools import chain
from itertools import product
from itertools import repeat

import numpy as np
import typing_extensions as tp
from arraykit import resolve_dtype
from arraykit import resolve_dtype_iter

from static_frame.core.container_util import index_from_optional_constructor
from static_frame.core.index import Index
from static_frame.core.index_base import IndexBase
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.util import DEFAULT_FAST_SORT_KIND
from static_frame.core.util import TCallableAny
from static_frame.core.util import TDepthLevel
from static_frame.core.util import TIndexCtor
from static_frame.core.util import TIndexCtorSpecifier
from static_frame.core.util import TIndexHierarchyCtor
from static_frame.core.util import TLabel
from static_frame.core.util import TName
from static_frame.core.util import TSortKinds
from static_frame.core.util import TUFunc
from static_frame.core.util import dtype_from_element
from static_frame.core.util import iterable_to_array_1d
from static_frame.core.util import ufunc_dtype_to_dtype
from static_frame.core.util import ufunc_unique
from static_frame.core.util import ufunc_unique1d

if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame  # pragma: no cover
    from static_frame.core.series import Series  # pragma: no cover
    TNDArrayAny = np.ndarray[tp.Any, tp.Any] #pragma: no cover
    TDtypeAny = np.dtype[tp.Any] #pragma: no cover
    TFrameAny = Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]] #pragma: no cover


#-------------------------------------------------------------------------------
def extrapolate_column_fields(
        columns_fields: tp.Sequence[TLabel],
        group: tp.Tuple[TLabel, ...],
        data_fields: tp.Sequence[TLabel],
        func_fields: tp.Iterable[TLabel],
        ) -> tp.Iterable[TLabel]:
    '''Determine columns to be reatined from gruop and data fields.
    Used in Frame.pivot.

    Args:
        group: a unique label from the the result of doing a group-by with the `columns_fields`.
    '''
    # NOTE: this will work correctly with no_func=True

    columns_fields_len = len(columns_fields)
    data_fields_len = len(data_fields)

    sub_columns: tp.Iterable[TLabel]

    if columns_fields_len == 1 and data_fields_len == 1:
        if not func_fields:
            sub_columns = group # already a tuple
        else:
            sub_columns = [group + (label,) for label in func_fields]
    elif columns_fields_len == 1 and data_fields_len > 1:
        # create a sub heading for each data field
        if not func_fields:
            sub_columns = list(product(group, data_fields))
        else:
            sub_columns = list(product(group, data_fields, func_fields))
    elif columns_fields_len > 1 and data_fields_len == 1:
        if not func_fields:
            sub_columns = (group,)
        else:
            sub_columns = [group + (label,) for label in func_fields]
    else: # group is already a tuple of the partial column label; need to extend with each data field
        if not func_fields:
            sub_columns = [group + (field,) for field in data_fields]
        else:
            sub_columns = [group + (field, label) for field in data_fields for label in func_fields]

    return sub_columns

def pivot_records_dtypes(
        dtype_map: Series[tp.Any, tp.Any],
        data_fields: tp.Iterable[TLabel],
        func_single: tp.Optional[TUFunc],
        func_map: tp.Sequence[tp.Tuple[TLabel, TUFunc]]
        ) -> tp.Iterator[tp.Optional[TDtypeAny]]:
    '''
    Iterator of ordered dtypes, providing multiple dtypes per field when func_map is provided.
    '''
    for field in data_fields:
        dtype: TDtypeAny = dtype_map[field]
        if func_single:
            yield ufunc_dtype_to_dtype(func_single, dtype)
        else: # we assume
            for _, func in func_map:
                yield ufunc_dtype_to_dtype(func, dtype)

def pivot_records_items_to_frame(
        *,
        blocks: TypeBlocks,
        group_fields_iloc: tp.List[int],
        group_depth: int,
        data_fields_iloc: tp.Iterable[int],
        func_single: tp.Optional[TUFunc],
        func_map: tp.Sequence[tp.Tuple[TLabel, TUFunc]],
        func_no: bool,
        kind: TSortKinds,
        columns_constructor: TIndexCtor,
        columns: tp.Sequence[TLabel],
        index_constructor: TIndexCtor,
        dtypes: tp.Tuple[tp.Optional[TDtypeAny], ...],
        frame_cls: tp.Type[TFrameAny],
        ) -> TFrameAny:
    '''
    Given a Frame and pivot parameters, perform the group by ont he group_fields and within each group,
    '''
    group_key: tp.List[int] | int = group_fields_iloc if group_depth > 1 else group_fields_iloc[0]
    record_size = len(data_fields_iloc) * (1 if (func_single or func_no) else len(func_map)) # type: ignore

    index_labels: tp.List[TLabel] = []
    arrays: tp.List[tp.List[tp.Any]] = [list() for _ in range(record_size)]

    part: TypeBlocks
    for label, _, part in blocks.group(axis=0, key=group_key, kind=kind):
        index_labels.append(label) # type: ignore
        if func_no:
            if len(part) != 1:
                raise RuntimeError('pivot requires aggregation of values; provide a `func` argument.')
            for i, column_key in enumerate(data_fields_iloc):
                arrays[i].append(part._extract(0, column_key))
        elif func_single:
            for i, column_key in enumerate(data_fields_iloc):
                arrays[i].append(func_single(part._extract_array_column(column_key)))
        else:
            i = 0
            for column_key in data_fields_iloc:
                values = part._extract_array_column(column_key)
                for _, func in func_map:
                    arrays[i].append(func(values))
                    i += 1

    def gen() -> tp.Iterator[TNDArrayAny]:
        for b, dtype in zip(arrays, dtypes):
            if dtype is None:
                array, _ = iterable_to_array_1d(b)
            else:
                array = np.array(b, dtype=dtype)
            array.flags.writeable = False
            yield array

    tb = TypeBlocks.from_blocks(gen())
    return frame_cls(tb,
            index=index_constructor(index_labels),
            columns=columns_constructor(columns),
            own_data=True,
            own_index=True,
            own_columns=True,
            )


def pivot_records_items_to_blocks(*,
        blocks: TypeBlocks,
        group_fields_iloc: tp.List[int],
        group_depth: int,
        data_fields_iloc: tp.Iterable[int],
        func_single: tp.Optional[TUFunc],
        func_map: tp.Sequence[tp.Tuple[TLabel, TUFunc]],
        func_no: bool,
        fill_value: tp.Any,
        fill_value_dtype: TDtypeAny,
        index_outer: 'IndexBase',
        dtypes: tp.Tuple[tp.Optional[TDtypeAny], ...],
        kind: TSortKinds,
        ) -> tp.List[TNDArrayAny]:
    '''
    Given a Frame and pivot parameters, perform the group by ont he group_fields and within each group,
    '''
    # NOTE: this delivers results by label, row for use in a Frame.from_records_items constructor
    group_key: tp.List[int] | int = group_fields_iloc if group_depth > 1 else group_fields_iloc[0]
    arrays: tp.List[tp.Union[tp.List[tp.Any], TNDArrayAny]] = []

    for dtype in dtypes:
        if dtype is None:
            # we can use fill_value here, as either it will be completely replaced (and not effect dtype evaluation) or be needed (and already there)
            arrays.append([fill_value] * len(index_outer))
        else:
            arrays.append(np.empty(len(index_outer), dtype=dtype))

    # try to use the dtype specified; fill values at end if necessary
    # collect all possible ilocs, and remove as observed; if any remain, we have fill targets
    iloc_not_found: tp.Set[int] = set(range(len(index_outer)))
    # each group forms a row, each label a value in the index
    for label, _, part in blocks.group(axis=0, key=group_key, kind=kind):
        iloc: int = index_outer._loc_to_iloc(label) #type: ignore
        iloc_not_found.remove(iloc)
        if func_no:
            if len(part) != 1:
                raise RuntimeError('pivot requires aggregation of values; provide a `func` argument.')
            for arrays_key, column_key in enumerate(data_fields_iloc):
                # this is equivalent to extracting a row, but doing so would force a type consolidation
                arrays[arrays_key][iloc] = part._extract(0, column_key)
        elif func_single:
            for arrays_key, column_key in enumerate(data_fields_iloc):
                arrays[arrays_key][iloc] = func_single(part._extract_array_column(column_key))
        else:
            arrays_key = 0
            for column_key in data_fields_iloc:
                values = part._extract_array_column(column_key)
                for _, func in func_map:
                    arrays[arrays_key][iloc] = func(values)
                    arrays_key += 1

    if iloc_not_found:
        # we did not fill all arrays and have values that need to be filled
        # order does not matter
        fill_targets = list(iloc_not_found)
        # mutate in place then make immutable
        for arrays_key in range(len(arrays)): #pylint: disable=C0200
            array = arrays[arrays_key]
            if not array.__class__ is np.ndarray: # a list
                array, _ = iterable_to_array_1d(array, count=len(index_outer))
                array.flags.writeable = True
                arrays[arrays_key] = array # restore new array
            dtype_resolved = resolve_dtype(array.dtype, fill_value_dtype) # type: ignore
            if array.dtype != dtype_resolved: # type: ignore
                array = array.astype(dtype_resolved) #type: ignore
                arrays[arrays_key] = array # re-assign new array
            array[fill_targets] = fill_value # type: ignore
            array.flags.writeable = False # type: ignore
    else:
        for arrays_key in range(len(arrays)): #pylint: disable=C0200
            array = arrays[arrays_key]
            if not array.__class__ is np.ndarray: # a list
                array, _ = iterable_to_array_1d(array, count=len(index_outer))
                arrays[arrays_key] = array # re-assign new array
            array.flags.writeable = False # type: ignore
    return arrays # type: ignore # we have converted all sequences to arrays at this point



def pivot_items_to_block(*,
        blocks: TypeBlocks,
        group_fields_iloc: tp.List[int],
        group_depth: int,
        data_field_iloc: int,
        func_single: tp.Optional[TUFunc],
        dtype: tp.Optional[TDtypeAny],
        fill_value: tp.Any,
        fill_value_dtype: TDtypeAny,
        index_outer: 'IndexBase',
        kind: TSortKinds,
        ) -> TNDArrayAny:
    '''
    Specialized generator of pairs for when we have only one data_field and one function.
    '''
    from static_frame.core.series import Series
    group_key: tp.List[int] | int = (group_fields_iloc if group_depth > 1
            else group_fields_iloc[0])

    if func_single and dtype is not None:
        array = np.full(len(index_outer),
                fill_value,
                dtype=resolve_dtype(dtype, fill_value_dtype),
                )
        for label, _, values in blocks.group_extract(
                axis=0,
                key=group_key,
                extract=data_field_iloc,
                kind=kind,
                ):
            array[index_outer._loc_to_iloc(label)] = func_single(values)
        array.flags.writeable = False
        return array

    if func_single and dtype is None:
        def gen() -> tp.Iterable[tp.Tuple[TLabel, tp.Any]]:
            for label, _, values in blocks.group_extract(
                    axis=0,
                    key=group_key,
                    extract=data_field_iloc,
                    kind=kind,
                    ):
                yield index_outer._loc_to_iloc(label), func_single(values) # pyright: ignore

        post = Series[tp.Any, tp.Any].from_items(gen())
        if len(post) == len(index_outer):
            array = np.empty(len(index_outer), dtype=post.dtype)
        else:
            array = np.full(len(index_outer),
                    fill_value,
                    dtype=resolve_dtype(post.dtype, fill_value_dtype),
                    )
        array[post.index.values] = post.values
        array.flags.writeable = False
        return array

    # func_no scenario as no mapping here
    if group_depth == 1:
        labels = [index_outer._loc_to_iloc(label) for label in blocks._extract_array_column(group_key)] # type: ignore
    else:
        # NOTE: might replace _extract_array_column with an iterator of tuples
        labels = [index_outer._loc_to_iloc(tuple(label)) for label in blocks._extract_array(column_key=group_key)]

    values = blocks._extract_array_column(data_field_iloc)
    if len(values) == len(index_outer):
        array = np.empty(len(index_outer), dtype=dtype)
    else:
        array = np.full(len(index_outer),
                fill_value,
                dtype=resolve_dtype(values.dtype, fill_value_dtype),
                )
    array[labels] = values
    array.flags.writeable = False
    return array

def pivot_items_to_frame(*,
        blocks: TypeBlocks,
        group_fields_iloc: tp.List[int],
        group_depth: int,
        data_field_iloc: int,
        func_single: tp.Optional[TCallableAny],
        frame_cls: tp.Type[TFrameAny],
        name: TName,
        dtype: TDtypeAny | None,
        index_constructor: TIndexCtor,
        columns_constructor: TIndexCtor,
        kind: TSortKinds,
        ) -> TFrameAny:
    '''
    Specialized generator of pairs for when we have only one data_field and one function.
    This version returns a Frame.
    '''

    from static_frame.core.series import Series
    group_key: tp.List[int] | int = (group_fields_iloc if group_depth > 1
            else group_fields_iloc[0])

    if func_single:
        labels: tp.List[TLabel] = []
        values = []
        for label, _, v in blocks.group_extract(
                axis=0,
                key=group_key,
                extract=data_field_iloc,
                kind=kind,
                ):
            labels.append(label)
            values.append(func_single(v))

        if dtype is None:
            array, _ = iterable_to_array_1d(values, count=len(values))
        else:
            array = np.array(values, dtype=dtype)
        array.flags.writeable = False
        index = index_constructor(labels)
        return frame_cls.from_elements(array,
                index=index,
                own_index=True,
                columns=(name,),
                columns_constructor=columns_constructor,
                )
    # func_no scenario
    if group_depth == 1:
        index = index_constructor(blocks._extract_array_column(group_key)) # type: ignore[arg-type]
    else:
        index = index_constructor(tuple(label) for label in blocks._extract_array(column_key=group_key))

    array = blocks._extract_array_column(data_field_iloc)
    return frame_cls.from_elements(array,
            index=index,
            own_index=True,
            columns=(name,),
            columns_constructor=columns_constructor,
            )


def pivot_core(
        *,
        frame: TFrameAny,
        index_fields: tp.List[TLabel],
        columns_fields: tp.List[TLabel],
        data_fields: tp.List[TLabel],
        func_fields: tp.Tuple[TLabel, ...],
        func_single: tp.Optional[TUFunc],
        func_map: tp.Sequence[tp.Tuple[TLabel, TUFunc]],
        fill_value: object = np.nan,
        index_constructor: TIndexCtorSpecifier = None,
        kind: TSortKinds = DEFAULT_FAST_SORT_KIND,
        ) -> TFrameAny:
    '''Core implementation of Frame.pivot(). The Frame has already been reduced to just relevant columns, and all fields groups are normalized as lists of hashables.
    '''
    from static_frame.core.frame import Frame
    from static_frame.core.series import Series

    func_no = func_single is None and func_map == ()

    data_fields_len = len(data_fields)
    index_depth = len(index_fields)

    # all are lists of hashables; get converted to lists of integers
    columns_loc_to_iloc = frame.columns._loc_to_iloc
    index_fields_iloc: tp.List[int] = columns_loc_to_iloc(index_fields) #type: ignore
    data_fields_iloc: tp.List[int] = columns_loc_to_iloc(data_fields) #type: ignore
    columns_fields_iloc: tp.List[int] = columns_loc_to_iloc(columns_fields) #type: ignore

    # For data fields, we add the field name, not the field values, to the columns.
    columns_name = tuple(columns_fields)
    if data_fields_len > 1 or not columns_fields:
        # if no columns_fields, have to add values label
        columns_name = tuple(chain(columns_fields, ('values',)))
    if len(func_map) > 1:
        columns_name = columns_name + ('func',)

    columns_depth = len(columns_name)
    columns_constructor: TIndexCtor

    if columns_depth == 1:
        columns_name = columns_name[0] # type: ignore
        columns_constructor = partial(frame._COLUMNS_CONSTRUCTOR, name=columns_name)
    else:
        columns_constructor = partial(frame._COLUMNS_HIERARCHY_CONSTRUCTOR.from_labels,
                depth_reference=columns_depth,
                name=columns_name)

    dtype_single: TDtypeAny | None
    dtype_map = frame.dtypes # returns a Series

    dtypes_per_data_fields: tp.Tuple[TDtypeAny | None, ...]
    if func_no:
        dtypes_per_data_fields = tuple(dtype_map[field] for field in data_fields)
        if data_fields_len == 1:
            dtype_single = dtype_map[data_fields[0]]
    else:
        dtypes_per_data_fields = tuple(pivot_records_dtypes(
                dtype_map=dtype_map,
                data_fields=data_fields,
                func_single=func_single,
                func_map=func_map,
                ))
        if func_single and data_fields_len == 1:
            dtype_single = ufunc_dtype_to_dtype(func_single, dtype_map[data_fields[0]])

    fill_value_dtype = dtype_from_element(fill_value)
    blocks = frame._blocks

    #---------------------------------------------------------------------------
    # First major branch: if we are only grouping be index fields. This can be done in a single group-by operation on those fields. The final index is not known until the group-by is performed.
    if not columns_fields: # group by is only index_fields
        columns = data_fields if (func_no or func_single) else tuple(
                product(data_fields, func_fields)
                )
        # NOTE: at this time we do not automatically give back an IndexHierarchy when index_depth is == 1, as the order of the resultant values may not be hierarchable.
        name_index = index_fields[0] if index_depth == 1 else tuple(index_fields)
        if index_constructor:
            index_constructor = partial(index_constructor, name=name_index)
        else:
            index_constructor = partial(Index, name=name_index)

        if len(columns) == 1:
            # length of columns is equal to length of datafields, func_map not needed
            f = pivot_items_to_frame(blocks=blocks,
                    group_fields_iloc=index_fields_iloc,
                    group_depth=index_depth,
                    data_field_iloc=data_fields_iloc[0],
                    func_single=func_single,
                    frame_cls=frame.__class__,
                    name=columns[0],
                    dtype=dtype_single,
                    index_constructor=index_constructor,
                    columns_constructor=columns_constructor,
                    kind=kind,
                    )
        else:
            f = pivot_records_items_to_frame(
                    blocks=blocks,
                    group_fields_iloc=index_fields_iloc,
                    group_depth=index_depth,
                    data_fields_iloc=data_fields_iloc,
                    func_single=func_single,
                    func_map=func_map,
                    func_no=func_no,
                    kind=kind,
                    columns_constructor=columns_constructor,
                    columns=columns,
                    index_constructor=index_constructor,
                    dtypes=dtypes_per_data_fields,
                    frame_cls=frame.__class__,
                    )
        columns_final = (f.columns.rename(columns_name) if columns_depth == 1
                else columns_constructor(f.columns)) # type: ignore
        return f.relabel(columns=columns_final)

    #---------------------------------------------------------------------------
    # Second major branch: we are grouping by index and columns fields. This is done with an outer and inner gruop by. The index is calculated ahead of time.

    # avoid doing a multi-column-style selection if not needed
    if len(columns_fields) == 1:
        retuple_group_label = True
    else:
        retuple_group_label = False

    columns_loc_to_iloc = frame.columns._loc_to_iloc
    # group by on 1 or more columns fields
    # NOTE: explored doing one group on index and columns that insert into pre-allocated arrays, but that proved slower than this approach
    group_key: int | tp.List[int] = columns_fields_iloc if len(columns_fields_iloc) > 1 else columns_fields_iloc[0]

    index_outer = pivot_outer_index(blocks=blocks, # 16%
                index_fields=index_fields,
                index_fields_iloc=index_fields_iloc,
                index_depth=index_depth,
                index_constructor=index_constructor,
                )

    # collect subframes based on an index of tuples and columns of tuples (if depth > 1)
    sub_blocks = []
    sub_columns_collected: tp.List[TLabel] = []

    for group, _, sub in blocks.group(axis=0, key=group_key, kind=kind): # 40%
        # derive the column fields represented by this group
        sub_columns = extrapolate_column_fields(
                columns_fields,
                group if not retuple_group_label else (group,), # type: ignore
                data_fields,
                func_fields,
                )
        sub_columns_collected.extend(sub_columns)

        # if sub_columns length is 1, that means that we only need to extract one column out of the sub blocks

        if len(sub_columns) == 1: # type: ignore
            sub_blocks.append(pivot_items_to_block(blocks=sub, # 40%
                            group_fields_iloc=index_fields_iloc,
                            group_depth=index_depth,
                            data_field_iloc=data_fields_iloc[0],
                            func_single=func_single,
                            dtype=dtype_single,
                            index_outer=index_outer,
                            fill_value=fill_value,
                            fill_value_dtype=fill_value_dtype,
                            kind=kind,
                            ))
        else:
            sub_blocks.extend(pivot_records_items_to_blocks(
                            blocks=sub,
                            group_fields_iloc=index_fields_iloc,
                            group_depth=index_depth,
                            data_fields_iloc=data_fields_iloc,
                            func_single=func_single,
                            func_map=func_map,
                            func_no=func_no,
                            fill_value=fill_value,
                            fill_value_dtype=fill_value_dtype,
                            index_outer=index_outer,
                            dtypes=dtypes_per_data_fields,
                            kind=kind,
                            ))

    tb = TypeBlocks.from_blocks(sub_blocks)
    return frame.__class__(tb,
            index=index_outer,
            columns=columns_constructor(sub_columns_collected), # type: ignore
            own_data=True,
            own_index=True,
            own_columns=True,
            )


#-------------------------------------------------------------------------------

def pivot_outer_index(
        blocks: TypeBlocks,
        index_fields: tp.Sequence[TLabel],
        index_fields_iloc: tp.Sequence[int],
        index_depth: int,
        index_constructor: TIndexCtorSpecifier = None,
        ) -> IndexBase:

    index_iloc = index_fields_iloc if index_depth > 1 else index_fields_iloc[0]

    if index_depth == 1:
        index_values = ufunc_unique1d(
                blocks._extract_array_column(index_iloc), # type: ignore
                )
        index_values.flags.writeable = False
        name = index_fields[0]
        index_inner = index_from_optional_constructor(
                index_values,
                default_constructor=partial(Index, name=name),
                explicit_constructor=None if index_constructor is None else partial(index_constructor, name=name),
                )
    else: # > 1
        # NOTE: this might force type an undesirable consolidation
        index_values = ufunc_unique(
                blocks._extract_array(column_key=index_iloc), # type: ignore
                axis=0)
        index_values.flags.writeable = False
        # NOTE: if index_types need to be provided to an IH here, they must be partialed in the single-argument index_constructor
        name = tuple(index_fields)
        index_inner = index_from_optional_constructor( # type: ignore
                index_values,
                default_constructor=partial(
                        IndexHierarchy.from_values_per_depth,
                        name=name,
                        ),
                explicit_constructor=None if index_constructor is None else partial(index_constructor, name=name),
                ).flat() # pyright: ignore
    return index_inner


#-------------------------------------------------------------------------------

class PivotIndexMap(tp.NamedTuple):
    targets_unique: tp.Iterable[TLabel]
    target_depth: int
    target_select: TNDArrayAny
    group_to_target_map: tp.Dict[tp.Optional[TLabel], tp.Dict[tp.Any, int]]
    group_depth: int
    group_select: TNDArrayAny
    group_to_dtype: tp.Dict[TLabel | None, TDtypeAny]

def pivot_index_map(*,
        index_src: IndexBase,
        depth_level: TDepthLevel,
        dtypes_src: TNDArrayAny | None, # array of dtypes
        ) -> PivotIndexMap:
    '''
    Args:
        dtypes_src: must be of length equal to axis
    '''
    # We are always moving levels from one axis to another; after application, the expanded axis will always be hierarchical, while the contracted axis may or may not be. From the contract axis, we need to divide the depths into two categories: targets (the depths to be moved and added to expand axis) and groups (unique combinations that remain on the contract axis after removing targets).

    # Unique target labels are added to labels on the expand axis; unique group labels become the new contract axis.

    target_select = np.full(index_src.depth, False)
    target_select[depth_level] = True
    group_select = ~target_select

    group_arrays = []
    target_arrays = []
    for i, v in enumerate(target_select):
        if v:
            target_arrays.append(index_src.values_at_depth(i))
        else:
            group_arrays.append(index_src.values_at_depth(i))

    group_depth = len(group_arrays)
    target_depth = len(target_arrays)
    group_to_dtype: tp.Dict[tp.Optional[TLabel], TDtypeAny] = {}
    targets_unique: tp.Iterable[TLabel]

    group_to_target_map: tp.Dict[tp.Any, tp.Dict[tp.Any, int]]
    if group_depth == 0:
        # targets must be a tuple
        group_to_target_map = {
                None: {v: idx for idx, v in enumerate(zip(*target_arrays))}
                }
        targets_unique = [k for k in group_to_target_map[None]]
        if dtypes_src is not None:
            group_to_dtype[None] = resolve_dtype_iter(dtypes_src)
    else:
        group_to_target_map = defaultdict(dict)
        targets_unique = {} # Store targets in order observed

        for axis_idx, (group, target, dtype) in enumerate(zip( # type: ignore
                zip(*group_arrays), # get tuples of len 1 to depth
                zip(*target_arrays),
                (dtypes_src if dtypes_src is not None else repeat(None)),
                )):
            if group_depth == 1:
                group = group[0]
            # targets are transfered labels; groups are the new columns
            group_to_target_map[group][target] = axis_idx
            targets_unique[target] = None #type: ignore

            if dtypes_src is not None:
                if group in group_to_dtype:
                    group_to_dtype[group] = resolve_dtype(group_to_dtype[group], dtype)
                else:
                    group_to_dtype[group] = dtype

    return PivotIndexMap( #pylint: disable=E1120
            targets_unique=targets_unique,
            target_depth=target_depth,
            target_select=target_select,
            group_to_target_map=group_to_target_map,
            group_depth=group_depth,
            group_select=group_select,
            group_to_dtype=group_to_dtype
            )


#-------------------------------------------------------------------------------
class PivotDeriveConstructors(tp.NamedTuple):
    contract_dst: tp.Optional[tp.Iterable[TLabel]]
    contract_constructor: TIndexCtorSpecifier
    expand_constructor: TIndexCtorSpecifier

def pivot_derive_constructors(*,
        contract_src: IndexBase,
        expand_src: IndexBase,
        group_select: TNDArrayAny, # Boolean
        group_depth: int,
        target_select: TNDArrayAny,
        group_to_target_map: tp.Dict[tp.Optional[TLabel], tp.Dict[tp.Any, int]],
        expand_is_columns: bool,
        frame_cls: tp.Type[TFrameAny],
        ) -> PivotDeriveConstructors:
    '''
    pivot_stack: columns is contract, index is expand
    pivot_unstack: index is contract, columns is expand
    '''
    # NOTE: group_select, target_select operate on the contract axis
    if expand_is_columns:
        contract_cls = Index
        contract_cls_hierarchy = IndexHierarchy
        expand_cls_hierarchy = frame_cls._COLUMNS_HIERARCHY_CONSTRUCTOR
    else: # contract is columns
        contract_cls = frame_cls._COLUMNS_CONSTRUCTOR
        contract_cls_hierarchy = frame_cls._COLUMNS_HIERARCHY_CONSTRUCTOR
        expand_cls_hierarchy = IndexHierarchy

    # NOTE: not propagating name attr, as not obvious how it should when depths are exiting and entering

    # contract axis may or may not be IndexHierarchy after extracting depths
    if contract_src.depth == 1: # will removed that one level, thus need IndexAuto
        contract_dst = None
        contract_constructor = contract_cls
    else:
        contract_src_types = contract_src.index_types.values
        contract_dst_types = contract_src_types[group_select]
        if group_depth == 0:
            contract_dst = None
            contract_constructor = contract_cls
        elif group_depth == 1:
            contract_dst = list(group_to_target_map.keys())
            contract_constructor = contract_dst_types[0]
        else:
            contract_dst = list(group_to_target_map.keys())
            contract_constructor = partial( #type: ignore
                    contract_cls_hierarchy.from_labels,
                    index_constructors=contract_dst_types,
                    )

    # expand axis will always be IndexHierarchy after adding depth
    if expand_src.depth == 1:
        expand_types = [expand_src.__class__]
    else:
        expand_types = expand_src.index_types.values.tolist()

    if contract_src.depth == 1:
        expand_types.append(contract_src.__class__)
    else:
        expand_types.extend(contract_src_types[target_select])

    expand_constructor: TIndexHierarchyCtor = partial(
            expand_cls_hierarchy.from_labels,
            index_constructors=expand_types,
            # name=expand_src.name,
            )

    # NOTE: expand_dst labels will come from the values generator
    return PivotDeriveConstructors( #pylint: disable=E1120
            contract_dst=contract_dst,
            contract_constructor=contract_constructor,
            expand_constructor=expand_constructor,
            )


