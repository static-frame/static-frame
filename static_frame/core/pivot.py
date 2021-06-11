import typing as tp
from functools import partial
from collections import defaultdict
from itertools import repeat
from itertools import product

import numpy as np
from arraykit import resolve_dtype
from arraykit import resolve_dtype_iter

from static_frame.core.index_base import IndexBase
from static_frame.core.index import Index
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.util import DepthLevelSpecifier
from static_frame.core.util import IndexConstructor
from static_frame.core.util import UFunc


if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame #pylint: disable=W0611 #pragma: no cover



#-------------------------------------------------------------------------------
# for Frame.pivot
def extrapolate_column_fields(
        columns_fields: tp.Sequence[tp.Hashable],
        group: tp.Tuple[tp.Hashable, ...],
        data_fields: tp.Sequence[tp.Hashable],
        func_fields: tp.Iterable[tp.Hashable],
        ) -> tp.Iterable[tp.Hashable]:
    '''Used in Frame.pivot.

    Args:
        group: a unique label from the the result of doing a group-by with the `columns_fields`.
    '''
    columns_fields_len = len(columns_fields)
    data_fields_len = len(data_fields)

    sub_columns: tp.Iterable[tp.Hashable]

    if columns_fields_len == 1 and data_fields_len == 1:
        if not func_fields:
            sub_columns = group # already a tuple
        else:
            sub_columns = [group + (label,) for label in func_fields]
    elif columns_fields_len == 1 and data_fields_len > 1: # create a sub heading for each data field
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
        frame: 'Frame',
        data_fields: tp.Iterable[tp.Hashable],
        func_single: tp.Optional[UFunc],
        func_map: tp.Sequence[tp.Tuple[tp.Hashable, UFunc]]
        ) -> tp.Iterator[np.dtype]:
    dtypes = frame.dtypes
    for field in data_fields:
        dtype = dtypes[field]
        if func_single:
            yield dtype
        else: # we assume
            for _, func in func_map:
                yield None # do not know what func result will be


def pivot_records_items(
        frame: 'Frame',
        group_fields: tp.Iterable[tp.Hashable],
        group_depth: int,
        data_fields: tp.Iterable[tp.Hashable],
        func_single: tp.Optional[UFunc],
        func_map: tp.Sequence[tp.Tuple[tp.Hashable, UFunc]]
        ) -> tp.Iterator[tp.Tuple[tp.Hashable, tp.Sequence[tp.Any]]]:

    take_group_index = group_depth > 1

    for group_index, part in frame.iter_group_items(group_fields):
        label = group_index if take_group_index else group_index[0]
        record = []
        part_columns_loc_to_iloc = part.columns._loc_to_iloc
        for field in data_fields:
            values = part._blocks._extract_array(
                    row_key=None,
                    column_key=part_columns_loc_to_iloc(field),
                    )
            if func_single:
                if len(values) == 1:
                    record.append(values[0])
                else:
                    record.append(func_single(values))
            else:
                for _, func in func_map:
                    if len(values) == 1:
                        record.append(values[0])
                    else:
                        record.append(func(values))
        yield label, record

def pivot_items(
        frame: 'Frame',
        group_fields: tp.Iterable[tp.Hashable],
        group_depth: int,
        data_fields: tp.Sequence[tp.Hashable],
        func_single: UFunc,
        ) -> tp.Iterator[tp.Tuple[tp.Hashable, tp.Any]]:
    '''
    Specialized generator of Pairs for when group_fields has been reduced to a single column.
    '''
    take_group = group_depth > 1

    for group, sub in frame.iter_group_items(group_fields):
        label = group if take_group else group[0]
        values = sub._blocks._extract_array(
                row_key=None,
                column_key=sub.columns._loc_to_iloc(data_fields[0]),
                )
        if len(values) == 1:
            yield label, values[0]
        else: # can be sure we only have func_single
            yield label, func_single(values)

#-------------------------------------------------------------------------------
class PivotIndexMap(tp.NamedTuple):
    targets_unique: tp.Iterable[tp.Hashable]
    target_depth: int
    target_select: np.ndarray
    group_to_target_map: tp.Dict[tp.Optional[tp.Hashable], tp.Dict[tp.Any, int]]
    group_depth: int
    group_select: np.ndarray
    group_to_dtype: tp.Dict[tp.Optional[tp.Hashable], np.dtype]

def pivot_index_map(*,
        index_src: IndexBase,
        depth_level: DepthLevelSpecifier,
        dtypes_src: tp.Optional[tp.Sequence[np.dtype]],
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
    group_to_dtype: tp.Dict[tp.Optional[tp.Hashable], np.dtype] = {}
    targets_unique: tp.Iterable[tp.Hashable]

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
        targets_unique = dict() # Store targets in order observed

        for axis_idx, (group, target, dtype) in enumerate(zip(
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
            group_to_target_map=group_to_target_map, #type: ignore
            group_depth=group_depth,
            group_select=group_select,
            group_to_dtype=group_to_dtype
            )


#-------------------------------------------------------------------------------
class PivotDeriveConstructors(tp.NamedTuple):
    contract_dst: tp.Optional[tp.Iterable[tp.Hashable]]
    contract_constructor: IndexConstructor
    expand_constructor: IndexConstructor

def pivot_derive_constructors(*,
        contract_src: IndexBase,
        expand_src: IndexBase,
        group_select: np.ndarray, # Boolean
        group_depth: int,
        target_select: np.ndarray,
        # target_depth: int,
        group_to_target_map: tp.Dict[tp.Hashable, tp.Tuple[tp.Hashable]],
        expand_is_columns: bool,
        frame_cls: tp.Type['Frame'],
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
        contract_src_types = contract_src.index_types.values #type: ignore
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
        expand_types = list(expand_src._levels.index_types()) #type: ignore

    if contract_src.depth == 1:
        expand_types.append(contract_src.__class__)
    else:
        expand_types.extend(contract_src_types[target_select])

    expand_constructor = partial(
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


