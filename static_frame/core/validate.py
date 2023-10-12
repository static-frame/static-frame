from __future__ import annotations

import types
import typing
import warnings
from collections import deque
from collections.abc import MutableMapping
from collections.abc import Sequence
from enum import Enum
from functools import wraps
from inspect import BoundArguments
from inspect import Signature
from itertools import chain
from itertools import repeat

import numpy as np
import typing_extensions as tp

from static_frame.core.frame import Frame
from static_frame.core.index import Index
from static_frame.core.index_base import IndexBase
from static_frame.core.index_datetime import IndexDatetime
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.series import Series
from static_frame.core.util import TLabel

# _UnionGenericAlias comes from tp.Union, UnionType from | expressions
# tp.Optional returns a _UnionGenericAlias with later Python, but a _GenericAlias with 3.8

if tp.TYPE_CHECKING:
    DtypeAny = np.dtype[tp.Any] # pylint: disable=W0611 #pragma: no cover


def _iter_generic_classes() -> tp.Iterable[tp.Type[tp.Any]]:
    if t := getattr(types, 'GenericAlias', None):
        yield t
    if t := getattr(typing, '_GenericAlias', None):
        yield t # pyright: ignore

GENERIC_TYPES = tuple(_iter_generic_classes())

def _iter_union_classes() -> tp.Iterable[tp.Type[tp.Any]]:
    if t := getattr(types, 'UnionType', None):
        yield t
    if t := getattr(typing, '_UnionGenericAlias', None):
        yield t # pyright: ignore

UNION_TYPES = tuple(_iter_union_classes())

def _iter_unpack_classes() -> tp.Iterable[tp.Type[tp.Any]]:
    # NOTE: type extensions Unpack is not equal to typing.Unpack
    if t := getattr(tp, 'Unpack', None):
        yield t
    if t := getattr(typing, 'Unpack', None):
        yield t # pyright: ignore

UNPACK_TYPES = tuple(_iter_unpack_classes())

def is_generic(hint: tp.Any) -> bool:
    return isinstance(hint, GENERIC_TYPES)

def is_union(hint: tp.Any) -> bool:
    if UNION_TYPES:
        return isinstance(hint, UNION_TYPES)
    elif isinstance(hint, GENERIC_TYPES):
        return tp.get_origin(hint) is tp.Union
    return False #pragma: no cover

def is_unpack(hint: tp.Any) -> bool:
    # NOTE: cannot use isinstance or issubclass with Unpack
    return hint in UNPACK_TYPES

#-------------------------------------------------------------------------------

TParent = tp.Tuple[tp.Any, ...]
# A validation record can be used to queue checks or report errors
TValidation = tp.Tuple[tp.Any, tp.Any, TParent]

#-------------------------------------------------------------------------------
# error reporting, presentation

ERROR_MESSAGE_TYPE = object()

def to_name(v: tp.Any,
        func_to_str: tp.Callable[..., str] = str,
        ) -> str:
    if is_generic(v):
        if hasattr(v, '__name__'):
            name = v.__name__
        else:
            # for older Python, not all generics have __name__
            origin = tp.get_origin(v)
            if hasattr(origin, '__name__'):
                name = origin.__name__
            elif is_unpack(origin): # needed for backwards compat
                name = 'Unpack'
            else:
                name = str(origin)
        s = f'{name}[{", ".join(to_name(q) for q in tp.get_args(v))}]'
    elif hasattr(v, '__name__'):
        s = v.__name__
    elif v is ...:
        s = '...'
    else:
        s = func_to_str(v)
    return s

def to_signature(
        sig: BoundArguments,
        hints: tp.Mapping[str, tp.Any]) -> str:
    msg = []
    for k in sig.arguments:
        msg.append(f'{k}: {to_name(hints.get(k, tp.Any))}')
    r = to_name(hints.get('return', tp.Any))
    return f'({", ".join(msg)}) -> {r}'



class ClinicResult:
    '''A ``ClinicResult`` instance stores zero or more error messages resulting from a check.
    '''
    __slots__ = ('_log',)

    _LINE = '─'
    _CORNER = '└'
    _WIDTH = 4

    def __init__(self, log: tp.Sequence[TValidation]) -> None:
        self._log = log

    def __iter__(self) -> tp.Iterator[TValidation]:
        return self._log.__iter__()

    def __len__(self) -> int:
        return len(self._log)

    def __bool__(self) -> bool:
        '''Return True if there are validation issues.
        '''
        return bool(self._log)

    @property
    def validated(self) -> bool:
        return not bool(self._log)

    @classmethod
    def _get_indent(cls, size: int) -> str:
        if size < 1:
            return ''
        c_width = cls._WIDTH - 2
        l_width = cls._WIDTH * (size - 1)
        return f'{" " * l_width}{cls._CORNER}{cls._LINE * c_width} '

    def to_str(self) -> str:
        '''Return error messages as a formatted string with line breaks and indentation.
        '''
        msg = []
        for v, h, p in self._log:
            if p:
                # path = ', '.join(to_name(n) for n in p)
                path_components = []
                for i, pc in enumerate(p):
                    path_components.append(f'{self._get_indent(i)}{to_name(pc)}')
                path = '\n'.join(path_components)
                i_next = i + 1
            else:
                path = ''
                i_next = 1

            if v is ERROR_MESSAGE_TYPE: # in this case, do not use the value
                error_msg = f'{h}'
            else:
                error_msg = f'Expected {to_name(h)}, provided {to_name(type(v))} invalid'

            if not path:
                msg.append(f'\n{error_msg}')
            else:
                msg.append(f'\nIn {path}\n{self._get_indent(i_next)}{error_msg}')

        return ''.join(msg)

    def __repr__(self) -> str:
        log_len = len(self)
        return f'<ClinicResult: {log_len} {"errors" if log_len != 1 else "error"}>'


class ClinicError(TypeError):
    '''A TypeError subclass for exposing check errors.
    '''
    def __init__(self, cr: ClinicResult) -> None:
        TypeError.__init__(self, cr.to_str())


#-------------------------------------------------------------------------------

class Constraint:
    '''Base class of all run-time constraints, deployed in Annotated generics.
    '''
    __slots__: tp.Tuple[str, ...] = ()

    def iter_error_log(self,
            value: tp.Any,
            hint: tp.Any,
            parent: TParent,
            ) -> tp.Iterator[TValidation]:
        raise NotImplementedError() #pragma: no cover

    def __repr__(self) -> str:
        args = ', '.join(to_name(getattr(self, v)) for v in self.__slots__)
        return f'{self.__class__.__name__}({args})'

class Name(Constraint):
    '''Constraint to validate the name of a container.
    '''
    __slots__ = ('_name',)

    def __init__(self, name: TLabel):
        self._name = name

    def iter_error_log(self,
            value: tp.Any,
            hint: tp.Any,
            parent: TParent,
            ) -> tp.Iterator[TValidation]:
        # returning anything is an error
        if (n := value.name) != self._name:
            yield ERROR_MESSAGE_TYPE, f'Expected name {self._name!r}, provided name {n!r}', parent

class Len(Constraint):
    '''Constraint to validate the length of a container.
    '''

    __slots__ = ('_len',)

    def __init__(self, len: int):
        self._len = len

    def iter_error_log(self,
            value: tp.Any,
            hint: tp.Any,
            parent: TParent,
            ) -> tp.Iterator[TValidation]:
        if (vl := len(value)) != self._len:
            yield ERROR_MESSAGE_TYPE, f'Expected length {self._len}, provided length {vl}', parent


# might accept regular expression objects as label entries?
class Labels(Constraint):
    __slots__ = ('_labels',)

    def __init__(self, *labels: tp.Sequence[TLabel]):
        self._labels: tp.Sequence[TLabel] = labels

    def __repr__(self) -> str:
        args = ', '.join(to_name(v, func_to_str=repr) for v in self._labels)
        return f'{self.__class__.__name__}({args})'


    @staticmethod
    def _prepare_remainder(labels: tp.Sequence[TLabel]) -> str:
        # always drop leading ellipses
        if labels[0] is ...:
            labels = labels[1:]
        return ', '.join((repr(l) if l is not ... else '...') for l in labels)

    def iter_error_log(self,
            value: tp.Any,
            hint: tp.Any,
            parent: TParent,
            ) -> tp.Iterator[TValidation]:

        if not isinstance(value, IndexBase):
            yield ERROR_MESSAGE_TYPE, f'Expected {self} to be used on Index or IndexHierarchy, not provided {to_name(type(value))}', parent
        else:
            pos_e = 0 # position expected
            len_e = len(self._labels)

            for label_p in value:
                if pos_e >= len_e:
                    yield ERROR_MESSAGE_TYPE, f'Expected labels exhausted at provided {label_p!r}', parent
                    break
                label_e = self._labels[pos_e]

                if label_e is not ...:
                    if label_p != label_e:
                        yield ERROR_MESSAGE_TYPE, f'Expected {label_e!r}, provided {label_p!r}', parent
                    pos_e += 1
                # label_e is an Ellipses; either find next as match or continue with Ellipses
                elif pos_e + 1 < len_e: # more expected labels available
                    label_next_e = self._labels[pos_e + 1]
                    if label_next_e is ...:
                        yield ERROR_MESSAGE_TYPE, 'Expected cannot be defined with adjacent ellipses', parent
                        break
                    if label_p == label_next_e:
                        pos_e += 2 # skip the compared value, prepare to get next
                # else, last expected value is Ellipses
            else: # no break, evaluate final conditions
                if pos_e == len_e - 1 and label_e is ...:
                    pass # ended on elipses
                elif pos_e < len_e:
                    remainder = self._prepare_remainder(self._labels[pos_e:])
                    yield ERROR_MESSAGE_TYPE, f'Expected has unmatched labels {remainder}', parent


class Validator(Constraint):
    '''Apply a constraint to a container with an arbitrary function.
    '''

    __slots__ = ('_validator',)

    def __init__(self, validator: tp.Callable[..., bool]):
        self._validator: tp.Callable[..., bool] = validator

    @staticmethod
    def _prepare_callable(validator: tp.Callable[..., bool]) -> str:
        return validator.__name__

    def iter_error_log(self,
            value: tp.Any,
            hint: tp.Any,
            parent: TParent,
            ) -> tp.Iterator[TValidation]:
        post = self._validator(value)
        if post is False:
            yield ERROR_MESSAGE_TYPE, f'{to_name(type(value))} failed validation with {self._prepare_callable(self._validator)}', parent

#-------------------------------------------------------------------------------
# handlers for getting components out of generics

def iter_sequence_checks(
        value: tp.Any,
        hint: tp.Any,
        parent: TParent,
        ) -> tp.Iterable[TValidation]:
    [h_component] = tp.get_args(hint)
    for v in value:
        yield v, h_component, parent

def iter_tuple_checks(
        value: tp.Any,
        hint: tp.Any,
        parent: TParent,
        ) -> tp.Iterable[TValidation]:
    h_components = tp.get_args(hint)
    if h_components[-1] is ...:
        if len(h_components) != 2 or h_components[0] is ...:
            yield ERROR_MESSAGE_TYPE, 'Invalid ellipses usage', parent
        else:
            h = h_components[0]
            for v in value:
                yield v, h, parent
    else:
        if (h_len := len(h_components)) != len(value):
            msg = f'Expected tuple length of {h_len}, provided tuple length of {len(value)}'
            yield ERROR_MESSAGE_TYPE, msg, parent
        for v, h in zip(value, h_components):
            # NOTE: find bad usage of ellipses
            yield v, h, parent

def iter_mapping_checks(
        value: tp.Any,
        hint: tp.Any,
        parent: TParent,
        ) -> tp.Iterable[TValidation]:
    [h_keys, h_values] = tp.get_args(hint)
    for v, h in chain(
            zip(value.keys(), repeat(h_keys)),
            zip(value.values(), repeat(h_values)),
            ):
        yield v, h, parent


def iter_typeddict(
        value: tp.Any,
        hint: tp.Any,
        parent: TParent,
        ) -> tp.Iterable[TValidation]:

    # hint is a typedict class; returns dict of key name and value
    hints = tp.get_type_hints(hint, include_extras=True)
    total = hint.__total__

    for k, hint_key in hints.items(): # iterating hints retains order
        parent_k = parent + (f'Key {k!r}',)
        k_found = k in value
        if not total or (is_generic(hint_key) and tp.get_origin(hint_key) is tp.NotRequired):
            required = False
        else:
            required = True

        if k_found:
            yield value[k], hint_key, parent_k
        elif not k_found and required:
            yield ERROR_MESSAGE_TYPE, 'Expected key not provided', parent_k

    # get over-specified keys
    if keys_os := value.keys() - hints.keys():
        yield ERROR_MESSAGE_TYPE, f"Keys provided not expected: {', '.join(f'{k!r}' for k in sorted(keys_os))}", parent

# NOTE: For SF containers, we create an instance of dtype.type() so as to not modify h_generic, as it might be Union or other generic that cannot be wrapped in a tp.Type. This returns a "sample" instance of the type that can be used for testing. Caching this value does not seem a benefit.

def iter_series_checks(
        value: tp.Any,
        hint: tp.Any,
        parent: TParent,
        ) -> tp.Iterable[TValidation]:
    h_index, h_generic = tp.get_args(hint) # there must be two
    yield value.index, h_index, parent
    yield value.dtype.type(), h_generic, parent

def iter_index_checks(
        value: tp.Any,
        hint: tp.Any,
        parent: TParent,
        ) -> tp.Iterable[TValidation]:
    [h_generic] = tp.get_args(hint)
    yield value.dtype.type(), h_generic, parent

def iter_index_hierarchy_checks(
        value: tp.Any,
        hint: tp.Any,
        parent: TParent,
        ) -> tp.Iterable[TValidation]:
    h_generics = tp.get_args(hint)
    h_len = len(h_generics)

    unpack_pos = -1
    for i, h in enumerate(h_generics):
        # must get_origin to id unpack; origin of simplet types is None
        if is_unpack(tp.get_origin((h))):
            unpack_pos = i
            break

    if h_len == 1 and unpack_pos == 0:
        [h_tuple] = tp.get_args(h_generics[0])
        assert issubclass(tuple, tp.get_origin(h_tuple))
        # to support usage of Ellipses, treat this as a hint of a corresponding tuple of types
        yield tuple(value.index_at_depth(i) for i in range(value.depth)), h_tuple, parent

    else:
        col_count = value.shape[1]

        if unpack_pos == -1: # no unpack
            if h_len != col_count:
                # if no unpack and lengths are not equal
                yield ERROR_MESSAGE_TYPE, f'Expected IndexHierarchy has {h_len} dtype, provided IndexHierarchy has {col_count} depth', parent
            else:
                yield from zip(
                        (value.index_at_depth(i) for i in range(value.depth)),
                        h_generics,
                        repeat(parent),
                        )

        else: # unpack found
            if h_len > col_count + 1:
                # if 1 unpack, there cannot be more than width h_generics + 1 for the Unpack
                yield ERROR_MESSAGE_TYPE, f'Expected IndexHierarchy has {h_len - 1} depth (excluding Unpack), provided IndexHierarchy has {col_count} depth', parent
            else:
                h_types_post = h_len - unpack_pos - 1
                depth_post_unpack = col_count - h_types_post

                indexes = tuple(value.index_at_depth(i) for i in range(value.depth))

                index_pre = indexes[:unpack_pos]
                index_unpack = indexes[unpack_pos: depth_post_unpack]
                index_post = indexes[depth_post_unpack:]

                h_pre = h_generics[:unpack_pos]
                h_unpack = h_generics[unpack_pos]
                h_post = h_generics[unpack_pos + 1:]

                # if len(index_pre) != len(h_pre):
                #     yield ERROR_MESSAGE_TYPE, f'Expected IndexHierarchy has {len(h_pre)} depth before Unpack, provided IndexHierarchy has {len(index_pre)} alignable depth', parent
                # if len(index_post) != len(h_post):
                #     yield ERROR_MESSAGE_TYPE, f'Expected IndexHierarchy has {len(h_post)} depth after Unpack, provided IndexHierarchy has {len(index_post)} alignable depth', parent
                # else:

                col_pos = 0
                for index, h in zip(index_pre, h_pre):
                    yield index, h, parent + (f'Depth {col_pos}',)
                    col_pos += 1

                if index_unpack: # we may not have dtypes to compare if fewer
                    [h_tuple] = tp.get_args(h_unpack)
                    assert issubclass(tuple, tp.get_origin(h_tuple))
                    yield tuple(index_unpack), h_tuple, parent + (f'Depths {col_pos} to {depth_post_unpack - 1}',)

                col_pos = depth_post_unpack
                for index, h in zip(index_post, h_post):
                    yield index, h, parent + (f'Depth {col_pos}',)
                    col_pos += 1


def iter_frame_checks(
        value: tp.Any,
        hint: tp.Any,
        parent: TParent,
        ) -> tp.Iterable[TValidation]:

    # NOTE: note: at runtime TypeVarTuple defaults do not return anything
    h_index, h_columns, *h_types = tp.get_args(hint)
    h_types_len = len(h_types)

    yield value.index, h_index, parent
    yield value.columns, h_columns, parent

    unpack_pos = -1
    for i, h in enumerate(h_types):
        # must get_origin to id unpack; origin of simplet types is None
        if is_unpack(tp.get_origin((h))):
            unpack_pos = i
            break

    if h_types_len == 1 and unpack_pos == 0:
        [h_tuple] = tp.get_args(h_types[0])
        assert issubclass(tuple, tp.get_origin(h_tuple))
        # to support usage of Ellipses, treat this as a hint of a corresponding tuple of types
        yield tuple(d.type() for d in value._blocks._iter_dtypes()), h_tuple, parent

    else:
        col_count = value.shape[1]

        if unpack_pos == -1: # no unpack
            if h_types_len != col_count:
                # if no unpack and lengths are not equal
                yield ERROR_MESSAGE_TYPE, f'Expected Frame has {h_types_len} dtype, provided Frame has {col_count} dtype', parent
            else:
                for dt, h in zip(value._blocks._iter_dtypes(), h_types):
                    yield dt.type(), h, parent

        else: # unpack found
            if h_types_len > col_count + 1:
                # if 1 unpack, there cannot be more than width h_types + 1 for the Unpack
                yield ERROR_MESSAGE_TYPE, f'Expected Frame has {h_types_len - 1} dtype (excluding Unpack), provided Frame has {col_count} dtype', parent
            else:
                # in terms of column position, we need to find the first column position after the unpack
                # if unpack pos is 0, and 5 types, 4 are post
                # if unpack pos is 3, and 5 types, 1 is post
                h_types_post = h_types_len - unpack_pos - 1
                col_post_unpack = col_count - h_types_post

                dts = tuple(value._blocks._iter_dtypes())

                dt_pre = dts[:unpack_pos]
                dt_unpack = dts[unpack_pos: col_post_unpack]
                dt_post = dts[col_post_unpack:]

                h_pre = h_types[:unpack_pos]
                h_unpack = h_types[unpack_pos]
                h_post = h_types[unpack_pos + 1:]

                # if len(dt_pre) != len(h_pre):
                #     yield ERROR_MESSAGE_TYPE, f'Expected Frame has {len(h_pre)} dtype before Unpack, provided Frame has {len(dt_pre)} alignable dtype', parent
                # elif len(dt_post) != len(h_post):
                #     yield ERROR_MESSAGE_TYPE, f'Expected Frame has {len(h_post)} dtype after Unpack, provided Frame has {len(dt_post)} alignable dtype', parent
                # else:

                col_pos = 0
                for dt, h in zip(dt_pre, h_pre):
                    yield dt.type(), h, parent + (f'Field {col_pos}',)
                    col_pos += 1

                if dt_unpack: # we may not have dtypes to compare if fewer
                    [h_tuple] = tp.get_args(h_unpack)
                    assert issubclass(tuple, tp.get_origin(h_tuple))
                    yield tuple(d.type() for d in dt_unpack), h_tuple, parent + (f'Fields {col_pos} to {col_post_unpack - 1}',)

                col_pos = col_post_unpack
                for dt, h in zip(dt_post, h_post):
                    yield dt.type(), h, parent + (f'Field {col_pos}',)
                    col_pos += 1


def iter_ndarray_checks(
        value: tp.Any,
        hint: tp.Any,
        parent: TParent,
        ) -> tp.Iterable[TValidation]:
    h_shape, h_dtype = tp.get_args(hint)
    yield value.dtype, h_dtype, parent

def iter_dtype_checks(
        value: tp.Any,
        hint: tp.Any,
        parent: TParent,
        ) -> tp.Iterable[TValidation]:
    [h_generic] = tp.get_args(hint)
    yield value.type(), h_generic, parent

#-------------------------------------------------------------------------------

def _check(
        value: tp.Any,
        hint: tp.Any,
        parent: TParent = (),
        fail_fast: bool = False,
        ) -> ClinicResult:

    # Check queue: queue all checks
    q = deque(((value, hint, parent),))
    # Error log: any entry is considered an error
    e_log: tp.List[TValidation] = []

    def tee_error_or_check(records: tp.Iterable[TValidation]) -> None:
        for record in records:
            if record[0] is ERROR_MESSAGE_TYPE:
                e_log.append(record)
            else:
                q.append(record)

    while q:
        if fail_fast and e_log:
            return ClinicResult(e_log)

        v, h, p = q.popleft()
        # an ERROR_MESSAGE_TYPE should only be used as a place holder in error logs, not queued checkls
        assert v is not ERROR_MESSAGE_TYPE

        if h is tp.Any:
            continue

        p_next = p + (h,)
        if is_union(h):
            # NOTE: must check union types first as tp.Union matches as generic type
            u_log: tp.List[TValidation] = []
            for c_hint in tp.get_args(h): # get components
                # handing one pair at a time with a secondary call will allow nested types in the union to be evaluated on their own
                c_log = _check(v, c_hint, p_next, fail_fast)
                if not c_log: # no error found, can exit
                    break
                u_log.extend(c_log)
            else: # no breaks, so no matches within union
                e_log.extend(u_log)
        elif isinstance(h, Constraint):
            e_log.extend(h.iter_error_log(v, h, p_next))
        elif is_generic(h):
            # have a generic container
            origin = tp.get_origin(h)

            if origin is type: # a tp.Type[x] generic
                [t] = tp.get_args(h)
                try: # the v should be a subclass of t
                    t_check = issubclass(t, v)
                except TypeError:
                    t_check = False
                if t_check:
                    continue
                e_log.append((v, h, p))
            elif origin == tp.Annotated: # NOTE: cannot use `is` due backwards compat
                h_type, *h_annotations = tp.get_args(h)
                # perform the un-annotated check
                q.append((v, h_type, p_next))
                for h_annotation in h_annotations:
                    if isinstance(h_annotation, Constraint):
                        q.append((v, h_annotation, p_next))
            elif origin == tp.Literal: # NOTE: cannot use is due backwards compat
                l_log: tp.List[TValidation] = []
                for l_hint in tp.get_args(h): # get components
                    c_log = _check(v, l_hint, p_next, fail_fast)
                    if not c_log: # no error found, can exit
                        break
                    l_log.extend(c_log)
                else: # no breaks, so no matches within union
                    e_log.extend(l_log)
            elif origin == tp.Required or origin == tp.NotRequired:
                # semantics handled elsewhere, just unpack
                [h_type] = tp.get_args(h)
                q.append((v, h_type, p_next))
            else:
                if not isinstance(v, origin):
                    e_log.append((v, origin, p))
                    continue

                if isinstance(v, tuple):
                    tee_error_or_check(iter_tuple_checks(v, h, p_next))
                elif isinstance(v, Sequence):
                    tee_error_or_check(iter_sequence_checks(v, h, p_next))
                elif isinstance(v, MutableMapping):
                    tee_error_or_check(iter_mapping_checks(v, h, p_next))

                elif isinstance(v, Index):
                    tee_error_or_check(iter_index_checks(v, h, p_next))
                elif isinstance(v, IndexHierarchy):
                    tee_error_or_check(iter_index_hierarchy_checks(v, h, p_next))
                elif isinstance(v, Series):
                    tee_error_or_check(iter_series_checks(v, h, p_next))
                elif isinstance(v, Frame):
                    tee_error_or_check(iter_frame_checks(v, h, p_next))

                elif isinstance(v, np.ndarray):
                    tee_error_or_check(iter_ndarray_checks(v, h, p_next))
                elif isinstance(v, np.dtype):
                    tee_error_or_check(iter_dtype_checks(v, h, p_next))
                else:
                    raise NotImplementedError(f'no handling for generic {origin}') #pragma: no cover
        elif tp.is_typeddict(h):
            tee_error_or_check(iter_typeddict(v, h, p_next))
        elif not isinstance(h, type): # h is a value from a literal
            # must check type: https://peps.python.org/pep-0586/#equivalence-of-two-literals
            if type(v) != type(h) or v != h: # pylint: disable=C0123
                e_log.append((v, h, p))
        else: # h is a non-generic type
            # special cases
            if v.__class__ is bool:
                if h is bool:
                    continue
            # general case
            elif isinstance(v, h):
                continue
            e_log.append((v, h, p))

    return ClinicResult(e_log)

#-------------------------------------------------------------------------------
# public interfaces

def _value_to_hint(value: tp.Any) -> tp.Any: # tp._GenericAlias
    if isinstance(value, type):
        return tp.Type[value]

    if isinstance(value, tuple):
        return value.__class__.__class_getitem__(tuple(_value_to_hint(v) for v in value))

    if isinstance(value, Sequence) and not isinstance(value, str):
        if not len(value):
            return value.__class__.__class_getitem__(tp.Any) # type: ignore[attr-defined]

        # as classes may not be hashable, we key to string name to from a set; this is imperfect
        ut = {v.__class__.__name__: v.__class__ for v in value}
        if len(ut) == 1:
            return value.__class__.__class_getitem__(ut[next(iter(ut.keys()))]) # type: ignore[attr-defined]

        hu = tp.Union.__getitem__(tuple(ut.values())) # pyright: ignore
        return value.__class__.__class_getitem__(hu) # type: ignore[attr-defined]

    if isinstance(value, MutableMapping):
        if not len(value):
            return value.__class__.__class_getitem__((tp.Any, tp.Any)) # type: ignore[attr-defined]

        keys_ut = {k.__class__.__name__: k.__class__ for k in value.keys()}
        values_ut = {v.__class__.__name__: v.__class__ for v in value.values()}

        if len(keys_ut) == 1:
            kt = keys_ut[next(iter(keys_ut.keys()))]
        else:
            kt = tp.Union.__getitem__(tuple(keys_ut.values())) # pyright: ignore

        if len(values_ut) == 1:
            vt = values_ut[next(iter(values_ut.keys()))]
        else:
            vt = tp.Union.__getitem__(tuple(values_ut.values())) # pyright: ignore

        return value.__class__.__class_getitem__((kt, vt)) # type: ignore[attr-defined]

    # --------------------------------------------------------------------------
    # SF containers

    if isinstance(value, Frame):
        hints = [_value_to_hint(value.index), _value_to_hint(value.columns)]
        hints.extend(dt.type().__class__ for dt in value._blocks._iter_dtypes())
        return value.__class__.__class_getitem__(tuple(hints)) # type: ignore

    if isinstance(value, Series):
        return value.__class__[_value_to_hint(value.index), value.dtype.type().__class__] # type: ignore

    # must come before index
    if isinstance(value, IndexDatetime):
        return value.__class__

    if isinstance(value, Index):
        return value.__class__[value.dtype.type().__class__] # type: ignore

    if isinstance(value, IndexHierarchy):
        hints = list(_value_to_hint(value.index_at_depth(i)) for i in range(value.depth))
        return value.__class__.__class_getitem__(tuple(hints)) # type: ignore

    if isinstance(value, np.dtype):
        return np.dtype.__class_getitem__(value.type().__class__)

    if isinstance(value, np.ndarray):
        return value.__class__.__class_getitem__(_value_to_hint(value.dtype))

    return value.__class__


class TypeClinic:
    __slots__ = ('_value',)

    INTERFACE = (
        'to_hint',
        'check',
        'warn',
        '__call__',
        '__repr__',
        )

    def __init__(self, value: tp.Any, /):
        self._value = value

    def to_hint(self) -> tp.Any:
        '''Return the type hint (the type and/or generic aliases necessary) to represent the object given at initialization.
        '''
        # NOTE: this can cache as value assumed immutable
        return _value_to_hint(self._value)

    def __repr__(self) -> str:
        '''Return a compact string representation of the type hint (the type and/or generic aliases necessary) to represent the object given at initialization.
        '''
        return to_name(self.to_hint())

    def check(self,
            hint: tp.Any,
            /, *,
            fail_fast: bool = False,
            ) -> None:
        '''Given a hint (a type and/or generic alias), raise a ``ClinicError`` exception describing the result of the check if an error is found.

        Args:
            fail_fast: If True, return on first failure. If False, all failures are discovered and reported.
        '''
        if cr := self(hint, fail_fast=fail_fast):
            raise ClinicError(cr)

    def warn(self,
            hint: tp.Any,
            /, *,
            fail_fast: bool = False,
            category: tp.Type[Warning] = UserWarning,
            ) -> None:
        '''Given a hint (a type and/or generic alias), issue a warning describing the result of the check if an error is found.

        Args:
            fail_fast: If True, return on first failure. If False, all failures are discovered and reported.
            category: The ``Warning`` subclass to be used for issueing the warning.
        '''
        if cr := self(hint, fail_fast=fail_fast):
            warnings.warn(cr.to_str(), category)


    def __call__(self,
            hint: tp.Any,
            /, *,
            fail_fast: bool = False,
            ) -> ClinicResult:
        '''Given a hint (a type and/or generic alias), return a ``ClinicResult`` object describing the result of the check.

        Args:
            fail_fast: If True, return on first failure. If False, all failures are discovered and reported.
        '''
        return _check(self._value, hint, fail_fast=fail_fast)



class ErrorAction(Enum):
    RAISE = 0
    WARN = 1
    RETURN = 2

def _check_interface(
        func: tp.Callable[..., tp.Any],
        args: tp.Any,
        kwargs: tp.Any,
        fail_fast: bool,
        error_action: ErrorAction,
        category: tp.Type[Warning] = UserWarning,
        ) -> tp.Any:
    # include_extras insures that Annotated generics are returned
    hints = tp.get_type_hints(func, include_extras=True)

    sig = Signature.from_callable(func)
    sig_bound = sig.bind(*args, **kwargs)
    sig_bound.apply_defaults()
    sig_str = to_signature(sig_bound, hints)
    parent = (f'args of {sig_str}',)

    for k, v in sig_bound.arguments.items():
        if h_p := hints.get(k, None):
            if cr := _check(v, h_p, parent, fail_fast=fail_fast):
                if error_action is ErrorAction.RAISE:
                    raise ClinicError(cr)
                elif error_action is ErrorAction.WARN:
                    warnings.warn(cr.to_str(), category)
                elif error_action is ErrorAction.RETURN:
                    return cr

    post = func(*args, **kwargs)

    if h_return := hints.get('return', None):
        if cr := _check(post,
                h_return,
                (f'return of {sig_str}',),
                fail_fast=fail_fast,
                ):
            if error_action is ErrorAction.RAISE:
                raise ClinicError(cr)
            elif error_action is ErrorAction.WARN:
                warnings.warn(cr.to_str(), category)
            elif error_action is ErrorAction.RETURN:
                return cr

    return post


TVFunc = tp.TypeVar('TVFunc', bound=tp.Callable[..., tp.Any])

class InterfaceClinic:
    '''A family of decorators for run-time type checking and data validation.
    '''

    @tp.overload
    @staticmethod
    def check(func: TVFunc) -> TVFunc: ...

    @tp.overload
    @staticmethod
    def check(*, fail_fast: bool) -> tp.Callable[[TVFunc], TVFunc]: ...

    @tp.overload
    @staticmethod
    def check(func: None, *, fail_fast: bool) -> tp.Callable[[TVFunc], TVFunc]: ...

    @staticmethod
    def check(
            func: TVFunc | None = None,
            *,
            fail_fast: bool = False,
            ) -> tp.Any:
        '''A function decorator to perform run-time checking of function arguments and return values based on the function type annotations, including type hints and ``Constraint`` subclasses. Raises ``ClinicError`` on failure.
        '''

        def decorator(func: TVFunc) -> TVFunc:
            @wraps(func)
            def wrapper(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
                return _check_interface(func,
                        args,
                        kwargs,
                        fail_fast,
                        ErrorAction.RAISE,
                        )
            return tp.cast(TVFunc, wrapper)

        if func is not None:
            return decorator(func)
        return decorator


    @tp.overload
    @staticmethod
    def warn(func: TVFunc) -> TVFunc: ...

    @tp.overload
    @staticmethod
    def warn(*, fail_fast: bool, category: tp.Type[Warning]) -> tp.Callable[[TVFunc], TVFunc]: ...

    @tp.overload
    @staticmethod
    def warn(func: None, *, fail_fast: bool, category: tp.Type[Warning]) -> tp.Callable[[TVFunc], TVFunc]: ...

    @staticmethod
    def warn(
            func: TVFunc | None = None,
            *,
            fail_fast: bool = False,
            category: tp.Type[Warning] = UserWarning,
            ) -> tp.Any:
        '''A function decorator to perform run-time checking of function arguments and return values based on the function type annotations, including type hints and ``Constraint`` subclasses. Issues a warning on failure.
        '''

        def decorator(func: TVFunc) -> TVFunc:
            @wraps(func)
            def wrapper(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
                return _check_interface(func,
                        args,
                        kwargs,
                        fail_fast,
                        ErrorAction.WARN,
                        category,
                        )
            return tp.cast(TVFunc, wrapper)

        if func is not None:
            return decorator(func)
        return decorator

