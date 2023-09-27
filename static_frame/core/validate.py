import types
import typing
from collections import deque
from collections.abc import MutableMapping
from collections.abc import Sequence
from functools import wraps
from inspect import Signature
from itertools import chain
from itertools import repeat

import numpy as np
import typing_extensions as tp

from static_frame.core.frame import Frame
from static_frame.core.index import Index
from static_frame.core.index_base import IndexBase
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.series import Series
from static_frame.core.util import TLabel

# _UnionGenericAlias comes from tp.Union, UnionType from | expressions
# tp.Optional returns a _UnionGenericAlias with later Python, but a _GenericAlias with 3.8

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


def is_union(hint: tp.Any) -> bool:
    if UNION_TYPES:
        return isinstance(hint, UNION_TYPES)
    elif isinstance(hint, GENERIC_TYPES):
        return tp.get_origin(hint) is tp.Union
    return False

def is_unpack(hint: tp.Any) -> bool:
    # NOTE: cannot use isinstance or issubclass with Unpack
    if hint in UNPACK_TYPES:
        return True
    return False

TParent = tp.Tuple[tp.Any, ...]
# A validation record can be used to queue checks or report errors
TValidation = tp.Tuple[tp.Any, tp.Any, TParent]

#-------------------------------------------------------------------------------
# error reporting, presentation

ERROR_MESSAGE_TYPE = object()

def to_name(v: tp.Any) -> str:
    if isinstance(v, GENERIC_TYPES):
        # for older Python, not all generics have __name__
        if not (name := getattr(v, '__name__', '')):
            name = str(v)
        return f'{name}[{", ".join(to_name(q) for q in tp.get_args(v))}]'
    if hasattr(v, '__name__'):
        return v.__name__ # type: ignore[no-any-return]
    if v is ...:
        return '...'
    return str(v)

class CheckError(TypeError):

    def __init__(self, log: tp.Iterable[TValidation]) -> None:
        msg = []
        for v, h, p in log:
            if p:
                path = ', '.join(to_name(n) for n in p)
            else:
                path = ''

            if v is ERROR_MESSAGE_TYPE: # in this case, do not use the value
                if path:
                    prefix = f'Failed check in {path}:'
                else:
                    prefix = 'Failed check:'
                msg.append(f'{prefix} {h}.')
            else:
                if path:
                    prefix = f'In {path}: expected'
                else:
                    prefix = 'Expected'
                msg.append(f'{prefix} {to_name(h)}, provided {to_name(type(v))} invalid.')

        TypeError.__init__(self, ' '.join(msg))

#-------------------------------------------------------------------------------

class Constraint:
    __slots__: tp.Tuple[str, ...] = ()

    def iter_error_log(self,
            value: tp.Any,
            hint: tp.Any,
            parent: TParent,
            ) -> tp.Iterator[TValidation]:
        raise NotImplementedError()

    def __repr__(self) -> str:
        args = ', '.join(repr(getattr(self, v)) for v in self.__slots__)
        return f'{self.__class__.__name__}({args})'

class Name(Constraint):
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
            yield ERROR_MESSAGE_TYPE, f'expected name {self._name!r}, provided name {n!r}', parent


class Len(Constraint):
    __slots__ = ('_len',)

    def __init__(self, len: int):
        self._len = len

    def iter_error_log(self,
            value: tp.Any,
            hint: tp.Any,
            parent: TParent,
            ) -> tp.Iterator[TValidation]:
        if (vl := len(value)) != self._len:
            yield ERROR_MESSAGE_TYPE, f'expected length {self._len}, provided length {vl}', parent


# might accept regular expression objects as label entries?
class Labels(Constraint):
    __slots__ = ('_labels',)

    def __init__(self, *labels: tp.Sequence[TLabel]):
        self._labels: tp.Sequence[TLabel] = labels

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
            yield ERROR_MESSAGE_TYPE, f'expected {self} to be used on Index or IndexHierarchy, not provided {to_name(type(value))}', parent
        else:
            pos_e = 0 # position expected
            len_e = len(self._labels)

            for label_p in value:
                if pos_e >= len_e:
                    yield ERROR_MESSAGE_TYPE, f'expected labels exhausted at provided {label_p!r}', parent
                    break
                label_e = self._labels[pos_e]

                if label_e is not ...:
                    if label_p != label_e:
                        yield ERROR_MESSAGE_TYPE, f'expected {label_e!r}, provided {label_p!r}', parent
                    pos_e += 1
                # label_e is an Ellipses; either find next as match or continue with Ellipses
                elif pos_e + 1 < len_e: # more expected labels available
                    label_next_e = self._labels[pos_e + 1]
                    if label_next_e is ...:
                        yield ERROR_MESSAGE_TYPE, 'expected cannot be defined with adjacent ellipses', parent
                        break
                    if label_p == label_next_e:
                        pos_e += 2 # skip the compared value, prepare to get next
                # else, last expected value is Ellipses
            else: # no break, evaluate final conditions
                if pos_e == len_e - 1 and label_e is ...:
                    pass # ended on elipses
                elif pos_e < len_e:
                    remainder = self._prepare_remainder(self._labels[pos_e:])
                    yield ERROR_MESSAGE_TYPE, f'expected has unmatched labels {remainder}', parent


class Validator(Constraint):
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

def iter_tuple_checks(value: tp.Any,
        hint: tp.Any,
        parent: TParent,
        ) -> tp.Iterable[TValidation]:
    h_components = tp.get_args(hint)
    if h_components[-1] is ...:
        if (h_len := len(h_components)) != 2:
            yield ERROR_MESSAGE_TYPE, 'invalid ellipses usage', parent
        else:
            h = h_components[0]
            for v in value:
                yield v, h, parent
    else:
        if (h_len := len(h_components)) != len(value):
            msg = f'expected tuple length of {h_len}, provided tuple length of {len(value)}'
            yield ERROR_MESSAGE_TYPE, msg, parent
        for v, h in zip(value, h_components):
            yield v, h, parent

def iter_mapping_checks(value: tp.Any,
        hint: tp.Any,
        parent: TParent,
        ) -> tp.Iterable[TValidation]:
    [h_keys, h_values] = tp.get_args(hint)
    for v, h in chain(
            zip(value.keys(), repeat(h_keys)),
            zip(value.values(), repeat(h_values)),
            ):
        yield v, h, parent


# NOTE: we create an instance of dtype.type() so as to not modify h_generic, as it might be Union or other generic that cannot be wrapped in a tp.Type

def iter_series_checks(value: tp.Any,
        hint: tp.Any,
        parent: TParent,
        ) -> tp.Iterable[TValidation]:
    h_index, h_generic = tp.get_args(hint) # there must be two
    yield value.index, h_index, parent
    yield value.dtype.type(), h_generic, parent

def iter_index_checks(value: tp.Any,
        hint: tp.Any,
        parent: TParent,
        ) -> tp.Iterable[TValidation]:
    [h_generic] = tp.get_args(hint)
    yield value.dtype.type(), h_generic, parent

def iter_index_hierarchy_checks(value: tp.Any,
        hint: tp.Any,
        parent: TParent,
        ) -> tp.Iterable[TValidation]:
    h_generics = tp.get_args(hint)
    h_len = len(h_generics)

    if h_len == 1 and is_unpack(tp.get_origin(h_generics[0])):
        # if using tp.Unpack, or the *tp.Tuple[] notation, the origin of this generic will be tp.Unpack
        [h_tuple] = tp.get_args(h_generics[0])
        assert issubclass(tuple, tp.get_origin(h_tuple))
        # to support usage of Ellipses, treat this as a hint of a corresponding tuple of Index
        yield tuple(value.index_at_depth(i) for i in range(value.depth)), h_tuple, parent
    else:
        if h_len != value.depth:
            # give expected first
            yield ERROR_MESSAGE_TYPE, f'expected IndexHierarchy depth of {h_len}, provided depth of {value.depth}', parent
        for i in range(value.depth):
            yield value.index_at_depth(i), h_generics[i], parent

def iter_frame_checks(value: tp.Any,
        hint: tp.Any,
        parent: TParent,
        ) -> tp.Iterable[TValidation]:

    # NOTE: not sure how this works with defaults in TypeVar
    h_index, h_columns, *h_types = tp.get_args(hint)
    h_types_len = len(h_types)

    yield value.index, h_index, parent
    yield value.columns, h_columns, parent

    if h_types_len == 1 and is_unpack(tp.get_origin(h_types[0])):
        # if using tp.Unpack, or the *tp.Tuple[] notation, the origin of this generic will be tp.Unpack
        [h_tuple] = tp.get_args(h_types[0])
        assert issubclass(tuple, tp.get_origin(h_tuple))
        # to support usage of Ellipses, treat this as a hint of a corresponding tuple of types
        yield tuple(d.type() for d in value._blocks._iter_dtypes()), h_tuple, parent
    else:
        if h_types_len != value.shape[1]:
            # give expected first
            yield ERROR_MESSAGE_TYPE, f'expected Frame has {h_types_len} dtype, provided Frame has {value.shape[1]} dtype', parent
        for dt, h in zip(value._blocks._iter_dtypes(), h_types):
            yield dt.type(), h, parent


def iter_ndarray_checks(value: tp.Any,
        hint: tp.Any,
        parent: TParent,
        ) -> tp.Iterable[TValidation]:
    h_shape, h_dtype = tp.get_args(hint)
    yield value.dtype, h_dtype, parent

def iter_dtype_checks(value: tp.Any,
        hint: tp.Any,
        parent: TParent,
        ) -> tp.Iterable[TValidation]:
    [h_generic] = tp.get_args(hint)
    yield value.type(), h_generic, parent

#-------------------------------------------------------------------------------

def check(
        value: tp.Any,
        hint: tp.Any,
        parent: TParent = (),
        fail_fast: bool = False,
        ) -> tp.Iterable[TValidation]:

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
            return e_log

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
                c_log = check(v, c_hint, p_next, fail_fast)
                if not c_log: # no error found, can exit
                    break
                u_log.extend(c_log)
            else: # no breaks, so no matches within union
                e_log.extend(u_log)
        elif isinstance(h, Constraint):
            e_log.extend(h.iter_error_log(v, h, p_next))
        elif isinstance(h, GENERIC_TYPES):
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
            elif origin == tp.Annotated: # NOTE: cannot use is due backwards compat
                h_type, *h_annotations = tp.get_args(h)
                # perform the un-annoitated check
                q.append((v, h_type, p_next))
                for h_annotation in h_annotations:
                    if isinstance(h_annotation, Constraint):
                        q.append((v, h_annotation, p_next))
            else:
                if origin == tp.Literal: # NOTE: cannot use is due backwards compat
                    l_log: tp.List[TValidation] = []
                    for l_hint in tp.get_args(h): # get components
                        c_log = check(v, l_hint, p_next, fail_fast)
                        if not c_log: # no error found, can exit
                            break
                        l_log.extend(c_log)
                    else: # no breaks, so no matches within union
                        e_log.extend(l_log)
                    continue

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
                    raise NotImplementedError(f'no handling for generic {origin}')
        elif not isinstance(h, type):
            # h is value from a literal
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

    return e_log

#-------------------------------------------------------------------------------
# public interfaces

def check_type(
        value: tp.Any,
        hint: tp.Any,
        parent: TParent = (),
        *,
        fail_fast: bool = False,
        ) -> None:
    e_log = check(value, hint, parent, fail_fast)
    if e_log:
        raise CheckError(e_log)


TVFunc = tp.TypeVar('TVFunc', bound=tp.Callable[..., tp.Any])

@tp.overload
def check_interface(func: TVFunc) -> TVFunc: ...

@tp.overload
def check_interface(func: None, *, fail_fast: bool) -> tp.Callable[[TVFunc], TVFunc]: ...

def check_interface(
        func: TVFunc | None = None,
        *,
        fail_fast: bool = False,
        ) -> tp.Any:

    def decorator(func: TVFunc) -> TVFunc:

        @wraps(func)
        def wrapper(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            sig = Signature.from_callable(func)
            sig_bound = sig.bind(*args, **kwargs)
            sig_bound.apply_defaults()
            parent = (f'args of {sig}',)
            for k, v in sig_bound.arguments.items():
                if (h_p := sig.parameters[k].annotation) != Signature.empty:
                    check_type(v, h_p, parent, fail_fast=fail_fast)

            post = func(*args, **kwargs)

            if (h_return := sig.return_annotation) != Signature.empty:
                check_type(post, h_return, (f'return of {sig}',), fail_fast=fail_fast)

            return post

        return tp.cast(TVFunc, wrapper)

    if func is not None:
        return decorator(func) # type: ignore

    return decorator







