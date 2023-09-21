import types
import typing
from collections import deque

import numpy as np
import typing_extensions as tp

# from static_frame.core.container import ContainerBase
# from static_frame.core.frame import Frame
from static_frame.core.index import Index
# from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.series import Series

# _UnionGenericAlias comes from tp.Union, UnionType from | expressions
# tp.Optional returns a _UnionGenericAlias with later Python, but a _GenericAlias with 3.8

def _iter_generic_classes() -> tp.Iterable[tp.Type[tp.Any]]:
    if hasattr(types, 'GenericAlias'):
        yield types.GenericAlias
    if hasattr(typing, '_GenericAlias'):
        yield typing._GenericAlias # pyright: ignore

GENERIC_TYPES = tuple(_iter_generic_classes())

def iter_union_classes() -> tp.Iterable[tp.Type[tp.Any]]:
    if hasattr(types, 'UnionType'):
        yield types.UnionType
    if hasattr(typing, '_UnionGenericAlias'):
        yield typing._UnionGenericAlias # pyright: ignore

UNION_TYPES = tuple(iter_union_classes())

def is_union(hint: tp.Any) -> bool:
    if UNION_TYPES:
        return isinstance(hint, UNION_TYPES)
    elif isinstance(hint, GENERIC_TYPES):
        return tp.get_origin(hint) is tp.Union
    return False


TParent = tp.Tuple[tp.Any, ...]
TValidation = tp.Tuple[tp.Any, tp.Any, TParent]

class CheckError(TypeError):
    @staticmethod
    def get_name(v: tp.Any) -> str:
        if isinstance(v, GENERIC_TYPES):
            return str(v)
        if hasattr(v, '__name__'):
            return v.__name__ # type: ignore[no-any-return]
        return str(v)

    def __init__(self, log: tp.Iterable[TValidation]) -> None:
        msg = []
        for v, h, p in log:
            if p:
                path = ', '.join(self.get_name(n) for n in p)
                prefix = f'In {path}, provided'
            else:
                prefix = 'Provided'
            msg.append(f'{prefix} {self.get_name(type(v))} invalid for {self.get_name(h)}.')
        TypeError.__init__(self, ' '.join(msg))

#-------------------------------------------------------------------------------
# handlers for getting components out of generics
# NOTE: we create an instance of dtype.type() so as to not modify h_generic, as it might be Union or other generic that cannot be wrapped in a tp.Type

def get_series_pairs(value: tp.Any, hint: tp.Any, parent: TParent) -> tp.Iterable[TValidation]:
    h_index, h_generic = tp.get_args(hint) # there must be two
    yield value.index, h_index, parent
    yield value.dtype.type(), h_generic, parent

def get_index_pairs(value: tp.Any, hint: tp.Any, parent: TParent) -> tp.Iterable[TValidation]:
    [h_generic] = tp.get_args(hint)
    yield value.dtype.type(), h_generic, parent

def get_ndarray_pairs(value: tp.Any, hint: tp.Any, parent: TParent) -> tp.Iterable[TValidation]:
    h_shape, h_dtype = tp.get_args(hint)
    yield value.dtype, h_dtype, parent

def get_dtype_pairs(value: tp.Any, hint: tp.Any, parent: TParent) -> tp.Iterable[TValidation]:
    [h_generic] = tp.get_args(hint)
    yield value.type(), h_generic, parent

#-------------------------------------------------------------------------------

def check(
        value: tp.Any,
        hint: tp.Any,
        fail_fast: bool = False,
        parent: TParent = (),
        ) -> tp.Iterable[TValidation]:

    q = deque(((value, hint, parent),))
    log: tp.List[TValidation] = []

    while q:
        if fail_fast and log:
            return log

        v, h, p = q.popleft()
        p_next = p + (h,)

        if h is tp.Any:
            continue

        if is_union(h):
            # NOTE: must check union types first as tp.Union matches as generic type
            u_log: tp.List[TValidation] = []
            for c_hint in tp.get_args(h): # get components
                # handing one pair at a time with a secondary call will allow nested types in the union to be evaluated on their own
                c_log = check(v, c_hint, fail_fast, p_next)
                if not c_log: # no error found, can exit
                    break
                u_log.extend(c_log)
            else: # no breaks, so no matches within union
                log.extend(u_log)

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
                log.append((v, h, p))
            else:
                if not isinstance(v, origin):
                    log.append((v, origin, p))
                    continue

                if isinstance(v, Index):
                    q.extend(get_index_pairs(v, h, p_next))
                elif isinstance(v, Series):
                    q.extend(get_series_pairs(v, h, p_next))
                elif isinstance(v, np.ndarray):
                    q.extend(get_ndarray_pairs(v, h, p_next))
                elif isinstance(v, np.dtype):
                    q.extend(get_dtype_pairs(v, h, p_next))
                else:
                    raise NotImplementedError(f'no handling for generic {origin}')

        else:
            assert isinstance(h, type)
            # special cases
            if v.__class__ is bool:
                if h is bool:
                    continue
            # general case
            elif isinstance(v, h):
                continue
            log.append((v, h, p))


    return log


def check_type(value: tp.Any, hint: tp.Any, fail_fast: bool = False) -> None:
    log = check(value, hint, fail_fast)
    if log:
        raise CheckError(log)


TVFunc = tp.TypeVar('TVFunc', bound=tp.Callable[..., tp.Any])

def check_interface(func: TVFunc) -> TVFunc:
    return func







