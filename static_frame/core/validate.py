import types
import typing
from collections import deque

import typing_extensions as tp

from static_frame.core.container import ContainerBase

# _UnionGenericAlias comes from tp.Union, UnionType from | expressions
# tp.Optional returns a _UnionGenericAlias
if hasattr(types, 'UnionType') and hasattr(types, 'GenericAlias'):
    UNION_TYPES = (typing._UnionGenericAlias, types.UnionType) # type: ignore
    GENERIC_TYPES = (typing._GenericAlias, types.GenericAlias) # type: ignore
else:
    UNION_TYPES = typing._UnionGenericAlias # type: ignore
    GENERIC_TYPES = typing._GenericAlias # type: ignore


class ValidationError(TypeError):
    def __init__(self, pairs: tp.Sequence[tp.Tuple[tp.Any, tp.Any]]) -> None:
        pass


def get_series_pairs() -> tp.Iterator[tp.Tuple[tp.Any, tp.Any]]:
    # yield pairs to compare
    pass


def validate_pair(value: tp.Any, hint: tp.Any) -> None:

    q = deque(((value, hint),))
    log = []

    while q:
        v, h = q.popleft()
        print(v, h)
        # import ipdb; ipdb.set_trace()
        if h is tp.Any:
            continue

        if isinstance(h, UNION_TYPES):
            # NOTE: must check union types first as tp.Union matches as generic type
            # import ipdb; ipdb.set_trace()
            u_log = []
            for c_hint in tp.get_args(h): # get components
                c_log = validate_pair(v, c_hint)
                if not c_log: # no error found, can exit
                    break
                else: # find all errors
                    u_log.extend(c_log)
            else: # not one break, so no matches within union
                log.extend(u_log)
                continue

            # any of these need to match

        elif isinstance(h, GENERIC_TYPES):
            # have a generic container
            origin = tp.get_origin(h)
            if origin is type: # a tp.Type[x] generic
                [t] = tp.get_args(h) # this is the type
                try:
                    check = issubclass(t, v)
                except TypeError:
                    check = False
                if check:
                    continue
                else:
                    log.append((v, h))
            elif isinstance(v, ContainerBase):
                args = tp.get_args(h)
                # next: enque v, origin to get a type check
                # enque the component checks
            else:
                raise NotImplementedError(f'no handling for generic {origin}')

        elif issubclass(ContainerBase, h):
            # handle SF containers
            if isinstance(value, h):
                continue
            else:
                log.append((v, h))
                continue

        elif isinstance(h, type):
            # special cases
            if v.__class__ is bool:
                if h is bool:
                    continue
                else:
                    log.append((v, h))
                    continue
            # general case
            if isinstance(v, h):
                continue
            else:
                log.append((v, h))
                continue
        else:
            pass

    return log


def validate_pair_raises(value: tp.Any, hint: tp.Any) -> None:
    log = validate_pair(value, hint)
    if log:
        raise TypeError(log)

TVFunc = tp.TypeVar('TVFunc', bound=tp.Callable[..., tp.Any])

def validate(func: TVFunc) -> TVFunc:
    pass