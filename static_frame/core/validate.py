import types
import typing
from collections import deque

import typing_extensions as tp

from static_frame.core.container import ContainerBase


class ValidationError(TypeError):
    def __init__(self, pairs: tp.Sequence[tp.Tuple[tp.Any, tp.Any]]) -> None:
        pass


def get_series_pairs():
    # yield pairs to compare
    pass


def validate_pair(value: tp.Any, hint: tp.Any) -> None:

    q = deque(((value, hint),))
    log = []

    while q:
        v, h = q.popleft()
        # import ipdb; ipdb.set_trace()
        if h is tp.Any:
            continue

        if isinstance(h, ContainerBase):
            if isinstance(value, h):
                continue
            else:
                log.append((v, h))
                continue

        elif isinstance(h, typing._GenericAlias):
            # have a generic container
            origin = tp.get_origin(h)
            if isinstance(origin, ContainerBase):
                args = tp.get_args(h)
                # next: enque v, origin to get a type check
                # enque the component checks

        elif isinstance(h, (typing._UnionGenericAlias, types.UnionType)):
            # _UnionGenericAlias comes from tp.Union, UnionType from | expressions
            # tp.Optional returns a _UnionGenericAlias
            pass
        else:
            pass

    if log:
        raise TypeError(log)


TVFunc = tp.TypeVar('TVFunc', bound=tp.Callable[..., tp.Any])

def validate(func: TVFunc) -> TVFunc:
    pass