import types
import typing
from collections import deque

import typing_extensions as tp

from static_frame.core.container import ContainerBase
from static_frame.core.frame import Frame
from static_frame.core.index import Index
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.series import Series

# _UnionGenericAlias comes from tp.Union, UnionType from | expressions
# tp.Optional returns a _UnionGenericAlias
if hasattr(types, 'UnionType') and hasattr(types, 'GenericAlias'):
    UNION_TYPES = (typing._UnionGenericAlias, types.UnionType) # type: ignore
    GENERIC_TYPES = (typing._GenericAlias, types.GenericAlias) # type: ignore
else:
    UNION_TYPES = typing._UnionGenericAlias # type: ignore
    GENERIC_TYPES = typing._GenericAlias # type: ignore

TPair = tp.Tuple[tp.Any, tp.Any]

class ValidationError(TypeError):
    def __init__(self, pairs: tp.Sequence[TPair]) -> None:
        pass


#-------------------------------------------------------------------------------
# handlers for getting components out of generics
# NOTE: we create an instance of dtype.type() so as to not modify h_generic, as it might be Union or other generic that cannot be wrapped in a tp.Type

def get_series_pairs(value: tp.Any, hint: tp.Any) -> tp.Iterable[TPair]:
    h_index, h_generic = tp.get_args(hint) # there must be two
    yield value.index, h_index
    yield value.dtype.type(), h_generic
    # yield value.dtype.type, tp.Type[h_generic]

def get_index_pairs(value: tp.Any, hint: tp.Any) -> tp.Iterable[TPair]:
    [h_generic] = tp.get_args(hint) # there must be two
    yield value.dtype.type(), h_generic


#-------------------------------------------------------------------------------

def validate_pair(
        value: tp.Any,
        hint: tp.Any,
        ) -> tp.Iterable[TPair]:

    q = deque(((value, hint),))
    log: tp.List[TPair] = []

    while q:
        v, h = q.popleft()
        # print(v, h)
        # import ipdb; ipdb.set_trace()

        if h is tp.Any:
            continue

        if isinstance(h, UNION_TYPES):
            # NOTE: must check union types first as tp.Union matches as generic type
            u_log: tp.List[TPair] = []
            for c_hint in tp.get_args(h): # get components
                # handing one pair at a time with a secondary call will allow nested types in the union to be evaluated on their own
                c_log = validate_pair(v, c_hint)
                if not c_log: # no error found, can exit
                    break
                else: # find all errors
                    u_log.extend(c_log)
            else: # not one break, so no matches within union
                log.extend(u_log)

        elif isinstance(h, GENERIC_TYPES): # type: ignore[unreachable]
            # have a generic container
            origin = tp.get_origin(h)
            if origin is type: # a tp.Type[x] generic
                [t] = tp.get_args(h) # this is the type
                try: # the value should be a subclass of t
                    check = issubclass(t, v)
                except TypeError:
                    check = False
                if check:
                    continue
                else:
                    log.append((v, h))

            elif isinstance(v, ContainerBase):
                # type check that instance v is of type origin
                if not isinstance(v, origin):
                    log.append((v, origin))
                    continue
                # collect pairs to check for component parts (which might be generic)
                if issubclass(Index, origin):
                    q.extend(get_index_pairs(v, h))
                elif issubclass(Series, origin):
                    q.extend(get_series_pairs(v, h))
                else:
                    raise NotImplementedError(f'no handling for generic {origin}')
            else:
                raise NotImplementedError(f'no handling for generic {origin}')

        elif issubclass(ContainerBase, h):
            # handle non-generic SF containers
            if isinstance(value, h):
                continue
            else:
                log.append((v, h))

        elif isinstance(h, type):
            # special cases
            if v.__class__ is bool:
                if h is bool:
                    continue
                else:
                    log.append((v, h))
            # general case
            elif isinstance(v, h):
                continue
            else:
                log.append((v, h))
        else:
            pass

    return log


def validate_pair_raises(value: tp.Any, hint: tp.Any) -> None:
    log = validate_pair(value, hint)
    if log:
        raise TypeError(log)


TVFunc = tp.TypeVar('TVFunc', bound=tp.Callable[..., tp.Any])

def validate(func: TVFunc) -> TVFunc:
    return func