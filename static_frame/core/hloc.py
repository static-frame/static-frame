import typing as tp

from static_frame.core.util import EMPTY_TUPLE
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import GetItemKeyType


class HLocMeta(type):

    def __getitem__(cls, key: GetItemKeyType) -> 'HLoc':
        if not isinstance(key, tuple) or key is EMPTY_TUPLE:
            key = (key,)
        return cls(key) #type: ignore [no-any-return]

class HLoc(metaclass=HLocMeta):
    '''
    A simple wrapper for embedding hierarchical specificiations for :obj:`static_frame.IndexHierarchy` within a single axis argument of a ``loc`` selection.

    Implemented as a container of hierarchical keys that defines NULL slices for all lower dimensions that are not defined at construction.
    '''

    STATIC = True
    __slots__ = (
            'key',
            )

    def __init__(self, key: tp.Tuple[GetItemKeyType]) -> None:
        self.key = key

    def __iter__(self) -> tp.Iterator[GetItemKeyType]:
        return self.key.__iter__()

    def __len__(self) -> int:
        return self.key.__len__()

    def __repr__(self) -> str:
        def gen_nested_keys() -> tp.Iterator[str]:
            for key in self.key:
                if key.__class__ is not slice:
                    yield str(key)
                    continue

                if key == NULL_SLICE:
                    yield ':'
                    continue

                result = ':' if key.start is None else f'{key.start}' # type: ignore [union-attr]

                if key.stop is not None: # type: ignore [union-attr]
                    result += str(key.stop) # type: ignore [union-attr]

                if key.step is not None and key.step != 1: # type: ignore [union-attr]
                    result += f':{key.step}' # type: ignore [union-attr]

                yield result

        return f'<HLoc[{",".join(gen_nested_keys())}]>'
