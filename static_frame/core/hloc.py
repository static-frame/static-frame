import typing as tp

from static_frame.core.util import GetItemKeyType
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import EMPTY_TUPLE


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
        contains_slices = False

        # This is usually very small (i.e. <10 items), so it's very cheap to
        # pay the price of an extra loop to check for slices.
        for key in self.key:
            if key.__class__ is slice:
                contains_slices = True
                break

        def gen_nested_keys() -> tp.Iterator[str]:
            for key in self.key:
                if key.__class__ is slice:
                    yield str(key)
                    continue

                if key == NULL_SLICE:
                    yield ':'
                    continue

                if key.start is None:  # type: ignore [union-attr]
                    result = ':'
                else:
                    result = f'{key.start}:' # type: ignore [union-attr]

                if key.stop is not None: # type: ignore [union-attr]
                    result += str(key.stop) # type: ignore [union-attr]

                if key.step is not None and key.step != 1: # type: ignore [union-attr]
                    result += f':{key.step}' # type: ignore [union-attr]

                yield result

        if not contains_slices:
            if len(self.key) == 1:
                return f'HLoc[{self.key[0]}]'

            # self.key is a tuple, so we strip off the parentheses.
            return f'HLoc[{str(self.key)[1:-1]}]'

        return f'<HLoc[{",".join(gen_nested_keys())}]>'
