import typing as tp

from static_frame.core.util import GetItemKeyType


class HLocMeta(type):

    def __getitem__(cls, key: GetItemKeyType) -> 'HLoc':
        if not isinstance(key, tuple):
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

    def __init__(self, key: tp.Sequence[GetItemKeyType]):
        self.key = key

    def __iter__(self) -> tp.Iterator[GetItemKeyType]:
        return self.key.__iter__()

    def __len__(self) -> int:
        return self.key.__len__()
