import typing as tp


from static_frame.core.util import GetItemKeyType
from static_frame.core.util import KEY_MULTIPLE_TYPES
from static_frame.core.util import NULL_SLICE


class HLocMeta(type):

    def __getitem__(cls,
            key: GetItemKeyType
            ) -> tp.Iterable[GetItemKeyType]:
        if not isinstance(key, tuple):
            key = (key,)
        # NOTE: tp.case is a performance hit and should be removed
        return tp.cast(tp.Iterable[GetItemKeyType], cls(key))

class HLoc(metaclass=HLocMeta):
    '''A wrapper for embedding hierarchical specificiations for :obj:`static_frame.IndexHierarchy` within a single axis argument of a ``loc`` selection.

    Implemented as a container of hiearchical keys that defiines NULL slices for all lower dimensions that are not defined at construction.
    '''

    __slots__ = (
            'key',
            )

    def __init__(self, key: tp.Sequence[GetItemKeyType]):
        self.key = key

    def __iter__(self) -> tp.Iterator[GetItemKeyType]:
        return self.key.__iter__()

    def __len__(self) -> int:
        return self.key.__len__()

    def __getitem__(self, key: int) -> GetItemKeyType:
        '''
        Each key reprsents a hierarchical level; if a key is not specified, the default should be to return the null slice.
        '''
        if key >= len(self.key):
            return NULL_SLICE
        return self.key.__getitem__(key)

    def has_key_multiple(self) -> bool:
        return any(isinstance(k, KEY_MULTIPLE_TYPES) for k in self.key)
