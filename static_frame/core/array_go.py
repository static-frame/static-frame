
import typing as tp

import numpy as np

from static_frame.core.util import immutable_filter


class ArrayGO:
    '''
    A grow only, one-dimensional, object type array, specifically for usage in IndexHierarchy IndexLevel objects.
    '''

    _array: tp.Optional[np.ndarray]
    _array_mutable: tp.Optional[tp.List[tp.Any]]

    __slots__ = (
            '_dtype',
            '_array',
            '_array_mutable',
            '_recache',
            )

    # NOTE: this can be implemented with one array, where we overallocate for growth, then grow as needed, or with an array and list. Since most instaces will not need to grow (only edge nodes), overall efficiency might be greater with a list

    def __init__(self,
            iterable: tp.Union[np.ndarray, tp.List[object]],
            *,
            own_iterable: bool = False) -> None:
        '''
        Args:
            own_iterable: flag iterable as ownable by this instance.
        '''

        self._dtype = object # only object arrays are supported

        if isinstance(iterable, np.ndarray):
            if own_iterable:
                self._array = iterable
                self._array.flags.writeable = False
            else:
                self._array = immutable_filter(iterable)
            if self._array.dtype != self._dtype:
                raise NotImplementedError('only object arrays are supported')
            self._recache = False
            self._array_mutable = None
        else: # assume it is a list or listable
            self._array = None
            self._recache = True
            # always call list to get new object, or realize a generator
            if own_iterable:
                self._array_mutable = iterable
            else:
                self._array_mutable = list(iterable)

    def _update_array_cache(self) -> None:
        if self._array_mutable is not None:
            if self._array is not None:
                len_base = len(self._array)
                array = np.empty(
                        len_base + len(self._array_mutable),
                        self._dtype)
                array[:len_base] = self._array
                array[len_base:] = self._array_mutable
                array.flags.writeable = False
                self._array = array
                self._array_mutable = None
            else:
                self._array = np.array(self._array_mutable, self._dtype)
                self._array.flags.writeable = False
                self._array_mutable = None
        self._recache = False

    def __iter__(self) -> tp.Iterator[tp.Any]:
        if self._recache:
            self._update_array_cache()
        return iter(self._array) #type: ignore

    def __getitem__(self, key: tp.Any) -> tp.Any:
        if self._recache:
            self._update_array_cache()
        return self._array.__getitem__(key) #type: ignore

    def __len__(self) -> int:
        if self._recache:
            self._update_array_cache()
        return len(self._array) #type: ignore

    def append(self, value: tp.Iterable[object]) -> None:
        if self._array_mutable is None:
            self._array_mutable = []
        self._array_mutable.append(value)
        self._recache = True

    def extend(self, values: tp.Iterable[object]) -> None:
        if self._array_mutable is None:
            self._array_mutable = []
        self._array_mutable.extend(values)
        self._recache = True

    @property
    def values(self) -> np.ndarray:
        '''Return the immutable labels array
        '''
        if self._recache:
            self._update_array_cache()
        return self._array


    def copy(self) -> 'ArrayGO':
        '''Return a new ArrayGO with an immutable array from this ArrayGO
        '''
        if self._recache:
            self._update_array_cache()
        return self.__class__(self._array, own_iterable=True)
