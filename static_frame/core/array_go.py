
import numpy as np

from static_frame.core.util import immutable_filter


class ArrayGO:
    '''
    A grow only, one-dimensional, object type array, specifically for usage in IndexHierarchy IndexLevel objects.
    '''

    __slots__ = (
            '_dtype',
            '_array',
            '_array_mutable',
            '_recache',
            )

    # NOTE: this can be implemented with one array, where we overallocate for growth, then grow as needed, or with an array and list. Since most instaces will not need to grow (only edge nodes), overall efficiency might be greater with a list

    def __init__(self,
            iterable,
            *,
            dtype=object,
            own_iterable=False):
        '''
        Args:
            own_iterable: flag iterable as ownable by this instance.
        '''

        self._dtype = dtype

        if isinstance(iterable, np.ndarray):
            if own_iterable:
                self._array = iterable
                self._array.flags.writeable = False
            else:
                self._array = immutable_filter(iterable)
            assert self._array.dtype == self._dtype
            self._recache = False
            self._array_mutable = None
        else:
            self._array = None
            self._recache = True
            # always call list to get new object, or realize a generator
            if own_iterable:
                self._array_mutable = iterable
            else:
                self._array_mutable = list(iterable)

    def _update_array_cache(self):
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

    def __iter__(self):
        if self._recache:
            self._update_array_cache()
        return self._array.__iter__()

    def __getitem__(self, key):
        if self._recache:
            self._update_array_cache()
        return self._array.__getitem__(key)

    def __len__(self):
        if self._recache:
            self._update_array_cache()
        return self._array.__len__()

    def append(self, value):
        if self._array_mutable is None:
            self._array_mutable = []
        self._array_mutable.append(value)
        self._recache = True

    def extend(self, values):
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
        return self.__class__(self._array, dtype=self._dtype)