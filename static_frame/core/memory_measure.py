import typing as tp
from collections import abc
from itertools import chain
from sys import getsizeof

import numpy as np

from static_frame.core.util import DTYPE_OBJECT_KIND

class MaterializedArray:
    '''Wrapper of array that delivers the sizeof as the fully realized size, ignoring any potential sharing of memory.
    '''

    __slots__ = ('_array',)
    BASE_ARRAY_BYTES = getsizeof(np.array(()))

    def __init__(self, array: np.ndarray):
        self._array = array

    def __sizeof__(self) -> int:
        return self.BASE_ARRAY_BYTES + self._array.nbytes


class MemoryMeasure:

    @staticmethod
    def _iter_iterable(obj: tp.Any) -> tp.Iterator[tp.Any]:
        '''
        Generates the iterable children that have not been counted by a getsizeof call on the parent object
        '''
        if hasattr(obj, '__iter__') and not isinstance(obj, str):
            if obj.__class__ is np.ndarray and obj.dtype.kind == DTYPE_OBJECT_KIND:
                # NOTE: iter(obj) would return slices for multi-dimensional arrays
                yield from (obj[loc] for loc in np.ndindex(obj.shape))
            elif isinstance(obj, (abc.Sequence, abc.Set)):
                yield from obj
            elif isinstance(obj, dict):
                yield from chain.from_iterable(obj.items())
            else:
                # The full size of the object is included in its getsizeof call
                # e.g. FrozenAutoMap, integer numpy arrays, int, float, etc.
                pass

    @staticmethod
    def _iter_slots(obj: tp.Any) -> tp.Iterator[tp.Any]:
        '''
        Generates an iterable of the values of all slot-based attributes in an object, including the slots contained in the object's parent classes based on the MRO
        '''
        # NOTE: This does NOT support 'single-string' slots (i.e. __slots__ = 'foo')
        slots = chain.from_iterable(
                cls.__slots__ for cls in obj.__class__.__mro__
                if hasattr(cls, '__slots__'))
        attrs = (getattr(obj, slot) for slot in slots
                if slot != '__weakref__' and hasattr(obj, slot))
        yield from attrs

    @classmethod
    def nested_sizable_elements(cls,
            obj: tp.Any,
            *,
            seen: tp.Set[int],
            ) -> tp.Iterator[tp.Any]:
        '''
        Generates an iterable of all objects the parent object has references to, including nested references. This function considers both the iterable unsized children (based on _iter_iterable) and the sizable
        attributes listed in its slots. The resulting generator is in pre-order and includes the parent object at the end.
        '''
        if id(obj) in seen:
            return
        seen.add(id(obj))
        is_array = obj.__class__ is np.ndarray

        if is_array and obj.dtype.kind != DTYPE_OBJECT_KIND:
            pass # non-object arrays report included elements
        else:
            for el in cls._iter_iterable(obj):
                yield from cls.nested_sizable_elements(el, seen=seen)

        for el in cls._iter_slots(obj):
            yield from cls.nested_sizable_elements(el, seen=seen)

        if is_array and obj.base is not None:
            # include the base array for numpy slices / views only if that base has not been seen
            yield from cls.nested_sizable_elements(obj.base, seen=seen)

        yield obj

def getsizeof_total(
        obj: tp.Any,
        *,
        seen: tp.Union[None, tp.Set[tp.Any]] = None,
        ) -> int:
    '''
    Returns the total size of the object and its references, including nested refrences
    '''
    seen = set() if seen is None else seen
    total = sum(getsizeof(el) for el in
            MemoryMeasure.nested_sizable_elements(obj, seen=seen))
    return total

