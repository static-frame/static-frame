import typing as tp
from collections import abc
from itertools import chain
from sys import getsizeof

import numpy as np

from static_frame.core.util import DTYPE_OBJECT_KIND


class MemoryMeasure:

    @staticmethod
    def _unsized_children(obj: tp.Any) -> tp.Iterator[tp.Any]:
        '''
        Generates the iterable children that have not been counted by a getsizeof call on the parent object
        '''
        # Check if iterable or a string first for fewer isinstance calls on common types
        if hasattr(obj, '__iter__') and not isinstance(obj, str):
            if obj.__class__ is np.ndarray and obj.dtype.kind == DTYPE_OBJECT_KIND:
                # Only return the referenced python objects not counted by numpy.
                # NOTE: iter(obj) would return slices for multi-dimensional object arrays
                yield from (obj[loc] for loc in np.ndindex(obj.shape))
                # What about numpy array references, double-check the data
            elif (
                isinstance(obj, abc.Sequence) # tuple, list
                or isinstance(obj, abc.Set) # set, frozenset
            ):
                yield from obj
            elif isinstance(obj, dict):
                yield from chain.from_iterable(obj.items())
            else:
                # The full size of the object is included in its getsizeof call
                # e.g. FrozenAutoMap, integer numpy arrays, int, float, etc.
                pass

    @staticmethod
    def _sizable_slot_attrs(obj: tp.Any) -> tp.Iterator[tp.Any]:
        '''
        Generates an iterable of the values of all slot-based attributes in an object, including the slots contained in the object's parent classes based on the MRO
        '''
        # NOTE: This does NOT support 'single-string' slots (i.e. __slots__ = 'foo')
        slots = chain.from_iterable(cls.__slots__ for cls in obj.__class__.__mro__ if hasattr(cls, '__slots__'))
        attrs = (getattr(obj, slot) for slot in slots if slot != '__weakref__' and hasattr(obj, slot))
        yield from attrs

    @classmethod
    def nested_sizable_elements(cls,
            obj: tp.Any,
            *,
            seen: tp.Set[int],
            ) -> tp.Iterator[tp.Any]:
        '''
        Generates an iterable of all objects the parent object has references to, including nested references. This function considers both the iterable unsized children (based on _unsized_children) and the sizable
        attributes listed in its slots. The resulting generator is in pre-order and includes the parent object at the end.
        '''
        if id(obj) in seen:
            return
        seen.add(id(obj))

        for el in cls._unsized_children(obj):
            yield from cls.nested_sizable_elements(el, seen=seen)
        for el in cls._sizable_slot_attrs(obj):
            yield from cls.nested_sizable_elements(el, seen=seen)

        if obj.__class__ is np.ndarray and obj.base is not None:
            # include the base array for numpy slices / views only if that base has not been seen
            yield from cls.nested_sizable_elements(obj.base, seen=seen)

        yield obj

def getsizeof_total(obj: tp.Any, *, seen: tp.Union[None, tp.Set[tp.Any]] = None) -> int:
    '''
    Returns the total size of the object and its references, including nested refrences
    '''
    seen = set() if seen is None else seen
    total = sum(getsizeof(el) for el in MemoryMeasure.nested_sizable_elements(obj, seen=seen))
    return total

