import typing as tp
from collections import abc
from enum import Enum
from itertools import chain
from sys import getsizeof

import numpy as np

from static_frame.core.util import DTYPE_OBJECT_KIND


class MaterializedArray:
    '''Wrapper of array that delivers the sizeof as the fully realized size, ignoring any potential sharing of memory.
    '''

    __slots__ = (
            '_array',
            '_data_only',
            )
    BASE_ARRAY_BYTES = getsizeof(np.array(()))

    def __init__(self,
            array: np.ndarray,
            data_only: bool = False,
            ):
        self._array = array
        self._data_only = data_only

    def __sizeof__(self) -> int:
        if self._data_only:
            # NOTE: when called with getsizeof, the value here is
            return self._array.nbytes # type: ignore
        return self.BASE_ARRAY_BYTES + self._array.nbytes # type: ignore


# data only / all object components
# local / shared : arrays only, do we include reference data
# materialized / not: arrays only, just measure nbytes

from typing import NamedTuple

class MFConfig(NamedTuple):
    data_only: bool
    local_only: bool
    materialized: bool

class MeasureFormat(Enum):
    LOCAL = MFConfig(data_only=False, local_only=True, materialized=False) # only the array data unique to the array, ignoring referenced data
    SHARED = MFConfig(data_only=False, local_only=False, materialized=False) # array data unique to the array and any referenced array data
    MATERIALIZED = MFConfig(data_only=False, local_only=False, materialized=True) # ignore sharing get overall size based on data footprint
    MATERIALIZED_DATA = MFConfig(data_only=True, local_only=False, materialized=True) # just get data foot print, ignore all other components

MF = MeasureFormat


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
            format: MeasureFormat = MeasureFormat.SHARED,
            seen: tp.Set[int],
            ) -> tp.Iterator[tp.Any]:
        '''
        Generates an iterable of all objects the parent object has references to, including nested references. This function considers both the iterable unsized children (based on _iter_iterable) and the sizable
        attributes listed in its slots. The resulting generator is in pre-order and includes the parent object at the end.
        '''
        if id(obj) in seen:
            return
        seen.add(id(obj))

        if obj.__class__ is np.ndarray:

            if format.value.materialized:
                obj = MaterializedArray(obj, data_only=format.value.data_only)

            else:
                # non-object arrays report included elements
                if obj.dtype.kind == DTYPE_OBJECT_KIND:
                    for el in cls._iter_iterable(obj):
                        yield from cls.nested_sizable_elements(el, seen=seen, format=format)

                if not format.value.local_only and obj.base is not None:
                    # include the base array for numpy slices / views only if that base has not been seen
                    yield from cls.nested_sizable_elements(obj.base, seen=seen, format=format)

        elif obj.__class__ is MaterializedArray:
            # if a MaterializedArray was passed direclty in
            pass

        elif not format.value.data_only: # not array
            for el in cls._iter_iterable(obj): # will not yield anything if no __iter__
                yield from cls.nested_sizable_elements(el, seen=seen, format=format)
            # arrays do not have slots
            for el in cls._iter_slots(obj):
                yield from cls.nested_sizable_elements(el, seen=seen, format=format)

        # import ipdb; ipdb.set_trace()
        yield obj



def getsizeof_total(
        obj: tp.Any,
        *,
        format: MF = MF.SHARED,
        seen: tp.Union[None, tp.Set[tp.Any]] = None,
        ) -> int:
    '''
    Returns the total size of the object and its references, including nested refrences
    '''
    seen = set() if seen is None else seen

    def gen() -> tp.Iterator[int]:
        for component in MemoryMeasure.nested_sizable_elements(obj, seen=seen, format=format):
            # import ipdb; ipdb.set_trace()
            # if format is MF.MATERIALIZED_DATA:
            if format.value.data_only and component.__class__ is MaterializedArray:
                yield component.__sizeof__() # call directly to avoid gc ovehead addition
                # ignore all other components
            else:
                yield getsizeof(component)

    return sum(gen())
