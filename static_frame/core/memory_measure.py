import typing as tp
from collections import abc
from enum import Enum
from itertools import chain
from sys import getsizeof
from typing import NamedTuple

import numpy as np

from static_frame.core.util import DTYPE_OBJECT_KIND
from static_frame.core.util import bytes_to_size_label

if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame  # pylint: disable=W0611 #pragma: no cover


class MFConfig(NamedTuple):
    local_only: bool # only data locally owned by arrays, or all referenced data
    materialized: bool # measure byte payload nbytes (regardless of sharing)
    data_only: bool # only array byte payloads, or all objects

class MeasureFormat(Enum):
    LOCAL = MFConfig(
            local_only=True,
            materialized=False,
            data_only=False,
            )
    LOCAL_MATERIALIZED = MFConfig(
            local_only=True,
            materialized=True,
            data_only=False,
            )
    LOCAL_MATERIALIZED_DATA = MFConfig(
            local_only=True,
            materialized=True,
            data_only=True,
            )
    # NOTE MFConfig(local_only=True, materialized=False, data_only=True) is invalid

    REFERENCED = MFConfig(
            local_only=False,
            materialized=False,
            data_only=False,
            )
    REFERENCED_MATERIALIZED = MFConfig(
            local_only=False,
            materialized=True,
            data_only=False,
            )
    REFERENCED_MATERIALIZED_DATA = MFConfig(
            local_only=False,
            materialized=True,
            data_only=True,
            )
    # NOTE MFConfig(local_only=False, materialized=False, data_only=True) is invalid

FORMAT_TO_DISPLAY = {
        MeasureFormat.LOCAL: 'L',
        MeasureFormat.LOCAL_MATERIALIZED: 'LM',
        MeasureFormat.LOCAL_MATERIALIZED_DATA: 'LMD',
        MeasureFormat.REFERENCED: 'R',
        MeasureFormat.REFERENCED_MATERIALIZED: 'RM',
        MeasureFormat.REFERENCED_MATERIALIZED_DATA: 'RMD',
        }

class MaterializedArray:
    '''Wrapper of array that delivers the sizeof as the fully realized size, ignoring any potential sharing of memory.
    '''

    __slots__ = (
            '_array',
            '_format',
            )
    BASE_ARRAY_BYTES = getsizeof(np.array(()))

    def __init__(self,
            array: np.ndarray,
            format: MeasureFormat = MeasureFormat.LOCAL,
            ):
        self._array = array
        self._format = format

    def __sizeof__(self) -> int:
        size = 0
        if self._format.value.local_only and self._array.base is not None:
            pass # all data referenced externally
        else:
            size += self._array.nbytes

        if not self._format.value.data_only:
            size += self.BASE_ARRAY_BYTES

        return size


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
                # e.g. SF Containers, FrozenAutoMap, integer numpy arrays, int, float, etc.
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
            format: MeasureFormat = MeasureFormat.REFERENCED,
            seen: tp.Set[int],
            skip_parent: bool = False,
            ) -> tp.Iterator[tp.Any]:
        '''
        Generates an iterable of all objects the parent object has references to, including nested references. This function considers both the iterable unsized children (based on _iter_iterable) and the sizable
        attributes listed in its slots. The resulting generator is in pre-order and includes the parent object at the end.
        '''
        from static_frame.core.container import ContainerBase

        if id(obj) in seen:
            return
        seen.add(id(obj))

        if obj.__class__ is np.ndarray:
            if format.value.materialized:
                obj = MaterializedArray(obj, format=format)
            else:
                # non-object arrays report included elements
                if obj.dtype.kind == DTYPE_OBJECT_KIND:
                    for el in cls._iter_iterable(obj):
                        yield from cls.nested_sizable_elements(el, seen=seen, format=format)

                if not format.value.local_only and obj.base is not None:
                    # include the base array for numpy slices / views only if that base has not been seen
                    yield from cls.nested_sizable_elements(obj.base, seen=seen, format=format)

        if obj.__class__ is np.ndarray or obj.__class__ is MaterializedArray:
            # classes that naturally report total size without introspection of iterables or slots
            pass
        else:
            # elif not format.value.data_only: # not array
            for el in cls._iter_iterable(obj): # will not yield if no __iter__
                yield from cls.nested_sizable_elements(el, seen=seen, format=format)
            # arrays do not have slots
            for el in cls._iter_slots(obj):
                yield from cls.nested_sizable_elements(el, seen=seen, format=format)

        if not skip_parent:
            yield obj


def memory_total(
        obj: tp.Any,
        *,
        format: MeasureFormat = MeasureFormat.REFERENCED,
        seen: tp.Union[None, tp.Set[tp.Any]] = None,
        skip_parent: bool = False,
        ) -> int:
    '''
    Returns the total size of the object and its references, including nested refrences
    '''
    seen = set() if seen is None else seen

    def gen() -> tp.Iterator[int]:
        for component in MemoryMeasure.nested_sizable_elements(obj,
                seen=seen,
                format=format,
                skip_parent=skip_parent,
                ):
            if format.value.data_only and component.__class__ is MaterializedArray:
                yield component.__sizeof__() # call directly to avoid gc ovehead addition
            else:
                yield getsizeof(component)

    return sum(gen())


def memory_display(
        obj: tp.Any,
        label_component_pairs: tp.Iterable[tp.Tuple[str, tp.Any]],
        *,
        size_label: bool = True,
        ) -> 'Frame':
    '''
    Args:
        label_component_pairs: provide paris of label, attribute component
    '''
    from static_frame.core.frame import Frame

    parts = chain(label_component_pairs, (('Total', obj),))

    def gen() -> tp.Iterator[tp.Tuple[tp.Tuple[str, ...], tp.List[int]]]:
        for label, part in parts:
            sizes = []
            for format in MeasureFormat:
                # NOTE: not sharing seen accross evaluations
                sizes.append(memory_total(part, format=format))
            yield label, sizes

    if hasattr(obj, 'name') and obj.name is not None:
        name = f'<{obj.__class__.__name__}: {obj.name}>'
    else:
        name = f'<{obj.__class__.__name__}>'

    f = Frame.from_records_items(
            gen(),
            columns=(FORMAT_TO_DISPLAY[mf] for mf in MeasureFormat),
            name=name,
            )
    if size_label:
        f = f.iter_element().apply(bytes_to_size_label, name=name)
    return f # type: ignore


