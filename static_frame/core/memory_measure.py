from __future__ import annotations

from collections import abc
from itertools import chain
from sys import getsizeof
from typing import NamedTuple

import numpy as np
import typing_extensions as tp

from static_frame.core.display_config import DisplayConfig
from static_frame.core.util import DTYPE_OBJECT_KIND
from static_frame.core.util import EMPTY_ARRAY
from static_frame.core.util import bytes_to_size_label

if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame  # pragma: no cover
    TNDArrayAny = np.ndarray[tp.Any, tp.Any] #pragma: no cover
    TFrameAny = Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]] #pragma: no cover

class MFConfig(NamedTuple):
    local_only: bool # only data locally owned by arrays, or all referenced data
    materialized: bool # measure byte payload nbytes (regardless of sharing)
    data_only: bool # only array byte payloads, or all objects


class MeasureFormatMeta(type):
    def __iter__(cls) -> tp.Iterator[MFConfig]:
        return (v for (k, v) in vars(cls).items() if not k.startswith('_'))

class MeasureFormat(metaclass=MeasureFormatMeta):
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
    BASE_ARRAY_BYTES = getsizeof(EMPTY_ARRAY)

    def __init__(self,
            array: TNDArrayAny,
            format: MFConfig = MeasureFormat.LOCAL,
            ):
        self._array = array
        self._format = format

    def __sizeof__(self) -> int:
        size = 0
        if self._format.local_only and self._array.base is not None:
            pass # all data referenced externally
        else:
            size += self._array.nbytes

        if not self._format.data_only:
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
            format: MFConfig = MeasureFormat.REFERENCED,
            seen: tp.Set[int],
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
            if format.materialized:
                obj = MaterializedArray(obj, format=format)
            else: # non-object arrays report included elements
                if obj.dtype.kind == DTYPE_OBJECT_KIND:
                    for el in cls._iter_iterable(obj):
                        yield from cls.nested_sizable_elements(el, seen=seen, format=format)
                if not format.local_only and obj.base is not None:
                    # include the base array for numpy slices / views only if that base has not been seen
                    yield from cls.nested_sizable_elements(obj.base, seen=seen, format=format)
            yield obj

        elif obj.__class__ is MaterializedArray:
            yield obj

        else:
            for el in cls._iter_iterable(obj): # will not yield if no __iter__
                yield from cls.nested_sizable_elements(el, seen=seen, format=format)
            # arrays do not have slots
            for el in cls._iter_slots(obj):
                yield from cls.nested_sizable_elements(el, seen=seen, format=format)
            yield obj


def memory_total(
        obj: tp.Any,
        *,
        format: MFConfig = MeasureFormat.REFERENCED,
        seen: tp.Union[None, tp.Set[tp.Any]] = None,
        ) -> int:
    '''
    Returns the total size of the object and its references, including nested refrences
    '''
    seen = set() if seen is None else seen

    def gen() -> tp.Iterator[int]:
        for component in MemoryMeasure.nested_sizable_elements(obj,
                seen=seen,
                format=format,
                ):
            if format.data_only and component.__class__ is MaterializedArray:
                yield component.__sizeof__() # call directly to avoid gc ovehead addition
            else:
                yield getsizeof(component)

    return sum(gen())

class MemoryDisplay:
    '''A simple container for capturing and displaying memory usage in bytes for StaticFrame containers.
    '''

    __slots__ = (
            '_frame',
            '_repr',
            )

    @classmethod
    def from_any(cls,
            obj: tp.Any,
            label_component_pairs: tp.Iterable[tp.Tuple[str, tp.Any]] = (),
            ) -> 'MemoryDisplay':
        '''Given any slotted object, return a :obj:`MemoryDisplay` instance.

        '''
        from static_frame.core.frame import Frame

        parts = chain(label_component_pairs, (('Total', obj),))

        def gen() -> tp.Iterator[tp.Tuple[tp.Tuple[str, ...], tp.List[int]]]:
            for label, part in parts:
                sizes = []
                for format in MeasureFormat:
                    # NOTE: not sharing seen accross evaluations
                    sizes.append(memory_total(part, format=format))
                yield label, sizes # pyright: ignore

        if hasattr(obj, 'name') and obj.name is not None:
            name = f'<{obj.__class__.__name__}: {obj.name}>'
        else:
            name = f'<{obj.__class__.__name__}>'

        f: TFrameAny = Frame.from_records_items(
                gen(),
                columns=(FORMAT_TO_DISPLAY[mf] for mf in MeasureFormat),
                name=name,
                )
        return cls(f)

    def __init__(self, frame: TFrameAny):
        '''Initialize an instance with a ``Frame`` of byte counts.
        '''
        from static_frame.core.frame import Frame
        self._frame = frame

        f_size = self._frame.iter_element().apply(bytes_to_size_label)

        def gen() -> tp.Iterator[tp.Sequence[str]]:
            for row_old in f_size.iter_tuple(axis=1):
                row_new = []
                for e in row_old:
                    row_new.extend(e.split(' '))
                yield row_new

        f: TFrameAny = Frame.from_records(gen(), index=f_size.index)
        columns = [
                f_size.columns[i // 2] if i % 2 == 0
                else f'{f_size.columns[i // 2]}u'.ljust(5)
                for i in range(len(f.columns))
                ]
        f = f.relabel(columns=columns)
        dc = DisplayConfig(type_show=False)
        self._repr: str = f.display(config=dc).__repr__()

    def __repr__(self) -> str:
        return self._repr

    def to_frame(self) -> TFrameAny:
        '''Return a Frame of byte counts.
        '''
        return self._frame

