from __future__ import annotations

import warnings

import typing_extensions as tp


class ErrorInit(RuntimeError):
    '''Error in Container initialization.
    '''

class ErrorInitSeries(ErrorInit):
    '''Error in Series initialization.
    '''

class ErrorInitFrame(ErrorInit):
    '''Error in Frame (and derived Frame) initialization.
    '''

class ErrorInitIndex(ErrorInit):
    '''Error in IndexBase (and derived Index) initialization.
    '''

class ErrorInitColumns(ErrorInitIndex):
    '''Error in IndexBase (and derived Index) initialization of columns.
    '''

class ErrorInitIndexNonUnique(ErrorInitIndex):
    '''Error in IndexBase initialization due to non-unique values.
    '''

class ErrorInitBus(ErrorInit):
    '''Error in Bus initialization.
    '''

class ErrorInitQuilt(ErrorInit):
    '''Error in Quilt initialization.
    '''

class ErrorInitYarn(ErrorInit):
    '''Error in Yarn initialization.
    '''

class ErrorInitStore(ErrorInit):
    '''Error in Store initialization.
    '''

class ErrorInitStoreConfig(ErrorInit):
    '''Error in StoreConfig initialization.
    '''

#-------------------------------------------------------------------------------

class LocEmpty(RuntimeError):
    pass

class LocInvalid(RuntimeError):
    pass

class AxisInvalid(RuntimeError):
    pass

class RelabelInvalid(RuntimeError):
    def __init__(self, *args: tp.Any) -> None:
        super().__init__('Relabelling with an unordered iterable is not permitted.', *args)

class BatchIterableInvalid(RuntimeError):
    def __init__(self, *args: tp.Any) -> None:
        super().__init__('Batch iterable does not yield expected pair of label, container.', *args)

class InvalidDatetime64Comparison(RuntimeError):
    def __init__(self, *args: tp.Any) -> None:
        super().__init__('Cannot perform set operations on datetime64 of different units; use astype to align units before comparison.', *args)

class InvalidDatetime64Initializer(RuntimeError):
    pass

class InvalidFillValue(RuntimeError):
    pass

def invalid_fill_value_factory(fill_value: tp.Any, context: str) -> InvalidFillValue:
    msg = f'{fill_value} not supported in the context of {context}.'
    return InvalidFillValue(msg)


class InvalidWindowLabel(IndexError):
    pass

def invalid_window_label_factory(label_iloc: int) -> InvalidWindowLabel:
    msg = f'A label cannot be assigned to the window for position {label_iloc}; set `label_missing_raises` to `False` or update `label_shift` to select an appropriate label relative to the window.'
    return InvalidWindowLabel(msg)


class GrowOnlyInvalid(RuntimeError):
    def __init__(self, *args: tp.Any) -> None:
        super().__init__('Cannot perform an in-place grow-only operation due to the class of the columns Index.')

#-------------------------------------------------------------------------------

class StoreFileMutation(RuntimeError):
    '''
    A Stores file was mutated in an unexpected way.
    '''

class StoreParameterConflict(RuntimeError):
    pass

class StoreLabelNonUnique(RuntimeError):
    pass

def store_label_non_unique_factory(label: str) -> StoreLabelNonUnique:
    msg = f'Store label "{label}" is not unique.'
    return StoreLabelNonUnique(msg)


class NotImplementedAxis(NotImplementedError):
    def __init__(self, *args: tp.Any) -> None:
        super().__init__('Iteration along this axis is too inefficient; create a consolidated Frame with Quilt.to_frame()', *args)

class ImmutableTypeError(TypeError):
    pass

def immutable_type_error_factory(
            cls: tp.Type[tp.Any],
            interface: str,
            key: tp.Any,
            value: tp.Any,
            ) -> ImmutableTypeError:
    from static_frame.core.store_client_mixin import StoreClientMixin
    if issubclass(cls, StoreClientMixin):
        # no assign interface
        msg = f'{cls.__name__} is immutable.'
    else:
        # only provide reprs for simple types
        classes = (bool, int, float, str, bytes, tuple, list)
        if isinstance(key, classes) and isinstance(value, classes):
            example = f'`{cls.__name__}.assign{"." if interface else ""}{interface}[{key!r}]({value!r})`'
        else:
            example = f'`{cls.__name__}.assign{"." if interface else ""}{interface}[key](value)`'
        msg = f'{cls.__name__} is immutable; use {example} to derive a modified container.'

    return ImmutableTypeError(msg)

#-------------------------------------------------------------------------------
# NOTE: these are derived from ValueError to match NumPy convention

class ErrorNPYEncode(ValueError):
    '''
    Error encoding an NPY file.
    '''

class ErrorNPYDecode(ValueError):
    '''
    Error decoding an NPY file.
    '''

class ErrorNotTruthy(ValueError):
    def __init__(self, *args: tp.Any) -> None:
        super().__init__('The truth value of a container is ambiguous. For a truthy indicator of non-empty status, use the `size` attribute.', *args)

#-------------------------------------------------------------------------------

def deprecated(message: str = '') -> None:
    # using UserWarning to get out of pytest with  -p no:warnings
    warnings.warn(message, UserWarning, stacklevel=2) #pragma: no cover

