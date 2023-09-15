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
    def __init__(self) -> None:
        super().__init__('Relabelling with an unordered iterable is not permitted.')

class BatchIterableInvalid(RuntimeError):
    def __init__(self) -> None:
        super().__init__('Batch iterable does not yield expected pair of label, container.')

class InvalidDatetime64Comparison(RuntimeError):
    def __init__(self) -> None:
        super().__init__('Cannot perform set operations on datetime64 of different units; use astype to align units before comparison.')

class InvalidDatetime64Initializer(RuntimeError):
    pass

class InvalidFillValue(RuntimeError):
    def __init__(self, fill_value: tp.Any, context: str) -> None:
        super().__init__(f'{fill_value} not supported in the context of {context}.')


class InvalidWindowLabel(IndexError):
    def __init__(self, label_iloc: int) -> None:
        super().__init__(f'A label cannot be assigned to the window for position {label_iloc}; set `label_missing_raises` to `False` or update `label_shift` to select an appropriate label relative to the window.')


#-------------------------------------------------------------------------------

class StoreFileMutation(RuntimeError):
    '''
    A Stores file was mutated in an unexpected way.
    '''

class StoreParameterConflict(RuntimeError):
    pass

class StoreLabelNonUnique(RuntimeError):
    def __init__(self, label: str) -> None:
        super().__init__(f'Store label "{label}" is not unique.')

class NotImplementedAxis(NotImplementedError):
    def __init__(self) -> None:
        super().__init__('Iteration along this axis is too inefficient; create a consolidated Frame with Quilt.to_frame()')


#-------------------------------------------------------------------------------
# NOTE: these are dervied from ValueError to match NumPy convention

class ErrorNPYEncode(ValueError):
    '''
    Error encoding an NPY file.
    '''

class ErrorNPYDecode(ValueError):
    '''
    Error decoding an NPY file.
    '''

class ErrorNotTruthy(ValueError):
    def __init__(self) -> None:
        super().__init__('The truth value of a container is ambiguous. For a truthy indicator of non-empty status, use the `size` attribute.')

#-------------------------------------------------------------------------------

def deprecated(message: str = '') -> None:
    # using UserWarning to get out of pytest with  -p no:warnings
    warnings.warn(message, UserWarning, stacklevel=2) #pragma: no cover

