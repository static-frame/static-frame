import warnings

class ErrorInit(RuntimeError):
    '''Error in Container initialization.
    '''

class ErrorInitTypeBlocks(ErrorInit):
    '''Error in TypeBlocks initialization.
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

class ErrorInitIndexNonUnique(ErrorInitIndex):
    '''Error in IndexBase initialization due to non-unique values.
    '''

class ErrorInitIndexLevel(ErrorInit):
    '''Error in IndexBase (and derived Index) initialization.
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
        super().__init__('Batch iterable does not yield expected pair of label, Frame.')

#-------------------------------------------------------------------------------

class StoreFileMutation(RuntimeError):
    '''
    A Stores file was mutated in an unexpected way.
    '''

class StoreParameterConflict(RuntimeError):
    '''
    A Stores file was mutated in an unexpected way.
    '''

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

#-------------------------------------------------------------------------------

def deprecated(message: str = '') -> None:
    # using UserWarning to get out of pytest with  -p no:warnings
    warnings.warn(message, UserWarning, stacklevel=2) #pragma: no cover

