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



#-------------------------------------------------------------------------------

class StoreFileMutation(RuntimeError):
    '''
    A Stores file was mutated in an unexpected way.
    '''
#-------------------------------------------------------------------------------

def deprecated(message: str = '') -> None:
    # using UserWarning to get out of pytest with  -p no:warnings
    warnings.warn(message, UserWarning, stacklevel=2) #pragma: no cover

