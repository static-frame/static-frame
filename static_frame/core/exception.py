

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

class ErrorInitIndexLevel(ErrorInit):
    '''Error in IndexBase (and derived Index) initialization.
    '''

class ErrorInitBus(ErrorInit):
    '''Error in Bus initialization.
    '''

class ErrorInitStore(ErrorInit):
    '''Error in Store initialization.
    '''



class LocEmpty(RuntimeError):
    pass

class LocInvalid(RuntimeError):
    pass


