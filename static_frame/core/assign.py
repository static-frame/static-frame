from static_frame.core.util import EMPTY_TUPLE

class Assign:
    '''
    Common base class for SeriesAssign and FrameAssign classes.
    '''
    __slots__ = EMPTY_TUPLE

    INTERFACE = (
        '__call__',
        'apply',
        )


