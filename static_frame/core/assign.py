import typing as tp


class Assign:
    '''
    Common base class for SeriesAssign and FrameAssign classes.
    '''
    __slots__ = ()

    INTERFACE: tp.Tuple[str, ...] = (
        '__call__',
        'apply',
        )


