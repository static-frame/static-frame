from __future__ import annotations

import typing_extensions as tp


class Assign:
    '''
    Common base class for SeriesAssign and FrameAssign classes.
    '''
    __slots__ = ()

    _INTERFACE: tp.Tuple[str, ...] = (
        '__call__',
        'apply',
        )


