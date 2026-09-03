from __future__ import annotations


class Assign:
    """
    Common base class for SeriesAssign and FrameAssign classes.
    """

    __slots__ = ()

    _INTERFACE: tuple[str, ...] = (
        '__call__',
        'apply',
    )
