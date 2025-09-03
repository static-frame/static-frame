from __future__ import annotations

from abc import ABCMeta

import typing_extensions as tp

from static_frame.core.doc_str import doc_inject

if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame

    TFrameAny = Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]]


class InterfaceMeta(ABCMeta):
    """Lowest level metaclass for providing interface property on class."""

    @property
    @doc_inject()
    def interface(cls) -> TFrameAny:
        """{}"""
        from static_frame.core.interface import InterfaceSummary

        return InterfaceSummary.to_frame(cls)  # type: ignore
