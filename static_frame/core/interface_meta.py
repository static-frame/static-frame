from __future__ import annotations

import typing_extensions as tp

from static_frame.core.doc_str import doc_inject

if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame  # pragma: no cover

    TFrameAny = Frame[
        tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]
    ]  # pragma: no cover


class InterfaceMeta(type):
    """Lowest level metaclass for providing interface property on class."""

    @property
    @doc_inject()
    def interface(cls) -> TFrameAny:
        """{}"""
        from static_frame.core.interface import InterfaceSummary

        return InterfaceSummary.to_frame(cls)  # type: ignore
