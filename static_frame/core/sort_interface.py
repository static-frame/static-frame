import typing_extensions as tp  # pragma: no cover

from static_frame.core.util import SortStatus, TNDArrayIntDefault  # pragma: no cover


class SortInterface(tp.Protocol):  # pragma: no cover
    """Defines the interface for objects that can be sorted"""

    def __copy__(self) -> tp.Self:  # pragma: no cover
        """
        Return a shallow copy of this container, with no data copied.
        """
        raise NotImplementedError()  # pragma: no cover

    def _reverse(self, axis: int = 0) -> tp.Self:  # pragma: no cover
        """
        Return a reversed copy of this container, with no data copied.
        """
        raise NotImplementedError()  # pragma: no cover

    def _apply_ordering(  # pragma: no cover
        self,
        order: TNDArrayIntDefault,
        sort_status: SortStatus,
        axis: int = 0,
    ) -> tp.Self:
        """
        Return a copy of this container with the specified ordering applied along the index of axis
        """
        raise NotImplementedError()  # pragma: no cover


TSortInterface = tp.TypeVar('TSortInterface', bound=SortInterface)  # pragma: no cover
