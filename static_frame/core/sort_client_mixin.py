import typing as tp

from static_frame.core.util import SortStatus, TNDArrayIntDefault


class SortClientMixin:
    __slots__ = ()

    def __copy__(self) -> tp.Self:
        """
        Return a shallow copy of this container, with no data copied.
        """
        raise NotImplementedError()  # pragma: no cover

    def _reverse(self, axis: int = 0) -> tp.Self:
        """
        Return a reversed copy of this container, with no data copied.
        """
        raise NotImplementedError()  # pragma: no cover

    def _apply_ordering(
        self,
        order: TNDArrayIntDefault,
        sort_status: SortStatus,
        axis: int = 0,
    ) -> tp.Self:
        """
        Return a copy of this container with the specified ordering applied along the index of axis
        """
        raise NotImplementedError()


TSortClient = tp.TypeVar('TSortClient', bound=SortClientMixin)
