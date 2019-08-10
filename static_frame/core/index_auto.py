
import typing as tp

from static_frame.core.index import Index
from static_frame.core.index_hierarchy import IndexHierarchy

from static_frame.core.index import IndexGO

from static_frame.core.util import IndexInitializer
from static_frame.core.util import IndexConstructor
from static_frame.core.util import IndexAutoInitializer

from static_frame.core.util import DEFAULT_INT_DTYPE



# could create trival subclasses for these indices, but the type would would not always describe the instance; for example, an IndexAutoGO could grow inot non-contiguous integer index, as loc_is_iloc is reevaluated with each append can simply go to false.
#
# class IndexAuto(Index):
#     pass

# class IndexAutoGO(IndexGO):
#     pass


class IndexAutoFactory:

    # @staticmethod
    # def _get_labels(initializer: IndexAutoInitializer) -> tp.Iterable[int]:
    #     if isinstance(initializer, int):
    #         size = initializer
    #     elif hasattr(initializer, '__len__'):
    #         size = len(initializer)
    #     else:
    #         size = len(list(initializer))

    #     # NOTE: might be faster to directly create array of labels here and set to immutable
    #     return range(size)

    @classmethod
    def from_is_go(cls,
            initializer: IndexAutoInitializer,
            *,
            is_go: bool,
            ) -> tp.Union[Index, IndexGO]:
        '''
        Args:
            initializer: An integer, or a sizable iterable.
            is_go: Boolean if this should be a grow-only index.
        '''
        labels = range(initializer)
        constructor = IndexGO if is_go else Index
        return constructor(
                labels=labels,
                loc_is_iloc=True, # th
                dtype=DEFAULT_INT_DTYPE
                )

    @classmethod
    def from_constructor(cls,
            initializer: IndexAutoInitializer,
            *,
            constructor: IndexConstructor,
            ) -> tp.Union[Index, IndexHierarchy]:

        labels = range(initializer)
        return constructor(labels)