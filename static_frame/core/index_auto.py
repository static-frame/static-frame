
import typing as tp

from static_frame.core.index import Index
from static_frame.core.index_hierarchy import IndexHierarchy

from static_frame.core.index import IndexGO

from static_frame.core.util import IndexInitializer
from static_frame.core.util import IndexConstructor

from static_frame.core.util import DEFAULT_INT_DTYPE


IndexAutoInitializer = int


# could create trival subclasses for these indices, but the type would would not always describe the instance; for example, an IndexAutoGO could grow inot non-contiguous integer index, as loc_is_iloc is reevaluated with each append can simply go to false.
#
# class IndexAuto(Index):
#     pass

# class IndexAutoGO(IndexGO):
#     pass


class IndexAutoFactory:

    @classmethod
    def from_is_static(cls,
            initializer: IndexAutoInitializer,
            *,
            is_static: bool,
            ) -> tp.Union[Index, IndexGO]:
        '''
        Args:
            initializer: An integer, or a sizable iterable.
            is_static: Boolean if this should be a static (not grow-only) index.
        '''
        labels = range(initializer)
        constructor = Index if is_static else IndexGO
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



IndexAutoFactoryType = tp.Type[IndexAutoFactory]
