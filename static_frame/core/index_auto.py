
import typing as tp

# from static_frame.core.index_base import IndexBase


from static_frame.core.index import Index
# from static_frame.core.index_hierarchy import IndexHierarchy

from static_frame.core.index import IndexGO

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

    # @classmethod
    # def from_is_static(cls,
    #         initializer: IndexAutoInitializer,
    #         *,
    #         is_static: bool,
    #         ) -> tp.Union[Index, IndexGO]:
    #     '''
    #     Args:
    #         initializer: An integer, or a sizable iterable.
    #         is_static: Boolean if this should be a static (not grow-only) index.
    #     '''
    #     labels = range(initializer)
    #     constructor = Index if is_static else IndexGO
    #     return constructor(
    #             labels=labels,
    #             loc_is_iloc=True,
    #             dtype=DEFAULT_INT_DTYPE
    #             )

    # @classmethod
    # def from_constructor(cls,
    #         initializer: IndexAutoInitializer,
    #         *,
    #         constructor: IndexConstructor,
    #         ) -> tp.Union[Index, IndexHierarchy]:

    #     labels = range(initializer)
    #     return constructor(labels)


    @classmethod
    def from_optional_constructor(cls,
            initializer: IndexAutoInitializer,
            *,
            default_constructor: IndexConstructor,
            explicit_constructor: tp.Optional[IndexConstructor] = None,
            ) -> tp.Union[Index, IndexGO]:

        labels = range(initializer)

        if explicit_constructor:
            return explicit_constructor(labels)

        else: # get from default constructor
            constructor = Index if default_constructor.STATIC else IndexGO
            return constructor(
                    labels=labels,
                    loc_is_iloc=True,
                    dtype=DEFAULT_INT_DTYPE
                    )


IndexAutoFactoryType = tp.Type[IndexAutoFactory]
