
import typing as tp


from static_frame.core.index import Index
from static_frame.core.index import IndexGO
from static_frame.core.util import PositionsAllocator
from static_frame.core.index_base import IndexBase  # pylint: disable = W0611
from static_frame.core.util import DTYPE_INT_DEFAULT
from static_frame.core.util import IndexConstructor
from static_frame.core.util import CallableOrMapping
from static_frame.core.util import IndexInitializer


IndexAutoInitializer = int


# could create trival subclasses for these indices, but the type would would not always describe the instance; for example, an IndexAutoGO could grow inot non-contiguous integer index, as loc_is_iloc is reevaluated with each append can simply go to false.

class IndexAutoFactory:

    @classmethod
    def from_optional_constructor(cls,
            initializer: IndexAutoInitializer,
            *,
            default_constructor: tp.Type['IndexBase'],
            explicit_constructor: tp.Optional[IndexConstructor] = None,
            ) -> 'IndexBase':

        # get an immutable array, shared from positions allocator
        labels = PositionsAllocator.get(initializer)

        if explicit_constructor:
            return explicit_constructor(labels)

        else: # get from default constructor
            constructor = Index if default_constructor.STATIC else IndexGO
            return constructor(
                    labels=labels,
                    loc_is_iloc=True,
                    dtype=DTYPE_INT_DEFAULT
                    )


IndexAutoFactoryType = tp.Type[IndexAutoFactory]
RelabelInput = tp.Union[CallableOrMapping, IndexAutoFactoryType, IndexInitializer]
