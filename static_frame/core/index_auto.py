
import typing as tp
from functools import partial

from static_frame.core.index import Index
from static_frame.core.index import IndexGO
from static_frame.core.util import PositionsAllocator
from static_frame.core.index_base import IndexBase  # pylint: disable = W0611
from static_frame.core.util import DTYPE_INT_DEFAULT
from static_frame.core.util import IndexConstructor
from static_frame.core.util import CallableOrMapping
from static_frame.core.util import IndexInitializer
from static_frame.core.util import NameType



class IndexDefaultFactory:
    '''
    Token class to be used to only provide a ``name`` to a default constructor of an Index.
    '''
    __slots__ = ('_name',)

    def __init__(self, name: NameType):
        self._name = name

    def __call__(self, constructor: IndexConstructor) -> IndexConstructor:
        '''Partial the passeed constructor with the ``name``.
        '''
        return partial(constructor, name=self._name)

IndexAutoInitializer = int

# could create trival subclasses for these indices, but the type would would not always describe the instance; for example, an IndexAutoGO could grow inot non-contiguous integer index, as loc_is_iloc is reevaluated with each append can simply go to false.

class IndexAutoFactory:
    __slots__ = ('_size', '_name')

    @classmethod
    def from_optional_constructor(cls,
            initializer: IndexAutoInitializer, # size
            *,
            default_constructor: tp.Type['IndexBase'],
            explicit_constructor: tp.Optional[tp.Union[IndexConstructor, IndexDefaultFactory]] = None,
            ) -> 'IndexBase':

        # get an immutable array, shared from positions allocator
        labels = PositionsAllocator.get(initializer)

        if explicit_constructor:
            if isinstance(explicit_constructor, IndexDefaultFactory):
                # partial the default constructor with a name argument
                return explicit_constructor(default_constructor)(labels)
            return explicit_constructor(labels)

        else: # get from default constructor
            constructor = Index if default_constructor.STATIC else IndexGO
            return constructor(
                    labels=labels,
                    loc_is_iloc=True,
                    dtype=DTYPE_INT_DEFAULT
                    )

    def __init__(self,
            size: IndexAutoInitializer,
            *,
            name: NameType = None,
            ):
        self._size = size
        self._name = name

    def to_index(self,
            *,
            default_constructor: tp.Type['IndexBase'],
            explicit_constructor: tp.Optional[tp.Union[IndexConstructor, IndexDefaultFactory]] = None,
            ) -> 'IndexBase':
        '''Called by index_from_optional_constructor.
        '''
        return self.from_optional_constructor(self._size,
                default_constructor=default_constructor,
                explicit_constructor=explicit_constructor,
                )



IndexAutoFactoryType = tp.Type[IndexAutoFactory]
RelabelInput = tp.Union[CallableOrMapping, IndexAutoFactoryType, IndexInitializer]
