
import typing as tp

from static_frame.core.index_base import IndexBase
from static_frame.core.index import Index
from static_frame.core.util import IndexInitializer


# take integers for size; otherwise, extract size from any other index initializer
IndexAutoInitializer = tp.Union[int, IndexInitializer]

class IndexAuto(Index):

    def __init__(self,
            initializer: IndexInitializer,
            *,
            name: tp.Hashable = None
            ):

        if isinstance(initializer, int):
            size = initializer
        elif hasattr(initializer, '__len__'):
            size = len(initializer)
        else:
            size = len(list(initializer))

        # produce a default name? auto{size}

        Index.__init__(self,
                labels=range(size),
                name=name
                loc_is_iloc=True,
                dtype=DEFAULT_INT_DTYPE
                )


class IndexAutoGO(IdnexGO):
    pass