
import typing as tp
import numpy as np
from numpy import char as npc

from static_frame.core.util import DT64_YEAR

if tp.TYPE_CHECKING:

    from static_frame.core.frame import Frame  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index import Index  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchy  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.series import Series  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.type_blocks import TypeBlocks  #pylint: disable = W0611 #pragma: no cover

# only ContainerOperand subclasses
TContainer = tp.TypeVar('TContainer', 'Index', 'IndexHierarchy', 'Series', 'Frame', 'TypeBlocks')

ToBlocksType = tp.Callable[[], tp.Iterator[np.ndarray]]
ToContainerType = tp.Callable[[np.ndarray], TContainer]


class InterfaceDatetime(tp.Generic[TContainer]):

    __slots__ = (
        '_func_to_blocks', # function that returns iterable of arrays
        '_func_to_container', # partialed function that will return a new container
        )


    def __init__(self,
            func_to_blocks: ToBlocksType,
            func_to_container: ToContainerType[TContainer]
            ) -> None:
        self._func_to_blocks: ToBlocksType = func_to_blocks
        self._func_to_container: ToContainerType[TContainer] = func_to_container


    @property
    def year(self) -> TContainer:
        def blocks() -> tp.Iterator[np.ndarray]:
            for block in self._func_to_blocks():
                # NOTE: need same special handling for integers the Index construciton uses
                yield block.astype(DT64_YEAR)

        return self._func_to_container(blocks())


    @property
    def month(self) -> TContainer:
        pass


    @property
    def day(self) -> TContainer:
        pass