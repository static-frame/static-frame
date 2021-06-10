import typing as tp
import re
from functools import partial

import numpy as np

from static_frame.core.node_selector import Interface
from static_frame.core.node_selector import TContainer
from static_frame.core.util import array_from_element_apply
from static_frame.core.util import DTYPE_STR
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import AnyCallable
from static_frame.core.util import DTYPE_STR_KINDS

if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index import Index  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchy  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.series import Series  #pylint: disable = W0611 #pragma: no cover


BlocksType = tp.Iterable[np.ndarray]
ToContainerType = tp.Callable[[tp.Iterator[np.ndarray]], TContainer]

class InterfaceRe(Interface[TContainer]):

    __slots__ = (
            '_blocks',
            '_blocks_to_container',
            '_pattern',
            )
    INTERFACE = (
            'search',
            'match',
            'fullmatch',
            'split',
            'findall',
            'sub',
            'subn',
            'groups', # property
            )


    def __init__(self,
            blocks: BlocksType,
            blocks_to_container: ToContainerType[TContainer],
            pattern: str,
            flags: int = 0,
            ) -> None:
        self._blocks: BlocksType = blocks
        self._blocks_to_container: ToContainerType[TContainer] = blocks_to_container

        self._pattern = re.compile(pattern, flags)

    @staticmethod
    def _process_blocks(*,
            blocks: BlocksType,
            func: AnyCallable,
            dtype: np.dtype,
            ) -> tp.Iterator[np.ndarray]:
        '''
        Element-wise processing of a methods on objects in a block
        '''
        for block in blocks:
            if block.dtype not in DTYPE_STR_KINDS:
                block = block.astype(DTYPE_STR)

            # resultant array is immutable
            array = array_from_element_apply(
                    array=block,
                    func=func,
                    dtype=dtype,
                    )
            yield array


    #---------------------------------------------------------------------------
    def search(self, pos=0, endpos=None):
        if endpos:
            func = lambda s: self._pattern.search(
                    s, pos=pos, endpos=endpos) is not None
        else:
            func = lambda s: self._pattern.search(
                    s, pos=pos) is not None

        block_gen = self._process_blocks(
                blocks=self._blocks,
                func=func,
                dtype=DTYPE_BOOL,
                )
        return self._blocks_to_container(block_gen)