import re
import typing as tp

import hashlib

import numpy as np

from static_frame.core.node_selector import Interface
# from static_frame.core.node_selector import InterfaceBatch
from static_frame.core.node_selector import TContainer
# from static_frame.core.util import DTYPE_BOOL
# from static_frame.core.util import DTYPE_OBJECT
# from static_frame.core.util import DTYPE_STR
# from static_frame.core.util import DTYPE_STR_KINDS
# from static_frame.core.util import AnyCallable
# from static_frame.core.util import array_from_element_apply

if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame  # pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index import Index  # pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchy  # pylint: disable = W0611 #pragma: no cover
    from static_frame.core.series import Series  # pylint: disable = W0611 #pragma: no cover


# BlocksType = tp.Iterable[np.ndarray]
ToContainerType = tp.Callable[[tp.Iterator[np.ndarray]], TContainer]

class InterfaceHashlib(Interface[TContainer]):

    __slots__ = (
            '_to_bytes',
            '_include_name',
            '_include_class',
            '_encoding',
            )

    INTERFACE = (
            'sha256',
            )

    def __init__(self,
            to_bytes: tp.Callable[[], bytes],
            include_name: bool,
            include_class: bool,
            encoding: str,
            ) -> None:
        self._to_bytes = to_bytes
        self._include_name = include_name
        self._include_class = include_class
        self._encoding = encoding


    @property
    def bytes(self) -> bytes:
        return self._to_bytes(
                include_name=self._include_name,
                include_class=self._include_class,
                encoding=self._encoding,
                )

    def sha256(self) -> bytes:
        return hashlib.sha256(self.bytes)







