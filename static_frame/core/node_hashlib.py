import hashlib
import re
import typing as tp

import numpy as np

# from static_frame.core.node_selector import InterfaceBatch
from static_frame.core.node_selector import Interface
from static_frame.core.node_selector import TContainer

# from static_frame.core.util import DTYPE_BOOL
# from static_frame.core.util import DTYPE_OBJECT
# from static_frame.core.util import DTYPE_STR
# from static_frame.core.util import DTYPE_STR_KINDS
# from static_frame.core.util import AnyCallable
# from static_frame.core.util import array_from_element_apply

if tp.TYPE_CHECKING:
    from hashlib import _Hash

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
            'md5',
            'sha256',
            'sha512',
            'sha3_256',
            'sha3_512',
            'shake_128',
            'shake_256',
            'blake2b',
            'blake2s',
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

    def md5(self) -> '_Hash':
        return hashlib.md5(self.bytes)

    def sha256(self) -> '_Hash':
        return hashlib.sha256(self.bytes)

    def sha512(self) -> '_Hash':
        return hashlib.sha512(self.bytes)

    def sha3_256(self) -> '_Hash':
        return hashlib.sha3_256(self.bytes)

    def sha3_512(self) -> '_Hash':
        return hashlib.sha3_512(self.bytes)

    def shake_128(self) -> '_Hash':
        return hashlib.shake_128(self.bytes)

    def shake_256(self) -> '_Hash':
        return hashlib.shake_256(self.bytes)

    def blake2b(self, *,
            digest_size: int = 64,
            key: bytes = b'',
            salt: bytes = b'',
            person: bytes = b'',
            fanout: int = 1,
            depth: int = 1,
            leaf_size: int = 0,
            node_offset: int = 0,
            node_depth: int = 0,
            inner_size: int = 0,
            last_node: bool = False,
            # usedforsecurity: bool = True, # py 3.9
            ) -> '_Hash':
        return hashlib.blake2b(
                self.bytes,
                digest_size=digest_size,
                key=key,
                salt=salt,
                person=person,
                fanout=fanout,
                depth=depth,
                leaf_size=leaf_size,
                node_offset=node_offset,
                node_depth=node_depth,
                inner_size=inner_size,
                last_node=last_node,
                # usedforsecurity=usedforsecurity,
                )

    def blake2s(self, *,
            digest_size: int = 32,
            key: bytes = b'',
            salt: bytes = b'',
            person: bytes = b'',
            fanout: int = 1,
            depth: int = 1,
            leaf_size: int = 0,
            node_offset: int = 0,
            node_depth: int = 0,
            inner_size: int = 0,
            last_node: bool = False,
            # usedforsecurity: bool = True,
            ) -> '_Hash':
        return hashlib.blake2s(
                self.bytes,
                digest_size=digest_size,
                key=key,
                salt=salt,
                person=person,
                fanout=fanout,
                depth=depth,
                leaf_size=leaf_size,
                node_offset=node_offset,
                node_depth=node_depth,
                inner_size=inner_size,
                last_node=last_node,
                # usedforsecurity=usedforsecurity,
                )


