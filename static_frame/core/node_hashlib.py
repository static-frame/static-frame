from __future__ import annotations

import hashlib

import typing_extensions as tp

if tp.TYPE_CHECKING:
    from hashlib import _Hash  # pylint: disable = E0611 #pragma: no cover
    from hashlib import _VarLenHash  # pylint: disable = E0611 #pragma: no cover


class InterfaceHashlib:

    __slots__ = (
            '_to_bytes',
            '_include_name',
            '_include_class',
            '_encoding',
            )

    _INTERFACE = (
            'to_bytes',
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
            to_bytes: tp.Callable[[bool, bool, str], bytes],
            *,
            include_name: bool,
            include_class: bool,
            encoding: str,
            ) -> None:
        '''Interfacefor deriving cryptographic hashes from this container, pre-loaded with byte signatures from the calling container.

        Args:
            include_name: Whether container name is included in the bytes signature.
            include_class: Whether class name is included in the byte signature.
            encoding: Encoding to use for converting strings to bytes.
        '''
        self._to_bytes = to_bytes
        self._include_name = include_name
        self._include_class = include_class
        self._encoding = encoding

    def __call__(self,
            include_name: tp.Optional[bool] = None,
            include_class: tp.Optional[bool] = None,
            encoding: tp.Optional[str] = None,
            ) -> 'InterfaceHashlib':
        '''Interfacefor deriving cryptographic hashes from this container, pre-loaded with byte signatures from the calling container.

        Args:
            include_name: Whether container name is included in the bytes signature.
            include_class: Whether class name is included in the byte signature.
            encoding: Encoding to use for converting strings to bytes.
        '''
        return self.__class__(
                to_bytes=self._to_bytes,
                include_name=include_name if include_name is not None else self._include_name,
                include_class=include_class if include_class is not None else self._include_class,
                encoding=encoding if encoding is not None else self._encoding,
                )

    def to_bytes(self) -> bytes:
        '''Return the byte signature for this container, suitable for passing to a cryptographic hash function.
        '''
        return self._to_bytes(
                self._include_name,
                self._include_class,
                self._encoding,
                )

    def md5(self) -> '_Hash':
        return hashlib.md5(self.to_bytes())

    def sha256(self) -> '_Hash':
        return hashlib.sha256(self.to_bytes())

    def sha512(self) -> '_Hash':
        return hashlib.sha512(self.to_bytes())

    def sha3_256(self) -> '_Hash':
        return hashlib.sha3_256(self.to_bytes())

    def sha3_512(self) -> '_Hash':
        return hashlib.sha3_512(self.to_bytes())

    def shake_128(self) -> '_VarLenHash':
        return hashlib.shake_128(self.to_bytes())

    def shake_256(self) -> '_VarLenHash':
        return hashlib.shake_256(self.to_bytes())

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
                self.to_bytes(),
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
                self.to_bytes(),
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


