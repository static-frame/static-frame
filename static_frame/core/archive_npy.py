from zipfile import ZipFile
from zipfile import ZIP_STORED
import json
import struct
from ast import literal_eval
import os
import mmap
import typing as tp

import numpy as np

from static_frame.core.util import PathSpecifier
from static_frame.core.util import DTYPE_OBJECT_KIND
from static_frame.core.util import list_to_tuple
from static_frame.core.util import EMPTY_TUPLE
from static_frame.core.util import NameType
from static_frame.core.container_util import ContainerMap

from static_frame.core.exception import ErrorNPYDecode
from static_frame.core.exception import ErrorNPYEncode

if tp.TYPE_CHECKING:
    import pandas as pd #pylint: disable=W0611 #pragma: no cover
    from static_frame.core.index_base import IndexBase #pylint: disable=W0611,C0412 #pragma: no cover
    from static_frame.core.frame import Frame #pylint: disable=W0611,C0412 #pragma: no cover


HeaderType = tp.Tuple[np.dtype, bool, tp.Tuple[int, ...]]
HeaderDecodeCacheType = tp.Dict[bytes, HeaderType]

#-------------------------------------------------------------------------------

class NPYConverter:
    '''Optimized implementation based on numpy/lib/format.py
    '''
    MAGIC_PREFIX = b'\x93NUMPY' + bytes((1, 0)) # version 1.0
    MAGIC_LEN = len(MAGIC_PREFIX)
    ARRAY_ALIGN = 64
    STRUCT_FMT = '<H' # version 1.0
    STRUCT_FMT_SIZE = struct.calcsize(STRUCT_FMT)
    ENCODING = 'latin1' # version 1.0
    BUFFERSIZE_NUMERATOR = 16 * 1024 ** 2 # ~16 MB
    NDITER_FLAGS = ('external_loop', 'buffered', 'zerosize_ok')

    @classmethod
    def _header_encode(cls, header: str) -> bytes:
        '''
        Takes a string header, and attaches the prefix and padding to it.
        This is hard-coded to only use Version 3.0
        '''
        center = header.encode(cls.ENCODING)
        hlen = len(center) + 1

        padlen = cls.ARRAY_ALIGN - (
               (cls.MAGIC_LEN + cls.STRUCT_FMT_SIZE + hlen) % cls.ARRAY_ALIGN
               )
        prefix = cls.MAGIC_PREFIX + struct.pack(cls.STRUCT_FMT, hlen + padlen)
        postfix = b' ' * padlen + b'\n'
        return prefix + center + postfix

    @classmethod
    def to_npy(cls, file: tp.IO[bytes], array: np.ndarray) -> None:
        '''Write an NPY 3.0 file to the open, writeable, binary file given by ``file``.
        '''
        dtype = array.dtype
        if dtype.kind == DTYPE_OBJECT_KIND:
            raise ErrorNPYEncode('no support for object dtypes')
        if dtype.names is not None:
            raise ErrorNPYEncode('no support for structured arrays')
        if array.ndim == 0 or array.ndim > 2:
            raise ErrorNPYEncode('no support for ndim == 0 or greater than two.')

        flags = array.flags
        fortran_order = flags.f_contiguous

        # NOTE: do not know how to create array with itmesize 0, assume array.itemsize > 0
        # NOTE: derived numpy configuration
        buffersize = max(cls.BUFFERSIZE_NUMERATOR // array.itemsize, 1)

        header = f'{{"descr":"{dtype.str}","fortran_order":{fortran_order},"shape":{array.shape}}}'
        file.write(cls._header_encode(header))

        # NOTE: this works but forces copying everything in memory

        # if flags.f_contiguous and not flags.c_contiguous:
        #     file.write(array.T.tobytes())
        # else:
        #     file.write(array.tobytes())

        # NOTE: this approach works with normal open files (but not zip files) and is not materially faster than using buffering
        # if isinstance(file, io.BufferedWriter):
        #     if fortran_order and not flags.c_contiguous:
        #         array.T.tofile(file)
        #     else:
        #         array.tofile(file)

        # NOTE: this might be made more efficient by creating an ArrayKit function that extracts bytes directly, avoiding creating an array for each chunk.
        if fortran_order and not flags.c_contiguous:
            for chunk in np.nditer(
                    array,
                    flags=cls.NDITER_FLAGS,
                    buffersize=buffersize,
                    order='F',
                    ):
                file.write(chunk.tobytes('C'))
        else:
            for chunk in np.nditer(
                    array,
                    flags=cls.NDITER_FLAGS,
                    buffersize=buffersize,
                    order='C',
                    ):
                file.write(chunk.tobytes('C'))

    @classmethod
    def _header_decode(cls,
            file: tp.IO[bytes],
            header_decode_cache: HeaderDecodeCacheType,
            ) -> HeaderType:
        '''Extract and decode the header.
        '''
        length_size = file.read(cls.STRUCT_FMT_SIZE)
        length_header = struct.unpack(cls.STRUCT_FMT, length_size)[0]
        header = file.read(length_header)
        if header not in header_decode_cache:
            dtype_str, fortran_order, shape = literal_eval(
                    header.decode(cls.ENCODING)
                    ).values()
            header_decode_cache[header] = np.dtype(dtype_str), fortran_order, shape
        return header_decode_cache[header]

    @classmethod
    def from_npy(cls,
            file: tp.IO[bytes],
            header_decode_cache: HeaderDecodeCacheType,
            memory_map: bool = False,
            ) -> tp.Tuple[np.ndarray, tp.Optional[mmap.mmap]]:
        '''Read an NPY 1.0 file.
        '''
        if cls.MAGIC_PREFIX != file.read(cls.MAGIC_LEN):
            raise ErrorNPYDecode('Invalid NPY header found.')

        dtype, fortran_order, shape = cls._header_decode(file, header_decode_cache)
        if dtype.kind == DTYPE_OBJECT_KIND:
            raise ErrorNPYDecode('no support for object dtypes')

        ndim = len(shape)
        if ndim == 1:
            size = shape[0]
        elif ndim == 2:
            size = shape[0] * shape[1]
        else:
            raise ErrorNPYDecode(f'No support for {ndim}-dimensional arrays')

        if memory_map:
            # offset calculations derived from numpy/core/memmap.py
            offset_header = file.tell()
            byte_count = offset_header + size * dtype.itemsize
            # ALLOCATIONGRANULARITY is 4096 on linux, if offset_header is 64 (or less than 4096), this will set offset_mmap to zero
            offset_mmap = offset_header - offset_header % mmap.ALLOCATIONGRANULARITY
            byte_count -= offset_mmap
            offset_array = offset_header - offset_mmap
            mm = mmap.mmap(file.fileno(),
                    byte_count,
                    access=mmap.ACCESS_READ,
                    offset=offset_mmap,
                    )
            # will always be immutable
            array = np.ndarray(shape,
                    dtype=dtype,
                    buffer=mm,
                    offset=offset_array,
                    order='F' if fortran_order else 'C',
                    )
            # assert not array.flags.writeable
            return array, mm

        # NOTE: we cannot use np.from_file, as the file object from a Zip is not a normal file
        # NOTE: np.frombuffer produces a read-only view on the existing data
        # array = np.frombuffer(file.read(size * dtype.itemsize), dtype=dtype)
        array = np.frombuffer(file.read(), dtype=dtype)

        if fortran_order and ndim == 2:
            array.shape = (shape[1], shape[0])
            array = array.transpose()
        else:
            array.shape = shape
        # assert not array.flags.writeable
        return array, None

#-------------------------------------------------------------------------------
class Archive:
    '''Abstraction of a read/write archive, such as a directory or a zip archive. Holds state over the life of writing / reading a Frame.
    '''
    FILE_META = '__meta__.json'

    __slots__ = (
            'labels',
            'memory_map',
            '_archive',
            '_closable',
            '_header_decode_cache',
            )

    labels: tp.FrozenSet[str]
    memory_map: bool
    _header_decode_cache: HeaderDecodeCacheType

    def __init__(self,
            fp: PathSpecifier,
            writeable: bool,
            memory_map: bool,
            ):
        raise NotImplementedError() #pragma: no cover

    def write_array(self, name: str, array: np.ndarray) -> None:
        raise NotImplementedError() #pragma: no cover

    def read_array(self, name: str) -> np.ndarray:
        raise NotImplementedError() #pragma: no cover

    def write_metadata(self, content: tp.Any) -> None:
        raise NotImplementedError() #pragma: no cover

    def read_metadata(self) -> tp.Any:
        raise NotImplementedError() #pragma: no cover

    def close(self) -> None:
        for f in getattr(self, '_closable', EMPTY_TUPLE):
            f.close()

class ArchiveZip(Archive):
    __slots__ = (
            'labels',
            'memory_map',
            '_archive',
            '_closable',
            '_header_decode_cache',
            )

    def __init__(self,
            fp: PathSpecifier,
            writeable: bool,
            memory_map: bool,
            ):

        mode = 'w' if writeable else 'r'
        self._archive = ZipFile(fp,
                mode=mode,
                compression=ZIP_STORED,
                allowZip64=True,
                )
        if not writeable:
            self.labels = frozenset(self._archive.namelist())
            self._header_decode_cache = {}
        if memory_map:
            raise RuntimeError(f'Cannot memory_map with {self}')

        self.memory_map = memory_map

    def __del__(self) -> None:
        self._archive.close()

    def write_array(self, name: str, array: np.ndarray) -> None:
        # NOTE: zip only has 'w' mode, not 'wb'
        # NOTE: force_zip64 required for large files
        f = self._archive.open(name, 'w', force_zip64=True)
        try:
            NPYConverter.to_npy(f, array)
        finally:
            f.close()

    def read_array(self, name: str) -> np.ndarray:
        f = self._archive.open(name)
        try:
            array, _ = NPYConverter.from_npy(f, self._header_decode_cache)
        finally:
            f.close()
        array.flags.writeable = False
        return array

    def write_metadata(self, content: tp.Any) -> None:
        self._archive.writestr(self.FILE_META, json.dumps(content))

    def read_metadata(self) -> tp.Any:
        return json.loads(self._archive.read(self.FILE_META))

class ArchiveDirectory(Archive):
    __slots__ = (
            'labels',
            'memory_map',
            '_archive',
            '_closable',
            '_header_decode_cache',
            )

    def __init__(self,
            fp: PathSpecifier,
            writeable: bool,
            memory_map: bool,
            ):

        self._archive = fp
        if not os.path.exists(self._archive):
            if writeable:
                os.mkdir(fp)
            else:
                raise RuntimeError(f'Atttempting to read from a non-existant directory: {fp}')
        elif not os.path.isdir(self._archive):
            raise RuntimeError(f'A directory must be provided, not {fp}')

        if not writeable:
            self._header_decode_cache = {}
            self.labels = frozenset(f.name for f in os.scandir(self._archive))

        self.memory_map = memory_map

    def write_array(self, name: str, array: np.ndarray) -> None:
        fp = os.path.join(self._archive, name)
        f = open(fp, 'wb')
        try:
            NPYConverter.to_npy(f, array)
        finally:
            f.close()

    def read_array(self, name: str) -> np.ndarray:
        fp = os.path.join(self._archive, name)
        if self.memory_map:
            if not hasattr(self, '_closable'):
                self._closable = []

            f = open(fp, 'rb')
            try:
                array, mm = NPYConverter.from_npy(f,
                        self._header_decode_cache,
                        self.memory_map,
                        )
            finally:
                f.close() # NOTE: can close the file after creating memory map
            # self._closable.append(f)
            self._closable.append(mm)
            return array

        f = open(fp, 'rb')
        try:
            array, _ = NPYConverter.from_npy(f,
                    self._header_decode_cache,
                    self.memory_map,
                    )
        finally:
            f.close()
        return array

    def write_metadata(self, content: tp.Any) -> None:
        fp = os.path.join(self._archive, self.FILE_META)
        f = open(fp, 'w')
        try:
            f.write(json.dumps(content))
        finally:
            f.close()

    def read_metadata(self) -> tp.Any:
        fp = os.path.join(self._archive, self.FILE_META)
        f = open(fp, 'r')
        try:
            post = json.loads(f.read())
        finally:
            f.close()
        return post

#-------------------------------------------------------------------------------
class ArchiveFrameConverter:
    KEY_NAMES = '__names__'
    KEY_TYPES = '__types__'
    KEY_DEPTHS = '__depths__'
    KEY_TYPES_INDEX = '__types_index__'
    KEY_TYPES_COLUMNS = '__types_columns__'
    FILE_TEMPLATE_VALUES_INDEX = '__values_index_{}__.npy'
    FILE_TEMPLATE_VALUES_COLUMNS = '__values_columns_{}__.npy'
    FILE_TEMPLATE_BLOCKS = '__blocks_{}__.npy'

    ARCHIVE_CLS: tp.Type[Archive]

    @staticmethod
    def _index_encode(
            *,
            metadata: tp.Dict[str, tp.Hashable],
            archive: Archive,
            index: 'IndexBase',
            key_template_values: str,
            key_types: str,
            depth: int,
            include: bool,
            ) -> None:
        '''
        Args:
            metadata: mutates in place with json components
        '''
        if depth == 1 and index._map is None: # type: ignore
            pass # do not store anything
        elif include:
            if depth == 1:
                archive.write_array(key_template_values.format(0), index.values)
            else:
                for i in range(depth):
                    archive.write_array(key_template_values.format(i), index.values_at_depth(i))
                metadata[key_types] = [cls.__name__ for cls in index.index_types.values] # type: ignore

    @classmethod
    def to_archive(cls,
            *,
            frame: 'Frame',
            fp: PathSpecifier,
            include_index: bool = True,
            include_columns: bool = True,
            consolidate_blocks: bool = False,
            ) -> None:
        '''
        Write a :obj:`Frame` as an npz file.
        '''
        metadata: tp.Dict[str, tp.Any] = {}
        metadata[cls.KEY_NAMES] = [frame._name,
                frame._index._name,
                frame._columns._name,
                ]
        # do not store Frame class as caller will determine
        metadata[cls.KEY_TYPES] = [
                frame._index.__class__.__name__,
                frame._columns.__class__.__name__,
                ]

        # store shape, index depths
        depth_index = frame._index.depth
        depth_columns = frame._columns.depth

        if consolidate_blocks:
            # NOTE: by taking iter, can avoid 2x memory in some circumstances
            block_iter = frame._blocks._reblock()
        else:
            block_iter = iter(frame._blocks._blocks)

        archive = cls.ARCHIVE_CLS(fp,
                writeable=True,
                memory_map=False,
                )

        cls._index_encode(
                metadata=metadata,
                archive=archive,
                index=frame._index,
                key_template_values=cls.FILE_TEMPLATE_VALUES_INDEX,
                key_types=cls.KEY_TYPES_INDEX,
                depth=depth_index,
                include=include_index,
                )

        cls._index_encode(
                metadata=metadata,
                archive=archive,
                index=frame._columns,
                key_template_values=cls.FILE_TEMPLATE_VALUES_COLUMNS,
                key_types=cls.KEY_TYPES_COLUMNS,
                depth=depth_columns,
                include=include_columns,
                )

        for i, array in enumerate(block_iter):
            archive.write_array(cls.FILE_TEMPLATE_BLOCKS.format(i), array)

        metadata[cls.KEY_DEPTHS] = [
                i + 1, # block count
                depth_index,
                depth_columns]

        archive.write_metadata(metadata)

    @staticmethod
    def _index_decode(*,
            archive: Archive,
            metadata: tp.Dict[str, tp.Any],
            key_template_values: str,
            key_types: str,
            depth: int,
            cls_index: tp.Type['IndexBase'],
            name: NameType,
            ) -> tp.Optional['IndexBase']:
        '''Build index or columns.
        '''
        from static_frame.core.type_blocks import TypeBlocks

        if key_template_values.format(0) not in archive.labels:
            index = None
        elif depth == 1:
            index = cls_index(archive.read_array(key_template_values.format(0)),
                    name=name,
                    )
        else:
            index_tb = TypeBlocks.from_blocks(
                    archive.read_array(key_template_values.format(i))
                    for i in range(depth)
                    )
            index_constructors = [ContainerMap.str_to_cls(name)
                    for name in metadata[key_types]]
            index = cls_index._from_type_blocks(index_tb, # type: ignore
                    name=name,
                    index_constructors=index_constructors,
                    )
        return index

    @classmethod
    def _from_archive(cls,
            *,
            constructor: tp.Type['Frame'],
            fp: PathSpecifier,
            memory_map: bool = False,
            ) -> tp.Tuple['Frame', Archive]:
        '''
        Create a :obj:`Frame` from an npz file.
        '''
        from static_frame.core.type_blocks import TypeBlocks

        archive = cls.ARCHIVE_CLS(fp,
                writeable=False,
                memory_map=memory_map,
                )
        metadata = archive.read_metadata()

        # JSON will bring back tuple `name` attributes as lists; these must be converted to tuples to be hashable. Alternatives (like storing repr and using literal_eval) are slower than JSON.
        name, name_index, name_columns = (list_to_tuple(n) for n in metadata[cls.KEY_NAMES])

        block_count, depth_index, depth_columns = metadata[cls.KEY_DEPTHS]
        cls_index, cls_columns = (ContainerMap.str_to_cls(name)
                for name in metadata[cls.KEY_TYPES])

        index = cls._index_decode(
                archive=archive,
                metadata=metadata,
                key_template_values=cls.FILE_TEMPLATE_VALUES_INDEX,
                key_types=cls.KEY_TYPES_INDEX,
                depth=depth_index,
                cls_index=cls_index,
                name=name_index,
                )

        columns = cls._index_decode(
                archive=archive,
                metadata=metadata,
                key_template_values=cls.FILE_TEMPLATE_VALUES_COLUMNS,
                key_types=cls.KEY_TYPES_COLUMNS,
                depth=depth_columns,
                cls_index=cls_columns,
                name=name_columns,
                )

        tb = TypeBlocks.from_blocks(
                archive.read_array(cls.FILE_TEMPLATE_BLOCKS.format(i))
                for i in range(block_count)
                )

        f = constructor(tb,
                own_data=True,
                index=index,
                own_index = False if index is None else True,
                columns=columns,
                own_columns = False if columns is None else True,
                name=name,
                )

        return f, archive


    @classmethod
    def from_archive(cls,
            *,
            constructor: tp.Type['Frame'],
            fp: PathSpecifier,
            ) -> 'Frame':
        '''
        Create a :obj:`Frame` from an npz file.
        '''
        f, _ = cls._from_archive(constructor=constructor,
                fp=fp,
                memory_map=False,
                )
        return f


    @classmethod
    def from_archive_mmap(cls,
            *,
            constructor: tp.Type['Frame'],
            fp: PathSpecifier,
            ) -> tp.Tuple['Frame', tp.Callable[[], None]]:
        '''
        Create a :obj:`Frame` from an npz file.
        '''
        f, archive = cls._from_archive(constructor=constructor,
                fp=fp,
                memory_map=True,
                )
        return f, archive.close


class NPZFrameConverter(ArchiveFrameConverter):
    ARCHIVE_CLS = ArchiveZip

class NPYFrameConverter(ArchiveFrameConverter):
    ARCHIVE_CLS = ArchiveDirectory




# for converting from components, unstructured Frames


class ArchiveComponentsConverter:
    pass


class NPZComponentsConverter(ArchiveComponentsConverter):
    ARCHIVE_CLS = ArchiveZip

class NPYComponentsConverter(ArchiveComponentsConverter):
    ARCHIVE_CLS = ArchiveDirectory

