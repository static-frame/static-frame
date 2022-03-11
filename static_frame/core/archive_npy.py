from zipfile import ZipFile
from zipfile import ZIP_STORED
import json
import struct
from ast import literal_eval
import os
import mmap
import typing as tp
from types import TracebackType
from io import UnsupportedOperation

import numpy as np

from static_frame.core.interface_meta import InterfaceMeta

from static_frame.core.util import PathSpecifier
from static_frame.core.util import DTYPE_OBJECT_KIND
from static_frame.core.util import list_to_tuple
from static_frame.core.util import EMPTY_TUPLE
from static_frame.core.util import NameType
from static_frame.core.util import IndexInitializer
from static_frame.core.util import concat_resolved

from static_frame.core.container_util import index_many_concat
from static_frame.core.container_util import index_many_set
from static_frame.core.container_util import ContainerMap

from static_frame.core.index_base import IndexBase
from static_frame.core.index import Index
from static_frame.core.index_datetime import dtype_to_index_cls

from static_frame.core.exception import ErrorNPYDecode
from static_frame.core.exception import ErrorNPYEncode
from static_frame.core.exception import AxisInvalid
from static_frame.core.exception import ErrorInitIndexNonUnique


if tp.TYPE_CHECKING:
    import pandas as pd #pylint: disable=W0611 #pragma: no cover
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
    STRUCT_FMT = '<H' # version 1.0, unsigned short
    STRUCT_FMT_SIZE = struct.calcsize(STRUCT_FMT) # 2 bytes
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
        '''Write an NPY 1.0 file to the open, writeable, binary file given by ``file``. NPY 1.0 is used as structured arrays are not supported.
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
        # unpack tuple of one element
        length_header, = struct.unpack(cls.STRUCT_FMT, length_size)
        header = file.read(length_header)
        if header not in header_decode_cache:
            # eval dict and strip values, relying on order
            dtype_str, fortran_order, shape = literal_eval(
                    header.decode(cls.ENCODING)
                    ).values()
            header_decode_cache[header] = np.dtype(dtype_str), fortran_order, shape
        return header_decode_cache[header]

    @classmethod
    def header_from_npy(cls,
            file: tp.IO[bytes],
            header_decode_cache: HeaderDecodeCacheType,
            ) -> HeaderType:
        '''Utility method to just read the header.
        '''
        if cls.MAGIC_PREFIX != file.read(cls.MAGIC_LEN):
            raise ErrorNPYDecode('Invalid NPY header found.') # COV_MISSING
        return cls._header_decode(file, header_decode_cache)

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
    _archive: tp.Union[ZipFile, PathSpecifier]


    def __init__(self,
            fp: PathSpecifier,
            writeable: bool,
            memory_map: bool,
            ):
        raise NotImplementedError() #pragma: no cover

    def __del__(self) -> None:
        pass

    def write_array(self, name: str, array: np.ndarray) -> None:
        raise NotImplementedError() #pragma: no cover

    def read_array(self, name: str) -> np.ndarray:
        raise NotImplementedError() #pragma: no cover

    def read_array_header(self, name: str) -> HeaderType:
        raise NotImplementedError() #pragma: no cover

    def size_array(self, name: str) -> int:
        raise NotImplementedError() #pragma: no cover

    def write_metadata(self, content: tp.Any) -> None:
        raise NotImplementedError() #pragma: no cover

    def read_metadata(self) -> tp.Any:
        raise NotImplementedError() #pragma: no cover

    def size_metadata(self) -> int:
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

    _archive: ZipFile

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

    def read_array_header(self, name: str) -> HeaderType:
        '''Alternate reader for status displays.
        '''
        f = self._archive.open(name)
        try:
            header = NPYConverter.header_from_npy(f, self._header_decode_cache)
        finally:
            f.close()
        return header

    def size_array(self, name: str) -> int:
        return self._archive.getinfo(name).file_size

    def write_metadata(self, content: tp.Any) -> None:
        self._archive.writestr(self.FILE_META, json.dumps(content))

    def read_metadata(self) -> tp.Any:
        return json.loads(self._archive.read(self.FILE_META))

    def size_metadata(self) -> int:
        return self._archive.getinfo(self.FILE_META).file_size

class ArchiveDirectory(Archive):
    __slots__ = (
            'labels',
            'memory_map',
            '_archive',
            '_closable',
            '_header_decode_cache',
            )

    _archive: PathSpecifier

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

    def read_array_header(self, name: str) -> HeaderType:
        '''Alternate reader for status displays.
        '''
        fp = os.path.join(self._archive, name)
        f = open(fp, 'rb')
        try:
            header = NPYConverter.header_from_npy(f, self._header_decode_cache)
        finally:
            f.close()
        return header

    def size_array(self, name: str) -> int:
        fp = os.path.join(self._archive, name)
        return os.path.getsize(fp)

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

    def size_metadata(self) -> int:
        fp = os.path.join(self._archive, self.FILE_META)
        return os.path.getsize(fp)

#-------------------------------------------------------------------------------
class Label:
    KEY_NAMES = '__names__'
    KEY_TYPES = '__types__'
    KEY_DEPTHS = '__depths__'
    KEY_TYPES_INDEX = '__types_index__'
    KEY_TYPES_COLUMNS = '__types_columns__'
    FILE_TEMPLATE_VALUES_INDEX = '__values_index_{}__.npy'
    FILE_TEMPLATE_VALUES_COLUMNS = '__values_columns_{}__.npy'
    FILE_TEMPLATE_BLOCKS = '__blocks_{}__.npy'


class ArchiveIndexConverter:
    '''Utility methods for converting Index or index components.
    '''

    @staticmethod
    def index_encode(
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

    @staticmethod
    def array_encode(
            *,
            metadata: tp.Dict[str, tp.Hashable],
            archive: Archive,
            array: tp.Union[np.ndarray, tp.Iterable[tp.Hashable]],
            key_template_values: str,
            ) -> None:
        '''
        Args:
            metadata: mutates in place with json components
        '''
        assert array.ndim == 1 # type: ignore
        archive.write_array(key_template_values.format(0), array)

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


class ArchiveFrameConverter:
    _ARCHIVE_CLS: tp.Type[Archive]

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
        metadata[Label.KEY_NAMES] = [frame._name,
                frame._index._name,
                frame._columns._name,
                ]
        # do not store Frame class as caller will determine
        metadata[Label.KEY_TYPES] = [
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

        archive = cls._ARCHIVE_CLS(fp,
                writeable=True,
                memory_map=False,
                )

        ArchiveIndexConverter.index_encode(
                metadata=metadata,
                archive=archive,
                index=frame._index,
                key_template_values=Label.FILE_TEMPLATE_VALUES_INDEX,
                key_types=Label.KEY_TYPES_INDEX,
                depth=depth_index,
                include=include_index,
                )

        ArchiveIndexConverter.index_encode(
                metadata=metadata,
                archive=archive,
                index=frame._columns,
                key_template_values=Label.FILE_TEMPLATE_VALUES_COLUMNS,
                key_types=Label.KEY_TYPES_COLUMNS,
                depth=depth_columns,
                include=include_columns,
                )

        for i, array in enumerate(block_iter):
            archive.write_array(Label.FILE_TEMPLATE_BLOCKS.format(i), array)

        metadata[Label.KEY_DEPTHS] = [
                i + 1, # block count
                depth_index,
                depth_columns]

        archive.write_metadata(metadata)

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

        archive = cls._ARCHIVE_CLS(fp,
                writeable=False,
                memory_map=memory_map,
                )
        metadata = archive.read_metadata()

        # JSON will bring back tuple `name` attributes as lists; these must be converted to tuples to be hashable. Alternatives (like storing repr and using literal_eval) are slower than JSON.
        name, name_index, name_columns = (list_to_tuple(n)
                for n in metadata[Label.KEY_NAMES])

        block_count, depth_index, depth_columns = metadata[Label.KEY_DEPTHS]
        cls_index, cls_columns = (ContainerMap.str_to_cls(name)
                for name in metadata[Label.KEY_TYPES])

        index = ArchiveIndexConverter._index_decode(
                archive=archive,
                metadata=metadata,
                key_template_values=Label.FILE_TEMPLATE_VALUES_INDEX,
                key_types=Label.KEY_TYPES_INDEX,
                depth=depth_index,
                cls_index=cls_index,
                name=name_index,
                )

        columns = ArchiveIndexConverter._index_decode(
                archive=archive,
                metadata=metadata,
                key_template_values=Label.FILE_TEMPLATE_VALUES_COLUMNS,
                key_types=Label.KEY_TYPES_COLUMNS,
                depth=depth_columns,
                cls_index=cls_columns,
                name=name_columns,
                )

        tb = TypeBlocks.from_blocks(
                archive.read_array(Label.FILE_TEMPLATE_BLOCKS.format(i))
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
    _ARCHIVE_CLS = ArchiveZip

class NPYFrameConverter(ArchiveFrameConverter):
    _ARCHIVE_CLS = ArchiveDirectory



#-------------------------------------------------------------------------------
# for converting from components, unstructured Frames

class ArchiveComponentsConverter(metaclass=InterfaceMeta):
    '''
    A family of methods to write NPY/NPZ from things other than a Frame, or multi-frame collections like a Bus/Yarn/Quilt but with the intention of production a consolidate Frame, not just a zip of Frames.
    '''
    _ARCHIVE_CLS: tp.Type[Archive]

    __slots__ = (
            '_archive',
            '_writeable',
            )

    def __init__(self, fp: PathSpecifier, mode: str = 'r') -> None:
        if mode == 'w':
            writeable = True
        elif mode == 'r':
            writeable = False
        else:
            raise RuntimeError('Invalid value for mode; use "w" or "r"')

        self._writeable = writeable # not explicitly stored in Archive instance
        self._archive = self._ARCHIVE_CLS(fp,
                writeable=self._writeable,
                memory_map=False,
                )

    def __enter__(self) -> 'ArchiveComponentsConverter':
        '''When entering a context manager, a handle to this instance is returned.
        '''
        return self

    def __exit__(self,
            type: tp.Type[BaseException],
            value: BaseException,
            traceback: TracebackType,
            ) -> None:
        '''When exiting a context manager, resources are closed as necessary.
        '''
        self._archive.close()
        self._archive.__del__() # force closing resources

    @property
    def contents(self) -> 'Frame':
        '''
        Return a :obj:`Frame` indicating name, dtype, shape, and bytes, of Archive components.
        '''
        if self._writeable:
            raise UnsupportedOperation('Open with mode "r" to get contents.')

        from static_frame.core.frame import Frame
        def gen() -> tp.Iterator[tp.Tuple[tp.Any, ...]]:
            # metadata is in labels; sort by ext,ension first to put at top
            for name in sorted(
                    self._archive.labels,
                    key=lambda fn: tuple(reversed(fn.split('.')))
                    ):
                if name == self._archive.FILE_META:
                    yield (name, self._archive.size_metadata()) + ('', '', '')
                else:
                    header = self._archive.read_array_header(name)
                    yield (name, self._archive.size_array(name)) + header

        f = Frame.from_records(gen(),
                columns=('name', 'size', 'dtype', 'fortran', 'shape'),
                name=str(self._archive._archive),
                )
        return f.set_index('name', drop=True) #type: ignore

    @property
    def nbytes(self) -> int:
        '''
        Return numer of bytes stored in this archive.
        '''
        if self._writeable:
            raise UnsupportedOperation('Open with mode "r" to get nbytes.')

        def gen() -> tp.Iterator[int]:
            # metadata is in labels; sort by ext,ension first to put at top
            for name in self._archive.labels:
                if name == self._archive.FILE_META:
                    yield self._archive.size_metadata()
                else:
                    yield self._archive.size_array(name)
        return sum(gen())

    def from_arrays(self,
            blocks: tp.Iterable[np.ndarray],
            *,
            index: tp.Optional[IndexInitializer] = None,
            columns: tp.Optional[IndexInitializer] = None,
            name: NameType = None,
            axis: int = 0,
            ) -> None:
        '''
        Given an iterable of arrays, write out an NPZ or NPY directly, without building up intermediary :obj:`Frame`. If axis 0, the arrays are vertically stacked; if axis 1, they are horizontally stacked. For both axis, if included, indices must be of appropriate length.

        Args:
            blocks:
            *,
            index: An array, :obj:`Index`, or :obj:`IndexHierarchy`.
            columns: An array, :obj:`Index`, or :obj:`IndexHierarchy`.
            name:
            axis:
        '''
        if not self._writeable:
            raise UnsupportedOperation('Open with mode "w" to write.')

        metadata: tp.Dict[str, tp.Any] = {}

        if isinstance(index, IndexBase):
            depth_index = index.depth
            name_index = index.name
            cls_index = index.__class__
            ArchiveIndexConverter.index_encode(
                    metadata=metadata,
                    archive=self._archive,
                    index=index,
                    key_template_values=Label.FILE_TEMPLATE_VALUES_INDEX,
                    key_types=Label.KEY_TYPES_INDEX,
                    depth=depth_index,
                    include=True,
                    )
        elif index is not None:
            if index.__class__ is not np.ndarray:
                raise RuntimeError('index argument must be an Index, IndexHierarchy, or 1D np.ndarray')

            depth_index = 1
            name_index = None
            cls_index = dtype_to_index_cls(True, index.dtype) #type: ignore
            ArchiveIndexConverter.array_encode(
                    metadata=metadata,
                    archive=self._archive,
                    array=index,
                    key_template_values=Label.FILE_TEMPLATE_VALUES_INDEX,
                    )
        else:
            depth_index = 1
            name_index = None
            cls_index = Index

        if isinstance(columns, IndexBase):
            depth_columns = columns.depth
            name_columns = columns.name
            cls_columns = columns.__class__
            ArchiveIndexConverter.index_encode(
                    metadata=metadata,
                    archive=self._archive,
                    index=columns,
                    key_template_values=Label.FILE_TEMPLATE_VALUES_COLUMNS,
                    key_types=Label.KEY_TYPES_COLUMNS,
                    depth=depth_columns,
                    include=True,
                    )
        elif columns is not None:
            if columns.__class__ is not np.ndarray:
                raise RuntimeError('index argument must be an Index, IndexHierarchy, or 1D np.ndarray')

            depth_columns = 1 # only support 1D
            name_columns = None
            cls_columns = dtype_to_index_cls(True, columns.dtype) #type: ignore
            ArchiveIndexConverter.array_encode(
                    metadata=metadata,
                    archive=self._archive,
                    array=columns,
                    key_template_values=Label.FILE_TEMPLATE_VALUES_COLUMNS,
                    )
        else:
            depth_columns = 1 # only support 1D
            name_columns = None
            cls_columns = Index

        metadata[Label.KEY_NAMES] = [name,
                name_index,
                name_columns,
                ]
        # do not store Frame class as caller will determine
        metadata[Label.KEY_TYPES] = [
                cls_index.__name__,
                cls_columns.__name__,
                ]

        if axis == 1:
            rows = 0
            for i, array in enumerate(blocks):
                if not rows:
                    rows = array.shape[0]
                else:
                    if array.shape[0] != rows:
                        raise RuntimeError('incompatible block shapes')
                self._archive.write_array(Label.FILE_TEMPLATE_BLOCKS.format(i), array)
        elif axis == 0:
            # for now, just vertically concat and write, though this has a 2X memory requirement
            resolved = concat_resolved(blocks, axis=0)
            # if this results in an obect array, an exception will be raised
            self._archive.write_array(Label.FILE_TEMPLATE_BLOCKS.format(0), resolved)
            i = 0
        else:
            raise AxisInvalid(f'invalid axis {axis}')

        metadata[Label.KEY_DEPTHS] = [
                i + 1, # block count
                depth_index,
                depth_columns]
        self._archive.write_metadata(metadata)

    def from_frames(self,
            frames: tp.Iterable['Frame'],
            *,
            include_index: bool = True,
            include_columns: bool = True,
            axis: int = 0,
            union: bool = True,
            name: NameType = None,
            fill_value: object = np.nan,
            ) -> None:
        '''Given an iterable of Frames, write out an NPZ or NPY directly, without building up an intermediary Frame. If axis 0, the Frames must be block compatible; if axis 1, the Frames must have the same number of rows. For both axis, if included, concatenated indices must be unique or aligned.

        Args:
            frames:
            *
            include_index:
            include_columns:
            axis:
            union:
            name:
            fill_value:

        '''
        if not self._writeable:
            raise UnsupportedOperation('Open with mode "w" to write.')

        from static_frame.core.type_blocks import TypeBlocks
        from static_frame.core.frame import Frame

        frames = [f if isinstance(f, Frame) else f.to_frame(axis) for f in frames] # type: ignore

        # NOTE: based on Frame.from_concat
        if axis == 1: # stacks columns (extends rows horizontally)
            if include_columns:
                try:
                    columns = index_many_concat(
                            (f._columns for f in frames),
                            Index,
                            )
                except ErrorInitIndexNonUnique:
                    raise RuntimeError('Column names after horizontal concatenation are not unique; set include_columns to None to ignore.')
            else:
                columns = None

            if include_index:
                index = index_many_set(
                        (f._index for f in frames),
                        Index,
                        union=union,
                        )
            else:
                raise RuntimeError('Must include index for horizontal alignment.')

            def blocks() -> tp.Iterator[np.ndarray]:
                for f in frames:
                    if len(f.index) != len(index) or (f.index != index).any():
                        f = f.reindex(index=index, fill_value=fill_value)
                    for block in f._blocks._blocks:
                        yield block

        elif axis == 0: # stacks rows (extends columns vertically)
            if include_index:
                try:
                    index = index_many_concat((f._index for f in frames), Index)
                except ErrorInitIndexNonUnique:
                    raise RuntimeError('Index names after vertical concatenation are not unique; set include_index to None to ignore')
            else:
                index = None

            if include_columns:
                columns = index_many_set(
                        (f._columns for f in frames),
                        Index,
                        union=union,
                        )
            else:
                raise RuntimeError('Must include columns for vertical alignment.')

            def blocks() -> tp.Iterator[np.ndarray]:
                type_blocks = []
                previous_f: tp.Optional[Frame] = None
                block_compatible = True
                reblock_compatible = True

                for f in frames:
                    if len(f.columns) != len(columns) or (f.columns != columns).any():
                        f = f.reindex(columns=columns, fill_value=fill_value)

                    type_blocks.append(f._blocks)
                    # column size is all the same by this point
                    if previous_f is not None: # after the first
                        if block_compatible:
                            block_compatible &= f._blocks.block_compatible(
                                    previous_f._blocks,
                                    axis=1) # only compare columns
                        if reblock_compatible:
                            reblock_compatible &= f._blocks.reblock_compatible(
                                    previous_f._blocks)
                    previous_f = f

                yield from TypeBlocks.vstack_blocks_to_blocks(
                        type_blocks=type_blocks,
                        block_compatible=block_compatible,
                        reblock_compatible=reblock_compatible,
                        )
        else:
            raise AxisInvalid(f'no support for {axis}')

        self.from_arrays(
                blocks=blocks(),
                index=index,
                columns=columns,
                name=name,
                axis=1, # blocks are normalized for horizontal concat
                )

class NPZ(ArchiveComponentsConverter):
    '''Utility object for reading characteristics from, or writing new, NPZ files from arrays or :obj:`Frame`.
    '''
    _ARCHIVE_CLS = ArchiveZip

    # def from_npy(self, fp: PathSpecifier) -> None: # writes an NPZ from an NPY
    #     pass

class NPY(ArchiveComponentsConverter):
    '''Utility object for reading characteristics from, or writing new, NPY directories from arrays or :obj:`Frame`.
    '''
    _ARCHIVE_CLS = ArchiveDirectory

    # def from_npz(self, fp: PathSpecifier) -> None: # writes an NPZ from an NPY
    #     pass





