from __future__ import annotations

import io
import json
import mmap
import os
import shutil
import struct
from ast import literal_eval
from io import UnsupportedOperation
from types import TracebackType
from zipfile import ZIP_STORED
from zipfile import ZipFile

import numpy as np
import typing_extensions as tp

from static_frame.core.archive_zip import ZipFilePartRO
from static_frame.core.archive_zip import ZipFileRO
from static_frame.core.container_util import ContainerMap
from static_frame.core.container_util import index_many_concat
from static_frame.core.container_util import index_many_to_one
from static_frame.core.exception import AxisInvalid
from static_frame.core.exception import ErrorInitIndexNonUnique
from static_frame.core.exception import ErrorNPYDecode
from static_frame.core.exception import ErrorNPYEncode
from static_frame.core.index import Index
from static_frame.core.index_base import IndexBase
from static_frame.core.index_datetime import dtype_to_index_cls
from static_frame.core.interface_meta import InterfaceMeta
from static_frame.core.metadata import NPYLabel
from static_frame.core.util import DTYPE_OBJECT_KIND
from static_frame.core.util import JSONTranslator
from static_frame.core.util import ManyToOneType
from static_frame.core.util import TLabel
from static_frame.core.util import TName
from static_frame.core.util import TPathSpecifier
from static_frame.core.util import TPathSpecifierOrIO
from static_frame.core.util import concat_resolved

if tp.TYPE_CHECKING:
    import pandas as pd  # pylint: disable=W0611 #pragma: no cover

    from static_frame.core.frame import Frame  # pylint: disable=W0611,C0412 #pragma: no cover
    from static_frame.core.generic_aliases import TFrameAny  # pylint: disable=W0611,C0412 #pragma: no cover

    TNDArrayAny = np.ndarray[tp.Any, tp.Any] #pragma: no cover
    TDtypeAny = np.dtype[tp.Any] #pragma: no cover
    HeaderType = tp.Tuple[TDtypeAny, bool, tp.Tuple[int, ...]] #pragma: no cover
    HeaderDecodeCacheType = tp.Dict[bytes, HeaderType] #pragma: no cover

#-------------------------------------------------------------------------------


TNDIterFlags = tp.Sequence[tp.Literal['external_loop', 'buffered', 'zerosize_ok']]

class NPYConverter:
    '''Optimized implementation based on numpy/lib/format.py
    '''
    MAGIC_PREFIX = b'\x93NUMPY' + bytes((1, 0)) # version 1.0
    MAGIC_LEN = len(MAGIC_PREFIX)
    ARRAY_ALIGN = 64
    STRUCT_FMT = '<H' # version 1.0, unsigned short
    STRUCT_FMT_SIZE = struct.calcsize(STRUCT_FMT) # 2 bytes
    MAGIC_AND_STRUCT_FMT_SIZE_LEN = MAGIC_LEN + STRUCT_FMT_SIZE
    ENCODING = 'latin1' # version 1.0
    BUFFERSIZE_NUMERATOR = 16 * 1024 ** 2 # ~16 MB
    NDITER_FLAGS: TNDIterFlags = ('external_loop', 'buffered', 'zerosize_ok')

    @classmethod
    def _header_encode(cls, header: str) -> bytes:
        '''
        Takes a string header, and attaches the prefix and padding to it.
        This is hard-coded to only use Version 1.0
        '''
        center = header.encode(cls.ENCODING)
        hlen = len(center) + 1

        padlen = cls.ARRAY_ALIGN - (
               (cls.MAGIC_AND_STRUCT_FMT_SIZE_LEN + hlen) % cls.ARRAY_ALIGN
               )
        prefix = cls.MAGIC_PREFIX + struct.pack(cls.STRUCT_FMT, hlen + padlen)
        postfix = b' ' * padlen + b'\n'
        return prefix + center + postfix

    @classmethod
    def to_npy(cls, file: tp.IO[bytes], array: TNDArrayAny) -> None:
        '''Write an NPY 1.0 file to the open, writeable, binary file given by ``file``. NPY 1.0 is used as structured arrays are not supported.
        '''
        dtype = array.dtype
        if dtype.kind == DTYPE_OBJECT_KIND:
            preview = repr(array)
            raise ErrorNPYEncode(
                f'No support for object dtypes: {preview[:40]}{"..." if len(preview) > 40 else ""}')
        if dtype.names is not None:
            raise ErrorNPYEncode('No support for structured arrays')
        if array.ndim == 0 or array.ndim > 2:
            raise ErrorNPYEncode('No support for ndim other than 1 and 2.')

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
                file.write(chunk.tobytes('C')) # type: ignore
        else:
            for chunk in np.nditer(
                    array,
                    flags=cls.NDITER_FLAGS,
                    buffersize=buffersize,
                    order='C',
                    ):
                file.write(chunk.tobytes('C')) # type: ignore

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
            ) -> tp.Tuple[TNDArrayAny, tp.Optional[mmap.mmap]]:
        '''Read an NPY 1.0 file.
        '''
        if cls.MAGIC_PREFIX != file.read(cls.MAGIC_LEN):
            raise ErrorNPYDecode('Invalid NPY header found.')

        dtype, fortran_order, shape = cls._header_decode(file, header_decode_cache)
        dtype_kind = dtype.kind
        if dtype_kind == DTYPE_OBJECT_KIND:
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
            array: TNDArrayAny = np.ndarray(shape,
                    dtype=dtype,
                    buffer=mm,
                    offset=offset_array,
                    order='F' if fortran_order else 'C',
                    )
            # assert not array.flags.writeable
            return array, mm

        if dtype_kind == 'M' or dtype_kind == 'm' or file.__class__ is not ZipFilePartRO: # type: ignore
            # NOTE: produces a read-only view on the existing data
            array = np.frombuffer(file.read(), dtype=dtype)
        else:
            # NOTE: using readinto shown to be faster than frombuffer, particularly in the context of tall Frames
            array = np.empty(size, dtype=dtype)
            file.readinto(array.data) # type: ignore
            array.flags.writeable = False

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
            '_memory_map',
            '_archive',
            '_closable',
            '_header_decode_cache',
            )

    _memory_map: bool
    _header_decode_cache: HeaderDecodeCacheType
    _archive: tp.Any # defined below tp.Union[ZipFile, ZipFileRO, TPathSpecifier]

    # set per subclass
    FUNC_REMOVE_FP: tp.Callable[[TPathSpecifier], None]

    def __init__(self,
            fp: TPathSpecifierOrIO,
            writeable: bool,
            memory_map: bool,
            ):
        raise NotImplementedError() #pragma: no cover

    def __del__(self) -> None:
        pass

    def __contains__(self, name: str) -> bool:
        raise NotImplementedError() # pragma: no cover

    def labels(self) -> tp.Iterator[str]:
        raise NotImplementedError() # pragma: no cover

    def write_array(self, name: str, array: TNDArrayAny) -> None:
        raise NotImplementedError() #pragma: no cover

    def read_array(self, name: str) -> TNDArrayAny:
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
        for f in getattr(self, '_closable', ()):
            f.close()

class ArchiveZip(Archive):

    '''Archives based on a new ZipFile per Frame; ZipFile creation happens on __init__.
    '''
    __slots__ = ()

    _archive: tp.Union[ZipFile, ZipFileRO]

    FUNC_REMOVE_FP = os.remove

    def __init__(self,
            fp: TPathSpecifier, # might be a BytesIO object
            writeable: bool,
            memory_map: bool,
            ):

        if writeable:
            self._archive = ZipFile(fp, # pylint: disable=R1732
                mode='w',
                compression=ZIP_STORED,
                allowZip64=True,
                )
        else:
            self._archive = ZipFileRO(fp)
            self._header_decode_cache = {}

        if memory_map:
            raise RuntimeError(f'Cannot memory_map with {self}')

        self._memory_map = memory_map

    def __del__(self) -> None:
        # Note: If the fp we were given didn't exist, _archive also doesn't exist.
        archive = getattr(self, '_archive', None)
        if archive:
            archive.close()

    def __contains__(self, name: str) -> bool:
        try:
            self._archive.getinfo(name)
        except KeyError:
            return False
        return True

    def labels(self) -> tp.Iterator[str]:
        yield from self._archive.namelist()

    def write_array(self, name: str, array: TNDArrayAny) -> None:
        # NOTE: zip only has 'w' mode, not 'wb'
        # NOTE: force_zip64 required for large files
        f = self._archive.open(name, 'w', force_zip64=True) # type: ignore # pylint: disable=R1732
        try:
            NPYConverter.to_npy(f, array)
        finally:
            f.close()

    def read_array(self, name: str) -> TNDArrayAny:
        f = self._archive.open(name) # pylint: disable=R1732
        try:
            array, _ = NPYConverter.from_npy(f, self._header_decode_cache)
        finally:
            f.close()
        array.flags.writeable = False
        return array

    def read_array_header(self, name: str) -> HeaderType:
        '''Alternate reader for status displays.
        '''
        f = self._archive.open(name) # pylint: disable=R1732
        try:
            header = NPYConverter.header_from_npy(f, self._header_decode_cache)
        finally:
            f.close()
        return header

    def size_array(self, name: str) -> int:
        return self._archive.getinfo(name).file_size

    def write_metadata(self, content: tp.Any) -> None:
        # writestr is a method on the ZipFile
        self._archive.writestr(
                self.FILE_META,
                json.dumps(content),
                )

    def read_metadata(self) -> tp.Any:
        return json.loads(self._archive.read(self.FILE_META))

    def size_metadata(self) -> int:
        return self._archive.getinfo(self.FILE_META).file_size

class ArchiveDirectory(Archive):
    '''Archive interface to a directory, where the directory is created on write and NPY files are authored into the files system.
    '''
    __slots__ = ()

    _archive: TPathSpecifier
    FUNC_REMOVE_FP = shutil.rmtree

    def __init__(self,
            fp: TPathSpecifier,
            writeable: bool,
            memory_map: bool,
            ):

        if writeable:
            # because an error in writing will remove the entire directory, we requires the directory to be newly created
            if os.path.exists(fp):
                raise RuntimeError(f'Atttempting to write to an existant directory: {fp}')
            os.mkdir(fp)
        else:
            if not os.path.exists(fp):
                raise RuntimeError(f'Atttempting to read from a non-existant directory: {fp}')
            if not os.path.isdir(fp):
                raise RuntimeError(f'A directory must be provided, not {fp}')
            self._header_decode_cache = {}

        self._archive = fp
        self._memory_map = memory_map

    def labels(self) -> tp.Iterator[str]:
        # NOTE: should this filter?
        yield from (f.name for f in os.scandir(self._archive)) #type: ignore

    def __contains__(self, name: str) -> bool:
        fp = os.path.join(self._archive, name)
        return os.path.exists(fp)

    def write_array(self, name: str, array: TNDArrayAny) -> None:
        fp = os.path.join(self._archive, name)
        f = open(fp, 'wb') # pylint: disable=R1732
        try:
            NPYConverter.to_npy(f, array)
        finally:
            f.close()

    def read_array(self, name: str) -> TNDArrayAny:
        fp = os.path.join(self._archive, name)
        if self._memory_map:
            if not hasattr(self, '_closable'):
                self._closable = []

            f = open(fp, 'rb') # pylint: disable=R1732
            try:
                array, mm = NPYConverter.from_npy(f,
                        self._header_decode_cache,
                        self._memory_map,
                        )
            finally:
                f.close() # NOTE: can close the file after creating memory map
            # self._closable.append(f)
            self._closable.append(mm)
            return array

        f = open(fp, 'rb') # pylint: disable=R1732
        try:
            array, _ = NPYConverter.from_npy(f,
                    self._header_decode_cache,
                    self._memory_map,
                    )
        finally:
            f.close()
        return array

    def read_array_header(self, name: str) -> HeaderType:
        '''Alternate reader for status displays.
        '''
        fp = os.path.join(self._archive, name)
        f = open(fp, 'rb') # pylint: disable=R1732
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
        f = open(fp, 'w', encoding='utf-8') # pylint: disable=R1732
        try:
            f.write(json.dumps(content))
        finally:
            f.close()

    def read_metadata(self) -> tp.Any:
        fp = os.path.join(self._archive, self.FILE_META)
        f = open(fp, 'r', encoding='utf-8') # pylint: disable=R1732
        try:
            post = json.loads(f.read())
        finally:
            f.close()
        return post

    def size_metadata(self) -> int:
        fp = os.path.join(self._archive, self.FILE_META)
        return os.path.getsize(fp)


class ArchiveZipWrapper(Archive):
    '''Archive based on a shared (and already open/created) ZipFile.
    '''
    __slots__ = ('prefix', '_delimiter')

    _archive: ZipFile

    def __init__(self,
            zf: ZipFile,
            writeable: bool,
            memory_map: bool,
            delimiter: str,
            ):

        self._archive = zf
        self.prefix = '' # must be directly set by clients
        self._delimiter = delimiter

        if not writeable:
            self._header_decode_cache = {}
        if memory_map:
            raise RuntimeError(f'Cannot memory_map with {self}')
        self._memory_map = memory_map

    def labels(self) -> tp.Iterator[str]:
        '''Only return unique outer-directory labels, not all contents (NPY) in the file. These labels are exclusively string (they are added post processing with label_encoding).
        '''
        dir_last = '' # dir name cannot be an empty sting
        for name in self._archive.namelist():
            # split on the last observed separator
            if name.endswith(self._delimiter):
                continue #pragma: no cover
            dir_current, _ = name.rsplit(self._delimiter, maxsplit=1)
            if dir_current != dir_last:
                dir_last = dir_current
                yield dir_current

    def __del__(self) -> None:
        # let the creator of the zip perform any cleanup
        pass

    def __contains__(self, name: str) -> bool:
        name = f'{self.prefix}{self._delimiter}{name}'
        try:
            self._archive.getinfo(name)
        except KeyError:
            return False
        return True

    def write_array(self, name: str, array: TNDArrayAny) -> None:
        # NOTE: force_zip64 required for large files
        name = f'{self.prefix}{self._delimiter}{name}'
        f = self._archive.open(name, 'w', force_zip64=True)
        try:
            NPYConverter.to_npy(f, array)
        finally:
            f.close()

    def read_array(self, name: str) -> TNDArrayAny:
        name = f'{self.prefix}{self._delimiter}{name}'
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
        name = f'{self.prefix}{self._delimiter}{name}'
        f = self._archive.open(name)
        try:
            header = NPYConverter.header_from_npy(f, self._header_decode_cache)
        finally:
            f.close()
        return header

    def size_array(self, name: str) -> int:
        name = f'{self.prefix}{self._delimiter}{name}'
        return self._archive.getinfo(name).file_size

    def write_metadata(self, content: tp.Any) -> None:
        name = f'{self.prefix}{self._delimiter}{self.FILE_META}'
        self._archive.writestr(name,
                json.dumps(content),
                )

    def read_metadata(self) -> tp.Any:
        name = f'{self.prefix}{self._delimiter}{self.FILE_META}'
        return json.loads(self._archive.read(name))

    def size_metadata(self) -> int:
        name = f'{self.prefix}{self._delimiter}{self.FILE_META}'
        return self._archive.getinfo(name).file_size


#-------------------------------------------------------------------------------

class ArchiveIndexConverter:
    '''Utility methods for converting Index or index components.
    '''

    @staticmethod
    def index_encode(
            *,
            metadata: tp.Dict[str, TLabel],
            archive: Archive,
            index: 'IndexBase',
            key_template_values: str,
            key_types: str,
            depth: int,
            include: bool,
            ) -> None:
        '''
        Args:
            metadata: mutates in place with json components for class names of index types.
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
            metadata: tp.Dict[str, TLabel],
            archive: Archive,
            array: TNDArrayAny,
            key_template_values: str,
            ) -> None:
        '''
        Args:
            metadata: mutates in place with json components
        '''
        assert array.ndim == 1
        archive.write_array(key_template_values.format(0), array)

    @staticmethod
    def index_decode(*,
            archive: Archive,
            metadata: tp.Dict[str, tp.Any],
            key_template_values: str,
            key_types: str, # which key to fetch IH component types
            depth: int,
            cls_index: tp.Type['IndexBase'],
            name: TName,
            ) -> tp.Optional['IndexBase']:
        '''Build index or columns.
        '''
        from static_frame.core.type_blocks import TypeBlocks

        if key_template_values.format(0) not in archive:
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

    @staticmethod
    def frame_encode(*,
            archive: Archive,
            frame: TFrameAny,
            include_index: bool = True,
            include_columns: bool = True,
            consolidate_blocks: bool = False,
            ) -> None:
        metadata: tp.Dict[str, tp.Any] = {}

        # NOTE: isolate custom pre-json encoding only where needed: on `name` attributes; the name might be nested tuples, so we cannot assume that name is just a string
        metadata[NPYLabel.KEY_NAMES] = [
                JSONTranslator.encode_element(frame._name),
                JSONTranslator.encode_element(frame._index._name),
                JSONTranslator.encode_element(frame._columns._name),
                ]
        # do not store Frame class as caller will determine
        metadata[NPYLabel.KEY_TYPES] = [
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

        ArchiveIndexConverter.index_encode(
                metadata=metadata,
                archive=archive,
                index=frame._index,
                key_template_values=NPYLabel.FILE_TEMPLATE_VALUES_INDEX,
                key_types=NPYLabel.KEY_TYPES_INDEX,
                depth=depth_index,
                include=include_index,
                )
        ArchiveIndexConverter.index_encode(
                metadata=metadata,
                archive=archive,
                index=frame._columns,
                key_template_values=NPYLabel.FILE_TEMPLATE_VALUES_COLUMNS,
                key_types=NPYLabel.KEY_TYPES_COLUMNS,
                depth=depth_columns,
                include=include_columns,
                )
        i = 0
        for i, array in enumerate(block_iter, 1):
            archive.write_array(NPYLabel.FILE_TEMPLATE_BLOCKS.format(i-1), array)

        metadata[NPYLabel.KEY_DEPTHS] = [
                i, # block count
                depth_index,
                depth_columns]

        archive.write_metadata(metadata)


    @classmethod
    def to_archive(cls,
            *,
            frame: TFrameAny,
            fp: TPathSpecifierOrIO,
            include_index: bool = True,
            include_columns: bool = True,
            consolidate_blocks: bool = False,
            ) -> None:
        '''
        Write a :obj:`Frame` as an npz file.
        '''
        archive = cls._ARCHIVE_CLS(fp,
                writeable=True,
                memory_map=False,
                )
        try:
            cls.frame_encode(
                    archive=archive,
                    frame=frame,
                    include_index=include_index,
                    include_columns=include_columns,
                    consolidate_blocks=consolidate_blocks,
                    )
        except ErrorNPYEncode:
            archive.close()
            archive.__del__() # force cleanup
            # fp can be BytesIO in a to_npz/to_zip_npz scenario
            if not isinstance(fp, io.IOBase) and os.path.exists(fp):  # type: ignore[arg-type]
                cls._ARCHIVE_CLS.FUNC_REMOVE_FP(fp)  # type: ignore[arg-type]
            raise


    @classmethod
    def frame_decode(cls,
            *,
            archive: Archive,
            constructor: tp.Type[TFrameAny],
            ) -> TFrameAny:
        '''
        Create a :obj:`Frame` from an npz file.
        '''
        from static_frame.core.type_blocks import TypeBlocks

        metadata = archive.read_metadata()

        # NOTE: we isolate custom post-JSON decoding to only where it is needed: the name attributes. JSON will bring back tuple `name` attributes as lists; these must be converted to tuples to be hashable. Alternatives (like storing repr and using literal_eval) are slower than JSON.
        names = metadata[NPYLabel.KEY_NAMES]

        name = JSONTranslator.decode_element(names[0])
        name_index = JSONTranslator.decode_element(names[1])
        name_columns = JSONTranslator.decode_element(names[2])

        block_count, depth_index, depth_columns = metadata[NPYLabel.KEY_DEPTHS]

        cls_index: tp.Type[IndexBase]
        cls_columns: tp.Type[IndexBase]
        cls_index, cls_columns = (ContainerMap.str_to_cls(name) # type: ignore
                for name in metadata[NPYLabel.KEY_TYPES])

        index = ArchiveIndexConverter.index_decode(
                archive=archive,
                metadata=metadata,
                key_template_values=NPYLabel.FILE_TEMPLATE_VALUES_INDEX,
                key_types=NPYLabel.KEY_TYPES_INDEX,
                depth=depth_index,
                cls_index=cls_index,
                name=name_index,
                )

        # we need to align the mutability of the constructor with the Index type on the columns
        if constructor.STATIC != cls_columns.STATIC:
            if constructor.STATIC:
                cls_columns = cls_columns._IMMUTABLE_CONSTRUCTOR #type: ignore
            else:
                cls_columns = cls_columns._MUTABLE_CONSTRUCTOR #type: ignore

        columns = ArchiveIndexConverter.index_decode(
                archive=archive,
                metadata=metadata,
                key_template_values=NPYLabel.FILE_TEMPLATE_VALUES_COLUMNS,
                key_types=NPYLabel.KEY_TYPES_COLUMNS,
                depth=depth_columns,
                cls_index=cls_columns,
                name=name_columns,
                )

        if block_count:
            tb = TypeBlocks.from_blocks(
                    archive.read_array(NPYLabel.FILE_TEMPLATE_BLOCKS.format(i))
                    for i in range(block_count)
                    )
        else:
            tb = TypeBlocks.from_zero_size_shape()

        f = constructor(tb,
                own_data=True,
                index=index,
                own_index = False if index is None else True,
                columns=columns,
                own_columns = False if columns is None else True,
                name=name,
                )
        return f

    @classmethod
    def from_archive(cls,
            *,
            constructor: tp.Type[TFrameAny],
            fp: TPathSpecifierOrIO,
            ) -> TFrameAny:
        '''
        Create a :obj:`Frame` from an npz file.
        '''
        archive = cls._ARCHIVE_CLS(fp,
                writeable=False,
                memory_map=False,
                )
        f = cls.frame_decode(
                archive=archive,
                constructor=constructor,
                )
        return f

    @classmethod
    def from_archive_mmap(cls,
            *,
            constructor: tp.Type[TFrameAny],
            fp: TPathSpecifier,
            ) -> tp.Tuple[TFrameAny, tp.Callable[[], None]]:
        '''
        Create a :obj:`Frame` from an npz file.
        '''
        archive = cls._ARCHIVE_CLS(fp,
                writeable=False,
                memory_map=True,
                )
        f = cls.frame_decode(
                archive=archive,
                constructor=constructor,
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

    def __init__(self, fp: TPathSpecifier, mode: str = 'r') -> None:
        if mode == 'w':
            writeable = True
        elif mode == 'r':
            writeable = False
        else:
            raise RuntimeError('Invalid value for mode; use "w" or "r"')

        self._writeable = writeable
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
    def contents(self) -> TFrameAny:
        '''
        Return a :obj:`Frame` indicating name, dtype, shape, and bytes, of Archive components.
        '''
        if self._writeable:
            raise UnsupportedOperation('Open with mode "r" to get contents.')

        from static_frame.core.frame import Frame
        def gen() -> tp.Iterator[tp.Tuple[tp.Any, ...]]:
            # metadata is in labels; sort by ext,ension first to put at top
            for name in sorted(
                    self._archive.labels(),
                    key=lambda fn: tuple(reversed(fn.split('.')))
                    ):
                # NOTE: will not work with ArchiveZipWrapper
                if name == self._archive.FILE_META:
                    yield (name, self._archive.size_metadata()) + ('', '', '')
                else:
                    header = self._archive.read_array_header(name)
                    yield (name, self._archive.size_array(name)) + header

        f: TFrameAny = Frame.from_records(gen(),
                columns=('name', 'size', 'dtype', 'fortran', 'shape'),
                name=str(self._archive._archive),
                )
        return f.set_index('name', drop=True)

    @property
    def nbytes(self) -> int:
        '''
        Return numer of bytes stored in this archive.
        '''
        if self._writeable:
            raise UnsupportedOperation('Open with mode "r" to get nbytes.')

        def gen() -> tp.Iterator[int]:
            # metadata is in labels; sort by extension first to put at top
            for name in self._archive.labels():
                # NOTE: will not work with ArchiveZipWrapper
                if name == self._archive.FILE_META:
                    yield self._archive.size_metadata()
                else:
                    yield self._archive.size_array(name)
        return sum(gen())

    def from_arrays(self,
            blocks: tp.Iterable[TNDArrayAny],
            *,
            index: TNDArrayAny | IndexBase | None = None,
            columns: TNDArrayAny | IndexBase | None = None,
            name: TName = None,
            axis: int = 0,
            ) -> None:
        '''
        Given an iterable of arrays, write out an NPZ or NPY directly, without building up intermediary :obj:`Frame`. If axis 0, the arrays are vertically stacked; if axis 1, they are horizontally stacked. For both axis, if included, indices must be of appropriate length.

        Args:
            blocks:
            *
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
                    key_template_values=NPYLabel.FILE_TEMPLATE_VALUES_INDEX,
                    key_types=NPYLabel.KEY_TYPES_INDEX,
                    depth=depth_index,
                    include=True,
                    )
        elif index is not None:
            if index.__class__ is not np.ndarray:
                raise RuntimeError('index argument must be an Index, IndexHierarchy, or 1D np.ndarray')

            depth_index = 1
            name_index = None
            cls_index = dtype_to_index_cls(True, index.dtype)
            ArchiveIndexConverter.array_encode(
                    metadata=metadata,
                    archive=self._archive,
                    array=index,
                    key_template_values=NPYLabel.FILE_TEMPLATE_VALUES_INDEX,
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
                    key_template_values=NPYLabel.FILE_TEMPLATE_VALUES_COLUMNS,
                    key_types=NPYLabel.KEY_TYPES_COLUMNS,
                    depth=depth_columns,
                    include=True,
                    )
        elif columns is not None:
            if columns.__class__ is not np.ndarray:
                raise RuntimeError('columns argument must be an Index, IndexHierarchy, or 1D np.ndarray')

            depth_columns = 1 # only support 1D
            name_columns = None
            cls_columns = dtype_to_index_cls(True, columns.dtype)
            ArchiveIndexConverter.array_encode(
                    metadata=metadata,
                    archive=self._archive,
                    array=columns,
                    key_template_values=NPYLabel.FILE_TEMPLATE_VALUES_COLUMNS,
                    )
        else:
            depth_columns = 1 # only support 1D
            name_columns = None
            cls_columns = Index

        metadata[NPYLabel.KEY_NAMES] = [name,
                name_index,
                name_columns,
                ]
        # do not store Frame class as caller will determine
        metadata[NPYLabel.KEY_TYPES] = [
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
                self._archive.write_array(NPYLabel.FILE_TEMPLATE_BLOCKS.format(i), array)
        elif axis == 0:
            # for now, just vertically concat and write, though this has a 2X memory requirement
            resolved = concat_resolved(blocks, axis=0)
            # if this results in an obect array, an exception will be raised
            self._archive.write_array(NPYLabel.FILE_TEMPLATE_BLOCKS.format(0), resolved)
            i = 0
        else:
            raise AxisInvalid(f'invalid axis {axis}')

        metadata[NPYLabel.KEY_DEPTHS] = [
                i + 1, # block count
                depth_index,
                depth_columns]
        self._archive.write_metadata(metadata)

    def from_frames(self,
            frames: tp.Iterable[TFrameAny],
            *,
            include_index: bool = True,
            include_columns: bool = True,
            axis: int = 0,
            union: bool = True,
            name: TName = None,
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

        from static_frame.core.frame import Frame
        from static_frame.core.type_blocks import TypeBlocks

        frames = [f if isinstance(f, Frame) else f.to_frame(axis) for f in frames] # type: ignore
        index: tp.Optional[IndexBase]

        # NOTE: based on Frame.from_concat
        if axis == 1: # stacks columns (extends rows horizontally)
            if include_columns:
                try:
                    columns = index_many_concat(
                            (f._columns for f in frames),
                            Index,
                            )
                except ErrorInitIndexNonUnique:
                    raise RuntimeError('Column names after horizontal concatenation are not unique; set include_columns to None to ignore.') from None
            else:
                columns = None

            if include_index:
                index = index_many_to_one(
                        (f._index for f in frames),
                        Index,
                        many_to_one_type=ManyToOneType.UNION if union else ManyToOneType.INTERSECT,
                        )
            else:
                raise RuntimeError('Must include index for horizontal alignment.')

            def blocks() -> tp.Iterator[TNDArrayAny]:
                for f in frames:
                    if len(f.index) != len(index) or (f.index != index).any(): # type: ignore
                        f = f.reindex(index=index, fill_value=fill_value)
                    for block in f._blocks._blocks:
                        yield block

        elif axis == 0: # stacks rows (extends columns vertically)
            if include_index:
                try:
                    index = index_many_concat((f._index for f in frames), Index)
                except ErrorInitIndexNonUnique:
                    raise RuntimeError('Index names after vertical concatenation are not unique; set include_index to None to ignore') from None
            else:
                index = None

            if include_columns:
                columns = index_many_to_one(
                        (f._columns for f in frames),
                        Index,
                        many_to_one_type=ManyToOneType.UNION if union else ManyToOneType.INTERSECT,
                        )
            else:
                raise RuntimeError('Must include columns for vertical alignment.')

            def blocks() -> tp.Iterator[TNDArrayAny]:
                type_blocks = []
                previous_f: tp.Optional[TFrameAny] = None
                block_compatible = True
                reblock_compatible = True

                for f in frames:
                    if len(f.columns) != len(columns) or (f.columns != columns).any(): # type: ignore
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

class NPY(ArchiveComponentsConverter):
    '''Utility object for reading characteristics from, or writing new, NPY directories from arrays or :obj:`Frame`.
    '''
    _ARCHIVE_CLS = ArchiveDirectory


