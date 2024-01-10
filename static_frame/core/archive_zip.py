'''
Optimized reader of ZIP files. Based largely on CPython, Lib/zipfile/__init__.py

'''
import binascii
import io
import os
import struct
import typing as tp
from zipfile import ZIP_STORED
from zipfile import BadZipFile

ZIP64_LIMIT = (1 << 31) - 1
ZIP_FILECOUNT_LIMIT = (1 << 16) - 1
ZIP_MAX_COMMENT = (1 << 16) - 1


# DEFAULT_VERSION = 20
# ZIP64_VERSION = 45
# BZIP2_VERSION = 46
# LZMA_VERSION = 63
# we recognize (but not necessarily support) all features up to that version
# MAX_EXTRACT_VERSION = 63

# Below are some formats and associated data for reading/writing headers using
# the struct module.  The names and structures of headers/records are those used
# in the PKWARE description of the ZIP file format:
#     http://www.pkware.com/documents/casestudies/APPNOTE.TXT
# (URL valid as of January 2008)

# The "end of central directory" structure, magic number, size, and indices
# (section V.I in the format document)
_END_ARCHIVE_STRUCT = b"<4s4H2LH"
_END_ARCHIVE_STRING = b"PK\005\006"
_END_ARCHIVE_SIZE = struct.calcsize(_END_ARCHIVE_STRUCT)

_ECD_SIGNATURE = 0
_ECD_DISK_NUMBER = 1
_ECD_DISK_START = 2
_ECD_ENTRIES_THIS_DISK = 3
_ECD_ENTRIES_TOTAL = 4
_ECD_SIZE = 5
_ECD_OFFSET = 6
_ECD_COMMENT_SIZE = 7
# These last two indices are not part of the structure as defined in the
# spec, but they are used internally by this module as a convenience
_ECD_COMMENT = 8
_ECD_LOCATION = 9

# The "central directory" structure, magic number, size, and indices
# of entries in the structure (section V.F in the format document)
_CENTRAL_DIR_STRUCT = "<4s4B4HL2L5H2L"
_CENTRAL_DIR_STRING = b"PK\001\002"
_CENTRAL_DIR_SIZE = struct.calcsize(_CENTRAL_DIR_STRUCT)

# indexes of entries in the central directory structure
_CD_SIGNATURE = 0
_CD_CREATE_VERSION = 1
_CD_CREATE_SYSTEM = 2
_CD_EXTRACT_VERSION = 3
_CD_EXTRACT_SYSTEM = 4
_CD_FLAG_BITS = 5
_CD_COMPRESS_TYPE = 6
_CD_TIME = 7
_CD_DATE = 8
_CD_CRC = 9
_CD_COMPRESSED_SIZE = 10
_CD_UNCOMPRESSED_SIZE = 11
_CD_FILENAME_LENGTH = 12
_CD_EXTRA_FIELD_LENGTH = 13
_CD_COMMENT_LENGTH = 14
_CD_DISK_NUMBER_START = 15
_CD_INTERNAL_FILE_ATTRIBUTES = 16
_CD_EXTERNAL_FILE_ATTRIBUTES = 17
_CD_LOCAL_HEADER_OFFSET = 18

# General purpose bit flags
# Zip Appnote: 4.4.4 general purpose bit flag: (2 bytes)
_MASK_ENCRYPTED = 1 << 0
# Bits 1 and 2 have different meanings depending on the compression used.
_MASK_COMPRESS_OPTION_1 = 1 << 1
# _MASK_COMPRESS_OPTION_2 = 1 << 2
# _MASK_USE_DATA_DESCRIPTOR: If set, crc-32, compressed size and uncompressed
# size are zero in the local header and the real values are written in the data
# descriptor immediately following the compressed data.
_MASK_USE_DATA_DESCRIPTOR = 1 << 3
# Bit 4: Reserved for use with compression method 8, for enhanced deflating.
# _MASK_RESERVED_BIT_4 = 1 << 4
_MASK_COMPRESSED_PATCH = 1 << 5
_MASK_STRONG_ENCRYPTION = 1 << 6
# _MASK_UNUSED_BIT_7 = 1 << 7
# _MASK_UNUSED_BIT_8 = 1 << 8
# _MASK_UNUSED_BIT_9 = 1 << 9
# _MASK_UNUSED_BIT_10 = 1 << 10
_MASK_UTF_FILENAME = 1 << 11
# Bit 12: Reserved by PKWARE for enhanced compression.
# _MASK_RESERVED_BIT_12 = 1 << 12
# _MASK_ENCRYPTED_CENTRAL_DIR = 1 << 13
# Bit 14, 15: Reserved by PKWARE
# _MASK_RESERVED_BIT_14 = 1 << 14
# _MASK_RESERVED_BIT_15 = 1 << 15

# The "local file header" structure, magic number, size, and indices
# (section V.A in the format document)
_FILE_HEADER_STRUCT = "<4s2B4HL2L2H"
_FILE_HEADER_STRING = b"PK\003\004"
_FILE_HEADER_SIZE = struct.calcsize(_FILE_HEADER_STRUCT)

_FH_SIGNATURE = 0
_FH_EXTRACT_VERSION = 1
_FH_EXTRACT_SYSTEM = 2
_FH_GENERAL_PURPOSE_FLAG_BITS = 3
_FH_COMPRESSION_METHOD = 4
_FH_LAST_MOD_TIME = 5
_FH_LAST_MOD_DATE = 6
_FH_CRC = 7
_FH_COMPRESSED_SIZE = 8
_FH_UNCOMPRESSED_SIZE = 9
_FH_FILENAME_LENGTH = 10
_FH_EXTRA_FIELD_LENGTH = 11

# The "Zip64 end of central directory locator" structure, magic number, and size
_END_ARCHIVE64_LOCATOR_STRUCT = "<4sLQL"
_END_ARCHIVE64_LOCATOR_STRING = b"PK\x06\x07"
_END_ARCHIVE64_LOCATOR_SIZE = struct.calcsize(_END_ARCHIVE64_LOCATOR_STRUCT)

# The "Zip64 end of central directory" record, magic number, size, and indices
# (section V.G in the format document)
_END_ARCHIVE64_STRUCT = "<4sQ2H2L4Q"
_END_ARCHIVE64_STRING = b"PK\x06\x06"
_END_ARCHIVE64_SIZE = struct.calcsize(_END_ARCHIVE64_STRUCT)

_CD64_SIGNATURE = 0
_CD64_DIRECTORY_RECSIZE = 1
_CD64_CREATE_VERSION = 2
_CD64_EXTRACT_VERSION = 3
_CD64_DISK_NUMBER = 4
_CD64_DISK_NUMBER_START = 5
_CD64_NUMBER_ENTRIES_THIS_DISK = 6
_CD64_NUMBER_ENTRIES_TOTAL = 7
_CD64_DIRECTORY_SIZE = 8
_CD64_OFFSET_START_CENTDIR = 9

_DD_SIGNATURE = 0x08074b50



TEndArchive = tp.List[tp.Union[bytes, int]]

def _end_archive64_update(fpin: tp.IO[bytes],
        offset: int,
        endrec: TEndArchive,
        ) -> TEndArchive:
    '''
    Read the ZIP64 end-of-archive records and use that to update endrec
    '''
    try:
        fpin.seek(offset - _END_ARCHIVE64_LOCATOR_SIZE, 2)
    except OSError:
        # If the seek fails, the file is not large enough to contain a ZIP64
        # end-of-archive record, so just return the end record we were given.
        return endrec

    data = fpin.read(_END_ARCHIVE64_LOCATOR_SIZE)
    if len(data) != _END_ARCHIVE64_LOCATOR_SIZE:
        return endrec

    sig, diskno, reloff, disks = struct.unpack(_END_ARCHIVE64_LOCATOR_STRUCT, data)
    if sig != _END_ARCHIVE64_LOCATOR_STRING:
        return endrec

    if diskno != 0 or disks > 1:
        raise BadZipFile("zipfiles that span multiple disks are not supported")

    # Assume no 'zip64 extensible data'
    fpin.seek(offset - _END_ARCHIVE64_LOCATOR_SIZE - _END_ARCHIVE64_SIZE, 2)
    data = fpin.read(_END_ARCHIVE64_SIZE)
    if len(data) != _END_ARCHIVE64_SIZE:
        return endrec

    (
    sig,
    sz,
    create_version,
    read_version,
    disk_num,
    disk_dir,
    dircount,
    dircount2,
    dirsize,
    diroffset
    ) = struct.unpack(_END_ARCHIVE64_STRUCT, data)

    if sig != _END_ARCHIVE64_STRING:
        return endrec

    # Update the original endrec using data from the ZIP64 record
    endrec[_ECD_SIGNATURE] = sig
    endrec[_ECD_DISK_NUMBER] = disk_num
    endrec[_ECD_DISK_START] = disk_dir
    endrec[_ECD_ENTRIES_THIS_DISK] = dircount
    endrec[_ECD_ENTRIES_TOTAL] = dircount2
    endrec[_ECD_SIZE] = dirsize
    endrec[_ECD_OFFSET] = diroffset
    return endrec


def _extract_end_archive(fpin) -> TEndArchive | None:
    '''Return data from the "End of Central Directory" record, or None.

    The data is a list of the nine items in the ZIP "End of central dir"
    record followed by a tenth item, the file seek offset of this record.'''

    # Determine file size
    fpin.seek(0, 2) # seek to end
    filesize = fpin.tell()

    # Check to see if this is ZIP file with no archive comment (the
    # "end of central directory" structure should be the last item in the
    # file if this is the case).
    try:
        fpin.seek(-_END_ARCHIVE_SIZE, 2)
    except OSError:
        return None

    endrec: TEndArchive

    data = fpin.read()
    if (len(data) == _END_ARCHIVE_SIZE and
            data[0:4] == _END_ARCHIVE_STRING and
            data[-2:] == b"\000\000"):
        # the signature is correct and there's no comment, unpack structure
        endrec = list(struct.unpack(_END_ARCHIVE_STRUCT, data))
        # Append a blank comment and record start offset
        endrec.append(b"")
        endrec.append(filesize - _END_ARCHIVE_SIZE)
        # Try to read the "Zip64 end of central directory" structure
        return _end_archive64_update(fpin, -_END_ARCHIVE_SIZE, endrec)

    # Either this is not a ZIP file, or it is a ZIP file with an archive
    # comment.  Search the end of the file for the "end of central directory"
    # record signature. The comment is the last item in the ZIP file and may be
    # up to 64K long.  It is assumed that the "end of central directory" magic
    # number does not appear in the comment.
    comment_max_start = max(filesize - (1 << 16) - _END_ARCHIVE_SIZE, 0)
    fpin.seek(comment_max_start, 0)

    data = fpin.read()
    start = data.rfind(_END_ARCHIVE_STRING)
    if start >= 0:
        # found the magic number; attempt to unpack and interpret
        recData = data[start: start + _END_ARCHIVE_SIZE]
        if len(recData) != _END_ARCHIVE_SIZE:
            return None # Zip file is corrupted.

        endrec = list(struct.unpack(_END_ARCHIVE_STRUCT, recData))
        # comment = data[
        #         start + _END_ARCHIVE_SIZE:
        #         start + _END_ARCHIVE_SIZE + endrec[_ECD_COMMENT_SIZE]
        #         ]
        endrec.append(b'') # ignore comment
        endrec.append(comment_max_start + start)

        # Try to read the "Zip64 end of central directory" structure
        return _end_archive64_update(fpin,
                comment_max_start + start - filesize,
                endrec,
                )
    # Unable to find a valid end of central directory structure
    return None


class ZipInfoRO:
    '''Class with attributes describing each file in the ZIP archive.'''

    __slots__ = (
        'filename',
        'flag_bits',
        'header_offset',
        'file_size',
    )

    def __init__(self,
            filename: str = "NoName",
            ):
        self.filename = filename
        self.flag_bits = 0
        self.header_offset = 0
        self.file_size = 0

#-------------------------------------------------------------------------------

# class _FileSharedRO:
#     '''
#     This wrapper around an IO bytes stream takes a close function at initialization, and exclusively uses that on close() with the composed file instance.
#     '''
#     __slots__ = (
#             '_file',
#             '_pos',
#             '_close',
#             )

#     def __init__(self,
#             file,
#             pos: int,
#             close: tp.Callable[..., None]):
#         '''
#         Args:
#             pos: the start position, just after the header
#         '''
#         self._file = file
#         self._pos = pos
#         self._close = close # callable

#     # def __enter__(self):
#     #     return self

#     # def __exit__(self, type, value, traceback):
#     #     self.close()

#     @property
#     def seekable(self):
#         return self._file.seekable

#     def tell(self) -> int:
#         return self._pos

#     def seek(self, offset: int, whence: int = 0) -> int:
#         self._file.seek(offset, whence)
#         self._pos = self._file.tell()
#         return self._pos

#     def read(self, n: int = -1):
#         self._file.seek(self._pos)
#         data = self._file.read(n)
#         self._pos = self._file.tell()
#         return data

#     def close(self) -> None:
#         if self._file is not None:
#             file = self._file
#             self._file = None
#             self._close(file)

#-------------------------------------------------------------------------------

# class _ZipFilePartRO(io.BufferedIOBase):
#     '''File-like object for reading an archive member.
#        Is returned by ZipFileRO.open().
#     '''
#     # Max size supported by decompressor.
#     MAX_N = 1 << 31 - 1
#     MIN_READ_SIZE = 4096 # Read from compressed files in 4k blocks.
#     MAX_SEEK_READ = 1 << 24 # Chunk size to read during seek

#     __slots__ = (
#         '_file',
#         '_compress_left',
#         '_left',
#         '_eof',
#         '_readbuffer',
#         '_offset',
#         '_orig_file_start',
#         '_orig_file_size',
#         )

#     def __init__(self,
#                 file: _FileSharedRO,
#                 zinfo: ZipInfoRO,
#                 ):

#         self._file = file
#         self._compress_left = zinfo.file_size
#         self._left = zinfo.file_size

#         self._eof = False
#         self._readbuffer = b''
#         self._offset = 0

#         self._orig_file_start = file.tell()
#         self._orig_file_size = zinfo.file_size


#     # def readline(self, limit=-1):
#     #     '''Read and return a line from the stream.

#     #     If limit is specified, at most limit bytes will be read.
#     #     '''

#     #     if limit < 0:
#     #         # Shortcut common case - newline found in buffer.
#     #         i = self._readbuffer.find(b'\n', self._offset) + 1
#     #         if i > 0:
#     #             line = self._readbuffer[self._offset: i]
#     #             self._offset = i
#     #             return line

#     #     return io.BufferedIOBase.readline(self, limit)

#     def peek(self, n: int = 1) -> bytes:
#         '''Returns buffered bytes without advancing the position.'''
#         if n > len(self._readbuffer) - self._offset:
#             chunk = self.read(n)
#             if len(chunk) > self._offset:
#                 self._readbuffer = chunk + self._readbuffer[self._offset:]
#                 self._offset = 0
#             else:
#                 self._offset -= len(chunk)

#         # Return up to 512 bytes to reduce allocation overhead for tight loops.
#         return self._readbuffer[self._offset: self._offset + 512]

#     def readable(self) -> bool:
#         if self.closed:
#             raise ValueError("I/O operation on closed file.")
#         return True


#     def _read2(self, n: int) -> bytes:
#         if self._compress_left <= 0:
#             return b''

#         n = max(n, self.MIN_READ_SIZE)
#         n = min(n, self._compress_left)

#         data = self._file.read(n)
#         self._compress_left -= len(data)
#         if not data:
#             raise EOFError

#         return data

#     def _read1(self, n: int) -> bytes:
#         # Read up to n compressed bytes with at most one read() system call,
#         if self._eof or n <= 0:
#             return b''

#         data = self._read2(n)
#         self._eof = self._compress_left <= 0

#         data = data[:self._left]
#         self._left -= len(data)
#         if self._left <= 0:
#             self._eof = True

#         return data

#     def read(self, n: int = -1) -> bytes:
#         '''Read and return up to n bytes.
#         If the argument is omitted, None, or negative, data is read and returned until EOF is reached.
#         '''
#         if self.closed:
#             raise ValueError("read from closed file.")

#         if n < 0:
#             buf = self._readbuffer[self._offset:]
#             self._readbuffer = b''
#             self._offset = 0
#             while not self._eof:
#                 buf += self._read1(self.MAX_N) # will set self._eof
#             return buf

#         end = n + self._offset
#         if end < len(self._readbuffer):
#             buf = self._readbuffer[self._offset: end]
#             self._offset = end
#             return buf

#         n = end - len(self._readbuffer)
#         buf = self._readbuffer[self._offset:]
#         self._readbuffer = b''
#         self._offset = 0
#         while n > 0 and not self._eof:
#             data = self._read1(n)
#             if n < len(data):
#                 self._readbuffer = data
#                 self._offset = n
#                 buf += data[:n]
#                 break
#             buf += data
#             n -= len(data)
#         return buf

#     def close(self) -> None:
#         try:
#             self._file.close()
#         finally:
#             super().close()

#     def seekable(self) -> bool:
#         if self.closed:
#             raise ValueError("I/O operation on closed file.")
#         return True

#     def seek(self, offset, whence=os.SEEK_SET) -> int:
#         if self.closed:
#             raise ValueError("seek on closed file.")

#         curr_pos = self.tell()
#         if whence == os.SEEK_SET:
#             new_pos = offset
#         elif whence == os.SEEK_CUR:
#             new_pos = curr_pos + offset
#         elif whence == os.SEEK_END:
#             new_pos = self._orig_file_size + offset
#         else:
#             raise ValueError("whence must be os.SEEK_SET (0), os.SEEK_CUR (1), or os.SEEK_END (2)")

#         if new_pos > self._orig_file_size:
#             new_pos = self._orig_file_size

#         if new_pos < 0:
#             new_pos = 0

#         read_offset = new_pos - curr_pos
#         buff_offset = read_offset + self._offset

#         if buff_offset >= 0 and buff_offset < len(self._readbuffer):
#             # Just move the _offset index if the new position is in the _readbuffer
#             self._offset = buff_offset
#             read_offset = 0

#         elif read_offset > 0:
#             # seek actual file taking already buffered data into account
#             read_offset -= len(self._readbuffer) - self._offset
#             self._file.seek(read_offset, os.SEEK_CUR)
#             self._left -= read_offset
#             read_offset = 0
#             self._readbuffer = b''
#             self._offset = 0

#         elif read_offset < 0:
#             # Position is before the current position. Reset the ZipFilePartRO
#             self._file.seek(self._orig_file_start)
#             self._compress_left = self._orig_file_size
#             self._left = self._orig_file_size
#             self._readbuffer = b''
#             self._offset = 0
#             self._eof = False
#             read_offset = new_pos

#         while read_offset > 0:
#             read_len = min(self.MAX_SEEK_READ, read_offset)
#             self.read(read_len)
#             read_offset -= read_len

#         return self.tell()

#     def tell(self) -> int:
#         if self.closed:
#             raise ValueError("tell on closed file.")
#         filepos = self._orig_file_size - self._left - len(self._readbuffer) + self._offset
#         return filepos


#-------------------------------------------------------------------------------

class ZipFilePartRO:
    '''
    This wrapper around an IO bytes stream takes a close function at initialization, and exclusively uses that on close() with the composed file instance.
    '''
    __slots__ = (
            '_file',
            '_pos',
            '_close',
            # '_pos_start',
            '_pos_end',
            '_file_size',
            )

    def __init__(self,
            file: tp.IO[bytes],
            close: tp.Callable[..., None],
            zinfo: ZipInfoRO,
            ):
        '''
        Args:
            pos: the start position, just after the header
        '''
        self._file = file
        self._pos = + zinfo.header_offset
        self._close = close # callable

        # self._pos_start = self._pos
        # self._orig_file_size = zinfo.file_size
        self._file_size = zinfo.file_size # main data size after header

        self._pos_end = -1 # self._pos + zinfo.file_size


    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    @property
    def seekable(self):
        return self._file.seekable

    def update_pos_end(self):
        assert self._pos_end < 0 # only allow once
        self._pos_end = self._pos + self._file_size

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        # NOTE: this presently permits unbound seeking in the complete zip file, thus we limit seeking to those relative to current posiion
        if whence != 1:
            raise NotImplementedError('start- or end-relative seeks are not permitted.')

        self._file.seek(self._pos)
        self._file.seek(offset, whence)
        self._pos = self._file.tell()
        return self._pos

    def read(self, n: int = -1):
        self._file.seek(self._pos)
        if n < 0:
            assert self._pos_end >= 0
            n_read = self._pos_end - self._pos
        else:
            n_read = n
        data = self._file.read(n_read)
        self._pos = self._file.tell()
        return data

    def readinto(self, buffer: tp.IO[bytes]) -> int:
        self._file.seek(self._pos)
        # assert self._pos_end >= 0
        # n_read = self._pos_end - self._pos

        count = self._file.readinto(buffer)
        self._pos = self._file.tell()
        return count


    def close(self) -> None:
        if self._file is not None:
            file = self._file
            self._file = None
            self._close(file)

#-------------------------------------------------------------------------------
class ZipFileRO:

    __slots__ = (
        '_name_to_info',
        '_file_passed',
        '_file_name',
        '_file',
        '_file_ref_count',
        )

    @staticmethod
    def _yield_zinfos(file: tp.IO[bytes]) -> tp.Iterator[ZipInfoRO]:
        '''Read in the table of contents for the ZIP file.'''
        try:
            endrec: TEndArchive = _extract_end_archive(file)
        except OSError:
            raise BadZipFile("File is not a zip file")
        if not endrec:
            raise BadZipFile("File is not a zip file")

        size_cd = endrec[_ECD_SIZE]
        offset_cd = endrec[_ECD_OFFSET]

        # "concat" is zero, unless zip was concatenated to another file
        concat = endrec[_ECD_LOCATION] - size_cd - offset_cd
        if endrec[_ECD_SIGNATURE] == _END_ARCHIVE64_STRING:
            # If Zip64 extension structures are present, account for them
            concat -= (_END_ARCHIVE64_SIZE + _END_ARCHIVE64_LOCATOR_SIZE)

        start_cd = offset_cd + concat # Position of start of central directory
        if start_cd < 0:
            raise BadZipFile("Bad offset for central directory")

        file.seek(start_cd, 0)
        data = file.read(size_cd)
        file_cd = io.BytesIO(data)

        total = 0
        filename_length = 0
        extra_length = 0
        comment_length = 0

        while total < size_cd:
            cdir_size = file_cd.read(_CENTRAL_DIR_SIZE)
            if len(cdir_size) != _CENTRAL_DIR_SIZE:
                raise BadZipFile("Truncated central directory")

            cdir = struct.unpack(_CENTRAL_DIR_STRUCT, cdir_size)
            if cdir[_CD_SIGNATURE] != _CENTRAL_DIR_STRING:
                raise BadZipFile("Bad magic number for central directory")
            if cdir[_CD_COMPRESS_TYPE] != ZIP_STORED:
                raise BadZipFile("Cannot process compressed zips")

            filename_length = cdir[_CD_FILENAME_LENGTH]
            filename = file_cd.read(filename_length)

            flags = cdir[_CD_FLAG_BITS]

            if flags & _MASK_UTF_FILENAME: # UTF-8 file names extension
                filename = filename.decode('utf-8')
            else: # Historical ZIP filename encoding
                filename = filename.decode('cp437')

            zinfo = ZipInfoRO(filename)

            extra_length = cdir[_CD_EXTRA_FIELD_LENGTH]
            comment_length = cdir[_CD_COMMENT_LENGTH]
            file_cd.seek(extra_length + comment_length, 1)

            zinfo.header_offset = cdir[_CD_LOCAL_HEADER_OFFSET] + concat
            zinfo.flag_bits = flags
            zinfo.file_size = cdir[_CD_UNCOMPRESSED_SIZE]

            yield zinfo

            total = (
                    total +
                    _CENTRAL_DIR_SIZE +
                    filename_length +
                    extra_length +
                    comment_length
                    )


    def __init__(self, file):
        '''Open the ZIP file with mode read 'r', write 'w', exclusive create 'x',
        or append 'a'.'''


        if isinstance(file, os.PathLike):
            file = os.fspath(file)

        if isinstance(file, str):
            self._file_passed = False
            self._file_name = file
            self._file = io.open(file, 'rb')
        else:
            self._file_passed = True
            self._file = file
            self._file_name = getattr(file, 'name', None)

        self._file_ref_count = 1

        try:
            self._name_to_info = {
                    zinfo.filename: zinfo for zinfo in self._yield_zinfos(self._file)
                    }
        except:
            fp = self._file
            self._file = None
            self._close(fp)
            raise

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __repr__(self):
        result = [f'<{self.__class__.__name__}']

        if self._file is not None:
            if self._file_passed:
                result.append(' file=%r' % self._file)
            elif self._file_name is not None:
                result.append(' filename=%r' % self._file_name)
        else:
            result.append(' [closed]')
        result.append('>')
        return ''.join(result)

    def namelist(self):
        '''Return a list of file names in the archive.'''
        # return [data.filename for data in self.filelist]
        return list(self._name_to_info.keys())

    def infolist(self):
        '''Return a list of class ZipInfoRO instances for files in the
        archive.'''
        # return self.filelist
        return self._name_to_info.values()

    def getinfo(self, name: str):
        '''Return the instance of ZipInfoRO given 'name'.'''
        info = self._name_to_info.get(name)
        if info is None:
            raise KeyError(
                'There is no item named %r in the archive' % name)
        return info

    def read(self, name: str):
        '''Return file bytes for name.'''
        with self.open(name) as file:
            return file.read()

    def open(self, name: str) -> ZipFilePartRO:
        '''Return file-like object for 'name'.

        name is a string for the file name within the ZIP file
        '''
        if not self._file:
            raise ValueError("Attempt to use ZIP archive that was already closed")

        zinfo = self.getinfo(name)

        self._file_ref_count += 1
        # file_shared = _FileSharedRO(self._file,
        #         zinfo.header_offset,
        #         self._close,
        #         )

        file_shared = ZipFilePartRO(
                self._file,
                self._close,
                zinfo,
                )

        try:
            fheader = file_shared.read(_FILE_HEADER_SIZE)
            if len(fheader) != _FILE_HEADER_SIZE:
                raise BadZipFile("Truncated file header")

            fheader = struct.unpack(_FILE_HEADER_STRUCT, fheader)
            if fheader[_FH_SIGNATURE] != _FILE_HEADER_STRING:
                raise BadZipFile("Bad magic number for file header")

            file_shared.seek(fheader[_FH_FILENAME_LENGTH], 1)

            if fheader[_FH_EXTRA_FIELD_LENGTH]:
                file_shared.seek(fheader[_FH_EXTRA_FIELD_LENGTH], 1)

            if zinfo.flag_bits & _MASK_COMPRESSED_PATCH: # Zip 2.7: compressed patched data
                raise NotImplementedError("compressed patched data (flag bit 5)")
            if zinfo.flag_bits & _MASK_STRONG_ENCRYPTION:
                raise NotImplementedError("strong encryption (flag bit 6)")

            is_encrypted = zinfo.flag_bits & _MASK_ENCRYPTED
            if is_encrypted:
                raise NotImplementedError('no support for encryption')

            file_shared.update_pos_end()
            return file_shared
        except:
            file_shared.close()
            raise

    def __del__(self):
        '''Call the "close()" method in case the user forgot.'''
        self.close()

    def close(self):
        '''Close the file, and for mode 'w', 'x' and 'a' write the ending
        records.'''
        # NOTE: in some __del__ scenarios _file is no longer present
        if not hasattr(self, '_file') or self._file is None:
            return

        file = self._file
        self._file = None
        self._close(file)


    def _close(self, file: tp.IO[bytes]):
        assert self._file_ref_count > 0
        self._file_ref_count -= 1
        if not self._file_ref_count and not self._file_passed:
            file.close()

