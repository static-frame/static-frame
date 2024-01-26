from __future__ import annotations

import io
import os
from os import PathLike
from struct import calcsize
from struct import error as StructError
from struct import unpack
from types import TracebackType
from zipfile import ZIP_STORED
from zipfile import BadZipFile

import typing_extensions as tp

# Optimized reader of uncompressed ZIP files. Based largely on CPython, Lib/zipfile/__init__.py. This ZIP reader removes CRC checking as well as file locks around in the object returned from open(). This is deemed acceptable as this is only used with NPZ files, which are not compressed, are read in a single thread, and are often bundled in (an outer) ZIP archives, such as those produced by Bus.to_zip_npz(). When unpacking such ZIP archives of NPZ, compression is still supported and CRC checking is performed. If the standard ZipFile reader is used on such a ZIP NPZ, CRC checking would actually be done twice, as the full bytes for the file are read into a BytesIO object and use to create new ZipFile instance for loading as an NPZ.

#-------------------------------------------------------------------------------

# Below are some formats and associated data for reading/writing headers using
# the struct module.  The names and structures of headers/records are those used
# in the PKWARE description of the ZIP file format:
#     http://www.pkware.com/documents/casestudies/APPNOTE.TXT
# (URL valid as of January 2008)

# The "end of central directory" structure, magic number, size, and indices
# (section V.I in the format document)
_END_ARCHIVE_STRUCT = b"<4s4H2LH"
_END_ARCHIVE_STRING = b"PK\005\006"
_END_ARCHIVE_SIZE = calcsize(_END_ARCHIVE_STRUCT)

_ECD_SIGNATURE = 0
_ECD_DISK_NUMBER = 1
_ECD_DISK_START = 2
_ECD_ENTRIES_THIS_DISK = 3
_ECD_ENTRIES_TOTAL = 4
_ECD_SIZE = 5
_ECD_OFFSET = 6
_ECD_COMMENT_SIZE = 7
_ECD_COMMENT = 8
_ECD_LOCATION = 9

# The "central directory" structure, magic number, size, and indices of entries in the structure (section V.F in the format document)
_CENTRAL_DIR_STRUCT = "<4s4B4HL2L5H2L"
_CENTRAL_DIR_STRING = b"PK\001\002"
_CENTRAL_DIR_SIZE = calcsize(_CENTRAL_DIR_STRUCT)

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
_MASK_ENCRYPTED = 1 << 0
_MASK_COMPRESSED_PATCH = 1 << 5
_MASK_STRONG_ENCRYPTION = 1 << 6
_MASK_UTF_FILENAME = 1 << 11

# The "local file header" structure, magic number, size, and indices (section V.A in the format document)
_FILE_HEADER_STRUCT = "<4s2B4HL2L2H"
_FILE_HEADER_STRING = b"PK\003\004"
_FILE_HEADER_SIZE = calcsize(_FILE_HEADER_STRUCT)

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
_END_ARCHIVE64_LOCATOR_SIZE = calcsize(_END_ARCHIVE64_LOCATOR_STRUCT)
# The "Zip64 end of central directory" record, magic number, size, and indices (section V.G in the format document)
_END_ARCHIVE64_STRUCT = "<4sQ2H2L4Q"
_END_ARCHIVE64_STRING = b"PK\x06\x06"
_END_ARCHIVE64_SIZE = calcsize(_END_ARCHIVE64_STRUCT)

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

#-------------------------------------------------------------------------------

TEndArchive = tp.List[tp.Union[bytes, int]]

def _end_archive64_update(
        fpin: tp.IO[bytes],
        offset: int,
        endrec: TEndArchive,
        ) -> TEndArchive:
    '''
    Read the ZIP64 end-of-archive records and use that to update endrec
    '''
    try:
        fpin.seek(offset - _END_ARCHIVE64_LOCATOR_SIZE, 2)
    except OSError: #pragma: no cover
        # If the seek fails, the file is not large enough to contain a ZIP64
        # end-of-archive record, so just return the end record we were given.
        return endrec #pragma: no cover

    data = fpin.read(_END_ARCHIVE64_LOCATOR_SIZE)
    if len(data) != _END_ARCHIVE64_LOCATOR_SIZE:
        return endrec #pragma: no cover

    sig, diskno, reloff, disks = unpack(_END_ARCHIVE64_LOCATOR_STRUCT, data)
    if sig != _END_ARCHIVE64_LOCATOR_STRING:
        return endrec #pragma: no cover

    if diskno != 0 or disks > 1:
        raise BadZipFile("zipfiles that span multiple disks are not supported") #pragma: no cover

    # Assume no 'zip64 extensible data'
    fpin.seek(offset - _END_ARCHIVE64_LOCATOR_SIZE - _END_ARCHIVE64_SIZE, 2)
    data = fpin.read(_END_ARCHIVE64_SIZE)
    if len(data) != _END_ARCHIVE64_SIZE:
        return endrec #pragma: no cover

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
    ) = unpack(_END_ARCHIVE64_STRUCT, data)

    if sig != _END_ARCHIVE64_STRING:
        return endrec #pragma: no cover

    # Update the original endrec using data from the ZIP64 record
    endrec[_ECD_SIGNATURE] = sig
    endrec[_ECD_DISK_NUMBER] = disk_num
    endrec[_ECD_DISK_START] = disk_dir
    endrec[_ECD_ENTRIES_THIS_DISK] = dircount
    endrec[_ECD_ENTRIES_TOTAL] = dircount2
    endrec[_ECD_SIZE] = dirsize
    endrec[_ECD_OFFSET] = diroffset
    return endrec


def _extract_end_archive(file: tp.IO[bytes]) -> TEndArchive:
    '''Return data from the "End of Central Directory" record, or None.

    The data is a list of the nine items in the ZIP "End of central dir"
    record followed by a tenth item, the file seek offset of this record.'''

    # Determine file size
    file.seek(0, 2) # seek to end
    filesize = file.tell()

    # Check to see if this is ZIP file with no archive comment (the
    # "end of central directory" structure should be the last item in the
    # file if this is the case).
    try:
        file.seek(-_END_ARCHIVE_SIZE, 2)
    except OSError as e: #pragma: no cover
        raise BadZipFile('Unable to find a valid end of central directory structure') from e #pragma: no cover

    endrec: TEndArchive
    data = file.read()
    if (len(data) == _END_ARCHIVE_SIZE and
            data[0:4] == _END_ARCHIVE_STRING and
            data[-2:] == b"\000\000"):
        # the signature is correct and there's no comment, unpack structure
        endrec = list(unpack(_END_ARCHIVE_STRUCT, data))
        # Append a blank comment and record start offset
        endrec.append(b"")
        endrec.append(filesize - _END_ARCHIVE_SIZE)
        # Try to read the "Zip64 end of central directory" structure
        return _end_archive64_update(file, -_END_ARCHIVE_SIZE, endrec)

    # Either this is not a ZIP file, or it is a ZIP file with an archive   comment.  Search the end of the file for the "end of central directory" record signature. The comment is the last item in the ZIP file and may be up to 64K long.  It is assumed that the "end of central directory" magic number does not appear in the comment.
    comment_max_start = max(filesize - (1 << 16) - _END_ARCHIVE_SIZE, 0)
    file.seek(comment_max_start, 0)

    data = file.read()
    start = data.rfind(_END_ARCHIVE_STRING)
    if start >= 0:
        # found the magic number; attempt to unpack and interpret
        data = data[start: start + _END_ARCHIVE_SIZE]
        if len(data) != _END_ARCHIVE_SIZE:
            raise BadZipFile('Corrupted ZIP.') #pragma: no cover

        endrec = list(unpack(_END_ARCHIVE_STRUCT, data))
        endrec.append(b'') # ignore comment
        endrec.append(comment_max_start + start)

        # Try to read the "Zip64 end of central directory" structure
        return _end_archive64_update(file,
                comment_max_start + start - filesize,
                endrec,
                )
    raise BadZipFile('Unable to find a valid end of central directory structure') #pragma: no cover

#-------------------------------------------------------------------------------

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

class ZipFilePartRO(io.BufferedIOBase):
    '''
    This wrapper around an IO bytes stream takes a close function at initialization, and exclusively uses that on close() with the composed file instance.
    '''
    __slots__ = (
            '_file',
            '_pos',
            '_close',
            '_file_size',
            '_pos_end',
            )

    def __init__(self,
            file: tp.IO[bytes],
            close: tp.Callable[..., None],
            zinfo: ZipInfoRO,
            ) -> None:
        '''
        Args:
            pos: the start position, just after the header
        '''
        self._file: tp.IO[bytes] | None = file
        self._pos = + zinfo.header_offset
        self._close = close # callable
        self._file_size = zinfo.file_size # main data size after header
        self._pos_end = -1 # self._pos + zinfo.file_size

    def __enter__(self) -> tp.Self:
        return self

    def __exit__(self,
            type: tp.Type[BaseException] | None,
            value: BaseException | None,
            traceback: TracebackType | None,
            ) -> None:
        self.close()

    def seekable(self) -> bool:
        if self._file is None:
            raise ValueError("I/O operation on closed file.")
        return self._file.seekable()

    def update_pos_end(self) -> None:
        assert self._pos_end < 0 # only allow once
        self._pos_end = self._pos + self._file_size

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if self._file is None:
            raise ValueError("I/O operation on closed file.")
        if whence != 1:
            # NOTE:seeking permits moving along the complete zip file; limit to relative seeks
            raise NotImplementedError('start- or end-relative seeks are not permitted.')
        self._file.seek(self._pos)
        self._file.seek(offset, whence)
        self._pos = self._file.tell()
        return self._pos

    def read(self, n: int | None = -1) -> bytes:
        if self._file is None:
            raise ValueError("I/O operation on closed file.")

        self._file.seek(self._pos)

        if n is None or n < 0:
            assert self._pos_end >= 0
            n_read = self._pos_end - self._pos
        else:
            n_read = n

        data = self._file.read(n_read)
        self._pos = self._file.tell()
        return data

    def readinto(self, buffer: tp.Buffer) -> int:
        if self._file is None:
            raise ValueError("I/O operation on closed file.")

        self._file.seek(self._pos)
        count: int = self._file.readinto(buffer) # type: ignore
        self._pos = self._file.tell()
        return count

    def close(self) -> None:
        if self._file is not None:
            file = self._file
            self._file = None
            self._close(file)

    def write(self, data: tp.Buffer, /) -> int:
        raise NotImplementedError() #pragma: no cover


#-------------------------------------------------------------------------------

@tp.overload
def yield_zinfos(
        file: tp.IO[bytes],
        filename_only: tp.Literal[True],
        ) -> tp.Iterator[str]: ...

@tp.overload
def yield_zinfos(
        file: tp.IO[bytes],
        filename_only: tp.Literal[False],
        ) -> tp.Iterator[ZipInfoRO]: ...

def yield_zinfos(
        file: tp.IO[bytes],
        filename_only: bool,
        ) -> tp.Iterator[ZipInfoRO | str]:
    '''Read in the table of contents for the ZIP file.'''
    try:
        endrec: TEndArchive = _extract_end_archive(file)
    except OSError as e: #pragma: no cover
        raise BadZipFile("File is not a zip file") from e #pragma: no cover

    size_cd: int = endrec[_ECD_SIZE] # type: ignore[assignment]
    offset_cd: int = endrec[_ECD_OFFSET] # type: ignore[assignment]
    end_location: int = endrec[_ECD_LOCATION] # type: ignore[assignment]
    # "concat" is zero, unless zip was concatenated to another file
    concat: int = end_location - size_cd - offset_cd
    if endrec[_ECD_SIGNATURE] == _END_ARCHIVE64_STRING:
        # If Zip64 extension structures are present, account for them
        concat -= (_END_ARCHIVE64_SIZE + _END_ARCHIVE64_LOCATOR_SIZE)

    start_cd = offset_cd + concat # Position of start of central directory
    if start_cd < 0:
        raise BadZipFile("Bad offset for central directory") #pragma: no cover

    file.seek(start_cd, 0)
    file_cd = io.BytesIO(file.read(size_cd))

    total = 0
    filename_length = 0
    extra_length = 0
    comment_length = 0

    while total < size_cd:
        cdir_size = file_cd.read(_CENTRAL_DIR_SIZE)
        if len(cdir_size) != _CENTRAL_DIR_SIZE:
            raise BadZipFile("Truncated central directory") #pragma: no cover

        cdir = unpack(_CENTRAL_DIR_STRUCT, cdir_size)
        if cdir[_CD_SIGNATURE] != _CENTRAL_DIR_STRING:
            raise BadZipFile("Bad magic number for central directory") #pragma: no cover

        if not filename_only and cdir[_CD_COMPRESS_TYPE] != ZIP_STORED:
            raise BadZipFile("Cannot process compressed zips")

        filename_length = cdir[_CD_FILENAME_LENGTH]
        flags = cdir[_CD_FLAG_BITS]

        filename_bytes = file_cd.read(filename_length)
        # check for UTF-8 file name extension, otherweise use historical ZIP filename encoding
        filename = filename_bytes.decode('utf-8' if (flags & _MASK_UTF_FILENAME) else 'cp437')

        extra_length = cdir[_CD_EXTRA_FIELD_LENGTH]
        comment_length = cdir[_CD_COMMENT_LENGTH]

        if filename_only:
            yield filename
            file_cd.seek(extra_length + comment_length, 1)
        else:
            zinfo = ZipInfoRO(filename)
            # these might be updated if extra exists
            header_offset = cdir[_CD_LOCAL_HEADER_OFFSET]
            file_size = cdir[_CD_UNCOMPRESSED_SIZE]
            compressed_size = cdir[_CD_COMPRESSED_SIZE]

            extra = file_cd.read(extra_length)

            # read extra data for ZIP64 adjustments to capped sizes
            while len(extra) >= 4:
                tp, ln = unpack('<HH', extra[:4])
                if ln + 4 > len(extra):
                    raise BadZipFile(f'Corrupt extra field {tp:04x} (size={ln})') #pragma: no cover
                if tp == 1:
                    ep = extra[4: ln + 4]
                    try:
                        if file_size in (0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF):
                            [file_size] = unpack('<Q', ep[:8])
                            ep = ep[8:]
                        if compressed_size == 0xFFFF_FFFF:
                            [compressed_size] = unpack('<Q', ep[:8])
                            ep = ep[8:]
                        if header_offset == 0xFFFF_FFFF:
                            [header_offset] = unpack('<Q', ep[:8])
                    except StructError: #pragma: no cover
                        raise BadZipFile('Corrupt zip64 extra field.') from None #pragma: no cover
                extra = extra[ln + 4:]

            file_cd.seek(comment_length, 1)

            zinfo.header_offset = header_offset + concat
            zinfo.flag_bits = flags
            zinfo.file_size = file_size
            yield zinfo

        total = (
                total +
                _CENTRAL_DIR_SIZE +
                filename_length +
                extra_length +
                comment_length
                )


class ZipFileRO:

    __slots__ = (
        '_name_to_info',
        '_file_passed',
        '_file_name',
        '_file',
        '_file_ref_count',
        )

    def __init__(self, file: PathLike[str] | str | tp.IO[bytes]) -> None:
        '''Open the ZIP file with mode read 'r', write 'w', exclusive create 'x',
        or append 'a'.'''
        if isinstance(file, os.PathLike):
            file = os.fspath(file)

        self._file: tp.IO[bytes] | None

        if isinstance(file, str):
            self._file_passed = False
            self._file_name = file
            self._file = io.open(file, 'rb')  #pylint: disable=R1732
        else:
            self._file_passed = True
            self._file = file
            self._file_name = getattr(file, 'name', '')

        assert self._file is not None
        self._file_ref_count = 1

        try:
            self._name_to_info = {
                    zinfo.filename: zinfo for zinfo in yield_zinfos(self._file, False)
                    }
        except BadZipFile:
            fp = self._file
            self._file = None
            self._close(fp)
            raise

    def __enter__(self) -> tp.Self:
        return self

    def __exit__(self,
            type: tp.Type[BaseException],
            value: BaseException,
            traceback: TracebackType,
            ) -> None:
        self.close()

    def __repr__(self) -> str:
        result = [f'<{self.__class__.__name__}']
        if self._file is not None:
            if self._file_passed:
                result.append(f' file={self._file!r}')
            elif self._file_name:
                result.append(f' filename={self._file_name!r}')
        else:
            result.append(' [closed]')
        result.append('>')
        return ''.join(result)

    def namelist(self) -> tp.List[str]:
        '''Return a list of file names in the archive.'''
        # return [data.filename for data in self.filelist]
        return list(self._name_to_info.keys())

    def infolist(self) -> tp.List[ZipInfoRO]:
        '''Return a list of class ZipInfoRO instances for files in the
        archive.'''
        return list(self._name_to_info.values())

    def __len__(self) -> int:
        return len(self._name_to_info)

    def getinfo(self, name: str) -> ZipInfoRO:
        '''Return the instance of ZipInfoRO given 'name'.'''
        if not (zinfo := self._name_to_info.get(name)):
            raise KeyError(f'There is no item named {name!r} in the archive')
        return zinfo

    def writestr(self, name: str, data: str) -> None:
        raise NotImplementedError() #pragma: no cover

    def read(self, name: str) -> bytes:
        '''Return file bytes for name.'''
        with self.open(name) as file:
            return file.read()

    def open(self, name: str) -> tp.IO[bytes]:
        '''Return file-like object for 'name'.

        name is a string for the file name within the ZIP file
        '''
        if not self._file:
            raise ValueError("Attempt to use ZIP archive that was already closed")

        zinfo = self.getinfo(name)

        self._file_ref_count += 1

        file_shared = ZipFilePartRO(
                self._file,
                self._close,
                zinfo,
                )
        try:
            fheader_bytes = file_shared.read(_FILE_HEADER_SIZE)
            if len(fheader_bytes) != _FILE_HEADER_SIZE:
                raise BadZipFile("Truncated file header") #pragma: no cover

            fheader = unpack(_FILE_HEADER_STRUCT, fheader_bytes)
            if fheader[_FH_SIGNATURE] != _FILE_HEADER_STRING:
                raise BadZipFile("Bad magic number for file header") #pragma: no cover

            file_shared.seek(fheader[_FH_FILENAME_LENGTH], 1)

            if fheader[_FH_EXTRA_FIELD_LENGTH]:
                file_shared.seek(fheader[_FH_EXTRA_FIELD_LENGTH], 1)

            if zinfo.flag_bits & _MASK_COMPRESSED_PATCH: # Zip 2.7: compressed patched data
                raise NotImplementedError("compressed patched data (flag bit 5)") #pragma: no cover
            if zinfo.flag_bits & _MASK_STRONG_ENCRYPTION:
                raise NotImplementedError("strong encryption (flag bit 6)") #pragma: no cover

            is_encrypted = zinfo.flag_bits & _MASK_ENCRYPTED
            if is_encrypted:
                raise NotImplementedError('no support for encryption') #pragma: no cover

            file_shared.update_pos_end()
            return file_shared # type: ignore # could cast
        except: #pragma: no cover
            file_shared.close() #pragma: no cover
            raise #pragma: no cover

    def __del__(self) -> None:
        '''Call the "close()" method in case the user forgot.'''
        self.close()

    def close(self) -> None:
        # NOTE: in some __del__ scenarios _file is no longer present
        if not hasattr(self, '_file') or self._file is None:
            return

        file = self._file
        self._file = None
        self._close(file)

    def _close(self, file: tp.IO[bytes]) -> None:
        '''Close function passed on to ZipFilePartRO instances from open()
        '''
        assert self._file_ref_count > 0
        self._file_ref_count -= 1
        if not self._file_ref_count and not self._file_passed:
            file.close()


def zip_namelist(fp: PathLike[str] | str) -> tp.Iterator[str]:
    '''High-performance routine to list the contents of a zip. This will work with both compressed and uncompressed zips.
    '''
    with open(fp, 'rb') as file:  #pylint: disable=R1732
        yield from yield_zinfos(file, True)
