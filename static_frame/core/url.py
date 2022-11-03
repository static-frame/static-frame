import os
import tempfile
import typing as tp
from io import BytesIO
from io import IOBase
from io import StringIO
from pathlib import Path
from types import TracebackType
from urllib import request

from zipfile import ZipFile


class StringIOTemporaryFile(StringIO):
    '''Subclass of a StringIO that reads from a managed file that is deleted when this instance goes out of scope.
    '''

    def __init__(self, fp: Path) -> None:
        self._fp = fp
        self._file = open(fp, 'r')
        super().__init__()

    def __del__(self) -> None:
        self._file.close()
        os.unlink(self._fp)
        super().__del__()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._file.seek(offset)

    def read(self, size: tp.Optional[int] =-1) -> str:
        return self._file.read(size)

    def readline(self, size: tp.Optional[int] = -1) -> str: # type: ignore
        return self._file.readline(size)

    def __iter__(self) -> tp.Iterator[str]: # type: ignore
        return self._file.__iter__()

class BytesIOTemporaryFile(BytesIO):
    '''Subclass of a BytesIO that reads from a managed file that is deleted when this instance goes out of scope.
    '''

    def __init__(self, fp: Path) -> None:
        self._fp = fp
        self._file = open(fp, 'rb')
        super().__init__()

    def __del__(self) -> None:
        self._file.close()
        os.unlink(self._fp)
        super().__del__()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._file.seek(offset)

    def read(self, size: tp.Optional[int] = -1) -> bytes:
        return self._file.read(size)

    def readline(self, size: tp.Optional[int] = -1) -> bytes:
        return self._file.readline(size)

    def __iter__(self) -> tp.Iterator[bytes]:
        return self._file.__iter__()


class MaybeTemporaryFile:
    '''Provide one context manager that, if an `fp` is given, work as a normal file; if no `fp` is given, produce a temporary file.
    '''
    def __init__(self, fp: tp.Optional[Path], mode: str):

        if fp:
            self._f = open(fp, mode=mode)
        else:
            self._f = tempfile.NamedTemporaryFile(mode=mode,
                suffix=None,
                delete=False,
                )

    def __enter__(self) -> tp.IO[tp.Any]:
        return self._f.__enter__()

    def __exit__(self,
            type: tp.Type[BaseException],
            value: BaseException,
            traceback: TracebackType,
            ) -> None:
        self._f.__exit__(type, value, traceback)


def url_adapter_file(
        url: tp.Union[str, request.Request],
        encoding: tp.Optional[str] = 'utf-8',
        in_memory: bool = True,
        buffer_size: int = 8192,
        fp: tp.Optional[Path] = None,
        ) -> tp.Union[Path, StringIO, BytesIO]:

    with request.urlopen(url) as response:
        if in_memory:
            if encoding:
                return StringIO(response.read().decode(encoding))
            else:
                return BytesIO(response.read())

        # not in-memory, write a file
        with MaybeTemporaryFile(fp=fp, mode='w' if encoding else 'wb') as f:
            fp_written = Path(f.name)
            if encoding:
                extract = lambda: response.read(buffer_size).decode(encoding)
            else:
                extract = lambda: response.read(buffer_size)

            while True:
                b = extract() # type: ignore
                if b:
                    f.write(b)
                else:
                    break
            if fp:
                return fp_written
            if encoding:
                return StringIOTemporaryFile(fp_written)
            return BytesIOTemporaryFile(fp_written)


def url_adapter_zip(
        url: tp.Union[str, request.Request],
        encoding: tp.Optional[str] = 'utf-8',
        in_memory: bool = True,
        buffer_size: int = 8192,
        fp: tp.Optional[Path] = None,
        ) -> tp.Union[Path, StringIO, BytesIO]:

    archive: tp.Union[Path, BytesIO]

    with request.urlopen(url) as response:
        if in_memory:
            archive = BytesIO(response.read())
        else:
            with tempfile.NamedTemporaryFile(mode='wb',
                    suffix='zip',
                    delete=False,
                    ) as f:
                archive = Path(f.name)
                while True:
                    b = response.read(buffer_size)
                    if b:
                        f.write(b)
                    else:
                        break

    with ZipFile(archive) as zf:
        names = zf.namelist()
        if len(names) > 1:
            raise RuntimeError(f'more than one file found in zip archive: {names}')
        name = names.pop()
        data = zf.read(name)

    if in_memory:
        if encoding:
            return StringIO(data.decode(encoding))
        else:
            return BytesIO(data)

    # not in-memory, write a file, delete archive
    os.unlink(archive)

    with MaybeTemporaryFile(fp=fp, mode='w' if encoding else 'wb') as f:
        fp_written = Path(f.name)
        if encoding:
            f.write(data.decode(encoding))
        else:
            f.write(data)

        if fp:
            return fp_written
        if encoding:
            return StringIOTemporaryFile(fp_written)
        return BytesIOTemporaryFile(fp_written)


def URL(url: tp.Union[str, request.Request],
        *,
        encoding: tp.Optional[str] = 'utf-8',
        in_memory: bool = True,
        buffer_size: int = 8192,
        unzip: bool = True,
        fp: tp.Optional[tp.Union[Path, str]] = None,
        ) -> tp.Union[Path, StringIO, BytesIO]:
    '''
    Args:
        encoding: Defaults to UTF-8; if None, binary data is collected.
        in_memory: if True, data is loaded into memory; if False, a temporary file is written.
    '''
    if fp is not None:
        if in_memory:
            raise RuntimeError('If supplying a fp set in_memory to False')
        if isinstance(fp, str):
            fp = Path(fp)

    if url.endswith('.zip') and unzip:
        return url_adapter_zip(url, encoding, in_memory, buffer_size, fp)
    return url_adapter_file(url, encoding, in_memory, buffer_size, fp)



