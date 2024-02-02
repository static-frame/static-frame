from __future__ import annotations

import gzip
import os
import tempfile
from io import BytesIO
from io import StringIO
from pathlib import Path
from types import TracebackType
from urllib import request
from urllib.parse import quote
from urllib.parse import urlparse
from urllib.parse import urlunparse
from zipfile import ZipFile

import typing_extensions as tp

from static_frame.core.doc_str import doc_inject


class StringIOTemporaryFile(StringIO):
    '''Subclass of a StringIO that reads from a managed file that is deleted when this instance goes out of scope.
    '''

    def __init__(self, fp: Path, encoding: str) -> None:
        self._fp = fp
        self._file = open(fp, 'r', encoding=encoding) # pylint: disable=R1732
        super().__init__()

    def __del__(self) -> None:
        self._file.close()
        os.unlink(self._fp)
        super().__del__()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._file.seek(offset, whence)

    def read(self, size: tp.Optional[int] =-1) -> str:
        return self._file.read(size)

    def readline(self, size: int = -1) -> str: # type: ignore
        return self._file.readline(size)

    def __iter__(self) -> tp.Iterator[str]: # type: ignore
        return self._file.__iter__()

class BytesIOTemporaryFile(BytesIO):
    '''Subclass of a BytesIO that reads from a managed file that is deleted when this instance goes out of scope.
    '''

    def __init__(self, fp: Path) -> None:
        self._fp = fp
        self._file = open(fp, 'rb') # pylint: disable=R1732
        super().__init__()

    def __del__(self) -> None:
        self._file.close()
        os.unlink(self._fp)
        super().__del__()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._file.seek(offset, whence)

    def read(self, size: tp.Optional[int] = -1) -> bytes:
        return self._file.read(size)

    def readline(self, size: tp.Optional[int] = -1) -> bytes:
        return self._file.readline(size)

    def __iter__(self) -> tp.Iterator[bytes]:
        return self._file.__iter__()

#-------------------------------------------------------------------------------

class MaybeTemporaryFile:
    '''Provide one context manager that, if an `fp` is given, works as a normal file; if no `fp` is given, produce a temporary file.
    '''
    def __init__(self, fp: tp.Optional[Path], mode: str, encoding: str):

        if fp:
            self._f = open(fp, mode=mode, encoding=encoding) # pylint: disable=R1732
        else:
            self._f = tempfile.NamedTemporaryFile(mode=mode, # pylint: disable=R1732
                suffix=None,
                delete=False,
                encoding=encoding,
                )

    def __enter__(self) -> tp.IO[tp.Any]:
        return self._f.__enter__()

    def __exit__(self,
            type: tp.Type[BaseException],
            value: BaseException,
            traceback: TracebackType,
            ) -> None:
        self._f.__exit__(type, value, traceback)


#-------------------------------------------------------------------------------

WWWReturnType = tp.Union[Path, StringIO, BytesIO]

class WWW:
    '''Utilities for downloading resources from the world-wide-web.
    '''
    __slots__ = ()


    @staticmethod
    def _url_prepare(url: str) -> str:
        '''Remove leading trailing white space, quote the path component to handle spaces. This does not handling spaces in queries
        '''
        url_parts = urlparse(url.strip())
        return urlunparse(
                url_parts._replace(path=quote(url_parts.path))
                )

    @classmethod
    def _download_archive(cls,
            url: tp.Union[str, request.Request],
            in_memory: bool,
            buffer_size: int,
            extension: str,
            ) -> tp.Union[Path, BytesIO]:
        archive: tp.Union[Path, BytesIO]

        if isinstance(url, str):
            url = cls._url_prepare(url)

        with request.urlopen(url) as response:
            if in_memory:
                archive = BytesIO(response.read())
            else:
                with tempfile.NamedTemporaryFile(mode='wb',
                        suffix=extension,
                        delete=False,
                        ) as f:
                    archive = Path(f.name)
                    while True:
                        b = response.read(buffer_size)
                        if b:
                            f.write(b)
                        else:
                            break
        return archive

    @staticmethod
    def _resolve_fp_and_in_memory(
            in_memory: tp.Optional[bool],
            fp: tp.Optional[tp.Union[Path, str]] = None,
            ) -> tp.Tuple[bool, tp.Optional[Path]]:
        '''
        If an fp is given and in_memory is True, error; else, in_memory is set to False; if an fp is not given and in_memory is None, default to True, else use in_memory.
        '''
        if fp is not None:
            if in_memory is True:
                raise RuntimeError('If supplying an `fp`, `in_memory` cannot be True.')
            in_memory = False
            if isinstance(fp, str): # just to pass Path
                fp = Path(fp)
        else:
            in_memory = True if in_memory is None else in_memory
        return in_memory, fp

    @staticmethod
    def _write_maybe_temporary(
            fp: tp.Optional[Path],
            encoding: str,
            extractor: tp.Callable[[], tp.Union[str, bytes]]
            ) -> WWWReturnType:
        with MaybeTemporaryFile(fp=fp,
                mode='w' if encoding else 'wb',
                encoding=encoding,
                ) as f:
            fp_written = Path(f.name)

            while True: # can use iter() function with for
                b = extractor()
                if b:
                    f.write(b)
                else:
                    break
            if fp:
                return fp_written
            if encoding:
                return StringIOTemporaryFile(fp_written, encoding=encoding)
            return BytesIOTemporaryFile(fp_written)

    #---------------------------------------------------------------------------
    @classmethod
    @doc_inject(selector='www')
    def from_file(cls,
            url: tp.Union[str, request.Request],
            *,
            encoding: str = 'utf-8',
            in_memory: tp.Optional[bool] = None,
            buffer_size: int = 8192,
            fp: tp.Optional[tp.Union[Path, str]] = None,
            ) -> WWWReturnType:
        '''
        {doc}

        Args:
            {url}
            {encoding}
            {in_memory}
            {buffer_size}
            {fp}
        '''
        in_memory, fp = cls._resolve_fp_and_in_memory(in_memory, fp)

        with request.urlopen(url) as response:
            if in_memory:
                if encoding:
                    return StringIO(response.read().decode(encoding))
                else:
                    return BytesIO(response.read())

            if encoding:
                extractor = lambda: response.read(buffer_size).decode(encoding)
            else:
                extractor = lambda: response.read(buffer_size)

            return cls._write_maybe_temporary(
                    fp=fp,
                    encoding=encoding,
                    extractor=extractor,
                    )


    @classmethod
    @doc_inject(selector='www')
    def from_zip(cls,
            url: tp.Union[str, request.Request],
            *,
            encoding: str = 'utf-8',
            in_memory: tp.Optional[bool] = None,
            buffer_size: int = 8192,
            fp: tp.Optional[tp.Union[Path, str]] = None,
            component: tp.Optional[str] = None,
            ) -> WWWReturnType:
        '''
        {doc}

        Args:
            {url}
            {encoding}
            {in_memory}
            {buffer_size}
            {fp}
            {component}
        '''
        in_memory, fp = cls._resolve_fp_and_in_memory(in_memory, fp)

        archive = cls._download_archive(url=url,
                in_memory=in_memory,
                buffer_size=buffer_size,
                extension='.zip',
                )

        with ZipFile(archive) as zf:
            names = zf.namelist()
            if component:
                name = component
            else:
                if len(names) > 1:
                    samples = ', '.join(names[:20])
                    etc = '...' if len(samples) > 20 else ''
                    raise RuntimeError(f'More than one file found in zip archive ({samples}{etc}); name a single file with the `component` argument.')
                name = names.pop()
            data = zf.read(name)

        data_io: tp.Union[StringIO, BytesIO]
        if encoding:
            data_io = StringIO(data.decode(encoding))
        else:
            data_io = BytesIO(data)

        if in_memory:
            return data_io

        # not in-memory, write a file, delete archive
        os.unlink(archive) # type: ignore
        return cls._write_maybe_temporary(
                fp=fp,
                encoding=encoding,
                extractor=data_io.read,
                )


    @classmethod
    @doc_inject(selector='www')
    def from_gzip(cls,
            url: tp.Union[str, request.Request],
            *,
            encoding: str = 'utf-8',
            in_memory: tp.Optional[bool] = None,
            buffer_size: int = 8192,
            fp: tp.Optional[tp.Union[Path, str]] = None,
            ) -> WWWReturnType:
        '''
        {doc}

        Args:
            {url}
            {encoding}
            {in_memory}
            {buffer_size}
            {fp}
        '''
        in_memory, fp = cls._resolve_fp_and_in_memory(in_memory, fp)

        archive = cls._download_archive(url=url,
                in_memory=in_memory,
                buffer_size=buffer_size,
                extension='.gzip',
                )

        with gzip.open(archive) as gz:
            data = gz.read()

        data_io: tp.Union[StringIO, BytesIO]
        if encoding:
            data_io = StringIO(data.decode(encoding))
        else:
            data_io = BytesIO(data)

        if in_memory:
            return data_io

        # not in-memory, write a file, delete archive
        os.unlink(archive) # type: ignore
        return cls._write_maybe_temporary(
                fp=fp,
                encoding=encoding,
                extractor=data_io.read,
                )
