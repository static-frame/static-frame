from __future__ import annotations

import gzip
import io
# import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch
from zipfile import ZipFile

import typing_extensions as tp

from static_frame.core.frame import Frame
from static_frame.core.www import WWW
from static_frame.core.www import BytesIOTemporaryFile
from static_frame.core.www import StringIOTemporaryFile
from static_frame.test.test_case import TestCase
from static_frame.test.test_case import temp_file

URL = 'http://foo'

def prepare_mock(mock: MagicMock, content: tp.Union[str, bytes]) -> None:
    if isinstance(content, str):
        payload = io.BytesIO(bytes(content, encoding='utf-8'))
    else:
        payload = io.BytesIO(content)
    cm = MagicMock()
    cm.__enter__.return_value.read = payload.read
    mock.return_value = cm

def load_zip_in_bytes(content: tp.Dict[str, tp.Union[str, bytes]]) -> io.BytesIO:
    archive = io.BytesIO()
    with ZipFile(archive, mode='w') as zf:
        for label, value in content.items():
            if isinstance(value, bytes):
                zf.writestr(label, value)
            else:
                zf.writestr(label, value.encode('utf-8'))
    archive.seek(0)
    return archive

def load_gzip_in_bytes(content: tp.Union[str, bytes]) -> io.BytesIO:
    archive = io.BytesIO()
    with gzip.open(archive, mode='wb') as gz:
        if isinstance(content, bytes):
            gz.write(content)
        else:
            gz.write(content.encode('utf-8'))
    archive.seek(0)
    return archive



class TestUnit(TestCase):

    def test_www_from_file_a(self) -> None:
        with patch('urllib.request.urlopen') as mock:
            prepare_mock(mock, 'foo')

            post: io.StringIO = WWW.from_file(URL, encoding='utf-8', in_memory=True) # type: ignore
            self.assertTrue(isinstance(post, io.StringIO))
            self.assertEqual(post.read(), 'foo')

    def test_www_from_file_b(self) -> None:
        with patch('urllib.request.urlopen') as mock:
            prepare_mock(mock, 'bar')

            post: io.StringIO = WWW.from_file(URL, encoding='utf-8', in_memory=False) # type: ignore
            self.assertTrue(isinstance(post, StringIOTemporaryFile))
            self.assertEqual('bar', post.read())

    def test_www_from_file_c(self) -> None:
        with patch('urllib.request.urlopen') as mock:
            prepare_mock(mock, 'bar')

            post: io.BytesIO = WWW.from_file(URL, encoding=None, in_memory=True) # type: ignore
            self.assertTrue(isinstance(post, io.BytesIO))
            self.assertEqual(post.read(), b'bar')

    def test_www_from_file_d(self) -> None:
        with patch('urllib.request.urlopen') as mock:
            prepare_mock(mock, 'foo')

            post: io.BytesIO = WWW.from_file(URL, encoding=None, in_memory=False) # type: ignore

            self.assertTrue(isinstance(post, BytesIOTemporaryFile))
            self.assertEqual(b'foo', post.read())


    def test_www_from_file_from_delimited_a(self) -> None:
        with patch('urllib.request.urlopen') as mock:
            prepare_mock(mock, 'a,b,c\n1,True,x\n20,False,y\n')

            post = Frame.from_csv(WWW.from_file(URL))
            self.assertEqual(post.to_pairs(),
                    (('a', ((0, 1), (1, 20))), ('b', ((0, True), (1, False))), ('c', ((0, 'x'), (1, 'y')))))

    # def test_url_from_delimited_b(self) -> None:1

    #     url = 'https://stats.govt.nz/assets/Uploads/Business-financial-data/Business-financial-data-June-2022-quarter/Download-data/business-financial-data-june-2022-quarter-csv.zip'

    #     post = URL(url)
    #     import ipdb; ipdb.set_trace()

    #---------------------------------------------------------------------------

    def test_string_io_temp_file_a(self) -> None:
        content = 'foo\nbar'
        with temp_file('.txt') as fp:
            with open(fp, 'w', encoding='utf-8') as f:
                f.write(content)

            siotf = StringIOTemporaryFile(fp, encoding='utf-8')

            self.assertEqual(siotf.read(), content)
            siotf.seek(0)

            self.assertEqual(siotf.readline(), 'foo\n')
            siotf.seek(0)

            self.assertEqual(tuple(siotf), ('foo\n', 'bar'))

            del siotf
            self.assertFalse(os.path.exists(fp))

            # restore file so context manager can clean up
            with open(fp, 'w', encoding='utf-8') as f:
                f.write(content)

    def test_bytes_io_temp_file_a(self) -> None:
        content = b'foo\nbar'

        with temp_file('.txt') as fp:
            with open(fp, 'wb') as f:
                f.write(content)

            siotf = BytesIOTemporaryFile(fp)

            self.assertEqual(siotf.read(), content)
            siotf.seek(0)

            self.assertEqual(siotf.readline(), b'foo\n')
            siotf.seek(0)

            self.assertEqual(tuple(siotf), (b'foo\n', b'bar'))

            del siotf
            self.assertFalse(os.path.exists(fp))

            # restore file so context manager can clean up
            with open(fp, 'wb') as f:
                f.write(content)

    #---------------------------------------------------------------------------

    def test_download_archive_a(self) -> None:
        archive = load_zip_in_bytes({'foo': 'a,b,c\n1,True,x\n2,False,y\n'})

        with patch('urllib.request.urlopen') as mock:
            prepare_mock(mock, archive.read())

            post = WWW._download_archive('http://foo',
                    in_memory=True,
                    buffer_size=1024,
                    extension='.zip')

            archive.seek(0)
            assert isinstance(post, io.BytesIO)
            self.assertEqual(post.read(), archive.read())

    def test_download_archive_b(self) -> None:
        content = b'a,b,c\n1,True,x\n2,False,y\n'
        archive = load_zip_in_bytes({'foo': content})

        with patch('urllib.request.urlopen') as mock:
            prepare_mock(mock, archive.read())

            post = WWW._download_archive(URL,
                    in_memory=False,
                    buffer_size=1024,
                    extension='.zip')
            self.assertIsInstance(post, Path)

            with ZipFile(post) as zf:
                contained = zf.read('foo')
                self.assertEqual(contained, content)

    #---------------------------------------------------------------------------
    def test_www_from_zip_a(self) -> None:
        content = b'a,b,c\n1,True,x\n2,False,y\n'
        archive = load_zip_in_bytes({'foo': content, 'bar': content})

        with patch('urllib.request.urlopen') as mock:
            prepare_mock(mock, archive.read())
            with self.assertRaises(RuntimeError):
                post = WWW.from_zip(URL)

    def test_www_from_zip_b(self) -> None:
        content1 = 'a,b,c\n1,True,x\n2,False,y\n'
        content2 = 'p,q\nTrue,x\nFalse,y\n'
        archive = load_zip_in_bytes({'foo': content1, 'bar': content2})

        with patch('urllib.request.urlopen') as mock:
            prepare_mock(mock, archive.read())
            post: io.StringIO = WWW.from_zip(URL, component='foo') # type: ignore
            self.assertEqual(post.read(), content1)

    def test_www_from_zip_c(self) -> None:
        content2 = 'p,q\nTrue,x\nFalse,y\n'
        archive = load_zip_in_bytes({'bar': content2})

        with patch('urllib.request.urlopen') as mock:
            prepare_mock(mock, archive.read())
            post: io.StringIO = WWW.from_zip(URL) # type: ignore
            self.assertEqual(post.read(), content2)

    def test_www_from_zip_d(self) -> None:
        content2 = b'p,q\nTrue,x\nFalse,y\n'
        archive = load_zip_in_bytes({'bar': content2})

        with patch('urllib.request.urlopen') as mock:
            prepare_mock(mock, archive.read())
            post: io.BytesIO = WWW.from_zip(URL, encoding=None) # type: ignore
            self.assertEqual(post.read(), content2)

    def test_www_from_zip_e(self) -> None:
        content2 = 'p,q\nTrue,x\nFalse,y\n'
        archive = load_zip_in_bytes({'bar': content2})

        with patch('urllib.request.urlopen') as mock:
            prepare_mock(mock, archive.read())

            with temp_file('.txt') as fp:
                with self.assertRaises(RuntimeError):
                    _ = WWW.from_zip(URL, fp=fp, in_memory=True)

                post: Path = WWW.from_zip(URL, fp=fp) # type: ignore
                self.assertEqual(str(post), fp)

                with open(post, encoding='utf-8') as postf:
                    self.assertEqual(postf.read(), content2)

    #---------------------------------------------------------------------------
    def test_www_from_gzip_a(self) -> None:
        content = 'p,q\nTrue,x\nFalse,y\n'
        archive = load_gzip_in_bytes(content)

        with patch('urllib.request.urlopen') as mock:
            prepare_mock(mock, archive.read())
            post: io.StringIO = WWW.from_gzip(URL) # type: ignore
            self.assertEqual(post.read(), content)

    def test_www_from_gzip_b(self) -> None:
        content = b'p,q\nTrue,x\nFalse,y\n'
        archive = load_gzip_in_bytes(content)

        with patch('urllib.request.urlopen') as mock:
            prepare_mock(mock, archive.read())
            post: io.BytesIO = WWW.from_gzip(URL, encoding=None) # type: ignore
            self.assertEqual(post.read(), content)

    def test_www_from_gzip_c(self) -> None:
        content = 'p,q\nTrue,x\nFalse,y\n'
        archive = load_gzip_in_bytes(content)

        with patch('urllib.request.urlopen') as mock:
            prepare_mock(mock, archive.read())

            with temp_file('.txt') as fp:
                post = WWW.from_gzip(URL, fp=fp)
                self.assertEqual(str(post), fp)

                with open(post, encoding='utf-8') as postf:
                    self.assertEqual(postf.read(), content)

    #---------------------------------------------------------------------------
    def test_frame_from_json_a(self) -> None:

        content = '''[
        {
        "userId": 1,
        "id": 1,
        "title": "delectus aut autem",
        "completed": false
        },
        {
        "userId": 1,
        "id": 2,
        "title": "quis ut nam facilis et officia qui",
        "completed": false
        }]'''
        with patch('urllib.request.urlopen') as mock:
            prepare_mock(mock, content)
            post = Frame.from_json_records(WWW.from_file(url=URL))
            self.assertEqual(post.shape, (2, 4))
            self.assertEqual([dt.kind for dt in post.dtypes.values], ['i', 'i', 'U', 'b'])


    #---------------------------------------------------------------------------
    def test_www_url_prepare_a(self) -> None:
        self.assertEqual(
                WWW._url_prepare('https://app.foo.com/client/T1H60/C062S7'),
                'https://app.foo.com/client/T1H60/C062S7'
                )

    def test_www_url_prepare_b(self) -> None:
        self.assertEqual(
                WWW._url_prepare('https://app.foo.com/path to/spaced files'),
                'https://app.foo.com/path%20to/spaced%20files'
                )

    def test_www_url_prepare_c(self) -> None:
        self.assertEqual(
                WWW._url_prepare('https://www.example.com/search?query=hello+world&sort=desc'),
                'https://www.example.com/search?query=hello+world&sort=desc'
                )

    def test_www_url_prepare_d(self) -> None:
        self.assertEqual(
                WWW._url_prepare('https://0cx70.execute-api.qs.as.com/stage/api/foo/RM nc QB/tyPa/data?startRow=0&endRow=20000&format=list'),
                'https://0cx70.execute-api.qs.as.com/stage/api/foo/RM%20nc%20QB/tyPa/data?startRow=0&endRow=20000&format=list'
                )


if __name__ == '__main__':
    unittest.main()
