import io
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from static_frame.core.frame import Frame
from static_frame.core.url import URL
from static_frame.core.url import BytesIOTemporaryFile
from static_frame.core.url import StringIOTemporaryFile
from static_frame.test.test_case import TestCase
from static_frame.test.test_case import temp_file

url = 'http://foo'

def prepare_mock(mock: MagicMock, content: str) -> None:
    payload = io.BytesIO(bytes(content, encoding='utf-8'))
    cm = MagicMock()
    cm.__enter__.return_value.read = payload.read
    mock.return_value = cm

class TestUnit(TestCase):

    def test_url_a(self) -> None:
        with patch('urllib.request.urlopen') as mock:
            prepare_mock(mock, 'foo')

            post = URL(url, encoding='utf-8', in_memory=True)
            self.assertTrue(isinstance(post, io.StringIO))
            self.assertEqual(post.read(), 'foo') # type: ignore

    def test_url_b(self) -> None:
        with patch('urllib.request.urlopen') as mock:
            prepare_mock(mock, 'bar')

            post = URL(url, encoding='utf-8', in_memory=False)
            self.assertTrue(isinstance(post, StringIOTemporaryFile))
            self.assertEqual('bar', post.read()) # type: ignore

    def test_url_c(self) -> None:
        with patch('urllib.request.urlopen') as mock:
            prepare_mock(mock, 'bar')

            post = URL(url, encoding=None, in_memory=True)
            self.assertTrue(isinstance(post, io.BytesIO))
            self.assertEqual(post.read(), b'bar') # type: ignore

    def test_url_d(self) -> None:
        with patch('urllib.request.urlopen') as mock:
            prepare_mock(mock, 'foo')

            post = URL(url, encoding=None, in_memory=False)

            self.assertTrue(isinstance(post, BytesIOTemporaryFile))
            self.assertEqual(b'foo', post.read()) # type: ignore


    def test_url_from_delimited_a(self) -> None:
        with patch('urllib.request.urlopen') as mock:
            prepare_mock(mock, 'a,b,c\n1,True,x\n20,False,y\n')

            post = Frame.from_csv(URL(url))
            self.assertEqual(post.to_pairs(),
                    (('a', ((0, 1), (1, 20))), ('b', ((0, True), (1, False))), ('c', ((0, 'x'), (1, 'y')))))

    # def test_url_from_delimited_b(self) -> None:

    #     url = 'https://stats.govt.nz/assets/Uploads/Business-financial-data/Business-financial-data-June-2022-quarter/Download-data/business-financial-data-june-2022-quarter-csv.zip'

    #     post = URL(url)
    #     import ipdb; ipdb.set_trace()


    def test_string_io_temp_file_a(self) -> None:
        content = 'foo\nbar'
        with temp_file('.txt') as fp:
            with open(fp, 'w') as f:
                f.write(content)

            siotf = StringIOTemporaryFile(fp)

            self.assertEqual(siotf.read(), content)
            siotf.seek(0)

            self.assertEqual(siotf.readline(), 'foo\n')
            siotf.seek(0)

            self.assertEqual(tuple(siotf), ('foo\n', 'bar'))

            del siotf
            self.assertFalse(os.path.exists(fp))

            # restore file so context manager can clean up
            with open(fp, 'w') as f:
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




if __name__ == '__main__':
    unittest.main()