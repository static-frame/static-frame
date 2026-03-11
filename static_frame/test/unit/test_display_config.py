from __future__ import annotations

from static_frame import DisplayConfig
from static_frame.core.display_config import DisplayFormatHTMLTable
from static_frame.core.style_config import StyleConfig
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):
    def test_display_config_a(self) -> None:
        post = DisplayConfig.interface
        self.assertTrue(post.shape[0] > 40)
        self.assertTrue(post.shape[1] == 3)

    def test_display_format_html_table_a(self) -> None:
        sc = StyleConfig()
        post = DisplayFormatHTMLTable.markup_row(['a', 'b'], 1, -1, sc)
        self.assertEqual(list(post), ['<tr>', '<th>a</th>', '<th>b</th>', '</tr>'])

    def test_display_format_html_table_b(self) -> None:
        sc = StyleConfig()
        post = DisplayFormatHTMLTable.markup_row(['a', 'b'], 0, -1, sc)
        self.assertEqual(list(post), ['<tr>', '<th>a</th>', '<th>b</th>', '</tr>'])


if __name__ == '__main__':
    import unittest

    unittest.main()
