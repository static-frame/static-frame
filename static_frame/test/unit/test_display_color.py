from __future__ import annotations

from static_frame.core.display_color import HexColor
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    def test_hex_str_to_int_a(self) -> None:
        post = HexColor._hex_str_to_int('aqua')
        self.assertEqual(post, 65535)

    def test_format_html_a(self) -> None:
        post = HexColor.format_html('aqua', 'test')
        self.assertEqual(post, '<span style="color: #ffff">test</span>')

    def test_hex_color_format_a(self) -> None:

        msg = HexColor.format_terminal('#4b006e', 'test')
        self.assertEqual(msg, '\x1b[38;5;53mtest\x1b[0m')

        msg = HexColor.format_terminal(0xaaaaaa, 'test')
        self.assertEqual(msg, '\x1b[38;5;248mtest\x1b[0m')

        msg = HexColor.format_terminal('0xaaaaaa', 'test')
        self.assertEqual(msg, '\x1b[38;5;248mtest\x1b[0m')

        msg = HexColor.format_terminal('#040273', 'test')
        self.assertEqual(msg, '\x1b[38;5;4mtest\x1b[0m')


if __name__ == '__main__':
    import unittest
    unittest.main()
