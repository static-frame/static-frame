

from static_frame.test.test_case import TestCase

# assuming located in the same directory
# from static_frame import Index
# from static_frame import IndexGO
# from static_frame import Series
# from static_frame import Frame
# from static_frame import FrameGO
# from static_frame import TypeBlocks
# from static_frame import Display
# from static_frame import mloc
# from static_frame import DisplayConfig
# from static_frame import DisplayConfigs

from static_frame.core.display_color import HexColor

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
