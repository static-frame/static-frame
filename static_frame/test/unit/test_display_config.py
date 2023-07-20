from __future__ import annotations

from static_frame import DisplayConfig
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    def test_display_config_a(self) -> None:
        post = DisplayConfig.interface
        self.assertTrue(post.shape[0] > 40)
        self.assertTrue(post.shape[1] == 3)


if __name__ == '__main__':
    import unittest
    unittest.main()
