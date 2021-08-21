
import unittest

from static_frame.core.style_config import StyleConfig
from static_frame.core.style_config import style_config_css_factory
from static_frame.core.frame import Frame
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    #---------------------------------------------------------------------------
    def test_style_config_a(self) -> None:

        sc = StyleConfig()
        self.assertEqual(sc.frame(), '')
        self.assertEqual(sc.apex(3, (1,1)), ('3', ''))
        self.assertEqual(sc.values(3, (1,1)), ('3', ''))
        self.assertEqual(sc.index(3), ('3', ''))
        self.assertEqual(sc.columns(3), ('3', ''))


    def test_style_config_css_factory_a(self) -> None:
        f = Frame()
        sc = StyleConfig(f)
        self.assertTrue(style_config_css_factory(sc, f) is sc)


if __name__ == '__main__':
    unittest.main()

