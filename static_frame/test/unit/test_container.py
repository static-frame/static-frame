import unittest

from static_frame.core.container import ContainerOperand
from static_frame.test.test_case import TestCase
from static_frame.core.interface import UFUNC_AXIS_SKIPNA
from static_frame.core.interface import UFUNC_SHAPE_SKIPNA


class TestUnit(TestCase):

    def test_container_attrs(self) -> None:

        for attr in UFUNC_AXIS_SKIPNA.keys() | UFUNC_SHAPE_SKIPNA.keys():
            c = ContainerOperand
            self.assertTrue(hasattr(c, attr))

        with self.assertRaises(NotImplementedError):
            c().display()




if __name__ == '__main__':
    unittest.main()
