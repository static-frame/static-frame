
import unittest
import pydoc
from static_frame.test.test_case import TestCase

class TestUnit(TestCase):

    # this test is slow
    def test_interface_help_a(self) -> None:

        for target in self.get_containers():
            post = pydoc.render_doc(target, renderer=pydoc.plaintext) # type: ignore
            self.assertTrue(len(post) > 0)

if __name__ == '__main__':
    unittest.main()

