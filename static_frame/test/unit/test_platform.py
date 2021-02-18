
from contextlib import redirect_stdout
from io import StringIO

import static_frame as sf
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    def test_platform_a(self):
        post = sf.Platform.to_series()
        self.assertTrue(post.shape[0] > 10)

        f = StringIO()
        with redirect_stdout(f):
            sf.Platform.display()

        msg = f.getvalue()
        self.assertTrue(len(msg) > 100)
