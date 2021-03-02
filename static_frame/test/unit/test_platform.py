


import static_frame as sf
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    def test_platform_a(self) -> None:
        post = sf.Platform.to_series()
        self.assertTrue(post.shape[0] > 10)

        d = sf.Platform.display()
        self.assertTrue(len(str(d)) > 100)
