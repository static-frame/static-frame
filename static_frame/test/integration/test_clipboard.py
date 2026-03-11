import numpy as np

from static_frame import Frame
from static_frame.test.test_case import TestCase, skip_linux_no_display, skip_mac_pyle310


class TestUnit(TestCase):
    # NOTE: this test will end up clearing the user's clipboard

    @skip_mac_pyle310
    @skip_linux_no_display
    def test_frame_to_clipboard_a(self) -> None:
        records = (
            (2, 'a', False),
            (3, 'b', False),
        )
        f1 = Frame.from_records(
            records, columns=('r', 's', 't'), index=('w', 'x'), dtypes={'r': np.int64}
        )

        f1.to_clipboard()
        f2 = Frame.from_clipboard(index_depth=1)
        self.assertTrue(f2.equals(f1, compare_dtype=True))


if __name__ == '__main__':
    import unittest

    unittest.main()
