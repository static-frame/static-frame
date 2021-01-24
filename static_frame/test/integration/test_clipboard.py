

from static_frame import Frame
from static_frame.test.test_case import TestCase
from static_frame.test.test_case import skip_linux_no_display


class TestUnit(TestCase):

    # NOTE: this test will end up clearing the user's clipboard

    @skip_linux_no_display #type: ignore
    def test_frame_to_clipboard_a(self) -> None:
        records = (
                (2, 'a', False),
                (3, 'b', False),
                )
        f1 = Frame.from_records(records,
                columns=('r', 's', 't'),
                index=('w', 'x'))

        f1.to_clipboard()
        f2 = Frame.from_clipboard(index_depth=1)
        self.assertTrue(f2.equals(f1, compare_dtype=True))



