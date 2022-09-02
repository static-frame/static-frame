
from static_frame.test.test_case import TestCase
from doc.build_example import to_string_io
from doc.build_example import get_examples_fp


class TestUnit(TestCase):

    SKIP_NEXT = frozenset((
        '>>> ih.mean()\n',
        '>>> ih.median()\n',
        '>>> ih.std()\n',
        '>>> ih.var()\n',
        '>>> bt\n',
        '>>> bt.T\n',
        '>>> bt.via_container\n',
        '>>> repr(bt)\n',
        '>>> str(bt)\n',
        '>>> repr(hl)\n',
        '>>> str(hl)\n',
        '>>> repr(il)\n',
        '>>> str(il)\n',
        '>>> sf.FillValueAuto.from_default()\n',
        '>>> repr(fva)\n',
        '>>> str(fva)\n',
        ))

    def test_example_gen(self) -> None:
        current = to_string_io()
        fp = get_examples_fp()
        skip = False
        with open(fp) as past:
            for line_past in past:
                line_current = current.readline()
                if line_current in self.SKIP_NEXT:
                    skip = True
                    self.assertEqual(line_past, line_current)
                elif skip:
                    skip = False # disable
                else:
                    self.assertEqual(line_past, line_current)
                # import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    import unittest
    unittest.main()
