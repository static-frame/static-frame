from collections import Counter

import typing_extensions as tp

from doc.build_example import TAG_END
from doc.build_example import TAG_START
from doc.build_example import get_examples_fp
from doc.build_example import to_string_io
from doc.build_fine_tune import get_corpus
from static_frame.test.test_case import TestCase
from static_frame.test.test_case import skip_win

# clipboard does not work on some platforms / GitHub CI, third-party packages might change repr
SKIP_COMPARE = frozenset((
    'from_clipboard()',
    'to_clipboard()',
    'from_arrow()',
    'to_arrow()',
    'to_pandas()',
    'mloc',
    'values',
    ))

class TestUnit(TestCase):

    @skip_win
    def test_example_gen(self) -> None:
        # NOTE: comparing the direct output is problematic as different platforms might have subtle differences in float representations; thus, we just copmare exaples size and exercise example generation

        current = to_string_io()
        fp = get_examples_fp()

        def count(lines: tp.Iterable[str], counter: tp.Counter[str]) -> None:
            current = ''
            for line in lines:
                if line.startswith(TAG_START):
                    current = line.rstrip()
                    if current.split('-', maxsplit=1)[1] in SKIP_COMPARE:
                        current = ''
                    continue
                if current:
                    counter[current] += 1
                    continue
                if line.startswith(TAG_END):
                    current = ''

        counts_current: tp.Counter[str] = Counter()
        counts_past: tp.Counter[str] = Counter()

        with open(fp, encoding='utf-8') as past:
            count(past, counts_past)
        count(current.readlines(), counts_current)

        for key in sorted(counts_current.keys() | counts_past.keys()):
            with self.subTest(key):
                self.assertEqual(counts_current[key], counts_past[key], key)



    @skip_win
    def test_fine_tune_gen(self) -> None:
        get_corpus().validate_all()

if __name__ == '__main__':
    import unittest
    unittest.main()
