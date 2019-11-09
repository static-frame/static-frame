


import unittest

# import numpy as np

from static_frame.test.test_case import TestCase
from static_frame.performance.main import performance
from static_frame.performance.main import performance_tables_from_records
from static_frame.performance import core
from static_frame.performance.main import yield_classes

class TestUnit(TestCase):

    def test_performance(self) -> None:

        core.SampleData.create()
        records = []
        for cls in sorted(yield_classes(core, '*'),
                key=lambda c: c.__name__):
            records.append(performance(core, cls))

        f, f_display = performance_tables_from_records(records)

        self.assertTrue(f['sf/pd'].mean() < 5)
        self.assertTrue(len(f) > 55)

if __name__ == '__main__':
    unittest.main()
