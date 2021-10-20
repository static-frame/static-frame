import unittest

import frame_fixtures as ff

from static_frame.test.test_case import TestCase
from static_frame.core.pivot import pivot_records_items

class TestUnit(TestCase):

    def test_pivot_records_items_a(self) -> None:
        frame = ff.parse('s(3,6)|v(int,str,bool)|c(I,str)|i(I,int)')
        group_fields = ['zUvW',] # needs to be valif loc selection
        group_depth = 1
        data_fields = ['zkuW', 'z2Oo']
        func_single = sum
        func_map = None
        loc_to_iloc = frame.columns.loc_to_iloc
        post = tuple(pivot_records_items(
                blocks=frame._blocks,
                group_fields_iloc=loc_to_iloc(group_fields),
                group_depth=group_depth,
                data_fields_iloc=loc_to_iloc(data_fields),
                func_single=func_single,
                func_map=func_map,
                ))
        self.assertEqual(post,
                ((False, [201945, 1]), (True, [129017, False])))


    def test_pivot_records_items_b(self) -> None:
        frame = ff.parse('s(3,6)|v(int,str,bool)|c(I,str)|i(I,int)')
        group_fields = ['zUvW',] # needs to be valif loc selection
        group_depth = 1
        data_fields = ['zkuW', 'z2Oo']
        func_single = None
        func_map = (('zkuW', sum), ('z2Oo', min))
        loc_to_iloc = frame.columns.loc_to_iloc

        post = tuple(pivot_records_items(
                blocks=frame._blocks,
                group_fields_iloc=loc_to_iloc(group_fields),
                group_depth=group_depth,
                data_fields_iloc=loc_to_iloc(data_fields),
                func_single=func_single,
                func_map=func_map,
                ))
        self.assertEqual(post,
                ((False, [201945, 35021, 1, False]),
                (True, [129017, 129017, False, False]))
                )



if __name__ == '__main__':
    unittest.main()


