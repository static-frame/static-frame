import unittest

import frame_fixtures as ff

from static_frame.test.test_case import TestCase
from static_frame.core.index import Index
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.pivot import pivot_records_items
# from static_frame.core.pivot import pivot_core

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


    def test_pivot_core_a(self) -> None:

        frame = ff.parse('s(20,4)|v(int)').assign[0].apply(lambda s: s % 4).assign[1].apply(lambda s: s % 3)

        # by default we get a tuple index
        post1 = frame.pivot([0, 1])
        self.assertEqual(post1.index.name, (0, 1))
        self.assertIs(post1.index.__class__, Index)
        self.assertTrue(post1.to_pairs(),
                ((2, (((0, 0), 463099), ((0, 1), -88017), ((0, 2), 35021), ((1, 0), 92867), ((1, 2), 96520), ((2, 0), 172133), ((2, 1), 279191), ((2, 2), 13448), ((3, 0), 255338), ((3, 1), 372807), ((3, 2), 155574))), (3, (((0, 0), 348362), ((0, 1), 175579), ((0, 2), 105269), ((1, 0), 58768), ((1, 2), 13448), ((2, 0), 84967), ((2, 1), 239151), ((2, 2), 170440), ((3, 0), 269300), ((3, 1), 204528), ((3, 2), 493169))))
                )

        # can provide index constructor
        post2 = frame.pivot([0, 1],
                index_constructor=IndexHierarchy.from_labels)
        self.assertEqual(post2.index.name, (0, 1))
        self.assertIs(post2.index.__class__, IndexHierarchy)
        self.assertTrue(post2.to_pairs(),
                ((2, (((0, 0), 463099), ((0, 1), -88017), ((0, 2), 35021), ((1, 0), 92867), ((1, 2), 96520), ((2, 0), 172133), ((2, 1), 279191), ((2, 2), 13448), ((3, 0), 255338), ((3, 1), 372807), ((3, 2), 155574))), (3, (((0, 0), 348362), ((0, 1), 175579), ((0, 2), 105269), ((1, 0), 58768), ((1, 2), 13448), ((2, 0), 84967), ((2, 1), 239151), ((2, 2), 170440), ((3, 0), 269300), ((3, 1), 204528), ((3, 2), 493169))))
                )


if __name__ == '__main__':
    unittest.main()


