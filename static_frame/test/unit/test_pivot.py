import numpy as np
import frame_fixtures as ff

from static_frame.test.test_case import TestCase
from static_frame.core.frame import Frame
from static_frame.core.index import Index
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.pivot import pivot_items_to_block
from static_frame.core.pivot import pivot_items_to_frame
# from static_frame.core.pivot import pivot_records_items


class TestUnit(TestCase):

    def test_pivot_items_to_block_a(self) -> None:
        f = ff.parse('s(6,4)|v(int)').assign[0](
                range(6)
                )
        group_fields_iloc = [0]
        index_outer = Index(f[0].values.tolist())

        post = pivot_items_to_block(
                blocks=f._blocks,
                group_fields_iloc=group_fields_iloc,
                group_depth=1,
                data_field_iloc=3,
                func_single=None,
                dtype=np.dtype(int),
                fill_value=0,
                fill_value_dtype=np.dtype(int),
                index_outer=index_outer,
                kind='mergesort',
                )
        self.assertEqual(post.tolist(),
                [129017,  35021, 166924, 122246, 197228, 105269]
                )

    def test_pivot_items_to_frame_a(self) -> None:
        f = ff.parse('s(6,4)|v(int)').assign[0](
                range(6)
                )

        post = pivot_items_to_frame(
                blocks=f._blocks,
                group_fields_iloc=[0],
                group_depth=1,
                data_field_iloc=3,
                func_single=lambda x: str(x) if x % 2 else sum(x),
                frame_cls=Frame,
                name='foo',
                dtype=None,
                index_constructor=Index,
                columns_constructor=Index,
                kind='mergesort',
                )
        self.assertEqual(post.to_pairs(),
                (('foo', ((0, '[129017]'), (1, '[35021]'), (2, 166924), (3, 122246), (4, 197228), (5, '[105269]'))),))



    def test_pivot_items_to_frame_b(self) -> None:
        f = ff.parse('s(6,4)|v(int)').assign[0](
                range(6)
                )
        post = pivot_items_to_frame(
                blocks=f._blocks,
                group_fields_iloc=[0, 1],
                group_depth=2,
                data_field_iloc=3,
                func_single=None,
                frame_cls=Frame,
                name='foo',
                dtype=np.dtype(int),
                index_constructor=IndexHierarchy.from_labels,
                columns_constructor=Index,
                kind='mergesort',
                )
        self.assertEqual(post.to_pairs(),
                (('foo', (((0, 162197), 129017), ((1, -41157), 35021), ((2, 5729), 166924), ((3, -168387), 122246), ((4, 140627), 197228), ((5, 66269), 105269))),),
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
    import unittest
    unittest.main()
