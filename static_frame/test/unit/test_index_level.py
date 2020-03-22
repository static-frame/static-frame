
import unittest
import numpy as np

from collections import OrderedDict

from static_frame import Index
from static_frame import IndexGO
from static_frame import IndexDate
from static_frame import ILoc
from static_frame import HLoc

from static_frame import IndexHierarchy
# from static_frame import IndexHierarchyGO
from static_frame import IndexLevel
from static_frame import IndexLevelGO
# from static_frame import HLoc

from static_frame.core.exception import ErrorInitIndexLevel
from static_frame.core.array_go import ArrayGO
from static_frame.test.test_case import TestCase

class TestUnit(TestCase):

    def test_index_level_a(self) -> None:
        groups = IndexGO(('A', 'B'))
        observations = IndexGO(('x', 'y'))
        targets = np.array(
                (IndexLevelGO(index=observations), IndexLevelGO(observations, offset=2)))

        level0 = IndexLevelGO(index=groups, targets=ArrayGO(targets))
        level1 = level0.to_index_level()

        groups = IndexGO(('C', 'D'))
        observations = IndexGO(('x', 'y', 'z'))
        targets = np.array(
                (IndexLevelGO(index=observations), IndexLevelGO(observations, offset=3)))

        level2 = IndexLevelGO(index=groups, targets=ArrayGO(targets))

        with self.assertRaises(RuntimeError):
            level0.extend(IndexLevelGO(index=observations))

        level0.extend(level2)

        self.assertEqual(len(level0.get_labels()), 10)
        self.assertEqual(len(level1.get_labels()), 4)
        self.assertEqual(len(level2.get_labels()), 6)

        assert level0.targets is not None

        self.assertEqual([lvl.offset for lvl in level0.targets], [0, 2, 4, 7])


    def test_index_level_b(self) -> None:
        with self.assertRaises(ErrorInitIndexLevel):
            _ = IndexLevel(('A', 'B')) #type: ignore


    def test_index_level_dtypes_all_a(self) -> None:
        level0 = IndexLevel(index=Index(('a', 'b')), targets=None)
        post = tuple(level0.dtypes_all())
        self.assertEqual(post[0], np.dtype('<U1'))
        self.assertEqual(len(post), 1)

    #---------------------------------------------------------------------------

    def test_index_level_dtypes_a(self) -> None:
        groups = IndexGO(('A', 'B'))
        observations = IndexGO(('x', 'y'))
        targets = ArrayGO(
                (IndexLevelGO(index=observations),
                IndexLevelGO(observations, offset=2)))
        level0 = IndexLevelGO(index=groups, targets=targets)
        self.assertEqual([d.kind for d in level0.dtypes()], ['U', 'U'])


    def test_index_level_dtypes_b(self) -> None:
        idx1 = Index(('A', 'B'))
        idx2 = IndexDate.from_date_range('2019-01-05', '2019-01-08')
        idx3 = Index((1, 2))

        hidx = IndexHierarchy.from_product(idx1, idx2, idx3)

        self.assertEqual([dt.kind for dt in hidx._levels.dtypes()],
                ['U', 'M', 'i'],
                )


    def test_index_level_dtypes_c(self) -> None:
        level0 = IndexLevel(index=Index(('a', 'b')), targets=None)
        post = tuple(level0.dtypes())
        self.assertEqual(post[0], np.dtype('<U1'))
        self.assertEqual(len(post), 1)


    #---------------------------------------------------------------------------

    def test_index_level_index_types_a(self) -> None:
        idx1 = Index(('A', 'B'))
        idx2 = IndexDate.from_date_range('2019-01-05', '2019-01-08')
        idx3 = Index((1, 2))
        hidx = IndexHierarchy.from_product(idx1, idx2, idx3)
        self.assertEqual(
                [it.__name__ for it in hidx._levels.index_types()],
                ['Index', 'IndexDate', 'Index'])


    def test_index_level_index_types_b(self) -> None:
        level0 = IndexLevel(index=Index(('a', 'b')), targets=None)
        post = tuple(level0.index_types())
        self.assertEqual(post[0], Index)
        self.assertEqual(len(post), 1)


    #---------------------------------------------------------------------------
    def test_index_level_contains_a(self) -> None:
        level0 = IndexLevel(index=Index(('a', 'b')), targets=None)
        self.assertFalse(('c',) in level0)
        self.assertTrue(('a',) in level0)



    #---------------------------------------------------------------------------

    def test_index_level_extend_a(self) -> None:
        level0 = IndexLevelGO(index=IndexGO(('a', 'b')), targets=None)
        with self.assertRaises(RuntimeError):
            level0.extend(level0)


    #---------------------------------------------------------------------------


    def test_index_level_get_labels_a(self) -> None:
        groups = IndexGO(('A', 'B'))
        observations = IndexGO(('x', 'y'))
        targets = ArrayGO(
                (IndexLevelGO(index=observations), IndexLevelGO(observations, offset=2)))

        level0 = IndexLevelGO(index=groups, targets=targets)

        self.assertEqual(level0.get_labels().tolist(),
                [['A', 'x'], ['A', 'y'], ['B', 'x'], ['B', 'y']])

        self.assertEqual(level0.get_labels().dtype.kind, 'U')

        groups = IndexGO((1, 2))
        observations = IndexGO((10, 20))
        targets = ArrayGO(
                (IndexLevelGO(index=observations), IndexLevelGO(observations, offset=2)))
        level1 = IndexLevelGO(index=groups, targets=targets)
        self.assertEqual(level1.get_labels().dtype.kind, 'i')

    #---------------------------------------------------------------------------

    def test_index_level_leaf_loc_to_iloc_a(self) -> None:

        groups = Index(('A', 'B', 'C'))
        dates = IndexDate.from_date_range('2018-01-01', '2018-01-04')
        observations = Index(('x', 'y'))

        lvl2a = IndexLevel(index=observations)
        lvl2b = IndexLevel(index=observations, offset=2)
        lvl2c = IndexLevel(index=observations, offset=4)
        lvl2d = IndexLevel(index=observations, offset=6)

        lvl2_targets = ArrayGO((lvl2a, lvl2b, lvl2c, lvl2d))

        lvl1a = IndexLevel(index=dates,
                targets=lvl2_targets, offset=0)
        lvl1b = IndexLevel(index=dates,
                targets=lvl2_targets, offset=len(lvl1a))
        lvl1c = IndexLevel(index=dates,
                targets=lvl2_targets, offset=len(lvl1a) * 2)

        # we need as many targets as len(index)
        lvl0 = IndexLevel(index=groups,
                targets=ArrayGO((lvl1a, lvl1b, lvl1c)))

        self.assertEqual(lvl0.leaf_loc_to_iloc(('B', '2018-01-04', 'y'),), 15)
        self.assertEqual(lvl0.leaf_loc_to_iloc(('A', '2018-01-01', 'y')), 1)


    def test_index_level_leaf_loc_to_iloc_b(self) -> None:

        level0 = IndexLevel(index=Index(('a', 'b')), targets=None)
        self.assertEqual(level0.leaf_loc_to_iloc(('b',)), 1)
        self.assertEqual(level0.leaf_loc_to_iloc(ILoc[1]), 1) #type: ignore


    def test_index_level_leaf_loc_to_iloc_c(self) -> None:

        groups = Index(('A', 'B', 'C'))
        dates = IndexDate.from_date_range('2018-01-01', '2018-01-04')
        observations = Index(('x', 'y'))

        lvl2a = IndexLevel(index=observations)
        lvl2b = IndexLevel(index=observations, offset=2)
        lvl2c = IndexLevel(index=observations, offset=4)
        lvl2d = IndexLevel(index=observations, offset=6)

        lvl2_targets = ArrayGO((lvl2a, lvl2b, lvl2c, lvl2d))

        lvl1a = IndexLevel(index=dates,
                targets=lvl2_targets, offset=0)
        lvl1b = IndexLevel(index=dates,
                targets=lvl2_targets, offset=len(lvl1a))
        lvl1c = IndexLevel(index=dates,
                targets=lvl2_targets, offset=len(lvl1a) * 2)

        lvl0 = IndexLevel(index=groups,
                targets=ArrayGO((lvl1a, lvl1b, lvl1c)))

        with self.assertRaises(KeyError):
            lvl0.leaf_loc_to_iloc(('A'))

        self.assertEqual(lvl0.leaf_loc_to_iloc(('A', '2018-01-01', 'y')), 1)


    #---------------------------------------------------------------------------

    def test_index_level_loc_to_iloc_a(self) -> None:

        level0 = IndexLevel(index=Index(('a', 'b')), targets=None)
        with self.assertRaises(KeyError):
            level0.loc_to_iloc('a')


    def test_index_level_loc_to_iloc_b(self) -> None:
        level0 = IndexLevel(index=Index(('a', 'b')), targets=None)
        with self.assertRaises(KeyError):
            level0.loc_to_iloc(HLoc['c',])


    #---------------------------------------------------------------------------

    def test_index_level_append_a(self) -> None:

        category = IndexGO(('I', 'II'))
        groups = IndexGO(('A', 'B'))
        observations = IndexGO(('x', 'y'))

        lvl2a = IndexLevelGO(index=observations)
        lvl2b = IndexLevelGO(index=observations, offset=2)
        lvl2_targets = ArrayGO((lvl2a, lvl2b))
        # must defensively copy index
        assert id(lvl2a.index) != id(lvl2b.index)

        lvl1a = IndexLevelGO(index=groups,
                targets=lvl2_targets,
                offset=0)


        # must create new index levels for each lower node
        lvl2c = IndexLevelGO(index=observations)
        lvl2d = IndexLevelGO(index=observations, offset=2)
        lvl2_targets = ArrayGO((lvl2c, lvl2d))
        # must defensively copy index
        assert id(lvl2c.index) != id(lvl2d.index)

        lvl1b = IndexLevelGO(index=groups,
                targets=lvl2_targets,
                offset=len(lvl1a))

        # we need as many targets as len(index)
        lvl0 = IndexLevelGO(index=category,
                targets=ArrayGO((lvl1a, lvl1b)))

        with self.assertRaises(RuntimeError):
            # RuntimeError: level for extension does not have necessary levels.
            lvl0.extend(lvl1b)

        lvl0.append(('II', 'B', 'z')) # depth not found is 2
        self.assertEqual(
                [lvl0.loc_to_iloc(tuple(x)) for x in lvl0.get_labels()],
                list(range(9)))

        lvl0.append(('II', 'C', 'a')) # depth not found is 1
        self.assertEqual(
                [lvl0.loc_to_iloc(tuple(x)) for x in lvl0.get_labels()],
                list(range(10)))

        lvl0.append(('III', 'A', 'a')) # 0

        self.assertEqual(
                [lvl0.loc_to_iloc(tuple(x)) for x in lvl0.get_labels()],
                list(range(11)))

        self.assertEqual(
                lvl0.get_labels().tolist(),
                [['I', 'A', 'x'], ['I', 'A', 'y'], ['I', 'B', 'x'], ['I', 'B', 'y'], ['II', 'A', 'x'], ['II', 'A', 'y'], ['II', 'B', 'x'], ['II', 'B', 'y'], ['II', 'B', 'z'], ['II', 'C', 'a'], ['III', 'A', 'a']])

    def test_index_level_append_b(self) -> None:

        groups = IndexGO(('A', 'B'))
        observations = IndexGO(('x', 'y'))
        lvl2a = IndexLevelGO(index=observations)
        lvl2b = IndexLevelGO(index=observations, offset=2)
        lvl2_targets = ArrayGO((lvl2a, lvl2b))
        lvl1a = IndexLevelGO(index=groups,
                targets=lvl2_targets,
                offset=0)

        with self.assertRaises(RuntimeError):
            lvl1a.append((1, 2, 3))

        lvl1a.append((1, 2))
        self.assertEqual(lvl1a.get_labels().tolist(),
                [['A', 'x'], ['A', 'y'], ['B', 'x'], ['B', 'y'], [1, 2]])



    #---------------------------------------------------------------------------



    def test_index_level_iter_a(self) -> None:
        OD = OrderedDict
        tree = OD([
                ('I', OD([
                        ('A', (1, 2)), ('B', (1, 2, 3)), ('C', (2, 3))
                        ])
                ),
                ('II', OD([
                        ('A', (1, 2, 3)), ('B', (1,))
                        ])
                ),
                ])

        levels = IndexHierarchy._tree_to_index_level(tree)

        post = list(levels.iter(0))
        self.assertEqual(post, ['I', 'II'])

        post = list(levels.iter(1))
        self.assertEqual(post, ['A', 'B', 'C', 'A', 'B'])

        post = list(levels.iter(2))
        self.assertEqual(post, [1, 2, 1, 2, 3, 2, 3, 1, 2, 3, 1])





    def test_index_level_label_widths_at_depth_a(self) -> None:
        OD = OrderedDict
        tree = OD([
                ('I', OD([
                        ('A', (1, 2)), ('B', (1, 2, 3)), ('C', (2, 3))
                        ])
                ),
                ('II', OD([
                        ('A', (1,)), ('B', (1,))
                        ])
                ),
                ('III', OD([
                        ('A', (1, 2, 3)), ('B', (1,))
                        ])
                ),
                ])

        levels = IndexHierarchy._tree_to_index_level(tree)

        post0 = tuple(levels.label_widths_at_depth(0))
        post1 = tuple(levels.label_widths_at_depth(1))
        post2 = tuple(levels.label_widths_at_depth(2))

        self.assertEqual(post0, (('I', 7), ('II', 2), ('III', 4)))

        self.assertEqual(post1,
            (('A', 2), ('B', 3), ('C', 2), ('A', 1), ('B', 1), ('A', 3), ('B', 1))
        )

        self.assertEqual(post2,
            (((1, 1), (2, 1), (1, 1), (2, 1), (3, 1), (2, 1), (3, 1), (1, 1), (1, 1), (1, 1), (2, 1), (3, 1), (1, 1)))
        )


if __name__ == '__main__':
    unittest.main()


