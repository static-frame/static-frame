
import unittest
from collections import OrderedDict
import copy

import numpy as np

from static_frame import Frame
from static_frame import HLoc
from static_frame import ILoc
from static_frame import Index
from static_frame import IndexDate
from static_frame import IndexGO
from static_frame import IndexHierarchy
from static_frame import IndexLevel
from static_frame import IndexLevelGO
from static_frame.core.array_go import ArrayGO
from static_frame.core.exception import ErrorInitIndexLevel
from static_frame.test.test_case import skip_win
from static_frame.test.test_case import TestCase


from static_frame.core.util import EMPTY_ARRAY

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

        self.assertEqual(len(level0.values), 10)
        self.assertEqual(len(level1.values), 4)
        self.assertEqual(len(level2.values), 6)

        assert level0.targets is not None

        self.assertEqual([lvl.offset for lvl in level0.targets], [0, 2, 4, 7])

        self.assertEqual(level2.depth, next(level2.depths()))


    def test_index_level_b(self) -> None:
        with self.assertRaises(ErrorInitIndexLevel):
            _ = IndexLevel(('A', 'B')) #type: ignore


    def test_index_level_dtypes_all_a(self) -> None:
        level0 = IndexLevel(index=Index(('a', 'b')), targets=None)
        post = tuple(level0.dtypes_iter())
        self.assertEqual(post[0], np.dtype('<U1'))
        self.assertEqual(len(post), 1)
        self.assertEqual(level0.depth, next(level0.depths()))


    #---------------------------------------------------------------------------

    def test_index_level_dtypes_a(self) -> None:
        groups = IndexGO(('A', 'B'))
        observations = IndexGO(('x', 'y'))
        targets = ArrayGO(
                (IndexLevelGO(index=observations),
                IndexLevelGO(observations, offset=2)))
        level0 = IndexLevelGO(index=groups, targets=targets)
        self.assertEqual([d.kind for d in level0.dtype_per_depth()], ['U', 'U'])


    def test_index_level_dtypes_b(self) -> None:
        idx1 = Index(('A', 'B'))
        idx2 = IndexDate.from_date_range('2019-01-05', '2019-01-08')
        idx3 = Index((1, 2))

        hidx = IndexHierarchy.from_product(idx1, idx2, idx3)

        self.assertEqual([dt.kind for dt in hidx._levels.dtype_per_depth()],
                ['U', 'M', 'i'],
                )


    def test_index_level_dtypes_c(self) -> None:
        level0 = IndexLevel(index=Index(('a', 'b')), targets=None)
        post = tuple(level0.dtype_per_depth())
        self.assertEqual(post[0], np.dtype('<U1'))
        self.assertEqual(len(post), 1)

    #---------------------------------------------------------------------------
    @skip_win #type: ignore
    def test_index_level_dtypes_per_depth_a(self) -> None:
        hidx = IndexHierarchy.from_labels((('a', 1, 'x'), ('a', 2, 'y'), ('b', 1, 'foo'), ('b', 1, 'bar')))
        lvl = hidx._levels

        self.assertEqual(
                tuple(lvl.dtypes_at_depth(0)),
                (np.dtype('<U1'),))
        self.assertEqual(
                tuple(lvl.dtypes_at_depth(1)),
                (np.dtype('int64'), np.dtype('int64')))
        self.assertEqual(
                tuple(lvl.dtypes_at_depth(2)),
                (np.dtype('<U1'), np.dtype('<U1'), np.dtype('<U3'))
                )


    def test_index_level_values_at_depth_a(self) -> None:
        hidx = IndexHierarchy.from_labels((('a', 1, 'x'), ('a', 2, 'y'), ('b', 1, 'foo'), ('b', 1, 'bar')))
        lvl = hidx._levels
        self.assertEqual(lvl.values_at_depth(2).tolist(), ['x', 'y', 'foo', 'bar'])
        self.assertEqual(lvl.depth, next(lvl.depths()))

    def test_index_level_values_at_depth_b(self) -> None:

        hidx = IndexHierarchy.from_labels((('a', 1, 'x'), ('a', 2, 'y'), ('b', 1, 'foo'), ('b', 2, None)))
        lvl = hidx._levels
        self.assertEqual(lvl.values_at_depth(2).tolist(), ['x', 'y', 'foo', None])


    #---------------------------------------------------------------------------

    def test_index_level_index_types_a(self) -> None:
        idx1 = Index(('A', 'B'))
        idx2 = IndexDate.from_date_range('2019-01-05', '2019-01-08')
        idx3 = Index((1, 2))
        hidx = IndexHierarchy.from_product(idx1, idx2, idx3)
        lvl = hidx._levels
        self.assertEqual(
                [it.__name__ for it in lvl.index_types()],
                ['Index', 'IndexDate', 'Index'])
        self.assertEqual(lvl.depth, next(lvl.depths()))


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

        self.assertEqual(level0.values.tolist(),
                [['A', 'x'], ['A', 'y'], ['B', 'x'], ['B', 'y']])

        self.assertEqual(level0.values.dtype.kind, 'U')

        groups = IndexGO((1, 2))
        observations = IndexGO((10, 20))
        targets = ArrayGO(
                (IndexLevelGO(index=observations), IndexLevelGO(observations, offset=2)))
        level1 = IndexLevelGO(index=groups, targets=targets)
        self.assertEqual(level1.values.dtype.kind, 'i')

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
        self.assertEqual(level0.leaf_loc_to_iloc(ILoc[1]), 1)


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

    def test_index_level_leaf_loc_to_iloc_d(self) -> None:

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

        levels = IndexLevel.from_tree(tree)
        with self.assertRaises(KeyError):
            levels.leaf_loc_to_iloc(('II', 'B', 1, 'a'))

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
                [lvl0.loc_to_iloc(tuple(x)) for x in lvl0.values],
                list(range(9)))

        lvl0.append(('II', 'C', 'a')) # depth not found is 1
        self.assertEqual(
                [lvl0.loc_to_iloc(tuple(x)) for x in lvl0.values],
                list(range(10)))

        lvl0.append(('III', 'A', 'a')) # 0

        self.assertEqual(
                [lvl0.loc_to_iloc(tuple(x)) for x in lvl0.values],
                list(range(11)))

        self.assertEqual(
                lvl0.values.tolist(),
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
        self.assertEqual(lvl1a.values.tolist(),
                [['A', 'x'], ['A', 'y'], ['B', 'x'], ['B', 'y'], [1, 2]])


    def test_index_level_append_c(self) -> None:

        levels1 = IndexLevelGO(IndexGO(()), depth_reference=3)
        levels1.append(('III', 'A', 1))

        self.assertEqual(len(tuple(levels1.dtypes_at_depth(0))), 1)
        self.assertEqual(len(tuple(levels1.dtypes_at_depth(1))), 1)
        self.assertEqual(len(tuple(levels1.dtypes_at_depth(2))), 1)

        self.assertEqual(levels1.values.tolist(),
                [['III', 'A', 1]]
                )

        levels1.append(('III', 'A', 2))

        self.assertEqual(levels1.values.tolist(),
                [['III', 'A', 1], ['III', 'A', 2]])


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

        levels = IndexLevel.from_tree(tree)

        post = list(levels.index_at_depth(0))
        self.assertEqual(post, ['I', 'II'])

        post = list(levels.index_at_depth(1))
        self.assertEqual(post, ['A', 'B', 'C', 'A', 'B'])

        post = list(levels.index_at_depth(2))
        self.assertEqual(post, [1, 2, 1, 2, 3, 2, 3, 1, 2, 3, 1])


    def test_index_level_iter_b(self) -> None:
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

        levels = IndexLevel.from_tree(tree)
        tuples = tuple(levels)
        self.assertEqual(
                tuples,
                (('I', 'A', 1), ('I', 'A', 2), ('I', 'B', 1), ('I', 'B', 2), ('I', 'B', 3), ('I', 'C', 2), ('I', 'C', 3), ('II', 'A', 1), ('II', 'A', 2), ('II', 'A', 3), ('II', 'B', 1))
                )


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

        levels = IndexLevel.from_tree(tree)

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


    def test_index_level_label_widths_at_depth_b(self) -> None:
        f = Frame.from_dict(
            dict(a=(1,2,3,4), b=(True, False, True, False), c=list('qrst')))
        f = f.set_index_hierarchy(['a', 'b'])

        post1 = tuple(f.index._levels.label_widths_at_depth(0))
        self.assertEqual(post1, ((1, 1), (2, 1), (3, 1), (4, 1)))
        post2 = tuple(f.index._levels.label_widths_at_depth(1))
        self.assertEqual(post2, ((True, 1), (False, 1), (True, 1), (False, 1)))


    def test_index_level_label_widths_at_depth_c(self) -> None:
        labels = (
                ('I', 'A', 1),
                ('I', 'B', 2),
                ('II', 'C', 3),
                ('II', 'C', 4),
                )
        ih = IndexHierarchy.from_labels(labels)
        lavels = ih._levels

        self.assertEqual(tuple(lavels.label_widths_at_depth(0)),
                (('I', 2), ('II', 2))
        )
        self.assertEqual(tuple(lavels.label_widths_at_depth(1)),
                (('A', 1), ('B', 1), ('C', 2))
        )
        self.assertEqual(tuple(lavels.label_widths_at_depth(2)),
                ((1, 1), (2, 1), (3, 1), (4, 1))
        )

    def test_index_level_label_widths_at_depth_d(self) -> None:
        labels = (
                ('I', 'A', 1),
                ('I', 'B', 2),
                ('I', 'B', 3),
                ('II', 'C', 3),
                ('II', 'C', 4),
                )
        ih = IndexHierarchy.from_labels(labels)
        lavels = ih._levels

        self.assertEqual(tuple(lavels.label_widths_at_depth(0)),
                (('I', 3), ('II', 2))
        )
        self.assertEqual(tuple(lavels.label_widths_at_depth(1)),
                (('A', 1), ('B', 2), ('C', 2))
        )
        self.assertEqual(tuple(lavels.label_widths_at_depth(2)),
                ((1, 1), (2, 1), (3, 1), (3, 1), (4, 1))
        )


    def test_index_levels_with_tuple_a(self) -> None:
        OD = OrderedDict
        tree = OD([
                (('I', 'I'), OD([
                        ('A', (1, 2)), ('B', (1, 2, 3)), ('C', (2, 3))
                        ])
                ),
                (('II', 'II'), OD([
                        ('A', (1,)), ('B', (1,))
                        ])
                ),
                ])

        levels = IndexLevel.from_tree(tree)
        self.assertEqual(levels.depth, 3)
        self.assertEqual(levels.loc_to_iloc((('II', 'II'), 'B', 1)), 8)



    #---------------------------------------------------------------------------
    def test_index_levels_equals_a(self) -> None:
        OD = OrderedDict
        tree1 = OD([
                (('I', 'I'), OD([
                        ('A', (1, 2)), ('B', (1, 2, 3)), ('C', (2, 3))
                        ])
                ),
                (('II', 'II'), OD([
                        ('A', (1,)), ('B', (1,))
                        ])
                ),
                ])

        levels1 = IndexLevel.from_tree(tree1)

        tree2 = OD([
                (('I', 'I'), OD([
                        ('A', (1, 2)), ('B', (1, 2, 3)), ('C', (2, 3))
                        ])
                ),
                (('II', 'II'), OD([
                        ('A', (1,)), ('B', (1,))
                        ])
                ),
                ])

        levels2 = IndexLevel.from_tree(tree2)

        tree3 = OD([
                (('I', 'I'), OD([
                        ('A', (1, 2)), ('B', (1, 2, 3)), ('C', (2, 3))
                        ])
                ),
                (('II', 'II'), OD([
                        ('A', (1,)), ('B', (0,)) # diff
                        ])
                ),
                ])

        levels3 = IndexLevel.from_tree(tree3)



        self.assertTrue(levels1.equals(levels1))
        self.assertTrue(levels1.equals(levels2))
        self.assertTrue(levels2.equals(levels1))

        self.assertFalse(levels2.equals(levels3))

    def test_index_levels_equals_b(self) -> None:

        idx1 = Index(('a', 'b', 'c', 'd', 'e'))
        idx2 = Index(range(10))
        levels1 = IndexHierarchy.from_product(idx1, idx2)._levels

        idx3 = Index(('a', 'b', 'c', 'd', 'e'))
        idx4 = Index(range(10))
        levels2 = IndexHierarchy.from_product(idx3, idx4)._levels

        self.assertTrue(levels1.equals(levels2))

    def test_index_levels_equals_c(self) -> None:

        OD = OrderedDict
        tree1 = OD([
                ('I', OD([
                        ('A', (1, 2)), ('B', (1, 2)),
                        ])
                ),
                ('II', OD([
                        ('A', (1, 2)),
                        ])
                ),
                ])

        levels1 = IndexLevel.from_tree(tree1)

        tree2 = OD([
                ('I', OD([
                        ('A', (1, 2)),
                        ])
                ),
                ('II', OD([
                        ('A', (1, 2)),
                        ])
                ),
                ])

        levels2 = IndexLevel.from_tree(tree2)

        tree3 = OD([
                ('I', (1, 2)),
                ('II', (1, 2)),
                ])

        levels3 = IndexLevel.from_tree(tree3)

        self.assertFalse(levels1.equals(levels1.values, compare_class=True))
        self.assertFalse(levels1.equals(levels1.values, compare_class=False))

        # differing length
        self.assertFalse(levels1.equals(levels2))

        # differeing depth
        self.assertFalse(levels2.equals(levels3))


    def test_index_levels_equals_d(self) -> None:

        levels1 = IndexLevel(Index(('a', 'b', 'c')), targets=None)
        levels2 = IndexLevel(Index(('a', 'b', 'c')), targets=None)

        self.assertTrue(levels1.equals(levels2))

    #---------------------------------------------------------------------------

    def test_index_levels_to_type_blocks_a(self) -> None:

        levels1 = IndexLevel(Index(()), targets=None, depth_reference=3)
        tb = levels1.to_type_blocks()
        # NOTE: this will be updated to (0, 0) with IndexLevel support for zero size
        self.assertEqual(tb.shape, (0, 3))

    #---------------------------------------------------------------------------

    def test_index_level_depth_reference_a(self) -> None:
        dtype = np.dtype

        lvl1 = IndexLevel(Index(()), depth_reference=3)
        self.assertEqual(lvl1.depth, 3)

        self.assertEqual(tuple(lvl1.dtype_per_depth()),
                (dtype('float64'), dtype('float64'), dtype('float64')))

        self.assertEqual(tuple(lvl1.index_array_at_depth(0)), (EMPTY_ARRAY,))

        self.assertEqual(lvl1.values_at_depth(0).tolist(), EMPTY_ARRAY.tolist())

        tb = lvl1.to_type_blocks()
        self.assertEqual(tb.shape, (0, 3))

    #---------------------------------------------------------------------------
    def test_index_level_from_depth_a(self) -> None:

        lvl1 = IndexLevel.from_depth(4)
        self.assertEqual(lvl1.depth, 4)
        self.assertEqual(len(lvl1), 0)

    #---------------------------------------------------------------------------
    def test_index_level_labels_at_depth_a(self) -> None:

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
        levels = IndexLevel.from_tree(tree)

        self.assertEqual(tuple(levels.labels_at_depth(0)),
                ('I', 'I', 'I', 'I', 'I', 'I', 'I', 'II', 'II', 'III', 'III', 'III', 'III'))
        self.assertEqual(tuple(levels.labels_at_depth(1)),
                ('A', 'A', 'B', 'B', 'B', 'C', 'C', 'A', 'B', 'A', 'A', 'A', 'B'))
        self.assertEqual(tuple(levels.labels_at_depth(2)),
                (1, 2, 1, 2, 3, 2, 3, 1, 1, 1, 2, 3, 1))


    #---------------------------------------------------------------------------

    def test_index_level_deepcopy_a(self) -> None:

        groups = IndexGO(('A', 'B'))
        observations = IndexGO(('x', 'y'))

        lvl2a = IndexLevelGO(index=observations)
        lvl2b = IndexLevelGO(index=observations, offset=2)
        lvl2_targets = ArrayGO((lvl2a, lvl2b))
        assert id(lvl2a.index) != id(lvl2b.index)

        lvl1a = IndexLevelGO(index=groups,
                targets=lvl2_targets,
                offset=0)

        post = copy.deepcopy(lvl1a)
        self.assertEqual(lvl1a.values.tolist(), post.values.tolist())
        self.assertEqual(
                id(post.targets[0].index._labels), #type: ignore
                id(post.targets[1].index._labels)) #type: ignore


if __name__ == '__main__':
    unittest.main()

