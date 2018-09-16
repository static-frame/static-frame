
import unittest
import numpy as np

from collections import OrderedDict

from static_frame import Index
from static_frame import IndexGO
from static_frame import IndexDate
from static_frame import Series
from static_frame import Frame
from static_frame import FrameGO
from static_frame import IndexYearMonth
from static_frame import IndexYear

from static_frame import IndexHierarchy
from static_frame import IndexHierarchyGO
from static_frame import IndexLevel
from static_frame import IndexLevelGO
from static_frame import HLoc

from static_frame.test.test_case import TestCase

class TestUnit(TestCase):


    def test_index_level_a(self):
        groups = IndexGO(('A', 'B'))
        observations = IndexGO(('x', 'y'))
        targets = np.array(
                (IndexLevelGO(index=observations), IndexLevelGO(observations, offset=2)))

        level0 = IndexLevelGO(index=groups, targets=targets)
        level1 = level0.to_index_level()

        groups = IndexGO(('C', 'D'))
        observations = IndexGO(('x', 'y', 'z'))
        targets = np.array(
                (IndexLevelGO(index=observations), IndexLevelGO(observations, offset=3)))

        level2 = IndexLevelGO(index=groups, targets=targets)

        level0.extend(level2)

        self.assertEqual(len(level0.get_labels()), 10)
        self.assertEqual(len(level1.get_labels()), 4)
        self.assertEqual(len(level2.get_labels()), 6)

        # import ipdb; ipdb.set_trace()
        self.assertEqual([lvl.offset for lvl in level0.targets], [0, 2, 4, 7])

        # import ipdb; ipdb.set_trace()






    def test_level_leaf_loc_to_iloc_a(self):

        groups = Index(('A', 'B', 'C'))
        dates = IndexDate.from_date_range('2018-01-01', '2018-01-04')
        observations = Index(('x', 'y'))

        lvl2a = IndexLevel(index=observations)
        lvl2b = IndexLevel(index=observations, offset=2)
        lvl2c = IndexLevel(index=observations, offset=4)
        lvl2d = IndexLevel(index=observations, offset=6)
        lvl2_targets = np.array((lvl2a, lvl2b, lvl2c, lvl2d))

        lvl1a = IndexLevel(index=dates,
                targets=lvl2_targets, offset=0)
        lvl1b = IndexLevel(index=dates,
                targets=lvl2_targets, offset=len(lvl1a))
        lvl1c = IndexLevel(index=dates,
                targets=lvl2_targets, offset=len(lvl1a) * 2)

        # we need as many targets as len(index)
        lvl0 = IndexLevel(index=groups,
                targets=np.array((lvl1a, lvl1b, lvl1c)))

        self.assertEqual(lvl0.leaf_loc_to_iloc(('B', '2018-01-04', 'y'),), 15)
        self.assertEqual(lvl0.leaf_loc_to_iloc(('A', '2018-01-01', 'y')), 1)


    def test_level_append_a(self):

        category = IndexGO(('I', 'II'))
        groups = IndexGO(('A', 'B'))
        observations = IndexGO(('x', 'y'))

        lvl2a = IndexLevelGO(index=observations)
        lvl2b = IndexLevelGO(index=observations, offset=2)
        lvl2_targets = np.array((lvl2a, lvl2b))
        # must defensively copy index
        assert id(lvl2a.index) != id(lvl2b.index)

        lvl1a = IndexLevelGO(index=groups,
                targets=lvl2_targets,
                offset=0)


        # must create new index levels for each lower node
        lvl2c = IndexLevelGO(index=observations)
        lvl2d = IndexLevelGO(index=observations, offset=2)
        lvl2_targets = np.array((lvl2c, lvl2d))
        # must defensively copy index
        assert id(lvl2c.index) != id(lvl2d.index)

        lvl1b = IndexLevelGO(index=groups,
                targets=lvl2_targets,
                offset=len(lvl1a))

        # we need as many targets as len(index)
        lvl0 = IndexLevelGO(index=category,
                targets=np.array((lvl1a, lvl1b)))


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




    def test_hierarchy_loc_to_iloc_a(self):


        groups = Index(('A', 'B', 'C'))
        dates = IndexDate.from_date_range('2018-01-01', '2018-01-04')
        observations = Index(('x', 'y'))


        lvl2a = IndexLevel(index=observations)
        lvl2b = IndexLevel(index=observations, offset=2)
        lvl2c = IndexLevel(index=observations, offset=4)
        lvl2d = IndexLevel(index=observations, offset=6)
        lvl2_targets = np.array((lvl2a, lvl2b, lvl2c, lvl2d))


        lvl1a = IndexLevel(index=dates,
                targets=lvl2_targets, offset=0)
        lvl1b = IndexLevel(index=dates,
                targets=lvl2_targets, offset=len(lvl1a))
        lvl1c = IndexLevel(index=dates,
                targets=lvl2_targets, offset=len(lvl1a) * 2)

        # we need as many targets as len(index)
        lvl0 = IndexLevel(index=groups,
                targets=np.array((lvl1a, lvl1b, lvl1c)))


        self.assertEqual(len(lvl2a), 2)
        self.assertEqual(len(lvl1a), 8)
        self.assertEqual(len(lvl0), 24)

        self.assertEqual(list(lvl2a.depths()),
                [1])
        self.assertEqual(list(lvl1a.depths()),
                [2, 2, 2, 2])
        self.assertEqual(list(lvl0.depths()),
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

        ih = IndexHierarchy(lvl0)

        self.assertEqual(len(ih), 24)

        post = ih.loc_to_iloc(HLoc[
                ['A', 'B', 'C'],
                slice('2018-01-01', '2018-01-04'),
                ['x', 'y']])
        # this will break if we recognize this can be a slice
        self.assertEqual(post, list(range(len(ih))))

        post = ih.loc_to_iloc(HLoc[
                ['A', 'B', 'C'],
                slice('2018-01-01', '2018-01-04'),
                'x'])

        self.assertEqual(post, list(range(0, len(ih), 2)))

        post = ih.loc_to_iloc(HLoc[
                'C',
                '2018-01-03',
                'y'])

        self.assertEqual(post, 21)

        post = ih.loc_to_iloc(HLoc['B', '2018-01-03':, 'y'])
        self.assertEqual(post, [13, 15])


        post = ih.loc_to_iloc(HLoc[['B', 'C'], '2018-01-03'])
        self.assertEqual(post, [12, 13, 20, 21])

        post = ih.loc_to_iloc(HLoc[['A', 'C'], :, 'y'])
        self.assertEqual(post, [1, 3, 5, 7, 17, 19, 21, 23])

        post = ih.loc_to_iloc(HLoc[['A', 'C'], :, 'x'])
        self.assertEqual(post, [0, 2, 4, 6, 16, 18, 20, 22])



    def test_hierarchy_from_product_a(self):

        groups = Index(('A', 'B', 'C'))
        dates = IndexDate.from_date_range('2018-01-01', '2018-01-04')
        observations = Index(('x', 'y'))

        ih = IndexHierarchy.from_product(groups, dates, observations)


    def test_hierarchy_from_tree_a(self):

        OD = OrderedDict

        tree = OD([('A', (1, 2, 3, 4)), ('B', (1, 2))])

        ih = IndexHierarchy.from_tree(tree)

        self.assertEqual(ih.to_frame().to_pairs(0),
                ((0, ((0, 'A'), (1, 'A'), (2, 'A'), (3, 'A'), (4, 'B'), (5, 'B'))), (1, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 1), (5, 2))))
                )


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

        ih = IndexHierarchy.from_tree(tree)
        self.assertEqual(ih.to_frame().to_pairs(0),
                ((0, ((0, 'I'), (1, 'I'), (2, 'I'), (3, 'I'), (4, 'I'), (5, 'I'), (6, 'I'), (7, 'II'), (8, 'II'), (9, 'II'), (10, 'II'))), (1, ((0, 'A'), (1, 'A'), (2, 'B'), (3, 'B'), (4, 'B'), (5, 'C'), (6, 'C'), (7, 'A'), (8, 'A'), (9, 'A'), (10, 'B'))), (2, ((0, 1), (1, 2), (2, 1), (3, 2), (4, 3), (5, 2), (6, 3), (7, 1), (8, 2), (9, 3), (10, 1))))
                )


    def test_hierarchy_from_labels_a(self):

        labels = (('I', 'A', 1),
                ('I', 'A', 2),
                ('I', 'B', 1),
                ('I', 'B', 2),
                ('II', 'A', 1),
                ('II', 'A', 2),
                ('II', 'B', 1),
                ('II', 'B', 2),
                )

        ih = IndexHierarchy.from_labels(labels)
        self.assertEqual(len(ih), 8)
        self.assertEqual(ih._depth, 3)


        labels = (('I', 'A', 1),
                ('I', 'A', 2),
                ('I', 'B', 1),
                ('II', 'B', 2),
                )

        ih = IndexHierarchy.from_labels(labels)
        self.assertEqual(len(ih), 4)
        self.assertEqual(ih._depth, 3)


    def test_hierarchy_from_labels_b(self):

        labels = (('I', 'A'),
                ('I', 'B'),
                )

        ih = IndexHierarchy.from_labels(labels)

        self.assertEqual(ih.to_frame().to_pairs(0),
                ((0, ((0, 'I'), (1, 'I'))), (1, ((0, 'A'), (1, 'B')))))

    def test_hierarchy_loc_to_iloc_b(self):
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

        ih = IndexHierarchy.from_tree(tree)

        post = ih.loc_to_iloc(HLoc['I', 'B', 1])
        self.assertEqual(post, 2)

        post = ih.loc_to_iloc(HLoc['I', 'B', 3])
        self.assertEqual(post, 4)

        post = ih.loc_to_iloc(HLoc['II', 'A', 3])
        self.assertEqual(post, 9)

        post = ih.loc_to_iloc(HLoc['II', 'A'])
        self.assertEqual(post, slice(7, 10))

        post = ih.loc_to_iloc(HLoc['I', 'C'])
        self.assertEqual(post, slice(5, 7))


        post = ih.loc_to_iloc(HLoc['I', ['A', 'C']])
        self.assertEqual(post, [0, 1, 5, 6])


        post = ih.loc_to_iloc(HLoc[:, 'A', :])
        self.assertEqual(post, [0, 1, 7, 8, 9])


        post = ih.loc_to_iloc(HLoc[:, 'C', 3])
        self.assertEqual(post, 6)


        post = ih.loc_to_iloc(HLoc[:, :, 3])
        self.assertEqual(post, [4, 6, 9])

        post = ih.loc_to_iloc(HLoc[:, :, 1])
        self.assertEqual(post, [0, 2, 7, 10])

        # TODO: not sure what to do when a multiple selection, [1, 2], is a superset of the leaf index; we do not match with a normal loc
        # ih.loc_to_iloc((slice(None), slice(None), [1,2]))


    def test_hierarchy_loc_to_iloc_c(self):
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

        ih = IndexHierarchy.from_tree(tree)

        # TODO: add additional validaton
        post = ih.loc[('I', 'B', 2): ('II', 'A', 2)]
        self.assertTrue(len(post), 6)

        post = ih.loc[[('I', 'B', 2), ('II', 'A', 2)]]
        self.assertTrue(len(post), 2)


    def test_hierarchy_contains_a(self):
        labels = (('I', 'A'),
                ('I', 'B'),
                )
        ih = IndexHierarchy.from_labels(labels)

        self.assertTrue(('I', 'A') in ih)


    def test_hierarchy_display_a(self):
        OD = OrderedDict
        tree = OD([
                ('I', OD([
                        ('A', (1, 2)), ('B', (1, 2))
                        ])
                ),
                ('II', OD([
                        ('A', (1, 2)), ('B', (1, 2))
                        ])
                ),
                ])

        ih = IndexHierarchy.from_tree(tree)

        post = ih.display()
        self.assertEqual(len(post), 10)

        s = Series(range(8), index=ih)
        post = s.display()
        self.assertEqual(len(post), 10)


    def test_hierarchy_loc_a(self):
        OD = OrderedDict
        tree = OD([
                ('I', OD([
                        ('A', (1, 2)), ('B', (1, 2))
                        ])
                ),
                ('II', OD([
                        ('A', (1, 2)), ('B', (1, 2))
                        ])
                ),
                ])

        ih = IndexHierarchy.from_tree(tree)

        s = Series(range(8), index=ih)

        self.assertEqual(
                s.loc[HLoc['I']].values.tolist(),
                [0, 1, 2, 3])

        self.assertEqual(
                s.loc[HLoc[:, 'A']].values.tolist(),
                [0, 1, 4, 5])

    def test_hierarchy_series_a(self):
        f = IndexHierarchy.from_tree
        s1 = Series(23, index=f(dict(a=(1,2,3))))
        self.assertEqual(s1.values.tolist(), [23, 23, 23])

        f = IndexHierarchy.from_product
        s2 = Series(3, index=f(Index(('a', 'b')), Index((1,2))))
        self.assertEqual(s2.to_pairs(),
                ((('a', 1), 3), (('a', 2), 3), (('b', 1), 3), (('b', 2), 3)))


    def test_hierarchy_frame_a(self):
        OD = OrderedDict
        tree = OD([
                ('I', OD([
                        ('A', (1,)), ('B', (1, 2))
                        ])
                ),
                ('II', OD([
                        ('A', (1,)), ('B', (1, 2))
                        ])
                ),
                ])

        ih = IndexHierarchy.from_tree(tree)

        data = np.arange(6*6).reshape(6, 6)
        f1 = Frame(data, index=ih, columns=ih)
        # self.assertEqual(len(f.to_pairs(0)), 8)


        f2 = f1.assign.loc[('I', 'B', 2), ('II', 'A', 1)](200)

        post = f2.to_pairs(0)
        self.assertEqual(post,
                ((('I', 'A', 1), ((('I', 'A', 1), 0), (('I', 'B', 1), 6), (('I', 'B', 2), 12), (('II', 'A', 1), 18), (('II', 'B', 1), 24), (('II', 'B', 2), 30))), (('I', 'B', 1), ((('I', 'A', 1), 1), (('I', 'B', 1), 7), (('I', 'B', 2), 13), (('II', 'A', 1), 19), (('II', 'B', 1), 25), (('II', 'B', 2), 31))), (('I', 'B', 2), ((('I', 'A', 1), 2), (('I', 'B', 1), 8), (('I', 'B', 2), 14), (('II', 'A', 1), 20), (('II', 'B', 1), 26), (('II', 'B', 2), 32))), (('II', 'A', 1), ((('I', 'A', 1), 3), (('I', 'B', 1), 9), (('I', 'B', 2), 200), (('II', 'A', 1), 21), (('II', 'B', 1), 27), (('II', 'B', 2), 33))), (('II', 'B', 1), ((('I', 'A', 1), 4), (('I', 'B', 1), 10), (('I', 'B', 2), 16), (('II', 'A', 1), 22), (('II', 'B', 1), 28), (('II', 'B', 2), 34))), (('II', 'B', 2), ((('I', 'A', 1), 5), (('I', 'B', 1), 11), (('I', 'B', 2), 17), (('II', 'A', 1), 23), (('II', 'B', 1), 29), (('II', 'B', 2), 35))))
        )


        f3 = f1.assign.loc[('I', 'B', 2):, HLoc[:, :, 2]](200)

        self.assertEqual(f3.to_pairs(0),
                ((('I', 'A', 1), ((('I', 'A', 1), 0), (('I', 'B', 1), 6), (('I', 'B', 2), 12), (('II', 'A', 1), 18), (('II', 'B', 1), 24), (('II', 'B', 2), 30))), (('I', 'B', 1), ((('I', 'A', 1), 1), (('I', 'B', 1), 7), (('I', 'B', 2), 13), (('II', 'A', 1), 19), (('II', 'B', 1), 25), (('II', 'B', 2), 31))), (('I', 'B', 2), ((('I', 'A', 1), 2), (('I', 'B', 1), 8), (('I', 'B', 2), 200), (('II', 'A', 1), 200), (('II', 'B', 1), 200), (('II', 'B', 2), 200))), (('II', 'A', 1), ((('I', 'A', 1), 3), (('I', 'B', 1), 9), (('I', 'B', 2), 15), (('II', 'A', 1), 21), (('II', 'B', 1), 27), (('II', 'B', 2), 33))), (('II', 'B', 1), ((('I', 'A', 1), 4), (('I', 'B', 1), 10), (('I', 'B', 2), 16), (('II', 'A', 1), 22), (('II', 'B', 1), 28), (('II', 'B', 2), 34))), (('II', 'B', 2), ((('I', 'A', 1), 5), (('I', 'B', 1), 11), (('I', 'B', 2), 200), (('II', 'A', 1), 200), (('II', 'B', 1), 200), (('II', 'B', 2), 200))))
        )



    def test_hierarchy_frame_b(self):
        OD = OrderedDict
        tree = OD([
                ('I', OD([
                        ('A', (1,)), ('B', (1, 2))
                        ])
                ),
                ('II', OD([
                        ('A', (1,)), ('B', (1, 2))
                        ])
                ),
                ])

        ih = IndexHierarchyGO.from_tree(tree)
        data = np.arange(6*6).reshape(6, 6)
        # TODO: this only works if own_columns is True for now
        f1 = FrameGO(data, index=range(6), columns=ih, own_columns=True)
        f1[('II', 'B', 3)] = 0

        f2 = f1[HLoc[:, 'B']]
        self.assertEqual(f2.shape, (6, 5))
        self.assertEqual(f2.to_pairs(0),
                ((('I', 'B', 1), ((0, 1), (1, 7), (2, 13), (3, 19), (4, 25), (5, 31))), (('I', 'B', 2), ((0, 2), (1, 8), (2, 14), (3, 20), (4, 26), (5, 32))), (('II', 'B', 1), ((0, 4), (1, 10), (2, 16), (3, 22), (4, 28), (5, 34))), (('II', 'B', 2), ((0, 5), (1, 11), (2, 17), (3, 23), (4, 29), (5, 35))), (('II', 'B', 3), ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0))))
                )

        f3 = f1[HLoc[:, :, 1]]
        self.assertEqual(f3.to_pairs(0), ((('I', 'A', 1), ((0, 0), (1, 6), (2, 12), (3, 18), (4, 24), (5, 30))), (('I', 'B', 1), ((0, 1), (1, 7), (2, 13), (3, 19), (4, 25), (5, 31))), (('II', 'A', 1), ((0, 3), (1, 9), (2, 15), (3, 21), (4, 27), (5, 33))), (('II', 'B', 1), ((0, 4), (1, 10), (2, 16), (3, 22), (4, 28), (5, 34)))))


        f4 = f1.loc[[2, 5], HLoc[:, 'A']]
        self.assertEqual(f4.to_pairs(0),
                ((('I', 'A', 1), ((2, 12), (5, 30))), (('II', 'A', 1), ((2, 15), (5, 33)))))



    def test_hierarchy_index_go_a(self):

        OD = OrderedDict
        tree1 = OD([
                ('I', OD([
                        ('A', (1,)), ('B', (1, 2))
                        ])
                ),
                ('II', OD([
                        ('A', (1,)), ('B', (1, 2))
                        ])
                ),
                ])
        ih1 = IndexHierarchyGO.from_tree(tree1)

        tree2 = OD([
                ('III', OD([
                        ('A', (1,)), ('B', (1, 2))
                        ])
                ),
                ('IV', OD([
                        ('A', (1,)), ('B', (1, 2))
                        ])
                ),
                ])
        ih2 = IndexHierarchyGO.from_tree(tree2)

        ih1.extend(ih2)

        # self.assertEqual(ih1.loc_to_iloc(('IV', 'B', 2)), 11)
        self.assertEqual(len(ih2), 6)

        # need tuple here to distinguish from iterable type selection
        self.assertEqual([ih1.loc_to_iloc(tuple(v)) for v in ih1.values],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                )



    def test_hierarchy_relabel_a(self):

        labels = (
                ('I', 'A'),
                ('I', 'B'),
                ('II', 'A'),
                ('II', 'B'),
                )

        ih = IndexHierarchy.from_labels(labels)

        ih.relabel({('I', 'B'): ('I', 'C')})

        ih2 = ih.relabel({('I', 'B'): ('I', 'C')})

        self.assertEqual(ih2.values.tolist(),
                [['I', 'A'], ['I', 'C'], ['II', 'A'], ['II', 'B']])

        with self.assertRaises(Exception):
            ih3 = ih.relabel({('I', 'B'): ('I', 'C', 1)})

        ih3 = ih.relabel(lambda x: tuple(e.lower() for e in x))

        self.assertEqual(
                ih3.values.tolist(),
                [['i', 'a'], ['i', 'b'], ['ii', 'a'], ['ii', 'b']])



    def test_hierarchy_intersection_a(self):

        labels = (
                ('I', 'A'),
                ('I', 'B'),
                ('II', 'A'),
                ('II', 'B'),
                )

        ih1 = IndexHierarchy.from_labels(labels)

        labels = (
                ('II', 'A'),
                ('II', 'B'),
                ('III', 'A'),
                ('III', 'B'),
                )

        ih2 = IndexHierarchy.from_labels(labels)

        post = ih1.intersection(ih2)
        self.assertEqual(post.values.tolist(),
                [['II', 'A'], ['II', 'B']])

        post = ih1.union(ih2)
        self.assertEqual(post.values.tolist(),
                [['I', 'A'], ['I', 'B'], ['II', 'A'], ['II', 'B'], ['III', 'A'], ['III', 'B']])

if __name__ == '__main__':
    unittest.main()


