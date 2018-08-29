
import unittest
import numpy as np

from collections import OrderedDict

from static_frame import Index
from static_frame import IndexGO
from static_frame import IndexDate
from static_frame import Series
from static_frame import Frame
from static_frame import IndexYearMonth
from static_frame import IndexYear

from static_frame import IndexHierarchy
from static_frame import IndexLevel
from static_frame import HLoc

from static_frame.test.test_case import TestCase

class TestUnit(TestCase):



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

        post = ih.loc_to_iloc((
                ['A', 'B', 'C'],
                slice('2018-01-01', '2018-01-04'),
                ['x', 'y']))
        # this will break if we recognize this can be a slice
        self.assertEqual(post, list(range(len(ih))))

        post = ih.loc_to_iloc((
                ['A', 'B', 'C'],
                slice('2018-01-01', '2018-01-04'),
                'x'))

        self.assertEqual(post, list(range(0, len(ih), 2)))

        post = ih.loc_to_iloc((
                'C',
                '2018-01-03',
                'y'))

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

        post = ih.loc_to_iloc(('I', 'B', 1))
        self.assertEqual(post, 2)

        post = ih.loc_to_iloc(('I', 'B', 3))
        self.assertEqual(post, 4)

        post = ih.loc_to_iloc(('II', 'A', 3))
        self.assertEqual(post, 9)

        post = ih.loc_to_iloc(('II', 'A', slice(None)))
        self.assertEqual(post, slice(7, 10))

        post = ih.loc_to_iloc(('I', 'C', slice(None)))
        self.assertEqual(post, slice(5, 7))


        post = ih.loc_to_iloc(('I', ['A', 'C'], slice(None)))
        self.assertEqual(post, [0, 1, 5, 6])


        post = ih.loc_to_iloc((slice(None), 'A', slice(None)))
        self.assertEqual(post, [0, 1, 7, 8, 9])


        post = ih.loc_to_iloc((slice(None), 'C', 3))
        self.assertEqual(post, 6)


        post = ih.loc_to_iloc((slice(None), slice(None), 3))
        self.assertEqual(post, [4, 6, 9])

        post = ih.loc_to_iloc((slice(None), slice(None), 1))
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

    def test_hierarchy_frame_a(self):
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

        data = np.arange(8*8).reshape(8, 8)
        f = Frame(data, index=ih, columns=ih)
        self.assertEqual(len(f.to_pairs(0)), 8)




if __name__ == '__main__':
    unittest.main()


