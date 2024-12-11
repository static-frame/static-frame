from __future__ import annotations

import copy
import datetime
import pickle
import unittest
from collections import OrderedDict
from functools import wraps
from hashlib import sha256

import frame_fixtures as ff
import numpy as np
import typing_extensions as tp

from static_frame import DisplayConfig
from static_frame import Frame
from static_frame import FrameGO
from static_frame import HLoc
from static_frame import ILoc
from static_frame import Index
from static_frame import IndexDate
from static_frame import IndexGO
from static_frame import IndexHierarchy
from static_frame import IndexHierarchyGO
from static_frame import IndexNanosecond
from static_frame import IndexNanosecondGO
from static_frame import IndexYear
from static_frame import IndexYearMonth
from static_frame import IndexYearMonthGO
from static_frame import Series
from static_frame import TypeBlocks
from static_frame.core.exception import ErrorInitIndex
from static_frame.core.exception import ErrorInitIndexNonUnique
from static_frame.core.index_auto import IndexAutoConstructorFactory
from static_frame.core.index_base import IndexBase
from static_frame.core.index_hierarchy import build_indexers_from_product
from static_frame.test.test_case import TestCase
from static_frame.test.test_case import skip_win
from static_frame.test.test_case import temp_file

SelfT = tp.TypeVar('SelfT')


def run_with_static_and_grow_only(func: tp.Callable[[SelfT, tp.Type[IndexHierarchy]], None]) -> tp.Callable[[SelfT], None]:
    '''
    Run a unit test using both `IndexHierarchy` and `IndexHierarchyGO`
    '''
    @wraps(func)
    def inner(self: SelfT) -> None:
        func(self, IndexHierarchy)
        func(self, IndexHierarchyGO)
    return inner


class TestUnit(TestCase):

    def _assert_to_tree_consistency(self, ih1: IndexHierarchy) -> None:
        # Ensure all IndexHierarchy's created using `from_tree` return the same tree using `to_tree`
        tree = ih1.to_tree()
        ih2 = IndexHierarchy.from_tree(tree)
        self.assertTrue(ih1.equals(ih2))

    #--------------------------------------------------------------------------

    def test_hierarchy_slotted_a(self) -> None:

        labels = (('I', 'A'), ('I', 'B'))
        ih1 = IndexHierarchy.from_labels(labels, name='foo')

        with self.assertRaises(AttributeError):
            ih1.g = 30 # type: ignore #pylint: disable=E0237
        with self.assertRaises(AttributeError):
            ih1.__dict__ #pylint: disable=W0104

    def test_hierarchy_init_a(self) -> None:

        labels = (('I', 'A'), ('I', 'B'))

        ih1 = IndexHierarchy.from_labels(labels, name='foo')
        ih2 = IndexHierarchy(ih1)
        self.assertEqual(ih1.name, 'foo')
        self.assertEqual(ih2.name, 'foo')

    def test_hierarchy_init_b(self) -> None:

        labels = (
                ('I', 'A'),
                ('I', 'B'),
                ('II', 'B'),
                ('I', 'C')
                )

        ih = IndexHierarchy.from_labels(labels)
        self.assertEqual(tuple(ih.iter_label()), labels)

    def test_hierarchy_init_c(self) -> None:

        labels = (
                ('I', 'A'),
                ('I', 'B'),
                ('II', 'B'),
                ('III', 'B'),
                ('III', 'A')
                )

        ih1 = IndexHierarchy.from_labels(labels)
        self.assertEqual(ih1.values.tolist(),
            [['I', 'A'], ['I', 'B'], ['II', 'B'], ['III', 'B'], ['III', 'A']])

    def test_hierarchy_init_d(self) -> None:

        labels = (
                ('I', 'A'),
                ('I', 'B'),
                ('II', 'B'),
                ('III', 'B'),
                ('III', 'B')
                )
        with self.assertRaises(ErrorInitIndex):
            ih1 = IndexHierarchy.from_labels(labels)

    def test_hierarchy_init_e(self) -> None:

        labels = (
                ('I', 'A'),
                ('I', 'B'),
                ('II', 'B'),
                ('III', 'B'),
                ('I', 'B'),
                )

        with self.assertRaises(RuntimeError):
            ih1 = IndexHierarchy.from_labels(labels)

    def test_hierarchy_init_f(self) -> None:

        labels = (
                ('I', 'A'),
                ('I', 'B'),
                ('II', 'B'),
                ('III', 'B'),
                ('I', 'B'),
                )

        with self.assertRaises(ErrorInitIndexNonUnique):
            ih1 = IndexHierarchy.from_labels(labels)

    def test_hierarchy_init_g(self) -> None:

        labels = (
                ('I', 'A', 1),
                ('I', 'B', 1),
                ('II', 'A', 1),
                ('II', 'A', 2),
                ('II', 'A', 1),
                )
        with self.assertRaises(ErrorInitIndex):
            ih1 = IndexHierarchy.from_labels(labels)

    def test_hierarchy_init_h(self) -> None:

        labels = (
                ('I', 'A', 1),
                ('I', 'B', 1),
                ('II', 'A', 1),
                ('II', 'A', 2),
                ('II', 'B', 1),
                ('II', 'A', 3),
                )

        ih1 = IndexHierarchy.from_labels(labels)
        self.assertEqual(tuple(ih1.iter_label()), labels)

        ih2 = IndexHierarchy.from_labels(labels, reorder_for_hierarchy=True)
        self.assertEqual(ih1.shape, ih2.shape)
        self.assertEqual(tuple(ih2.iter_label()),
                (('I', 'A', 1), ('I', 'B', 1), ('II', 'A', 1), ('II', 'A', 2), ('II', 'A', 3), ('II', 'B', 1))
                )

    def test_hierarchy_init_i(self) -> None:
        with self.assertRaises(RuntimeError):
            ih1 = IndexHierarchy((3,))  # type: ignore

    def test_hierarchy_init_j(self) -> None:

        labels = (('I', 'A'), ('I', 'B'))

        ih1 = IndexHierarchy.from_labels(labels, name=('a', 'b', 'c'))

        # can access as a .name, but not a .names
        self.assertEqual(ih1.name, ('a', 'b', 'c'))
        # names does not use name as it is the wrong size
        self.assertEqual(ih1.names, ('__index0__', '__index1__'))

    def test_hierarchy_init_k(self) -> None:

        labels = (('I', 'A'), ('I', 'B'))
        ih1 = IndexHierarchy.from_labels(labels, name='foo')

        # Cannot provide blocks in this case
        with self.assertRaises(ErrorInitIndex):
            _ = IndexHierarchy(ih1, blocks=ih1._blocks)

        # Cannot provide indexers in this case
        with self.assertRaises(ErrorInitIndex):
            _ = IndexHierarchy(ih1, indexers=ih1._indexers)

        ih2 = IndexHierarchy(ih1)
        self.assertTrue(ih2.equals(ih1, compare_dtype=True))

    def test_hierarchy_init_l(self) -> None:

        indices = [Index(tuple('ABC')) for _ in range(2)]
        indexers = [[0, 1, 2], [0, 1, 2]]

        # Indexers must be numpy arrays
        with self.assertRaises(ErrorInitIndex):
            IndexHierarchy(indices, indexers=indexers)

        indexers = [np.array(indexer) for indexer in indexers]

        # Indexers must be read-only
        with self.assertRaises(ErrorInitIndex):
            IndexHierarchy(indices, indexers=indexers)

    def test_hierarchy_init_m(self) -> None:

        indices: tp.List[Index] = []
        indexers = np.array([[0, 1, 2], [0, 1, 2]])
        indexers.flags.writeable = False

        # Indices must be have more than 1 element
        with self.assertRaises(ErrorInitIndex):
            IndexHierarchy(indices, indexers=indexers)

        indices = [Index(tuple('ABC'))]

        # Indices must be have more than 1 element
        with self.assertRaises(ErrorInitIndex):
            IndexHierarchy(indices, indexers=indexers)

        indices = [Index(tuple('ABC')), Index(range(3))]
        IndexHierarchy(indices, indexers=indexers)

    #---------------------------------------------------------------------------

    def test_hierarchy_mloc_a(self) -> None:

        labels = (('I', 'A'), ('I', 'B'))

        ih1 = IndexHierarchy.from_labels(labels, name='foo')
        # per type block size
        self.assertEqual(ih1.size, 4)

        ih2 = IndexHierarchy(ih1)
        self.assertEqual(ih2.mloc.tolist(), ih1.mloc.tolist())

    def test_hierarchy_mloc_b(self) -> None:

        labels = (('I', 'A'), ('I', 'B'))

        ih1 = IndexHierarchy.from_labels(labels, name='foo')
        post = ih1.mloc
        self.assertEqual(post.tolist(), ih1._blocks.mloc.tolist())

    def test_hierarchy_mloc_c(self) -> None:

        labels = (('I', 'A'), ('I', 'B'))

        ih1 = IndexHierarchyGO.from_labels(labels, name='foo')
        ih1.append(('I', 'C'))
        post = ih1.mloc
        self.assertEqual(post.tolist(), ih1._blocks.mloc.tolist())

        ih1.append(('I', 'D'))
        self.assertEqual(ih1.size, 8)

    def test_hierarchy_size_a(self) -> None:

        labels = (('I', 'A'), ('I', 'B'))

        ih1 = IndexHierarchy.from_labels(labels, name='foo')
        self.assertTrue(ih1.nbytes > 500)

    def test_hierarchy_size_b(self) -> None:

        labels = (('I', 'A'), ('I', 'B'))

        ih1 = IndexHierarchyGO.from_labels(labels, name='foo')
        ih1.append(('I', 'C'))
        self.assertTrue(ih1.nbytes > 500)

    def test_hierarchy_bool_a(self) -> None:

        labels = (('I', 'A'), ('I', 'B'))

        ih1 = IndexHierarchy.from_labels(labels, name='foo')

        with self.assertRaises(ValueError):
            bool(ih1)

    #---------------------------------------------------------------------------

    def test_hierarchy_loc_to_iloc_a(self) -> None:
        groups = Index(('A', 'B', 'C'))
        dates = IndexDate.from_date_range('2018-01-01', '2018-01-04')
        observations = Index(('x', 'y'))

        ih = IndexHierarchy.from_product(groups, dates, observations)

        self.assertEqual(len(ih), 24)

        post = ih._loc_to_iloc(HLoc[
                ['A', 'B', 'C'],
                slice('2018-01-01', '2018-01-04'),
                np.array(['x', 'y'])])

        self.assertEqual(list(post), list(range(len(ih)))) # type: ignore

        post = ih._loc_to_iloc(HLoc[
                ['A', 'B', 'C'],
                slice('2018-01-01', '2018-01-04', 2), # selects across full region from first start
                np.array(['x', 'y'])])

        self.assertEqual(list(post), [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]) # type: ignore

        post = ih._loc_to_iloc(HLoc[
                ['A', 'B', 'C'],
                slice('2018-01-01', '2018-01-04'),
                ['x', 'y']])

        self.assertEqual(list(post), list(range(len(ih)))) # type: ignore

        post = ih._loc_to_iloc(HLoc[
                ['A', 'B', 'C'],
                slice(None, '2018-01-04'),
                'x'])

        self.assertEqual(list(post), list(range(0, len(ih), 2))) # type: ignore

        post = ih._loc_to_iloc(HLoc['C', '2018-01-03', 'y'])
        self.assertEqual(post, 21)

        post = ih._loc_to_iloc(HLoc['B', '2018-01-03':, 'y'])  # type: ignore
        # NOTE: we assume that the depth 1 selection is independent of 'B' selection
        self.assertEqual(list(post), [13, 15]) # type: ignore

        post = ih._loc_to_iloc(HLoc[['B', 'C'], '2018-01-03'])
        self.assertEqual(list(post), [12, 13, 20, 21]) # type: ignore

        post = ih._loc_to_iloc(HLoc[['A', 'C'], :, 'y'])
        self.assertEqual(list(post), [1, 3, 5, 7, 17, 19, 21, 23]) # type: ignore

        post = ih._loc_to_iloc(HLoc[['A', 'C'], :, 'x'])
        self.assertEqual(list(post), [0, 2, 4, 6, 16, 18, 20, 22]) # type: ignore

    def test_hierarchy_loc_to_iloc_b(self) -> None:
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
        self._assert_to_tree_consistency(ih)

        post = ih._loc_to_iloc(HLoc['I', 'B', 1])
        self.assertEqual(post, 2)

        post = ih._loc_to_iloc(HLoc['I', 'B', 3])
        self.assertEqual(post, 4)

        post = ih._loc_to_iloc(HLoc['II', 'A', 3])
        self.assertEqual(post, 9)

        post = ih._loc_to_iloc(HLoc['II', 'A'])
        self.assertEqual(list(post), [7, 8, 9])

        post = ih._loc_to_iloc(HLoc['I', 'C'])
        self.assertEqual(list(post), [5, 6])

        post = ih._loc_to_iloc(HLoc['I', ['A', 'C']])
        self.assertEqual(list(post), [0, 1, 5, 6])

        post = ih._loc_to_iloc(HLoc[:, 'A', :])
        self.assertEqual(list(post), [0, 1, 7, 8, 9])

        post = ih._loc_to_iloc(HLoc[:, 'C', 3])
        self.assertEqual(post, [6])

        post = ih._loc_to_iloc(HLoc[:, :, 3])
        self.assertEqual(list(post), [4, 6, 9])

        post = ih._loc_to_iloc(HLoc[:, :, 1])
        self.assertEqual(list(post), [0, 2, 7, 10])

        self.assertEqual(
                list(ih._loc_to_iloc(HLoc[:, :, [1, 2]])),
                [0, 1, 2, 3, 5, 7, 8, 10]
                )

    def test_hierarchy_loc_to_iloc_c(self) -> None:
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
        self._assert_to_tree_consistency(ih)

        # TODO: add additional validaton
        post1 = ih.loc[('I', 'B', 2): ('II', 'A', 2)]  # type: ignore
        self.assertTrue(len(post1), 6)

        post2 = ih.loc[[('I', 'B', 2), ('II', 'A', 2)]]
        self.assertTrue(len(post2), 2)

    def test_hierarchy_loc_to_iloc_d(self) -> None:

        labels = (
                ('I', 'A', 1),
                ('I', 'B', 1),
                ('II', 'A', 1),
                ('II', 'A', 2),
                ('II', 'B', 1),
                ('II', 'B', 2),
                )

        ih = IndexHierarchy.from_labels(labels)

        # selection with an Index objext
        iloc1 = ih._loc_to_iloc(Index((labels[2], labels[5])))
        self.assertEqual(iloc1, [2, 5])

        iloc2 = ih._loc_to_iloc(Index(labels))
        self.assertEqual(iloc2, [0, 1, 2, 3, 4, 5])

    def test_hierarchy_loc_to_iloc_e(self) -> None:

        labels = (
                ('I', 'A', 1),
                ('I', 'B', 1),
                ('II', 'A', 1),
                ('II', 'A', 2),
                ('II', 'B', 1),
                ('II', 'B', 2),
                )

        ih1 = IndexHierarchy.from_labels(labels)

        ih2 = IndexHierarchy.from_labels(labels[:3])
        ih3 = IndexHierarchy.from_labels(labels[-3:])

        # selection with an IndexHierarchy
        self.assertEqual(ih1._loc_to_iloc(ih2), [0, 1, 2])
        self.assertEqual(ih1._loc_to_iloc(ih3), [3, 4, 5])

        # Depth 2 != 3
        sel = IndexHierarchy.from_labels([(0, 1)])
        with self.assertRaises(KeyError):
            ih1._loc_to_iloc(sel)

        # Depth 4 != 3
        sel = IndexHierarchy.from_labels([(0, 1, 2, 3)])
        with self.assertRaises(KeyError):
            ih1._loc_to_iloc(sel)

    def test_hierarchy_loc_to_iloc_f(self) -> None:

        labels = (
                ('I', 'A', 1),
                ('I', 'B', 1),
                ('II', 'A', 1),
                ('II', 'A', 2),
                ('II', 'B', 1),
                ('II', 'B', 2),
                )

        ih1 = IndexHierarchy.from_labels(labels)

        # selection with Boolean and non-Bolean Series
        a1 = ih1._loc_to_iloc(Series((True, True), index=(labels[1], labels[4])))
        self.assertEqual(a1.tolist(), [1, 4]) # type: ignore

        a2 = ih1._loc_to_iloc(Series((labels[5], labels[2], labels[4])))
        self.assertEqual(a2, [5, 2, 4])

    def test_hierarchy_loc_to_iloc_g(self) -> None:

        labels = (
                ('I', 'A', -1),
                ('I', 'B', 1),
                ('II', 'A', 1),
                ('II', 'C', 2),
                ('III', 'B', 1),
                ('III', 'C', 3),
                )

        ih1 = IndexHierarchy.from_labels(labels)

        self.assertEqual(
                list(ih1._loc_to_iloc(HLoc[slice(None), ['A', 'C']])),
                [0, 2, 3, 5]
                )

        self.assertEqual(
                list(ih1._loc_to_iloc(HLoc[slice(None), ['A', 'C'], [-1, 3]])),
                [0, 5]
                )

    def test_hierarchy_loc_to_iloc_h(self) -> None:

        labels = (
                ('I', 'A', -1),
                ('I', 'B', 1),
                ('II', 'A', 1),
                ('II', 'C', 2),
                ('III', 'B', 1),
                ('III', 'C', 2),
                ('III', 'C', 3),
                )

        ih1 = IndexHierarchy.from_labels(labels)
        sel1 = ih1.values_at_depth(1) == 'C'
        post1 = ih1._loc_to_iloc(HLoc[slice(None), sel1])
        self.assertEqual(list(post1), [3, 5, 6])

        sel2 = ih1.values_at_depth(2) == 3
        post2 = ih1._loc_to_iloc(HLoc[slice(None), slice(None), sel2])
        self.assertEqual(post2, [6])

    def test_hierarchy_loc_to_iloc_i(self) -> None:

        labels = (
                ('I', 'A', -1),
                ('I', 'B', 1),
                ('II', 'A', 3),
                ('II', 'C', 2),
                ('III', 'B', 1),
                ('III', 'C', 2),
                ('III', 'C', 3),
                )

        ih1 = IndexHierarchy.from_labels(labels)

        post1 = ih1._loc_to_iloc(ILoc[4])
        self.assertEqual(post1, 4)

        # ILoc context is outermost, not local
        post1 = ih1._loc_to_iloc(HLoc[slice(None), ILoc[[0, -1]], 3])
        self.assertEqual(post1, [6])

        post2 = ih1._loc_to_iloc(HLoc[['I', 'III'], 'B', 1])
        self.assertEqual(list(post2), [1, 4])

    def test_hierarchy_loc_to_iloc_j(self) -> None:

        labels = (
                ('I', 'X', 1),
                ('I', 'X', 2),
                ('I', 'W', 0),
                ('I', 'W', 1),
                ('II', 'R', 1),
                ('II', 'R', 2),
                ('II', 'P', 0),
                ('II', 'P', 1),
                )
        ih1 = IndexHierarchy.from_labels(labels)

        self.assertEqual(list(ih1._loc_to_iloc(HLoc[ILoc[-4:], :, 1])), [4, 7])
        self.assertEqual(list(ih1._loc_to_iloc(HLoc[:, :, 1])), [0, 3, 4, 7])
        self.assertEqual(list(ih1._loc_to_iloc(HLoc[:, :, ILoc[-2:]])), [6, 7])
        self.assertEqual(list(ih1._loc_to_iloc(HLoc[:, ILoc[2:6], 1])), [3, 4])

    def test_hierarchy_loc_to_iloc_k(self) -> None:

        labels = (
                ('I', 'X', 1),
                ('I', 'X', 2),
                ('II', 'R', 0),
                ('II', 'R', 1),
                ('II', 'R', 2),
                ('II', 'R', 3),
                ('II', 'B', 3),
                ('II', 'A', 2),
                )
        ih1 = IndexHierarchy.from_labels(labels)

        post1 = ih1._loc_to_iloc(HLoc['II', ILoc[-5:], [2, 3]])
        self.assertEqual(list(post1), list(range(4, 8)))

        post2 = ih1._loc_to_iloc(HLoc[:, :, ILoc[-4]])
        self.assertEqual(post2, 4)

    def test_hierarchy_loc_to_iloc_m(self) -> None:
        idx = Index(range(20), loc_is_iloc=True)
        idx_alt = Index(range(20))

        tree = {'a':idx, 'b':idx}
        ih1 = IndexHierarchy.from_tree(tree)

        tree_alt = {'a':idx_alt, 'b':idx_alt}
        ih1_alt = IndexHierarchy.from_tree(tree_alt)

        post1 = ih1._loc_to_iloc(HLoc['b'])
        self.assertEqual(list(post1), list(ih1_alt._loc_to_iloc(HLoc['b'])))
        self.assertEqual(list(post1), list(range(20, 40)))

        post2 = ih1._loc_to_iloc(HLoc['b', 10:12])
        self.assertEqual(list(post2), list(ih1_alt._loc_to_iloc(HLoc['b', 10:12])))
        self.assertEqual(list(post2), [30, 31, 32])

        post3 = ih1._loc_to_iloc(HLoc['b', [0, 10, 19]])
        self.assertEqual(list(post3), list(ih1_alt._loc_to_iloc(HLoc['b', [0, 10, 19]])))
        self.assertEqual(list(post3), [20, 30, 39])

        post4 = ih1._loc_to_iloc(HLoc['b', 11])
        self.assertEqual(post4, ih1_alt._loc_to_iloc(HLoc['b', 11]))
        self.assertEqual(post4, 31)

        post5 = ih1._loc_to_iloc(
                HLoc['b', ~(ih1.values_at_depth(1) % 3).astype(bool)])
        self.assertEqual(list(post5), list(ih1_alt._loc_to_iloc(
                HLoc['b', ~(ih1.values_at_depth(1) % 3).astype(bool)])))
        self.assertEqual(list(post5), [20, 23, 26, 29, 32, 35, 38])

        post6 = ih1._loc_to_iloc(HLoc['b', np.array([0, 10, 19])])
        self.assertEqual(list(post6), list(ih1_alt._loc_to_iloc(HLoc['b', np.array([0, 10, 19])])))
        self.assertEqual(list(post6), [20, 30, 39])

    def test_hierarchy_loc_to_iloc_n(self) -> None:
        idx = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        post = idx._loc_to_iloc(np.array([False, True, False, True]))
        self.assertEqual(post.tolist(), [1, 3]) #type: ignore

    def test_hierarchy_loc_to_iloc_o(self) -> None:
        idx1 = Index(range(3), loc_is_iloc=True)
        idx2 = Index(('a', 'b', 'c'))
        ih1 = IndexHierarchy.from_product(idx2, idx1)

        self.assertEqual(ih1.loc_to_iloc(('b', 1)), 4)

        self.assertEqual(ih1.loc_to_iloc(slice(('b', 1), ('c', 1))),
                slice(4, 8, None))

        self.assertEqual(ih1.loc_to_iloc([('a', 1), ('b', 1), ('c', 0)]),
                [1, 4, 6])

        self.assertEqual(ih1.loc_to_iloc(ih1.values_at_depth(1) == 1).tolist(), #type: ignore
                [1, 4, 7])

    def test_hierarchy_loc_to_iloc_p(self) -> None:
        idx1 = Index(range(3), loc_is_iloc=True)
        idx2 = Index(('a', 'b', 'c'))
        ih1 = IndexHierarchy.from_product(idx2, idx1)

        with self.assertRaises(TypeError):
            ih1._loc_to_iloc(HLoc[slice(None, None, ('a', 2))])

        with self.assertRaises(TypeError):
            ih1._loc_to_iloc(slice(None, None, ('a', 2)))

        self.assertEqual(ih1.loc_to_iloc(('b', 1)), 4)

        self.assertEqual(ih1.loc_to_iloc(slice(('b', 1), ('c', 1))),
                slice(4, 8, None))

        self.assertEqual(ih1.loc_to_iloc([('a', 1), ('b', 1), ('c', 0)]),
                [1, 4, 6])

        self.assertEqual(ih1.loc_to_iloc(ih1.values_at_depth(1) == 1).tolist(), #type: ignore
                [1, 4, 7])

    def test_hierarchy_loc_to_iloc_q(self) -> None:
        labels = (('I', 'A'), ('I', 'B'), ('II', 'A'), ('III', 'B'))
        ih = IndexHierarchy.from_labels(labels, name='foo')

        with self.assertRaises(TypeError):
            ih._loc_to_iloc(HLoc[slice('I', 'III', '?')])

    def test_hierarchy_loc_to_iloc_r(self) -> None:
        labels = [
                (1, 'dd', 0),
                (1, 'b', 0),
                (2, 'cc', 0),
                (2, 'ee', 0),
                ]

        ih = IndexHierarchy.from_labels(labels)
        selections = [
                ih.loc[HLoc[1,'dd']],
                ih.loc[HLoc[(1,'dd')]],

                ih.loc[HLoc[[1],'dd']],
                ih.loc[HLoc[([1],'dd')]],

                ih.loc[HLoc[1,['dd']]],
                ih.loc[HLoc[(1,['dd'])]],

                ih.loc[HLoc[[1],['dd']]],
                ih.loc[HLoc[([1],['dd'])]],
                ]

        for i in range(len(selections) - 1):
            for j in range(i, len(selections)):
                self.assertTrue(selections[i].equals(selections[j]), msg=(i, j))

    def test_hierarchy_loc_to_iloc_s(self) -> None:

        # https://github.com/static-frame/static-frame/issues/554
        ih = ff.parse('v(bool)|i((I,ID),(int,dtD))|s(4,4)').index.sort()
        start = ih.values_at_depth(1)[0]
        end = ih.values_at_depth(1)[-1]

        post = ih._loc_to_iloc(HLoc[:, start:end])
        self.assertListEqual(list(post), [0, 1, 2, 3])

        ih = IndexHierarchy.from_labels(
            [
                [4, 0],
                [2, 1],
                [0, 2],
                [3, 3],
                [1, 4],
            ]
        )

        ih2 = ih.sort()

        self.assertListEqual(
            list(ih2), [
                (0, 2),
                (1, 4),
                (2, 1),
                (3, 3),
                (4, 0),
            ]
        )

        for idx1, idx2 in zip(ih._indices, ih2._indices):
            self.assertTrue(idx1.equals(idx2))

        post = ih2._loc_to_iloc(HLoc[:, 4:1])
        self.assertListEqual(list(post), [1, 2])

    def test_hierarchy_loc_to_iloc_t(self) -> None:
        # https://github.com/static-frame/static-frame/issues/610
        ih = IndexHierarchy.from_labels(
            [
                ("a", ("b", "c")),
                ("a", "d"),
            ]
        )

        assert 0 == ih.loc_to_iloc(("a", ("b", "c")))
        assert [0] == ih.loc_to_iloc([("a", ("b", "c"))])
        assert 1 == ih.loc_to_iloc(("a", "d"))
        assert [1] == ih.loc_to_iloc([("a", "d")])

    def test_hierarchy_loc_to_iloc_u(self) -> None:
        labels1 = (
                ('c_II', 'B', 1),
                ('c_II', 'A', 1),
                ('c_I', 'C', 1),
                ('c_I', 'B', 1),
                ('c_I', 'A', 1),
                ('c_I', 'D', 1),
        )

        ih = IndexHierarchy.from_labels(labels1)
        post = ih._loc_to_iloc(ih)

        self.assertEqual(list(post), [0, 1, 2, 3, 4, 5])

    def test_hierarchy_loc_to_iloc_v(self) -> None:
        labels1 = (
                (0, 0, 1),
                (0, 0, 2),
                (1, 0, 3),
                (0, 1, 1),
                (0, 1, 2),
                (1, 1, 3),
        )
        ih = IndexHierarchy.from_labels(labels1)
        post1: list[int] = ih._loc_to_iloc(ih.values.astype(object))
        post2: list[int] = ih._loc_to_iloc(ih.values).tolist()

        self.assertListEqual(post1, list(range(6)))
        self.assertListEqual(sorted(post2), post1)  # No guarantee of order for post2

    #---------------------------------------------------------------------------

    def test_hierarchy_loc_to_iloc_index_hierarchy_a(self) -> None:
        ih1 = IndexHierarchy.from_labels([(0, 1), (0, 2), (1, 0), (1, 1)])
        ih2 = IndexHierarchy.from_labels([(1, 1), (0, 2), (0, 1)])

        post = ih1._loc_to_iloc_index_hierarchy(ih2)

        self.assertListEqual(post, [3, 1, 0])

    def test_hierarchy_loc_to_iloc_index_hierarchy_b(self) -> None:
        ih1 = IndexHierarchy.from_labels([(1, 1), (0, 2), (0, 1)])

        post = ih1._loc_to_iloc_index_hierarchy(ih1)

        self.assertListEqual(post, [0, 1, 2])

    def test_hierarchy_loc_to_iloc_index_hierarchy_c(self) -> None:
        ih1 = IndexHierarchy.from_labels([(1, 1), (0, 2), (0, 1)])

        post = ih1._loc_to_iloc_index_hierarchy(ih1.iloc[1:])

        self.assertListEqual(post, [1, 2])

    def test_hierarchy_loc_to_iloc_index_hierarchy_d(self) -> None:
        ih1 = IndexHierarchy.from_labels([(1, 1)])
        ih2 = IndexHierarchyGO.from_labels([(2, 2)])
        ih2.append((3, 3))

        with self.assertRaises(KeyError):
            ih1._loc_to_iloc_index_hierarchy(ih2)

    def test_hierarchy_loc_to_iloc_index_hierarchy_e(self) -> None:
        ih1 = IndexHierarchyGO.from_labels([(0, 0)])
        ih2 = IndexHierarchyGO.from_labels([(0, 0)])
        ih1.append((1, 1))
        ih2.append((0, 1)) # Valid theoretical combination from ih1, except it doesn't actually exist

        with self.assertRaises(KeyError):
            ih1._loc_to_iloc_index_hierarchy(ih2)

    def test_hierarchy_loc_to_iloc_index_hierarchy_f(self) -> None:
        labels1 = (
                ('c_II', 'B', 1),
                ('c_II', 'A', 1),
                ('c_I', 'C', 1),
                ('c_I', 'B', 1),
                ('c_I', 'A', 1),
                ('c_I', 'D', 1),
        )
        labels2 = (
                ('c_I', 'A', 1),
                ('c_I', 'B', 1),
                ('c_I', 'C', 1),
                ('c_II', 'A', 1),
                ('c_II', 'B', 1),
        )

        ih1 = IndexHierarchy.from_labels(labels1)
        ih2 = IndexHierarchy.from_labels(labels2)

        post = ih1._loc_to_iloc_index_hierarchy(ih2)

        self.assertListEqual(post, [4, 3, 2, 1, 0])

    #---------------------------------------------------------------------------

    def test_hierarchy_extract_iloc_a(self) -> None:

        labels = (
                ('I', 'A', 1),
                ('I', 'B', 1),
                ('II', 'A', 1),
                ('II', 'A', 2),
                ('II', 'B', 1),
                ('II', 'B', 2),
                )

        ih1 = IndexHierarchy.from_labels(labels)

        ih2 = ih1._extract_iloc(slice(None)) # will get a copy
        self.assertTrue((ih1.values == ih2.values).all())
        # reduces to a tuple
        ih3 = ih1._extract_iloc(3)
        self.assertEqual(ih3, ('II', 'A', 2))

    def test_hierarchy_extract_iloc_b(self) -> None:

        labels = (
                ('I', 'A', 1),
                ('I', 'B', 1),
                ('II', 'A', 1),
                ('II', 'A', 2),
                ('II', 'B', 1),
                ('II', 'B', 2),
                )

        ih1 = IndexHierarchyGO.from_labels(labels)
        ih2 = ih1.iloc[:0]
        self.assertEqual(ih1.depth, ih2.depth)

        assert isinstance(ih2, IndexHierarchyGO)

        ih2.append(('a', 'b', 'c'))
        ih2.append(('a', 'b', 'd'))
        self.assertEqual(ih2.shape, (2, 3))

        self.assertEqual(ih2.dtypes.values.tolist(),
                [np.dtype('<U1'), np.dtype('<U1'), np.dtype('<U1')])

        self.assertEqual(ih2.values.tolist(),
                [['a', 'b', 'c'], ['a', 'b', 'd']]
                )

    def test_hierarchy_extract_iloc_c(self) -> None:

        labels = (
                ('I', 'A', 1),
                ('I', 'B', 1),
                ('II', 'A', 1),
                ('II', 'A', 2),
                ('II', 'B', 1),
                ('II', 'B', 2),
                )
        ih1 = IndexHierarchy.from_labels(labels)
        ih2 = ih1[:0]
        assert isinstance(ih2, IndexHierarchy)
        self.assertEqual(ih2._blocks.shape, (0, 3))
        self.assertEqual(ih2.shape, (0, 3))

    def test_hierarchy_extract_iloc_d(self) -> None:

        labels = (
                ('I', 'A', 1),
                ('I', 'B', 1),
                ('II', 'A', 1),
                ('II', 'A', 2),
                ('II', 'B', 1),
                ('II', 'B', 2),
                )

        ih1 = IndexHierarchy.from_labels(labels)

        post = ih1._extract_iloc(None)
        self.assertTrue(post.equals(ih1))
        self.assertIs(post, ih1)

    def test_hierarchy_extract_iloc_e(self) -> None:
        ih1 = IndexHierarchyGO.from_labels((('a', 'a'), ('b','b')))
        ih2 = ih1[None]
        ih2.append(('c', 'c'))
        self.assertEqual(ih1.values.tolist(),
                [['a', 'a'], ['b', 'b']])

        self.assertEqual(ih2.values.tolist(),
                [['a', 'a'], ['b', 'b'], ['c', 'c']])

    #---------------------------------------------------------------------------

    def test_hierarchy_extract_getitem_astype_a(self) -> None:

        labels = (
                ('I', 'A', 1),
                ('I', 'B', 1),
                ('II', 'A', 1),
                ('II', 'A', 2),
                ('II', 'B', 1),
                ('II', 'B', 2),
                )

        ih1 = IndexHierarchy.from_labels(labels)

        with self.assertRaises(KeyError):
            ih1._extract_getitem_astype(('A', 1))

    def test_hierarchy_extract_getitem_astype_b(self) -> None:

        labels = (
                ('1', '3', 1),
                ('1', '4', 1),
                ('11', '3', 1),
                ('11', '3', 2),
                ('11', '4', 1),
                ('11', '4', 2),
                )

        ih1 = IndexHierarchy.from_labels(labels)
        ih2 = ih1.astype[:2](int, consolidate_blocks=True)
        self.assertTrue(ih2._blocks.unified)

    #--------------------------------------------------------------------------

    def test_hierarchy_from_product_a(self) -> None:

        groups = Index(('A', 'B', 'C'))
        dates = IndexDate.from_date_range('2018-01-01', '2018-01-04')
        observations = Index(('x', 'y'))

        ih = IndexHierarchy.from_product(groups, dates, observations)

    def test_hierarchy_from_product_b(self) -> None:

        with self.assertRaises(RuntimeError):
            IndexHierarchy.from_product((1, 2))

    def test_hierarchy_from_product_c(self) -> None:

        groups = ('A', 'B', 'C')
        dates = ('2018-01-01', '2018-01-04')

        with self.assertRaises(ErrorInitIndex):
            # mis-matched length
            _ = IndexHierarchy.from_product(groups, dates, index_constructors=(Index,))

        ih = IndexHierarchy.from_product(groups, dates, index_constructors=(Index, IndexDate))

        self.assertEqual(ih.index_types.values.tolist(), [Index, IndexDate])
        self.assertEqual(ih.values.tolist(),
                [['A', datetime.date(2018, 1, 1)], ['A', datetime.date(2018, 1, 4)], ['B', datetime.date(2018, 1, 1)], ['B', datetime.date(2018, 1, 4)], ['C', datetime.date(2018, 1, 1)], ['C', datetime.date(2018, 1, 4)]])

    def test_hierarchy_from_product_d(self) -> None:

        groups = ('2021-01-01', '2021-01-02')
        dates = ('2018-01-01', '2018-01-04')

        ih = IndexHierarchy.from_product(groups, dates, index_constructors=IndexDate)
        self.assertEqual(ih.index_types.values.tolist(), [IndexDate, IndexDate])
        self.assertEqual(ih.values.tolist(),
                [[datetime.date(2021, 1, 1), datetime.date(2018, 1, 1)], [datetime.date(2021, 1, 1), datetime.date(2018, 1, 4)], [datetime.date(2021, 1, 2), datetime.date(2018, 1, 1)], [datetime.date(2021, 1, 2), datetime.date(2018, 1, 4)]])

    def test_hierarchy_from_product_e(self) -> None:
        with self.assertRaises(ErrorInitIndex):
            IndexHierarchy.from_product(range(2), range(2), range(2), index_constructors=[Index for _ in range(2)])

        with self.assertRaises(ErrorInitIndex):
            IndexHierarchy.from_product(range(2), range(2), range(2), index_constructors=[Index for _ in range(8)])

    def test_hierarchy_from_product_f(self) -> None:

        groups = np.array(('1954', '2020'), np.datetime64)
        dates = np.array(('2018-01-01', '2018-01-04'), np.datetime64)
        ih = IndexHierarchy.from_product(groups, dates,
                index_constructors=IndexAutoConstructorFactory)
        self.assertEqual([str(dt) for dt in ih.dtypes.values],
                ['datetime64[Y]', 'datetime64[D]'],
                )

    #--------------------------------------------------------------------------

    def test_hierarchy_from_empty(self) -> None:
        ih1 = IndexHierarchy._from_empty(
            (),
            depth_reference=2,
        )
        self.assertEqual(ih1.shape, (0, 2))

        ih2 = IndexHierarchy._from_empty(
            (),
            name=tuple('ABC'),
            depth_reference=3,
        )
        self.assertEqual(ih2.shape, (0, 3))

        ih3 = IndexHierarchy._from_empty(
            np.array(()),
            name=tuple('ABC'),
            depth_reference=3,
        )
        self.assertEqual(ih3.shape, (0, 3))

        ih4 = IndexHierarchy._from_empty(
            IndexHierarchy._from_empty((), depth_reference=2).values
        )
        self.assertEqual(ih4.shape, (0, 2))

        with self.assertRaises(ErrorInitIndex):
            IndexHierarchy._from_empty(())

        with self.assertRaises(ErrorInitIndex):
            IndexHierarchy._from_empty((), depth_reference=1)

        with self.assertRaises(ErrorInitIndex):
            IndexHierarchy._from_empty(np.array(()))

        with self.assertRaises(ErrorInitIndex):
            IndexHierarchy._from_empty(np.array((), ndmin=2))

    #--------------------------------------------------------------------------

    def test_hierarchy_from_tree_a(self) -> None:
        OD = OrderedDict
        tree = OD([('A', (1, 2, 3, 4)), ('B', (1, 2))])

        ih = IndexHierarchy.from_tree(tree)
        self._assert_to_tree_consistency(ih)

        self.assertEqual(ih.to_frame().to_pairs(0),
                ((0, ((0, 'A'), (1, 'A'), (2, 'A'), (3, 'A'), (4, 'B'), (5, 'B'))), (1, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 1), (5, 2))))
                )

    def test_hierarchy_from_tree_b(self) -> None:
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
        self._assert_to_tree_consistency(ih)
        self.assertEqual(ih.to_frame().to_pairs(0),
                ((0, ((0, 'I'), (1, 'I'), (2, 'I'), (3, 'I'), (4, 'I'), (5, 'I'), (6, 'I'), (7, 'II'), (8, 'II'), (9, 'II'), (10, 'II'))), (1, ((0, 'A'), (1, 'A'), (2, 'B'), (3, 'B'), (4, 'B'), (5, 'C'), (6, 'C'), (7, 'A'), (8, 'A'), (9, 'A'), (10, 'B'))), (2, ((0, 1), (1, 2), (2, 1), (3, 2), (4, 3), (5, 2), (6, 3), (7, 1), (8, 2), (9, 3), (10, 1))))
                )

    #---------------------------------------------------------------------------

    def test_hierarchy_from_labels_a(self) -> None:

        labels1 = (('I', 'A', 1),
                ('I', 'A', 2),
                ('I', 'B', 1),
                ('I', 'B', 2),
                ('II', 'A', 1),
                ('II', 'A', 2),
                ('II', 'B', 1),
                ('II', 'B', 2),
                )

        ih = IndexHierarchy.from_labels(labels1)
        self.assertEqual(len(ih), 8)
        self.assertEqual(ih.depth, 3)

        self.assertEqual([ih._loc_to_iloc(x) for x in labels1],
                [0, 1, 2, 3, 4, 5, 6, 7])


        labels2 = (('I', 'A', 1),
                ('I', 'A', 2),
                ('I', 'B', 1),
                ('II', 'B', 2),
                )

        ih = IndexHierarchy.from_labels(labels2)
        self.assertEqual(len(ih), 4)
        self.assertEqual(ih.depth, 3)

        self.assertEqual([ih._loc_to_iloc(x) for x in labels2], [0, 1, 2, 3])

    def test_hierarchy_from_labels_b(self) -> None:

        labels = (('I', 'A'), ('I', 'B'))

        ih = IndexHierarchy.from_labels(labels)

        self.assertEqual(ih.to_frame().to_pairs(0),
                ((0, ((0, 'I'), (1, 'I'))), (1, ((0, 'A'), (1, 'B')))))

    def test_hierarchy_from_labels_c(self) -> None:

        with self.assertRaises(ErrorInitIndex):
            ih = IndexHierarchy.from_labels(tuple())

        with self.assertRaises(ErrorInitIndex):
            ih = IndexHierarchy.from_labels(tuple(), depth_reference=1)


        ih = IndexHierarchy.from_labels(np.array(()).reshape(0, 3))
        self.assertEqual(ih.shape, (0, 3))

        with self.assertRaises(ErrorInitIndex):
            # if depth_reference provided, must match iterable
            ih = IndexHierarchy.from_labels(np.array(()).reshape(0, 3), depth_reference=2)

    def test_hierarchy_from_labels_d(self) -> None:

        with self.assertRaises(RuntimeError):
            ih = IndexHierarchy.from_labels([(3,), (4,)])

    def test_hierarchy_from_labels_e(self) -> None:

        index_constructors = (Index, IndexDate)

        labels = (
            ('a', '2019-01-01'),
            ('a', '2019-02-01'),
            ('b', '2019-01-01'),
            ('b', '2019-02-01'),
        )

        with self.assertRaises(ErrorInitIndex):
            ih = IndexHierarchy.from_labels(labels, index_constructors=(Index,))


        ih = IndexHierarchy.from_labels(labels, index_constructors=index_constructors)

        self.assertEqual(ih.loc[HLoc[:, '2019-02']].values.tolist(),
                [['a', datetime.date(2019, 2, 1)],
                ['b', datetime.date(2019, 2, 1)]])

        self.assertEqual(ih.loc[HLoc[:, '2019']].values.tolist(),
                [['a', datetime.date(2019, 1, 1)],
                ['a', datetime.date(2019, 2, 1)],
                ['b', datetime.date(2019, 1, 1)],
                ['b', datetime.date(2019, 2, 1)]])

        self.assertEqual(ih.loc[HLoc[:, '2019-02-01']].values.tolist(),
                [['a', datetime.date(2019, 2, 1)],
                ['b', datetime.date(2019, 2, 1)]]
                )

    def test_hierarchy_from_labels_f(self) -> None:

        labels1 = (('I', 'A', 1),
                ('I', 'A', 2),
                (None, 'B', 1),
                ('I', None, 2),
                ('II', 'A', 1),
                (None, 'A', 2),
                (None, 'B', 1),
                (None, 'B', 2),
                )

        ih = IndexHierarchy.from_labels(labels1, continuation_token=None)

        self.assertEqual(ih.values.tolist(),
                [['I', 'A', 1], ['I', 'A', 2], ['I', 'B', 1], ['I', 'B', 2], ['II', 'A', 1], ['II', 'A', 2], ['II', 'B', 1], ['II', 'B', 2]]
                )

    def test_hierarchy_from_labels_g(self) -> None:

        labels = (('II', 'A', 1),
                ('I', 'B', 1),
                ('II', 'B', 2),
                ('I', 'A', 2),
                ('I', 'B', 2),
                ('II', 'A', 2),
                ('II', 'B', 1),
                ('I', 'A', 1),
                )

        ih1 = IndexHierarchy.from_labels(labels)
        self.assertEqual(tuple(ih1.iter_label()), labels)

        ih2 = IndexHierarchy.from_labels(labels, reorder_for_hierarchy=True)
        self.assertEqual(ih1.shape, ih2.shape)
        self.assertEqual(tuple(ih2.iter_label()),
                (('II', 'A', 1),
                 ('II', 'A', 2),
                 ('II', 'B', 2),
                 ('II', 'B', 1),
                 ('I', 'A', 2),
                 ('I', 'A', 1),
                 ('I', 'B', 1),
                 ('I', 'B', 2))
                )

    def test_hierarchy_from_labels_h(self) -> None:

        labels = (
                ('I', 'A', 1),
                ('II', 'B', 1),
                ('', 'A', 3),
                ('', 'A', 2),
                ('', 'A', 1),
                ('I', 'B', 1),
                )

        ih  = IndexHierarchy.from_labels(labels, reorder_for_hierarchy=True, continuation_token='')
        self.assertEqual(tuple(ih.iter_label()),
                (('I', 'A', 1),
                 ('I', 'B', 1),
                 ('II', 'A', 3),
                 ('II', 'A', 2),
                 ('II', 'A', 1),
                 ('II', 'B', 1))
                )

    def test_hierarchy_from_labels_i(self) -> None:
        labels = (('I', 'A', 1),
                ('I', 'A', 2),
                ('I', 'B'),
                ('II', 'B', 2),
                )
        with self.assertRaises(ErrorInitIndex):
            IndexHierarchy.from_labels(labels)

    def test_hierarchy_from_labels_j(self) -> None:

        labels1 = ((None, None, 1),
                ('I', 'A', 2),
                (None, 'B', 1),
                ('I', None, 2),
                ('II', 'A', 1),
                (None, 'A', 2),
                (None, 'B', 1),
                (None, 'B', 2),
                )

        ih = IndexHierarchy.from_labels(labels1, continuation_token=None)

        self.assertEqual(ih.values.tolist(),
                [[None, None, 1], ['I', 'A', 2], ['I', 'B', 1], ['I', 'B', 2], ['II', 'A', 1], ['II', 'A', 2], ['II', 'B', 1], ['II', 'B', 2]]
                )

    def test_hierarchy_from_labels_k(self) -> None:

        ih = IndexHierarchy.from_labels((),
                depth_reference=3,
                reorder_for_hierarchy=True,
                )
        self.assertEqual(ih.shape, (0, 3))
        self.assertEqual(ih.depth, 3)


    #---------------------------------------------------------------------------

    def test_hierarchy_from_index_items_a(self) -> None:

        idx1 = Index(('A', 'B', 'C'))
        idx2 = Index(('x', 'y'))
        idx3 = Index((4, 5, 6))

        ih = IndexHierarchy.from_index_items(dict(a=idx1, b=idx2, c=idx3).items())

        self.assertEqual(
                ih.values.tolist(),
                [['a', 'A'], ['a', 'B'], ['a', 'C'], ['b', 'x'], ['b', 'y'], ['c', 4], ['c', 5], ['c', 6]]
                )

    def test_hierarchy_from_index_items_b(self) -> None:

        idx1 = Index(('A', 'B', 'C'))
        idx2 = Index(('x', 'y'))
        idx3 = Index((4, 5, 6))

        ih = IndexHierarchyGO.from_index_items(dict(a=idx1, b=idx2, c=idx3).items())
        ih.append(('c', 7))

        self.assertEqual(ih.values.tolist(),
                [['a', 'A'], ['a', 'B'], ['a', 'C'], ['b', 'x'], ['b', 'y'], ['c', 4], ['c', 5], ['c', 6], ['c', 7]])

    def test_hierarchy_from_index_items_c(self) -> None:
        ih = IndexHierarchy.from_index_items(())
        self.assertEqual((0, 2), ih.shape)

    def test_hierarchy_from_index_items_d(self) -> None:
        ih1 = IndexHierarchy.from_labels((('x', 0), ('y', 1)))
        ih2 = IndexHierarchy.from_labels((('p', 0), ('q', 1)))
        ih3 = IndexHierarchy.from_index_items((('a', ih1), ('b', ih2)))

        assert ih3.shape == (len(ih1) + len(ih2), 3)

        assert set(ih3._indices[0]) == {"a", "b"}
        assert set(ih3._indices[1]) == {"x", "y", "p", "q"}
        assert set(ih3._indices[2]) == {0, 1}

        assert ih3._indices[1].equals(ih1._indices[0].union(ih2._indices[0]))
        assert ih3._indices[2].equals(ih1._indices[1].union(ih2._indices[1]))

        assert tuple(ih3) == (
                ('a', 'x', 0),
                ('a', 'y', 1),
                ('b', 'p', 0),
                ('b', 'q', 1),
                )

    def test_hierarchy_from_index_items_e(self) -> None:
        ih1 = IndexHierarchy.from_labels((('x', 0), ('y', 1)))
        ih2 = IndexHierarchy.from_labels((('p', "2022-12-30"), ('q', "2022-12-31")), index_constructors=[Index, IndexDate])
        ih3 = IndexHierarchy.from_index_items((('a', ih1), ('b', ih2)))

        assert ih3.shape == (len(ih1) + len(ih2), 3)

        assert set(ih3._indices[0]) == {"a", "b"}
        assert set(ih3._indices[1]) == {"x", "y", "p", "q"}
        # Note the downcast to Python date objects
        assert set(ih3._indices[2]) == {0, 1, datetime.date(2022, 12, 30), datetime.date(2022, 12, 31)}

        assert ih3._indices[1].equals(ih1._indices[0].union(ih2._indices[0]))

        # Note the downcast effect here as well
        assert set(ih3._indices[2]) == set(ih1._indices[1].union(ih2._indices[1]))

        assert tuple(ih3) == (
                ('a', 'x', 0),
                ('a', 'y', 1),
                ('b', 'p', datetime.date(2022, 12, 30)),
                ('b', 'q', datetime.date(2022, 12, 31)),
                )

    def test_hierarchy_from_index_items_f(self) -> None:
        ih = IndexHierarchy.from_labels((('x', 0), ('y', 1)))
        idx = Index(('A', 'B', 'C'))

        with self.assertRaises(ErrorInitIndex):
            IndexHierarchy.from_index_items((('a', ih), ('b', idx)))

        with self.assertRaises(ErrorInitIndex):
            IndexHierarchy.from_index_items((('a', idx), ('b', ih)))

    def test_hierarchy_from_index_items_g(self) -> None:
        idx1 = Index(('A', 'B', 'C'))
        idx2 = Index(('x', 'y'))

        ih = IndexHierarchy.from_index_items({"2020-12-30": idx1, "2020-12-31": idx2}.items(), index_constructor=IndexDate)

        assert ih.shape == (len(idx1) + len(idx2), 2)
        assert tuple(ih.astype(str)) == (
                ('2020-12-30', 'A'),
                ('2020-12-30', 'B'),
                ('2020-12-30', 'C'),
                ('2020-12-31', 'x'),
                ('2020-12-31', 'y'),
        )

    #---------------------------------------------------------------------------

    def test_hierarchy_from_labels_delimited_a(self) -> None:

        labels = ("'I' 'A'", "'I' 'B'")

        ih = IndexHierarchy.from_labels_delimited(labels)

        self.assertEqual(ih.values.tolist(),
                [['I', 'A'], ['I', 'B']])

    def test_hierarchy_from_labels_delimited_b(self) -> None:

        labels = (
                "'I' 'A' 0",
                "'I' 'A' 1",
                "'I' 'B' 0",
                "'I' 'B' 1",
                "'II' 'A' 0",
                )

        ih = IndexHierarchy.from_labels_delimited(labels)

        self.assertEqual(ih.values.tolist(),
                [['I', 'A', 0], ['I', 'A', 1], ['I', 'B', 0], ['I', 'B', 1], ['II', 'A', 0]]
                )

    def test_hierarchy_from_labels_delimited_c(self) -> None:

        labels = (
                "['I' 'A' 0]",
                "['I' 'A' 1]",
                "['I' 'B' 0]",
                "['I' 'B' 1]",
                "['II' 'A' 0]",
                )

        ih = IndexHierarchy.from_labels_delimited(labels)

        self.assertEqual(ih.values.tolist(),
                [['I', 'A', 0], ['I', 'A', 1], ['I', 'B', 0], ['I', 'B', 1], ['II', 'A', 0]]
                )

    def test_hierarchy_from_labels_delimited_d(self) -> None:

        labels = (
                "'I' 'A' 0",
                "'I' 'A' 1",
                "'I' 'B' 0",
                "'I' B 1",
                "'II' 'A' 0",
                )

        with self.assertRaises(ValueError):
            IndexHierarchy.from_labels_delimited(labels)

    def test_hierarchy_from_labels_delimited_e(self) -> None:

        labels = (
                "'I' 'A' 0",
                "'I' 'A' 1",
                "'I' 'B' 0",
                "'I'",
                "'II' 'A' 0",
                )

        with self.assertRaises(RuntimeError):
            IndexHierarchy.from_labels_delimited(labels)
    #---------------------------------------------------------------------------

    def test_hierarchy_from_type_blocks_a(self) -> None:
        f1 = Frame.from_element('a', index=range(3), columns=('a',))
        f2 = Frame.from_items((('a', tuple('AABB')), ('b', (1, 2, 1, 2))))
        f3 = Frame.from_items((('a', tuple('AABA')), ('b', (1, 2, 1, 2))))

        with self.assertRaises(ErrorInitIndex):
            IndexHierarchy._from_type_blocks(f1._blocks)

        with self.assertRaises(ErrorInitIndex):
            IndexHierarchy._from_type_blocks(f2._blocks, index_constructors=(IndexDate,))

        with self.assertRaises(ErrorInitIndex):
            IndexHierarchy._from_type_blocks(f3._blocks)

    def test_hierarchy_from_type_blocks_b(self) -> None:
        f1 = Frame.from_items((
                ('a', tuple('ABAB')),
                ('b', (1, 2, 1, 2)),
                ('c', (1, 2, 1, 2)))
                )
        with self.assertRaises(ErrorInitIndex):
            ih = IndexHierarchy._from_type_blocks(f1._blocks)

    def test_hierarchy_from_type_blocks_c(self) -> None:
        f1 = Frame.from_items((
                ('a', tuple('ABAB')),
                ('b', (1, 2, 1, 2)),
                ('c', (1, 2, 1, 2)))
                )

        with self.assertRaises(ErrorInitIndex):
            ih = IndexHierarchy._from_type_blocks(f1._blocks, index_constructors=Index)

    def test_hierarchy_from_type_blocks_d(self) -> None:
        f1 = Frame.from_items((
                ('a', tuple('ABAB')),
                ('b', (1, 2, 1, 2)),
                ('c', (1, 2, 1, 2)))
                )

        with self.assertRaises(ErrorInitIndex):
            ih = IndexHierarchy._from_type_blocks(f1._blocks, index_constructors=[Index for _ in range(2)])

        with self.assertRaises(ErrorInitIndex):
            ih = IndexHierarchy._from_type_blocks(f1._blocks, index_constructors=[Index for _ in range(8)])

    def test_hierarchy_from_type_blocks_e(self) -> None:

        str_dates = np.arange(4).reshape(2,2).astype("datetime64[D]").astype(str)

        ih1 = IndexHierarchy._from_type_blocks(TypeBlocks.from_blocks(str_dates))
        self.assertSetEqual(set(ih1.dtypes.values), {np.dtype("<U28")})

        ih2 = IndexHierarchy._from_type_blocks(TypeBlocks.from_blocks(str_dates), index_constructors=[Index, IndexDate])
        self.assertSetEqual(set(ih2.dtypes.values), {np.dtype("<U28"), np.dtype("<M8[D]")})

        ih3 = IndexHierarchy._from_type_blocks(TypeBlocks.from_blocks(str_dates), index_constructors=IndexDate)
        self.assertSetEqual(set(ih3.dtypes.values), {np.dtype("<M8[D]")})

    #---------------------------------------------------------------------------

    def test_hierarchy_from_values_per_depth_a(self) -> None:

        # NOTE: This will consolidate dtypes
        arrays1 = np.array((
                ('II', 'A', '1'),
                ('I', 'B', '1'),
                ('II', 'B', '2'),
                ('I', 'A', '2'),
                ('I', 'B', '2'),
                ('II', 'A', '2'),
                ('II', 'B', '1'),
                ('I', 'A', '1'),
                ))

        ih1 = IndexHierarchy.from_values_per_depth(arrays1)
        self.assertTrue((np.array(tuple(ih1.iter_label())) == arrays1).all())

        arrays2 = [
            np.array(['II', 'I', 'II', 'I', 'I', 'II', 'II', 'I']),
            np.array(['A', 'B', 'B', 'A', 'B', 'A', 'B', 'A']),
            np.array(['1', '1', '2', '2', '2', '2', '1', '1']),
        ]
        ih2 = IndexHierarchy.from_values_per_depth(arrays2)
        self.assertTrue(ih1.equals(ih2))

    def test_hierarchy_from_values_per_depth_b(self) -> None:

        arrays = [(1, 2), (1,)]

        with self.assertRaises(ErrorInitIndex):
            _ = IndexHierarchy.from_values_per_depth(arrays)


    def test_hierarchy_from_values_per_depth_c(self) -> None:
        a1 = np.array(('1954', '1954', '2020'), np.datetime64)
        a2 = np.array(('2018-01-01', '2018-01-04', '2018-01-04'), np.datetime64)
        ih = IndexHierarchy.from_values_per_depth((a1, a2),
                index_constructors=IndexAutoConstructorFactory)
        self.assertEqual([str(dt) for dt in ih.dtypes.values],
                ['datetime64[Y]', 'datetime64[D]'],
                )
        self.assertEqual(ih.name, None)


    def test_hierarchy_from_values_per_depth_d(self) -> None:
        IACF = IndexAutoConstructorFactory
        a1 = np.array(('1954', '1954', '2020'), np.datetime64)
        a2 = np.array(('2018-01-01', '2018-01-04', '2018-01-04'), np.datetime64)
        ih = IndexHierarchy.from_values_per_depth((a1, a2),
                index_constructors=(IACF('a'), IACF('b')),
                )
        self.assertEqual([str(dt) for dt in ih.dtypes.values],
                ['datetime64[Y]', 'datetime64[D]'],
                )
        self.assertEqual(ih.name, ('a', 'b'))

    def test_hierarchy_from_values_per_depth_e(self) -> None:
        ih = IndexHierarchy.from_values_per_depth((('a', 'a', None, None), ('2012', '2008', '1933', '2008')), index_constructors=(Index, IndexDate))
        self.assertEqual(ih.dtypes.values.tolist(), [np.dtype('O'), np.dtype('<M8[D]')])
        self.assertEqual(ih.shape, (4, 2))

    def test_hierarchy_from_values_per_depth_f(self) -> None:
        ih1 = IndexHierarchy.from_values_per_depth(((), ()))
        self.assertEqual(ih1.shape, (0, 2))

        ih2 = IndexHierarchy.from_values_per_depth(np.array(()).reshape(0, 2))
        self.assertEqual(ih2.shape, (0, 2))

    def test_hierarchy_from_values_per_depth_g(self) -> None:
        with self.assertRaises(RuntimeError):
            ih1 = IndexHierarchy.from_values_per_depth(())
        ih2 = IndexHierarchy.from_values_per_depth((), depth_reference=4)
        self.assertEqual(ih2.shape, (0, 4))

    #---------------------------------------------------------------------------

    def test_hierarchy_contains_a(self) -> None:
        labels = (('I', 'A'), ('I', 'B'))
        ih = IndexHierarchy.from_labels(labels)

        self.assertIn(('I', 'A'), ih)

    def test_hierarchy_contains_b(self) -> None:
        labels = (('I', 'A'), ('I', 'B'))
        ih = IndexHierarchy.from_labels(labels)

        self.assertIn(Index(labels), ih)

    def test_hierarchy_contains_c(self) -> None:
        ih = IndexHierarchy.from_product((True, False), (True, False))

        self.assertIn((True, False), ih)
        with self.assertRaises(IndexError):
            np.array((True, False)) in ih # type: ignore

        with self.assertRaises(RuntimeError):
            (True, False, True, False) in ih #pylint: disable=W0104

    def test_hierarchy_contains_d(self) -> None:
        labels = ((True, 'A'), ('I', 'B'))
        ih = IndexHierarchy.from_labels(labels)

        key = HLoc[:, 'A'] # type: ignore

        ih2 = ih.loc[key]
        self.assertEqual(tuple(ih2), ((True, 'A'),))

        self.assertIn(key, ih)

    def test_hierarchy_extract_a(self) -> None:
        idx = IndexHierarchy.from_product(['A', 'B'], [1, 2])

        self.assertEqual(idx.iloc[1], ('A', 2))
        self.assertEqual(idx.loc[('B', 1)], ('B', 1))
        self.assertEqual(idx[2], ('B', 1)) #pylint: disable=E1136
        self.assertEqual(idx.loc[HLoc['B', 1]], ('B', 1))

    def test_hierarchy_iter_a(self) -> None:
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
        self._assert_to_tree_consistency(ih)

        # this iterates over numpy arrays, which can be used with contains
        self.assertEqual([k in ih for k in ih], #pylint: disable=E1133
                [True, True, True, True, True, True, True, True]
                )

    def test_hierarchy_rename_a(self) -> None:
        labels = (('a', 1), ('a', 2), ('b', 1), ('b', 2))
        ih1 = IndexHierarchy.from_labels(labels, name='foo')
        self.assertEqual(ih1.name, 'foo')
        ih2 = ih1.rename(None)
        self.assertEqual(ih2.name, None)

    def test_hierarchy_reversed(self) -> None:
        labels = (('a', 1), ('a', 2), ('b', 1), ('b', 2))
        hier_idx = IndexHierarchy.from_labels(labels)
        self.assertTrue(
            all(hidx_1 == hidx_2
                for hidx_1, hidx_2 in zip(reversed(hier_idx), reversed(labels)))
        )

    def test_hierarchy_keys_a(self) -> None:
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

        ih = IndexHierarchyGO.from_tree(tree)
        self._assert_to_tree_consistency(ih)

        self.assertEqual([k in ih for k in ih], #pylint: disable=E1133
                [True, True, True, True, True, True, True, True]
                )

        ih.append(('III', 'A', 1))

        self.assertEqual(set(ih),
                {('I', 'B', 1), ('I', 'A', 2), ('II', 'B', 2), ('II', 'A', 2), ('I', 'A', 1), ('III', 'A', 1), ('II', 'B', 1), ('II', 'A', 1), ('I', 'B', 2)}
                )

    def test_hierarchy_display_a(self) -> None:
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
        self._assert_to_tree_consistency(ih)

        post = ih.display()
        self.assertEqual(len(post), 10)

        s = Series(range(8), index=ih)
        post = s.display()
        self.assertEqual(len(post), 11)

    def test_hierarchy_loc_a(self) -> None:
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
        self._assert_to_tree_consistency(ih)

        s = Series(range(8), index=ih)

        self.assertEqual(
                s.loc[HLoc['I']].values.tolist(),
                [0, 1, 2, 3])

        self.assertEqual(
                s.loc[HLoc[:, 'A']].values.tolist(),
                [0, 1, 4, 5])

    def test_hierarchy_loc_b(self) -> None:
        ih1 = IndexHierarchy.from_labels([(1,'dd',0),(1,'b',0),(2,'cc',0),(2,'ee',0)])

        ih2 = ih1.loc[HLoc[(1,['dd'])]]

        [[a],[b],[c]] = ih2._indexers
        self.assertEqual((a, b, c), (0, 0, 0))

        [[a],[b],[c]] = ih2._indices
        self.assertEqual((a, b, c), (1, 'dd', 0))

    def test_hierarchy_loc_c(self) -> None:
        ih1 = IndexHierarchy.from_labels([(1,'dd',0),(1,'b',0),(2,'cc',0),(2,'ee',0)])

        with self.assertRaises(RuntimeError):
            ih1.loc[1, 'dd'] # pylint: disable=pointless-statement

        with self.assertRaises(RuntimeError):
            ih1.loc[1, :] # pylint: disable=pointless-statement

        with self.assertRaises(RuntimeError):
            ih1.loc[:, 'dd'] # pylint: disable=pointless-statement

        with self.assertRaises(RuntimeError):
            ih1.loc[:, :, 0] # pylint: disable=pointless-statement

        with self.assertRaises(RuntimeError):
            ih1.loc[(1, 'dd')] # pylint: disable=pointless-statement

        with self.assertRaises(RuntimeError):
            ih1.loc[(1, 'dd'):] # pylint: disable=pointless-statement

        with self.assertRaises(RuntimeError):
            ih1.loc[Index([(1, 'dd')])]

        with self.assertRaises(RuntimeError):
            ih1.loc[Series([(1, 'dd')])]

    def test_hierarchy_loc_d(self) -> None:
        # https://github.com/static-frame/static-frame/issues/455
        ih1 = IndexHierarchy.from_labels([(0, 1), (0, 2), (1, 0), (1, 1)])
        ih2 = IndexHierarchy.from_labels([(1, 1), (0, 2), (0, 1)])

        self.assertEqual(tuple(ih2), tuple(ih1.loc[ih2]))

    def test_hierarchy_series_a(self) -> None:
        f1 = IndexHierarchy.from_tree
        tree = dict(a=(1,2,3))
        s1 = Series.from_element(23, index=f1(tree))
        self._assert_to_tree_consistency(f1(tree))
        self.assertEqual(s1.values.tolist(), [23, 23, 23])

        f2 = IndexHierarchy.from_product
        s2 = Series.from_element(3, index=f2(Index(('a', 'b')), Index((1,2))))
        self.assertEqual(s2.to_pairs(),
                ((('a', 1), 3), (('a', 2), 3), (('b', 1), 3), (('b', 2), 3)))

    def test_hierarchy_frame_a(self) -> None:
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
        self._assert_to_tree_consistency(ih)

        data = np.arange(6*6).reshape(6, 6)
        f1 = Frame(data, index=ih, columns=ih)
        # self.assertEqual(len(f.to_pairs(0)), 8)


        f2 = f1.assign.loc[('I', 'B', 2), ('II', 'A', 1)](200)

        post = f2.to_pairs(0)
        self.assertEqual(post,
                ((('I', 'A', 1), ((('I', 'A', 1), 0), (('I', 'B', 1), 6), (('I', 'B', 2), 12), (('II', 'A', 1), 18), (('II', 'B', 1), 24), (('II', 'B', 2), 30))), (('I', 'B', 1), ((('I', 'A', 1), 1), (('I', 'B', 1), 7), (('I', 'B', 2), 13), (('II', 'A', 1), 19), (('II', 'B', 1), 25), (('II', 'B', 2), 31))), (('I', 'B', 2), ((('I', 'A', 1), 2), (('I', 'B', 1), 8), (('I', 'B', 2), 14), (('II', 'A', 1), 20), (('II', 'B', 1), 26), (('II', 'B', 2), 32))), (('II', 'A', 1), ((('I', 'A', 1), 3), (('I', 'B', 1), 9), (('I', 'B', 2), 200), (('II', 'A', 1), 21), (('II', 'B', 1), 27), (('II', 'B', 2), 33))), (('II', 'B', 1), ((('I', 'A', 1), 4), (('I', 'B', 1), 10), (('I', 'B', 2), 16), (('II', 'A', 1), 22), (('II', 'B', 1), 28), (('II', 'B', 2), 34))), (('II', 'B', 2), ((('I', 'A', 1), 5), (('I', 'B', 1), 11), (('I', 'B', 2), 17), (('II', 'A', 1), 23), (('II', 'B', 1), 29), (('II', 'B', 2), 35))))
        )


        f3 = f1.assign.loc[('I', 'B', 2):, HLoc[:, :, 2]](200)  # type: ignore  # https://github.com/python/typeshed/pull/3024

        self.assertEqual(f3.to_pairs(0),
                ((('I', 'A', 1), ((('I', 'A', 1), 0), (('I', 'B', 1), 6), (('I', 'B', 2), 12), (('II', 'A', 1), 18), (('II', 'B', 1), 24), (('II', 'B', 2), 30))), (('I', 'B', 1), ((('I', 'A', 1), 1), (('I', 'B', 1), 7), (('I', 'B', 2), 13), (('II', 'A', 1), 19), (('II', 'B', 1), 25), (('II', 'B', 2), 31))), (('I', 'B', 2), ((('I', 'A', 1), 2), (('I', 'B', 1), 8), (('I', 'B', 2), 200), (('II', 'A', 1), 200), (('II', 'B', 1), 200), (('II', 'B', 2), 200))), (('II', 'A', 1), ((('I', 'A', 1), 3), (('I', 'B', 1), 9), (('I', 'B', 2), 15), (('II', 'A', 1), 21), (('II', 'B', 1), 27), (('II', 'B', 2), 33))), (('II', 'B', 1), ((('I', 'A', 1), 4), (('I', 'B', 1), 10), (('I', 'B', 2), 16), (('II', 'A', 1), 22), (('II', 'B', 1), 28), (('II', 'B', 2), 34))), (('II', 'B', 2), ((('I', 'A', 1), 5), (('I', 'B', 1), 11), (('I', 'B', 2), 200), (('II', 'A', 1), 200), (('II', 'B', 1), 200), (('II', 'B', 2), 200))))
        )

    def test_hierarchy_frame_b(self) -> None:
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
        self._assert_to_tree_consistency(ih)
        data = np.arange(6*6).reshape(6, 6)
        # TODO: this only works if own_columns is True for now
        f1 = FrameGO(data, index=range(6), columns=ih, own_columns=True)
        f1[('II', 'B', 3)] = 0

        f2 = f1[HLoc[:, 'B']]
        self.assertEqual(f2.shape, (6, 5))

        self.assertEqual(f2.to_pairs(),
                ((('I', 'B', 1), ((0, 1), (1, 7), (2, 13), (3, 19), (4, 25), (5, 31))), (('I', 'B', 2), ((0, 2), (1, 8), (2, 14), (3, 20), (4, 26), (5, 32))), (('II', 'B', 1), ((0, 4), (1, 10), (2, 16), (3, 22), (4, 28), (5, 34))), (('II', 'B', 2), ((0, 5), (1, 11), (2, 17), (3, 23), (4, 29), (5, 35))), (('II', 'B', 3), ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0))))
                )

        f3 = f1[HLoc[:, :, 1]]
        self.assertEqual(f3.to_pairs(), ((('I', 'A', 1), ((0, 0), (1, 6), (2, 12), (3, 18), (4, 24), (5, 30))), (('I', 'B', 1), ((0, 1), (1, 7), (2, 13), (3, 19), (4, 25), (5, 31))), (('II', 'A', 1), ((0, 3), (1, 9), (2, 15), (3, 21), (4, 27), (5, 33))), (('II', 'B', 1), ((0, 4), (1, 10), (2, 16), (3, 22), (4, 28), (5, 34)))))


        f4 = f1.loc[[2, 5], HLoc[:, 'A']]
        self.assertEqual(f4.to_pairs(0),
                ((('I', 'A', 1), ((2, 12), (5, 30))), (('II', 'A', 1), ((2, 15), (5, 33)))))

    def test_hierarchy_index_go_a(self) -> None:

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
        self._assert_to_tree_consistency(ih1)

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
        self._assert_to_tree_consistency(ih2)

        ih1.extend(ih2)

        self.assertEqual(ih1._loc_to_iloc(('IV', 'B', 2)), 11)
        self.assertEqual(len(ih2), 6)

        # need tuple here to distinguish from iterable type selection
        self.assertEqual([ih1._loc_to_iloc(tuple(v)) for v in ih1.values],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                )

    def test_hierarchy_index_go_b(self) -> None:
        labelsA = [
            ('A', 'D'),
            ('A', 'E'),
            ('B', 'D'),
            ('A', 'G'),
            ('B', 'E'),
        ]
        labelsB = [
            ('A', 'H'),
            ('A', 'F'),
            ('B', 'F'),
            ('C', 'D'),
            ('B', 'G'),
        ]

        ihgo = IndexHierarchyGO.from_labels(labelsA)
        ih2 = IndexHierarchy.from_labels(labelsB)

        ihgo.extend(ih2)

        expected = IndexHierarchy.from_labels(labelsA + labelsB)
        self.assertTrue(ihgo.equals(expected))

    def test_hierarchy_index_go_c(self) -> None:
        labelsA = [
            ('A', 'D'),
            ('A', 'E'),
            ('A', 'F'),
            ('B', 'D'),
            ('B', 'E'),
        ]
        labelsB = [
            ('A', 'G'),
            ('A', 'H'),
            ('A', 'I'),
            ('B', 'F'),
            ('B', 'G'),
        ]

        ihgo = IndexHierarchyGO.from_labels(labelsA)
        ih2 = IndexHierarchy.from_labels(labelsB)

        ihgo.extend(ih2)

        expected = IndexHierarchy.from_labels(labelsA + labelsB)
        self.assertTrue(ihgo.equals(expected))

    def test_hierarchy_index_go_d(self) -> None:
        labelsA = [
            ('A', 'D'),
            ('A', 'E'),
            ('A', 'F'),
            ('B', 'E'),
            ('B', 'D'),
        ]
        labelsB = [
            ('B', 'G'),
            ('A', 'G'),
            ('A', 'I'),
            ('A', 'H'),
            ('B', 'F'),
        ]
        labelsC = [
            ('C', 'G'),
            ('D', 'G'),
            ('C', 'I'),
            ('C', 'H'),
            ('F', 'F'),
        ]

        ihgo = IndexHierarchyGO.from_labels(labelsA)
        ih2 = IndexHierarchy.from_labels(labelsB)
        ih3 = IndexHierarchy.from_labels(labelsC)

        ihgo.extend(ih2)
        ihgo.extend(ih3)

        expected = IndexHierarchy.from_labels(labelsA + labelsB + labelsC)
        self.assertTrue(ihgo.equals(expected))

    def test_hierarchy_index_go_e(self) -> None:
        labelsA = [
            ('A', 'D'),
            ('A', 'E'),
            ('A', 'F'),
            ('B', 'E'),
            ('B', 'D'),
        ]

        labelB = ('C', 'C')
        labelsC = [
            ('B', 'G'),
            ('A', 'G'),
            ('A', 'I'),
            ('A', 'H'),
            ('B', 'F'),
        ]
        labelD = ('C', 'B')
        labelsE = [
            ('C', 'G'),
            ('D', 'G'),
            ('C', 'I'),
            ('C', 'H'),
            ('F', 'F'),
        ]
        labelF = ('C', 'A')

        ihgo = IndexHierarchyGO.from_labels(labelsA)
        ih2 = IndexHierarchy.from_labels(labelsC)
        ih3 = IndexHierarchy.from_labels(labelsE)

        ihgo.append(labelB)
        ihgo.extend(ih2)
        ihgo.append(labelD)
        ihgo.extend(ih3)
        ihgo.append(labelF)

        expected = IndexHierarchy.from_labels(labelsA + [labelB] + labelsC + [labelD] + labelsE + [labelF])

        self.assertTrue(ihgo.equals(expected))

    def test_hierarchy_index_go_f(self) -> None:
        labels = [
            ('A', 'D'),
            ('A', 'E'),
            ('A', 'F'),
            ('B', 'E'),
            ('B', 'D'),
        ]

        ihgo = IndexHierarchyGO.from_labels(labels)

        with self.assertRaises(RuntimeError):
            ihgo.append(('A', ))

        with self.assertRaises(RuntimeError):
            ihgo.append(('A', 'B', 'C'))

    #---------------------------------------------------------------------------

    @run_with_static_and_grow_only
    def test_hierarchy_relabel_a(self,
            index_class: tp.Type[IndexHierarchy]
            ) -> None:

        labels = (('I', 'A'), ('I', 'B'), ('II', 'A'), ('II', 'B'))

        ih = index_class.from_labels(labels)

        ih2 = ih.relabel({('I', 'B'): ('I', 'C')})

        self.assertEqual(ih2.values.tolist(),
                [['I', 'A'], ['I', 'C'], ['II', 'A'], ['II', 'B']])

        with self.assertRaises(Exception):
            ih3 = ih.relabel({('I', 'B'): ('I', 'C', 1)})

        ih3 = ih.relabel(lambda x: tuple(e.lower() for e in x))

        self.assertEqual(
                ih3.values.tolist(),
                [['i', 'a'], ['i', 'b'], ['ii', 'a'], ['ii', 'b']])

    def test_hierarchy_relabel_b(self) -> None:

        labels = (('I', 'A'), ('I', 'B'), ('II', 'A'), ('II', 'B'))

        ih = IndexHierarchyGO.from_labels(labels)
        ih.append((('I', 'D')))

        ih2 = ih.relabel({('I', 'B'): ('I', 'C')})

        self.assertEqual(ih2.values.tolist(),
                [['I', 'A'], ['I', 'C'], ['II', 'A'], ['II', 'B'], ['I', 'D']])

        with self.assertRaises(Exception):
            ih3 = ih.relabel({('I', 'B'): ('I', 'C', 1)})

        ih3 = ih.relabel(lambda x: tuple(e.lower() for e in x))

        self.assertEqual(
                ih3.values.tolist(),
                [['i', 'a'], ['i', 'b'], ['ii', 'a'], ['ii', 'b'], ['i', 'd']])

    @run_with_static_and_grow_only
    def test_hierarchy_relabel_at_depth_a(self,
            index_class: tp.Type[IndexHierarchy]
            ) -> None:

        idx1 = Index((True, False))
        idx2 = Index(tuple('abcde'))
        idx3 = Index(range(10))

        ih = index_class.from_product(idx1, idx2, idx3)

        actual = ih.relabel_at_depth(lambda x: x*2, [1, 2])
        expected = index_class.from_product(idx1, idx2 * 2, idx3 * 2)

        self.assertTrue(actual.equals(expected))

    @run_with_static_and_grow_only
    def test_hierarchy_relabel_at_depth_b(self,
            index_class: tp.Type[IndexHierarchy]
            ) -> None:

        labels = [
            (0, 0), # -> (0, 0)
            (0, 1), # -> (0, 1)
            (0, 2), # -> (0, 2)
            (1, 0), # -> (1, 0)
            (1, 1), # -> (1, 1)
            (1, 2), # -> (1, 2)
            (2, 3), # -> (2, 3)
        ]

        mapper = {0:1, 1:0, 2:1}

        ih = index_class.from_labels(labels)

        actual = ih.relabel_at_depth(mapper, 0)

        self.assertListEqual(actual.values.tolist(),
                [[1, 0], [1, 1], [1, 2], [0, 0], [0, 1], [0, 2], [1, 3]])

    @run_with_static_and_grow_only
    def test_hierarchy_relabel_at_depth_2d_single_depth(self,
            index_class: tp.Type[IndexHierarchy]
            ) -> None:

        ih = index_class.from_product(('I', 'II'), ('A', 'B'))

        # Mapping
        ih1 = ih.relabel_at_depth(dict(I=1), depth_level=0)
        self.assertEqual(ih1.values.tolist(),
                [[1, 'A'], [1, 'B'], ['II', 'A'], ['II', 'B']])

        # Function
        ih2 = ih.relabel_at_depth(lambda x: x*2, depth_level=1)
        self.assertEqual(ih2.values.tolist(),
                [['I', 'AA'], ['I', 'BB'], ['II', 'AA'], ['II', 'BB']])

        # Sequence
        ih3 = ih.relabel_at_depth(range(2**2), depth_level=[1])
        self.assertEqual(ih3.values.tolist(),
                [['I', 0], ['I', 1], ['II', 2], ['II', 3]])

    @run_with_static_and_grow_only
    def test_hierarchy_relabel_at_depth_2d_all_depths(self,
            index_class: tp.Type[IndexHierarchy]
            ) -> None:

        ih = index_class.from_product(('I', 'II'), ('A', 'B'))

        # Mapping
        ih1 = ih.relabel_at_depth(dict(I=1, B=2), depth_level=[0, 1])
        self.assertEqual(ih1.values.tolist(),
                [[1, 'A'], [1, 2], ['II', 'A'], ['II', 2]])

        # Func
        ih2 = ih.relabel_at_depth(lambda x: x*2, depth_level=[1, 0])
        self.assertEqual(ih2.values.tolist(),
                [['II', 'AA'], ['II', 'BB'], ['IIII', 'AA'], ['IIII', 'BB']])

        # Sequence
        ih3 = ih.relabel_at_depth(tuple('abcd'), depth_level=[1, 0])
        self.assertEqual(ih3.values.tolist(),
                [['a', 'a'], ['b', 'b'], ['c', 'c'], ['d', 'd']])

    @run_with_static_and_grow_only
    def test_hierarchy_relabel_at_depth_3d_single_depth(self,
            index_class: tp.Type[IndexHierarchy]
            ) -> None:

        ih = index_class.from_product(('I', 'II'), ('B', 'A'), (2, 1))

        # Mapping
        ih1 = ih.relabel_at_depth(dict(II=99), depth_level=0)
        self.assertTrue(ih1.equals(
            index_class.from_product(('I', 99), ('B', 'A'), (2, 1)))
            )

        # Function
        ih2 = ih.relabel_at_depth(lambda x: x.lower(), depth_level=1)
        self.assertTrue(ih2.equals(
            index_class.from_product(('I', 'II'), ('b', 'a'), (2, 1)))
            )

        # Sequence
        ih3 = ih.relabel_at_depth(np.arange(2**3), depth_level=[2,])
        self.assertEqual(ih3.values.tolist(),
                [['I', 'B', 0],
                 ['I', 'B', 1],
                 ['I', 'A', 2],
                 ['I', 'A', 3],
                 ['II', 'B', 4],
                 ['II', 'B', 5],
                 ['II', 'A', 6],
                 ['II', 'A', 7]])

    @run_with_static_and_grow_only
    def test_hierarchy_relabel_at_depth_3d_multiple_depths(self,
            index_class: tp.Type[IndexHierarchy]
            ) -> None:

        ih = index_class.from_product(('I', 'II'), ('B', 'A'), (2, 1))

        # Mapping
        ih1 = ih.relabel_at_depth({'II': 99, 1: 101}, depth_level=list((0, 2)))
        self.assertTrue(ih1.equals(
            index_class.from_product(('I', 99), ('B', 'A'), (2, 101)))
            )

        # Function
        ih2 = ih.relabel_at_depth(lambda x: x*3, depth_level=[2, 0])
        self.assertTrue(ih2.equals(
            index_class.from_product(('III', 'IIIIII'), ('B', 'A'), (6, 3)))
            )

        # Sequence
        ih3 = ih.relabel_at_depth(iter(range(9, 1, -1)), depth_level=list({0: None, 1: None}))
        self.assertEqual(ih3.values.tolist(),
                [[9, 9, 2],
                 [8, 8, 1],
                 [7, 7, 2],
                 [6, 6, 1],
                 [5, 5, 2],
                 [4, 4, 1],
                 [3, 3, 2],
                 [2, 2, 1]])

    @run_with_static_and_grow_only
    def test_hierarchy_relabel_at_depth_3d_all_depths(self,
            index_class: tp.Type[IndexHierarchy]
            ) -> None:

        ih = index_class.from_product(('I', 'II'), ('B', 'A'), (2, 1))

        # Mapping
        series_map = Series([0, True, None, '13'], index=['I', 'II', 'B', 2])
        ih1 = ih.relabel_at_depth(series_map, depth_level=[2,0,1])
        self.assertEqual(ih1.values.tolist(),
                [[0, None, '13'],
                 [0, None, 1],
                 [0, 'A', '13'],
                 [0, 'A', 1],
                 [1, None, '13'],
                 [1, None, 1],
                 [1, 'A', '13'],
                 [1, 'A', 1]])

        # Function
        numbers = (n for n in range(1, 100000000))
        def func(arg: tp.Any) -> int:
            return next(numbers)

        ih2 = ih.relabel_at_depth(func, depth_level=list(l for l in (1, 0, 2)))
        self.assertEqual(ih2.values.tolist(),
                [[1, 3, 5],
                 [1, 3, 6],
                 [1, 4, 5],
                 [1, 4, 6],
                 [2, 3, 5],
                 [2, 3, 6],
                 [2, 4, 5],
                 [2, 4, 6]])

        # Sequence
        ih3 = ih.relabel_at_depth(range(8), depth_level=list(range(3)))
        self.assertEqual(ih3.values.tolist(),
                [[0, 0, 0],
                 [1, 1, 1],
                 [2, 2, 2],
                 [3, 3, 3],
                 [4, 4, 4],
                 [5, 5, 5],
                 [6, 6, 6],
                 [7, 7, 7]])

    @run_with_static_and_grow_only
    def test_hierarchy_relabel_at_depth_bad_input(self,
            index_class: tp.Type[IndexHierarchy]
            ) -> None:
        ih = index_class.from_product(('I', 'II'), ('A', 'B'))

        # Iterable is not long enough!
        with self.assertRaises(ValueError):
            ih.relabel_at_depth(range(3), depth_level=0)

        # Iterable is too long!
        with self.assertRaises(ValueError):
            ih.relabel_at_depth(range(5), depth_level=0)

        # Depth levels are not unique
        with self.assertRaises(ValueError):
            ih.relabel_at_depth({}, depth_level=[0, 0])

        # Depth level is too shallow
        with self.assertRaises(ValueError):
            ih.relabel_at_depth({}, depth_level=2)

        # Depth level outside range positive
        with self.assertRaises(ValueError):
            ih.relabel_at_depth(range(4), depth_level=3)

        # Depth level outside range negative
        with self.assertRaises(ValueError):
            ih.relabel_at_depth(range(4), depth_level=-1)

        # No depth levels!
        with self.assertRaises(ValueError):
            ih.relabel_at_depth(lambda:None, depth_level=[])

    @run_with_static_and_grow_only
    def test_hierarchy_relabel_at_depth_properties(self,
            index_class: tp.Type[IndexHierarchy]
            ) -> None:
        ih = index_class.from_product(('I', 'II'), ('A', 'B'), (1, 2))

        # Identity function relabel
        ih1 = ih.relabel_at_depth(lambda x: x, depth_level=0)
        self.assertTrue(ih1.equals(ih))

        # Empty mapping relabel
        ih2 = ih.relabel_at_depth({}, depth_level=0)
        self.assertTrue(ih2.equals(ih))

        # Repeat column
        ih3 = ih.relabel_at_depth(ih._blocks._extract_array_column(0), depth_level=0)
        self.assertTrue(ih3.equals(ih))

    #---------------------------------------------------------------------------

    def test_hierarchy_rehierarch_a(self) -> None:
        ih1 = IndexHierarchy.from_product(('I', 'II'), ('B', 'A'), (2, 1))
        ih2 = ih1.rehierarch((1, 0, 2))

        self.assertEqual(ih2.values.tolist(),
                [['B', 'I', 2], ['B', 'I', 1], ['B', 'II', 2], ['B', 'II', 1], ['A', 'I', 2], ['A', 'I', 1], ['A', 'II', 2], ['A', 'II', 1]]
                )

        ih3 = ih1.rehierarch((2, 1, 0))
        self.assertEqual(
                ih3.values.tolist(),
                [[2, 'B', 'I'], [2, 'B', 'II'], [2, 'A', 'I'], [2, 'A', 'II'], [1, 'B', 'I'], [1, 'B', 'II'], [1, 'A', 'I'], [1, 'A', 'II']]
                )

    def test_hierarchy_rehierarch_b(self) -> None:
        labels = (
                ('I', 'A'),
                ('I', 'B'),
                ('II', 'C'),
                ('II', 'B'),
                ('II', 'D'),
                ('III', 'D'),
                ('IV', 'A'),
                )

        ih1 = IndexHierarchyGO.from_labels(labels)
        self.assertEqual(ih1.rehierarch([1, 0]).values.tolist(),
                [['A', 'I'], ['A', 'IV'], ['B', 'I'], ['B', 'II'], ['C', 'II'], ['D', 'II'], ['D', 'III']]
                )

    def test_hierarchy_rehierarch_c(self) -> None:
        labels = (
                ('I', 'A'),
                ('I', 'B'),
                ('II', 'A'),
                ('II', 'B'),
                )
        ih1 = IndexHierarchy.from_labels(labels)

        with self.assertRaises(RuntimeError):
            ih1.rehierarch([0, 0])

        with self.assertRaises(RuntimeError):
            ih1.rehierarch([0,])

    #---------------------------------------------------------------------------

    def test_hierarchy_set_operators_a(self) -> None:

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

        post1 = ih1.intersection(ih2)
        self.assertEqual(post1.values.tolist(),
                [['II', 'A'], ['II', 'B']])

        post2 = ih1.union(ih2)
        self.assertEqual(post2.values.tolist(),
                [['I', 'A'], ['II', 'A'], ['III', 'A'], ['I', 'B'], ['II', 'B'], ['III', 'B']])

        post3 = ih1.difference(ih2)
        self.assertEqual(post3.values.tolist(),
                [['I', 'A'], ['I', 'B']])

        post4 = ih1.difference(tuple(ih2))
        self.assertTrue(post3.equals(post4))
        self.assertTrue(post4.equals(post3))

    def test_hierarchy_set_operators_b(self) -> None:

        labels = (
                ('II', 'B'),
                ('II', 'A'),
                ('I', 'B'),
                ('I', 'A'),
                )

        ih1 = IndexHierarchy.from_labels(labels)
        ih2 = IndexHierarchy.from_labels(labels)

        post1 = ih1.union(ih2)
        self.assertEqual(post1.values.tolist(),
                [['II', 'B'], ['II', 'A'], ['I', 'B'], ['I', 'A']])

        post2 = ih1.intersection(ih2)
        self.assertEqual(post2.values.tolist(),
                [['II', 'B'], ['II', 'A'], ['I', 'B'], ['I', 'A']])

        post3 = ih1.difference(ih2)
        self.assertEqual(post3.values.tolist(), [])
        self.assertEqual(post3.shape, (0, 2))

    def test_hierarchy_set_operators_c(self) -> None:

        labels = (
                ('II', 'B'),
                ('II', 'A'),
                ('I', 'B'),
                ('I', 'A'),
                )

        ih1 = IndexHierarchy.from_labels((), depth_reference=2)
        ih2 = IndexHierarchy.from_labels(labels)

        post1 = ih1.union(ih2)
        self.assertEqual(post1.values.tolist(),
                [['II', 'B'], ['I', 'B'], ['II', 'A'], ['I', 'A']])

        post2 = ih1.intersection(ih2)
        self.assertEqual(post2.values.tolist(), [])

        post3 = ih1.difference(ih2)
        self.assertEqual(post3.values.tolist(), [])

    def test_hierarchy_set_operators_d(self) -> None:

        labels = (
                ('II', 'B'),
                ('II', 'A'),
                ('I', 'B'),
                ('I', 'A'),
                )

        ih1 = IndexHierarchyGO.from_labels(labels)
        ih2 = IndexHierarchy.from_labels((), depth_reference=2)

        post1 = ih1.union(ih2)
        self.assertEqual(post1.values.tolist(),
                [['II', 'B'], ['II', 'A'], ['I', 'B'], ['I', 'A']])

        post2 = ih1.intersection(ih2)
        self.assertEqual(post2.values.tolist(), [])

        ih1.append(('I', 'C'))

        post3 = ih1.difference(ih2)
        self.assertEqual(post3.values.tolist(),
                [['II', 'B'], ['II', 'A'], ['I', 'B'], ['I', 'A'], ['I', 'C']])

    def test_hierarchy_set_operators_e(self) -> None:
        dd = datetime.date
        i1 = IndexHierarchy.from_labels([[1, dd(2019, 1, 1)], [2, dd(2019, 1, 2)]], index_constructors=[Index, IndexDate])

        i2 = IndexHierarchy.from_labels([[2, dd(2019, 1, 2)], [3, dd(2019, 1, 3)]], index_constructors=[Index, IndexDate])

        i3 = i1.union(i2)

        self.assertEqual(
                i3.index_types.to_pairs(),
                ((0, Index), (1, IndexDate))
                )

    def test_hierarchy_set_operators_f1(self) -> None:
        dd = datetime.date

        i1 = IndexHierarchy.from_labels([[1, dd(2019, 1, 1)], [2, dd(2019, 1, 2)]], index_constructors=[Index, IndexDate], name='foo')

        i2 = IndexHierarchy.from_labels([[2, dd(2019, 1, 2)], [3, dd(2019, 1, 3)]], index_constructors=[Index, IndexDate], name='foo')

        i3 = i1.union(i2)

        self.assertEqual(
                i3.index_types.to_pairs(),
                ((0, Index), (1, IndexDate))
                )

        self.assertEqual(i3.name, 'foo')

    def test_hierarchy_set_operators_f2(self) -> None:
        dd = datetime.date

        i1 = IndexHierarchy.from_labels([[1, dd(2019, 1, 1)], [2, dd(2019, 1, 2)]], index_constructors=[Index, IndexDate], name='foo')

        i2 = IndexHierarchy.from_labels([[2, dd(2019, 1, 2)], [3, dd(2019, 1, 3)]], index_constructors=[Index, IndexDate])

        i3 = i1.union(i2)

        self.assertEqual(
                i3.index_types.to_pairs(),
                ((0, Index), (1, IndexDate))
                )
        # we do not propagate name if not matched
        self.assertEqual(i3.name, None)


    def test_hierarchy_set_operators_g(self) -> None:
        dd = datetime.date

        i1 = IndexHierarchy.from_labels([[1, dd(2019, 1, 1)], [2, dd(2019, 1, 2)]], index_constructors=[Index, IndexDate])

        i2 = IndexHierarchy.from_labels([[2, dd(2019, 1, 2)], [3, dd(2019, 1, 3)]],)

        i3 = i1.union(i2)

        # only if classes match do we pass on index types
        self.assertEqual(
                i3.index_types.to_pairs(),
                ((0, Index), (1, Index))
                )

    def test_hierarchy_set_operators_h(self) -> None:
        dd = datetime.date

        i1 = IndexHierarchy.from_labels([[1, dd(2019, 1, 1)], [2, dd(2019, 1, 2)]], index_constructors=[Index, IndexDate])

        i2 = IndexHierarchy.from_labels([[2, dd(2019, 1, 2), 'a'], [3, dd(2019, 1, 3), 'b']],)

        with self.assertRaises(ErrorInitIndex):
            i3 = i1.union(i2)

        with self.assertRaises(ErrorInitIndex):
            i3 = i1.union(np.arange(4))

    def test_hierarchy_set_operators_i(self) -> None:
        labels = (
                ('II', 'B'),
                ('II', 'A'),
                ('I', 'B'),
                ('I', 'A'),
                )

        ih1 = IndexHierarchy.from_labels(labels)

        f1 = Frame.from_records((('III', 1), ('III', 2)))

        with self.assertRaises(RuntimeError):
            _ = ih1.union(f1)

        ih2 = ih1.union(f1.values)
        self.assertEqual(ih2.values.tolist(),
                [['I', 'A'], ['I', 'B'], ['II', 'A'], ['II', 'B'], ['III', 1], ['III', 2]]
                )

    #---------------------------------------------------------------------------

    def test_hierarchy_set_operators_j(self) -> None:
        labels = (
                ('II', 'B'),
                ('II', 'A'),
                ('I', 'B'),
                ('I', 'A'),
                )

        ih1 = IndexHierarchy.from_labels(labels)

        ih2 = ih1.intersection(ih1.copy())
        self.assertEqual(id(ih1), id(ih2))
        self.assertTrue(ih1.equals(ih2))

        ih3 = ih1.union(ih1.copy())
        self.assertEqual(id(ih1), id(ih3))
        self.assertTrue(ih1.equals(ih3))

        ih4 = ih1.difference(ih1.copy())
        self.assertEqual(len(ih4), 0)

    def test_hierarchy_set_operators_k(self) -> None:
        labels = (
                ('II', 'B'),
                ('II', 'A'),
                ('I', 'B'),
                ('I', 'A'),
                )

        ih1 = IndexHierarchy.from_labels(labels)
        with self.assertRaises(RuntimeError):
            _ = ih1.intersection(['a', 'b'])

    def test_hierarchy_set_operators_l(self) -> None:
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

        labels = (
                ('II', 'A'),
                ('II', 'B'),
                ('IV', 'A'),
                ('IV', 'B'),
                )

        ih3 = IndexHierarchy.from_labels(labels)

        post1 = ih1.intersection(ih2, ih3)
        self.assertEqual(post1.values.tolist(),
                [['II', 'A'], ['II', 'B']])


        post2 = ih1.union(ih2, ih3)

        self.assertEqual(post2.values.tolist(),
                [['I', 'A'], ['II', 'A'], ['III', 'A'], ['IV', 'A'], ['I', 'B'], ['II', 'B'], ['III', 'B'], ['IV', 'B']]
                )

    def test_hierarchy_set_operators_m(self) -> None:
        labels = (
                ('II', 'B'),
                ('II', 'A'),
                ('I', 'B'),
                ('I', 'A'),
                )
        ih1 = IndexHierarchy.from_labels(labels)
        post = ih1.intersection((('I', 'B'), ('II', 'B')))
        self.assertEqual(post.values.tolist(),
                [['I', 'B'], ['II', 'B']]
                )

    def test_hierarchy_set_operators_n(self) -> None:
        # Test the short-circuit optimization for intersections when the result
        # will be empty
        ih1 = IndexHierarchy.from_product(('I', 'II'), ('A', 'B'))
        ih2 = IndexHierarchy.from_product(('II', 'III'), ('A', 'B'))
        ih3 = IndexHierarchy.from_product(('III', 'IV'), ('A', 'B'))

        post = ih1.intersection(ih2, ih3)
        assert len(post) == 0

    def test_hierarchy_set_operators_o(self) -> None:
        # Test the short-circuit optimization for differences when all elements are disjoint
        ih1 = IndexHierarchy.from_product(('I', 'II'), ('A', 'B'))
        ih2 = IndexHierarchy.from_product(('III', 'IV'), ('A', 'B'))
        ih3 = IndexHierarchy.from_product(('III', 'IV'), ('C', 'D'))

        post = ih1.difference(ih2, ih3)
        assert post.equals(ih1)

    def test_hierarchy_set_operators_p(self) -> None:
        # Add edge-case coverage for the generic 2D set approach invoked by IndexBase.
        ih = IndexHierarchy.from_product(('I', 'II'), ('A', 'B'))

        empty_mask = np.full(len(ih), False)

        post1 = IndexBase.intersection(ih, ih[empty_mask])
        post2 = IndexBase.intersection(ih, ih.values[empty_mask])
        post3 = IndexBase.intersection(ih[empty_mask], ih)

        post4 = IndexBase.difference(ih, ih.values[empty_mask])
        post5 = IndexBase.difference(ih, ih[empty_mask])
        post6 = IndexBase.difference(ih[empty_mask], ih)

        assert len(post1) == len(post2) == len(post3) == len(post6) == 0
        assert post4.equals(ih)
        assert post5.equals(ih)

        with self.assertRaises(RuntimeError):
            IndexBase.intersection(ih, np.array([[]]))

    #---------------------------------------------------------------------------

    def test_hierarchy_unary_operators_a(self) -> None:

        labels = (
                (1, 1),
                (1, 2),
                (2, 1),
                (2, 2),
                )
        ih1 = IndexHierarchyGO.from_labels(labels)
        ih1.append((3, 1))

        self.assertEqual((-ih1).tolist(),
                [[-1, -1], [-1, -2], [-2, -1], [-2, -2], [-3, -1]]
                )

    #---------------------------------------------------------------------------
    def test_hierarchy_binary_operators_a(self) -> None:

        labels = (
                (1, 1),
                (1, 2),
                (2, 1),
                (2, 2),
                )
        ih1 = IndexHierarchy.from_labels(labels)
        self.assertEqual((ih1*2).tolist(),
                [[2, 2], [2, 4], [4, 2], [4, 4]])

        self.assertEqual((-ih1).tolist(),
                [[-1, -1], [-1, -2], [-2, -1], [-2, -2]])

    def test_hierarchy_binary_operators_b(self) -> None:

        labels = (
                (1, 1),
                (1, 2),
                )
        ih1 = IndexHierarchy.from_labels(labels)

        labels = (
                (3, 3),
                (1, 2),
                )
        ih2 = IndexHierarchy.from_labels(labels)

        self.assertEqual((ih1 @ ih2).tolist(),
                [[4, 5], [5, 7]]
                )

        self.assertEqual((ih1.values @ ih2).tolist(),
                [[4, 5], [5, 7]]
                )

        self.assertEqual((ih1 @ ih2.values).tolist(),
                [[4, 5], [5, 7]]
                )

        self.assertEqual((ih1.values @ ih2.values).tolist(),
                [[4, 5], [5, 7]]
                )

    def test_hierarchy_binary_operators_c(self) -> None:

        labels = (
                (1, 1),
                (1, 2),
                (2, 1),
                (2, 2),
                )
        ih1 = IndexHierarchyGO.from_labels(labels)

        # by default, 1D multiplies by row (label)
        a1 = ih1 * Index((3, 4))
        self.assertEqual(a1.tolist(), [[3, 4], [3, 8], [6, 4], [6, 8]])

        a2 = ih1 + ih1
        self.assertEqual(a2.tolist(), [[2, 2], [2, 4], [4, 2], [4, 4]])

        a3 = ih1.via_T * Index((1, 2, 3, 4))
        self.assertEqual(a3.tolist(), [[1, 1], [2, 4], [6, 3], [8, 8]])

    def test_hierarchy_binary_operators_d(self) -> None:

        labels = (
                (1, 1),
                (1, 2),
                )
        ih1 = IndexHierarchy.from_labels(labels)

        labels = (
                (3, 3),
                (1, 2),
                )
        ih2 = IndexHierarchy.from_labels(labels)

        a1 = ih1 @ ih2
        a2 = ih1.values.tolist() @ ih2 # force rmatmul
        self.assertEqual(a1.tolist(), a2.tolist())

    def test_hierarchy_binary_operators_e(self) -> None:

        labels = (
                (1, 1, 1),
                (2, 2, 2),
                )
        ih1 = IndexHierarchy.from_labels(labels)

        labels = (
                (1, 1, 1),
                (2, 2, 2),
                )
        ih2 = IndexHierarchy.from_labels(labels)

        a1 = ih1 == ih2
        self.assertEqual(a1.tolist(), [[True, True, True], [True, True, True]])

    def test_hierarchy_binary_operators_f(self) -> None:

        a1 = np.arange(25).reshape(5,5)
        a2 = np.arange(start=24, stop=-1, step=-1).reshape(5,5)

        f1_idx_labels = [
                ['i_I', 1, 'i'],
                ['i_I', 2, 'i'],
                ['i_I', 3, 'i'],
                ['i_II', 1, 'i'],
                ['i_II', 3, 'i']]

        f2_idx_labels = [
                ['i_II', 2, 'i'],
                ['i_II', 1, 'i'],
                ['i_I', 3, 'i'],
                ['i_I', 1, 'i'],
                ['i_I', 4, 'i']]

        f1 = Frame(a1, index=IndexHierarchy.from_labels(f1_idx_labels))
        f2 = Frame(a2, index=IndexHierarchy.from_labels(f2_idx_labels))
        int_index = f1.index.intersection(f2.index)

        post = f1.reindex(int_index).index == f2.reindex(int_index).index
        self.assertEqual(post.tolist(),
                [[True, True, True], [True, True, True], [True, True, True]])

    def test_hierarchy_binary_operators_g(self) -> None:

        a1 = np.arange(25).reshape(5,5)
        a2 = np.arange(start=24, stop=-1, step=-1).reshape(5,5)
        f1_idx_labels = [
                ['i_I', 'i'],
                ['i_I', 2],
                ['i_I', 'iii'],
                ['i_II', 'i'],
                ['i_III', 'ii']]

        f2_idx_labels = [
                ['i_IV', 'i'],
                ['i_II', 'ii'],
                ['i_I', 2],
                ['i_I', 'ii'],
                ['i_I', 'iii']]

        f1 = Frame(a1, index=IndexHierarchy.from_labels(f1_idx_labels)) #type: ignore
        f2 = Frame(a2, index=IndexHierarchy.from_labels(f2_idx_labels)) #type: ignore
        int_index = f1.index.intersection(f2.index)

        post = f1.reindex(int_index).index == f2.reindex(int_index).index
        self.assertEqual(post.tolist(), [[True, True], [True, True]])

    def test_hierarchy_binary_operators_h(self) -> None:

        labels1 = (
                (1, 1),
                (2, 2),
                )
        ih1 = IndexHierarchy.from_labels(labels1)

        labels2 = (
                (1, 1, 1),
                (2, 2, 2),
                )
        ih2 = IndexHierarchy.from_labels(labels2)

        with self.assertRaises(NotImplementedError):
            _ = ih1 != ih2

        with self.assertRaises(NotImplementedError):
            _ = ih1 == ih2

        self.assertFalse(ih1.equals(ih2))

    def test_hierarchy_binary_operators_i(self) -> None:

        labels = (
                ('I', 'A'),
                ('I', 'B'),
                ('II', 'A'),
                ('II', 'B'),
                )

        ih1 = IndexHierarchy.from_labels(labels)

        ih2 = ih1 + '_'
        self.assertEqual(ih2.tolist(),
            [['I_', 'A_'], ['I_', 'B_'], ['II_', 'A_'], ['II_', 'B_']])

        ih3 = '_' + ih1
        self.assertEqual(ih3.tolist(),
            [['_I', '_A'], ['_I', '_B'], ['_II', '_A'], ['_II', '_B']])


        ih4 = ih1 * 2
        self.assertEqual(ih4.tolist(),
            [['II', 'AA'], ['II', 'BB'], ['IIII', 'AA'], ['IIII', 'BB']])

    def test_hierarchy_binary_operators_j(self) -> None:

        labels1 = (
                (1, 1),
                (2, 2),
                )
        ih1 = IndexHierarchy.from_labels(labels1)
        with self.assertRaises(ValueError):
            _ = ih1 * ih1.to_frame()

    #---------------------------------------------------------------------------
    def test_hierarchy_flat_a(self) -> None:

        labels = (
                ('I', 'A'),
                ('I', 'B'),
                ('II', 'A'),
                ('II', 'B'),
                )

        ih = IndexHierarchy.from_labels(labels)
        self.assertEqual(ih.flat().values.tolist(),
                [('I', 'A'), ('I', 'B'), ('II', 'A'), ('II', 'B')]
                )

    #---------------------------------------------------------------------------
    def test_hierarchy_add_level_a(self) -> None:

        labels = (
                ('I', 'A'),
                ('I', 'B'),
                ('II', 'A'),
                ('II', 'B'),
                )

        ih = IndexHierarchy.from_labels(labels)
        ih2 = ih.level_add('b')

        self.assertEqual(ih2.values.tolist(),
                [['b', 'I', 'A'], ['b', 'I', 'B'], ['b', 'II', 'A'], ['b', 'II', 'B']])
        self.assertEqual([ih2._loc_to_iloc(tuple(x)) for x in ih2.values],
                [0, 1, 2, 3])

    def test_hierarchy_add_level_b(self) -> None:

        labels = (
                ('I', 'A'),
                ('I', 'B'),
                ('II', 'A'),
                ('II', 'B'),
                )

        ih1 = IndexHierarchyGO.from_labels(labels)
        ih1.append(('III', 'A'))
        ih2 = ih1.level_add('x')

        self.assertEqual(ih1.values.tolist(),
                [['I', 'A'], ['I', 'B'], ['II', 'A'], ['II', 'B'], ['III', 'A']])

        self.assertEqual(ih2.values.tolist(),
                [['x', 'I', 'A'], ['x', 'I', 'B'], ['x', 'II', 'A'], ['x', 'II', 'B'], ['x', 'III', 'A']])

    def test_hierarchy_add_level_c(self) -> None:

        labels = ((1, 'A'), (1, 'B'), (2, 'A'), (2, 'B'))

        ih1 = IndexHierarchyGO.from_labels(labels)
        ih2 = ih1.level_add('x')
        # prove we reused the underlying block arrays
        self.assertEqual(ih1._blocks.mloc.tolist(), ih2._blocks.mloc[1:].tolist())

    def test_hierarchy_add_level_d(self) -> None:
        labels = ((1, 'A'), (1, 'B'), (2, 'A'), (2, 'B'))
        ih1 = IndexHierarchyGO.from_labels(labels)
        ih2 = ih1.level_add('1542-02', index_constructor=IndexYearMonth)

        self.assertEqual(ih2.index_types.values.tolist(),
                [IndexYearMonthGO, IndexGO, IndexGO],
                )
        self.assertTrue(
                (ih2.values_at_depth(0) == np.array(['1542-02', '1542-02', '1542-02', '1542-02'], dtype='datetime64[M]')).all()
                )

    def test_hierarchy_add_level_e(self) -> None:
        labels = ((1, 'A'), (1, 'B'), (2, 'A'), (2, 'B'))
        ih1 = IndexHierarchyGO.from_labels(labels)
        ih1.append(labels[0])

        with self.assertRaises(ErrorInitIndexNonUnique):
            ih1._update_array_cache()

    #---------------------------------------------------------------------------

    def test_hierarchy_drop_level_a(self) -> None:

        labels = (
                ('I', 'A', 1),
                ('I', 'B', 1),
                ('II', 'A', 1),
                ('II', 'B', 2),
                )

        ih = IndexHierarchy.from_labels(labels)
        ih2 = ih.level_drop(-1)

        self.assertEqual(ih2.values.tolist(),
                [['I', 'A'], ['I', 'B'], ['II', 'A'], ['II', 'B']])

    def test_hierarchy_drop_level_b(self) -> None:

        labels = (
                ('I', 'A', 1),
                ('I', 'B', 1),
                ('II', 'C', 1),
                ('II', 'C', 2),
                )

        ih = IndexHierarchy.from_labels(labels)
        ih2 = ih.level_drop(1)
        assert isinstance(ih2, IndexHierarchy)
        self.assertEqual(ih2.values.tolist(),
            [['A', 1], ['B', 1], ['C', 1], ['C', 2]])

        with self.assertRaises(ErrorInitIndex):
            ih2.level_drop(1)

    def test_hierarchy_drop_level_c(self) -> None:

        labels = (
                ('I', 'A', 1),
                ('I', 'B', 2),
                ('II', 'C', 3),
                ('II', 'C', 4),
                )

        ih = IndexHierarchy.from_labels(labels)
        self.assertEqual(ih.level_drop(1).values.tolist(),
                [['A', 1], ['B', 2], ['C', 3], ['C', 4]])

    def test_hierarchy_drop_level_d(self) -> None:

        labels = (
                ('A', 1),
                ('B', 2),
                ('C', 3),
                ('C', 4),
                )

        ih = IndexHierarchy.from_labels(labels)
        self.assertEqual(ih.level_drop(1).values.tolist(),
                [1, 2, 3, 4])

    def test_hierarchy_drop_level_e(self) -> None:

        ih = IndexHierarchy.from_product(('a',), (1,), ('x', 'y'))
        self.assertEqual(ih.level_drop(2).values.tolist(),
                ['x', 'y'])

        self.assertEqual(ih.level_drop(1).values.tolist(),
                [[1, 'x'], [1, 'y']])

    def test_hierarchy_drop_level_f(self) -> None:

        ih = IndexHierarchy.from_product(('a',), (1,), ('x',))
        self.assertEqual(ih.level_drop(1).values.tolist(),
                [[1, 'x']])

    def test_hierarchy_drop_level_g(self) -> None:

        ih = IndexHierarchy.from_product(('a',), (1,), ('x',))
        with self.assertRaises(ValueError):
            _ = ih.level_drop(0)

    def test_hierarchy_drop_level_h(self) -> None:

        labels = (
                ('I', 'A', 1, False),
                ('I', 'B', 2, True),
                ('II', 'C', 3, False),
                ('II', 'C', 4, True),
                )

        ih = IndexHierarchy.from_labels(labels)

        post1 = ih.level_drop(-1)
        assert isinstance(post1, IndexHierarchy) # mypy
        self.assertEqual(ih._blocks.mloc[:-1].tolist(), post1._blocks.mloc.tolist())

        with self.assertRaises(ErrorInitIndexNonUnique):
            ih.level_drop(-2)

        post2 = ih.level_drop(1)
        assert isinstance(post2, IndexHierarchy) # mypy
        self.assertEqual(ih._blocks.mloc[1:].tolist(), post2._blocks.mloc.tolist())

        post3 = ih.level_drop(2)
        assert isinstance(post3, IndexHierarchy) # mypy
        self.assertEqual(ih._blocks.mloc[2:].tolist(), post3._blocks.mloc.tolist())

    def test_hierarchy_drop_level_i(self) -> None:

        labels = (
                ('I', 'A', 1, False),
                ('I', 'B', 2, True),
                ('II', 'C', 3, None),
                ('II', 'C', 4, ...),
                )

        ih1 = IndexHierarchy.from_labels(labels, name=('a', 'b', 'c', 'd'))

        ih2 = ih1.level_drop(1)
        self.assertEqual(ih2.name, ('b', 'c', 'd'))

        ih3 = ih1.level_drop(2)
        self.assertEqual(ih3.name, ('c', 'd'))

        ih4 = ih1.level_drop(3)
        self.assertEqual(ih4.name, 'd')

    def test_hierarchy_drop_level_j(self) -> None:

        labels = (
                ('I', 'A', 1, False),
                ('II', 'B', 2, True),
                ('III', 'C', 3, None),
                ('IV', 'D', 4, ...),
                )

        ih1 = IndexHierarchy.from_labels(labels, name=('a', 'b', 'c', 'd'))

        ih2 = ih1.level_drop(-1)
        self.assertEqual(ih2.name, ('a', 'b', 'c'))

        ih3 = ih1.level_drop(-2)
        self.assertEqual(ih3.name, ('a', 'b'))

        ih4 = ih1.level_drop(-3)
        self.assertEqual(ih4.name, 'a')

    def test_hierarchy_drop_level_k(self) -> None:
        tree = {
            'f1': {'i_I': ('1',), 'i_II': ('2',)},
            'f2': {'c_I': ('A', ), 'c_II': ('B',)}
        }
        ih = IndexHierarchy.from_tree(tree)
        self._assert_to_tree_consistency(ih)
        post = ih.level_drop(1)

        # This used to raise `ValueError: negative dimensions are not allowed`
        # as the `ih` hadn't properly updated its internal cache before creation
        post.display()

    def test_hierarchy_drop_level_l(self) -> None:
        labels = (
                ('I', 'A', 1, False),
                ('I', 'B', 2, True),
                ('II', 'C', 3, False),
                ('II', 'D', 3, False),
                )
        ih = IndexHierarchy.from_labels(labels)
        with self.assertRaises(ErrorInitIndexNonUnique):
            post = ih.level_drop(2)

    def test_hierarchy_drop_level_m(self) -> None:
        ih1= IndexHierarchy.from_product(('a', 'b'), (1, 2), name=('x', 'y'))
        self.assertTrue(ih1._name_is_names())
        with self.assertRaises(ValueError):
            _ = ih1.level_drop(0)


    #---------------------------------------------------------------------------

    def test_hierarchy_drop_loc_a(self) -> None:

        labels = (
                ('I', 'A', 1),
                ('I', 'B', 1),
                ('II', 'A', 1),
                ('II', 'B', 2),
                )

        ih1 = IndexHierarchy.from_labels(labels)
        ih2 = ih1._drop_loc([('I', 'B', 1), ('II', 'B', 2)])

        self.assertEqual(ih2.to_frame().to_pairs(0),
                ((0, ((0, 'I'), (1, 'II'))), (1, ((0, 'A'), (1, 'A'))), (2, ((0, 1), (1, 1))))
                )

    def test_hierarchy_drop_loc_b(self) -> None:

        labels = (
                ('I', 'A', 1),
                ('I', 'B', 1),
                ('II', 'A', 1),
                ('II', 'B', 2),
                )

        ih1 = IndexHierarchyGO.from_labels(labels)

        ih1.append(('II', 'B', 3))
        ih2 = ih1._drop_loc([('I', 'B', 1), ('II', 'B', 2)])
        self.assertEqual(ih2.to_frame().to_pairs(0),
                ((0, ((0, 'I'), (1, 'II'), (2, 'II'))), (1, ((0, 'A'), (1, 'A'), (2, 'B'))), (2, ((0, 1), (1, 1), (2, 3))))
                )

    #---------------------------------------------------------------------------

    def test_hierarchy_boolean_loc(self) -> None:
        records = (
                ('a', 999999, 0.1),
                ('a', 201810, 0.1),
                ('b', 999999, 0.4),
                ('b', 201810, 0.4))
        f1 = Frame.from_records(records, columns=list('abc'))

        f1 = f1.set_index_hierarchy(['a', 'b'], drop=False)
        self.assertEqual(f1.index.names, ('a', 'b'))

        f2 = f1.loc[f1['b'] == 999999]

        self.assertEqual(f2.to_pairs(0),
                (('a', ((('a', 999999), 'a'), (('b', 999999), 'b'))), ('b', ((('a', 999999), 999999), (('b', 999999), 999999))), ('c', ((('a', 999999), 0.1), (('b', 999999), 0.4)))))

        f3 = f1.loc[Series([False, True], index=(('b', 999999), ('b', 201810)))]
        self.assertEqual(f3.to_pairs(0),
                (('a', ((('b', 201810), 'b'),)), ('b', ((('b', 201810), 201810),)), ('c', ((('b', 201810), 0.4),))))

    def test_hierarchy_name_a(self) -> None:

        idx1 = IndexHierarchy.from_product(list('ab'), list('xy'), name='q')
        self.assertEqual(idx1.name, 'q')

        idx2 = idx1.rename('w')
        self.assertEqual(idx2.name, 'w')
        # will provide one for each level
        self.assertEqual(idx2.names, ('__index0__', '__index1__'))

    def test_hierarchy_name_b(self) -> None:

        idx1 = IndexHierarchyGO.from_product(list('ab'), list('xy'), name='q')
        idx2 = idx1.rename('w')

        self.assertEqual(idx1.name, 'q')
        self.assertEqual(idx2.name, 'w')

        idx1.append(('c', 'c'))
        idx2.append(('x', 'x'))

        self.assertEqual(
                idx1.values.tolist(),
                [['a', 'x'], ['a', 'y'], ['b', 'x'], ['b', 'y'], ['c', 'c']]
                )

        self.assertEqual(
                idx2.values.tolist(),
                [['a', 'x'], ['a', 'y'], ['b', 'x'], ['b', 'y'], ['x', 'x']]
                )

    def test_hierarchy_name_c(self) -> None:

        idx1 = IndexHierarchyGO.from_product(list('ab'), list('xy'), name='q')
        idx2 = idx1.rename(('a', 'b', 'c'))

        # since the name attr is the wrong size, names use the generic from
        self.assertEqual(idx2.names, ('__index0__', '__index1__'))

    def test_hierarchy_name_d(self) -> None:

        idx1 = IndexHierarchy.from_product(list('ab'), list('xy'), name='q')
        self.assertEqual(idx1.name, 'q')

    def test_hierarchy_display_b(self) -> None:

        idx1 = IndexHierarchy.from_product(list('ab'), list('xy'), name='q')

        match = tuple(idx1.display(DisplayConfig(type_color=False)))

        self.assertEqual(
                match,
                (['<IndexHierarchy: q>', ''], ['a', 'x'], ['a', 'y'], ['b', 'x'], ['b', 'y'], ['<<U1>', '<<U1>'])
                )

    #---------------------------------------------------------------------------
    def test_hierarchy_to_frame_a(self) -> None:

        ih1 = IndexHierarchy.from_product(list('ab'), list('xy'), name='q')

        self.assertEqual(ih1.to_frame().to_pairs(0),
                ((0, ((0, 'a'), (1, 'a'), (2, 'b'), (3, 'b'))), (1, ((0, 'x'), (1, 'y'), (2, 'x'), (3, 'y'))))
                )

        f2 = ih1.to_frame_go()
        f2[-1] = None

        self.assertEqual(f2.to_pairs(0),
                ((0, ((0, 'a'), (1, 'a'), (2, 'b'), (3, 'b'))), (1, ((0, 'x'), (1, 'y'), (2, 'x'), (3, 'y'))), (-1, ((0, None), (1, None), (2, None), (3, None))))
                )

    def test_hierarchy_to_frame_b(self) -> None:

        ih1 = IndexHierarchy.from_product(list('ab'), [10.1, 20.2], name='q')
        f1 = ih1.to_frame()
        self.assertEqual(f1.dtypes.to_pairs(),
                ((0, np.dtype('<U1')), (1, np.dtype('float64')))
                )

    #---------------------------------------------------------------------------

    def test_hierarchy_to_html_datatables(self) -> None:

        ih1 = IndexHierarchy.from_product(list('ab'), list('xy'), name='q')

        with temp_file('.html', path=True) as fp:
            ih1.to_html_datatables(fp, show=False)
            with open(fp, encoding='utf-8') as file:
                data = file.read()
                self.assertTrue('SFTable' in data)
                self.assertTrue(len(data) > 1000)

    def test_hierarchy_to_pandas_a(self) -> None:

        idx1 = IndexHierarchy.from_product(list('ab'), list('xy'), name='q')

        pdidx = idx1.to_pandas()
        self.assertEqual(pdidx.name, 'q')
        self.assertEqual(
                idx1.values.tolist(),
                [list(x) for x in pdidx.values.tolist()])

    def test_hierarchy_from_pandas_a(self) -> None:
        import pandas

        pdidx = pandas.MultiIndex.from_product((('I', 'II'), ('A', 'B')))

        idx = IndexHierarchy.from_pandas(pdidx)

        self.assertEqual(idx.values.tolist(),
                [['I', 'A'], ['I', 'B'], ['II', 'A'], ['II', 'B']]
                )

    def test_hierarchy_from_pandas_b(self) -> None:
        import pandas

        pdidx = pandas.MultiIndex.from_product((('I', 'II'), ('A', 'B')))

        idx = IndexHierarchyGO.from_pandas(pdidx)

        self.assertEqual(idx.values.tolist(),
                [['I', 'A'], ['I', 'B'], ['II', 'A'], ['II', 'B']]
                )

        self.assertEqual(idx.values.tolist(),
                [['I', 'A'], ['I', 'B'], ['II', 'A'], ['II', 'B']])

        idx.append(('III', 'A'))

        self.assertEqual(idx.values.tolist(),
                [['I', 'A'], ['I', 'B'], ['II', 'A'], ['II', 'B'], ['III', 'A']])

    def test_hierarchy_from_pandas_c(self) -> None:
        import pandas

        pdidx = pandas.MultiIndex.from_tuples((('I', 'II'), ('I', 'II')))

        with self.assertRaises(ErrorInitIndex):
            IndexHierarchyGO.from_pandas(pdidx)

    def test_hierarchy_from_pandas_d(self) -> None:
        import pandas

        pdidx = pandas.MultiIndex.from_product(
            ((np.datetime64('2000-01-01'), np.datetime64('2000-01-02')), range(3)))

        idx1 = IndexHierarchyGO.from_pandas(pdidx)
        self.assertEqual([IndexNanosecondGO, IndexGO], idx1.index_types.values.tolist())

        idx2 = IndexHierarchy.from_pandas(pdidx)
        self.assertEqual([IndexNanosecond, Index], idx2.index_types.values.tolist())

    def test_hierarchy_from_pandas_unbloats(self) -> None:
        import pandas

        mi = pandas.MultiIndex([tuple("ABC"), tuple("DEF")], [[0, 0, 1], [1,2,2]])

        ih1 = IndexHierarchy.from_pandas(mi)
        ih2 = IndexHierarchy.from_values_per_depth([tuple("AAB"), tuple("EFF")])

        self.assertTrue(all(a.equals(b) for (a, b) in zip(ih1._indices, ih2._indices)))

        self.assertTrue(ih1.equals(ih2))
        self.assertTrue(ih2.equals(ih1))
        self.assertTrue(ih1.loc[ih2].equals(ih2.loc[ih1]))
        self.assertTrue(ih2.loc[ih1].equals(ih1.loc[ih2]))

    def test_hierarchy_from_pandas_fails_non_multiindex(self) -> None:
        import pandas

        i = pandas.Index(["ABC", "DEF"])
        with self.assertRaises(ErrorInitIndex):
            IndexHierarchy.from_pandas(i)

    #---------------------------------------------------------------------------

    def test_hierarchy_copy_a(self) -> None:

        labels = (('I', 'A'), ('I', 'B'), ('II', 'A'), ('II', 'B'))

        ih1 = IndexHierarchy.from_labels(labels)
        ih2 = ih1.copy()

        self.assertEqual(ih2.values.tolist(),
            [['I', 'A'], ['I', 'B'], ['II', 'A'], ['II', 'B']])

    def test_hierarchy_copy_b(self) -> None:

        labels = (('I', 'A'), ('I', 'B'), ('II', 'A'), ('II', 'B'))

        ih1 = IndexHierarchyGO.from_labels(labels)
        ih2 = ih1.copy()
        ih2.append(('II', 'C'))

        self.assertEqual(ih2.values.tolist(),
            [['I', 'A'], ['I', 'B'], ['II', 'A'], ['II', 'B'], ['II', 'C']]
            )

        self.assertEqual(ih1.values.tolist(),
            [['I', 'A'], ['I', 'B'], ['II', 'A'], ['II', 'B']]
            )

    def test_hierarchy_copy_c(self) -> None:

        labels = (('I', 'A'), ('I', 'B'), ('II', 'A'), ('II', 'B'))

        ih1 = IndexHierarchyGO.from_labels(labels)
        ih2 = ih1.copy()
        ih1.append(('II', 'C'))

        ih1._update_array_cache()

        self.assertEqual([i.tolist() for i in ih2._indexers],
            [[0, 0, 1, 1], [0, 1, 0, 1]]
            )

        self.assertEqual([i.tolist() for i in ih1._indexers],
            [[0, 0, 1, 1, 1], [0, 1, 0, 1, 2]]
            )

    def test_hierarchy_deepcopy_a(self) -> None:

        groups = Index(('A', 'B', 'C'))
        dates = IndexDate.from_date_range('2018-01-01', '2018-01-04')
        observations = Index(('x', 'y'))
        ih1 = IndexHierarchy.from_product(groups, dates, observations)

        ih2 = copy.deepcopy(ih1)
        self.assertEqual(ih1.values.tolist(), ih2.values.tolist())

    def test_hierarchy_deepcopy_b(self) -> None:


        idx1 = Index(('A', 'B', 'C'))
        idx2 = Index(('x', 'y'))
        idx3 = Index((4, 5, 6))

        ih1 = IndexHierarchyGO.from_index_items(dict(a=idx1, b=idx2, c=idx3).items())
        ih1.append(('c', 7))

        ih2 = copy.deepcopy(ih1)

        ih2.append(('c', 8))
        ih1.append(('d', 8))

        self.assertEqual(ih1.values.tolist(),
                [['a', 'A'], ['a', 'B'], ['a', 'C'], ['b', 'x'], ['b', 'y'], ['c', 4], ['c', 5], ['c', 6], ['c', 7], ['d', 8]]
                )

        self.assertEqual(ih2.values.tolist(),
                [['a', 'A'], ['a', 'B'], ['a', 'C'], ['b', 'x'], ['b', 'y'], ['c', 4], ['c', 5], ['c', 6], ['c', 7], ['c', 8]]
                )

    #---------------------------------------------------------------------------

    def test_hierarchy_ufunc_axis_skipna_a(self) -> None:

        ih1 = IndexHierarchy.from_product((10, 20), (3, 7))
        with self.assertRaises(NotImplementedError):
            _ = ih1.std()

        with self.assertRaises(NotImplementedError):
            _ = ih1.sum()

    def test_hierarchy_ufunc_shape_skipna_a(self) -> None:

        ih1 = IndexHierarchy.from_product((10, 20), (3, 7))
        with self.assertRaises(NotImplementedError):
            _ = ih1.cumprod()

        with self.assertRaises(NotImplementedError):
            _ = ih1.cumsum()


    #---------------------------------------------------------------------------

    def test_hierarchy_pickle_a(self) -> None:

        a = IndexHierarchy.from_product((10, 20), (3, 7))
        b = IndexHierarchy.from_product(('a', 'b'), ('x', 'y'))

        for index in (a, b):
            # force creating of ._labels
            self.assertTrue(len(index.values), len(index))

            pbytes = pickle.dumps(index)
            index_new = pickle.loads(pbytes)

            for v in index: # iter labels (tuples here)
                self.assertFalse(index_new.values.flags.writeable)
                self.assertEqual(index_new.loc[v], index.loc[v])

    #---------------------------------------------------------------------------
    def test_hierarchy_sort_a(self) -> None:

        ih1 = IndexHierarchy.from_product((1, 2), (30, 70))

        self.assertEqual(ih1.sort(ascending=False).values.tolist(),
            [[2, 70], [2, 30], [1, 70], [1, 30]]
            )

    def test_hierarchy_sort_b(self) -> None:

        ih1 = IndexHierarchy.from_labels(((1, 1000), (30, 25), (100, 3)))

        self.assertEqual(ih1.sort(key=lambda i: i / -1).values.tolist(),
                [[100, 3], [30, 25], [1, 1000]])

        self.assertEqual(ih1.sort(
                key=lambda i: i.values.sum(axis=1)).values.tolist(),
                [[30, 25], [100, 3], [1, 1000]],
                )

    def test_hierarchy_sort_c(self) -> None:

        ih1 = IndexHierarchy.from_product(('a', 'b'), (1, 5, 3, -4), ('y', 'z', 'x'))

        with self.assertRaises(RuntimeError):
            ih1.sort(ascending=(True, False))

        self.assertEqual(ih1.sort(ascending=(True, False, True)).values.tolist(),
            [['a', 5, 'x'], ['a', 5, 'y'], ['a', 5, 'z'], ['a', 3, 'x'], ['a', 3, 'y'], ['a', 3, 'z'], ['a', 1, 'x'], ['a', 1, 'y'], ['a', 1, 'z'], ['a', -4, 'x'], ['a', -4, 'y'], ['a', -4, 'z'], ['b', 5, 'x'], ['b', 5, 'y'], ['b', 5, 'z'], ['b', 3, 'x'], ['b', 3, 'y'], ['b', 3, 'z'], ['b', 1, 'x'], ['b', 1, 'y'], ['b', 1, 'z'], ['b', -4, 'x'], ['b', -4, 'y'], ['b', -4, 'z']]
            )

        self.assertEqual(ih1.sort(ascending=(True, False, False)).values.tolist(),
            [['a', 5, 'z'], ['a', 5, 'y'], ['a', 5, 'x'], ['a', 3, 'z'], ['a', 3, 'y'], ['a', 3, 'x'], ['a', 1, 'z'], ['a', 1, 'y'], ['a', 1, 'x'], ['a', -4, 'z'], ['a', -4, 'y'], ['a', -4, 'x'], ['b', 5, 'z'], ['b', 5, 'y'], ['b', 5, 'x'], ['b', 3, 'z'], ['b', 3, 'y'], ['b', 3, 'x'], ['b', 1, 'z'], ['b', 1, 'y'], ['b', 1, 'x'], ['b', -4, 'z'], ['b', -4, 'y'], ['b', -4, 'x']]
            )

        self.assertEqual(ih1.sort(ascending=(False, True, False)).values.tolist(),
            [['b', -4, 'z'], ['b', -4, 'y'], ['b', -4, 'x'], ['b', 1, 'z'], ['b', 1, 'y'], ['b', 1, 'x'], ['b', 3, 'z'], ['b', 3, 'y'], ['b', 3, 'x'], ['b', 5, 'z'], ['b', 5, 'y'], ['b', 5, 'x'], ['a', -4, 'z'], ['a', -4, 'y'], ['a', -4, 'x'], ['a', 1, 'z'], ['a', 1, 'y'], ['a', 1, 'x'], ['a', 3, 'z'], ['a', 3, 'y'], ['a', 3, 'x'], ['a', 5, 'z'], ['a', 5, 'y'], ['a', 5, 'x']]
            )

    def test_hierarchy_sort_d(self) -> None:
        ih1 = IndexHierarchy.from_labels((), depth_reference=2)
        ih2 = ih1.sort()
        self.assertEqual(ih1.shape, ih2.shape)

    #---------------------------------------------------------------------------
    def test_hierarchy_isin_a(self) -> None:

        ih1 = IndexHierarchy.from_product((1, 2), (30, 70), (2, 5))

        post = ih1.isin([(2, 30, 2),])
        self.assertEqual(post.dtype, bool)
        self.assertEqual(post.tolist(),
            [False, False, False, False, True, False, False, False])

        extract = ih1.loc[post]
        self.assertEqual(extract.values.shape, (1, 3))

    def test_hierarchy_isin_b(self) -> None:

        ih1 = IndexHierarchy.from_product((1, 2), (30, 70), (2, 5))

        with self.assertRaises(RuntimeError):
            ih1.isin([3,4,5]) #type: ignore # not an iterable of iterables

        post = ih1.isin(([3,4], [2,5,1,5]))
        self.assertEqual(post.sum(), 0)

    def test_hierarchy_isin_c(self) -> None:

        ih1 = IndexHierarchy.from_product((1, 2), ('a', 'b'), (2, 5))

        # multiple matches

        post1 = ih1.isin([(1, 'a', 5), (2, 'b', 2)])
        self.assertEqual(post1.tolist(),
                [False, True, False, False, False, False, True, False])

        post2 = ih1.isin(ih1)
        self.assertEqual(post2.sum(), len(ih1))

    def test_hierarchy_isin_d(self) -> None:

        ih1 = IndexHierarchy.from_product((1, 2), (30, 70), (2, 5))

        # Index is an iterable
        index_iter1 = (val for val in (2, 30, 2))
        index_non_iter = (1, 70, 5)

        post = ih1.isin([index_iter1, index_non_iter])
        self.assertEqual(post.dtype, bool)
        self.assertEqual(post.tolist(),
            [False, False, False, True, True, False, False, False])

        extract = ih1.loc[post]
        self.assertEqual(extract.values.shape, (2, 3))

    def test_hierarchy_roll_a(self) -> None:

        ih1 = IndexHierarchy.from_product((1, 2), (30, 70))

        self.assertEqual(ih1.roll(1).values.tolist(),
            [[2, 70], [1, 30], [1, 70], [2, 30]]
            )

        self.assertEqual(ih1.roll(2).values.tolist(),
            [[2, 30], [2, 70], [1, 30], [1, 70]]
            )

    def test_hierarchy_roll_b(self) -> None:

        ih1 = IndexHierarchy.from_labels((('a', 1), ('b', 20), ('c', 400), ('d', 50)))

        self.assertEqual(
                ih1.roll(1).values.tolist(),
                [['d', 50], ['a', 1], ['b', 20], ['c', 400]]
                )

        self.assertEqual(
                ih1.roll(-1).values.tolist(),
                [['b', 20], ['c', 400], ['d', 50], ['a', 1]]
                )

    def test_hierarchy_dtypes_a(self) -> None:
        idx1 = Index(('A', 'B'))
        idx2 = IndexDate.from_date_range('2019-01-05', '2019-01-08')
        idx3 = Index((1, 2))
        hidx = IndexHierarchy.from_product(idx1, idx2, idx3)

        self.assertEqual(
            [(x, y.kind) for x, y in hidx.dtypes.to_pairs()],
            [(0, 'U'), (1, 'M'), (2, 'i')]
            )

    def test_hierarchy_dtypes_b(self) -> None:
        idx1 = Index(('A', 'B'), name='a')
        idx2 = IndexDate.from_date_range('2019-01-05', '2019-01-08', name='b')
        idx3 = Index((1, 2), name='c')
        hidx = IndexHierarchy.from_product(idx1, idx2, idx3)

        self.assertEqual(
            [(x, y.kind) for x, y in hidx.dtypes.to_pairs()],
            [('a', 'U'), ('b', 'M'), ('c', 'i')]
            )

    def test_hierarchy_index_types_a(self) -> None:
        idx1 = Index(('A', 'B'))
        idx2 = IndexDate.from_date_range('2019-01-05', '2019-01-08')
        idx3 = Index((1, 2))
        hidx = IndexHierarchy.from_product(idx1, idx2, idx3)

        self.assertEqual(
            [(x, y.__name__) for x, y in hidx.index_types.to_pairs()],
            [(0, 'Index'), (1, 'IndexDate'), (2, 'Index')]
            )

    def test_hierarchy_index_types_b(self) -> None:
        idx1 = Index(('A', 'B'), name='a')
        idx2 = IndexDate.from_date_range('2019-01-05', '2019-01-08', name='b')
        idx3 = Index((1, 2), name='c')
        hidx = IndexHierarchy.from_product(idx1, idx2, idx3)

        self.assertEqual(
            [(x, y.__name__) for x, y in hidx.index_types.to_pairs()],
            [('a', 'Index'), ('b', 'IndexDate'), ('c', 'Index')]
            )

    #---------------------------------------------------------------------------
    def test_hierarchy_label_widths_at_depth_a(self) -> None:
        idx1 = Index(('A', 'B'), name='a')
        idx2 = IndexDate.from_date_range('2019-01-05', '2019-01-08', name='b')
        idx3 = Index((1, 2), name='c')
        hidx = IndexHierarchyGO.from_product(idx1, idx2, idx3)

        hidx.append(('B', np.datetime64('2019-01-05'), 3))

        self.assertEqual(tuple(hidx.label_widths_at_depth(0)),
                (('A', 8), ('B', 9))
                )

        self.assertEqual(tuple(hidx.label_widths_at_depth(1)),
                ((np.datetime64('2019-01-05'), 2),
                (np.datetime64('2019-01-06'), 2),
                (np.datetime64('2019-01-07'), 2),
                (np.datetime64('2019-01-08'), 2),
                (np.datetime64('2019-01-05'), 2),
                (np.datetime64('2019-01-06'), 2),
                (np.datetime64('2019-01-07'), 2),
                (np.datetime64('2019-01-08'), 2),
                (np.datetime64('2019-01-05'), 1),
                ))

        self.assertEqual(tuple(hidx.label_widths_at_depth(2)),
                ((1, 1), (2, 1), (1, 1), (2, 1), (1, 1), (2, 1), (1, 1), (2, 1), (1, 1), (2, 1), (1, 1), (2, 1), (1, 1), (2, 1), (1, 1), (2, 1), (3, 1))
                )

        self.assertEqual(tuple(hidx.label_widths_at_depth(2)), tuple(hidx.label_widths_at_depth([2])))


    def test_hierarchy_label_widths_at_depth_b(self) -> None:
        idx1 = Index(('A', 'B'), name='a')
        idx2 = IndexDate.from_date_range('2019-01-05', '2019-01-08', name='b')
        idx3 = Index((1, 2), name='c')
        hidx = IndexHierarchy.from_product(idx1, idx2, idx3)

        with self.assertRaises(NotImplementedError):
            _ = next(hidx.label_widths_at_depth(None))

        with self.assertRaises(NotImplementedError):
            _ = next(hidx.label_widths_at_depth([0, 1]))

    #---------------------------------------------------------------------------

    def test_hierarchy_astype_a(self) -> None:

        ih1 = IndexHierarchy.from_product((1, 2), ('a', 'b'), (2, 5))

        ih2 = ih1.astype[[0, 2]](float)

        self.assertEqual(ih2.dtypes.values.tolist(),
                [np.dtype('float64'), np.dtype('<U1'), np.dtype('float64')])

    def test_hierarchy_astype_b(self) -> None:

        ih1 = IndexHierarchy.from_product((1, 2), (100, 200))
        ih2 = ih1.astype(float)
        self.assertEqual(ih2.dtypes.values.tolist(),
                [np.dtype('float64'), np.dtype('float64')])

    def test_hierarchy_astype_c(self) -> None:
        ih1 = IndexHierarchy.from_product((1, 2), (100, 200), ('2020-01', '2020-03'))

        self.assertEqual(
                ih1.astype[[0, 1]](float).dtypes.to_pairs(),
                ((0, np.dtype('float64')), (1, np.dtype('float64')), (2, np.dtype('<U7')))
                )

    def test_hierarchy_astype_d(self) -> None:
        ih1 = IndexHierarchy.from_product(
            ('1945-01-02', '1843-07-07'), ('2020-01', '2020-03'))

        self.assertEqual(
                ih1.astype('datetime64[M]').index_types.to_pairs(),
                ((0, IndexYearMonth),
                (1, IndexYearMonth))
                )

    @skip_win
    def test_hierarchy_astype_e(self) -> None:
        ih1 = IndexHierarchy.from_product((1, 2), (100, 200), ('2020-01', '2020-03'))

        self.assertEqual(
                ih1.astype[[0, 1]](float).dtypes.to_pairs(),
                ((0, np.dtype('float64')), (1, np.dtype('float64')), (2, np.dtype('<U7')))
                )

        ih2 = ih1.astype[2]('datetime64[M]')

        self.assertEqual(
                ih2.dtypes.to_pairs(),
                ((0, np.dtype('int64')), (1, np.dtype('int64')), (2, np.dtype('<M8[M]')))
                )

        self.assertEqual(ih2.index_types.to_pairs(),
                ((0, Index), (1, Index), (2, IndexYearMonth))
                )

        post = ih2.loc[HLoc[:, 200, '2020-03']]
        self.assertEqual(post.shape, (2, 3))
        self.assertEqual(post.dtypes.to_pairs(),
                ((0, np.dtype('int64')), (1, np.dtype('int64')), (2, np.dtype('<M8[M]')))
                )

    def test_hierarchy_astype_f(self) -> None:

        idx1 = IndexHierarchy.from_labels([("1", 2), (3, "4"), ('5', '6')], name=(5, 6))
        idx2 = idx1.astype[np.array([True, False])](str)
        self.assertEqual([d.kind for d in idx2.dtypes.values], ['U', 'O'])

        idx3 = idx2.astype[np.array([True, True])](str)
        self.assertEqual([d.kind for d in idx3.dtypes.values], ['U', 'U'])

    def test_hierarchy_astype_g1(self) -> None:
        d = datetime.date
        days = [d(2020, 1, 1), d(2020, 1, 2), d(2020, 1, 3)]
        idx1 = IndexHierarchy.from_product(range(3), days)
        idx2 = idx1.astype[np.array([False, True])]("datetime64[D]")
        self.assertEqual(idx2.index_types.values.tolist(), [Index, IndexDate])

    def test_hierarchy_astype_g2(self) -> None:
        d = datetime.date
        days = [d(2020, 1, 1), d(2020, 1, 2), d(2020, 1, 3)]
        idx1 = IndexHierarchy.from_product(range(3), days)
        with self.assertRaises(KeyError):
            idx2 = idx1.astype[[False, True]]("datetime64[D]")

    def test_hierarchy_astype_g3(self) -> None:
        d = datetime.date
        days = [d(2020, 1, 1), d(2020, 1, 2), d(2020, 1, 3)]
        idx1 = IndexHierarchy.from_product(range(3), days)
        with self.assertRaises(RuntimeError):
            idx2 = idx1.astype[np.array([False, True])]((str, int, bool))

    def test_hierarchy_astype_h(self) -> None:
        d = datetime.date
        days = [d(2020, 1, 1), d(2020, 1, 2), d(2020, 1, 3)]
        idx1 = IndexHierarchy.from_product(range(3), days, (10, 20))
        idx2 = idx1.astype({0:str, 2:str})
        self.assertEqual([dt.kind for dt in idx2.dtypes.values], ['U', 'O', 'U'])

        idx3 = idx1.astype((float, np.datetime64, float))
        self.assertEqual([dt.kind for dt in idx3.dtypes.values], ['f', 'M', 'f'])

    def test_hierarchy_astype_i(self) -> None:

        ih1 = IndexHierarchy.from_product((1, 2), ('a', 'b'), (2, 5))
        with self.assertRaises(ValueError):
            _ = ih1.astype(float)
        with self.assertRaises(TypeError):
            ih1.astype(IndexDate)

    def test_hierarchy_astype_j(self) -> None:
        ih1 = IndexHierarchy.from_product((1, 2), ('a', 'b'), name=('x', 'y'))
        ih2 = ih1.astype[0](float)
        self.assertEqual(ih1.names, ih2.names)


    #---------------------------------------------------------------------------

    @skip_win
    def test_hierarchy_values_at_depth_a(self) -> None:
        ih1 = IndexHierarchy.from_product((1, 2), (100, 200), ('2020-01', '2020-03'))
        post = ih1.values_at_depth([0, 1])
        self.assertEqual(post.shape, (8, 2))
        self.assertEqual(post.dtype, np.dtype(int))
        self.assertEqual(ih1.values_at_depth(2).dtype, np.dtype('<U7'))

    def test_hierarchy_values_at_depth_b(self) -> None:
        ih1 = IndexHierarchyGO.from_product((1, 2), ('a', 'b'),)
        ih1.append((3, 'c'))
        post = ih1.values_at_depth(1)
        self.assertEqual(post.tolist(), ['a', 'b', 'a', 'b', 'c'])

    def test_hierarchy_index_at_depth_a(self) -> None:
        ih1 = IndexHierarchyGO.from_product((1, 2), (100, 200), ('2020-01', '2020-03'))
        ih1.append((1, 300, '2020-01'))

        for depth in range(3):
            assert ih1.index_at_depth(depth) is ih1._indices[depth]
            assert ih1.index_at_depth([depth]) == (ih1._indices[depth],)

        assert ih1.index_at_depth([0, 1]) == ih1.index_at_depth([1, 0])[::-1]
        assert ih1.index_at_depth([2, 0]) == (ih1._indices[2], ih1._indices[0])
        assert ih1.index_at_depth(list(range(3))) == tuple(ih1._indices)

    def test_hierarchy_indexer_at_depth_a(self) -> None:
        ih1 = IndexHierarchyGO.from_product((1, 2), (100, 200), ('2020-01', '2020-03'))
        ih1.append((1, 300, '2020-01'))

        for depth in range(3):
            assert (ih1.indexer_at_depth(depth) == ih1._indexers[depth]).all()
            assert (ih1.indexer_at_depth([depth]) == ih1._indexers[[depth]]).all()

        assert (ih1.indexer_at_depth([0, 1]) == ih1.indexer_at_depth([1, 0])[::-1]).all().all()
        assert (ih1.indexer_at_depth([2, 0]) == [ih1._indexers[2], ih1._indexers[0]]).all().all()
        assert (ih1.indexer_at_depth(list(range(3))) == ih1._indexers).all().all()

    #---------------------------------------------------------------------------

    def test_hierarchy_head_a(self) -> None:

        ih1 = IndexHierarchy.from_product((1, 2), ('a', 'b'), (2, 5))

        self.assertEqual(ih1.head().values.tolist(),
            [[1, 'a', 2], [1, 'a', 5], [1, 'b', 2], [1, 'b', 5], [2, 'a', 2]]
            )

    def test_hierarchy_tail_a(self) -> None:

        ih1 = IndexHierarchy.from_product((1, 2), ('a', 'b'), (2, 5))

        self.assertEqual(ih1.tail().values.tolist(),
            [[1, 'b', 5], [2, 'a', 2], [2, 'a', 5], [2, 'b', 2], [2, 'b', 5]]
            )

    #---------------------------------------------------------------------------

    def test_hierarchy_via_str_a(self) -> None:

        ih1 = IndexHierarchy.from_product(('i', 'ii'), ('a', 'b'))
        ih2 = ih1.via_str.upper()

        self.assertEqual(ih2.tolist(),
                [['I', 'A'], ['I', 'B'], ['II', 'A'], ['II', 'B']]
                )

    def test_hierarchy_via_str_b(self) -> None:

        ih1 = IndexHierarchyGO.from_product(('i', 'ii'), ('a', 'b'))
        ih1.append(('iii', 'a'))
        ih2 = ih1.via_str.upper()

        self.assertEqual(ih2.tolist(),
                [['I', 'A'], ['I', 'B'], ['II', 'A'], ['II', 'B'], ['III', 'A']]
                )

    def test_hierarchy_via_dt_a(self) -> None:
        index_constructors = (IndexYearMonth, IndexDate)

        labels = (
            ('2020-01', '2019-01-01'),
            ('2020-01', '2019-02-01'),
            ('2019-02', '2019-01-01'),
            ('2019-02', '2019-02-01'),
        )

        ih1 = IndexHierarchy.from_labels(labels, index_constructors=index_constructors)
        ih2 = ih1.via_dt.month

        self.assertEqual(
                ih2.tolist(),
                [[1, 1], [1, 2], [2, 1], [2, 2]]
                )

    def test_hierarchy_via_dt_b(self) -> None:
        index_constructors = (IndexDate, IndexDate)

        labels = (
            ('2020-01-03', '2019-01-01'),
            ('2020-01-03', '2019-02-01'),
            ('2019-02-05', '2019-01-01'),
            ('2019-02-05', '2019-02-01'),
        )

        ih1 = IndexHierarchy.from_labels(labels, index_constructors=index_constructors)
        ih2 = ih1.via_dt.isoformat()

        self.assertEqual(
            ih2.dtype, np.dtype('<U10'),
            )
        self.assertEqual(
                ih2.tolist(),
                [['2020-01-03', '2019-01-01'], ['2020-01-03', '2019-02-01'], ['2019-02-05', '2019-01-01'], ['2019-02-05', '2019-02-01']]
                )

        ih3 = ih1.via_dt.strftime('%y|%m|%d')
        self.assertEqual(
            ih3.dtype, np.dtype('<U8'),
            )

        self.assertEqual(
            ih3.tolist(),
            [['20|01|03', '19|01|01'], ['20|01|03', '19|02|01'], ['19|02|05', '19|01|01'], ['19|02|05', '19|02|01']]
            )

    def test_hierarchy_via_dt_c(self) -> None:
        index_constructors = (IndexYearMonth, IndexDate)

        labels = (
            ('2020-01', '2019-01-01'),
            ('2020-01', '2019-02-01'),
            ('2019-02', '2019-01-01'),
            ('2019-02', '2019-02-01'),
        )

        ih1 = IndexHierarchyGO.from_labels(labels, index_constructors=index_constructors)
        ih1.append(('2021-01', '2019-01-01'))
        ih2 = ih1.via_dt.month

        self.assertEqual(
                ih2.tolist(),
                [[1, 1], [1, 2], [2, 1], [2, 2], [1, 1]]
                )

    def test_hierarchy_via_re_a(self) -> None:
        index_constructors = (IndexYearMonth, IndexDate)

        labels = (
            ('2020-01', '2019-01-01'),
            ('2020-01', '2019-02-01'),
            ('2019-02', '2019-01-01'),
            ('2019-02', '2019-02-01'),
        )
        ih1 = IndexHierarchy.from_labels(labels, index_constructors=index_constructors)

        a1 = ih1.via_re('19').search()
        self.assertEqual(a1.tolist(),
                [[False, True], [False, True], [True, True], [True, True]]
                )

        a2 = ih1.via_re('-').sub('*')
        self.assertEqual(a2.tolist(),
                [['2020*01', '2019*01*01'], ['2020*01', '2019*02*01'], ['2019*02', '2019*01*01'], ['2019*02', '2019*02*01']]
                )

    def test_hierarchy_via_re_b(self) -> None:
        index_constructors = (IndexYearMonth, IndexDate)

        labels = (
            ('2020-01', '2019-01-01'),
            ('2020-01', '2019-02-01'),
            ('2019-02', '2019-01-01'),
            ('2019-02', '2019-02-01'),
        )
        ih1 = IndexHierarchyGO.from_labels(labels, index_constructors=index_constructors)
        ih1.append(('2021-01', '2019-01-01'))

        a1 = ih1.via_re('19').search()
        self.assertEqual(a1.tolist(),
                [[False, True], [False, True], [True, True], [True, True], [False, True]]
                )

        a2 = ih1.via_re('-').sub('*')
        self.assertEqual(a2.tolist(),
                [['2020*01', '2019*01*01'], ['2020*01', '2019*02*01'], ['2019*02', '2019*01*01'], ['2019*02', '2019*02*01'], ['2021*01', '2019*01*01']]
                )

    def test_hierarchy_via_values_a(self) -> None:
        ih1 = IndexHierarchy.from_product((0, 1), (10, 20))
        ih2 = ih1.via_values.apply(lambda b: -b)
        self.assertEqual(ih2.values.tolist(),
                [[0, -10], [0, -20], [-1, -10], [-1, -20]]
                )


    def test_hierarchy_via_values_b(self) -> None:
        ih1 = IndexHierarchy.from_product((0, 1), (10, 20))
        post = np.sum(ih1.values, axis=1)
        self.assertEqual(post.tolist(),
                [10, 20, 11, 21]
                )

    def test_hierarchy_via_values_c(self) -> None:
        ih1 = IndexHierarchyGO.from_product((0, 1), (2, 3))
        ih1.append((5, 3))
        post = np.power(ih1.via_values, 2) # type: ignore
        self.assertEqual(post.values.tolist(),
                [[0, 4], [0, 9], [1, 4], [1, 9], [25, 9]]
                )

    #---------------------------------------------------------------------------

    def test_hierarchy_equals_a(self) -> None:

        ih1 = IndexHierarchy.from_product((1, 2), ('a', 'b'), (2, 5))
        ih2 = IndexHierarchy.from_product((1, 2), ('a', 'b'), (2, 5))
        ih3 = IndexHierarchy.from_product((1, 2), ('a', 'b'), (2, 4))
        ih4 = IndexHierarchy.from_product((1, 2), ('a', 'b'), (2, 4), name='foo')

        self.assertTrue(ih1.equals(ih1))
        self.assertTrue(ih1.equals(ih2))
        self.assertTrue(ih2.equals(ih1))

        self.assertFalse(ih1.equals(ih3))
        self.assertFalse(ih3.equals(ih1))

        self.assertFalse(ih3.equals(ih4, compare_name=True))
        self.assertTrue(ih3.equals(ih4, compare_name=False))

    def test_hierarchy_equals_b(self) -> None:

        ih1 = IndexHierarchy.from_product((1, 2), ('a', 'b'), Index((2, 5), dtype=np.int64))
        ih2 = IndexHierarchy.from_product((1, 2), ('a', 'b'), Index((2, 5), dtype=np.int32))

        self.assertFalse(ih1.equals(ih2, compare_dtype=True))
        self.assertTrue(ih1.equals(ih2, compare_dtype=False))

    def test_hierarchy_equals_c(self) -> None:

        idx = IndexDate.from_year_month_range('2020-01', '2020-02')

        ih1 = IndexHierarchy.from_product((1, 2), ('a', 'b'), idx)
        ih2 = IndexHierarchy.from_product((1, 2), ('a', 'b'),
                Index(idx.values.astype(object)))

        self.assertFalse(ih1.equals(ih2, compare_class=True))
        self.assertTrue(ih1.equals(ih2, compare_class=False))

    def test_hierarchy_equals_d(self) -> None:

        ih1 = IndexHierarchy.from_product((1, 2), ('a', 'b'), (2, 5))
        ih2 = IndexHierarchyGO.from_product((1, 2), ('a', 'b'), (2, 5))

        self.assertFalse(ih1.equals(ih2, compare_class=True))
        self.assertTrue(ih1.equals(ih2, compare_class=False))

    def test_hierarchy_equals_e(self) -> None:
        ih1 = IndexHierarchyGO.from_product((1, 2), ('a', 'b'))
        ih2 = ih1.copy()
        self.assertTrue(ih1.equals(ih2))
        ih1.append((3, 'c'))
        ih2.append((3, 'c'))
        self.assertTrue(ih1.equals(ih2))

    #---------------------------------------------------------------------------
    def test_hierarchy_fillna_a(self) -> None:

        ih1 = IndexHierarchy.from_product((1, 2), ('a', 'b'), (2, None))
        ih2 = ih1.fillna(20)
        self.assertEqual(ih2.values.tolist(),
                [[1, 'a', 2], [1, 'a', 20], [1, 'b', 2], [1, 'b', 20], [2, 'a', 2], [2, 'a', 20], [2, 'b', 2], [2, 'b', 20]]
                )

    def test_hierarchy_fillna_b(self) -> None:

        ih1 = IndexHierarchyGO.from_product((1, 2), ('a', 'b'), (2, np.nan))
        ih1.append((3, 'c', np.nan))
        ih2 = ih1.fillna('foo')

        self.assertEqual(ih2.values.tolist(),
                [[1, 'a', 2.0], [1, 'a', 'foo'], [1, 'b', 2.0], [1, 'b', 'foo'], [2, 'a', 2.0], [2, 'a', 'foo'], [2, 'b', 2.0], [2, 'b', 'foo'], [3, 'c', 'foo']]
                )

    #---------------------------------------------------------------------------
    def test_hierarchy_fillfalsy_a(self) -> None:

        ih1 = IndexHierarchy.from_product((1, 2), ('', 'b'), (2, ''))
        ih2 = ih1.fillfalsy('x')
        self.assertEqual(ih2.values.tolist(),
                [[1, 'x', 2], [1, 'x', 'x'], [1, 'b', 2], [1, 'b', 'x'], [2, 'x', 2], [2, 'x', 'x'], [2, 'b', 2], [2, 'b', 'x']]
                )

    def test_hierarchy_fillfalsy_b(self) -> None:

        ih1 = IndexHierarchyGO.from_product((1, 2), ('', 'b'))
        ih1.append((3, ''))
        ih2 = ih1.fillfalsy('x')
        self.assertEqual(ih2.values.tolist(),
                [[1, 'x'], [1, 'b'], [2, 'x'], [2, 'b'], [3, 'x']]
                )

    #---------------------------------------------------------------------------
    def test_hierarchy_dropna_a(self) -> None:

        ih1 = IndexHierarchy.from_product((1, 'a'), (None, 'b'))
        ih2: IndexHierarchy = ih1.dropna(condition=np.any)
        self.assertEqual(ih2.values.tolist(), [[1, 'b'], ['a', 'b']])

        ih3: IndexHierarchy = ih1.dropna(condition=np.all)
        self.assertIs(ih3, ih1)

    def test_hierarchy_dropna_b(self) -> None:

        ih1 = IndexHierarchy.from_labels(((1, 'a'), (None, np.nan), (None, 'b')))
        ih2: IndexHierarchy = ih1.dropna(condition=np.all)
        self.assertEqual(ih2.values.tolist(), [[1, 'a'], [None, 'b']])

    #---------------------------------------------------------------------------
    def test_hierarchy_dropfalsy_a(self) -> None:

        ih1 = IndexHierarchy.from_product((1, 'a'), ('', 'b'))
        ih2: IndexHierarchy = ih1.dropfalsy(condition=np.any)
        self.assertEqual(ih2.values.tolist(), [[1, 'b'], ['a', 'b']])

        ih2 = ih1.dropna(condition=np.all)
        self.assertIs(ih2, ih1)

    #---------------------------------------------------------------------------
    def test_hierarchy_from_names_a(self) -> None:

        ih1 = IndexHierarchy.from_names(('foo', 'bar'))
        self.assertEqual(ih1.name, ('foo', 'bar'))
        self.assertEqual(ih1.shape, (0, 2))

        ih2 = IndexHierarchyGO.from_names(('x', 'y', 'z'))
        self.assertEqual(ih2.name, ('x', 'y', 'z'))
        self.assertEqual(ih2.shape, (0, 3))

        ih2.append(('A', 10, False))
        self.assertEqual(ih2.values.tolist(),
                [['A', 10, False]])

    def test_hierarchy_from_names_b(self) -> None:
        with self.assertRaises(ErrorInitIndex):
            IndexHierarchy.from_names(())

        with self.assertRaises(ErrorInitIndex):
            IndexHierarchy.from_names([])

    #---------------------------------------------------------------------------

    def test_hierarchy_iter_label_a(self) -> None:

        idx = IndexHierarchy.from_product(('I', 'II'), ('A', 'B'), (1, 2))

        self.assertEqual(list(idx.iter_label(0)), ['I', 'I', 'I', 'I', 'II', 'II', 'II', 'II'])
        self.assertEqual(list(idx.iter_label(1)), ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'])
        self.assertEqual(list(idx.iter_label(2)), [1, 2, 1, 2, 1, 2, 1, 2])

        self.assertEqual(list(idx.iter_label(0)), ['I', 'I', 'I', 'I', 'II', 'II', 'II', 'II'])
        self.assertEqual(list(idx.iter_label(1)), ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'])
        self.assertEqual(list(idx.iter_label(2)), [1, 2, 1, 2, 1, 2, 1, 2])

        post = idx.iter_label(1).apply(lambda x: x.lower())
        self.assertEqual(post.tolist(), ['a', 'a', 'b', 'b', 'a', 'a', 'b', 'b'])

    def test_hierarchy_iter_label_b(self) -> None:

        idx = IndexHierarchy.from_product(('I', 'II'), ('A', 'B'), (1, 2))
        self.assertEqual(list(idx.iter_label([0, 2])),
                [('I', 1), ('I', 2), ('I', 1), ('I', 2), ('II', 1), ('II', 2), ('II', 1), ('II', 2)])

        self.assertEqual(list(idx.iter_label([0, 2])),
                [('I', 1), ('I', 2), ('I', 1), ('I', 2), ('II', 1), ('II', 2), ('II', 1), ('II', 2)])

    def test_hierarchy_iter_label_c(self) -> None:

        idx = IndexHierarchy.from_product(('I', 'II'), ('A', 'B'), (1, 2))
        post = list(idx.iter_label())
        self.assertEqual(post,
                [('I', 'A', 1), ('I', 'A', 2), ('I', 'B', 1), ('I', 'B', 2), ('II', 'A', 1), ('II', 'A', 2), ('II', 'B', 1), ('II', 'B', 2)]
                )
        self.assertEqual(idx.iter_label().apply(lambda x: x[:2]).tolist(),
                [('I', 'A'), ('I', 'A'), ('I', 'B'), ('I', 'B'), ('II', 'A'), ('II', 'A'), ('II', 'B'), ('II', 'B')]
                )

    def test_hierarchy_iter_label_d(self) -> None:
        idx = IndexHierarchy.from_product(('A', 'B'), (1, 2))
        self.assertEqual(list(idx._iter_label_items()),
                [(0, ('A', 1)), (1, ('A', 2)), (2, ('B', 1)), (3, ('B', 2))]
                )

    #---------------------------------------------------------------------------
    def test_hierarchy_sample_a(self) -> None:
        idx1 = Index(('A', 'B'))
        idx2 = IndexDate.from_date_range('2019-01-05', '2019-01-08')
        idx3 = Index((1, 2))
        hidx = IndexHierarchy.from_product(idx1, idx2, idx3)

        self.assertEqual(hidx.sample(3, seed=4).values.tolist(),
                [['A', datetime.date(2019, 1, 5), 1], ['A', datetime.date(2019, 1, 8), 1], ['B', datetime.date(2019, 1, 7), 1]])

        self.assertEqual(hidx.sample(3, seed=4).index_types.values.tolist(),
                [Index, IndexDate, Index])

    #---------------------------------------------------------------------------
    def test_hierarchy_iloc_searchsorted_a(self) -> None:

        idx1 = Index(('A', 'B'))
        idx2 = Index((1, 2, 3))
        hidx = IndexHierarchy.from_product(idx1, idx2)

        self.assertEqual(hidx.iloc_searchsorted(('B', 1)).tolist(), 3)
        self.assertEqual(hidx.iloc_searchsorted([('A', 1), ('B', 2)]).tolist(), [0, 4])

    def test_hierarchy_iloc_searchsorted_b(self) -> None:
        ih1 = IndexHierarchy.from_product((1, 2), ('a', 'b'), (2, 5))
        with self.assertRaises(NotImplementedError):
            ih1.iloc_searchsorted(3)

    #---------------------------------------------------------------------------
    def test_hierarchy_loc_searchsorted_a(self) -> None:

        idx1 = Index(('A', 'B'))
        idx2 = Index((1, 2, 3))
        hidx = IndexHierarchy.from_product(idx1, idx2)

        self.assertEqual(hidx.loc_searchsorted(('B', 1)), ('B', 1))
        self.assertEqual(hidx.loc_searchsorted([('A', 1), ('B', 2)]).tolist(), # type: ignore
                [('A', 1), ('B', 2)])

    def test_hierarchy_loc_searchsorted_b(self) -> None:

        idx1 = Index(('A', 'B'))
        idx2 = Index((1, 2, 3))
        hidx = IndexHierarchy.from_product(idx1, idx2)

        self.assertEqual(hidx.loc_searchsorted(('B', 3),
                side_left=False,
                fill_value=None),
                None)
        self.assertEqual(hidx.loc_searchsorted([('A', 1), ('B', 3)], # type: ignore
                side_left=False,
                fill_value=None).tolist(),
                [('A', 2), None])

    def test_hierarchy_loc_searchsorted_c(self) -> None:
        idx1 = Index(('A', 'B'))
        idx2 = IndexDate.from_date_range('2019-01-05', '2019-01-08')
        idx3 = Index((1, 2))
        hidx = IndexHierarchy.from_product(idx1, idx2, idx3)

        self.assertEqual(hidx.loc_searchsorted(('B', '2019-01-07', 2)),
                ('B', np.datetime64('2019-01-07'), 2),
                )

        self.assertEqual(
                hidx.loc_searchsorted( # type: ignore
                [('B', '2019-01-07', 2), ('B', '2019-01-08', 1)]).tolist(),
                [('B', np.datetime64('2019-01-07'), 2),
                ('B', np.datetime64('2019-01-08'), 1)])

    #---------------------------------------------------------------------------
    def test_hierarchy_unique_a1(self) -> None:
        ih1 = IndexHierarchy.from_product((1, 2), ('a', 'b'), (2, 5))

        self.assertEqual(ih1.unique(0).tolist(), [1, 2])
        self.assertEqual(ih1.unique(1).tolist(), ['a', 'b'])
        self.assertEqual(ih1.unique(2).tolist(), [2, 5])

    def test_hierarchy_unique_a2(self) -> None:
        ih1 = IndexHierarchy.from_product((1, 2), ('a', 'b'), (2, 5))
        self.assertEqual(ih1.unique([0, 2]).tolist(),
                [(1, 2), (1, 5), (2, 2), (2, 5)])

    def test_hierarchy_unique_a3(self) -> None:
        ih1 = IndexHierarchy.from_product((1, 2), ('a', 'b'), (2, 5))
        self.assertEqual(ih1.unique([1, 2]).tolist(),
                [('a', 2), ('a', 5), ('b', 2), ('b', 5)])


    def test_hierarchy_unique_b(self) -> None:
        ih1 = IndexHierarchy.from_product((1, 2), ('a', 'b'), (2, 5))
        self.assertEqual(ih1.unique([1]).tolist(), ['a', 'b'])

    def test_hierarchy_unique_c(self) -> None:
        ih1 = IndexHierarchy.from_product((1, 2), ('a', 'b'), (2, 5))

        with self.assertRaises(RuntimeError):
            ih1.unique([0, 2], order_by_occurrence=True)

    #---------------------------------------------------------------------------

    def test_hierarchy_union_a(self) -> None:

        ih1 = IndexHierarchy.from_labels(((1, '2020-01-01'), (1, '2020-01-02'), (1, '2020-01-03')),
                index_constructors=(Index, IndexDate))

        ih2 = IndexHierarchy.from_labels(((1, '2020-01-01'), (1, '2020-01-02'), (1, '2020-01-05')),
                index_constructors=(Index, IndexDate))

        ih3 = ih1.union(ih2)
        self.assertEqual(ih3.index_types.to_pairs(),
            ((0, Index), (1, IndexDate)))

    #---------------------------------------------------------------------------

    def test_build_indexers_from_product_a(self) -> None:
        actual = build_indexers_from_product([3, 3])
        expected = np.array([
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
        ])
        self.assertTrue(np.array_equal(actual, expected))

    def test_build_indexers_from_product_b(self) -> None:
        actual = build_indexers_from_product([3, 4, 2])
        expected = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        ])
        self.assertTrue(np.array_equal(actual, expected))

    #---------------------------------------------------------------------------

    def test_hierarchy_ndim(self) -> None:
        ih = IndexHierarchy.from_labels(((1, 2), (2, 3)))
        self.assertEqual(ih.ndim, 2)

        ih = IndexHierarchy.from_labels(((1, 2, 3, 4), (2, 3, 4, 5)))
        self.assertEqual(ih.ndim, 2)

    #---------------------------------------------------------------------------

    def test_get_unique_labels_in_occurence_order(self) -> None:
        labels = [
            (1, 'A'),
            (3, 'B'),
            (2, 'C'),
            (0, 'F'),
            (4, 'D'),
            (5, 'E'),
        ]
        ih = IndexHierarchy.from_labels(labels)

        depth0 = ih.unique(depth_level=0, order_by_occurrence=True).tolist()
        depth1 = ih.unique(depth_level=1, order_by_occurrence=True).tolist()

        self.assertListEqual([1, 3, 2, 0, 4, 5], depth0)
        self.assertListEqual(list("ABCFDE"), depth1)

    #---------------------------------------------------------------------------

    def test_hierarchy_update_array_cache_edge_cases(self) -> None:
        ihgo = IndexHierarchyGO.from_product(range(5), range(5))

        original = copy.deepcopy(ihgo)
        ihgo.append((5, 5))
        post1 = IndexHierarchyGO(ihgo)
        self.assertIn((5, 5), set(tuple(post1.difference(original))))

        original = copy.deepcopy(ihgo)
        ihgo.append((5, 6))
        post2 = ihgo.copy()
        self.assertIn((5, 6), set(tuple(post2.difference(original))))

        original = copy.deepcopy(ihgo)
        ihgo.append((5, 7))
        post3 = ihgo._drop_iloc(-2)
        self.assertNotIn((5, 6), set(tuple(post3.difference(original))))
        self.assertIn((5, 7), set(tuple(post3.difference(original))))

        original = copy.deepcopy(ihgo)
        ihgo.append((5, 8))
        post4 = ihgo.relabel_at_depth(lambda x: x, depth_level=0)
        self.assertIn((5, 8), set(tuple(post4.difference(original))))

        original = copy.deepcopy(ihgo)
        ihgo.append((5, 9))
        post5 = ihgo.rehierarch((1, 0))
        self.assertIn((9, 5), set(tuple(post5.difference(original))))
        self.assertEqual(post5.shape, ihgo.shape)
        self.assertTrue(all(a.equals(b) for (a, b) in zip(post5._indices, ihgo._indices[::-1])))

        ihgo.append((5, 10))
        post6 = ihgo[-1]
        self.assertEqual(post6, (5, 10))

        ihgo.append((5, 11))
        post7 = ihgo * 15
        self.assertTrue(((post7 / 15).astype(int) == ihgo.values).all().all())

        # original = copy.deepcopy(ihgo)
        # post8 = ihgo.append((5, 12))
        # self.assertIn((5, 12), set(tuple(post8.difference(original))))

        original = copy.deepcopy(ihgo)
        ihgo2 = ihgo.copy()
        ihgo.append((5, 13))
        ihgo2.append((5, 13))
        post9 = ihgo * ihgo2
        self.assertEqual((5**2, 13**2), tuple(post9[-1]))
        self.assertEqual(post9.shape, ihgo.shape)

        ihgo.append((5, 14))
        post10 = list(reversed(ihgo))
        self.assertEqual((5, 14), post10[0])
        self.assertEqual(len(ihgo), len(post10))

        original = copy.deepcopy(ihgo)
        ihgo.append((1, 99))
        post11 = ihgo.sort()
        self.assertEqual((1, 99), ihgo.iloc[-1])
        self.assertEqual((5, 14), post11.iloc[-1])
        self.assertIn((1, 99), set(tuple(post11.difference(original))))

        original = copy.deepcopy(ihgo)
        ihgo.append((5, 15))
        post12 = ihgo.roll(1)
        self.assertTrue((post12.roll(-1) == ihgo.values).all().all())
        self.assertIn((5, 15), set(tuple(post12.difference(original))))

        ihgo.append((5, 16))
        post13 = ihgo._sample_and_key(seed=97)
        self.assertEqual(((5, 16),), tuple(post13[0]))
        self.assertEqual(len(ihgo) - 1, post13[1].tolist()[0])

        original = copy.deepcopy(ihgo)
        ihgo.append((5, 17))
        post14 = ihgo.to_frame()
        ihgo3 = post14.set_index_hierarchy(post14.columns).index
        self.assertIn((5, 17), set(tuple(ihgo3.difference(original))))

        ihgo.append((5, 18))
        post15 = ihgo.to_pandas()
        self.assertEqual(len(ihgo), len(post15))
        self.assertEqual(ihgo.depth, post15.nlevels)
        self.assertIn((5, 18), post15)

        ihgo.append((5, 19))
        post16 = ihgo.to_tree()
        self.assertEqual(len(post16), len(ihgo._indices[0]))
        self.assertEqual(len(ihgo), sum(map(len, post16.values())))
        self.assertIn(19, post16[5])

        ihgo = ihgo.relabel_at_depth(depth_level=1, mapper=ihgo.values_at_depth(1) + ihgo.positions)

        ihgo.append((5, 300))
        post17 = ihgo.level_drop(1)
        self.assertEqual(len(post17), len(ihgo))
        self.assertIn(300, set(post17))

        ihgo.append((5, 301))
        post18 = ihgo.astype(str)
        self.assertEqual(len(post18), len(ihgo))
        self.assertIn(("5", "301"), set(post18))

        original = copy.deepcopy(ihgo)
        ihgo.append((5, 302))
        post19 = str(ihgo)
        self.assertNotEqual(post19, str(original))
        self.assertEqual(post19, str(ihgo))

    #---------------------------------------------------------------------------

    def test_hierarchy_iloc_a(self) -> None:

        f = ff.parse('f(Fg)|v(int,bool,str)|i((IY,ID),(dtY,dtD))|c(Isg,dts)|s(6,2)')

        post = f.loc[HLoc[f.index.iloc[0]]]
        self.assertEqual(post.to_pairs(),
            ((np.datetime64('1970-01-01T09:38:35'), -88017), (np.datetime64('1970-01-01T01:00:48'), False)))

        ih = IndexHierarchy.from_labels(
            [("1990", "2000-03-14")],
            index_constructors=[IndexYear, IndexDate]
        )

        self.assertEqual(ih.loc[ih.iloc[0]],
            (np.datetime64('1990'), np.datetime64('2000-03-14')))

    #---------------------------------------------------------------------------
    def test_hierarchy_max_a(self) -> None:
        ih = IndexHierarchy.from_labels([[1, 2], [20, 1], [3, 4]])
        self.assertEqual(ih.max().tolist(), [20, 1])
        self.assertEqual(ih.min().tolist(), [1, 2])
        self.assertEqual(ih.loc_to_iloc(ih.max()), 1)
        self.assertEqual(ih.loc_to_iloc(ih.min()), 0)

        with self.assertRaises(NotImplementedError):
            _ = ih.max(axis=1)

    def test_hierarchy_max_b(self) -> None:
        ih = IndexHierarchy.from_labels([[3, 4], [20, 1], [2, 1], [22, None]])
        self.assertEqual(ih.max().tolist(), [20, 1])
        self.assertEqual(ih.min().tolist(), [2, 1])
        self.assertEqual(ih.loc_to_iloc(ih.max()), 1)
        self.assertEqual(ih.loc_to_iloc(ih.min()), 2)

        with self.assertRaises(TypeError):
            _ = ih.max(skipna=False)


    def test_hierarchy_max_c(self) -> None:
        ih = IndexHierarchy.from_labels([[1, 2], [20, 1], [3, 4]])
        with self.assertRaises(NotImplementedError):
            _ = ih.mean()

    def test_hierarchy_max_d(self) -> None:
        ih = IndexHierarchyGO.from_labels([[1, 2], [20, 1], [3, 4]])
        ih.append([-1, -4])
        post = ih.min()
        self.assertEqual(post.tolist(), [-1, -4])

    # def test_hierarchy_cumsum_a(self) -> None:
    #     ih = IndexHierarchyGO.from_labels([[1, 2], [20, 1], [3, 4]])
    #     ih.append([-1, np.nan])

    #     post = ih.cumsum()
    #     self.assertEqual(post.tolist(),
    #         [[1.0, 2.0], [21.0, 3.0], [24.0, 7.0], [23.0, 7.0]]
    #         )
    #     self.assertEqual(ih.cumsum(skipna=False).astype(str).tolist(),
    #         [['1.0', '2.0'], ['21.0', '3.0'], ['24.0', '7.0'], ['23.0', 'nan']]
    #         )

    #---------------------------------------------------------------------------
    def test_hierarchy_concat_a(self) -> None:

        f = ff.parse("f(Fg)|v(int,bool,str)|i((IY,ID),(dtY,dtD))|c(Isg,dts)|s(4,2)")
        f1 = f.iloc[:2]
        f2 = f.iloc[2:]

        f3 = Frame.from_concat((f1, f2)) # RuntimeError
        self.assertTrue(f.index.equals(f3.index, compare_dtype=True, compare_class=True))

    #---------------------------------------------------------------------------

    def test_hierarchy_to_signature_bytes_a(self) -> None:

        hidx1 = IndexHierarchy.from_product(Index((1, 2), dtype=np.int64), IndexDate.from_date_range('2019-01-05', '2019-01-08'), name='')

        post = hidx1._to_signature_bytes()

        self.assertEqual(sha256(post).hexdigest(), 'f24f3db67466e74241c077a00b3211f5895253cd51995254600b1d68a8af5696')

    def test_hierarchy_to_signature_bytes_b(self) -> None:

        hidx1 = IndexHierarchy.from_product(Index((1, 2)), IndexDate.from_date_range('2019-01-05', '2019-01-08'), name='')

        hidx2 = IndexHierarchy.from_product(Index((1, 2)), IndexDate.from_date_range('2019-01-06', '2019-01-09'), name='')

        post1 = hidx1._to_signature_bytes()
        post2 = hidx2._to_signature_bytes()

        self.assertNotEqual(sha256(post1).hexdigest(), sha256(post2).hexdigest())

    def test_hierarchy_to_signature_bytes_c(self) -> None:

        hidx1 = IndexHierarchy.from_product(Index((1, 2)), IndexDate.from_date_range('2019-01-05', '2019-01-08'), name='')

        hidx2 = IndexHierarchy.from_product(Index((1, 2)), IndexDate.from_date_range('2019-01-05', '2019-01-08'), name='foo')

        post1 = hidx1._to_signature_bytes(include_name=False)
        post2 = hidx2._to_signature_bytes(include_name=False)

        self.assertEqual(sha256(post1).hexdigest(), sha256(post2).hexdigest())

    #---------------------------------------------------------------------------

    def test_hierarchy_via_hashlib_a(self) -> None:
        hidx1 = IndexHierarchy.from_product(Index((1, 2), dtype=np.int64), IndexDate.from_date_range('2019-01-05', '2019-01-08'), name='')
        hd = hidx1.via_hashlib().sha256().hexdigest()
        self.assertEqual(hd, 'f24f3db67466e74241c077a00b3211f5895253cd51995254600b1d68a8af5696')

    def test_hierarchy_via_hashlib_b(self) -> None:
        hidx1 = IndexHierarchy.from_product(Index((1, 2), dtype=object), IndexDate.from_date_range('2019-01-05', '2019-01-08'), name='')
        with self.assertRaises(TypeError):
            hd = hidx1.via_hashlib().sha256().hexdigest()

    #---------------------------------------------------------------------------
    def test_hierarchy_hloc_a(self) -> None:
        labels = [
            ('a', 1, 10),
            ('a', 2, 10),
            ('b', 1, 10),
            ('b', 2, 20),
            ('b', 3, 10),
            ('b', 4, 20),
            ('b', 5, 10),
            ('c', 1, 10),
            ]
        ih1 = IndexHierarchy.from_labels(labels)
        ih2 = ih1.loc[HLoc['b', :, 20:]]
        self.assertEqual(len(ih2), 4)
        self.assertEqual(ih2.values.tolist(),
            [['b', 2, 20], ['b', 3, 10], ['b', 4, 20], ['b', 5, 10]]
            )

    def test_hierarchy_hloc_b(self) -> None:
        labels = [
            ('a', 1, 70),
            ('a', 2, 50),
            ('a', 3, 30),
            ('a', 4, 10),
            ('b', 1, 70),
            ('b', 2, 50),
            ('b', 3, 30),
            ('b', 4, 10),
            ]
        ih1 = IndexHierarchy.from_labels(labels)
        ih2 = ih1.loc[HLoc['b', :, 30:]]
        self.assertEqual(ih2.values.tolist(),
            [['b', 3, 30], ['b', 4, 10]]
            )
        # same result
        ih3 = ih1.loc[HLoc['b']].loc[HLoc[:, :, 30:]]
        self.assertEqual(ih3.values.tolist(),
            [['b', 3, 30], ['b', 4, 10]]
            )

    #---------------------------------------------------------------------------
    def test_hierarchy_isna_a(self) -> None:
        labels = [
            ('a', 0, 10),
            ('a', np.nan, 8),
            ]
        ih1 = IndexHierarchy.from_labels(labels)
        self.assertEqual(
                ih1.isna().tolist(),
                [[False, False, False], [False, True, False]]
                )

    def test_hierarchy_notna_a(self) -> None:
        labels = [
            ('a', 0, 10),
            ('a', np.nan, 8),
            ]
        ih1 = IndexHierarchy.from_labels(labels)
        self.assertEqual(
                ih1.notna().tolist(),
                [[True, True, True], [True, False, True]]
                )

    def test_hierarchy_isfalsy_a(self) -> None:
        labels = [
            ('a', 0, 10),
            ('', np.nan, 8),
            ]
        ih1 = IndexHierarchy.from_labels(labels)
        self.assertEqual(
                ih1.isfalsy().tolist(),
                [[False, True, False], [True, True, False]]
                )

    def test_hierarchy_notfalsy_a(self) -> None:
        labels = [
            ('a', 0, 10),
            ('', np.nan, 8),
            ]
        ih1 = IndexHierarchy.from_labels(labels)
        self.assertEqual(
                ih1.notfalsy().tolist(),
                [[True, False, True], [False, False, True]]
                )

if __name__ == '__main__':
    unittest.main()
