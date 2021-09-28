import typing as tp
import unittest

import numpy as np
import frame_fixtures as ff

from static_frame.test.test_case import TestCase
from static_frame.core.bus import Bus
from static_frame.core.frame import Frame
from static_frame.core.index import Index
from static_frame.core.index_level import TreeNodeT
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.exception import ErrorInitBus
from static_frame.core.exception import ErrorInitQuilt
from static_frame.core.exception import AxisInvalid
from static_frame.core.exception import ErrorInitYarn

from static_frame.core.axis_map import bus_to_hierarchy
from static_frame.core.axis_map import buses_to_hierarchy


class TestUnit(TestCase):

    def test_axis_map_b(self) -> None:

        f1 = ff.parse('s(4,4)|v(int,float)').rename('f1')
        f2 = ff.parse('s(4,4)|v(str)').rename('f2')
        f3 = ff.parse('s(4,4)|v(bool)').rename('f3')

        b1 = Bus.from_frames((f1, f2, f3))

        with self.assertRaises(AxisInvalid):
            bus_to_hierarchy(b1, deepcopy_from_bus=False, axis=3, init_exception_cls=ErrorInitQuilt)

    def test_index_map_a(self) -> None:

        f1 = ff.parse('s(4,4)|v(int,float)').rename('f1')
        f2 = ff.parse('s(4,4)|v(str)').rename('f2')
        f3 = ff.parse('s(4,4)|v(bool)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3), name='a')

        f4 = ff.parse('s(4,4)|v(int,float)').rename('f4')
        f5 = ff.parse('s(4,4)|v(str)').rename('f5')
        b2 = Bus.from_frames((f4, f5), name='b')

        post = buses_to_hierarchy((b1, b2), (b1.name, b2.name), deepcopy_from_bus=False, init_exception_cls=ErrorInitYarn)

        self.assertEqual(post.values.tolist(),
                [['a', 'f1'], ['a', 'f2'], ['a', 'f3'], ['b', 'f4'], ['b', 'f5']])

    def compare_trees(self, tree1: TreeNodeT, tree2: TreeNodeT) -> None:
        self.assertSequenceEqual(tree1.keys(), tree2.keys())

        for k, v1 in tree1.items():
            v2 = tree2[k]
            if isinstance(v1, Index):
                self.assertTrue(v1.equals(v2))
            else:
                self.compare_trees(v1, v2)

    def test_bus_to_hierarchy_a(self) -> None:
        f1 = ff.parse('s(4,4)|v(int,float)|c(I, str)').rename('f1')
        f2 = ff.parse('s(4,4)|v(str)|c(I, str)').rename('f2')
        f3 = ff.parse('s(4,4)|v(bool)|c(I, str)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3), name='a')

        indices = Index((0, 1, 2, 3))
        columns = Index(("zZbu", "ztsv", "zUvW", "zkuW"))

        for _, frame in b1.items():
            self.assertTrue(indices.equals(frame.index))
            self.assertTrue(columns.equals(frame.columns))


        def test_assertions(axis: int, flag: bool) -> None:
            hierarchy, opposite = bus_to_hierarchy(b1, axis=axis, deepcopy_from_bus=flag, init_exception_cls=ErrorInitBus)

            if axis == 0:
                expected_tree: tp.Dict[str, Index] = {
                    'f1': indices, 'f2': indices, 'f3': indices
                }
                expected_index = columns
            else:
                expected_index = indices
                expected_tree = {'f1': columns, 'f2': columns, 'f3': columns}

            self.compare_trees(hierarchy.to_tree(), expected_tree)
            self.assertTrue(expected_index.equals(opposite))

        for axis in (0, 1):
            for flag in (True, False):
                test_assertions(axis, flag)

    def test_bus_to_hierarchy_b(self) -> None:

        class CustomError(Exception):
            pass

        tree1 = dict(a_I=Index((1,2,3)), a_II=Index((1,2,3)))
        tree2 = dict(b_I=Index((1,2,3)), b_II=Index((1,2,3)))
        tree3 = dict(c_I=Index((1,2,3)), c_II=Index((1,2,3)))
        index1 = IndexHierarchy.from_tree(tree1)
        index2 = IndexHierarchy.from_tree(tree2)
        index3 = IndexHierarchy.from_tree(tree3)
        values = np.arange(36).reshape(6,6)

        # Align all the frames on columns!
        f1 = Frame(values, index=index1, columns=index1, name="f1")
        f2 = Frame(values, index=index2, columns=index1, name="f2")
        f3 = Frame(values, index=index3, columns=index1, name="f3")
        b1 = Bus.from_frames((f1, f2, f3))

        def test_assertions(hierarchy: IndexHierarchy, opposite: Index) -> None:
            expected_tree = dict(f1=tree1, f2=tree2, f3=tree3)
            self.compare_trees(hierarchy.to_tree(), expected_tree)
            self.assertTrue(index1.equals(opposite))

        test_assertions(*bus_to_hierarchy(b1, axis=0, deepcopy_from_bus=False, init_exception_cls=CustomError))
        test_assertions(*bus_to_hierarchy(b1, axis=0, deepcopy_from_bus=True, init_exception_cls=CustomError))

        # Cannot do this since the frames do not share the same index
        with self.assertRaises(CustomError):
            bus_to_hierarchy(b1, axis=1, deepcopy_from_bus=False, init_exception_cls=CustomError)

        with self.assertRaises(CustomError):
            bus_to_hierarchy(b1, axis=1, deepcopy_from_bus=True, init_exception_cls=CustomError)

        # Align all the frames on index!
        f1 = Frame(values, index=index1, columns=index1, name="f1")
        f2 = Frame(values, index=index1, columns=index2, name="f2")
        f3 = Frame(values, index=index1, columns=index3, name="f3")
        b1 = Bus.from_frames((f1, f2, f3))

        test_assertions(*bus_to_hierarchy(b1, axis=1, deepcopy_from_bus=False, init_exception_cls=CustomError))
        test_assertions(*bus_to_hierarchy(b1, axis=1, deepcopy_from_bus=True, init_exception_cls=CustomError))

        # Cannot do this since the frames do not share the same columns
        with self.assertRaises(CustomError):
            bus_to_hierarchy(b1, axis=0, deepcopy_from_bus=False, init_exception_cls=CustomError)

        with self.assertRaises(CustomError):
            bus_to_hierarchy(b1, axis=0, deepcopy_from_bus=True, init_exception_cls=CustomError)

    def test_buses_to_hierarchy_a(self) -> None:
        f1 = ff.parse('s(4,4)|v(int,float)').rename('f1')
        f2 = ff.parse('s(4,4)|v(str)').rename('f2')
        f3 = ff.parse('s(4,4)|v(bool)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3), name='a')

        f4 = ff.parse('s(4,4)|v(int,float)').rename('f4')
        f5 = ff.parse('s(4,4)|v(str)').rename('f5')
        b2 = Bus.from_frames((f4, f5), name='b')

        post = buses_to_hierarchy((b1, b2), (b1.name, b2.name), deepcopy_from_bus=False, init_exception_cls=ErrorInitYarn)

        self.assertEqual(post.values.tolist(),
                [['a', 'f1'], ['a', 'f2'], ['a', 'f3'], ['b', 'f4'], ['b', 'f5']])

    def test_buses_to_hierarchy_b(self) -> None:
        f1 = ff.parse('s(4,4)|v(int,float)').rename('f1')
        f2 = ff.parse('s(4,4)|v(str)').rename('f2')
        f3 = ff.parse('s(4,4)|v(bool)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3), name='foo')

        f4 = ff.parse('s(4,4)|v(int,float)').rename('f4')
        f5 = ff.parse('s(4,4)|v(str)').rename('f5')
        b2 = Bus.from_frames((f4, f5), name='foo')

        with self.assertRaises(ErrorInitYarn):
            _ = buses_to_hierarchy((b1, b2),
                    (b1.name, b2.name),
                    deepcopy_from_bus=False,
                    init_exception_cls=ErrorInitYarn)


if __name__ == '__main__':
    unittest.main()
