
import unittest
import copy

import numpy as np

from static_frame import mloc
from static_frame.core.array_go import ArrayGO
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):



    def test_array_init_a(self) -> None:
        with self.assertRaises(NotImplementedError):
            _ = ArrayGO(np.array((3, 4, 5)))

    def test_array_append_a(self) -> None:

        ag1 = ArrayGO(('a', 'b', 'c', 'd'))

        self.assertEqual([x for x in ag1],
            ['a', 'b', 'c', 'd'])

        self.assertEqual(ag1.values.tolist(),
                ['a', 'b', 'c', 'd'])


        ag1.append('e')
        ag1.extend(('f', 'g'))

        self.assertEqual(ag1.values.tolist(),
            ['a', 'b', 'c', 'd', 'e', 'f', 'g'])

        self.assertEqual([x for x in ag1],
            ['a', 'b', 'c', 'd', 'e', 'f', 'g'])


    def test_array_append_b(self) -> None:

        ag1 = ArrayGO(np.array(('a', 'b', 'c', 'd'), object))

        self.assertEqual([x for x in ag1],
            ['a', 'b', 'c', 'd'])

        self.assertEqual(ag1.values.tolist(),
                ['a', 'b', 'c', 'd'])


        ag1.append('e')
        ag1.extend(('f', 'g'))

        self.assertEqual(ag1.values.tolist(),
            ['a', 'b', 'c', 'd', 'e', 'f', 'g'])

        self.assertEqual([x for x in ag1],
            ['a', 'b', 'c', 'd', 'e', 'f', 'g'])


    def test_array_getitem_a(self) -> None:

        a = np.array(('a', 'b', 'c', 'd'), object)
        a.flags.writeable = False

        ag1 = ArrayGO(a)
        # insure no copy for immutable
        self.assertEqual(mloc(ag1.values), mloc(a))

        ag1.append('b')

        post = ag1[ag1.values == 'b']

        self.assertEqual(post.tolist(), ['b', 'b'])
        self.assertEqual(ag1[[2,1,1,1]].tolist(),
                ['c', 'b', 'b', 'b'])

    def test_array_copy_a(self) -> None:

        ag1 = ArrayGO(np.array(('a', 'b', 'c', 'd'), dtype=object))
        ag1.append('e')

        ag2 = ag1.copy()
        ag1.extend(('f', 'g'))

        self.assertEqual(ag1.values.tolist(),
                ['a', 'b', 'c', 'd', 'e', 'f', 'g'])

        self.assertEqual(ag2.values.tolist(),
                ['a', 'b', 'c', 'd', 'e'])


    def test_array_deepcopy_a(self) -> None:
        ag1 = ArrayGO(np.array(('a', 'b', 'c', 'd'), dtype=object))
        ag1.append('e')
        ag1.extend(('f', 'g'))
        ag2 = copy.deepcopy(ag1)
        self.assertEqual(ag1._array.tolist(), ag2._array.tolist()) #type: ignore

    def test_array_len_a(self) -> None:

        ag1 = ArrayGO(np.array(('a', 'b', 'c', 'd'), object))
        ag1.append('e')

        self.assertEqual(len(ag1), 5)


if __name__ == '__main__':
    unittest.main()




