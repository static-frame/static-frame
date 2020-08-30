import unittest

import numpy as np


from static_frame.core.container import _all
from static_frame.core.container import _any
from static_frame.core.container import _nanall
from static_frame.core.container import _nanany
from static_frame.core.container import _ufunc_logical_skipna
from static_frame.core.container import ContainerOperand
from static_frame.test.test_case import TestCase
from static_frame.test.test_case import UFUNC_AXIS_SKIPNA
from static_frame.test.test_case import UFUNC_SHAPE_SKIPNA


class TestUnit(TestCase):

    def test_container_attrs(self) -> None:

        for attr in UFUNC_AXIS_SKIPNA.keys() | UFUNC_SHAPE_SKIPNA.keys():
            c = ContainerOperand
            self.assertTrue(hasattr(c, attr))

        with self.assertRaises(NotImplementedError):
            c().display()

    def test_container_any_a(self) -> None:

        self.assertTrue(_nanany(np.array([np.nan, False, True])))
        self.assertTrue(_nanany(np.array(['foo', '', np.nan], dtype=object)))
        self.assertTrue(_nanany(np.array(['', None, 1], dtype=object)))

        self.assertFalse(_nanany(np.array([False, np.nan], dtype=object)))
        self.assertFalse(_nanany(np.array([False, None])))
        self.assertFalse(_nanany(np.array(['', np.nan], dtype=object)))
        self.assertFalse(_nanany(np.array(['', None], dtype=object)))


    def test_container_any_b(self) -> None:

        self.assertTrue(_any(np.array([False, True])))
        self.assertTrue(_any(np.array([False, True])))
        self.assertTrue(_any(np.array([False, True], dtype=object)))
        self.assertTrue(_any(np.array(['foo', ''])))
        self.assertTrue(_any(np.array(['foo', ''], dtype=object)))


        self.assertFalse(_any(np.array([False, False])))
        self.assertFalse(_any(np.array([False, False], dtype=object)))
        self.assertFalse(_any(np.array(['', ''])))
        self.assertFalse(_any(np.array(['', ''], dtype=object)))


        # self.assertTrue(
        #         np.isnan(_any(np.array([False, np.nan], dtype=object)))
        #         )
        # self.assertTrue(
        #         np.isnan(_any(np.array([False, None], dtype=object)))
        #         )




    def test_container_all_a(self) -> None:

        self.assertTrue(_nanall(np.array([np.nan, True, True], dtype=object)))
        self.assertTrue(_nanall(np.array([np.nan, True], dtype=object)))
        self.assertTrue(_nanall(np.array([np.nan, 1.0])))


        self.assertFalse(_nanall(np.array([None, False, False], dtype=object)))
        self.assertFalse(_nanall(np.array([np.nan, False, False], dtype=object)))
        self.assertFalse(_nanall(np.array([None, False, False], dtype=object)))


    def test_container_all_b(self) -> None:
        self.assertTrue(_all(np.array([True, True])))
        self.assertTrue(_all(np.array([1, 2])))


        self.assertFalse(_all(np.array([1, 0])))
        self.assertFalse(_all(np.array([False, False])))

        with self.assertRaises(TypeError):
            np.isnan(_all(np.array([False, np.nan], dtype=object)))
        with self.assertRaises(TypeError):
            np.isnan(_all(np.array([False, None], dtype=object)))



    def test_ufunc_logical_skipna_a(self) -> None:

        # empty arrays
        a1 = np.array([], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False), True)

        a1 = np.array([], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.any, skipna=False), False)


        # float arrays 1d
        a1 = np.array([2.4, 5.4], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=True), True)

        # skippna is False, but there is non NaN, so we do not raise
        a1 = np.array([2.4, 0], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False), False)

        a1 = np.array([0, np.nan, 0], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.any, skipna=True), False)

        with self.assertRaises(TypeError):
            a1 = np.array([0, np.nan, 0], dtype=float)
            self.assertEqual(_ufunc_logical_skipna(a1, np.any, skipna=False), True)


        # float arrays 2d
        a1 = np.array([[2.4, 5.4, 3.2], [2.4, 5.4, 3.2]], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False, axis=0).tolist(),
                [True, True, True])

        a1 = np.array([[2.4, 5.4, 3.2], [2.4, 5.4, 3.2]], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False, axis=1).tolist(),
                [True, True])

        a1 = np.array([[2.4, 5.4, 0], [2.4, 5.4, 3.2]], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False, axis=0).tolist(),
                [True, True, False])

        a1 = np.array([[2.4, 5.4, 0], [2.4, 5.4, 3.2]], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False, axis=1).tolist(),
                [False, True])


        # object arrays
        a1 = np.array([[2.4, 5.4, 0], [2.4, None, 3.2]], dtype=object)


        with self.assertRaises(TypeError):
            self.assertAlmostEqualValues(
                    _ufunc_logical_skipna(a1, np.all, skipna=False, axis=1).tolist(),
                    [False, np.nan])

        with self.assertRaises(TypeError):
            self.assertAlmostEqualValues(
                    _ufunc_logical_skipna(a1, np.any, skipna=False, axis=1).tolist(),
                    [True, np.nan])

        with self.assertRaises(TypeError):
            self.assertAlmostEqualValues(_ufunc_logical_skipna(a1, np.all, skipna=False, axis=0).tolist(),
                    [True, np.nan, False])

        with self.assertRaises(TypeError):
            self.assertAlmostEqualValues(_ufunc_logical_skipna(a1, np.any, skipna=False, axis=0).tolist(),
                    [True, np.nan, True])


        a2 = np.array([[2.4, 5.4, 0], [2.4, np.nan, 3.2]], dtype=object)

        with self.assertRaises(TypeError):
            self.assertAlmostEqualValues(
                    _ufunc_logical_skipna(a2, np.any, skipna=False, axis=1).tolist(),
                    [True, np.nan])

        with self.assertRaises(TypeError):
            self.assertAlmostEqualValues(_ufunc_logical_skipna(a2, np.all, skipna=False, axis=0).tolist(),
                    [True, np.nan, False])

        with self.assertRaises(TypeError):
            self.assertAlmostEqualValues(_ufunc_logical_skipna(a2, np.any, skipna=False, axis=0).tolist(),
                    [True, np.nan, True])


    def test_ufunc_logical_skipna_b(self) -> None:
        # object arrays

        a1 = np.array([['sdf', '', 'wer'], [True, False, True]], dtype=object)

        self.assertEqual(
                _ufunc_logical_skipna(a1, np.all, skipna=False, axis=0).tolist(),
                [True, False, True]
                )
        self.assertEqual(
                _ufunc_logical_skipna(a1, np.all, skipna=False, axis=1).tolist(),
                [False, False]
                )


        # string arrays
        a1 = np.array(['sdf', ''], dtype=str)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False, axis=0), False)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=True, axis=0), False)


        a1 = np.array([['sdf', '', 'wer'], ['sdf', '', 'wer']], dtype=str)
        self.assertEqual(
                _ufunc_logical_skipna(a1, np.all, skipna=False, axis=0).tolist(),
                [True,  False,  True])

        self.assertEqual(
                _ufunc_logical_skipna(a1, np.all, skipna=False, axis=1).tolist(),
                [False, False])

        self.assertEqual(
                _ufunc_logical_skipna(a1, np.any, skipna=False, axis=0).tolist(),
                [True,  False,  True])

        self.assertEqual(
                _ufunc_logical_skipna(a1, np.any, skipna=False, axis=1).tolist(),
                [True, True])


    def test_ufunc_logical_skipna_c(self) -> None:

        a1 = np.array([], dtype=float)
        with self.assertRaises(NotImplementedError):
            _ufunc_logical_skipna(a1, np.sum, skipna=True)


    def test_ufunc_logical_skipna_d(self) -> None:

        a1 = np.array(['2018-01-01', '2018-02-01'], dtype=np.datetime64)
        post1 = _ufunc_logical_skipna(a1, np.all, skipna=True)
        self.assertTrue(post1)

        a2 = np.array(['2018-01-01', '2018-02-01', None], dtype=np.datetime64)
        with self.assertRaises(TypeError):
            post2 = _ufunc_logical_skipna(a2, np.all, skipna=False)


    def test_ufunc_logical_skipna_e(self) -> None:

        a1 = np.array([['2018-01-01', '2018-02-01'],
                ['2018-01-01', '2018-02-01']], dtype=np.datetime64)
        post = _ufunc_logical_skipna(a1, np.all, skipna=True)
        self.assertEqual(post.tolist(), [True, True])




if __name__ == '__main__':
    unittest.main()
