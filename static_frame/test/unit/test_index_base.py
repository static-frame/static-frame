from __future__ import annotations

import typing as tp
import numpy as np

from static_frame.core.index_base import IndexBase
from static_frame.test.test_case import TestCase
from static_frame.core.index_datetime import IndexDatetime


class TestUnit(TestCase):
    def test_index_base_slotted_a(self) -> None:
        idx1 = IndexBase()

        with self.assertRaises(AttributeError):
            idx1.g = 30  # type: ignore
        with self.assertRaises(AttributeError):
            idx1.__dict__

    def test_index_base_not_implemented(self) -> None:
        idx1 = IndexBase()

        with self.assertRaises(NotImplementedError):
            idx1._ufunc_axis_skipna(
                axis=0,
                skipna=False,
                ufunc=np.sum,
                ufunc_skipna=np.nansum,
                composable=True,
                dtypes=(),
                size_one_unity=True,
            )

        with self.assertRaises(NotImplementedError):
            idx1._update_array_cache()

        with self.assertRaises(NotImplementedError):
            idx1.copy()

        with self.assertRaises(NotImplementedError):
            idx1.copy()

        with self.assertRaises(NotImplementedError):
            idx1.display()

        with self.assertRaises(NotImplementedError):
            idx1.from_labels(())


    def test_index_constructors_a(self) -> None:
        def all_subclasses(cls: tp.Type[IndexBase]) -> tp.Iterator[tp.Type[IndexBase]]:
            for sub in cls.__subclasses__():
                yield sub
                yield from all_subclasses(sub)

        for cls in all_subclasses(IndexBase):
            if cls is IndexDatetime:
                continue

            mutable_cls = cls._MUTABLE_CONSTRUCTOR
            immutable_cls = cls._IMMUTABLE_CONSTRUCTOR

            # Every subclass must resolve to correctly typed constructors
            self.assertTrue(immutable_cls.STATIC, f'{immutable_cls.__name__} should be STATIC=True')
            self.assertFalse(mutable_cls.STATIC, f'{mutable_cls.__name__} should be STATIC=False')

            # Name and symmetry checks
            self.assertEqual(mutable_cls.__name__, immutable_cls.__name__ + 'GO',
                f'{mutable_cls.__name__}._MUTABLE_CONSTRUCTOR should be named {cls.__name__}GO')

            stem = cls.__name__.replace('GO', '')
            self.assertTrue(mutable_cls.__name__.startswith(stem))
            self.assertTrue(immutable_cls.__name__.startswith(stem))

if __name__ == '__main__':
    import unittest

    unittest.main()
