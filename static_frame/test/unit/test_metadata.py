from __future__ import annotations

import frame_fixtures as ff
import numpy as np

from static_frame.core.index import Index
from static_frame.core.index_datetime import IndexDate
from static_frame.core.metadata import JSONMeta
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    def _round_trip_dtype(self, dt) -> None:
        self.assertEqual(np.dtype(JSONMeta._dtype_to_str(dt)), dt)

    def test_dtype_to_str_a(self) -> None:
        self.assertEqual(JSONMeta._dtype_to_str(np.dtype(np.float64)), '=f8')
        self._round_trip_dtype(np.dtype(np.float64))

    def test_dtype_to_str_b(self) -> None:
        self.assertEqual(JSONMeta._dtype_to_str(np.dtype(np.uint8)), '|u1')
        self._round_trip_dtype(np.dtype(np.uint8))

    def test_dtype_to_str_c(self) -> None:
        dt = np.array(('2022-01', '2022-03'), dtype=np.datetime64).dtype
        self.assertEqual(JSONMeta._dtype_to_str(dt), '=M8[M]')
        self._round_trip_dtype(dt)

    #---------------------------------------------------------------------------
    def test_to_metadata_a(self) -> None:
        f = ff.parse('s(2,3)|v(int64,float)|i(ID,dtD)').rename('a')
        md = JSONMeta.to_dict(f)
        self.assertEqual(md, {
                '__names__': ['a', None, None],
                '__types__': ['IndexDate', 'Index'],
                '__depths__': [3, 1, 1],
                '__dtypes__': ['=i8', '=f8', '=i8'],
                '__dtypes_columns__': ['=i8'],
                '__dtypes_index__': ['=M8[D]'],
                })

    def test_to_metadata_b(self) -> None:
        f = ff.parse('s(2,4)|v(int64,float)|i(ID,dtD)|c(IH, (int64, str))').rename('a')
        md = JSONMeta.to_dict(f)
        self.assertEqual(md, {
                '__names__': ['a', None, None],
                '__types__': ['IndexDate', 'IndexHierarchy'],
                '__types_columns__': ['Index', 'Index'],
                '__depths__': [4, 1, 2],
                '__dtypes__': ['=i8', '=f8', '=i8', '=f8'],
                '__dtypes_columns__': ['=i8', '=U4'],
                '__dtypes_index__': ['=M8[D]'],
                })

    def test_to_metadata_c(self) -> None:
        f = ff.parse('s(2,4)|v(bool)|i(Is,dts)|c(IH, (int64, str, str))').rename('a')
        md = JSONMeta.to_dict(f)
        self.assertEqual(md, {
                '__names__': ['a', None, None],
                '__types__': ['IndexSecond', 'IndexHierarchy'],
                '__types_columns__': ['Index', 'Index', 'Index'],
                '__depths__': [4, 1, 3],
                '__dtypes__': ['|b1', '|b1', '|b1', '|b1'],
                '__dtypes_columns__': ['=i8', '=U4', '=U4'],
                '__dtypes_index__': ['=M8[s]'],
                })

    def test_to_metadata_d(self) -> None:
        f = ff.parse('s(2,4)|v(bool)|c(Is,dts)|i(IH, (int64, str, str))').rename('a')
        md = JSONMeta.to_dict(f)
        self.assertEqual(md, {
                '__names__': ['a', None, None],
                '__types__': ['IndexHierarchy', 'IndexSecond'],
                '__types_index__': ['Index', 'Index', 'Index'],
                '__depths__': [4, 3, 1],
                '__dtypes__': ['|b1', '|b1', '|b1', '|b1'],
                '__dtypes_columns__': ['=M8[s]'],
                '__dtypes_index__': ['=i8', '=U4', '=U4'],
                })

    def test_to_metadata_e(self) -> None:
        f = ff.parse('s(2,4)|v(bool)|c(IH, (str, int64, str))|i(IH, (int64, str))').rename('a')
        md = JSONMeta.to_dict(f)
        self.assertEqual(md, {
                '__names__': ['a', None, None],
                '__types__': ['IndexHierarchy', 'IndexHierarchy'],
                '__types_index__': ['Index', 'Index'],
                '__types_columns__': ['Index', 'Index', 'Index'],
                '__depths__': [4, 2, 3],
                '__dtypes__': ['|b1', '|b1', '|b1', '|b1'],
                '__dtypes_columns__': ['=U4', '=i8', '=U4'],
                '__dtypes_index__': ['=i8', '=U4'],
                })

    #---------------------------------------------------------------------------
    def test_from_dict_to_ctors_a(self) -> None:
        f = ff.parse('s(2,3)|v(int,float)|i(ID,dtD)').rename('a', index='row', columns='col')
        md = JSONMeta.to_dict(f)
        index_ctor, columns_ctor = JSONMeta.from_dict_to_ctors(md, True)
        idx1 = index_ctor(('2022-01-01',))
        self.assertEqual(idx1.name, 'row')
        self.assertIs(idx1.__class__, IndexDate)

        idx2 = columns_ctor((3, 2))
        self.assertEqual(idx2.name, 'col')
        self.assertIs(idx2.__class__, Index)
        self.assertEqual(idx2.dtype.kind, 'i')


