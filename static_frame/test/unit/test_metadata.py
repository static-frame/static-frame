from __future__ import annotations

import frame_fixtures as ff

from static_frame.core.metadata import JSONMeta
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):


    #---------------------------------------------------------------------------
    def test_to_metadata_a(self) -> None:
        f = ff.parse('s(2,3)|v(int,float)|i(ID,dtD)').rename('a')
        md = JSONMeta.to_dict(f)
        self.assertEqual(md, {
                '__names__': ['a', None, None],
                '__types__': ['IndexDate', 'Index'],
                '__depths__': [3, 1, 1],
                })

    def test_to_metadata_b(self) -> None:
        f = ff.parse('s(2,4)|v(int,float)|i(ID,dtD)|c(IH, (int, str))').rename('a')
        md = JSONMeta.to_dict(f)
        self.assertEqual(md, {
                '__names__': ['a', None, None],
                '__types__': ['IndexDate', 'IndexHierarchy'],
                '__types_columns__': ['Index', 'Index'],
                '__depths__': [4, 1, 2],
                })

    def test_to_metadata_c(self) -> None:
        f = ff.parse('s(2,4)|v(bool)|i(IS,dts)|c(IH, (int, str, str))').rename('a')
        md = JSONMeta.to_dict(f)
        self.assertEqual(md, {
                '__names__': ['a', None, None],
                '__types__': ['IndexSecond', 'IndexHierarchy'],
                '__types_columns__': ['Index', 'Index', 'Index'],
                '__depths__': [1, 1, 3],
                })

    def test_to_metadata_d(self) -> None:
        f = ff.parse('s(2,4)|v(bool)|c(IS,dts)|i(IH, (int, str, str))').rename('a')
        md = JSONMeta.to_dict(f)
        self.assertEqual(md, {
                '__names__': ['a', None, None],
                '__types__': ['IndexHierarchy', 'IndexSecond'],
                '__types_index__': ['Index', 'Index', 'Index'],
                '__depths__': [1, 3, 1],
                })

    def test_to_metadata_d(self) -> None:
        f = ff.parse('s(2,4)|v(bool)|c(IH, (str, int, str))|i(IH, (int, str))').rename('a')
        md = JSONMeta.to_dict(f)
        self.assertEqual(md, {
                '__names__': ['a', None, None],
                '__types__': ['IndexHierarchy', 'IndexHierarchy'],
                '__types_index__': ['Index', 'Index'],
                '__types_columns__': ['Index', 'Index', 'Index'],
                '__depths__': [1, 2, 3],
                })
