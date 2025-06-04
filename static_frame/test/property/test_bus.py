import frame_fixtures as ff
from hypothesis import given

from static_frame.core.bus import Bus
from static_frame.core.frame import Frame
from static_frame.test.property.strategies import DTGroup, get_array_1d
from static_frame.test.test_case import TestCase, temp_file


class TestUnit(TestCase):
    @given(
        get_array_1d(min_size=0, max_size=20, dtype_group=DTGroup.NUMERIC_INT),
        get_array_1d(min_size=0, max_size=20, dtype_group=DTGroup.NUMERIC_INT),
    )
    def test_bus_max_persist_a(self, pos_start, pos_end):
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')
        f7 = ff.parse('s(2,4)').rename('f7')
        f8 = ff.parse('s(5,4)').rename('f8')
        f9 = ff.parse('s(2,7)').rename('f9')
        f10 = ff.parse('s(8,2)').rename('f10')
        f11 = ff.parse('s(4,9)').rename('f11')
        f12 = ff.parse('s(4,6)').rename('f12')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12))

        # include over-sized max_persist
        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)

            for max_persist in range(1, len(b1) + 1):
                b2 = Bus.from_zip_npz(fp, max_persist=max_persist)

                for iloc in pos_start % len(b2):
                    f = b2.iloc[iloc]
                    self.assertIsInstance(f, Frame)
                    self.assertEqual(f.name, f'f{int(iloc + 1)}')

                # force full iteration
                items = list(b2.items())
                self.assertEqual([f.name for _, f in items], list(b1.index))

                for iloc in pos_end % len(b2):
                    self.assertIsInstance(b2.iloc[iloc], Frame)

    @given(
        get_array_1d(min_size=0, max_size=20, dtype_group=DTGroup.NUMERIC_INT),
        get_array_1d(min_size=0, max_size=20, dtype_group=DTGroup.NUMERIC_INT),
    )
    def test_bus_max_persist_b(self, pos_start, pos_end):
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')
        f7 = ff.parse('s(2,4)').rename('f7')
        f8 = ff.parse('s(5,4)').rename('f8')
        f9 = ff.parse('s(2,7)').rename('f9')
        f10 = ff.parse('s(8,2)').rename('f10')
        f11 = ff.parse('s(4,9)').rename('f11')
        f12 = ff.parse('s(4,6)').rename('f12')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12))

        # include over-sized max_persist
        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)

            for max_persist in range(1, len(b1) + 1):
                b2 = Bus.from_zip_npz(fp, max_persist=max_persist)

                for iloc in pos_start % len(b2):
                    f = b2.iloc[iloc]
                    self.assertIsInstance(f, Frame)
                    self.assertEqual(f.name, f'f{int(iloc + 1)}')

                # force full iteration
                values = b2.values
                self.assertEqual([f.name for f in values], list(b1.index))

                for iloc in pos_end % len(b2):
                    self.assertIsInstance(b2.iloc[iloc], Frame)

                self.assertEqual([f.name for f in b2.iter_element()], list(b1.index))

    # ---------------------------------------------------------------------------

    @given(
        get_array_1d(min_size=0, max_size=20, dtype_group=DTGroup.NUMERIC_INT),
        get_array_1d(min_size=0, max_size=20, dtype_group=DTGroup.NUMERIC_INT),
    )
    def test_bus_persistant_a(self, pos_start, pos_end):
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')
        f7 = ff.parse('s(2,4)').rename('f7')
        f8 = ff.parse('s(5,4)').rename('f8')
        f9 = ff.parse('s(2,7)').rename('f9')
        f10 = ff.parse('s(8,2)').rename('f10')
        f11 = ff.parse('s(4,9)').rename('f11')
        f12 = ff.parse('s(4,6)').rename('f12')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12))

        # include over-sized max_persist
        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)
            b2 = Bus.from_zip_npz(fp)

            for iloc in pos_start % len(b2):
                f = b2.iloc[iloc]
                self.assertIsInstance(f, Frame)
                self.assertEqual(f.name, f'f{int(iloc + 1)}')

            # force full iteration
            items = list(b2.items())
            self.assertEqual([f.name for _, f in items], list(b1.index))

            for iloc in pos_end % len(b2):
                self.assertIsInstance(b2.iloc[iloc], Frame)

    @given(
        get_array_1d(min_size=0, max_size=20, dtype_group=DTGroup.NUMERIC_INT),
        get_array_1d(min_size=0, max_size=20, dtype_group=DTGroup.NUMERIC_INT),
    )
    def test_bus_persistant_b(self, pos_start, pos_end):
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')
        f7 = ff.parse('s(2,4)').rename('f7')
        f8 = ff.parse('s(5,4)').rename('f8')
        f9 = ff.parse('s(2,7)').rename('f9')
        f10 = ff.parse('s(8,2)').rename('f10')
        f11 = ff.parse('s(4,9)').rename('f11')
        f12 = ff.parse('s(4,6)').rename('f12')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12))

        # include over-sized max_persist
        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)
            b2 = Bus.from_zip_npz(fp)

            for iloc in pos_start % len(b2):
                f = b2.iloc[iloc]
                self.assertIsInstance(f, Frame)
                self.assertEqual(f.name, f'f{int(iloc + 1)}')

            # force full iteration
            values = b2.values
            self.assertEqual([f.name for f in values], list(b1.index))

            for iloc in pos_end % len(b2):
                self.assertIsInstance(b2.iloc[iloc], Frame)

            self.assertEqual([f.name for f in b2.iter_element()], list(b1.index))

    # ---------------------------------------------------------------------------
    @given(
        get_array_1d(min_size=0, max_size=20, dtype_group=DTGroup.NUMERIC_INT),
        get_array_1d(min_size=0, max_size=20, dtype_group=DTGroup.NUMERIC_INT),
    )
    def test_bus_persist_many_a(self, pos_start, pos_end):
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')
        f7 = ff.parse('s(2,4)').rename('f7')
        f8 = ff.parse('s(5,4)').rename('f8')
        f9 = ff.parse('s(2,7)').rename('f9')
        f10 = ff.parse('s(8,2)').rename('f10')
        f11 = ff.parse('s(4,9)').rename('f11')
        f12 = ff.parse('s(4,6)').rename('f12')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12))

        # include over-sized max_persist
        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)

            for max_persist in range(1, len(b1) + 1):
                b2 = Bus.from_zip_npz(fp, max_persist=max_persist)

                for iloc in pos_start % len(b2):
                    b2._update_mutable_max_persist_many(slice(iloc, None))

                for iloc in pos_start % len(b2):
                    f = b2.iloc[iloc]
                    self.assertIsInstance(f, Frame)
                    self.assertEqual(f.name, f'f{int(iloc + 1)}')

                for iloc in pos_end % len(b2):
                    b2._update_mutable_max_persist_many([iloc])

                for iloc in pos_end % len(b2):
                    self.assertIsInstance(b2.iloc[iloc], Frame)

    @given(
        get_array_1d(min_size=0, max_size=20, dtype_group=DTGroup.NUMERIC_INT),
        get_array_1d(min_size=0, max_size=20, dtype_group=DTGroup.NUMERIC_INT),
    )
    def test_bus_persist_many_b(self, pos_start, pos_end):
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')
        f7 = ff.parse('s(2,4)').rename('f7')
        f8 = ff.parse('s(5,4)').rename('f8')
        f9 = ff.parse('s(2,7)').rename('f9')
        f10 = ff.parse('s(8,2)').rename('f10')
        f11 = ff.parse('s(4,9)').rename('f11')
        f12 = ff.parse('s(4,6)').rename('f12')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12))

        # include over-sized max_persist
        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)

            b2 = Bus.from_zip_npz(fp)

            for iloc in pos_start % len(b2):
                b2._update_mutable_persistant_many(slice(iloc, None))

            for iloc in pos_start % len(b2):
                f = b2.iloc[iloc]
                self.assertIsInstance(f, Frame)
                self.assertEqual(f.name, f'f{int(iloc + 1)}')

            b2.unpersist()

            for iloc in pos_end % len(b2):
                b2._update_mutable_persistant_many([iloc])

            for iloc in pos_end % len(b2):
                self.assertIsInstance(b2.iloc[iloc], Frame)
