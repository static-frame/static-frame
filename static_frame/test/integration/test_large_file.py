from zipfile import ZipFile

import numpy as np

from static_frame.core.archive_zip import ZipFileRO, zip_namelist
from static_frame.core.frame import Frame, FrameGO
from static_frame.core.index_auto import IndexAutoFactory
from static_frame.test.test_case import TestCase, skip_pyle310, temp_file


class TestUnit(TestCase):
    def test_exceed_columns(self) -> None:
        f1 = Frame.from_element('x', index=('x',), columns=range(16384))

        with temp_file('.xlsx') as fp:
            with self.assertRaises(RuntimeError):
                # with the index, the limit is exceeded
                f1.to_xlsx(fp, include_index=True)

            f1.to_xlsx(fp, include_index=False)
            f2 = Frame.from_xlsx(fp, index_depth=0, columns_depth=1)
            # need to remove index on original for appropriate comparison
            self.assertEqualFrames(f1.relabel(index=IndexAutoFactory), f2)

    def test_exceed_rows(self) -> None:
        f1 = Frame.from_element('x', index=range(1048576), columns=('x',))

        with temp_file('.xlsx') as fp:
            with self.assertRaises(RuntimeError):
                # with the index, the limit is exceeded
                f1.to_xlsx(fp, include_columns=True)

            # NOTE: it takes almost 60s to write this file, so we will skip testing it
            # f1.to_xlsx(fp, include_columns=False)
            # f2 = Frame.from_xlsx(fp, index_depth=1, columns_depth=0)
            # # need to remove index on original for appropriate comparison
            # self.assertEqualFrames(f1.relabel(columns=IndexAutoFactory), f2)

    # ---------------------------------------------------------------------------
    @skip_pyle310
    def test_zip64_a(self) -> None:
        # need 536,870,911 64-bit floats to hit zip64 file size

        with temp_file('.npz') as fp:
            f = FrameGO(index=IndexAutoFactory(600_000_000))
            array = np.random.random_sample(600_000_000)
            array.flags.writeable = False

            for i in range(2):
                f[str(i)] = array
            f.to_npz(fp)
            del f
            del array

            with ZipFile(fp) as zf, ZipFileRO(fp) as zfro:
                with zf.open('__meta__.json') as part_zf:
                    with zfro.open('__meta__.json') as part_zfro:
                        self.assertEqual(part_zf.read(), part_zfro.read())

            post = list(zip_namelist(fp))
            self.assertEqual(
                post,
                [
                    '__values_columns_0__.npy',
                    '__blocks_0__.npy',
                    '__blocks_1__.npy',
                    '__meta__.json',
                ],
            )


if __name__ == '__main__':
    import unittest

    unittest.main()
