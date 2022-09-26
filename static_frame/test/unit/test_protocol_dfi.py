import numpy as np

from static_frame.core.protocol_dfi import ArrowCType
from static_frame.core.protocol_dfi import DFIBuffer
from static_frame.core.protocol_dfi import DFIColumn
from static_frame.core.protocol_dfi import DFIDataFrame
from static_frame.test.test_case import TestCase
from static_frame.core.protocol_dfi_abc import DlpackDeviceType


class TestUnit(TestCase):

    def test_arrow_ctype_a(self):
        self.assertEqual(ArrowCType.from_dtype(np.dtype(np.float64)), 'g')
        self.assertEqual(ArrowCType.from_dtype(np.dtype(np.float32)), 'f')
        self.assertEqual(ArrowCType.from_dtype(np.dtype(np.float16)), 'e')

        self.assertEqual(ArrowCType.from_dtype(np.dtype(np.int64)), 'l')
        self.assertEqual(ArrowCType.from_dtype(np.dtype(np.int8)), 'c')

        self.assertEqual(ArrowCType.from_dtype(np.dtype(bool)), 'b')

        self.assertEqual(ArrowCType.from_dtype(np.dtype(np.uint64)), 'L')
        self.assertEqual(ArrowCType.from_dtype(np.dtype(np.uint8)), 'C')

    def test_arrow_ctype_b(self):
        with self.assertRaises(NotImplementedError):
            ArrowCType.from_dtype(np.dtype(object))

    def test_arrow_ctype_c(self):
        self.assertEqual(ArrowCType.from_dtype(np.dtype(str)), 'u')

    def test_arrow_ctype_d(self):
        self.assertEqual(ArrowCType.from_dtype(np.dtype(np.datetime64('2022-01-01'))), 'tdm')

    def test_arrow_ctype_e(self):
        self.assertEqual(ArrowCType.from_dtype(np.dtype(np.datetime64('2022-01-01T01:01:01'))), 'tts')
        self.assertEqual(ArrowCType.from_dtype(np.dtype(np.datetime64('2022-01-01', 'ns'))), 'ttn')

    def test_arrow_ctype_f(self):
        with self.assertRaises(NotImplementedError):
            ArrowCType.from_dtype(np.dtype(np.datetime64('2022-01')))

    def test_arrow_ctype_g(self):
        with self.assertRaises(NotImplementedError):
            ArrowCType.from_dtype(np.dtype(complex))



    #---------------------------------------------------------------------------
    def test_dfi_buffer_a(self):
        dfib = DFIBuffer(np.array((True, False)))
        self.assertEqual(str(dfib), '<DFIBuffer: shape=(2,) dtype=|b1>')
        self.assertTrue(dfib.__array__().data.contiguous)

    def test_dfi_buffer_b(self):
        dfib = DFIBuffer((np.arange(12).reshape(6, 2) % 3 == 0)[:, 0])
        self.assertEqual(str(dfib), '<DFIBuffer: shape=(6,) dtype=|b1>')
        self.assertTrue(dfib.__array__().data.contiguous)

    def test_dfi_buffer_array_a(self):
        a1 = np.array((True, False))
        dfib = DFIBuffer(a1)
        self.assertEqual(dfib.__array__().tolist(), a1.tolist())

    def test_dfi_buffer_bufsize_a(self):
        a1 = np.array((True, False))
        dfib = DFIBuffer(a1)
        self.assertEqual(dfib.bufsize, 2)

    def test_dfi_buffer_ptr_a(self):
        a1 = np.array((True, False))
        dfib = DFIBuffer(a1)
        self.assertEqual(dfib.ptr, a1.__array_interface__['data'][0])

    def test_dfi_buffer_dlpack_a(self):
        a1 = np.array((True, False))
        dfib = DFIBuffer(a1)
        with self.assertRaises(NotImplementedError):
            dfib.__dlpack__()

    def test_dfi_buffer_dlpack_device_a(self):
        a1 = np.array((True, False))
        dfib = DFIBuffer(a1)
        self.assertEqual(dfib.__dlpack_device__(), (DlpackDeviceType.CPU, None))

        # import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    import unittest
    unittest.main()