from __future__ import annotations

import frame_fixtures as ff
import numpy as np

from static_frame.core.index import Index
from static_frame.core.protocol_dfi import ArrowCType
from static_frame.core.protocol_dfi import DFIBuffer
from static_frame.core.protocol_dfi import DFIColumn
from static_frame.core.protocol_dfi import DFIDataFrame
from static_frame.core.protocol_dfi import np_dtype_to_dfi_dtype
from static_frame.core.protocol_dfi_abc import ColumnNullType
from static_frame.core.protocol_dfi_abc import DlpackDeviceType
from static_frame.core.protocol_dfi_abc import DtypeKind
from static_frame.core.util import NAT
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    def test_arrow_ctype_a(self) -> None:
        self.assertEqual(ArrowCType.from_dtype(np.dtype(np.float64)), 'g')
        self.assertEqual(ArrowCType.from_dtype(np.dtype(np.float32)), 'f')
        self.assertEqual(ArrowCType.from_dtype(np.dtype(np.float16)), 'e')

        self.assertEqual(ArrowCType.from_dtype(np.dtype(np.int64)), 'l')
        self.assertEqual(ArrowCType.from_dtype(np.dtype(np.int8)), 'c')

        self.assertEqual(ArrowCType.from_dtype(np.dtype(bool)), 'b')

        self.assertEqual(ArrowCType.from_dtype(np.dtype(np.uint64)), 'L')
        self.assertEqual(ArrowCType.from_dtype(np.dtype(np.uint8)), 'C')

    def test_arrow_ctype_b(self) -> None:
        with self.assertRaises(NotImplementedError):
            ArrowCType.from_dtype(np.dtype(object))

    def test_arrow_ctype_c(self) -> None:
        self.assertEqual(ArrowCType.from_dtype(np.dtype(str)), 'u')

    def test_arrow_ctype_d(self) -> None:
        self.assertEqual(ArrowCType.from_dtype(np.dtype(np.datetime64('2022-01-01'))), 'tdm')

    def test_arrow_ctype_e(self) -> None:
        self.assertEqual(ArrowCType.from_dtype(np.dtype(np.datetime64('2022-01-01T01:01:01'))), 'tts')
        self.assertEqual(ArrowCType.from_dtype(np.dtype(np.datetime64('2022-01-01', 'ns'))), 'ttn')

    def test_arrow_ctype_f(self) -> None:
        with self.assertRaises(NotImplementedError):
            ArrowCType.from_dtype(np.dtype(np.datetime64('2022-01')))

    def test_arrow_ctype_g(self) -> None:
        with self.assertRaises(NotImplementedError):
            ArrowCType.from_dtype(np.dtype(complex))

    #---------------------------------------------------------------------------
    def test_np_dtype_to_dfi_dtype_a(self) -> None:
        self.assertEqual(np_dtype_to_dfi_dtype(
                np.dtype(bool)),
                (DtypeKind.BOOL, 8, 'b', '='),
                )

    def test_np_dtype_to_dfi_dtype_b(self) -> None:
        self.assertEqual(np_dtype_to_dfi_dtype(
                np.dtype(np.float64)),
                (DtypeKind.FLOAT, 64, 'g', '='),
                )

    def test_np_dtype_to_dfi_dtype_c(self) -> None:
        self.assertEqual(np_dtype_to_dfi_dtype(
                np.dtype(np.uint8)),
                (DtypeKind.UINT, 8, 'C', '='),
                )

    #---------------------------------------------------------------------------
    def test_dfi_buffer_a(self) -> None:
        dfib = DFIBuffer(np.array((True, False)))
        self.assertEqual(str(dfib), '<DFIBuffer: shape=(2,) dtype=|b1>')
        self.assertTrue(dfib.__array__().data.contiguous)

    def test_dfi_buffer_b(self) -> None:
        # only accept already-contiguous arrays
        with self.assertRaises(ValueError):
            dfib = DFIBuffer((np.arange(12).reshape(6, 2) % 3 == 0)[:, 0])

    def test_dfi_buffer_array_a(self) -> None:
        a1 = np.array((True, False))
        dfib = DFIBuffer(a1)
        self.assertEqual(dfib.__array__().tolist(), a1.tolist())

    def test_dfi_buffer_array_b(self) -> None:
        a1 = np.array((True, False))
        dfib = DFIBuffer(a1)
        self.assertEqual(dfib.__array__(str).tolist(), a1.astype(str).tolist())

    def test_dfi_buffer_bufsize_a(self) -> None:
        a1 = np.array((True, False))
        dfib = DFIBuffer(a1)
        self.assertEqual(dfib.bufsize, 2)

    def test_dfi_buffer_ptr_a(self) -> None:
        a1 = np.array((True, False))
        dfib = DFIBuffer(a1)
        self.assertEqual(dfib.ptr, a1.__array_interface__['data'][0])

    def test_dfi_buffer_dlpack_a(self) -> None:
        a1 = np.array((True, False))
        dfib = DFIBuffer(a1)
        with self.assertRaises(NotImplementedError):
            dfib.__dlpack__()

    def test_dfi_buffer_dlpack_device_a(self) -> None:
        a1 = np.array((True, False))
        dfib = DFIBuffer(a1)
        self.assertEqual(dfib.__dlpack_device__(), (DlpackDeviceType.CPU, None))

    #---------------------------------------------------------------------------

    def test_dfi_column_init_a(self) -> None:
        a1 = np.array((True, False))
        idx1 = Index(('a', 'b'))
        dfic = DFIColumn(a1, idx1)
        self.assertEqual(str(dfic), '<DFIColumn: shape=(2,) dtype=|b1>')

    def test_dfi_column_array_a(self) -> None:
        a1 = np.array((True, False))
        idx1 = Index(('a', 'b'))
        dfic = DFIColumn(a1, idx1)
        self.assertEqual(dfic.__array__().tolist(), a1.tolist())

    def test_dfi_column_array_b(self) -> None:
        a1 = np.array((True, False))
        idx1 = Index(('a', 'b'))
        dfic = DFIColumn(a1, idx1)
        self.assertEqual(dfic.__array__(str).tolist(), a1.astype(str).tolist())

    def test_dfi_column_size_a(self) -> None:
        a1 = np.array((True, False))
        idx1 = Index(('a', 'b'))
        dfic = DFIColumn(a1, idx1)
        self.assertEqual(dfic.size(), 2)

    def test_dfi_column_offset_a(self) -> None:
        a1 = np.array((True, False))
        idx1 = Index(('a', 'b'))
        dfic = DFIColumn(a1, idx1)
        self.assertEqual(dfic.offset, 0)

    def test_dfi_column_dtype_a(self) -> None:
        a1 = np.array((True, False))
        idx1 = Index(('a', 'b'))
        dfic = DFIColumn(a1, idx1)
        self.assertEqual(dfic.dtype, (DtypeKind.BOOL, 8, 'b', '='))

    def test_dfi_column_dtype_b(self) -> None:
        a1 = np.array((1.1, 2.2), dtype=np.float64)
        idx1 = Index(('a', 'b'))
        dfic = DFIColumn(a1, idx1)
        self.assertEqual(dfic.dtype, (DtypeKind.FLOAT, 64, 'g', '='))

    def test_dfi_column_dtype_c(self) -> None:
        a1 = np.array((1.1, 2.2), dtype=np.float16)
        idx1 = Index(('a', 'b'))
        dfic = DFIColumn(a1, idx1)
        self.assertEqual(dfic.dtype, (DtypeKind.FLOAT, 16, 'e', '='))

    def test_dfi_column_describe_categorical_a(self) -> None:
        a1 = np.array((1.1, 2.2), dtype=np.float64)
        idx1 = Index(('a', 'b'))
        dfic = DFIColumn(a1, idx1)
        with self.assertRaises(TypeError):
            dfic.describe_categorical()

    def test_dfi_column_describe_null_a(self) -> None:
        a1 = np.array((1.1, 2.2, np.nan), dtype=np.float64)
        idx1 = Index(('a', 'b', 'c'))
        dfic = DFIColumn(a1, idx1)
        self.assertEqual(dfic.describe_null, (ColumnNullType.USE_NAN, None))

    def test_dfi_column_describe_null_b(self) -> None:
        a1 = np.array(('2020-01-01', '2022-05-01', NAT), dtype=np.datetime64)
        idx1 = Index(('a', 'b', 'c'))
        dfic = DFIColumn(a1, idx1)
        self.assertEqual(dfic.describe_null, (ColumnNullType.USE_SENTINEL, NAT))
        post = dfic.get_buffers()

    def test_dfi_column_describe_null_c(self) -> None:
        a1 = np.array((3, 4))
        idx1 = Index(('a', 'b'))
        dfic = DFIColumn(a1, idx1)
        self.assertEqual(dfic.describe_null, (ColumnNullType.NON_NULLABLE, None))

    def test_dfi_column_null_count_a(self) -> None:
        a1 = np.array((1.1, 2.2, np.nan), dtype=np.float64)
        idx1 = Index(('a', 'b', 'c'))
        dfic = DFIColumn(a1, idx1)
        self.assertEqual(dfic.null_count, 1)

    def test_dfi_column_null_count_b(self) -> None:
        a1 = np.array(('2020-01', '2022-05', NAT), dtype=np.datetime64)
        idx1 = Index(('a', 'b', 'c'))
        dfic = DFIColumn(a1, idx1)
        self.assertEqual(dfic.null_count, 1)

    def test_dfi_column_null_count_c(self) -> None:
        a1 = np.array((3, 4))
        idx1 = Index(('a', 'b'))
        dfic = DFIColumn(a1, idx1)
        self.assertEqual(dfic.null_count, 0)

    def test_dfi_column_metadata_a(self) -> None:
        a1 = np.array((3, 4))
        idx1 = Index(('a', 'b'))
        dfic = DFIColumn(a1, idx1)
        [(mk, mv)] = dfic.metadata.items()
        self.assertEqual(mk, 'static-frame.index')
        self.assertTrue(mv.equals(mv))

    def test_dfi_column_num_chunks_a(self) -> None:
        a1 = np.array((3, 4))
        idx1 = Index(('a', 'b'))
        dfic = DFIColumn(a1, idx1)
        self.assertEqual(dfic.num_chunks(), 1)

    def test_dfi_column_chunks_a(self) -> None:
        a1 = np.arange(5)
        idx1 = Index(('a', 'b', 'c', 'd', 'e'))
        dfic = DFIColumn(a1, idx1)
        post = tuple(dfic.get_chunks(2))

        self.assertEqual(
                [c.__array__().tolist() for c in post],
                [[0, 1, 2], [3, 4]],
                )

    def test_dfi_column_chunks_b(self) -> None:
        a1 = np.arange(5)
        idx1 = Index(('a', 'b', 'c', 'd', 'e'))
        dfic = DFIColumn(a1, idx1)
        post = tuple(dfic.get_chunks(5))

        self.assertEqual(
                [c.__array__().tolist() for c in post],
                [[0], [1], [2], [3], [4]],
                )

    def test_dfi_column_chunks_c(self) -> None:
        a1 = np.arange(5)
        idx1 = Index(('a', 'b', 'c', 'd', 'e'))
        dfic = DFIColumn(a1, idx1)
        post = tuple(dfic.get_chunks(1))

        self.assertEqual(
                [c.__array__().tolist() for c in post],
                [[0, 1, 2, 3, 4]],
                )

    def test_dfi_column_get_buffers_a(self) -> None:
        a1 = np.array((1.1, 2.2, np.nan), dtype=np.float64)
        idx1 = Index(('a', 'b', 'c'))
        dfic = DFIColumn(a1, idx1)
        post = dfic.get_buffers()

        assert post['data'] is not None

        self.assertEqual(str(post['data'][0]), '<DFIBuffer: shape=(3,) dtype=<f8>')
        self.assertEqual(post['data'][1], (DtypeKind.FLOAT, 64, 'g', '='))

        assert post['validity'] is not None

        self.assertEqual(str(post['validity'][0]), '<DFIBuffer: shape=(3,) dtype=|b1>')
        self.assertEqual(post['validity'][1], (DtypeKind.BOOL, 8, 'b', '='))

        self.assertEqual(post['offsets'], None)

    def test_dfi_column_get_buffers_b(self) -> None:
        a1 = np.array((False, True, False), dtype=bool)
        idx1 = Index(('a', 'b', 'c'))
        dfic = DFIColumn(a1, idx1)
        post = dfic.get_buffers()

        assert post['data'] is not None

        self.assertEqual(str(post['data'][0]), '<DFIBuffer: shape=(3,) dtype=|b1>')
        self.assertEqual(post['data'][1], (DtypeKind.BOOL, 8, 'b', '='))

        self.assertEqual(post['validity'], None)
        self.assertEqual(post['offsets'], None)

    #---------------------------------------------------------------------------

    def test_dfi_df_init_a(self) -> None:
        f = ff.parse('s(3,2)|v(bool,float)')
        dfif1 = DFIDataFrame(f)
        dfif2 = dfif1.__dataframe__()
        self.assertEqual(str(dfif1), str(dfif2))

    def test_dfi_df_array_a(self) -> None:
        f = ff.parse('s(3,2)|v(bool)')
        dfif = DFIDataFrame(f)
        post = dfif.__array__(str).tolist()
        self.assertEqual(post,
                [['False', 'False'], ['False', 'False'], ['False', 'False']])

    def test_dfi_df_metadata_a(self) -> None:
        f = ff.parse('s(3,2)|v(bool,float)').rename('foo')
        dfif = DFIDataFrame(f)
        [(mk, mv), (mnk, mnv)] = dfif.metadata.items()
        self.assertEqual(mk, 'static-frame.index')
        self.assertTrue(mv.equals(f.index))
        self.assertEqual(mnk, 'static-frame.name')
        self.assertEqual(mnv, f.name)

    def test_dfi_df_num_columns_a(self) -> None:
        f = ff.parse('s(3,2)|v(bool,float)')
        dfif = DFIDataFrame(f)
        self.assertEqual(dfif.num_columns(), 2)

    def test_dfi_df_num_rows_a(self) -> None:
        f = ff.parse('s(3,2)|v(bool,float)')
        dfif = DFIDataFrame(f)
        self.assertEqual(dfif.num_rows(), 3)

    def test_dfi_df_num_chunks_a(self) -> None:
        f = ff.parse('s(3,2)|v(bool,float)')
        dfif = DFIDataFrame(f)
        self.assertEqual(dfif.num_chunks(), 1)

    def test_dfi_df_column_names_a(self) -> None:
        f = ff.parse('s(3,2)|v(bool,float)')
        dfif = DFIDataFrame(f)
        self.assertEqual(tuple(dfif.column_names()), (0, 1))

    def test_dfi_df_get_column_a(self) -> None:
        f = ff.parse('s(3,2)|v(bool,float)|c(I,str)')
        dfif = DFIDataFrame(f)
        self.assertEqual(str(dfif.get_column(0)), '<DFIColumn: shape=(3,) dtype=|b1>')

    def test_dfi_df_get_column_by_name_a(self) -> None:
        f = ff.parse('s(3,2)|v(bool,float)|c(I,str)')
        dfif = DFIDataFrame(f)
        self.assertEqual(str(dfif.get_column_by_name('zZbu')), '<DFIColumn: shape=(3,) dtype=|b1>')

    def test_dfi_df_get_columns_a(self) -> None:
        f = ff.parse('s(3,2)|v(bool,float)|c(I,str)')
        dfif = DFIDataFrame(f)
        self.assertEqual(
                [str(c) for c in dfif.get_columns()],
                ['<DFIColumn: shape=(3,) dtype=|b1>', '<DFIColumn: shape=(3,) dtype=<f8>']
                )

    def test_dfi_df_select_columns_a(self) -> None:
        f = ff.parse('s(3,4)|v(bool,float)|c(I,str)')
        dfif1 = DFIDataFrame(f)
        dfif2 = dfif1.select_columns([0, 3])
        self.assertEqual(str(dfif2), '<DFIDataFrame: shape=(3, 2)>')

    def test_dfi_df_select_columns_b(self) -> None:
        f = ff.parse('s(3,4)|v(bool,float)|c(I,str)')
        dfif1 = DFIDataFrame(f)
        dfif2 = dfif1.select_columns(range(2))
        self.assertEqual(str(dfif2), '<DFIDataFrame: shape=(3, 2)>')


    def test_dfi_df_select_columns_by_name_a(self) -> None:
        f = ff.parse('s(3,4)|v(bool,float)|c(I,str)')
        dfif1 = DFIDataFrame(f)
        dfif2 = dfif1.select_columns_by_name(['ztsv', 'zkuW'])
        self.assertEqual(str(dfif2), '<DFIDataFrame: shape=(3, 2)>')
        self.assertEqual(tuple(dfif2.column_names()), ('ztsv', 'zkuW'))

    def test_dfi_df_select_columns_by_name_b(self) -> None:
        f = ff.parse('s(3,4)|v(bool,float)|c(I,str)')
        dfif1 = DFIDataFrame(f)
        dfif2 = dfif1.select_columns_by_name((x for x in ('ztsv', 'zkuW')))
        self.assertEqual(str(dfif2), '<DFIDataFrame: shape=(3, 2)>')
        self.assertEqual(tuple(dfif2.column_names()), ('ztsv', 'zkuW'))


    def test_dfi_df_get_chunks_a(self) -> None:
        f = ff.parse('s(5,4)|v(bool,int64)|c(I,str)')
        dfif1 = DFIDataFrame(f)
        post = [df.__array__().tolist() for df in dfif1.get_chunks(2)]
        self.assertEqual(post,
            [[[False, 162197, True, 129017], [False, -41157, False, 35021], [False, 5729, False, 166924]], [[True, -168387, True, 122246], [False, 140627, False, 197228]]])

    def test_dfi_df_get_chunks_b(self) -> None:
        f = ff.parse('s(5,4)|v(bool,int64)|c(I,str)')
        dfif1 = DFIDataFrame(f)
        post = [str(df) for df in dfif1.get_chunks(1)]
        self.assertEqual(post, ['<DFIDataFrame: shape=(5, 4)>'])




if __name__ == '__main__':
    import unittest
    unittest.main()

