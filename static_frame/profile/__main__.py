import io
import os
import argparse
import typing as tp
import fnmatch
import timeit
from time import sleep
import cProfile
import pstats
import sys
import datetime
import tempfile
from enum import Enum
import shutil


from pyinstrument import Profiler #type: ignore
from line_profiler import LineProfiler #type: ignore
import gprof2dot #type: ignore

import numpy as np
import pandas as pd
import frame_fixtures as ff
import static_frame as sf

from static_frame.core.display_color import HexColor
from static_frame.core.util import AnyCallable, isin
from static_frame.test.test_case import temp_file


class PerfStatus(Enum):
    EXPLAINED_WIN = (True, True)
    EXPLAINED_LOSS = (True, False)
    UNEXPLAINED_WIN = (False, True)
    UNEXPLAINED_LOSS = (False, False)

    def __str__(self) -> str:
        if self.value[0]:
            v = '✓' # make a check mark
        else:
            v = '?'
        if self.value[1]:
            return HexColor.format_terminal('darkgreen', str(v))
        return HexColor.format_terminal('darkorange', str(v))

class FunctionMetaData(tp.NamedTuple):
    line_target: tp.Optional[AnyCallable] = None
    perf_status: tp.Optional[PerfStatus] = None
    explanation: str = ''

class PerfKey: pass

class Perf(PerfKey):
    NUMBER = 100_000

    def __init__(self) -> None:
        self.meta: tp.Optional[tp.Dict[str, FunctionMetaData]] = None

    def iter_function_names(self, pattern_func: str = '') -> tp.Iterator[str]:
        for name in sorted(dir(self)):
            if name == 'iter_function_names':
                continue
            if pattern_func and not fnmatch.fnmatch(
                    name, pattern_func.lower()):
               continue
            if not name.startswith('_') and callable(getattr(self, name)):
                yield name


class Native(PerfKey): pass
class Reference(PerfKey): pass
class ReferenceMissing(Reference):
    '''For classes that do not / cannot run a reference component.
    '''



#-------------------------------------------------------------------------------

class IndexIterLabelApply(Perf):
    NUMBER = 200

    def __init__(self) -> None:
        super().__init__()


        self.sfi_int = ff.parse('s(100,1)|i(I,int)|c(I,int)').index
        self.pdi_int = self.sfi_int.to_pandas()

class IndexIterLabelApply_N(IndexIterLabelApply, Native):

    def index_int(self) -> None:
        self.sfi_int.iter_label().apply(lambda s: s * 10)

    def index_int_dtype(self) -> None:
        self.sfi_int.iter_label().apply(lambda s: s * 10, dtype=int)

class IndexIterLabelApply_R(IndexIterLabelApply, Reference):

    def index_int(self) -> None:
        # Pandas Index to not have an apply
        pd.Series(self.pdi_int).apply(lambda s: s * 10)

    def index_int_dtype(self) -> None:
        # Pandas Index to not have an apply
        pd.Series(self.pdi_int).apply(lambda s: s * 10)



#-------------------------------------------------------------------------------
class SeriesIsNa(Perf):
    NUMBER = 10_000

    def __init__(self) -> None:
        super().__init__()

        f = ff.parse('s(1000,3)|v(float,object,bool)')
        f = f.assign.loc[(f.index % 12 == 0), 0](np.nan)
        f = f.assign.loc[(f.index % 12 == 0), 1](None)

        self.sfs_float = f.iloc[:, 0]
        self.sfs_object = f.iloc[:, 1]
        self.sfs_bool = f.iloc[:, 2]

        self.pds_float = f.iloc[:, 0].to_pandas()
        self.pds_object = f.iloc[:, 1].to_pandas()
        self.pds_bool = f.iloc[:, 2].to_pandas()

        self.meta = {
            'float_index_auto': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_WIN,
                ),
            'object_index_auto': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_WIN,
                ),
            'bool_index_auto': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_WIN, # not copying anything
                ),
            }

class SeriesIsNa_N(SeriesIsNa, Native):

    def float_index_auto(self) -> None:
        self.sfs_float.isna()

    def object_index_auto(self) -> None:
        self.sfs_object.isna()

    def bool_index_auto(self) -> None:
        self.sfs_bool.isna()

class SeriesIsNa_R(SeriesIsNa, Reference):

    def float_index_auto(self) -> None:
        self.pds_float.isna()

    def object_index_auto(self) -> None:
        self.pds_object.isna()

    def bool_index_auto(self) -> None:
        self.pds_bool.isna()

#-------------------------------------------------------------------------------
class SeriesDropNa(Perf):
    NUMBER = 200

    def __init__(self) -> None:
        super().__init__()

        f1 = ff.parse('s(100_000,3)|v(float,object,bool)')
        f1 = f1.assign.loc[(f1.index % 12 == 0), 0](np.nan)
        f1 = f1.assign.loc[(f1.index % 12 == 0), 1](None)

        self.sfs_float_auto = f1.iloc[:, 0]
        self.sfs_object_auto = f1.iloc[:, 1]
        self.sfs_bool_auto = f1.iloc[:, 2]

        self.pds_float_auto = f1.iloc[:, 0].to_pandas()
        self.pds_object_auto = f1.iloc[:, 1].to_pandas()
        self.pds_bool_auto = f1.iloc[:, 2].to_pandas()


        f2 = ff.parse('s(100_000,3)|v(float,object,bool)|i(I,str)|c(I,str)')
        f2 = f2.assign.loc[f2.index.via_str.find('u') >= 0, sf.ILoc[0]](np.nan)
        f2 = f2.assign.loc[f2.index.via_str.find('u') >= 0, sf.ILoc[1]](None)

        self.sfs_float_str = f2.iloc[:, 0]
        self.sfs_object_str = f2.iloc[:, 1]
        self.sfs_bool_str = f2.iloc[:, 2]

        self.pds_float_str = f2.iloc[:, 0].to_pandas()
        self.pds_object_str = f2.iloc[:, 1].to_pandas()
        self.pds_bool_str = f2.iloc[:, 2].to_pandas()

        self.meta = {
            'float_index_auto': FunctionMetaData(
                line_target=sf.Index.__init__,
                perf_status=PerfStatus.EXPLAINED_LOSS,
                ),
            'object_index_auto': FunctionMetaData(
                line_target=sf.Series.dropna,
                perf_status=PerfStatus.EXPLAINED_LOSS,
                ),
            'bool_index_auto': FunctionMetaData(
                line_target=sf.Series.dropna,
                perf_status=PerfStatus.EXPLAINED_WIN, # not copying anything
                ),

            'float_index_str': FunctionMetaData(
                line_target=sf.Index.__init__,
                perf_status=PerfStatus.EXPLAINED_LOSS,
                ),
            'object_index_str': FunctionMetaData(
                line_target=sf.Series.dropna,
                perf_status=PerfStatus.EXPLAINED_LOSS,
                ),
            'bool_index_str': FunctionMetaData(
                line_target=sf.Series.dropna,
                perf_status=PerfStatus.EXPLAINED_WIN,
                )
            }

class SeriesDropNa_N(SeriesDropNa, Native):

    def float_index_auto(self) -> None:
        s = self.sfs_float_auto.dropna()
        assert 99999 in s

    def object_index_auto(self) -> None:
        s = self.sfs_object_auto.dropna()
        assert 99999 in s

    def bool_index_auto(self) -> None:
        s = self.sfs_bool_auto.dropna()
        assert 99999 in s


    def float_index_str(self) -> None:
        s = self.sfs_float_str.dropna()
        assert 'zDa2' in s

    def object_index_str(self) -> None:
        s = self.sfs_object_str.dropna()
        assert 'zDa2' in s

    def bool_index_str(self) -> None:
        s = self.sfs_bool_str.dropna()
        assert 'zDa2' in s


class SeriesDropNa_R(SeriesDropNa, Reference):

    def float_index_auto(self) -> None:
        s = self.pds_float_auto.dropna()
        assert 99999 in s

    def object_index_auto(self) -> None:
        s = self.pds_object_auto.dropna()
        assert 99999 in s

    def bool_index_auto(self) -> None:
        s = self.pds_bool_auto.dropna()
        assert 99999 in s


    def float_index_str(self) -> None:
        s = self.pds_float_str.dropna()
        assert 'zDa2' in s

    def object_index_str(self) -> None:
        s = self.pds_object_str.dropna()
        assert 'zDa2' in s

    def bool_index_str(self) -> None:
        s = self.pds_bool_str.dropna()
        assert 'zDa2' in s




#-------------------------------------------------------------------------------
class SeriesFillNa(Perf):
    NUMBER = 100

    def __init__(self) -> None:
        super().__init__()

        f1 = ff.parse('s(100_000,2)|v(float,object)|i(I,str)|c(I,str)')
        f1 = f1.assign.loc[f1.index.via_str.find('u') >= 0, sf.ILoc[0]](np.nan)
        f1 = f1.assign.loc[f1.index.via_str.find('u') >= 0, sf.ILoc[1]](None)

        self.sfs_float_str = f1.iloc[:, 0]
        self.sfs_object_str = f1.iloc[:, 1]

        self.pds_float_str = f1.iloc[:, 0].to_pandas()
        self.pds_object_str = f1.iloc[:, 1].to_pandas()

        from static_frame.core.util import isna_array

        self.meta = {
            'float_index_str': FunctionMetaData(
                line_target=isna_array,
                perf_status=PerfStatus.EXPLAINED_WIN,
                ),
            'object_index_str': FunctionMetaData(
                line_target=isna_array,
                perf_status=PerfStatus.EXPLAINED_LOSS,
                explanation='isna_array does two passes on object arrays',
                ),
            }

class SeriesFillNa_N(SeriesFillNa, Native):

    def float_index_str(self) -> None:
        s = self.sfs_float_str.fillna(0.0)
        assert 'zDa2' in s

    def object_index_str(self) -> None:
        s = self.sfs_object_str.fillna('')
        assert 'zDa2' in s


class SeriesFillNa_R(SeriesFillNa, Reference):

    def float_index_str(self) -> None:
        s = self.pds_float_str.fillna(0.0)
        assert 'zDa2' in s

    def object_index_str(self) -> None:
        s = self.pds_object_str.fillna('')
        assert 'zDa2' in s




#-------------------------------------------------------------------------------
class SeriesDropDuplicated(Perf):
    NUMBER = 500

    def __init__(self) -> None:
        super().__init__()

        f = ff.parse('s(1000,3)|v(float,object,bool)|i(I,str)|c(I,str)')

        self.sfs_float = f.iloc[:, 0]
        self.sfs_float = self.sfs_float.assign.iloc[np.arange(len(self.sfs_float)) % 12 == 0](1.0)
        self.sfs_object = f.iloc[:, 1]
        self.sfs_object = self.sfs_object.assign.iloc[np.arange(len(self.sfs_object)) % 12 == 0](None)
        self.sfs_bool = f.iloc[:, 2]

        self.pds_float = self.sfs_float.to_pandas()
        self.pds_object = self.sfs_object.to_pandas()
        self.pds_bool = self.sfs_bool.to_pandas()


        self.meta = {
            'float_index_str': FunctionMetaData(
                line_target=sf.Index.__init__,
                perf_status=PerfStatus.EXPLAINED_WIN,
                ),
            'object_index_str': FunctionMetaData(
                line_target=sf.Series.drop_duplicated,
                perf_status=PerfStatus.EXPLAINED_WIN,
                ),
            'bool_index_str': FunctionMetaData(
                line_target=sf.Series.drop_duplicated,
                perf_status=PerfStatus.EXPLAINED_WIN,
                ),
            }

class SeriesDropDuplicated_N(SeriesDropDuplicated, Native):

    def float_index_str(self) -> None:
        s = self.sfs_float.drop_duplicated()
        assert 'zDr0' in s

    def object_index_str(self) -> None:
        s = self.sfs_object.drop_duplicated()
        assert 'zDr0' in s

    def bool_index_str(self) -> None:
        self.sfs_bool.drop_duplicated()

class SeriesDropDuplicated_R(SeriesDropDuplicated, Reference):

    def float_index_str(self) -> None:
        s = self.pds_float.drop_duplicates(keep=False)
        assert 'zDr0' in s

    def object_index_str(self) -> None:
        s = self.pds_object.drop_duplicates(keep=False)
        assert 'zDr0' in s

    def bool_index_str(self) -> None:
        self.pds_bool.drop_duplicates(keep=False)




#-------------------------------------------------------------------------------
class SeriesIterElementApply(Perf):
    NUMBER = 500

    def __init__(self) -> None:
        super().__init__()

        f = ff.parse('s(1000,3)|v(float,object,bool)|i(I,str)|c(I,str)')

        self.sfs_float = f.iloc[:, 0]
        self.sfs_object = f.iloc[:, 1]
        self.sfs_bool = f.iloc[:, 2]

        self.pds_float = f.iloc[:, 0].to_pandas()
        self.pds_object = f.iloc[:, 1].to_pandas()
        self.pds_bool = f.iloc[:, 2].to_pandas()


        from static_frame.core.util import prepare_iter_for_array

        self.meta = {
            'float_index_str': FunctionMetaData(
                line_target=prepare_iter_for_array,
                perf_status=PerfStatus.EXPLAINED_LOSS,
                explanation='prepare_iter_for_array() appears to be the biggest cost'
                ),
            'object_index_str': FunctionMetaData(
                line_target=prepare_iter_for_array,
                perf_status=PerfStatus.EXPLAINED_LOSS,
                explanation='prepare_iter_for_array() appears to be the biggest cost'
                ),
            'bool_index_str': FunctionMetaData(
                line_target=prepare_iter_for_array,
                perf_status=PerfStatus.EXPLAINED_LOSS, # not copying anything
                explanation='prepare_iter_for_array() appears to be the biggest cost'
                ),
            }

class SeriesIterElementApply_N(SeriesIterElementApply, Native):

    def float_index_str(self) -> None:
        self.sfs_float.iter_element().apply(lambda x: str(x))

    def object_index_str(self) -> None:
        self.sfs_object.iter_element().apply(lambda x: str(x))

    def bool_index_str(self) -> None:
        self.sfs_bool.iter_element().apply(lambda x: str(x))

class SeriesIterElementApply_R(SeriesIterElementApply, Reference):

    def float_index_str(self) -> None:
        self.pds_float.apply(lambda x: str(x))

    def object_index_str(self) -> None:
        self.pds_object.apply(lambda x: str(x))

    def bool_index_str(self) -> None:
        self.pds_bool.apply(lambda x: str(x))






#-------------------------------------------------------------------------------
class FrameDropNa(Perf):
    NUMBER = 100

    def __init__(self) -> None:
        super().__init__()

        f1 = ff.parse('s(100,100)|v(float)')
        f1 = f1.assign.loc[(f1.index % 12 == 0),:](np.nan)
        self.sff_float_auto_row = f1
        self.pdf_float_auto_row = f1.to_pandas()

        f2 = ff.parse('s(100,100)|v(float)')
        f2 = f2.assign.loc[:, (f2.columns % 12 == 0)](np.nan)
        self.sff_float_auto_column = f2
        self.pdf_float_auto_column = f2.to_pandas()


        f3 = ff.parse('s(100,100)|v(float)|i(I,str)|c(I,str)')
        f3 = f3.assign.loc[(np.arange(len(f3.index)) % 12 == 0),:](np.nan)
        self.sff_float_str_row = f3
        self.pdf_float_str_row = f3.to_pandas()

        f4 = ff.parse('s(100,100)|v(float)|i(I,str)|c(I,str)')
        f4 = f4.assign.loc[:, (np.arange(len(f4.columns)) % 12 == 0)](np.nan)
        self.sff_float_str_column = f4
        self.pdf_float_str_column = f4.to_pandas()

        self.meta = {
            'float_index_auto_row': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_WIN,
                ),
            'float_index_auto_column': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_WIN,
                ),
            'float_index_str_row': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_WIN, # not copying anything
                ),
            'float_index_str_column': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_WIN, # not copying anything
                ),
            }

class FrameDropNa_N(FrameDropNa, Native):

    def float_index_auto_row(self) -> None:
        self.sff_float_auto_row.dropna()

    def float_index_auto_column(self) -> None:
        self.sff_float_auto_column.dropna(axis=1)


    def float_index_str_row(self) -> None:
        self.sff_float_str_row.dropna()

    def float_index_str_column(self) -> None:
        self.sff_float_str_column.dropna(axis=1)


class FrameDropNa_R(FrameDropNa, Reference):

    def float_index_auto_row(self) -> None:
        self.pdf_float_auto_row.dropna()

    def float_index_auto_column(self) -> None:
        self.pdf_float_auto_column.dropna(axis=1)


    def float_index_str_row(self) -> None:
        self.pdf_float_str_row.dropna()

    def float_index_str_column(self) -> None:
        self.pdf_float_str_column.dropna(axis=1)

#-------------------------------------------------------------------------------

class FrameILoc(Perf):

    def __init__(self) -> None:
        super().__init__()

        self.sff1 = ff.parse('s(100,100)')
        self.pdf1 = pd.DataFrame(self.sff1.values)

        self.sff2 = ff.parse('s(100,100)|i(I,str)|c(I,str)')
        self.pdf2 = self.sff2.to_pandas()

        self.meta = {
            'element_index_auto': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_WIN,
                ),
            'element_index_str': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_WIN,
                ),
            }

class FrameILoc_N(FrameILoc, Native):

    def element_index_auto(self) -> None:
        self.sff1.iloc[50, 50]

    def element_index_str(self) -> None:
        self.sff2.iloc[50, 50]

class FrameILoc_R(FrameILoc, Reference):

    def element_index_auto(self) -> None:
        self.pdf1.iloc[50, 50]

    def element_index_str(self) -> None:
        self.pdf2.iloc[50, 50]

#-------------------------------------------------------------------------------

class FrameLoc(Perf):

    def __init__(self) -> None:
        super().__init__()

        self.sff1 = ff.parse('s(100,100)')
        self.pdf1 = pd.DataFrame(self.sff1.values)

        self.sff2 = ff.parse('s(100,100)|i(I,str)|c(I,str)')
        self.pdf2 = self.sff2.to_pandas()

        self.meta = {
            'element_index_auto': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_WIN,
                ),
            'element_index_str': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_WIN,
                ),
            }

class FrameLoc_N(FrameLoc, Native):

    def element_index_auto(self) -> None:
        self.sff1.loc[50, 50]

    def element_index_str(self) -> None:
        self.sff2.loc['zGuv', 'zGuv']

class FrameLoc_R(FrameLoc, Reference):

    def element_index_auto(self) -> None:
        self.pdf1.loc[50, 50]

    def element_index_str(self) -> None:
        self.pdf2.loc['zGuv', 'zGuv']


#-------------------------------------------------------------------------------

class FrameIterSeriesApply(Perf):
    NUMBER = 50

    def __init__(self) -> None:
        super().__init__()


        self.sff_float = ff.parse('s(1000,1000)|i(I,str)|c(I,int)')
        self.pdf_float = self.sff_float.to_pandas()

        self.sff_mixed = ff.parse('s(1000,1000)|v(int,float,bool,str)|i(I,str)|c(I,int)')
        self.pdf_mixed = self.sff_mixed.to_pandas()

        from static_frame.core.type_blocks import TypeBlocks
        from static_frame.core.util import iterable_to_array_1d
        from static_frame.core.util import prepare_iter_for_array

        self.meta = {
            'float_index_str_row': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_LOSS,
                line_target=prepare_iter_for_array,
                explanation='appears to all be in apply_iter gen exp'
                ),
            'float_index_str_row_dtype': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_LOSS,
                ),
            'float_index_str_column': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_WIN,
                ),
            'float_index_str_column_dtype': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_WIN,
                ),
            'mixed_index_str_row': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_LOSS,
                line_target=TypeBlocks._blocks_to_array,
                explanation='possible improvement with _blocks_to_array in C'
                ),
            'mixed_index_str_row_dtype': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_LOSS,
                line_target=iterable_to_array_1d
                ),
            'mixed_index_str_column': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_WIN,
                ),
            'mixed_index_str_column_dtype': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_WIN,
                ),
            }

class FrameIterSeriesApply_N(FrameIterSeriesApply, Native):

    def float_index_str_row(self) -> None:
        s = self.sff_float.iter_series(axis=1).apply(lambda s: s.mean())
        assert 'zwVN' in s.index

    def float_index_str_row_dtype(self) -> None:
        s = self.sff_float.iter_series(axis=1).apply(lambda s: s.mean(), dtype=float)
        assert 'zwVN' in s.index


    def float_index_str_column(self) -> None:
        s = self.sff_float.iter_series(axis=0).apply(lambda s: s.mean())
        assert -149082 in s.index

    def float_index_str_column_dtype(self) -> None:
        s = self.sff_float.iter_series(axis=0).apply(lambda s: s.mean(), dtype=float)
        assert -149082 in s.index


    def mixed_index_str_row(self) -> None:
        s = self.sff_mixed.iter_series(axis=1).apply(lambda s: s.iloc[-1])
        assert 'zwVN' in s.index

    def mixed_index_str_row_dtype(self) -> None:
        s = self.sff_mixed.iter_series(axis=1).apply(lambda s: s.iloc[-1], dtype=str)
        assert 'zwVN' in s.index


    def mixed_index_str_column(self) -> None:
        s = self.sff_mixed.iter_series(axis=0).apply(lambda s: s.iloc[-1])
        assert -149082 in s.index

    def mixed_index_str_column_dtype(self) -> None:
        s = self.sff_mixed.iter_series(axis=0).apply(lambda s: s.iloc[-1], dtype=str)
        assert -149082 in s.index



class FrameIterSeriesApply_R(FrameIterSeriesApply, Reference):

    def float_index_str_row(self) -> None:
        s = self.pdf_float.apply(lambda s: s.mean(), axis=1)
        assert 'zwVN' in s.index

    def float_index_str_row_dtype(self) -> None:
        s = self.pdf_float.apply(lambda s: s.mean(), axis=1)
        assert 'zwVN' in s.index


    def float_index_str_column(self) -> None:
        s = self.pdf_float.apply(lambda s: s.mean(), axis=0)
        assert -149082 in s.index

    def float_index_str_column_dtype(self) -> None:
        s = self.pdf_float.apply(lambda s: s.mean(), axis=0)
        assert -149082 in s.index


    def mixed_index_str_row(self) -> None:
        s = self.pdf_mixed.apply(lambda s: s.iloc[-1], axis=1)
        assert 'zwVN' in s.index

    def mixed_index_str_row_dtype(self) -> None:
        s = self.pdf_mixed.apply(lambda s: s.iloc[-1], axis=1)
        assert 'zwVN' in s.index


    def mixed_index_str_column(self) -> None:
        s = self.pdf_mixed.apply(lambda s: s.iloc[-1], axis=0)
        assert -149082 in s.index

    def mixed_index_str_column_dtype(self) -> None:
        s = self.pdf_mixed.apply(lambda s: s.iloc[-1], axis=0)
        assert -149082 in s.index

#-------------------------------------------------------------------------------
class FrameIterGroupApply(Perf):
    NUMBER = 1000

    def __init__(self) -> None:
        super().__init__()

        self.sff_int_index_str = ff.parse('s(1000,10)|v(int)|i(I,str)|c(I,str)').assign[sf.ILoc[0]].apply(lambda s: s % 10).assign[sf.ILoc[1]].apply(lambda s: s % 2)
        self.pdf_int_index_str = self.sff_int_index_str.to_pandas()


        self.sff_str_index_str = ff.parse('s(1000,10)|v(str)|i(I,str)|c(I,str)').assign[
                sf.ILoc[0]].apply(lambda s: s.iter_element().apply(
                        lambda e: chr(ord(e[3]) % 10 + 97))).assign[
                sf.ILoc[1]].apply(lambda s: s.iter_element().apply(
                        lambda e: chr(ord(e[3]) % 2 + 97)))

        self.pdf_str_index_str = self.sff_str_index_str.to_pandas()


        from static_frame.core.type_blocks import TypeBlocks
        # from static_frame.core.util import iterable_to_array_1d
        # from static_frame.core.util import prepare_iter_for_array

        self.meta = {
            'int_index_str_double': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_LOSS,
                ),
            }

class FrameIterGroupApply_N(FrameIterGroupApply, Native):

    def int_index_str_single(self) -> None:
        self.sff_int_index_str.iter_group('zZbu').apply(lambda f: len(f))

    def int_index_str_double(self) -> None:
        self.sff_int_index_str.iter_group(['zZbu', 'ztsv']).apply(lambda f: len(f))


    def str_index_str_single(self) -> None:
        self.sff_str_index_str.iter_group('zZbu').apply(lambda f: len(f))

    def str_index_str_double(self) -> None:
        self.sff_str_index_str.iter_group(['zZbu', 'ztsv']).apply(lambda f: len(f))


class FrameIterGroupApply_R(FrameIterGroupApply, Reference):

    def int_index_str_single(self) -> None:
        self.pdf_int_index_str.groupby('zZbu').apply(lambda f: len(f))

    def int_index_str_double(self) -> None:
        # NOTE: this produces a hierarchical index
        self.pdf_int_index_str.groupby(['zZbu', 'ztsv']).apply(lambda f: len(f))


    def str_index_str_single(self) -> None:
        self.pdf_str_index_str.groupby('zZbu').apply(lambda f: len(f))

    def str_index_str_double(self) -> None:
        # NOTE: this produces a hierarchical index
        self.pdf_str_index_str.groupby(['zZbu', 'ztsv']).apply(lambda f: len(f))


#-------------------------------------------------------------------------------
class Pivot(Perf):
    NUMBER = 150

    def __init__(self) -> None:
        super().__init__()

        self.sff1 = ff.parse('s(100_000,10)|v(int,str,bool)|c(I,str)|i(I,int)')
        self.pdf1 = self.sff1.to_pandas()

        # narrow eav table
        self.sff2 = ff.parse('s(100_000,3)|v(int,int,int)').assign[0].apply(
                lambda s: s % 6).assign[1].apply(
                lambda s: s % 12
                )
        self.pdf2 = self.sff2.to_pandas()

        # more than one data fields
        self.sff3 = ff.parse('s(100_000,4)|v(int,int,int,int)').assign[0].apply(
                lambda s: s % 6).assign[1].apply(
                lambda s: s % 12
                )

        from static_frame.core.pivot import pivot_outer_index
        from static_frame import TypeBlocks
        from static_frame.core.util import array_to_groups_and_locations
        self.meta = {
            'index1_columns0_data2': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_WIN,
                line_target=array_to_groups_and_locations,
                ),
            'index1_columns1_data1': FunctionMetaData(
                line_target=pivot_outer_index,
                perf_status=PerfStatus.EXPLAINED_LOSS,
                ),
            }

class Pivot_N(Pivot, Native):

    def index1_columns0_data2(self) -> None:
        post = self.sff1.pivot(index_fields='zUvW', data_fields=('zZbu', 'zkuW'))
        assert post.shape == (2, 2)

    def index1_columns1_data1(self) -> None:
        post = self.sff2.pivot(index_fields=0, columns_fields=1)
        assert post.shape == (6, 12)


class Pivot_R(Pivot, Reference):

    def index1_columns0_data2(self) -> None:
        post = self.pdf1.pivot_table(index='zUvW', values=('zZbu', 'zkuW'), aggfunc=np.nansum)
        assert post.shape == (2, 2)

    def index1_columns1_data1(self) -> None:
        post = self.pdf2.pivot_table(index=0, columns=1, aggfunc=np.nansum)
        assert post.shape == (6, 12)

#-------------------------------------------------------------------------------
class BusItemsZipPickle(Perf):
    NUMBER = 1

    def __init__(self) -> None:
        super().__init__()

        def items() -> tp.Iterator[tp.Tuple[str, sf.Frame]]:
            f = ff.parse(f's(2,2)|v(int)|i(I,str)|c(I,str)')
            for i in range(10_000):
                yield str(i), f

        frames = sf.Series.from_items(items(), dtype=object)
        _, self.fp = tempfile.mkstemp(suffix='.zip')
        b1 = sf.Bus(frames)
        b1.to_zip_pickle(self.fp)

        # self.meta = {
        #     'int_index_str_double': FunctionMetaData(
        #         perf_status=PerfStatus.EXPLAINED_LOSS,
        #         None
        #         ),
        #     }

    def __del__(self) -> None:
        os.unlink(self.fp)

class BusItemsZipPickle_N(BusItemsZipPickle, Native):

    def int_index_str(self) -> None:
        bus = sf.Bus.from_zip_pickle(self.fp, max_persist=100)
        for label, frame in bus.items():
           assert frame.shape[0] == 2

class BusItemsZipPickle_R(BusItemsZipPickle, ReferenceMissing):

    def int_index_str(self) -> None:
        pass

#-------------------------------------------------------------------------------
class FrameToParquet(Perf):
    NUMBER = 4

    def __init__(self) -> None:
        super().__init__()
        _, self.fp = tempfile.mkstemp(suffix='.zip')

        self.sff1 = ff.parse('s(10,10_000)|v(int,int,bool,float,float)|i(I,str)|c(I,str)')
        self.pdf1 = self.sff1.to_pandas()

        self.sff2 = ff.parse('s(10_000,10)|v(int,int,bool,float,float)|i(I,str)|c(I,str)')
        self.pdf2 = self.sff2.to_pandas()


        # self.meta = {
        #     'int_index_str_double': FunctionMetaData(
        #         perf_status=PerfStatus.EXPLAINED_LOSS,
        #         None
        #         ),
        #     }

    def __del__(self) -> None:
        os.unlink(self.fp)

class FrameToParquet_N(FrameToParquet, Native):

    def write_wide_mixed_index_str(self) -> None:
        self.sff1.to_parquet(self.fp)

    def write_tall_mixed_index_str(self) -> None:
        self.sff2.to_parquet(self.fp)


class FrameToParquet_R(FrameToParquet, Reference):

    def write_wide_mixed_index_str(self) -> None:
        self.pdf1.to_parquet(self.fp)

    def write_tall_mixed_index_str(self) -> None:
        self.pdf2.to_parquet(self.fp)



#-------------------------------------------------------------------------------
class FrameToNPZ(Perf):
    NUMBER = 1

    def __init__(self) -> None:
        super().__init__()
        _, self.fp = tempfile.mkstemp(suffix='.zip')

        self.sff1 = ff.parse('s(10,10_000)|v(int,bool,float)|i(I,str)|c(I,str)')

        # self.meta = {
        #     'int_index_str_double': FunctionMetaData(
        #         perf_status=PerfStatus.EXPLAINED_LOSS,
        #         None
        #         ),
        #     }

    def __del__(self) -> None:
        os.unlink(self.fp)

class FrameToNPZ_N(FrameToNPZ, Native):

    def wide_mixed_index_str(self) -> None:
        self.sff1.to_npz(self.fp)

class FrameToNPZ_R(FrameToNPZ, Reference):

    # NOTE: benchmark is SF to_parquet
    def wide_mixed_index_str(self) -> None:
        self.sff1.to_parquet(self.fp)


class FrameFromNPZ(Perf):
    NUMBER = 1

    def __init__(self) -> None:
        super().__init__()

        self.sff1 = ff.parse('s(10,10_000)|v(int,bool,float)|i(I,str)|c(I,str)')
        _, self.fp_npz = tempfile.mkstemp(suffix='.zip')
        self.sff1.to_npz(self.fp_npz)

        _, self.fp_parquet = tempfile.mkstemp(suffix='.parquet')
        self.sff1.to_parquet(self.fp_parquet)

        # self.meta = {
        #     'int_index_str_double': FunctionMetaData(
        #         perf_status=PerfStatus.EXPLAINED_LOSS,
        #         None
        #         ),
        #     }

    def __del__(self) -> None:
        os.unlink(self.fp_npz)
        os.unlink(self.fp_parquet)

class FrameFromNPZ_N(FrameFromNPZ, Native):

    def wide_mixed_index_str(self) -> None:
        sf.Frame.from_npz(self.fp_npz)

class FrameFromNPZ_R(FrameFromNPZ, Reference):

    # NOTE: benchmark is SF from_parquet
    def wide_mixed_index_str(self) -> None:
        sf.Frame.from_parquet(self.fp_parquet)


#-------------------------------------------------------------------------------
class Group(Perf):
    NUMBER = 150

    def __init__(self) -> None:
        super().__init__()

        self.sff1 = ff.parse('s(100_000,10)|v(int,str,bool)|c(I,str)|i(I,int)')
        self.pdf1 = self.sff1.to_pandas()

        # narrow eav table
        self.sff2 = ff.parse('s(100_000,3)|v(int,int,int)').assign[0].apply(
                lambda s: s % 6).assign[1].apply(
                lambda s: s % 100
                )
        self.pdf2 = self.sff2.to_pandas()

        from static_frame import Frame
        from static_frame import TypeBlocks
        from static_frame.core.util import array_to_groups_and_locations
        self.meta = {
            'wide_group_2': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_LOSS,
                line_target=Frame._axis_group_iloc_items,
                explanation='nearly identical, favoring slower'
                ),
            'tall_group_100': FunctionMetaData(
                perf_status=PerfStatus.EXPLAINED_LOSS,
                line_target=Frame._axis_group_iloc_items,
                ),
            }

class Group_N(Group, Native):

    def wide_group_2(self) -> None:
        post = tuple(self.sff1.iter_group_items('zUvW'))
        assert len(post) == 2

    def tall_group_100(self) -> None:
        post = tuple(self.sff2.iter_group_items(1))
        assert len(post) == 100


class Group_R(Group, Reference):

    def wide_group_2(self) -> None:
        post = tuple(self.pdf1.groupby('zUvW'))
        assert len(post) == 2

    def tall_group_100(self) -> None:
        post = tuple(self.pdf2.groupby(1))
        assert len(post) == 100




#-------------------------------------------------------------------------------
class FrameFromConcat(Perf):
    NUMBER = 50

    def __init__(self) -> None:
        super().__init__()
        self.tall_mixed_sff1 = [
            ff.parse('s(10_000,10)|v(int,str,int,bool)')
            for _ in range(20)
            ]
        self.tall_mixed_pdf1 = [f.to_pandas() for f in self.tall_mixed_sff1]


        self.tall_uniform_sff1 = [
            ff.parse('s(10_000,10)|v(float)')
            for _ in range(20)
            ]
        self.tall_uniform_pdf1 = [f.to_pandas() for f in self.tall_uniform_sff1]


        from static_frame import Frame
        # from static_frame import TypeBlocks
        # from static_frame.core.util import array_to_groups_and_locations
        # self.meta = {
        #     'tall_uniform_20': FunctionMetaData(
        #         # perf_status=PerfStatus.EXPLAINED_LOSS,
        #         line_target=Frame.from_concat,
        #         # explanation='nearly identical, favoring slower'
        #         ),
        #     }

class FrameFromConcat_N(FrameFromConcat, Native):

    def tall_mixed_20(self) -> None:
        f = sf.Frame.from_concat(self.tall_mixed_sff1, index=sf.IndexAutoFactory)
        assert f.shape == (200_000, 10)

    def tall_uniform_20(self) -> None:
        f = sf.Frame.from_concat(self.tall_uniform_sff1, index=sf.IndexAutoFactory)
        assert f.shape == (200_000, 10)

class FrameFromConcat_R(FrameFromConcat, Reference):

    def tall_mixed_20(self) -> None:
        df = pd.concat(self.tall_mixed_pdf1)
        assert df.shape == (200_000, 10)

    def tall_uniform_20(self) -> None:
        df = pd.concat(self.tall_uniform_pdf1)
        assert df.shape == (200_000, 10)


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def get_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
            description='Performance testing and profiling',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='''Example:

Performance comparison of all dropna tests:

python3 test_performance.py '*dropna' --performance

Profiling outpout for static-frame dropna:

python3 test_performance.py SeriesIntFloat_dropna --profile
            '''
            )
    p.add_argument('patterns',
            help='Names of classes to match using fn_match syntax',
            nargs='+',
            )
    # p.add_argument('--modules',
    #         help='Names of modules to find tests',
    #         nargs='+',
    #         default=('core',),
    #         )
    p.add_argument('--profile',
            help='Turn on profiling with cProfile',
            action='store_true',
            default=False,
            )
    p.add_argument('--graph',
            help='Produce a call graph of cProfile output',
            action='store_true',
            default=False,
            )
    p.add_argument('--instrument',
            help='Turn on instrumenting with pyinstrument',
            action='store_true',
            default=False,
            )
    p.add_argument('--performance',
            help='Turn on performance measurements',
            action='store_true',
            default=False,
            )
    p.add_argument('--line',
            help='Turn on line profiler',
            action='store_true',
            default=False,
            )
    return p


def yield_classes(
        pattern: str
        ) -> tp.Iterator[
                tp.Tuple[
                    tp.Dict[tp.Type[PerfKey], tp.Type[PerfKey]],
                    str]]:

    if '.' in pattern:
        pattern_cls, pattern_func = pattern.split('.')
    else:
        pattern_cls, pattern_func = pattern, '*'

    for cls_perf in Perf.__subclasses__(): # only get one level
        if pattern_cls and not fnmatch.fnmatch(
                cls_perf.__name__.lower(), pattern_cls.lower()):
            continue

        runners: tp.Dict[tp.Type[PerfKey], tp.Type[PerfKey]] = {Perf: cls_perf}

        for cls_runner in cls_perf.__subclasses__():
            for cls in (Native, Reference):
                if issubclass(cls_runner, cls):
                    runners[cls] = cls_runner
                    break
        yield runners, pattern_func


def profile(
        cls_runner: tp.Type[Perf],
        pattern_func: str,
        ) -> None:
    '''
    Profile the `sf` function from the supplied class.
    '''
    runner = cls_runner()
    for name in runner.iter_function_names(pattern_func):
        f = getattr(runner, name)
        pr = cProfile.Profile()

        pr.enable()
        for _ in range(runner.NUMBER):
            f()
        pr.disable()

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())

def graph(
        cls_runner: tp.Type[Perf],
        pattern_func: str,
        threshold_edge: float = 0.1,
        threshold_node: float = 0.5,
        ) -> None:
    '''
    Profile the `sf` function from the supplied class.
    '''
    runner = cls_runner()
    for name in runner.iter_function_names(pattern_func):
        _, fp = tempfile.mkstemp(suffix='', text=True)
        fp_pstat = fp + '.pstat'
        fp_dot = fp + '.dot'
        fp_png = fp + '.png'

        f = getattr(runner, name)
        pr = cProfile.Profile()

        pr.enable()
        for _ in range(runner.NUMBER):
            f()
        pr.disable()

        ps = pstats.Stats(pr)
        ps.dump_stats(fp_pstat)

        gprof2dot.main([
            '--format', 'pstats',
            '--output', fp_dot,
            '--edge-thres', threshold_edge, # 0.1 default
            '--node-thres', threshold_node, # 0.5 default
            fp_pstat
        ])
        os.system(f'dot {fp_dot} -Tpng -Gdpi=300 -o {fp_png}; eog {fp_png} &')

def instrument(
        cls_runner: tp.Type[Perf],
        pattern_func: str,
        timeline: bool = False,
        ) -> None:
    '''
    Profile the `sf` function from the supplied class.
    '''
    runner = cls_runner()
    for name in runner.iter_function_names(pattern_func):
        f = getattr(runner, name)
        profiler = Profiler(interval=0.0001) # default is 0.001, 1 ms

        if timeline:
            profiler.start()
            f()
            profiler.stop()
        else:
            profiler.start()
            for _ in range(runner.NUMBER):
                f()
            profiler.stop()

        print(profiler.output_text(unicode=True, color=True, timeline=timeline, show_all=True))

def line(
        cls_runner: tp.Type[Perf],
        pattern_func: str,
        ) -> None:
    runner = cls_runner()
    for name in runner.iter_function_names(pattern_func):
        f = getattr(runner, name)
        profiler = LineProfiler()
        if not runner.meta:
            raise NotImplementedError('must define runner.meta')
        profiler.add_function(runner.meta[name].line_target)
        profiler.enable()
        f()
        profiler.disable()
        profiler.print_stats()

#-------------------------------------------------------------------------------

PerformanceRecord = tp.MutableMapping[str,
        tp.Union[str, float, bool, tp.Optional[PerfStatus]]]

def performance(
        bundle: tp.Dict[tp.Type[PerfKey], tp.Type[PerfKey]],
        pattern_func: str,
        ) -> tp.Iterator[PerformanceRecord]:

    cls_perf = bundle[Perf]
    assert issubclass(cls_perf, Perf)

    cls_native = bundle[Native]
    cls_reference = bundle[Reference]

    # TODO: check native/ref have the same  iterations
    runner_n = cls_native()
    runner_r = cls_reference()
    assert isinstance(runner_n, Perf)

    for func_name in runner_n.iter_function_names(pattern_func):
        row: PerformanceRecord = {}
        row['name'] = f'{cls_perf.__name__}.{func_name}'
        row['iterations'] = cls_perf.NUMBER

        for label, runner in ((Native, runner_n), (Reference, runner_r)):
            if isinstance(runner, ReferenceMissing):
                row[label.__name__] = np.nan
            else:
                row[label.__name__] = timeit.timeit(
                        f'runner.{func_name}()',
                        globals=locals(),
                        number=cls_perf.NUMBER)

        row['n/r'] = row[Native.__name__] / row[Reference.__name__] #type: ignore
        row['r/n'] = row[Reference.__name__] / row[Native.__name__] #type: ignore
        row['win'] = row['r/n'] > .99 if not np.isnan(row['r/n']) else True #type: ignore

        if runner_n.meta is not None and func_name in runner_n.meta:
            row['status'] = runner_n.meta[func_name].perf_status
            row['explanation'] = runner_n.meta[func_name].explanation
        else:
            row['status'] = (PerfStatus.UNEXPLAINED_WIN if row['win']
                    else PerfStatus.UNEXPLAINED_LOSS)
            row['explanation'] = ''

        yield row


def performance_tables_from_records(
        records: tp.Iterable[PerformanceRecord],
        ) -> tp.Tuple[sf.Frame, sf.Frame]:

    name_root_last = None
    name_root_count = 0

    def format(key: tp.Tuple[tp.Any, str], v: object) -> str:
        nonlocal name_root_last
        nonlocal name_root_count

        if isinstance(v, float):
            if np.isnan(v):
                return ''
            return str(round(v, 4))
        if isinstance(v, (bool, np.bool_)):
            if v:
                return HexColor.format_terminal('green', str(v))
            return HexColor.format_terminal('orange', str(v))
        if isinstance(v, str):
            if key[1] == 'explanation':
                return HexColor.format_terminal('gray', v)
            if key[1] == 'name':
                name_root = v.split('.')[0]
                if name_root != name_root_last:
                    name_root_last = name_root
                    name_root_count += 1
                if name_root_count % 2:
                    return HexColor.format_terminal('lavender', v)
                return HexColor.format_terminal('lightslategrey', v)
        return str(v)

    frame = sf.FrameGO.from_dict_records(records)

    fields = ['Native', 'Reference', 'n/r', 'r/n']
    stats = sf.Frame.from_concat((
            frame[fields].min().rename('min'),
            frame[fields].max().rename('max'),
            frame[fields].mean().rename('mean'),
            frame[fields].median().rename('median'),
            frame[fields].std(ddof=1).rename('std')
            )).rename(index='name').unset_index()
    # import ipdb; ipdb.set_trace()
    composit = sf.Frame.from_concat((frame, stats), columns=frame.columns, index=sf.IndexAutoFactory)
    display = composit.iter_element_items().apply(format)
    # display = display[display.columns.drop.loc['status'].values.tolist() + ['status']]
    # display = display[[c for c in display.columns if '/' not in c]]
    return frame, display

def main() -> None:

    options = get_arg_parser().parse_args()
    records: tp.List[PerformanceRecord] = []

    for pattern in options.patterns:
        for bundle, pattern_func in yield_classes(pattern):
            if options.performance:
                records.extend(performance(bundle, pattern_func))
            if options.profile:
                profile(bundle[Native], pattern_func) #type: ignore
            if options.graph:
                graph(bundle[Native], pattern_func) #type: ignore
            if options.instrument:
                instrument(bundle[Native], pattern_func) #type: ignore
            if options.line:
                line(bundle[Native], pattern_func) #type: ignore

    itemize = False # make CLI option maybe

    if records:

        from static_frame import DisplayConfig

        print(str(datetime.datetime.now()))

        pairs = []
        pairs.append(('python', sys.version.split(' ')[0]))
        for package in (np, pd, sf):
            pairs.append((package.__name__, package.__version__))
        print('|'.join(':'.join(pair) for pair in pairs))

        frame, display = performance_tables_from_records(records)

        config = DisplayConfig(
                cell_max_width_leftmost=np.inf,
                cell_max_width=np.inf,
                type_show=False,
                display_rows=200,
                include_index=False,
                )
        print(display.display(config))

        if itemize:
            alt = display.T
            for c in alt.columns:
                print(c)
                print(alt[c].sort_values().display(config))

        # import ipdb; ipdb.set_trace()
        # if 'sf/pd' in frame.columns:
        #     print('mean: {}'.format(round(frame['sf/pd'].mean(), 6)))
        #     print('wins: {}/{}'.format((frame['sf/pd'] < 1.05).sum(), len(frame)))



if __name__ == '__main__':
    main()
