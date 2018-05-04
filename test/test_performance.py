import timeit
import cProfile
import pstats
import io
import collections
import typing as tp
import itertools as it
import argparse
import string
import hashlib
import fnmatch

import pandas as pd
import numpy as np

import static_frame as sf


#-------------------------------------------------------------------------------

def get_sample_series_float(size=10000):
    a1 = np.arange(size) * .001
    a1[size // 2:] = np.nan
    pds = pd.Series(a1)
    sfs = sf.Series(a1)
    return pds, sfs, a1

pds_int_float_10k, sfs_int_float_10k, npa_int_float_10k = get_sample_series_float(10000)



def get_sample_series_string_index_float_values(size=10000):
    a1 = np.arange(size) * .001
    a1[size // 2:] = np.nan
    # create hsa indices
    index = [hashlib.sha224(str(x).encode('utf-8')).hexdigest() for x in range(size)]
    pds = pd.Series(a1, index=index)
    sfs = sf.Series(a1, index=index)
    return pds, sfs, a1

pds_str_float_10k, sfs_str_float_10k, _ = get_sample_series_string_index_float_values(10000)



def get_sample_series_obj(size=10000):
    sample = [None, 3, 0.123, np.nan]
    a1 = np.array(sample * int(size / len(sample)))

    pds = pd.Series(a1)
    sfs = sf.Series(a1)

    return pds, sfs, a1

pds_obj_10k, sfs_obj_10k, npa_obj_10k = get_sample_series_obj(10000)


def get_sample_series_objstr(size=10000):
    sample = [None, 3, 0.123, np.nan, 'str']
    a1 = np.array(sample * int(size / len(sample)))

    pds = pd.Series(a1)
    sfs = sf.Series(a1)

    return pds, sfs, a1

pds_objstr_10k, sfs_objstr_10k, npa_objstr_10k = get_sample_series_objstr(10000)


#-------------------------------------------------------------------------------
# frame generators


def get_sample_frame_float_string_index(size=10000, columns=100):
    a1 = (np.arange(size * columns)).reshape((size, columns)) * .001
    # insert random nan in very other columns
    for col in range(0, 100, 2):
        a1[:100, col] = np.nan
    index = [hashlib.sha224(str(x).encode('utf-8')).hexdigest() for x in range(size)]
    columns = [hashlib.sha224(str(x).encode('utf-8')).hexdigest() for x in range(columns)]
    sfs = sf.Frame(a1, index=index, columns=columns)
    pds = pd.DataFrame(a1, index=index, columns=columns)
    return pds, sfs, a1

pdf_float_10k, sff_float_10k, npf_float_10k = get_sample_frame_float_string_index(10000)


class PerfTest:
    PD_NAME = 'pd'
    SF_NAME = 'sf'
    FUNCTION_NAMES = ('np', PD_NAME, SF_NAME)
    NUMBER = 500



#-------------------------------------------------------------------------------
# series tests

class SeriesIntFloat_isnull(PerfTest):
    @staticmethod
    def np():
        post = np.isnan(npa_int_float_10k)

    @staticmethod
    def pd():
        post = pds_int_float_10k.isnull()

    @staticmethod
    def sf():
        post = sfs_int_float_10k.isna()


class SeriesIntFloat_dropna(PerfTest):
    @staticmethod
    def np():
        post = npa_int_float_10k[np.isnan(npa_int_float_10k)]

    @staticmethod
    def pd():
        post = pds_int_float_10k.dropna()

    @staticmethod
    def sf():
        post = sfs_int_float_10k.dropna()


class SeriesIntFloat_fillna(PerfTest):
    @staticmethod
    def np():
        sel = np.isnan(npa_int_float_10k)
        post = npa_int_float_10k.copy()
        post[sel] = 0.0

    @staticmethod
    def pd():
        post = pds_int_float_10k.fillna(0.0)

    @staticmethod
    def sf():
        post = sfs_int_float_10k.fillna(0.0)






class SeriesStrFloat_isnull(PerfTest):

    @staticmethod
    def pd():
        post = pds_str_float_10k.isnull()

    @staticmethod
    def sf():
        post = sfs_str_float_10k.isna()


class SeriesStrFloat_dropna(PerfTest):

    @staticmethod
    def pd():
        post = pds_str_float_10k.dropna()

    @staticmethod
    def sf():
        post = sfs_str_float_10k.dropna()


class SeriesStrFloat_fillna(PerfTest):

    @staticmethod
    def pd():
        post = pds_str_float_10k.fillna(0.0)

    @staticmethod
    def sf():
        post = sfs_str_float_10k.fillna(0.0)






class SeriesIntObj_isnull(PerfTest):
    @staticmethod
    def pd():
        post = pds_obj_10k.isnull()

    @staticmethod
    def sf():
        post = sfs_obj_10k.isna()


class SeriesIntObj_dropna(PerfTest):
    @staticmethod
    def pd():
        post = pds_obj_10k.dropna()

    @staticmethod
    def sf():
        post = sfs_obj_10k.dropna()


class SeriesIntObj_fillna(PerfTest):

    @staticmethod
    def pd():
        post = pds_obj_10k.fillna(0.0)

    @staticmethod
    def sf():
        post = sfs_obj_10k.fillna(0.0)






class SeriesIntObjStr_isnull(PerfTest):
    @staticmethod
    def pd():
        post = pds_objstr_10k.isnull()

    @staticmethod
    def sf():
        post = sfs_objstr_10k.isna()


class SeriesIntObjStr_dropna(PerfTest):
    @staticmethod
    def pd():
        post = pds_objstr_10k.dropna()

    @staticmethod
    def sf():
        post = sfs_objstr_10k.dropna()


class SeriesIntObjStr_fillna(PerfTest):

    @staticmethod
    def pd():
        post = pds_obj_10k.fillna('wrong')

    @staticmethod
    def sf():
        post = sfs_obj_10k.fillna('wrong')





#-------------------------------------------------------------------------------
# frame tests

class FrameFloat_sum_skipna_axis0(PerfTest):
    @staticmethod
    def np():
        post = np.nansum(npf_float_10k, axis=0)
        assert post.shape == (100,)

    @staticmethod
    def pd():
        post = pdf_float_10k.sum(axis=0, skipna=True)
        assert post.shape == (100,)

    @staticmethod
    def sf():
        post = sff_float_10k.sum(axis=0, skipna=True)
        assert post.shape == (100,)


class FrameFloat_sum_skipna_axis1(PerfTest):
    @staticmethod
    def np():
        post = np.nansum(npf_float_10k, axis=1)
        assert post.shape == (10000,)

    @staticmethod
    def pd():
        post = pdf_float_10k.sum(axis=1, skipna=True)
        assert post.shape == (10000,)

    @staticmethod
    def sf():
        post = sff_float_10k.sum(axis=1, skipna=True)
        assert post.shape == (10000,)




class FrameFloat_dropna_any_axis0(PerfTest):

    @staticmethod
    def pd():
        post = pdf_float_10k.dropna(axis=0, how='any')
        assert post.shape == (9900, 100)

    @staticmethod
    def sf():
        post = sff_float_10k.dropna(axis=0, condition=np.any)
        assert post.shape == (9900, 100)


class FrameFloat_dropna_any_axis1(PerfTest):

    @staticmethod
    def pd():
        post = pdf_float_10k.dropna(axis=1, how='any')
        assert post.shape == (10000, 50)

    @staticmethod
    def sf():
        post = sff_float_10k.dropna(axis=1, condition=np.any)
        assert post.shape == (10000, 50)


class FrameFloat_isna(PerfTest):

    @staticmethod
    def np():
        post = np.isnan(npf_float_10k)

    @staticmethod
    def pd():
        post = pdf_float_10k.isnull()

    @staticmethod
    def sf():
        post = sff_float_10k.isna()





class FrameFloat_apply_axis0(PerfTest):

    @staticmethod
    def pd():
        func = lambda a: np.nanmean(a ** 2)
        post = pdf_float_10k.apply(func, axis=0) # apply to columns
        assert post.shape == (100,)
        assert post.sum() > 33501616.16668333

    @staticmethod
    def sf():
        func = lambda a: np.nanmean(a ** 2)
        post = sff_float_10k.iter_array(0).apply(func) # apply to columns
        assert post.shape == (100,)
        assert post.sum() > 33501616.16668333


class FrameFloat_apply_axis1(PerfTest):
    NUMBER = 10

    @staticmethod
    def pd():
        func = lambda a: np.nanmean(a ** 2)
        post = pdf_float_10k.apply(func, axis=1) # apply to rows
        assert post.shape == (10000,)
        assert post.sum() > 3333328333.8349

    @staticmethod
    def sf():
        func = lambda a: np.nanmean(a ** 2)
        post = sff_float_10k.iter_array(1).apply(func)
        assert post.shape == (10000,)
        assert post.sum() > 3333328333.8349

#-------------------------------------------------------------------------------
# frame creation and growth


class FrameFloat_add_series_partial(PerfTest):

    # 325 two character strings
    _index = list(''.join(x) for x in it.combinations(string.ascii_lowercase, 2))

    @classmethod
    def pd(cls):
        f1 = pd.DataFrame(index=cls._index)
        for col in range(100):
            s = pd.Series(col * .1, index=cls._index[col: col+20])
            f1[col] = s
        assert f1.sum().sum() == 9900.0

    @classmethod
    def sf(cls):
        f1 = sf.FrameGO(index=cls._index)
        for col in range(100):
            s = sf.Series(col * .1, index=cls._index[col: col+20])
            f1[col] = s
        assert f1.sum().sum() == 9900.0




#-------------------------------------------------------------------------------

def get_arg_parser():
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
    p.add_argument('--profile',
            help='Turn on profiling',
            action='store_true',
            default=False,
            )
    p.add_argument('--performance',
            help='Turn on performance measurements',
            action='store_true',
            default=False,
            )
    return p


def yield_classes(pattern: str):
    # this will not find children of children
    for cls in PerfTest.__subclasses__():
        if fnmatch.fnmatch(cls.__name__.lower(), pattern.lower()):
            yield cls

def profile(cls, function='sf'):
    '''
    Profile the `sf` function from the supplied class.
    '''

    f = getattr(cls, function)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(cls.NUMBER):
        f()
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())

def performance(cls) -> tp.Tuple[str, float, float, float]:
    #row = []
    row = collections.OrderedDict()
    row['name'] = cls.__name__
    for f in PerfTest.FUNCTION_NAMES:
        if hasattr(cls, f):
            result = timeit.timeit(cls.__name__ + '.' + f + '()',
                    globals=globals(),
                    number=cls.NUMBER)
            row[f] = result
        else:
            row[f] = np.nan
    row['sf/pd'] = row[PerfTest.SF_NAME] / row[PerfTest.PD_NAME]
    return row
    return tuple(row.values())


def main():

    options = get_arg_parser().parse_args()
    records = []
    for pattern in options.patterns:
        for cls in sorted(yield_classes(pattern), key=lambda c: c.__name__):
            print(cls.__name__)
            if options.performance:
                records.append(performance(cls))
            if options.profile:
                profile(cls)
    if records:
        df = pd.DataFrame.from_records(records)
        print(df)



    # df = pd.DataFrame.from_records(records, columns=('name',) + PerfTest.FUNCTION_NAMES)
    # print(df)


if __name__ == '__main__':
    main()

