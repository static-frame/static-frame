import timeit
import cProfile
import pstats
import io
import collections
import typing as tp
import argparse
import string
import hashlib
import fnmatch

import pandas as pd
import numpy as np

import static_frame as sf


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





class PerfTest:
    PD_NAME = 'pd'
    SF_NAME = 'sf'
    FUNCTION_NAMES = ('np', PD_NAME, SF_NAME)
    NUMBER = 2000




class SeriesIntFloat_isnull(PerfTest):
    @staticmethod
    def np():
        post = np.isnan(npa_int_float_10k)

    @staticmethod
    def pd():
        post = pds_int_float_10k.isnull()

    @staticmethod
    def sf():
        post = sfs_int_float_10k.isnull()


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
        post = sfs_str_float_10k.isnull()


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
        post = sfs_obj_10k.isnull()


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
        post = sfs_objstr_10k.isnull()


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


#                  name        np        pd         sf
# 0            PerfTest       NaN       NaN        NaN
# 1  SeriesFloat_dropna       NaN  0.433320  11.352377
# 2  SeriesFloat_isnull  0.034229  0.144864   0.059468
# %

#dropped ordered dict

#                  name        np        pd         sf
# 0  SeriesFloat_dropna  0.104048  0.432315  12.350630
# 1  SeriesFloat_isnull  0.034371  0.146767   0.060482

# stored positions array, alternate dict comp for non generators


# % python3 RALib/test/unit_test/test_static_frame_perf.py
#                  name        np        pd        sf
# 0  SeriesFloat_dropna  0.098362  0.431975  3.215246
# 1  SeriesFloat_isnull  0.034171  0.148821  0.059623


# % python3 RALib/test/unit_test/test_static_frame_perf.py
#                  name        np        pd        sf
# 0  SeriesFloat_dropna  0.104799  0.448937  3.668478
# 1  SeriesFloat_isnull  0.034877  0.158095  0.060943
# 2    SeriesObj_dropna       NaN  1.477193  9.369060
# 3    SeriesObj_isnull       NaN  0.926872  5.997276


# updated nan discovery in object types to use astype conversion

# % python3 RALib/test/unit_test/test_static_frame_perf.py
#                  name       np        pd        sf
# 0  SeriesFloat_dropna  0.10071  0.437749  3.617273
# 1  SeriesFloat_isnull  0.03505  0.150843  0.060352
# 2    SeriesObj_dropna      NaN  1.484937  4.759194
# 3    SeriesObj_isnull      NaN  0.937147  1.049971


# replaced dictionary comprehension with dictionary function

# % python3 RALib/test/unit_test/test_static_frame_perf.py
#                  name        np        pd        sf
# 0  SeriesFloat_dropna  0.096741  0.428629  3.133134
# 1  SeriesFloat_isnull  0.034854  0.145981  0.059680
# 2    SeriesObj_dropna       NaN  1.473483  4.400348
# 3    SeriesObj_isnull       NaN  0.934011  1.02889




#                       name        np        pd        sf      sf/pd
# 0    SeriesIntFloat_dropna  0.040793  0.169650  1.390540   8.196537
# 1    SeriesIntFloat_fillna  0.051688  0.131705  0.073481   0.557924
# 2    SeriesIntFloat_isnull  0.015110  0.063173  0.023108   0.365792
# 3   SeriesIntObjStr_dropna       NaN  0.706906  5.830331   8.247676
# 4   SeriesIntObjStr_fillna       NaN  0.781177  0.506214   0.648014
# 5   SeriesIntObjStr_isnull       NaN  0.364144  3.979732  10.929006
# 6      SeriesIntObj_dropna       NaN  0.578961  1.647710   2.845976
# 7      SeriesIntObj_fillna       NaN  1.728827  0.496637   0.287268
# 8      SeriesIntObj_isnull       NaN  0.365337  0.412665   1.129545
# 9    SeriesStrFloat_dropna       NaN  0.227070  4.779123  21.046928
# 10   SeriesStrFloat_fillna       NaN  0.168498  0.071351   0.423456
# 11   SeriesStrFloat_isnull       NaN  0.061955  0.021504   0.347098
