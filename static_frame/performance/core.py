
import io
import collections
import typing as tp
import itertools as it
import string
import hashlib

import pandas as pd  # type: ignore
import numpy as np  # type: ignore

import static_frame as sf

from static_frame.performance.perf_test import PerfTest

#-------------------------------------------------------------------------------

def get_sample_series_float(size: int = 10000) -> tp.Tuple[pd.Series, sf.Series, np.ndarray]:
    a1 = np.arange(size) * .001
    a1[size // 2:] = np.nan
    pds = pd.Series(a1)
    sfs = sf.Series(a1)
    return pds, sfs, a1

def get_sample_series_string_index_float_values(size: int = 10000) -> tp.Tuple[pd.Series, sf.Series, np.ndarray]:
    a1 = np.arange(size) * .001
    a1[size // 2:] = np.nan
    # create hsa indices
    index = [hashlib.sha224(str(x).encode('utf-8')).hexdigest() for x in range(size)]
    pds = pd.Series(a1, index=index)
    sfs = sf.Series(a1, index=index)
    return pds, sfs, a1

def get_sample_series_obj(size: int = 10000) -> tp.Tuple[pd.Series, sf.Series, np.ndarray]:
    sample = [None, 3, 0.123, np.nan]
    a1 = np.array(sample * int(size / len(sample)))

    pds = pd.Series(a1)
    sfs = sf.Series(a1)

    return pds, sfs, a1

def get_sample_series_objstr(size: int = 10000) -> tp.Tuple[pd.Series, sf.Series, np.ndarray]:
    sample = [None, 3, 0.123, np.nan, 'str']
    a1 = np.array(sample * int(size / len(sample)))

    pds = pd.Series(a1)
    sfs = sf.Series(a1)

    return pds, sfs, a1


#-------------------------------------------------------------------------------
# frame generators


def get_sample_frame_float_string_index(size: int = 10000, columns: int = 100) -> tp.Tuple[pd.DataFrame, sf.Frame, np.ndarray]:
    a1 = (np.arange(size * columns)).reshape((size, columns)) * .001
    # insert random nan in very other columns
    for col in range(0, 100, 2):
        a1[:100, col] = np.nan
    index = [hashlib.sha224(str(x).encode('utf-8')).hexdigest() for x in range(size)]
    cols = [hashlib.sha224(str(x).encode('utf-8')).hexdigest() for x in range(columns)]
    sff = sf.Frame(a1, index=index, columns=cols)
    pdf = pd.DataFrame(a1, index=index, columns=cols)
    return pdf, sff, a1


_mixed_types = ('foo', 'bar', True, None, 234.34, 90)

def _typed_array(dtype: type, size: int, shift: int = 0) -> np.ndarray:
    if dtype == float:
        return np.roll(np.arange(size) * .001, shift)
    if dtype == int:
        return np.roll(np.arange(size), shift)
    if dtype == bool:
        return np.roll(np.isin(np.arange(size) % 5, (1, 4)), shift)
    if dtype == object:
        return np.roll(np.array([_mixed_types[x % len(_mixed_types)] for x in range(size)]), shift)
    raise NotImplementedError()

def get_sample_frame_mixed_string_index(size: int = 10000, columns: int = 100) -> tp.Tuple[
            pd.DataFrame, sf.FrameGO, np.ndarray]:
    '''Get frames with mixed types.
    '''
    # produces 14950 strings
    source_ids = list(''.join(x) for x in it.combinations(string.ascii_lowercase, 4))
    assert size <= len(source_ids)

    index = source_ids[:size]
    cols = source_ids[:columns]

    dtypes = (float, int, object, bool)

    sff = sf.FrameGO(index=index)
    for idx, col in enumerate(cols):
        s = sf.Series(_typed_array(dtypes[idx % 4], size=size, shift=idx), index=index)
        sff[col] = s

    npf = sff.values

    pdf = pd.DataFrame(index=index)
    for idx, col in enumerate(cols):
        s = pd.Series(_typed_array(dtypes[idx % 4], size=size, shift=idx), index=index)
        pdf[col] = s


    return pdf, sff, npf


def get_series_float_h2d_str_index(size: int = 1000) -> tp.Tuple[pd.Series, sf.Series]:
    '''
    Get a hierarchical index with
    '''
    labels = list(''.join(x) for x in it.combinations(string.ascii_lowercase, 4))
    labels0 = labels[:int(size / 10)]
    labels1 = labels[:size]
    values = np.arange(len(labels0) * len(labels1)) * .001

    ih = sf.IndexHierarchy.from_product(labels0, labels1)
    sfs = sf.Series(values, index=ih)

    mi = pd.MultiIndex.from_product((labels0, labels1))
    pds = pd.Series(values, index=mi)
    return pds, sfs


def get_series_float_h3d_str_index(size: int = 1000) -> tp.Tuple[pd.Series, sf.Series]:
    '''
    Get a hierarchical index with
    '''
    labels = list(''.join(x) for x in it.combinations(string.ascii_lowercase, 4))
    labels0 = labels[:int(size / 100)]
    labels1 = labels[:int(size / 10)]
    labels2 = labels[:size]

    values = np.arange(len(labels0) * len(labels1) * len(labels2)) * .001

    ih = sf.IndexHierarchy.from_product(labels0, labels1, labels2)
    sfs = sf.Series(values, index=ih)

    mi = pd.MultiIndex.from_product((labels0, labels1, labels2))
    pds = pd.Series(values, index=mi)
    return pds, sfs

class SampleData:

    _store: tp.Dict[str, tp.Any] = {}

    @classmethod
    def create(cls) -> None:
        pds_int_float_10k, sfs_int_float_10k, npa_int_float_10k = get_sample_series_float(10000)
        pds_obj_10k, sfs_obj_10k, npa_obj_10k = get_sample_series_obj(10000)
        pds_str_float_10k, sfs_str_float_10k, _ = get_sample_series_string_index_float_values(10000)
        pds_objstr_10k, sfs_objstr_10k, npa_objstr_10k = get_sample_series_objstr(10000)
        pdf_float_10k, sff_float_10k, npf_float_10k = get_sample_frame_float_string_index(10000)
        pdf_mixed_10k, sff_mixed_10k, npf_mixed_10k = get_sample_frame_mixed_string_index()

        pds_float_h2d_str_index, sfs_float_h2d_str_index = get_series_float_h2d_str_index()
        pds_float_h3d_str_index, sfs_float_h3d_str_index = get_series_float_h3d_str_index()


        for k, v in locals().items():
            if k == 'cls' or k.startswith('__'):
                continue
            cls._store[k] = v

        # additional resources
        cls._store['label_str'] = list(
                ''.join(x) for x in it.combinations(string.ascii_lowercase, 4))

        cls._store['label_tuple2_int_10000'] = [(int(x / 10), x)
                for x in range(10000)]
        cls._store['label_tuple3_int_100000'] = [(int(x / 100), int(x / 10), x)
                for x in range(100000)]

    @classmethod
    def get(cls, key: str) -> tp.Any:
        return cls._store[key]



#-------------------------------------------------------------------------------
# index Tests

class IndexStr_init(PerfTest):
    '''Index construction for string labels.
    '''

    @classmethod
    def pd(cls) -> None:
        pd.Index(SampleData.get('label_str'))

    @classmethod
    def sf(cls) -> None:
        sf.Index(SampleData.get('label_str'))


class IndexHierarchy2d_from_product(PerfTest):

    NUMBER = 100

    _size0 = 100
    _size1 = 1000

    @classmethod
    def pd(cls) -> None:
        labels0 = SampleData.get('label_str')[:cls._size0]
        labels1 = SampleData.get('label_str')[:cls._size1]
        ih = pd.MultiIndex.from_product((labels0, labels1))
        assert len(ih) == cls._size0 * cls._size1

    @classmethod
    def sf(cls) -> None:
        labels0 = SampleData.get('label_str')[:cls._size0]
        labels1 = SampleData.get('label_str')[:cls._size1]
        ih = sf.IndexHierarchy.from_product(labels0, labels1)
        assert len(ih) == cls._size0 * cls._size1

class IndexHierarchy3d_from_product(PerfTest):

    NUMBER = 10

    _size0 = 10
    _size1 = 100
    _size2 = 1000

    @classmethod
    def pd(cls) -> None:
        labels0 = SampleData.get('label_str')[:cls._size0]
        labels1 = SampleData.get('label_str')[:cls._size1]
        labels2 = SampleData.get('label_str')[:cls._size2]
        ih = pd.MultiIndex.from_product((labels0, labels1, labels2))
        assert len(ih) == cls._size0 * cls._size1 * cls._size2

    @classmethod
    def sf(cls) -> None:
        labels0 = SampleData.get('label_str')[:cls._size0]
        labels1 = SampleData.get('label_str')[:cls._size1]
        labels2 = SampleData.get('label_str')[:cls._size2]
        ih = sf.IndexHierarchy.from_product(labels0, labels1, labels2)
        assert len(ih) == cls._size0 * cls._size1 * cls._size2


class IndexHierarchy2d_from_labels(PerfTest):

    NUMBER = 100

    @classmethod
    def pd(cls) -> None:
        ih = pd.MultiIndex.from_tuples(SampleData.get('label_tuple2_int_10000'))

    @classmethod
    def sf(cls) -> None:
        ih = sf.IndexHierarchy.from_labels(SampleData.get('label_tuple2_int_10000'))


class IndexHierarchy3d_from_labels(PerfTest):

    NUMBER = 10

    @classmethod
    def pd(cls) -> None:
        ih = pd.MultiIndex.from_tuples(SampleData.get('label_tuple3_int_100000'))

    @classmethod
    def sf(cls) -> None:
        ih = sf.IndexHierarchy.from_labels(SampleData.get('label_tuple3_int_100000'))


#-------------------------------------------------------------------------------
# series tests

class SeriesIntFloat_init(PerfTest):
    @staticmethod
    def pd() -> None:
        post = pd.Series(SampleData.get('npa_int_float_10k'))

    @staticmethod
    def sf() -> None:
        post = pd.Series(SampleData.get('npa_int_float_10k'))


class SeriesStrObj_init(PerfTest):
    @staticmethod
    def pd() -> None:
        a = SampleData.get('npa_obj_10k')
        post = pd.Series(a, index=SampleData.get('label_str')[:len(a)])

    @staticmethod
    def sf() -> None:
        a = SampleData.get('npa_obj_10k')
        post = sf.Series(a, index=SampleData.get('label_str')[:len(a)])




class SeriesIntFloat_isnull(PerfTest):
    @staticmethod
    def np() -> None:
        post = np.isnan(SampleData.get('npa_int_float_10k'))

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_int_float_10k').isnull()

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_int_float_10k').isna()


class SeriesIntFloat_dropna(PerfTest):
    @staticmethod
    def np() -> None:
        post = SampleData.get('npa_int_float_10k')[np.isnan(SampleData.get('npa_int_float_10k'))]

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_int_float_10k').dropna()

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_int_float_10k').dropna()


class SeriesIntFloat_fillna(PerfTest):
    @staticmethod
    def np() -> None:
        sel = np.isnan(SampleData.get('npa_int_float_10k'))
        post = SampleData.get('npa_int_float_10k').copy()
        post[sel] = 0.0

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_int_float_10k').fillna(0.0)

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_int_float_10k').fillna(0.0)


class SeriesIntFloat_fillna_forward(PerfTest):

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_int_float_10k').fillna(method='ffill')

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_int_float_10k').fillna_forward()


class SeriesIntFloat_apply(PerfTest):

    NUMBER = 50

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_int_float_10k').apply(str)

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_int_float_10k').iter_element().apply(str)






class SeriesStrFloat_isna(PerfTest):

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_str_float_10k').isnull()

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_str_float_10k').isna()


class SeriesStrFloat_dropna(PerfTest):

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_str_float_10k').dropna()

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_str_float_10k').dropna()


class SeriesStrFloat_fillna(PerfTest):

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_str_float_10k').fillna(0.0)

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_str_float_10k').fillna(0.0)


class SeriesStrFloat_fillna_forward(PerfTest):

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_str_float_10k').fillna(method='ffill')

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_str_float_10k').fillna_forward()



class SeriesStrFloat_apply(PerfTest):

    NUMBER = 50

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_str_float_10k').apply(str)

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_str_float_10k').iter_element().apply(str)




class SeriesIntObj_isnull(PerfTest):
    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_obj_10k').isnull()

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_obj_10k').isna()


class SeriesIntObj_dropna(PerfTest):
    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_obj_10k').dropna()

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_obj_10k').dropna()


class SeriesIntObj_fillna(PerfTest):

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_obj_10k').fillna(0.0)

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_obj_10k').fillna(0.0)


class SeriesIntObj_fillna_forward(PerfTest):
    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_obj_10k').fillna(method='ffill')

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_obj_10k').fillna_forward()


class SeriesIntObj_apply(PerfTest):

    NUMBER = 50

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_obj_10k').apply(str)

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_obj_10k').iter_element().apply(str)






class SeriesIntObjStr_isnull(PerfTest):
    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_objstr_10k').isnull()

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_objstr_10k').isna()

class SeriesIntObjStr_dropna(PerfTest):
    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_objstr_10k').dropna()

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_objstr_10k').dropna()

class SeriesIntObjStr_fillna(PerfTest):

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_objstr_10k').fillna('wrong')

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_objstr_10k').fillna('wrong')


class SeriesIntObjStr_fillna_forward(PerfTest):
    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_objstr_10k').fillna(method='ffill')

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_objstr_10k').fillna_forward()


class SeriesIntObjStr_apply(PerfTest):

    NUMBER = 50

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_objstr_10k').apply(str)

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_objstr_10k').iter_element().apply(str)






class SeriesFloatH2DString_loc_target(PerfTest):

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_float_h2d_str_index').loc[('abgu', 'abcf')]

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_float_h2d_str_index').loc[('abgu', 'abcf')]


class SeriesFloatH2DString_loc_slice(PerfTest):

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_float_h2d_str_index').loc[pd.IndexSlice[:, 'abcf']]

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_float_h2d_str_index').loc[sf.HLoc[:, 'abcf']]




class SeriesFloatH3DString_loc_target(PerfTest):
    '''
    Selecting single value from 3-level hierarchy.
    '''
    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_float_h3d_str_index').loc[('abce', 'abgu', 'afgx')]

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_float_h3d_str_index').loc[('abce', 'abgu', 'afgx')]


class SeriesFloatH3DString_loc_slice_target_slice(PerfTest):

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_float_h3d_str_index').loc[
                pd.IndexSlice[:, 'abcf', 'abcl':'abco']]  # type: ignore  # https://github.com/python/typeshed/pull/3024
        assert len(post) == 40

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_float_h3d_str_index').loc[
                sf.HLoc[:, 'abcf', 'abcl':'abco']]  # type: ignore  # https://github.com/python/typeshed/pull/3024
        assert len(post) == 40


class SeriesFloatH3DString_loc_slice_slice_target(PerfTest):

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pds_float_h3d_str_index').loc[
                pd.IndexSlice[:, :, 'abcl']]
        assert len(post) == 1000

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sfs_float_h3d_str_index').loc[
                sf.HLoc[:, :, 'abcl']]
        assert len(post) == 1000



#-------------------------------------------------------------------------------
# frame tests

# this is deemed invalid, as Pandas just holds a reference
# class FrameFloat_init(PerfTest):
#     @staticmethod
#     def pd() -> None:
#         post = pd.DataFrame(SampleData.get('npf_float_10k'))

#     @staticmethod
#     def sf() -> None:
#         post = sf.Frame(SampleData.get('npf_float_10k'))


class FrameStrFloat_init(PerfTest):
    @staticmethod
    def pd() -> None:
        data = SampleData.get('npf_float_10k')
        labels = SampleData.get('label_str')
        post = pd.DataFrame(data, index=labels[:data.shape[0]], columns=labels[:data.shape[1]])

    @staticmethod
    def sf() -> None:
        data = SampleData.get('npf_float_10k')
        labels = SampleData.get('label_str')
        post = sf.Frame(data, index=labels[:data.shape[0]], columns=labels[:data.shape[1]])


class FrameFloat_from_records(PerfTest):

    NUMBER = 10

    @staticmethod
    def pd() -> None:
        # make data into a list to force type identification
        post = pd.DataFrame.from_records(list(SampleData.get('npf_float_10k')))
        assert post.shape == (10000, 100)

    @staticmethod
    def sf() -> None:
        post = sf.Frame.from_records(list(SampleData.get('npf_float_10k')))
        assert post.shape == (10000, 100)



class FrameMixed_from_records(PerfTest):

    NUMBER = 10

    @staticmethod
    def pd() -> None:
        # make data into a list to force type identification
        post = pd.DataFrame.from_records(list(SampleData.get('npf_mixed_10k')))
        assert post.shape == (10000, 100)

    @staticmethod
    def sf() -> None:
        post = sf.Frame.from_records(list(SampleData.get('npf_mixed_10k')))
        # NOTE: this presently casts a mixed typed column into Bools
        assert post.shape == (10000, 100)





#-------------------------------------------------------------------------------
# frame util functions

class FrameFloat_sum_skipna_axis0(PerfTest):
    @staticmethod
    def np() -> None:
        post = np.nansum(SampleData.get('npf_float_10k'), axis=0)
        assert post.shape == (100,)

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pdf_float_10k').sum(axis=0, skipna=True)
        assert post.shape == (100,)

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sff_float_10k').sum(axis=0, skipna=True)
        assert post.shape == (100,)


class FrameFloat_sum_skipna_axis1(PerfTest):
    @staticmethod
    def np() -> None:
        post = np.nansum(SampleData.get('npf_float_10k'), axis=1)
        assert post.shape == (10000,)

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pdf_float_10k').sum(axis=1, skipna=True)
        assert post.shape == (10000,)

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sff_float_10k').sum(axis=1, skipna=True)
        assert post.shape == (10000,)


class FrameFloat_dropna_any_axis0(PerfTest):

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pdf_float_10k').dropna(axis=0, how='any')
        assert post.shape == (9900, 100)

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sff_float_10k').dropna(axis=0, condition=np.any)
        assert post.shape == (9900, 100)


class FrameFloat_dropna_any_axis1(PerfTest):

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pdf_float_10k').dropna(axis=1, how='any')
        assert post.shape == (10000, 50)

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sff_float_10k').dropna(axis=1, condition=np.any)
        assert post.shape == (10000, 50)


class FrameFloat_isna(PerfTest):

    @staticmethod
    def np() -> None:
        post = np.isnan(SampleData.get('npf_float_10k'))

    @staticmethod
    def pd() -> None:
        post = SampleData.get('pdf_float_10k').isnull()

    @staticmethod
    def sf() -> None:
        post = SampleData.get('sff_float_10k').isna()


class FrameFloat_apply_axis0(PerfTest):

    NUMBER = 50

    @staticmethod
    def pd() -> None:
        func = lambda a: np.nanmean(a ** 2)
        post = SampleData.get('pdf_float_10k').apply(func, axis=0) # apply to columns
        assert post.shape == (100,)
        assert post.sum() > 33501616.16668333

    @staticmethod
    def sf() -> None:
        func = lambda a: np.nanmean(a ** 2)
        post = SampleData.get('sff_float_10k').iter_array(0).apply(func) # apply to columns
        assert post.shape == (100,)
        assert post.sum() > 33501616.16668333


class FrameFloat_apply_axis1(PerfTest):
    NUMBER = 10

    @staticmethod
    def pd() -> None:
        func = lambda a: np.nanmean(a ** 2)
        post = SampleData.get('pdf_float_10k').apply(func, axis=1) # apply to rows
        assert post.shape == (10000,)
        assert post.sum() > 3333328333.8349

    @staticmethod
    def sf() -> None:
        func = lambda a: np.nanmean(a ** 2)
        post = SampleData.get('sff_float_10k').iter_array(1).apply(func)
        assert post.shape == (10000,)
        assert post.sum() > 3333328333.8349


#-------------------------------------------------------------------------------
# frame loc float


class FrameFloat_slice_loc_indices(PerfTest):

    @staticmethod
    def pd() -> None:
        for i in range(0, 10000, 1000):
            start = SampleData.get('pdf_float_10k').index[i]
            SampleData.get('pdf_float_10k').loc[start:]

    @staticmethod
    def sf() -> None:
        for i in range(0, 10000, 1000):
            start = SampleData.get('sff_float_10k').index.values[i]
            SampleData.get('sff_float_10k').loc[start:]


class FrameFloat_slice_loc_index(PerfTest):

    @staticmethod
    def pd() -> None:
        for i in range(0, 10000, 1000):
            start = SampleData.get('pdf_float_10k').index[i]
            SampleData.get('pdf_float_10k').loc[start]

    @staticmethod
    def sf() -> None:
        for i in range(0, 10000, 1000):
            start = SampleData.get('sff_float_10k').index.values[i]
            SampleData.get('sff_float_10k').loc[start]



class FrameFloat_slice_loc_columns(PerfTest):

    NUMBER = 100

    @staticmethod
    def pd() -> None:
        for i in range(0, 100, 10):
            start = SampleData.get('pdf_float_10k').index[i]
            SampleData.get('pdf_float_10k').loc[:, start:]

    @staticmethod
    def sf() -> None:
        for i in range(0, 100, 10):
            start = SampleData.get('sff_float_10k').index.values[i]
            SampleData.get('sff_float_10k').loc[:, start:]



class FrameFloat_slice_loc_column(PerfTest):

    @staticmethod
    def pd() -> None:
        for i in range(0, 100, 10):
            start = SampleData.get('pdf_float_10k').index[i]
            SampleData.get('pdf_float_10k').loc[:, start]

    @staticmethod
    def sf() -> None:
        for i in range(0, 100, 10):
            start = SampleData.get('sff_float_10k').index.values[i]
            SampleData.get('sff_float_10k').loc[:, start]


#-------------------------------------------------------------------------------
# frame loc mixed


class FrameMixed_slice_loc_indices(PerfTest):

    @staticmethod
    def pd() -> None:
        for i in range(0, 10000, 1000):
            start = SampleData.get('pdf_mixed_10k').index[i]
            SampleData.get('pdf_mixed_10k').loc[start:]

    @staticmethod
    def sf() -> None:
        for i in range(0, 10000, 1000):
            start = SampleData.get('sff_mixed_10k').index.values[i]
            SampleData.get('sff_mixed_10k').loc[start:]


class FrameMixed_slice_loc_index(PerfTest):

    @staticmethod
    def pd() -> None:
        for i in range(0, 10000, 1000):
            start = SampleData.get('pdf_mixed_10k').index[i]
            SampleData.get('pdf_mixed_10k').loc[start]

    @staticmethod
    def sf() -> None:
        for i in range(0, 10000, 1000):
            start = SampleData.get('sff_mixed_10k').index.values[i]
            SampleData.get('sff_mixed_10k').loc[start]



class FrameMixed_slice_loc_columns(PerfTest):

    @staticmethod
    def pd() -> None:
        for i in range(0, 100, 10):
            start = SampleData.get('pdf_mixed_10k').index[i]
            SampleData.get('pdf_mixed_10k').loc[:, start:]

    @staticmethod
    def sf() -> None:
        for i in range(0, 100, 10):
            start = SampleData.get('sff_mixed_10k').index.values[i]
            SampleData.get('sff_mixed_10k').loc[:, start:]



class FrameMixed_slice_loc_column(PerfTest):

    @staticmethod
    def pd() -> None:
        for i in range(0, 100, 10):
            start = SampleData.get('pdf_mixed_10k').index[i]
            SampleData.get('pdf_mixed_10k').loc[:, start]

    @staticmethod
    def sf() -> None:
        for i in range(0, 100, 10):
            start = SampleData.get('sff_mixed_10k').index.values[i]
            SampleData.get('sff_mixed_10k').loc[:, start]



#-------------------------------------------------------------------------------
# frame creation and growth

class FrameFloat_H1D_add_series_partial(PerfTest):
    '''Adding series that only partially match the index
    '''

    NUMBER = 50

    # 325 two character strings
    _index = list(''.join(x) for x in it.combinations(string.ascii_lowercase, 2))

    @classmethod
    def pd(cls) -> None:
        f1 = pd.DataFrame(index=cls._index)
        for col in range(100):
            s = pd.Series(col * .1, index=cls._index[col: col+20])
            f1[col] = s
        assert f1.sum().sum() == 9900.0

    @classmethod
    def sf(cls) -> None:
        f1 = sf.FrameGO(index=cls._index)
        for col in range(100):
            s = sf.Series(col * .1, index=cls._index[col: col+20])
            f1[col] = s
        assert f1.sum().sum() == 9900.0


class FrameFloat_H2D_add_series_partial(PerfTest):
    '''Adding series that only partially match the index
    '''
    NUMBER = 50
    _index_leaves = list(''.join(x) for x in it.combinations(string.ascii_lowercase, 2))

    @classmethod
    def sf(cls) -> None:
        index = sf.IndexHierarchy.from_product(list(string.ascii_lowercase),
                list(string.ascii_lowercase))
        f1 = sf.FrameGO(index=index)
        for col in range(100):
            s = sf.Series(col * .1, index=index[col: col+6])
            f1[col] = s
        assert f1.sum().sum() == 2970.0

    @classmethod
    def pd(cls) -> None:
        index = pd.MultiIndex.from_product((list(string.ascii_lowercase),
                list(string.ascii_lowercase)))
        f1 = pd.DataFrame(index=index)
        for col in range(100):
            s = pd.Series(col * .1, index=index[col: col+6])
            f1[col] = s
        assert f1.sum().sum() == 2970.0
