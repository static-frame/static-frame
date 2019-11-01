import typing as tp
import itertools as it
import string
import os
from urllib import request

import numpy as np
import pandas as pd
import static_frame as sf


from static_frame.performance.perf_test import PerfTest



class SampleData:

    _store: tp.Dict[str, tp.Any] = {}

    URL_CSV = 'https://data.ny.gov/api/views/xe9x-a24f/rows.csv?accessType=DOWNLOAD'
    FP_CSV = '/tmp/sf_pydata_2018.csv'
    URL_JSON = 'https://jsonplaceholder.typicode.com/photos'

    @classmethod
    def create(cls) -> None:

        if not os.path.exists(cls.FP_CSV):
            with request.urlopen(cls.URL_CSV) as response:
                with open(cls.FP_CSV, 'w') as f:
                    f.write(response.read().decode('utf-8'))

        cls._store['data_csv_fp'] = cls.FP_CSV
        cls._store['data_json_url'] = cls.URL_JSON

        labels_src = list(''.join(x) for x in it.combinations(string.ascii_lowercase, 4))
        assert len(labels_src) > 10000

        index = labels_src[:10000]
        columns = labels_src[:1000]

        data_float = np.random.rand(len(index), len(columns))

        # alt floats, Bools
        data_func = [
                lambda: np.random.rand(len(index)),
                lambda: np.random.randint(-1, 1, len(index)).astype(bool)
                ]

        cls._store['index'] = index
        cls._store['index_target'] = [idx for idx in index if 'd' in idx]
        cls._store['columns'] = columns
        cls._store['columns_target'] = [c for c in columns if 'd' in c]
        cls._store['data_float'] = data_float
        cls._store['data_func'] = data_func

        cls._store['sf.FrameFloat'] = sf.Frame(data_float, index=index, columns=columns)
        cls._store['pd.FrameFloat'] = pd.DataFrame(data_float, index=index, columns=columns)

        # mypy hates this:
        data_cols = {i: data_func[i%2]() for i in range(len(columns))}  # type: ignore
        cls._store['sf.FrameMixed'] = sf.Frame.from_dict(data_cols, index=index)
        cls._store['pd.FrameMixed'] = pd.DataFrame(data_cols, index=index)

    @classmethod
    def get(cls, key: str) -> tp.Any:
        return cls._store[key]


class FloatFrameStrLabel_100_Init(PerfTest):

    NUMBER = 100

    @classmethod
    def pd(cls) -> None:
        post = pd.DataFrame(SampleData.get('data_float'),
                index=SampleData.get('index'),
                columns=SampleData.get('columns')
                )
        assert post.shape == (10000, 1000)

    @classmethod
    def sf(cls) -> None:
        post = sf.Frame(SampleData.get('data_float'),
                index=SampleData.get('index'),
                columns=SampleData.get('columns')
                )
        assert post.shape == (10000, 1000)


class FloatFrameStrLabel_101_ApplyArrayAxis1(PerfTest):

    NUMBER = 10

    @classmethod
    def pd(cls) -> None:
        frame = SampleData.get('pd.FrameFloat')
        post = frame.apply(np.sum, axis=1, raw=True)
        assert post.shape == (10000,)


    @classmethod
    def sf(cls) -> None:
        frame = SampleData.get('sf.FrameFloat')
        post = frame.iter_array(axis=1).apply(np.sum)
        assert post.shape == (10000,)

class FloatFrameStrLabel_102_ApplyArrayAxis0(PerfTest):

    NUMBER = 10

    @classmethod
    def pd(cls) -> None:
        frame = SampleData.get('pd.FrameFloat')
        post = frame.apply(np.sum, axis=0, raw=True)
        assert post.shape == (1000,)

    @classmethod
    def sf(cls) -> None:
        frame = SampleData.get('sf.FrameFloat')
        post = frame.iter_array(axis=0).apply(np.sum)
        assert post.shape == (1000,)



class FloatFrameStrLabel_103_ApplySeriesAxis1(PerfTest):

    NUMBER = 2

    @classmethod
    def pd(cls) -> None:
        frame = SampleData.get('pd.FrameFloat')
        cols = SampleData.get('columns_target')
        post = frame.apply(lambda s: s[cols].sum(), axis=1)
        assert post.shape == (10000,)

    @classmethod
    def sf(cls) -> None:
        frame = SampleData.get('sf.FrameFloat')
        cols = SampleData.get('columns_target')
        post = frame.iter_series(axis=1).apply(lambda s: s[cols].sum())
        assert post.shape == (10000,)

class FloatFrameStrLabel_104_ApplySeriesAxis0(PerfTest):

    NUMBER = 2

    @classmethod
    def pd(cls) -> None:
        frame = SampleData.get('pd.FrameFloat')
        indices = SampleData.get('index_target')
        post = frame.apply(lambda s: s[indices].sum(), axis=0)
        assert post.shape == (1000,)

    @classmethod
    def sf(cls) -> None:
        frame = SampleData.get('sf.FrameFloat')
        indices = SampleData.get('index_target')
        post = frame.iter_series(axis=0).apply(lambda s: s[indices].sum())
        assert post.shape == (1000,)

class FloatFrameStrLabel_105_SubsetConcat(PerfTest):

    NUMBER = 2

    @classmethod
    def pd(cls) -> None:
        frame = SampleData.get('pd.FrameFloat')
        post = pd.concat((
                frame[[c for c in frame.columns if c.startswith('ad')]],
                frame[[c for c in frame.columns if c.startswith('ae')]]
                ), axis=1)
        assert post.shape == (10000, 441)

    @classmethod
    def sf(cls) -> None:
        frame = SampleData.get('sf.FrameFloat')
        post = sf.Frame.from_concat((
                frame[[c for c in frame.columns if c.startswith('ad')]],
                frame[[c for c in frame.columns if c.startswith('ae')]]
                ), axis=1)
        assert post.shape == (10000, 441)







class MixedFrameIntLabel_200_Init(PerfTest):

    NUMBER = 10

    @classmethod
    def pd(cls) -> None:

        data_func = SampleData.get('data_func')
        columns = SampleData.get('columns')
        index = SampleData.get('index')
        post = pd.DataFrame({i: data_func[i%2]() for i in range(len(columns))}, index=index)
        assert post.shape == (10000, 1000)

    @classmethod
    def sf(cls) -> None:
        data_func = SampleData.get('data_func')
        columns = SampleData.get('columns')
        index = SampleData.get('index')
        post = sf.Frame.from_dict({i: data_func[i%2]() for i in range(len(columns))}, index=index)
        assert post.shape == (10000, 1000)



class MixedFrameIntLabel_201_SubsetSumAxis0(PerfTest):

    NUMBER = 10

    @classmethod
    def pd(cls) -> None:

        frame = SampleData.get('pd.FrameMixed')
        post = frame[[c for c in frame.columns if c%2 == 0]].sum()
        assert post.shape == (500,)

    @classmethod
    def sf(cls) -> None:
        frame = SampleData.get('sf.FrameMixed')
        post = frame[[c for c in frame.columns if c%2 == 0]].sum()
        assert post.shape == (500,)



class MixedFrameIntLabel_202_SubsetSumAxis1(PerfTest):

    NUMBER = 10

    @classmethod
    def pd(cls) -> None:

        frame = SampleData.get('pd.FrameMixed')
        post = frame[[c for c in frame.columns if c%2 == 0]].sum(axis=1)
        assert post.shape == (10000,)

    @classmethod
    def sf(cls) -> None:
        frame = SampleData.get('sf.FrameMixed')
        post = frame[[c for c in frame.columns if c%2 == 0]].sum(axis=1)
        assert post.shape == (10000,)







class MixedFrame_300_CSV(PerfTest):

    NUMBER = 2

    @classmethod
    def pd(cls) -> None:
        data_fp = SampleData.get('data_csv_fp')
        post = pd.read_csv(data_fp)
        assert post.shape == (1654482, 19)

    @classmethod
    def sf(cls) -> None:
        data_fp = SampleData.get('data_csv_fp')
        post = sf.Frame.from_csv(data_fp)
        assert post.shape == (1654482, 19)




class MixedFrame_301_JSON(PerfTest):

    NUMBER = 6

    @classmethod
    def pd(cls) -> None:
        data_url = SampleData.get('data_json_url')
        post = pd.read_json(data_url)
        assert post.shape == (5000, 5)

    @classmethod
    def sf(cls) -> None:
        data_url = SampleData.get('data_json_url')
        post = sf.Frame.from_json_url(data_url)
        assert post.shape == (5000, 5)


class MixedFrame_302_CSVHybrid(PerfTest):

    NUMBER = 2

    @classmethod
    def pd(cls) -> None:
        data_fp = SampleData.get('data_csv_fp')
        post = pd.read_csv(data_fp)
        assert post.shape == (1654482, 19)

    @classmethod
    def sf(cls) -> None:
        data_fp = SampleData.get('data_csv_fp')
        post = sf.Frame.from_pandas(pd.read_csv(data_fp))
        assert post.shape == (1654482, 19)
