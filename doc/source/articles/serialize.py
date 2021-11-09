

import os
import timeit
import tempfile
import typing as tp
import pickle

import numpy as np
import frame_fixtures as ff
import static_frame as sf
import pandas as pd
from static_frame.core.display_color import HexColor



FF_wide = 's(10,10_000)|v(int,int,bool,float,float)|i(I,int)|c(I,str)'
FF_wide_col = 's(10,10_000)|v(int,bool,float)|i(I,int)|c(I,str)'
FF_wide_ext = 's(1000,10_000)|v(int,int,bool,float,float)|i(I,int)|c(I,str)'

FF_tall = 's(10_000,10)|v(int,int,bool,float,float)|i(I,int)|c(I,str)'
FF_tall_col = 's(10_000,10)|v(int,bool,float)|i(I,int)|c(I,str)'
FF_tall_ext = 's(10_000,1000)|v(int,int,bool,float,float)|i(I,int)|c(I,str)'

FF_square = 's(1_000,1_000)|v(float)|i(I,int)|c(I,str)'

class FileIOTest:
    NUMBER = 4
    SUFFIX = '.tmp'

    def __init__(self, fixture: str):
        self.fixture = ff.parse(fixture)
        _, self.fp = tempfile.mkstemp(suffix=self.SUFFIX)

    def __del__(self) -> None:
        os.unlink(self.fp)

    def __call__(self):
        raise NotImplementedError()



class SFReadParquet(FileIOTest):
    SUFFIX = '.parquet'

    def __init__(self, fixture: str):
        super().__init__(fixture)
        self.fixture.to_parquet(self.fp)

    def __call__(self):
        f = sf.Frame.from_parquet(self.fp, index_depth=1)
        _ = f.loc[34715, 'zZbu']

class SFWriteParquet(FileIOTest):
    SUFFIX = '.parquet'

    def __call__(self):
        self.fixture.to_parquet(self.fp)



class PDReadParquet(FileIOTest):
    SUFFIX = '.parquet'

    def __init__(self, fixture: str):
        super().__init__(fixture)
        df = self.fixture.to_pandas()
        df.to_parquet(self.fp)

    def __call__(self):
        f = pd.read_parquet(self.fp)
        _ = f.loc[34715, 'zZbu']

class PDWriteParquet(FileIOTest):
    SUFFIX = '.parquet'

    def __init__(self, fixture: str):
        super().__init__(fixture)
        self.df = self.fixture.to_pandas()

    def __call__(self):
        self.df.to_parquet(self.fp)




class SFReadNPZ(FileIOTest):
    SUFFIX = '.npz'

    def __init__(self, fixture: str):
        super().__init__(fixture)
        self.fixture.to_npz(self.fp)

    def __call__(self):
        f = sf.Frame.from_npz(self.fp)
        _ = f.loc[34715, 'zZbu']


class SFWriteNPZ(FileIOTest):
    SUFFIX = '.npz'

    def __call__(self):
        self.fixture.to_npz(self.fp)



class SFReadPickle(FileIOTest):
    SUFFIX = '.pickle'

    def __init__(self, fixture):
        super().__init__(fixture)
        self.file = open(self.fp, 'wb')
        pickle.dump(self.fixture, self.file)
        self.file.close()

    def __call__(self):
        with open(self.fp, 'rb') as f:
            f = pickle.load(f)
            _ = f.loc[34715, 'zZbu']


    def __del__(self) -> None:
        self.file.close()
        os.unlink(self.fp)


class SFWritePickle(FileIOTest):
    SUFFIX = '.pickle'

    def __init__(self, fixture):
        super().__init__(fixture)
        self.file = open(self.fp, 'wb')

    def __call__(self):
        pickle.dump(self.fixture, self.file)

    def __del__(self) -> None:
        self.file.close()
        os.unlink(self.fp)



#-------------------------------------------------------------------------------
def convert_size(size_bytes):
    import math
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

def get_sizes():
    records = []
    for label, fixture in (
            ('wide', FF_wide),
            ('wide_ext', FF_wide_ext),
            ('tall', FF_tall),
            ('tall_ext', FF_tall_ext),
            ('square', FF_square),
            ):
        f = ff.parse(fixture)
        record = [label, f.shape]

        _, fp = tempfile.mkstemp(suffix='.parquet')
        f.to_parquet(fp, include_index=True)
        size = os.path.getsize(fp)
        os.unlink(fp)
        record.append(convert_size(size))

        _, fp = tempfile.mkstemp(suffix='.npz')
        f.to_npz(fp, include_columns=True)
        size = os.path.getsize(fp)
        os.unlink(fp)
        record.append(convert_size(size))

        _, fp = tempfile.mkstemp(suffix='.pickle')
        file = open(fp, 'wb')
        pickle.dump(f, file)
        file.close()
        size = os.path.getsize(fp)
        os.unlink(fp)
        record.append(convert_size(size))

        records.append(record)

    f = sf.Frame.from_records(records, columns=('fixture', 'shape', 'parquet', 'npz', 'pickle')).set_index('fixture', drop=True)
    print(f)


def pandas_serialize():
    import pandas as pd
    df = ff.parse('s(10,10)|v(int,int,bool,float,float)|i(I,int)|c(I,str)').rename('foo').to_pandas().set_index(['zZbu', 'ztsv'])
    df = df.reindex(columns=pd.Index(df.columns, name='foo'))


    import ipdb; ipdb.set_trace()


#-------------------------------------------------------------------------------


def get_format():

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

        return str(v)

    return format


def run_test():
    records = []
    for label, fixture in (
            ('wide', FF_wide),
            ('wide_col', FF_wide_col),
            ('tall', FF_tall),
            ('tall_col', FF_tall_col),
            ('square', FF_square),
            ):
    # for label, fixture in (('square', FF_square),):
        for cls in (
                PDWriteParquet,
                SFWriteParquet,
                SFWriteNPZ,
                SFWritePickle,
                PDReadParquet,
                SFReadParquet,
                SFReadNPZ,
                SFReadPickle,
                ):
            runner = cls(fixture)
            record = [cls.__name__, cls.NUMBER, label]
            result = timeit.timeit(
                    f'runner()',
                    globals=locals(),
                    number=cls.NUMBER)
            record.append(result)
            records.append(record)

    f = sf.FrameGO.from_records(records,
            columns=('name', 'number', 'fixture', 'time')
            )

    display = f.iter_element_items().apply(get_format())

    config = sf.DisplayConfig(
            cell_max_width_leftmost=np.inf,
            cell_max_width=np.inf,
            type_show=False,
            display_rows=200,
            include_index=False,
            )
    print(display.display(config))
    # import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    # pandas_serialize()
    # get_sizes()
    run_test()

