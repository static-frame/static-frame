

import os
import timeit
import tempfile
import typing as tp
import pickle
import shutil

import matplotlib.pyplot as plt
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
        self.fp_dir = '/tmp/npy'

    def clear(self) -> None:
        os.unlink(self.fp)
        if os.path.exists(self.fp_dir):
            shutil.rmtree(self.fp_dir)

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



class PDReadParquetArrow(FileIOTest):
    SUFFIX = '.parquet'

    def __init__(self, fixture: str):
        super().__init__(fixture)
        df = self.fixture.to_pandas()
        df.to_parquet(self.fp)

    def __call__(self):
        f = pd.read_parquet(self.fp)
        _ = f.loc[34715, 'zZbu']

class PDWriteParquetArrow(FileIOTest):
    SUFFIX = '.parquet'

    def __init__(self, fixture: str):
        super().__init__(fixture)
        self.df = self.fixture.to_pandas()

    def __call__(self):
        self.df.to_parquet(self.fp)


class PDReadParquetFast(FileIOTest):
    SUFFIX = '.parquet'

    def __init__(self, fixture: str):
        super().__init__(fixture)
        df = self.fixture.to_pandas()
        df.to_parquet(self.fp, engine='fastparquet')

    def __call__(self):
        f = pd.read_parquet(self.fp, engine='fastparquet')
        _ = f.loc[34715, 'zZbu']

class PDWriteParquetFast(FileIOTest):
    SUFFIX = '.parquet'

    def __init__(self, fixture: str):
        super().__init__(fixture)
        self.df = self.fixture.to_pandas()

    def __call__(self):
        self.df.to_parquet(self.fp, engine='fastparquet')





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


    def clear(self) -> None:
        self.file.close()
        os.unlink(self.fp)


class SFWritePickle(FileIOTest):
    SUFFIX = '.pickle'

    def __init__(self, fixture):
        super().__init__(fixture)
        self.file = open(self.fp, 'wb')

    def __call__(self):
        pickle.dump(self.fixture, self.file)

    def clear(self) -> None:
        self.file.close()
        os.unlink(self.fp)



class SFReadNPY(FileIOTest):
    SUFFIX = '.npy'

    def __init__(self, fixture: str):
        super().__init__(fixture)
        self.fixture.to_npy(self.fp_dir)

    def __call__(self):
        # import ipdb; ipdb.set_trace()
        f = sf.Frame.from_npy(self.fp_dir)
        _ = f.loc[34715, 'zZbu']


class SFWriteNPY(FileIOTest):
    SUFFIX = '.npy'

    def __call__(self):
        self.fixture.to_npy(self.fp_dir)




class SFReadNPYMM(FileIOTest):
    SUFFIX = '.npy'

    def __init__(self, fixture: str):
        super().__init__(fixture)
        self.fixture.to_npy(self.fp_dir)

    def __call__(self):
        # import ipdb; ipdb.set_trace()
        f = sf.Frame.from_npy(self.fp_dir, memory_map=True)
        _ = f.loc[34715, 'zZbu']



def plot(frame: sf.Frame):
    from collections import defaultdict
    fig, axes = plt.subplots(2, 1)

    for axes_count, cat in enumerate(frame.iter_group('category')):
        ax = axes[axes_count]

        labels = [] # by fixture
        results = defaultdict(list)

        components = 0
        for label, sub in cat.iter_group_items('fixture'):
            labels.append(label)
            components = max(components, len(sub))
            for row in sub.iter_series(axis=1):
                results[row['name']].append(row['time'])

        width = 0.85  # the width of each group
        x = np.arange(len(labels))

        segment = width / components
        start = -width / 2
        for i, (label, values) in enumerate(results.items()):
            r = ax.bar(x + (start + (segment * i)), values, segment, label=label)
            # ax.bar_label(r, padding=3)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        # ax.set_ylabel('Scores')
        # ax.set_title('Scores by group and gender')
        ax.set_xticks(x, labels)
        ax.set_yscale('log')
        ax.legend()


    fig.tight_layout()
    fp = '/tmp/serialize.png'
    plt.savefig(fp, dpi=300)
    import os
    os.system(f'eog {fp}')


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



# This subclass of ndarray has some unpleasant interactions with some operations, because it doesn’t quite fit properly as a subclass. An alternative to using this subclass is to create the mmap object yourself, then create an ndarray with ndarray.__new__ directly, passing the object created in its ‘buffer=’ parameter.

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

from itertools import repeat
from itertools import chain

def run_test():
    records = []
    for label, fixture in (
            ('wide', FF_wide),
            # ('wide_col', FF_wide_col),
            ('tall', FF_tall),
            # ('tall_col', FF_tall_col),
            ('square', FF_square),
            ):
        cls_read = (
            PDReadParquetArrow,
            # PDReadParquetFast, # not faster!
            SFReadParquet,
            SFReadNPZ,
            SFReadNPY,
            SFReadPickle,
            # SFReadNPYMM,
            )
        cls_write = (
            PDWriteParquetArrow,
            # PDWriteParquetFast, # not faster!
            SFWriteParquet,
            SFWriteNPZ,
            SFWriteNPY,
            SFWritePickle,
            )

        for cls, category in chain(
                zip(cls_read, repeat('read')),
                zip(cls_write, repeat('write')),
                ):
            runner = cls(fixture)
            record = [cls.__name__, cls.NUMBER, category, label]
            try:
                result = timeit.timeit(
                        f'runner()',
                        globals=locals(),
                        number=cls.NUMBER)
            except OSError:
                result = np.nan
            finally:
                runner.clear()
            record.append(result)
            records.append(record)

    f = sf.FrameGO.from_records(records,
            columns=('name', 'number', 'category', 'fixture', 'time')
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

    plot(f)
    # import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    # pandas_serialize()
    # get_sizes()
    run_test()

