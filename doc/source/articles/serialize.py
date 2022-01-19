

import os
import timeit
import tempfile
import typing as tp
import pickle
import shutil
import sys
import os


import matplotlib.pyplot as plt
import numpy as np
import frame_fixtures as ff
import static_frame as sf
import pandas as pd

from static_frame.core.display_color import HexColor

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


class PDReadParquetArrowNoComp(FileIOTest):
    SUFFIX = '.parquet'

    def __init__(self, fixture: str):
        super().__init__(fixture)
        df = self.fixture.to_pandas()
        df.to_parquet(self.fp, compression=None)

    def __call__(self):
        f = pd.read_parquet(self.fp)
        _ = f.loc[34715, 'zZbu']

class PDWriteParquetArrowNoComp(FileIOTest):
    SUFFIX = '.parquet'

    def __init__(self, fixture: str):
        super().__init__(fixture)
        self.df = self.fixture.to_pandas()

    def __call__(self):
        self.df.to_parquet(self.fp, compression=None)


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



#-------------------------------------------------------------------------------
def scale(v):
    return int(v * 1)

FF_wide_uniform = f's({scale(100)},{scale(10_000)})|v(float)|i(I,int)|c(I,str)'
FF_wide_mixed   = f's({scale(100)},{scale(10_000)})|v(int,int,bool,float,float)|i(I,int)|c(I,str)'
FF_wide_columnar   = f's({scale(100)},{scale(10_000)})|v(int,bool,float)|i(I,int)|c(I,str)'


FF_tall_uniform = f's({scale(10_000)},{scale(100)})|v(float)|i(I,int)|c(I,str)'
FF_tall_mixed   = f's({scale(10_000)},{scale(100)})|v(int,int,bool,float,float)|i(I,int)|c(I,str)'
FF_tall_columnar   = f's({scale(10_000)},{scale(100)})|v(int,bool,float)|i(I,int)|c(I,str)'

FF_square_uniform = f's({scale(1_000)},{scale(1_000)})|v(float)|i(I,int)|c(I,str)'
FF_square_mixed   = f's({scale(1_000)},{scale(1_000)})|v(int,int,bool,float,float)|i(I,int)|c(I,str)'
FF_square_columnar   = f's({scale(1_000)},{scale(1_000)})|v(int,bool,float)|i(I,int)|c(I,str)'

#-------------------------------------------------------------------------------

def get_versions() -> str:
    import platform
    import pyarrow
    return f'OS: {platform.system()} / Pandas: {pd.__version__} / PyArrow: {pyarrow.__version__} / StaticFrame: {sf.__version__}\n'

def plot(frame: sf.Frame):
    fixture_total = len(frame['fixture'].unique())
    cat_total = len(frame['category'].unique())
    name_total = len(frame['name'].unique())

    fig, axes = plt.subplots(cat_total, fixture_total)

    # for legend
    name_replace = {
        PDReadParquetArrow.__name__: 'Parquet\n (pd, snappy)',
        PDWriteParquetArrow.__name__: 'Parquet\n (pd, snappy)',
        PDReadParquetArrowNoComp.__name__: 'Parquet\n(pd, no compression)',
        PDWriteParquetArrowNoComp.__name__: 'Parquet\n(pd, no compression)',
        PDReadParquetFast.__name__: 'Parquet\n(pd, FastParquet)',
        PDWriteParquetFast.__name__: 'Parquet\n(pd, FastParquet)',
        SFReadPickle.__name__: 'Pickle (sf)',
        SFWritePickle.__name__: 'Pickle (sf)',
        SFReadParquet.__name__: 'Parquet (sf)',
        SFWriteParquet.__name__: 'Parquet (sf)',
        SFReadNPZ.__name__: 'NPZ (sf)',
        SFWriteNPZ.__name__: 'NPZ (sf)',
        SFReadNPY.__name__: 'NPY (sf)',
        SFWriteNPY.__name__: 'NPY (sf)',
    }

    name_order = {
        PDReadParquetArrow.__name__: 0,
        PDWriteParquetArrow.__name__: 0,
        PDReadParquetArrowNoComp.__name__: 1,
        PDWriteParquetArrowNoComp.__name__: 1,
        PDReadParquetFast.__name__: 2,
        PDWriteParquetFast.__name__: 2,
        SFReadParquet.__name__: 3,
        SFWriteParquet.__name__: 3,
        SFReadNPZ.__name__: 4,
        SFWriteNPZ.__name__: 4,
        SFReadNPY.__name__: 5,
        SFWriteNPY.__name__: 5,
        SFReadPickle.__name__: 6,
        SFWritePickle.__name__: 6,
    }

    fixture_shape_map = {
        '1000x10': 'Tall',
        '100x100': 'Square',
        '10x1000': 'Wide',
        '10000x100': 'Tall',
        '1000x1000': 'Square',
        '100x10000': 'Wide',
        '100000x1000': 'Tall',
        '10000x10000': 'Square',
        '1000x100000': 'Wide',
    }

    cmap = plt.get_cmap('terrain')
    color = cmap(np.arange(name_total) / name_total)

    # categories are read, write
    for cat_count, (cat_label, cat) in enumerate(frame.iter_group_items('category')):
        for fixture_count, (fixture_label, fixture) in enumerate(
                cat.iter_group_items('fixture')):
            ax = axes[cat_count][fixture_count]

            # set order
            fixture = fixture.sort_values('name', key=lambda s:s.iter_element().map_all(name_order))
            results = fixture['time'].values.tolist()
            names = fixture['name'].values.tolist()
            x = np.arange(len(results))
            names_display = [name_replace[l] for l in names]
            post = ax.bar(names_display, results, color=color)

            # ax.set_ylabel()
            cat_io, cat_dtype = cat_label.split(' ')
            title = f'{cat_io.title()}\n{cat_dtype.title()}\n{fixture_shape_map[fixture_label]}'
            ax.set_title(title, fontsize=8)
            ax.set_box_aspect(0.75) # makes taller tan wide
            time_max = fixture['time'].max()
            ax.set_yticks([0, time_max * 0.5, time_max])
            ax.set_yticklabels(['',
                    f'{time_max * 0.5:.3f} (s)',
                    f'{time_max:.3f} (s)',
                    ], fontsize=6)
            # ax.set_xticks(x, names_display, rotation='vertical')
            ax.tick_params(
                    axis='x',
                    which='both',
                    bottom=False,
                    top=False,
                    labelbottom=False,
                    )


    fig.set_size_inches(6, 7) # width, height
    fig.legend(post, names_display, loc='center right', fontsize=8)
    # horizontal, vertical
    count = ff.parse(FF_tall_uniform).size
    fig.text(.14, .97, f'NPY & NPZ Performance: {count:.0e} Elements', fontsize=10)
    fig.text(.14, .92, get_versions(), fontsize=8)
    # get fixtures size reference
    shape_map = {shape: fixture_shape_map[shape] for shape in frame['fixture'].unique()}
    shape_msg = ' / '.join(f'{v}: {k}' for k, v in shape_map.items())
    fig.text(.14, .915, shape_msg, fontsize=8)

    fp = '/tmp/serialize.png'
    plt.subplots_adjust(
            left=0.05,
            bottom=0.05,
            right=0.75,
            top=0.82,
            wspace=-0.2, # width
            hspace=1.5,
            )
    # plt.rcParams.update({'font.size': 22})
    plt.savefig(fp, dpi=300)

    if sys.platform.startswith('linux'):
        os.system(f'eog {fp}&')
    else:
        os.system(f'open {fp}')

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
            ('wide_uniform', FF_wide_uniform),
            ('wide_mixed', FF_wide_mixed),
            ('wide_columnar', FF_wide_columnar),

            ('tall_uniform', FF_tall_uniform),
            ('tall_mixed', FF_tall_mixed),
            ('tall_columnar', FF_tall_columnar),

            ('square_uniform', FF_square_uniform),
            ('square_mixed', FF_square_mixed),
            ('square_columnar', FF_square_columnar),

            ):
        f = ff.parse(fixture)
        df = f.to_pandas()
        record = [label, f.shape]

        _, fp = tempfile.mkstemp(suffix='.parquet')
        df.to_parquet(fp)
        size_parquet = os.path.getsize(fp)
        os.unlink(fp)
        record.append(convert_size(size_parquet))

        _, fp = tempfile.mkstemp(suffix='.parquet')
        df.to_parquet(fp, compression=None)
        size_parquet_noc = os.path.getsize(fp)
        os.unlink(fp)
        record.append(convert_size(size_parquet_noc))

        _, fp = tempfile.mkstemp(suffix='.npz')
        f.to_npz(fp, include_columns=True)
        size_npz = os.path.getsize(fp)
        os.unlink(fp)
        record.append(convert_size(size_npz))

        _, fp = tempfile.mkstemp(suffix='.pickle')
        file = open(fp, 'wb')
        pickle.dump(f, file)
        file.close()
        size_pickle = os.path.getsize(fp)
        os.unlink(fp)
        record.append(convert_size(size_pickle))

        record.append(round(size_npz / size_parquet, 3))
        record.append(round(size_npz / size_parquet_noc, 3))

        records.append(record)

    f = sf.Frame.from_records(records,
            columns=('fixture',
            'shape',
            'parquet',
            'parquet_noc',
            'npz',
            'pickle',
            'npz/parquet',
            'npz/parquet_noc'
            )).set_index('fixture', drop=True)

    print(f.display_wide())


def pandas_serialize_test():
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

def fixture_to_pair(label: str, fixture: str) -> tp.Tuple[str, str, str]:
    # get a title
    f = ff.parse(fixture)
    return label, f'{f.shape[0]:}x{f.shape[1]}', fixture

def run_test():
    records = []
    for dtype_hetero, fixture_label, fixture in (
            fixture_to_pair('uniform', FF_wide_uniform),
            fixture_to_pair('mixed', FF_wide_mixed),
            fixture_to_pair('columnar', FF_wide_columnar),

            fixture_to_pair('uniform', FF_tall_uniform),
            fixture_to_pair('mixed', FF_tall_mixed),
            fixture_to_pair('columnar', FF_tall_columnar),

            fixture_to_pair('uniform', FF_square_uniform),
            fixture_to_pair('mixed', FF_square_mixed),
            fixture_to_pair('columnar', FF_square_columnar),
            ):
        cls_read = (
            PDReadParquetArrow,
            PDReadParquetArrowNoComp,
            # PDReadParquetFast, # not faster!
            # SFReadParquet,
            SFReadNPZ,
            SFReadNPY,
            SFReadPickle,
            # SFReadNPYMM,
            )
        cls_write = (
            PDWriteParquetArrow,
            PDWriteParquetArrowNoComp,
            # PDWriteParquetFast, # not faster!
            # SFWriteParquet,
            SFWriteNPZ,
            SFWriteNPY,
            SFWritePickle,
            )

        for cls, category_prefix in chain(
                zip(cls_read, repeat('read')),
                zip(cls_write, repeat('write')),
                ):
            runner = cls(fixture)
            category = f'{category_prefix} {dtype_hetero}'

            record = [cls.__name__, cls.NUMBER, category, fixture_label]
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
    # pandas_serialize_test()
    # get_sizes()
    run_test()

