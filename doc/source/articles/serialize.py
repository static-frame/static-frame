

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
import pandas as pd

sys.path.append(os.getcwd())

import static_frame as sf
from static_frame.core.display_color import HexColor
from static_frame.core.util import bytes_to_data_label


class FileIOTest:
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



class PDReadFeather(FileIOTest):
    SUFFIX = '.feather'

    def __init__(self, fixture: str):
        super().__init__(fixture)
        df = self.fixture.to_pandas().reset_index() # required for feather
        df.to_feather(self.fp)

    def __call__(self):
        f = pd.read_feather(self.fp)
        f = f.set_index('index')

class PDWriteFeather(FileIOTest):
    SUFFIX = '.feather'

    def __init__(self, fixture: str):
        super().__init__(fixture)
        self.df = self.fixture.to_pandas().reset_index() # required for feather

    def __call__(self):
        self.df.to_feather(self.fp)



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
        f, close = sf.Frame.from_npy_mmap(self.fp_dir)
        _ = f.loc[34715, 'zZbu']
        close()


#-------------------------------------------------------------------------------
NUMBER = 5

def scale(v):
    return int(v * 10)

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
    return f'OS: {platform.system()} / Pandas: {pd.__version__} / PyArrow: {pyarrow.__version__} / StaticFrame: {sf.__version__} / NumPy: {np.__version__}\n'

FIXTURE_SHAPE_MAP = {
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

def plot_performance(frame: sf.Frame):
    fixture_total = len(frame['fixture'].unique())
    cat_total = len(frame['category'].unique())
    name_total = len(frame['name'].unique())

    fig, axes = plt.subplots(cat_total, fixture_total)

    # for legend
    name_replace = {
        PDReadParquetArrow.__name__: 'Parquet\n(pd, snappy)',
        PDWriteParquetArrow.__name__: 'Parquet\n(pd, snappy)',
        PDReadParquetArrowNoComp.__name__: 'Parquet\n(pd, no compression)',
        PDWriteParquetArrowNoComp.__name__: 'Parquet\n(pd, no compression)',
        PDReadParquetFast.__name__: 'Parquet\n(pd, FastParquet)',
        PDWriteParquetFast.__name__: 'Parquet\n(pd, FastParquet)',
        PDReadFeather.__name__: 'Feather (pd)',
        PDWriteFeather.__name__: 'Feather (pd)',
        SFReadPickle.__name__: 'Pickle (sf)',
        SFWritePickle.__name__: 'Pickle (sf)',
        SFReadParquet.__name__: 'Parquet (sf)',
        SFWriteParquet.__name__: 'Parquet (sf)',
        SFReadNPZ.__name__: 'NPZ (sf)',
        SFWriteNPZ.__name__: 'NPZ (sf)',
        SFReadNPY.__name__: 'NPY (sf)',
        SFWriteNPY.__name__: 'NPY (sf)',
        SFReadNPYMM.__name__: 'NPY mmap (sf)'
    }

    name_order = {
        PDReadParquetArrow.__name__: 0,
        PDWriteParquetArrow.__name__: 0,
        PDReadParquetArrowNoComp.__name__: 1,
        PDWriteParquetArrowNoComp.__name__: 1,
        PDReadParquetFast.__name__: 2,
        PDWriteParquetFast.__name__: 2,
        PDReadFeather.__name__: 3,
        PDWriteFeather.__name__: 3,
        SFReadParquet.__name__: 4,
        SFWriteParquet.__name__: 4,
        SFReadNPZ.__name__: 5,
        SFWriteNPZ.__name__: 5,
        SFReadNPY.__name__: 6,
        SFWriteNPY.__name__: 6,
        SFReadNPYMM.__name__: 7,
        SFReadPickle.__name__: 8,
        SFWritePickle.__name__: 8,
    }

    # cmap = plt.get_cmap('terrain')
    cmap = plt.get_cmap('plasma')

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
            title = f'{cat_io.title()}\n{cat_dtype.title()}\n{FIXTURE_SHAPE_MAP[fixture_label]}'
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

    fig.set_size_inches(6, 3.5) # width, height
    fig.legend(post, names_display, loc='center right', fontsize=8)
    # horizontal, vertical
    count = ff.parse(FF_tall_uniform).size
    fig.text(.05, .97, f'NPZ Performance: {count:.0e} Elements, {NUMBER} Iterations', fontsize=10)
    fig.text(.05, .91, get_versions(), fontsize=6)
    # get fixtures size reference
    shape_map = {shape: FIXTURE_SHAPE_MAP[shape] for shape in frame['fixture'].unique()}
    shape_msg = ' / '.join(f'{v}: {k}' for k, v in shape_map.items())
    fig.text(.05, .91, shape_msg, fontsize=6)

    fp = '/tmp/serialize.png'
    plt.subplots_adjust(
            left=0.05,
            bottom=0.05,
            right=0.75,
            top=0.75,
            wspace=-0.2, # width
            hspace=1,
            )
    # plt.rcParams.update({'font.size': 22})
    plt.savefig(fp, dpi=300)

    if sys.platform.startswith('linux'):
        os.system(f'eog {fp}&')
    else:
        os.system(f'open {fp}')


#-------------------------------------------------------------------------------


def plot_size(frame: sf.Frame):
    # for legend
    name_replace = {
        'parquet': 'Parquet\n(pd, snappy)',
        'parquet_noc': 'Parquet\n(pd, no compression)',
        # PDReadFeather.__name__: 'Feather (pd)',
        'pickle': 'Pickle (sf)',
        'npz': 'NPZ (sf)',
        'npy': 'NPY (sf)',
    }

    fixture_total = len(frame)
    names = ('parquet', 'parquet_noc', 'npz', 'pickle')
    name_total = len(names)

    fig, axes = plt.subplots(3, 3)

    # cmap = plt.get_cmap('terrain')
    cmap = plt.get_cmap('plasma')
    color = cmap(np.arange(name_total) / name_total)

    sl_to_pos = {'tall': 0, 'square': 1, 'wide': 2}
    cl_to_pos = {'columnar': 0, 'mixed': 1, 'uniform': 2}

    # categories are read, write
    for fixture_count, (fixture_label, row) in enumerate(frame.iter_series_items(axis=1)):
        shape_label, dtype_label = fixture_label.split('_')

        # ax = axes[sl_to_pos[shape_label]][cl_to_pos[dtype_label]]
        ax = axes[cl_to_pos[dtype_label]][sl_to_pos[shape_label]]

        x = np.arange(name_total)
    #     names_display = [name_replace[l] for l in names]
        results = row[list(names)].values
        post = ax.bar(names, results, color=color)
        shape_key = f"{row['shape'][0]}x{row['shape'][1]}"
        title = f'{dtype_label.title()}\n{FIXTURE_SHAPE_MAP[shape_key]}'
        # title = f'{cat_io.title()}\n{cat_dtype.title()}\n{FIXTURE_SHAPE_MAP[fixture_label]}'


        ax.set_title(title, fontsize=8)
        ax.set_box_aspect(0.75) # makes taller tan wide
        size_max = results.max()
        ax.set_yticks([0, size_max * 0.5, size_max])
        ax.set_yticklabels(['',
                bytes_to_data_label(size_max * 0.5),
                bytes_to_data_label(size_max),
                ], fontsize=6)
        ax.tick_params(
                axis='x',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False,
                )

    fig.set_size_inches(6, 3.5) # width, height
    fig.legend(post, [name_replace[n] for n in names], loc='center right', fontsize=8)
    # horizontal, vertical
    count = ff.parse(FF_tall_uniform).size
    fig.text(.05, .97, f'NPZ Size: {count:.0e} Elements', fontsize=10)
    fig.text(.05, .91, get_versions(), fontsize=6)
    # get fixtures size reference
    shape_map = {shape: FIXTURE_SHAPE_MAP[f'{shape[0]}x{shape[1]}'] for shape in frame['shape'].unique()}
    shape_msg = ' / '.join(f'{v}: {k}' for k, v in shape_map.items())
    fig.text(.05, .91, shape_msg, fontsize=6)

    fp = '/tmp/serialize-size.png'
    plt.subplots_adjust(
            left=0.05,
            bottom=0.05,
            right=0.75,
            top=0.75,
            wspace=-0.2, # width
            hspace=1,
            )
    # plt.rcParams.update({'font.size': 22})
    plt.savefig(fp, dpi=300)

    if sys.platform.startswith('linux'):
        os.system(f'eog {fp}&')
    else:
        os.system(f'open {fp}')

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
        record.append(size_parquet)
        record.append(bytes_to_data_label(size_parquet))

        _, fp = tempfile.mkstemp(suffix='.parquet')
        df.to_parquet(fp, compression=None)
        size_parquet_noc = os.path.getsize(fp)
        os.unlink(fp)
        record.append(size_parquet_noc)
        record.append(bytes_to_data_label(size_parquet_noc))

        _, fp = tempfile.mkstemp(suffix='.npz')
        f.to_npz(fp, include_columns=True)
        size_npz = os.path.getsize(fp)
        os.unlink(fp)
        record.append(size_npz)
        record.append(bytes_to_data_label(size_npz))

        _, fp = tempfile.mkstemp(suffix='.pickle')
        file = open(fp, 'wb')
        pickle.dump(f, file)
        file.close()
        size_pickle = os.path.getsize(fp)
        os.unlink(fp)
        record.append(size_pickle)
        record.append(bytes_to_data_label(size_pickle))

        record.append(round(size_npz / size_parquet, 3))
        record.append(round(size_npz / size_parquet_noc, 3))

        records.append(record)

    f = sf.Frame.from_records(records,
            columns=('fixture',
            'shape',
            'parquet',
            'parquet_hr', # human readable
            'parquet_noc',
            'parquet_noc_hr',
            'npz',
            'npz_hr',
            'pickle',
            'pickle_hr',

            'npz/parquet',
            'npz/parquet_noc'
            )).set_index('fixture', drop=True)

    print(f.display_wide())
    plot_size(f)


def pandas_serialize_test():
    import pandas as pd
    df = ff.parse('s(10,10)|v(int,int,bool,float,float)|i(I,int)|c(I,str)').rename('foo').to_pandas().set_index(['zZbu', 'ztsv'])
    df = df.reindex(columns=pd.Index(df.columns, name='foo'))


    # feather
    # ValueError: feather does not support serializing <class 'pandas.core.indexes.base.Index'> for the index; you can .reset_index() to make the index into column(s)
    # feather must have string column names


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

CLS_READ = (
    PDReadParquetArrow,
    PDReadParquetArrowNoComp,
    # PDReadParquetFast, # not faster!
    # PDReadFeather,

    # SFReadParquet,
    SFReadNPZ,
    # SFReadNPY,
    SFReadNPYMM,
    SFReadPickle,
    )
CLS_WRITE = (
    # PDWriteParquetArrow,
    # PDWriteParquetArrowNoComp,
    # PDWriteParquetFast, # not faster!
    # SFWriteParquet,
    # PDWriteFeather,
    SFWriteNPZ,
    # SFWriteNPY,
    SFWritePickle,
    )


def run_test(
        include_read: bool = True,
        include_write: bool = True,
        ):
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

        for cls, category_prefix in chain(
                (zip(CLS_READ, repeat('read')) if include_read else ()),
                (zip(CLS_WRITE, repeat('write')) if include_write else ()),
                ):
            runner = cls(fixture)
            category = f'{category_prefix} {dtype_hetero}'

            record = [cls.__name__, NUMBER, category, fixture_label]
            try:
                result = timeit.timeit(
                        f'runner()',
                        globals=locals(),
                        number=NUMBER)
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

    plot_performance(f)

if __name__ == '__main__':
    # pandas_serialize_test()
    get_sizes()
    # run_test(include_read=True, include_write=False)
    # run_test(include_read=False, include_write=True)


