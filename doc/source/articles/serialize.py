
import hashlib
import os
import pickle
import shutil
import sys
import tempfile
import timeit
from pathlib import Path

import frame_fixtures as ff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typing_extensions as tp

sys.path.append(os.getcwd())

import static_frame as sf
from static_frame.core.display_color import HexColor
from static_frame.core.util import bytes_to_size_label


def ff_cached(fmt: str) -> sf.TFrameAny:
    h = hashlib.sha256(bytes(fmt, 'utf-8')).hexdigest()
    fp = Path('/tmp') / f"{h}.npz"
    if fp.exists():
        return sf.Frame.from_npz(fp)
    f = ff.parse(fmt)
    f.to_npz(fp)
    return f

class FileIOTest:
    SUFFIX = '.tmp'

    def __init__(self, fixture: str | Path):
        if isinstance(fixture, Path):
            self.fixture = sf.Frame.from_csv(fixture)
        else:
            self.fixture = ff_cached(fixture)

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
        # _ = f.loc[34715, 'zZbu']
        _ = f.loc[14863776, 'total_amount']

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
        # _ = f.loc[34715, 'zZbu']
        _ = f.loc[14863776, 'total_amount']

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
        # _ = f.loc[34715, 'zZbu']
        _ = f.loc[14863776, 'total_amount']

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
        # _ = f.loc[34715, 'zZbu']
        _ = f.loc[14863776, 'total_amount']

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
        df = self.fixture.to_pandas()
        df.to_feather(self.fp, compression='lz4')

    def __call__(self):
        f = pd.read_feather(self.fp)
        # _ = f.loc[34715, 'zZbu']
        _ = f.loc[14863776, 'total_amount']

class PDWriteFeather(FileIOTest):
    SUFFIX = '.feather'

    def __init__(self, fixture: str):
        super().__init__(fixture)
        self.df = self.fixture.to_pandas()

    def __call__(self):
        self.df.to_feather(self.fp, compression='lz4')


class PDReadFeatherNoComp(FileIOTest):
    SUFFIX = '.feather'

    def __init__(self, fixture: str):
        super().__init__(fixture)
        df = self.fixture.to_pandas()
        df.to_feather(self.fp, compression='uncompressed')

    def __call__(self):
        f = pd.read_feather(self.fp)
        # _ = f.loc[34715, 'zZbu']
        _ = f.loc[14863776, 'total_amount']


class PDWriteFeatherNoComp(FileIOTest):
    SUFFIX = '.feather'

    def __init__(self, fixture: str):
        super().__init__(fixture)
        self.df = self.fixture.to_pandas()

    def __call__(self):
        self.df.to_feather(self.fp, compression='uncompressed')



class SFReadNPZ(FileIOTest):
    SUFFIX = '.npz'

    def __init__(self, fixture: str):
        super().__init__(fixture)
        self.fixture.to_npz(self.fp)

    def __call__(self):
        f = sf.Frame.from_npz(self.fp)
        # _ = f.loc[34715, 'zZbu']
        _ = f.loc[14863776, 'total_amount']


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
            # _ = f.loc[34715, 'zZbu']
            _ = f.loc[14863776, 'total_amount']


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
        # _ = f.loc[34715, 'zZbu']
        _ = f.loc[14863776, 'total_amount']


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
        # _ = f.loc[34715, 'zZbu']
        _ = f.loc[14863776, 'total_amount']
        close()




#-------------------------------------------------------------------------------

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
FF_square_columnar = f's({scale(1_000)},{scale(1_000)})|v(int,bool,float)|i(I,int)|c(I,str)'



#-------------------------------------------------------------------------------

def seconds_to_display(seconds: float, number: int) -> str:
    seconds /= number
    if seconds < 1e-4:
        return f'{seconds * 1e6: .1f} (µs)'
    if seconds < 1e-1:
        return f'{seconds * 1e3: .1f} (ms)'
    return f'{seconds: .1f} (s)'

def get_versions() -> str:
    import platform

    import pyarrow
    return f'OS: {platform.system()} / Python: {platform.python_version()} / Pandas: {pd.__version__} / PyArrow: {pyarrow.__version__} / StaticFrame: {sf.__version__} / NumPy: {np.__version__}\n'

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
# for legend
CLS_NAME_TO_DISPLAY = {
    PDReadParquetArrow.__name__: 'Parquet\n(Pandas, snappy)',
    PDWriteParquetArrow.__name__: 'Parquet\n(Pandas, snappy)',
    PDReadParquetArrowNoComp.__name__: 'Parquet\n(Pandas, no compression)',
    PDWriteParquetArrowNoComp.__name__: 'Parquet\n(Pandas, no compression)',
    PDReadParquetFast.__name__: 'Parquet\n(Pandas, FastParquet)',
    PDWriteParquetFast.__name__: 'Parquet\n(Pandas, FastParquet)',
    PDReadFeather.__name__: 'Feather\n(Pandas, lz4)',
    PDWriteFeather.__name__: 'Feather\n(Pandas, lz4)',
    PDReadFeatherNoComp.__name__: 'Feather\n(Pandas, no compression)',
    PDWriteFeatherNoComp.__name__: 'Feather\n(Pandas, no compression)',

    SFReadPickle.__name__: 'Pickle (StaticFrame)',
    SFWritePickle.__name__: 'Pickle (StaticFrame)',
    SFReadParquet.__name__: 'Parquet (StaticFrame)',
    SFWriteParquet.__name__: 'Parquet (StaticFrame)',
    SFReadNPZ.__name__: 'NPZ (StaticFrame)',
    SFWriteNPZ.__name__: 'NPZ (StaticFrame)',
    SFReadNPY.__name__: 'NPY (StaticFrame)',
    SFWriteNPY.__name__: 'NPY (StaticFrame)',
    SFReadNPYMM.__name__: 'NPY mmap (StaticFrame)'
}

CLS_NAME_TO_ORDER = {
    PDReadParquetArrow.__name__: 0,
    PDWriteParquetArrow.__name__: 0,
    PDReadParquetArrowNoComp.__name__: 1,
    PDWriteParquetArrowNoComp.__name__: 1,
    PDReadParquetFast.__name__: 2,
    PDWriteParquetFast.__name__: 2,
    PDReadFeather.__name__: 3,
    PDWriteFeather.__name__: 3,
    PDReadFeatherNoComp.__name__: 4,
    PDWriteFeatherNoComp.__name__: 4,

    SFReadParquet.__name__: 5,
    SFWriteParquet.__name__: 5,
    SFReadNPZ.__name__: 6,
    SFWriteNPZ.__name__: 6,
    SFReadNPY.__name__: 7,
    SFWriteNPY.__name__: 7,
    SFReadNPYMM.__name__: 7,
    SFReadPickle.__name__: 8,
    SFWritePickle.__name__: 8,
}

def plot_ff_performance(
        frame: sf.Frame,
        *,
        number: int,
        fp: str = '/tmp/serialize.png',
        log_scale: bool = False,
        title: str = 'NPZ Performance',
        ):
    fixture_total = len(frame['fixture'].unique())
    cat_total = len(frame['category'].unique())
    name_total = len(frame['name'].unique())

    fig, axes = plt.subplots(cat_total, fixture_total, squeeze=False)


    # cmap = plt.get_cmap('terrain')
    cmap = plt.get_cmap('plasma')

    color = cmap(np.arange(name_total) / name_total)

    # categories are read, write
    for cat_count, (cat_label, cat) in enumerate(frame.iter_group_items('category')):
        for fixture_count, (fixture_label, fixture) in enumerate(
                cat.iter_group_items('fixture')):
            ax = axes[cat_count][fixture_count]
            fixture = fixture.sort_values('name', key=lambda s:s.iter_element().map_all(CLS_NAME_TO_ORDER))
            results = fixture['time'].values.tolist()

            x_labels = [f'{i}: {CLS_NAME_TO_DISPLAY[name]}' for i, name in
                    zip(range(1, len(results) + 1),
                    fixture['name'].values)
                    ]
            x_tick_labels = [str(l + 1) for l in range(len(x_labels))]
            x = np.arange(len(results))
            x_bar = ax.bar(x_labels, results, color=color)

            cat_io, cat_dtype = cat_label.split(' ')
            plot_title = f'{cat_dtype.title()}\n{FIXTURE_SHAPE_MAP[fixture_label]}'

            ax.set_title(plot_title, fontsize=6)
            ax.set_box_aspect(0.75) # makes taller tan wide

            time_max = fixture["time"].max()
            time_min = fixture["time"].min()

            if log_scale:
                ax.set_yscale('log')
                y_ticks = []
                for v in range(
                        math.floor(math.log(time_min, 10)),
                        math.floor(math.log(time_max, 10)) + 1,
                        ):
                    y_ticks.append(1 * pow(10, v))
                ax.set_yticks(y_ticks)
            else:
                y_ticks = [0, time_min, time_max * 0.5, time_max]
                y_labels = [
                    "",
                    seconds_to_display(time_min, number),
                    seconds_to_display(time_max * 0.5, number),
                    seconds_to_display(time_max, number),
                ]
                if time_min > time_max * 0.25:
                    # remove the min if it is greater than quarter
                    y_ticks.pop(1)
                    y_labels.pop(1)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_labels)

            ax.tick_params(
                axis="y",
                length=2,
                width=0.5,
                pad=1,
                labelsize=4,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(x_tick_labels)
            ax.tick_params(
                axis="x",
                length=2,
                width=0.5,
                pad=1,
                labelsize=4,
            )

    fig.set_size_inches(5, 3) # width, height
    fig.legend(x_bar, x_labels, loc='center right', fontsize=6)
    # horizontal, vertical
    count = ff_cached(FF_tall_uniform).size
    fig.text(.05, .96, f'{title}: {count:.0e} Elements, {number} Iterations', fontsize=10)
    fig.text(.05, .90, get_versions(), fontsize=6)
    # get fixtures size reference
    shape_map = {shape: FIXTURE_SHAPE_MAP[shape] for shape in frame['fixture'].unique()}
    shape_msg = ' / '.join(f'{v}: {k}' for k, v in shape_map.items())
    fig.text(.05, .90, shape_msg, fontsize=6)

    plt.subplots_adjust(
            left=0.05,
            bottom=0.05,
            right=0.75,
            top=0.75,
            wspace=-0.3, # width
            hspace=1,
            )
    # plt.rcParams.update({'font.size': 22})
    plt.savefig(fp, dpi=300)

    if sys.platform.startswith('linux'):
        os.system(f'eog {fp}&')
    else:
        os.system(f'open {fp}')


def plot_file_performance(
        frame: sf.Frame,
        *,
        number: int,
        fp: str = '/tmp/serialize.png',
        log_scale: bool = False,
        title: str = 'NPZ Performance',
        ):
    fixture_total = len(frame['fixture'].unique())
    cat_total = len(frame['category'].unique())
    name_total = len(frame['name'].unique())

    # NOTE cat_total, order flipped
    fig, axes = plt.subplots(fixture_total, cat_total, squeeze=False)
    cmap = plt.get_cmap('plasma')
    color = cmap(np.arange(name_total) / name_total)

    # categories are read, write
    for cat_count, (cat_label, cat) in enumerate(frame.iter_group_items('category')):
        for fixture_count, (fixture_label, fixture) in enumerate(
                cat.iter_group_items('fixture')):

            ax = axes[fixture_count][cat_count]
            fixture = fixture.sort_values('name', key=lambda s:s.iter_element().map_all(CLS_NAME_TO_ORDER))
            results = fixture['time'].values.tolist()

            x_labels = [f'{i}: {CLS_NAME_TO_DISPLAY[name]}' for i, name in
                    zip(range(1, len(results) + 1),
                    fixture['name'].values)
                    ]
            x_tick_labels = [str(l + 1) for l in range(len(x_labels))]
            x = np.arange(len(results))
            x_bar = ax.bar(x_labels, results, color=color)

            # NOTE: fixture_label is a Path
            plot_title = f'{cat_label.title()}\n{fixture_label.stem}'
            ax.set_title(plot_title, fontsize=6)
            ax.set_box_aspect(0.75) # makes taller tan wide

            time_max = fixture["time"].max()
            time_min = fixture["time"].min()

            if log_scale:
                ax.set_yscale('log')
                y_ticks = []
                for v in range(
                        math.floor(math.log(time_min, 10)),
                        math.floor(math.log(time_max, 10)) + 1,
                        ):
                    y_ticks.append(1 * pow(10, v))
                ax.set_yticks(y_ticks)
            else:
                y_ticks = [0, time_min, time_max * 0.5, time_max]
                y_labels = [
                    "",
                    seconds_to_display(time_min, number),
                    seconds_to_display(time_max * 0.5, number),
                    seconds_to_display(time_max, number),
                ]
                if time_min > time_max * 0.25:
                    # remove the min if it is greater than quarter
                    y_ticks.pop(1)
                    y_labels.pop(1)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_labels)

            ax.tick_params(
                axis="y",
                length=2,
                width=0.5,
                pad=1,
                labelsize=4,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(x_tick_labels)
            ax.tick_params(
                axis="x",
                length=2,
                width=0.5,
                pad=1,
                labelsize=4,
            )

    fig.set_size_inches(5, 2) # width, height
    fig.legend(x_bar, x_labels, loc='center right', fontsize=6)
    # horizontal, vertical
    fig.text(.05, .92, f'{title}: {number} Iterations', fontsize=10)
    fig.text(.05, .82, get_versions(), fontsize=6)

    plt.subplots_adjust(
            left=0.1,
            bottom=0.05,
            right=0.65,
            top=0.75,
            wspace=0.5, # width
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
    CLS_NAME_TO_DISPLAY = {
        'parquet': 'Parquet\n(Pandas, snappy)',
        'parquet_noc': 'Parquet\n(Pandas, no compression)',
        'feather': 'Feather\n(Pandas, lz4)',
        'feather_noc': 'Feather\n(Pandas, no compression)',
        'pickle': 'Pickle (StaticFrame)',
        'npz': 'NPZ (StaticFrame)',
        'npy': 'NPY (StaticFrame)',
    }

    # fixture_total = len(frame)
    names = ('parquet', 'parquet_noc', 'feather', 'feather_noc', 'npz')
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
        results = row[list(names)].values

        x_labels = [f'{i}: {CLS_NAME_TO_DISPLAY[name]}' for i, name in
                zip(range(1, name_total + 1), names)
                ]

        x_tick_labels = [str(l + 1) for l in range(len(x_labels))]
        x = np.arange(name_total)
        x_bar = ax.bar(x_labels, results, color=color)

        shape_key = f"{row['shape'][0]}x{row['shape'][1]}"
        plot_title = f'{dtype_label.title()}\n{FIXTURE_SHAPE_MAP[shape_key]}'

        ax.set_title(plot_title, fontsize=6)
        ax.set_box_aspect(0.75) # makes taller tan wide
        size_max = results.max()
        ax.set_yticks([0, size_max * 0.5, size_max])
        ax.set_yticklabels(['',
                bytes_to_size_label(size_max * 0.5),
                bytes_to_size_label(size_max),
                ], fontsize=6)
        ax.tick_params(
            axis="y",
            length=2,
            width=0.5,
            pad=1,
            labelsize=4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(x_tick_labels)
        ax.tick_params(
            axis="x",
            length=2,
            width=0.5,
            pad=1,
            labelsize=4,
            )

    fig.set_size_inches(5, 3) # width, height
    fig.legend(x_bar, x_labels, loc='center right', fontsize=6)
    # horizontal, vertical
    count = ff_cached(FF_tall_uniform).size
    fig.text(.05, .96, f'NPZ Size: {count:.0e} Elements', fontsize=10)
    fig.text(.05, .90, get_versions(), fontsize=6)
    # get fixtures size reference

    shape_order = [] #frame[['shape']].to_frame_go()
    cl = next(iter(cl_to_pos.keys())) # get one to draw examples
    for shape_label in sl_to_pos.keys():
        shape_order.append(frame.loc[f'{shape_label}_{cl}', 'shape'])

    shape_map = {f'{shape[0]}x{shape[1]}':
            FIXTURE_SHAPE_MAP[f'{shape[0]}x{shape[1]}']
            for shape in shape_order}

    shape_msg = ' / '.join(f'{v}: {k}' for k, v in shape_map.items())
    fig.text(.05, .90, shape_msg, fontsize=6)

    fp = '/tmp/serialize-size.png'
    plt.subplots_adjust(
            left=0.05,
            bottom=0.05,
            right=0.75,
            top=0.75,
            wspace=-0.3, # width
            hspace=1,
            )
    # plt.rcParams.update({'font.size': 22})
    plt.savefig(fp, dpi=300)

    if sys.platform.startswith('linux'):
        os.system(f'eog {fp}&')
    else:
        os.system(f'open {fp}')

def run_size_test():
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
        f = ff_cached(fixture)
        df = f.to_pandas()
        record = [label, f.shape]

        _, fp = tempfile.mkstemp(suffix='.parquet')
        df.to_parquet(fp)
        size_parquet = os.path.getsize(fp)
        os.unlink(fp)
        record.append(size_parquet)
        record.append(bytes_to_size_label(size_parquet))

        _, fp = tempfile.mkstemp(suffix='.parquet')
        df.to_parquet(fp, compression=None)
        size_parquet_noc = os.path.getsize(fp)
        os.unlink(fp)
        record.append(size_parquet_noc)
        record.append(bytes_to_size_label(size_parquet_noc))


        _, fp = tempfile.mkstemp(suffix='.feather')
        df.to_feather(fp, compression='lz4')
        size_feather = os.path.getsize(fp)
        os.unlink(fp)
        record.append(size_feather)
        record.append(bytes_to_size_label(size_feather))

        _, fp = tempfile.mkstemp(suffix='.feather')
        df.to_feather(fp, compression='uncompressed')
        size_feather = os.path.getsize(fp)
        os.unlink(fp)
        record.append(size_feather)
        record.append(bytes_to_size_label(size_feather))


        _, fp = tempfile.mkstemp(suffix='.npz')
        f.to_npz(fp, include_columns=True)
        size_npz = os.path.getsize(fp)
        os.unlink(fp)
        record.append(size_npz)
        record.append(bytes_to_size_label(size_npz))

        # _, fp = tempfile.mkstemp(suffix='.pickle')
        # file = open(fp, 'wb')
        # pickle.dump(f, file)
        # file.close()
        # size_pickle = os.path.getsize(fp)
        # os.unlink(fp)
        # record.append(size_pickle)
        # record.append(bytes_to_size_label(size_pickle))

        # record.append(round(size_npz / size_parquet, 3))
        # record.append(round(size_npz / size_parquet_noc, 3))

        records.append(record)

    f = sf.Frame.from_records(records,
            columns=('fixture',
            'shape',
            'parquet',
            'parquet_hr', # human readable
            'parquet_noc',
            'parquet_noc_hr',
            'feather',
            'feather_hr',
            'feather_noc',
            'feather_noc_hr',
            'npz',
            'npz_hr',
            )).set_index('fixture', drop=True)

    print(f.display_wide())
    plot_size(f)


def pandas_serialize_test():
    import pandas as pd
    df = ff_cached('s(10,10)|v(int,int,bool,float,float)|i(I,int)|c(I,str)').rename('foo').to_pandas().set_index(['zZbu', 'ztsv'])
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

from itertools import chain
from itertools import repeat


def fixture_to_pair(label: str, fixture: str) -> tp.Tuple[str, str, str]:
    # get a title
    f = ff_cached(fixture)
    return label, f'{f.shape[0]:}x{f.shape[1]}', fixture

CLS_READ = (
    PDReadParquetArrow,
    PDReadParquetArrowNoComp,
    # PDReadParquetFast, # not faster!
    PDReadFeather,
    PDReadFeatherNoComp,
    # SFReadParquet,
    SFReadNPZ,
    # SFReadNPY,
    # SFReadNPYMM,
    # SFReadPickle,
    )
CLS_WRITE = (
    PDWriteParquetArrow,
    PDWriteParquetArrowNoComp,
    # PDWriteParquetFast, # not faster!
    # SFWriteParquet,
    PDWriteFeather,
    PDWriteFeatherNoComp,
    SFWriteNPZ,
    # SFWriteNPY,
    # SFWritePickle,
    )


def run_ff_test(
        *,
        number: int,
        include_read: bool = True,
        include_write: bool = True,
        fp: str = '/tmp/serialize.png',
        ):
    assert not (include_read is True and include_write is True)
    title = 'NPZ Read Performance' if include_read else 'NPZ Write Performance'

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

            record = [cls.__name__, number, category, fixture_label]
            try:
                result = timeit.timeit(
                        f'runner()',
                        globals=locals(),
                        number=number)
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

    plot_ff_performance(f, number=number, fp=fp, title=title)


def run_file_test(
        *,
        number: int,
        fixture: str,
        fp: str = '/tmp/serialize.png',
        ):
    title = 'NPZ Performance'
    records = []

    for cls, category_prefix in chain(
            zip(CLS_READ, repeat('read')),
            zip(CLS_WRITE, repeat('write')),
            ):
        runner = cls(fixture)
        category = f'{category_prefix}'

        record = [cls.__name__, number, category, fixture]
        try:
            result = timeit.timeit(
                    f'runner()',
                    globals=locals(),
                    number=number)
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

    plot_file_performance(f, number=number, fp=fp, title=title)

if __name__ == '__main__':
    run_size_test()
    # run_file_test(number=10,
    #         fixture=Path('/tmp/yellow_tripdata_2010-01.csv'),
    #         fp='/tmp/serialize.png',
    #         )
    # run_ff_test(number=10, include_read=True, include_write=False, fp='/tmp/serialize-read.png')
    # run_ff_test(number=10, include_read=False, include_write=True, fp='/tmp/serialize-write.png')



    # run_ff_test(number=1, include_read=True, include_write=False, fp='/tmp/serialize-temp.png')
