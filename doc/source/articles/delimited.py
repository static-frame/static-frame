


import os
import shutil
import sys
import tempfile
import timeit
import typing as tp
from itertools import repeat


import frame_fixtures as ff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

import static_frame as sf
from static_frame.core.display_color import HexColor
from static_frame.core.util import bytes_to_size_label


class FileIOTest:
    SUFFIX = '.tmp'

    def __init__(self, fixture: str):
        self.fixture = ff.parse(fixture)
        _, self.fp = tempfile.mkstemp(suffix=self.SUFFIX)
        self.fixture.to_csv(self.fp, include_index=False)
        self.dtypes = dict(self.fixture.dtypes)
        self.format = list(self.dtypes.items())

    def __call__(self):
        raise NotImplementedError()



class SFTypeParse(FileIOTest):

    def __call__(self):
        f = sf.Frame.from_csv(self.fp, index_depth=0)
        assert f.shape == self.fixture.shape

class SFStr(FileIOTest):

    def __call__(self):
        f = sf.Frame.from_csv(self.fp, index_depth=0, dtypes=str)
        assert f.shape == self.fixture.shape

class SFTypeGiven(FileIOTest):

    def __call__(self):
        f = sf.Frame.from_csv(self.fp, index_depth=0, dtypes=self.dtypes)
        assert f.shape == self.fixture.shape


class PandasTypeParse(FileIOTest):

    def __call__(self):
        f = pd.read_csv(self.fp)
        assert f.shape == self.fixture.shape

class PandasStr(FileIOTest):

    def __call__(self):
        f = pd.read_csv(self.fp, dtype=str)
        assert f.shape == self.fixture.shape

class PandasTypeGiven(FileIOTest):

    def __call__(self):
        f = pd.read_csv(self.fp, dtype=self.dtypes)
        assert f.shape == self.fixture.shape


class NumpyGenfromtxtTypeParse(FileIOTest):

    def __call__(self):
        f = np.genfromtxt(self.fp, dtype=None, delimiter=',', encoding=None, names=True)
        assert len(f) == len(self.fixture)

# class NumpyLoadtxtTypeParse(FileIOTest):

#     def __call__(self):
#         import pdb; pdb.set_trace()
#         f = np.loadtxt(self.fp, dtype=self.format, delimiter=',', encoding=None, skiprows=1)
#         assert len(f) == len(self.fixture)

#-------------------------------------------------------------------------------
NUMBER = 2

def scale(v):
    return int(v * 1)

FF_wide_uniform = f's({scale(100)},{scale(10_000)})|v(float)|i(I,int)|c(I,str)'
FF_wide_mixed   = f's({scale(100)},{scale(10_000)})|v(int,int,bool,float,float)|i(I,int)|c(I,str)'
FF_wide_columnar = f's({scale(100)},{scale(10_000)})|v(int,bool,float)|i(I,int)|c(I,str)'


FF_tall_uniform = f's({scale(10_000)},{scale(100)})|v(float)|i(I,int)|c(I,str)'
FF_tall_mixed   = f's({scale(10_000)},{scale(100)})|v(int,int,bool,float,float)|i(I,int)|c(I,str)'
FF_tall_columnar   = f's({scale(10_000)},{scale(100)})|v(int,bool,float)|i(I,int)|c(I,str)'

FF_square_uniform = f's({scale(1_000)},{scale(1_000)})|v(float)|i(I,int)|c(I,str)'
FF_square_mixed   = f's({scale(1_000)},{scale(1_000)})|v(int,int,bool,float,float)|i(I,int)|c(I,str)'
FF_square_columnar = f's({scale(1_000)},{scale(1_000)})|v(int,bool,float)|i(I,int)|c(I,str)'

#-------------------------------------------------------------------------------


def plot_performance(frame: sf.Frame):
    fixture_total = len(frame['fixture'].unique())
    cat_total = len(frame['category'].unique())
    name_total = len(frame['name'].unique())

    fig, axes = plt.subplots(cat_total, fixture_total)

    # for legend
    name_replace = {
        SFTypeParse.__name__: 'StaticFrame\n(type parsing)',
        SFStr.__name__: 'StaticFrame\n(as string)',
        SFTypeGiven.__name__: 'StaticFrame\n(type given)',
        PandasTypeParse.__name__: 'Pandas\n(type parsing)',
        PandasStr.__name__: 'Pandas\n(as string)',
        PandasTypeGiven.__name__: 'Pandas\n(type given)',
        NumpyGenfromtxtTypeParse.__name__: 'NumPy genfromtxt\n(type parsing)',
        # NumpyLoadtxtTypeParse.__name__: 'NumPy loadtxt\n(type given)',
    }

    name_order = {
        SFTypeParse.__name__: 0,
        SFStr.__name__: 0,
        SFTypeGiven.__name__: 0,
        PandasTypeParse.__name__: 1,
        PandasStr.__name__: 1,
        PandasTypeGiven.__name__: 1,
        NumpyGenfromtxtTypeParse.__name__: 2,
        # NumpyLoadtxtTypeParse.__name__: 2,
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
    fig.text(.05, .97, f'Delimited Read Performance: {count:.0e} Elements, {NUMBER} Iterations', fontsize=10)
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

def get_versions() -> str:
    import platform
    return f'OS: {platform.system()} / Pandas: {pd.__version__} / StaticFrame: {sf.__version__} / NumPy: {np.__version__}\n'

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

def fixture_to_pair(label: str, fixture: str) -> tp.Tuple[str, str, str]:
    # get a title
    f = ff.parse(fixture)
    return label, f'{f.shape[0]:}x{f.shape[1]}', fixture

CLS_READ = (
    SFTypeParse,
    SFStr,
    SFTypeGiven,
    PandasTypeParse,
    PandasStr,
    PandasTypeGiven,
    # NumpyGenfromtxtTypeParse,
    )


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

        for cls, category_prefix in zip(CLS_READ, repeat('read')):
            runner = cls(fixture)
            category = f'{category_prefix} {dtype_hetero}'

            record = [cls.__name__, NUMBER, category, fixture_label]
            print(record)
            try:
                result = timeit.timeit(
                        f'runner()',
                        globals=locals(),
                        number=NUMBER)
            except OSError:
                result = np.nan
            finally:
                pass
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

    run_test()



