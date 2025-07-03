import time
import numpy as np
import sys
import os
import timeit
from itertools import repeat

import frame_fixtures as ff
import matplotlib.pyplot as plt
import numpy as np
import typing_extensions as tp

sys.path.append(os.getcwd())

import static_frame as sf
from static_frame.core.display_color import HexColor




#-------------------------------------------------------------------------------

class FTTest:
    SUFFIX = '.tmp'

    def __init__(self, fixture: str):
        self.sff = ff.parse(fixture)

    def __call__(self):
        raise NotImplementedError()


def proc(s):
    return s.loc[(s % 2) == 0].sum()


class IterSeriesA_Single(FTTest):
    def __call__(self):
        _ = self.sff.iter_series(axis=1).apply(proc)
        # import ipdb; ipdb.set_trace()

class IterSeriesA_Process_Workers2(FTTest):
    def __call__(self):
        _ = self.sff.iter_series(axis=1).apply_pool(proc,
                chunksize=10, use_threads=False, max_workers=2)


class IterSeriesA_Process_Workers4(FTTest):
    def __call__(self):
        _ = self.sff.iter_series(axis=1).apply_pool(proc,
                chunksize=10, use_threads=False, max_workers=4)


class IterSeriesA_Process_Workers8(FTTest):
    def __call__(self):
        _ = self.sff.iter_series(axis=1).apply_pool(proc,
                chunksize=10, use_threads=False, max_workers=8)


class IterSeriesA_Process_Workers16(FTTest):
    def __call__(self):
        _ = self.sff.iter_series(axis=1).apply_pool(proc,
                chunksize=10, use_threads=False, max_workers=16)





class IterSeriesA_Threads_Workers2(FTTest):
    def __call__(self):
        _ = self.sff.iter_series(axis=1).apply_pool(proc,
                chunksize=10, use_threads=True, max_workers=2)


class IterSeriesA_Threads_Workers4(FTTest):
    def __call__(self):
        _ = self.sff.iter_series(axis=1).apply_pool(proc,
                chunksize=10, use_threads=True, max_workers=4)

class IterSeriesA_Threads_Workers8(FTTest):
    def __call__(self):
        _ = self.sff.iter_series(axis=1).apply_pool(proc,
                chunksize=10, use_threads=True, max_workers=8)


class IterSeriesA_Threads_Workers16(FTTest):
    def __call__(self):
        _ = self.sff.iter_series(axis=1).apply_pool(proc,
                chunksize=10, use_threads=True, max_workers=16)







#-------------------------------------------------------------------------------
NUMBER = 10

def scale(v):
    return int(v * 1)

VALUES_UNIFORM = 'float'
VALUES_MIXED = 'int,int,int,int,bool,bool,bool,bool,float,float,float,float'
VALUES_COLUMNAR = 'int,bool,float'

FF_wide_uniform = f's({scale(100)},{scale(10_000)})|v({VALUES_UNIFORM})|i(I,int)|c(I,str)'
FF_wide_mixed   = f's({scale(100)},{scale(10_000)})|v({VALUES_MIXED})|i(I,int)|c(I,str)'
FF_wide_columnar = f's({scale(100)},{scale(10_000)})|v({VALUES_COLUMNAR})|i(I,int)|c(I,str)'


FF_tall_uniform = f's({scale(10_000)},{scale(100)})|v({VALUES_UNIFORM})|i(I,int)|c(I,str)'
FF_tall_mixed   = f's({scale(10_000)},{scale(100)})|v({VALUES_MIXED})|i(I,int)|c(I,str)'
FF_tall_columnar   = f's({scale(10_000)},{scale(100)})|v({VALUES_COLUMNAR})|i(I,int)|c(I,str)'

FF_square_uniform = f's({scale(1_000)},{scale(1_000)})|v({VALUES_UNIFORM})|i(I,int)|c(I,str)'
FF_square_mixed   = f's({scale(1_000)},{scale(1_000)})|v({VALUES_MIXED})|i(I,int)|c(I,str)'
FF_square_columnar = f's({scale(1_000)},{scale(1_000)})|v({VALUES_COLUMNAR})|i(I,int)|c(I,str)'

#-------------------------------------------------------------------------------

def seconds_to_display(seconds: float, number: int) -> str:
    seconds /= number
    if seconds < 1e-4:
        return f'{seconds * 1e6: .1f} (µs)'
    if seconds < 1e-1:
        return f'{seconds * 1e3: .1f} (ms)'
    return f'{seconds: .1f} (s)'

def plot_performance(frame: sf.Frame,
        *,
        number: int,
    ):
    fixture_total = len(frame['fixture'].unique())
    cat_total = len(frame['category'].unique())
    name_total = len(frame['name'].unique())

    fig, axes = plt.subplots(cat_total, fixture_total)

    # for legend
    name_replace = {
        # IterArrayA_Single.__name__: 'iter_array()',
        # IterArrayA_Threads_Workers4.__name__: 'iter_array(use_threads=True,\nmax_workers=4)',
        # IterArrayA_Threads_Workers16.__name__: 'iter_array(use_threads=True,\nmax_workers=16)',
        IterSeriesA_Single.__name__: 'iter_series.apply()',

        IterSeriesA_Process_Workers2.__name__: 'iter_series.apply_pool(\nuse_threads=False,\nmax_workers=2)',
        IterSeriesA_Process_Workers4.__name__: 'iter_series.apply_pool(\nuse_threads=False,\nmax_workers=4)',
        IterSeriesA_Process_Workers8.__name__: 'iter_series.apply_pool(\nuse_threads=False,\nmax_workers=8)',
        IterSeriesA_Process_Workers16.__name__: 'iter_series.apply_pool(\nuse_threads=False,\nmax_workers=16)',

        IterSeriesA_Threads_Workers2.__name__: 'iter_series.apply_pool(\nuse_threads=True,\nmax_workers=2)',
        IterSeriesA_Threads_Workers4.__name__: 'iter_series.apply_pool(\nuse_threads=True,\nmax_workers=4)',
        IterSeriesA_Threads_Workers8.__name__: 'iter_series.apply_pool(\nuse_threads=True,\nmax_workers=8)',
        IterSeriesA_Threads_Workers16.__name__: 'iter_series.apply_pool(\nuse_threads=True,\nmax_workers=16)',
    }

    name_order = {
        # IterArrayA_Single.__name__: 0,
        # IterArrayA_Threads_Workers4.__name__: 1,
        # IterArrayA_Threads_Workers16.__name__: 2,
        IterSeriesA_Single.__name__: 0,
        IterSeriesA_Process_Workers2.__name__: 1,
        IterSeriesA_Process_Workers4.__name__: 2,
        IterSeriesA_Process_Workers8.__name__: 3,
        IterSeriesA_Process_Workers16.__name__: 4,

        IterSeriesA_Threads_Workers2.__name__: 11,
        IterSeriesA_Threads_Workers4.__name__: 12,
        IterSeriesA_Threads_Workers8.__name__: 13,
        IterSeriesA_Threads_Workers16.__name__: 14,
    }

    # cmap = plt.get_cmap('terrain')
    cmap = plt.get_cmap('plasma')
    color_count = name_total
    color = cmap(np.arange(color_count) / color_count)

    # categories are read, write
    # import ipdb; ipdb.set_trace()
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
            title = f'{cat_label.title()}\n{FIXTURE_SHAPE_MAP[fixture_label]}'
            ax.set_title(title, fontsize=8)
            ax.set_box_aspect(0.75) # makes taller tan wide
            time_max = fixture['time'].max()
            time_min = fixture["time"].min()

            y_ticks = [0, time_min, time_max * 0.5, time_max]
            y_labels = ['',
                    seconds_to_display(time_min, number),
                    seconds_to_display(time_max * 0.5, number),
                    seconds_to_display(time_max, number),
                    ]

            if time_min > time_max * 0.333:
                # remove the min if it is greater than quarter
                y_ticks.pop(1)
                y_labels.pop(1)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=6)

            # ax.set_xticks(x, names_display, rotation='vertical')
            ax.tick_params(
                    axis='x',
                    which='both',
                    bottom=False,
                    top=False,
                    labelbottom=False,
                    )

    fig.set_size_inches(5.5, 3.5) # width, height
    fig.legend(post, names_display, loc='center right', fontsize=6)
    # horizontal, vertical
    count = ff.parse(FF_tall_uniform).size
    fig.text(.05, .96, f'Row-Wise Function Application: {count:.0e} Elements, {NUMBER} Iterations', fontsize=10)
    fig.text(.05, .90, get_versions(), fontsize=6)

    # get fixtures size reference
    shape_map = {shape: FIXTURE_SHAPE_MAP[shape] for shape in frame['fixture'].unique()}
    shape_msg = ' / '.join(f'{v}: {k}' for k, v in shape_map.items())
    fig.text(.05, .90, shape_msg, fontsize=6)

    fp = '/tmp/ft-perf.png'
    plt.subplots_adjust(
            left=0.05,
            bottom=0.05,
            right=0.75,
            top=0.75,
            wspace=0, # width
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
    py_version = sys.version[:sys.version.find('(')].strip()
    return f'OS: {platform.system()} / Python: {py_version} / StaticFrame: {sf.__version__} / NumPy: {np.__version__}\n'

FIXTURE_SHAPE_MAP = {
    '100x1': 'Tall',
    '10x10': 'Square',
    '1x100': 'Wide',
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


def fixture_to_pair(label: str, fixture: str) -> tp.Tuple[str, str, str]:
    # get a title
    f = ff.parse(fixture)
    return label, f'{f.shape[0]:}x{f.shape[1]}', fixture

CLS_READ = (
    IterSeriesA_Single,
    IterSeriesA_Process_Workers4,
    IterSeriesA_Process_Workers8,
    IterSeriesA_Process_Workers16,

    # IterSeriesA_Threads_Workers4,
    # IterSeriesA_Threads_Workers8,
    # IterSeriesA_Threads_Workers16,
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

        for cls in CLS_READ:
            runner = cls(fixture)
            category = f'{dtype_hetero}'

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


    config = sf.DisplayConfig(
            cell_max_width_leftmost=np.inf,
            cell_max_width=np.inf,
            type_show=False,
            display_rows=200,
            include_index=False,
            )
    print(f.display(config))

    plot_performance(f, number=NUMBER)

if __name__ == '__main__':

    run_test()




