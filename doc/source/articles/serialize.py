

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

FF_wide = f's({scale(100)},{scale(10_000)})|v(int,int,bool,float,float)|i(I,int)|c(I,str)'
FF_wide_uniform = f's({scale(100)},{scale(10_000)})|v(float)|i(I,int)|c(I,str)'
FF_wide_mixed   = f's({scale(100)},{scale(10_000)})|v(int,int,bool,float,float)|i(I,int)|c(I,str)'
FF_wide_columnar   = f's({scale(100)},{scale(10_000)})|v(int,bool,float)|i(I,int)|c(I,str)'


FF_tall = f's({scale(10_000)},{scale(100)})|v(int,int,bool,float,float)|i(I,int)|c(I,str)'
FF_tall_uniform = f's({scale(10_000)},{scale(100)})|v(float)|i(I,int)|c(I,str)'
FF_tall_mixed   = f's({scale(10_000)},{scale(100)})|v(int,int,bool,float,float)|i(I,int)|c(I,str)'
FF_tall_columnar   = f's({scale(10_000)},{scale(100)})|v(int,bool,float)|i(I,int)|c(I,str)'

FF_square = f's({scale(1_000)},{scale(1_000)})|v(float)|i(I,int)|c(I,str)'
FF_square_unifrom = f's({scale(1_000)},{scale(1_000)})|v(float)|i(I,int)|c(I,str)'
FF_square_mixed   = f's({scale(1_000)},{scale(1_000)})|v(int,int,bool,float,float)|i(I,int)|c(I,str)'
FF_square_columnar   = f's({scale(1_000)},{scale(1_000)})|v(int,bool,float)|i(I,int)|c(I,str)'

#-------------------------------------------------------------------------------

# def plot(frame: sf.Frame):
#     from collections import defaultdict
#     fig, axes = plt.subplots(2, 2)

#     # for legend
#     label_replace = {
#         PDReadParquetArrow.__name__: 'Parquet (pd, pyarrow)',
#         PDWriteParquetArrow.__name__: 'Parquet (pd, pyarrow)',
#         SFReadPickle.__name__: 'Pickle (sf)',
#         SFWritePickle.__name__: 'Pickle (sf)',
#         SFReadParquet.__name__: 'Parquet (sf)',
#         SFWriteParquet.__name__: 'Parquet (sf)',
#         SFReadNPZ.__name__: 'NPZ (sf)',
#         SFWriteNPZ.__name__: 'NPZ (sf)',
#         SFReadNPY.__name__: 'NPY (sf)',
#         SFWriteNPY.__name__: 'NPY (sf)',

#     }

#     def prepare_label(label: str) -> str:
#         return label_replace[label]

#     for axes_count, (cat_label, cat) in enumerate(frame.iter_group_items('category')):
#         for log in (1, 0): # this is row count
#             ax = axes[log][axes_count]

#             labels = [] # by fixture
#             results = defaultdict(list)

#             components = 0
#             for label, sub in cat.iter_group_items('fixture'):
#                 labels.append(label)
#                 components = max(components, len(sub))
#                 for row in sub.iter_series(axis=1):
#                     results[row['name']].append(row['time'])

#             width = 0.85  # the width of each group
#             x = np.arange(len(labels))

#             segment = width / components
#             start = -width / 2
#             for i, (label, values) in enumerate(results.items()):
#                 r = ax.bar(x + (start + (segment * i)),
#                         values,
#                         segment,
#                         label=prepare_label(label),
#                         )
#                 # ax.bar_label(r, padding=3)

#             # ax.set_ylabel()
#             title = f'{cat_label.title()} ({"log(s)" if log else "s"})'
#             ax.set_title(title)
#             ax.set_xticks(x, labels)
#             if log:
#                 ax.set_yscale('log')
#             ax.legend(fontsize='small')

#     fig.tight_layout()
#     fig.set_size_inches(10, 4)
#     fp = '/tmp/serialize.png'
#     plt.savefig(fp, dpi=300)
#     import os
#     os.system(f'open {fp}')



def plot(frame: sf.Frame):
    fixture_total = len(frame['fixture'].unique())
    cat_total = len(frame['category'].unique())
    name_total = len(frame['name'].unique())

    fig, axes = plt.subplots(cat_total, fixture_total)

    # for legend
    name_replace = {
        PDReadParquetArrow.__name__: 'Parquet (pd)',
        PDWriteParquetArrow.__name__: 'Parquet (pd)',
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
        SFReadParquet.__name__: 1,
        SFWriteParquet.__name__: 1,
        SFReadNPY.__name__: 2,
        SFWriteNPY.__name__: 2,
        SFReadNPZ.__name__: 3,
        SFWriteNPZ.__name__: 3,
        SFReadPickle.__name__: 4,
        SFWritePickle.__name__: 4,
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
            # import ipdb; ipdb.set_trace()
            x = np.arange(len(results))
            names_display = [name_replace[l] for l in names]
            post = ax.bar(names_display, results, color=color)
            # import ipdb; ipdb.set_trace()

            # ax.set_ylabel()
            title = f'{cat_label.title()}\n{fixture_label}'
            ax.set_title(title, fontsize=8)
            ax.set_box_aspect(0.75) # makes taller tan wide
            time_max = fixture['time'].max()
            ax.set_yticks([0, time_max * 0.5, time_max])
            ax.set_yticklabels(['',
                    f'{time_max * 0.5:.3f}',
                    f'{time_max:.3f}',
                    ], fontsize=6)
            # ax.set_xticks(x, names_display, rotation='vertical')
            ax.tick_params(
                    axis='x',
                    which='both',
                    bottom=False,
                    top=False,
                    labelbottom=False,
                    )


    fig.set_size_inches(6, 6) # width, height
    fig.legend(post, names_display, loc='center right', fontsize=8)
    fp = '/tmp/serialize.png'
    plt.subplots_adjust(left=0.05,
            bottom=0.05,
            right=0.85,
            top=0.90,
            wspace=-0.2, # width
            hspace=0.9,
            )
    # plt.rcParams.update({'font.size': 22})
    plt.savefig(fp, dpi=300)

    if sys.platform.startswith('linux'):
        os.system(f'eog {fp}')
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

            fixture_to_pair('uniform', FF_square_unifrom),
            fixture_to_pair('mixed', FF_square_mixed),
            fixture_to_pair('columnar', FF_square_columnar),
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

