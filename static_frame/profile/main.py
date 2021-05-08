import io
import os
import argparse
import typing as tp
import types
import fnmatch
import collections
import timeit
import cProfile
import pstats
import sys
import datetime

from pyinstrument import Profiler #type: ignore
import numpy as np
import pandas as pd
import static_frame as sf

import frame_fixtures as ff

class PerfKey: pass

class Perf(PerfKey):
    NUMBER = 100_000

    def iter_function_names(self, pattern_func: str = '') -> tp.Iterator[str]:
        for name in sorted(dir(self)):
            if name == 'iter_function_names':
                continue
            if pattern_func and not fnmatch.fnmatch(
                    name, pattern_func.lower()):
               continue
            if not name.startswith('_') and callable(getattr(self, name)):
                yield name


class Native(PerfKey): pass
class Reference(PerfKey): pass


#-------------------------------------------------------------------------------
class SeriesIsNa(Perf):
    NUMBER = 10_000

    def __init__(self) -> None:
        f = ff.parse('s(1000,3)|v(float,object,bool)')
        f = f.assign.loc[(f.index % 12 == 0), 0](np.nan)
        f = f.assign.loc[(f.index % 12 == 0), 1](None)

        self.sfs_float = f.iloc[:, 0]
        self.sfs_object = f.iloc[:, 1]
        self.sfs_bool = f.iloc[:, 2]

        self.pds_float = f.iloc[:, 0].to_pandas()
        self.pds_object = f.iloc[:, 1].to_pandas()
        self.pds_bool = f.iloc[:, 2].to_pandas()

class SeriesIsNa_N(SeriesIsNa, Native):

    def float_index_auto(self) -> None:
        self.sfs_float.isna()

    def object_index_auto(self) -> None:
        self.sfs_object.isna()

    def bool_index_auto(self) -> None:
        self.sfs_bool.isna()

class SeriesIsNa_R(SeriesIsNa, Reference):

    def float_index_auto(self) -> None:
        self.pds_float.isna()

    def object_index_auto(self) -> None:
        self.pds_object.isna()

    def bool_index_auto(self) -> None:
        self.pds_bool.isna()

#-------------------------------------------------------------------------------
class SeriesDropNa(Perf):
    NUMBER = 500

    def __init__(self) -> None:
        f = ff.parse('s(100_000,3)|v(float,object,bool)')
        f = f.assign.loc[(f.index % 12 == 0), 0](np.nan)
        f = f.assign.loc[(f.index % 12 == 0), 1](None)

        self.sfs_float = f.iloc[:, 0]
        self.sfs_object = f.iloc[:, 1]
        self.sfs_bool = f.iloc[:, 2]

        self.pds_float = f.iloc[:, 0].to_pandas()
        self.pds_object = f.iloc[:, 1].to_pandas()
        self.pds_bool = f.iloc[:, 2].to_pandas()

class SeriesDropNa_N(SeriesDropNa, Native):

    def float_index_auto(self) -> None:
        self.sfs_float.dropna()

    def object_index_auto(self) -> None:
        self.sfs_object.dropna()

    def bool_index_auto(self) -> None:
        self.sfs_bool.dropna()

class SeriesDropNa_R(SeriesDropNa, Reference):

    def float_index_auto(self) -> None:
        self.pds_float.dropna()

    def object_index_auto(self) -> None:
        self.pds_object.dropna()

    def bool_index_auto(self) -> None:
        self.pds_bool.dropna()

#-------------------------------------------------------------------------------

class FrameILoc(Perf):

    def __init__(self) -> None:
        self.sff1 = ff.parse('s(100,100)')
        self.pdf1 = pd.DataFrame(self.sff1.values)

        self.sff2 = ff.parse('s(100,100)|i(I,str)|c(I,str)')
        self.pdf2 = self.sff2.to_pandas()

class FrameILoc_N(FrameILoc, Native):

    def element_index_auto(self) -> None:
        self.sff1.iloc[50, 50]

    def element_index_str(self) -> None:
        self.sff2.iloc[50, 50]

class FrameILoc_R(FrameILoc, Reference):

    def element_index_auto(self) -> None:
        self.pdf1.iloc[50, 50]

    def element_index_str(self) -> None:
        self.pdf2.iloc[50, 50]

#-------------------------------------------------------------------------------

class FrameLoc(Perf):

    def __init__(self) -> None:
        self.sff1 = ff.parse('s(100,100)')
        self.pdf1 = pd.DataFrame(self.sff1.values)

        self.sff2 = ff.parse('s(100,100)|i(I,str)|c(I,str)')
        self.pdf2 = self.sff2.to_pandas()

class FrameLoc_N(FrameLoc, Native):

    def element_index_auto(self) -> None:
        self.sff1.loc[50, 50]

    def element_index_str(self) -> None:
        self.sff2.loc['zGuv', 'zGuv']

class FrameLoc_R(FrameLoc, Reference):

    def element_index_auto(self) -> None:
        self.pdf1.loc[50, 50]

    def element_index_str(self) -> None:
        self.pdf2.loc['zGuv', 'zGuv']


#-------------------------------------------------------------------------------

def get_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
            description='Performance testing and profiling',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='''Example:

Performance comparison of all dropna tests:

python3 test_performance.py '*dropna' --performance

Profiling outpout for static-frame dropna:

python3 test_performance.py SeriesIntFloat_dropna --profile
            '''
            )
    p.add_argument('patterns',
            help='Names of classes to match using fn_match syntax',
            nargs='+',
            )
    # p.add_argument('--modules',
    #         help='Names of modules to find tests',
    #         nargs='+',
    #         default=('core',),
    #         )
    p.add_argument('--profile',
            help='Turn on profiling with cProfile',
            action='store_true',
            default=False,
            )
    p.add_argument('--graph',
            help='Produce a call graph of cProfile output',
            action='store_true',
            default=False,
            )
    p.add_argument('--instrument',
            help='Turn on instrumenting with pyinstrument',
            action='store_true',
            default=False,
            )
    p.add_argument('--performance',
            help='Turn on performance measurements',
            action='store_true',
            default=False,
            )
    return p


def yield_classes(
        pattern: str
        ) -> tp.Iterator[tp.Dict[tp.Type[PerfKey], tp.Type[PerfKey]]]:

    if '.' in pattern:
        pattern_cls, pattern_func = pattern.split('.')
    else:
        pattern_cls, pattern_func = pattern, '*'

    for cls_perf in Perf.__subclasses__(): # only get one level
        if pattern_cls and not fnmatch.fnmatch(
                cls_perf.__name__.lower(), pattern_cls.lower()):
            continue

        runners: tp.Dict[tp.Type[PerfKey], tp.Type[PerfKey]] = {Perf: cls_perf}

        for cls_runner in cls_perf.__subclasses__():
            for cls in (Native, Reference):
                if issubclass(cls_runner, cls):
                    runners[cls] = cls_runner
                    break
        yield runners, pattern_func



def profile(
        cls_runner: tp.Type[Perf],
        pattern_func: str,
        ) -> None:
    '''
    Profile the `sf` function from the supplied class.
    '''
    runner = cls_runner()
    for name in runner.iter_function_names(pattern_func):
        f = getattr(runner, name)
        pr = cProfile.Profile()

        pr.enable()
        for _ in range(runner.NUMBER):
            f()
        pr.disable()

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())

def graph(
        cls_runner: tp.Type[Perf],
        pattern_func: str,
        ) -> None:
    '''
    Profile the `sf` function from the supplied class.
    '''
    runner = cls_runner()
    for name in runner.iter_function_names(pattern_func):
        f = getattr(runner, name)
        pr = cProfile.Profile()

        pr.enable()
        for _ in range(runner.NUMBER):
            f()
        pr.disable()

        ps = pstats.Stats(pr)
        ps.dump_stats('/tmp/tmp.pstat')

        import gprof2dot
        gprof2dot.main([
            '--format', 'pstats',
            '--output', '/tmp/tmp.dot',
            '--edge-thres', '0', # 0.1 default
            '--node-thres', '0', # 0.5 default
            '/tmp/tmp.pstat'
        ])
        os.system('dot /tmp/tmp.dot -Tpng -o /tmp/tmp.png; eog /tmp/tmp.png')


def instrument(
        cls_runner: tp.Type[Perf],
        pattern_func: str,
        timeline: bool = False,
        ) -> None:
    '''
    Profile the `sf` function from the supplied class.
    '''
    runner = cls_runner()
    for name in runner.iter_function_names(pattern_func):
        f = getattr(runner, name)
        profiler = Profiler(interval=0.0001) # default is 0.001, 1 ms

        if timeline:
            profiler.start()
            f()
            profiler.stop()
        else:
            profiler.start()
            for _ in range(runner.NUMBER):
                f()
            profiler.stop()

        print(profiler.output_text(unicode=True, color=True, timeline=timeline, show_all=True))


def line(
        cls_runner: tp.Type[Perf],
        pattern_func: str,
        ) -> None:
    runner = cls_runner()
    from line_profiler import main
    for name in runner.iter_function_names(pattern_func):
        f = getattr(runner, name)


#-------------------------------------------------------------------------------

PerformanceRecord = tp.MutableMapping[str, tp.Union[str, float, bool]]

def performance(
        bundle: tp.Dict[tp.Type[PerfKey], tp.Type[PerfKey]],
        pattern_func: str,
        ) -> tp.Iterator[PerformanceRecord]:

    cls_perf = bundle[Perf]
    assert issubclass(cls_perf, Perf)

    cls_native = bundle[Native]
    cls_reference = bundle[Reference]

    # TODO: check native/ref have the same  iterations
    runner_n = cls_native()
    runner_r = cls_reference()
    assert isinstance(runner_n, Perf)

    for func_name in runner_n.iter_function_names(pattern_func):
        row: PerformanceRecord = {}
        row['name'] = f'{cls_perf.__name__}.{func_name}'
        row['iterations'] = cls_perf.NUMBER

        for label, runner in ((Native, runner_n), (Reference, runner_r)):
            row[label.__name__] = timeit.timeit(
                    f'runner.{func_name}()',
                    globals=locals(),
                    number=cls_perf.NUMBER)
        yield row


def performance_tables_from_records(
        records: tp.Iterable[PerformanceRecord],
        ) -> tp.Tuple[sf.Frame, sf.Frame]:

    from static_frame.core.display_color import HexColor

    frame = sf.FrameGO.from_dict_records(records)
    # frame = frame.set_index('name', drop=True)

    # if PerfTest.SF_NAME in frame.columns and PerfTest.PD_NAME in frame.columns:
    frame['n/r'] = frame[Native.__name__] / frame[Reference.__name__]
    frame['r/n'] = frame[Reference.__name__] / frame[Native.__name__]
    frame['win'] = frame['r/n'] > .99

    def format(v: object) -> str:
        if isinstance(v, float):
            if np.isnan(v):
                return ''
            return str(round(v, 4))
        if isinstance(v, (bool, np.bool_)):
            if v:
                return HexColor.format_terminal('green', str(v))
            return HexColor.format_terminal('orange', str(v))

        return str(v)

    display = frame.iter_element().apply(format)
    # display = display[[c for c in display.columns if '/' not in c]]
    return frame, display

def main() -> None:

    options = get_arg_parser().parse_args()
    records: tp.List[PerformanceRecord] = []

    for pattern in options.patterns:
        for bundle, pattern_func in yield_classes(pattern):
            if options.performance:
                records.extend(performance(bundle, pattern_func))
            if options.profile:
                profile(bundle[Native], pattern_func) #type: ignore
            if options.graph:
                graph(bundle[Native], pattern_func) #type: ignore
            if options.instrument:
                instrument(bundle[Native], pattern_func) #type: ignore

    itemize = False # make CLI option maybe

    if records:

        from static_frame import DisplayConfig

        print(str(datetime.datetime.now()))

        pairs = []
        pairs.append(('python', sys.version.split(' ')[0]))
        for package in (np, pd, sf):
            pairs.append((package.__name__, package.__version__))
        print('|'.join(':'.join(pair) for pair in pairs))

        frame, display = performance_tables_from_records(records)

        config = DisplayConfig(
                cell_max_width_leftmost=np.inf,
                cell_max_width=np.inf,
                type_show=False,
                display_rows=200
                )
        print(display.display(config))

        if itemize:
            alt = display.T
            for c in alt.columns:
                print(c)
                print(alt[c].sort_values().display(config))

        # import ipdb; ipdb.set_trace()
        # if 'sf/pd' in frame.columns:
        #     print('mean: {}'.format(round(frame['sf/pd'].mean(), 6)))
        #     print('wins: {}/{}'.format((frame['sf/pd'] < 1.05).sum(), len(frame)))



if __name__ == '__main__':
    main()
