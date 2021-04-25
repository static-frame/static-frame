import io
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

class Perf:
    FUNCTIONS = ()
    NUMBER = 100_000

    def iter_function_names(self) -> tp.Iterator[str]:
        for name in dir(self):
            if name == 'iter_function_names':
                continue
            if not name.startswith('_') and callable(getattr(self, name)):
                yield name


class Native(Perf): pass
class Reference(Perf): pass

#-------------------------------------------------------------------------------

class FrameILoc(Perf):

    def __init__(self) -> None:
        self.f1 = ff.parse('s(100,100)')
        self.p1 = pd.DataFrame(self.f1.values)

        self.f2 = ff.parse('s(100,100)|i(I,str)|c(I,str)')
        self.p2 = self.f2.to_pandas()

class FrameILoc_N(FrameILoc, Native):

    def element_index_auto(self) -> None:
        self.f1.iloc[50, 50]

    def element_index_str(self) -> None:
        self.f2.iloc[50, 50]

class FrameILoc_R(FrameILoc, Reference):

    def element_index_auto(self) -> None:
        self.p1.iloc[50, 50]

    def element_index_str(self) -> None:
        self.p2.iloc[50, 50]

#-------------------------------------------------------------------------------

class FrameLoc(Perf):

    def __init__(self) -> None:
        self.f1 = ff.parse('s(100,100)')
        self.p1 = pd.DataFrame(self.f1.values)

        self.f2 = ff.parse('s(100,100)|i(I,str)|c(I,str)')
        self.p2 = self.f2.to_pandas()

class FrameLoc_N(FrameLoc, Native):

    def element_index_auto(self) -> None:
        self.f1.loc[50, 50]

    def element_index_str(self) -> None:
        self.f2.loc['zGuv', 'zGuv']

class FrameLoc_R(FrameLoc, Reference):

    def element_index_auto(self) -> None:
        self.p1.loc[50, 50]

    def element_index_str(self) -> None:
        self.p2.loc['zGuv', 'zGuv']

#

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
        ) -> tp.Iterator[tp.Dict[tp.Type[Perf], tp.Type[Perf]]]:

    for cls_perf in Perf.__subclasses__(): # only get one level
        if pattern and not fnmatch.fnmatch(
                cls_perf.__name__.lower(), pattern.lower()):
            continue
        runners = {Perf: cls_perf}
        for cls_runner in cls_perf.__subclasses__():
            for cls in (Native, Reference):
                if issubclass(cls_runner, cls):
                    runners[cls] = cls_runner
                    break
        yield runners



def profile(cls_runner: tp.Type[Perf]) -> None:
    '''
    Profile the `sf` function from the supplied class.
    '''

    runner = cls_runner()
    for name in runner.iter_function_names():
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

def instrument(cls_runner: tp.Type[Perf]) -> None:
    '''
    Profile the `sf` function from the supplied class.
    '''
    runner = cls_runner()
    for name in runner.iter_function_names():
        f = getattr(runner, name)
        profiler = Profiler()

        profiler.start()
        for _ in range(runner.NUMBER):
            f()
        profiler.stop()

        print(profiler.output_text(unicode=True, color=True))


PerformanceRecord = tp.MutableMapping[str, tp.Union[str, float, bool]]

def performance(
        bundle: tp.Dict[tp.Type[Perf], tp.Type[Perf]],
        ) -> tp.Iterator[PerformanceRecord]:

    cls_perf = bundle[Perf]
    cls_native = bundle[Native]
    cls_reference = bundle[Reference]

    # TODO: check native/ref have the same  iterations
    runner_n = cls_native()
    runner_r = cls_reference()

    for func_name in runner_n.iter_function_names():
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
        records: tp.Iterable[PerformanceRecord]
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
        for bundle in yield_classes(pattern):
            if options.performance:
                records.extend(performance(bundle))
            if options.profile:
                profile(bundle[Native])
            if options.instrument:
                instrument(bundle[Native])

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
