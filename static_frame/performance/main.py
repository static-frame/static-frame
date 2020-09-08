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


from static_frame.performance.perf_test import PerfTest


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
    p.add_argument('--modules',
            help='Names of modules to find tests',
            nargs='+',
            default=('core',),
            )
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
        module: types.ModuleType,
        pattern: str
        ) -> tp.Iterator[tp.Type[PerfTest]]:
    # this will not find children of children
    for attr_name, attr in vars(module).items():
        if attr_name.startswith('_'):
            continue
        if isinstance(attr, type) and issubclass(attr, PerfTest) and not attr is PerfTest:
            if fnmatch.fnmatch(attr_name.lower(), pattern.lower()):
                yield attr

def profile(cls: tp.Type[PerfTest],
        function: str = 'sf'
        ) -> None:
    '''
    Profile the `sf` function from the supplied class.
    '''

    f = getattr(cls, function)
    pr = cProfile.Profile()

    pr.enable()
    for _ in range(cls.NUMBER):
        f()
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())

def instrument(cls: tp.Type[PerfTest],
        function: str = 'sf'
        ) -> None:
    '''
    Profile the `sf` function from the supplied class.
    '''

    f = getattr(cls, function)
    profiler = Profiler()

    profiler.start()
    for _ in range(cls.NUMBER):
        f()
    profiler.stop()

    print(profiler.output_text(unicode=True, color=True))


PerformanceRecord = tp.MutableMapping[str, tp.Union[str, float]]

def performance(
        module: types.ModuleType,
        cls: tp.Type[PerfTest]
        ) -> PerformanceRecord:
    #row = []
    row: PerformanceRecord = collections.OrderedDict()
    row['name'] = cls.__name__
    row['iterations'] = cls.NUMBER
    for f in cls.FUNCTION_NAMES:
        if hasattr(cls, f):
            result = timeit.timeit(cls.__name__ + '.' + f + '()',
                    globals=vars(module),
                    number=cls.NUMBER)
            row[f] = result
        else:
            row[f] = np.nan
    return row


def performance_tables_from_records(
        records: tp.Iterable[PerformanceRecord]
        ) -> tp.Tuple[sf.Frame, sf.Frame]:

    frame = sf.FrameGO.from_dict_records(records)
    frame = frame.set_index('name', drop=True)

    if PerfTest.SF_NAME in frame.columns and PerfTest.PD_NAME in frame.columns:
        frame['sf/pd'] = frame[PerfTest.SF_NAME] / frame[PerfTest.PD_NAME]
        frame['pd_outperform'] = frame['sf/pd'].loc[frame['sf/pd'] > 1]

        frame['pd/sf'] = frame[PerfTest.PD_NAME] / frame[PerfTest.SF_NAME]
        frame['sf_outperform'] = frame['pd/sf'].loc[frame['pd/sf'] > 1]

    def format(v: object) -> str:
        if isinstance(v, float):
            if np.isnan(v):
                return ''
            return str(round(v, 4))
        return str(v)

    display = frame.iter_element().apply(format)
    display = display[[c for c in display.columns if '/' not in c]]
    return frame, display

def main() -> None:

    options = get_arg_parser().parse_args()

    module_targets = []
    for module in options.modules:
        if module == 'core':
            from static_frame.performance import core
            module_targets.append(core)
        elif module == 'adhoc':
            from static_frame.performance import adhoc
            module_targets.append(adhoc)
        elif module == 'pydata_2018':
            from static_frame.performance import pydata_2018
            module_targets.append(pydata_2018)
        elif module == 'pydata_2019':
            from static_frame.performance import pydata_2019
            module_targets.append(pydata_2019)
        elif module == 'iter_group_perf':
            from static_frame.performance import iter_group_perf
            module_targets.append(iter_group_perf)
        else:
            raise NotImplementedError()

    records = []

    for module in module_targets:
        module.SampleData.create()
        for pattern in options.patterns:
            for cls in sorted(yield_classes(module, pattern), key=lambda c: c.__name__):
                print(cls.__name__)
                if options.performance:
                    records.append(performance(module, cls))
                if options.profile:
                    profile(cls)
                if options.instrument:
                    instrument(cls)

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
        if 'sf/pd' in frame.columns:
            print('mean: {}'.format(round(frame['sf/pd'].mean(), 6)))
            print('wins: {}/{}'.format((frame['sf/pd'] < 1.05).sum(), len(frame)))



if __name__ == '__main__':
    main()
