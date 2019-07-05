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

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
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
            help='Turn on profiling',
            action='store_true',
            default=False,
            )
    p.add_argument('--performance',
            help='Turn on performance measurements',
            action='store_true',
            default=False,
            )
    return p


def yield_classes(module: types.ModuleType, pattern: str) -> tp.Iterator[tp.Type[PerfTest]]:
    # this will not find children of children
    for attr_name, attr in vars(module).items():
        if attr_name.startswith('_'):
            continue
        if isinstance(attr, type) and issubclass(attr, PerfTest) and not attr is PerfTest:
            if fnmatch.fnmatch(attr_name.lower(), pattern.lower()):
                yield attr

def profile(cls: tp.Type[PerfTest], function: str = 'sf') -> None:
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

def performance(module: types.ModuleType, cls: tp.Type[PerfTest]) -> tp.MutableMapping[str, tp.Union[str, float]]:
    #row = []
    row: tp.MutableMapping[str, tp.Union[str, float]] = collections.OrderedDict()
    row['name'] = cls.__name__
    for f in PerfTest.FUNCTION_NAMES:
        if hasattr(cls, f):
            result = timeit.timeit(cls.__name__ + '.' + f + '()',
                    globals=vars(module),
                    number=cls.NUMBER)
            row[f] = result
        else:
            row[f] = np.nan
    return row


def main() -> None:

    options = get_arg_parser().parse_args()

    module_targets = []
    for module in options.modules:
        if module == 'core':
            from static_frame.performance import core
            module_targets.append(core)
        elif module == 'pydata_2018':
            from static_frame.performance import pydata_2018
            module_targets.append(pydata_2018)
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
    if records:

        from static_frame import DisplayConfig

        print(str(datetime.datetime.now()))

        pairs = []
        pairs.append(('python', sys.version.split(' ')[0]))
        for package in (np, pd, sf):
            pairs.append((package.__name__, package.__version__))
        print('|'.join(':'.join(pair) for pair in pairs))

        frame = sf.FrameGO.from_records(records)
        frame = frame.set_index('name', drop=True)
        frame['sf/pd'] = frame[PerfTest.SF_NAME] / frame[PerfTest.PD_NAME]
        frame['pd_outperform'] = frame['sf/pd'].loc[frame['sf/pd'] > 1]

        frame['pd/sf'] = frame[PerfTest.PD_NAME] / frame[PerfTest.SF_NAME]
        frame['sf_outperform'] = frame['pd/sf'].loc[frame['pd/sf'] > 1]


        config = DisplayConfig(cell_max_width=80, type_show=False, display_rows=200)

        def format(v: object) -> str:
            if isinstance(v, float):
                if np.isnan(v):
                    return ''
                return str(round(v, 4))
            return str(v)
        present = frame.iter_element().apply(format)
        present = present[[c for c in present.columns if '/' not in c]]
        print(present.display(config))

        # import ipdb; ipdb.set_trace()
        print('mean: {}'.format(round(frame['sf/pd'].mean(), 6)))
        print('wins: {}/{}'.format((frame['sf/pd'] < 1.05).sum(), len(frame)))



if __name__ == '__main__':
    main()
