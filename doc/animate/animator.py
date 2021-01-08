import typing as tp
import time
import sys
import traceback
import random
import argparse
import subprocess

from static_frame.core.display_color import HexColor


class Line:
    pass

PAUSE_SHORT = Line()
PAUSE_LONG = Line()
PAUSE_FINAL = Line()

class Comment(Line):
    def __init__(self, message: str, color: int = 0xaaaaaa) -> None:
        self.message = message
        self.color = color

    def __iter__(self) -> tp.Iterator[str]:
        return HexColor.format_terminal(self.color, self.message).__iter__()

LineIter = tp.Iterator[tp.Union[Line, str]]

#-------------------------------------------------------------------------------
class LineGen:
    CMD_PREFIX = ''

    @staticmethod
    def lines() -> LineIter:
        raise NotImplementedError()

    @classmethod
    def get_animate_command(cls) -> str:
        if cls.CMD_PREFIX:
            return f"{cls.CMD_PREFIX} python3 doc/animate/animator.py --animate {cls.__name__}"
        return f"python3 doc/animate/animator.py --animate {cls.__name__}"



class DisplayConfig(LineGen):

    @staticmethod
    def lines() -> LineIter:
        yield 'import static_frame as sf'
        yield PAUSE_SHORT

        yield "f = sf.FrameGO.from_dict_records_items((('charm', {'symbol':'c', 'mass':1.3}), ('strange', {'symbol':'s', 'mass':0.1})))"
        yield "f['target'] = f['mass'] > 1"
        yield 'f'
        yield PAUSE_SHORT


        yield "sf.DisplayActive.set(sf.DisplayConfig(type_color_float='orange'))"
        yield 'f'
        yield PAUSE_SHORT

        yield "sf.DisplayActive.update(type_color_str='lightcoral', type_color_bool='mediumpurple')"
        yield 'f'
        yield PAUSE_SHORT


        yield "sf.DisplayActive.update(type_color_index=0x4682b4, type_color_frame=0xbfff)"
        yield 'f'
        yield PAUSE_SHORT

        yield 'f.display(sf.DisplayConfig(type_color=False))'
        yield PAUSE_SHORT

        yield 'f.display(sf.DisplayConfig(type_show=False, include_index=False))'
        yield PAUSE_FINAL


class LowMemoryOpsVerbose(LineGen):
    CMD_PREFIX = 'prlimit --as=800000000' # shown to cause expected memory error

    @staticmethod
    def lines() -> LineIter:
        # return lines of code to execute

        yield Comment("# This example demonstrates one of the many benefits of StaticFrame's use of immutable data by simulating a low-memory environment with prlimit. Let us start by importing numpy, pandas, and static_frame")
        yield PAUSE_SHORT

        yield 'import numpy as np'
        yield 'import pandas as pd'
        yield 'import static_frame as sf'
        yield PAUSE_SHORT

        yield Comment('# We will create a large 2D array of integers and a tuple of column labels.')
        yield PAUSE_SHORT

        yield 'a1 = np.arange(10_000_000).reshape(1_000_000, 10)'
        yield "columns = tuple('abcdefghij')"
        yield PAUSE_SHORT

        yield Comment('# Next, we create a Pandas DataFrame using that array.')
        yield PAUSE_SHORT


        yield 'df1 = pd.DataFrame(a1, columns=columns)'
        yield 'df1.shape'
        yield PAUSE_LONG


        yield Comment('# Pandas cannot rename the DataFrame without defensively copying the data, which in this low-memory environment causes a MemoryError.')
        yield PAUSE_SHORT

        yield 'df1.rename(columns=lambda x: x.upper())'
        yield PAUSE_LONG


        yield Comment('# Similarly, concatenating the DataFrame with itself results in a MemoryError.')
        yield PAUSE_SHORT

        yield 'pd.concat((df1, df1), axis=1, ignore_index=True)'
        yield PAUSE_LONG


        yield Comment('# To reuse the same array in StaticFrame, we can make it immutable.')
        yield PAUSE_SHORT

        yield 'a1.flags.writeable = False'
        yield PAUSE_SHORT

        yield 'f1 = sf.Frame(a1, columns=columns)'
        yield 'f1.shape'
        yield PAUSE_SHORT


        yield Comment('# As StaticFrame is built on immutable arrays, we can relabel the Frame without a MemoryError, as underlying data does not need to be copied.')
        yield PAUSE_SHORT

        yield 'f2 = f1.relabel(columns=lambda x: x.upper())'
        yield 'f2.shape'
        yield 'f2.columns'
        yield PAUSE_LONG


        yield Comment('# Similarly, while Pandas runs out of memory, StaticFrame can successfully concatenate the Frame.')
        yield PAUSE_SHORT

        yield 'f3 = sf.Frame.from_concat((f1, f2), axis=1)'
        yield 'f3.columns.values'
        yield 'f3.shape'
        yield PAUSE_LONG


class LowMemoryQuilt(LineGen):
    CMD_PREFIX = 'prlimit --as=800000000' # shown to cause expected memory error

    @staticmethod
    def lines() -> LineIter:
        yield 'import numpy as np'
        yield 'import pandas as pd'
        yield 'import static_frame as sf'
        yield 'from string import ascii_lowercase'
        # yield "df = pd.DataFrame(np.arange(6_000_000 * 4).reshape(6_000_000, 4), columns=tuple('wxyz'))"
        # yield "del df"
        yield "fp = '/tmp/test.zip'"
        yield 'config = sf.StoreConfig(include_index=True, index_depth=1)'
        yield "items = ((ascii_lowercase[i], sf.Frame(np.arange(2_000_000).reshape(500_000, 4), columns=tuple('wxyz'))) for i in range(12))"
        yield 'sf.Batch(items).to_zip_pickle(fp, config=config)'
        yield 'q1 = sf.Quilt.from_zip_pickle(fp, max_persist=1, retain_labels=True, config=config, deepcopy_from_bus=True)'
        yield 'q1.shape'
        yield "q1.iloc[1000000:1000010, 3:]"
        # yield 'sf.Batch(q1.iter_window_items(size=100_000, step=100_000)).mean().to_frame()'



#------------------------------------------------------------------------\------
class Animator:

    PROMPT = HexColor.format_terminal('lightgrey', '>>> ')
    CHAR_INTERVAL = 0.03 #0.07
    CHAR_JITTER = [x * .01 for x in range(6)] + [0.10, .12]

    @classmethod
    def print_char(cls, char: str) -> None:
        print(char, end='')
        sys.stdout.flush()
        # time.sleep(cls.CHAR_INTERVAL + random.choice(cls.CHAR_JITTER))

    @classmethod
    def pause(cls, interval: float) -> None:
        print(cls.PROMPT, end='')
        sys.stdout.flush()
        time.sleep(interval)
        print() # newline
        sys.stdout.flush()


    @classmethod
    def main(cls, func: tp.Callable[[], LineIter]) -> None:

        import static_frame as sf
        import numpy as np
        from string import ascii_lowercase

        for line in func():
            if line is PAUSE_SHORT:
                cls.pause(1)
                continue
            if line is PAUSE_LONG:
                cls.pause(2.5)
                continue
            if line is PAUSE_FINAL:
                cls.pause(5)
                continue

            assert isinstance(line, (Comment, str))

            print(cls.PROMPT, end='')
            for char in line:
                cls.print_char(char)
            cls.print_char('\n') # get new line

            if isinstance(line, Comment):
                continue

            # NOTE: exec puts variables in this scope; pass globals and locals eval, exec
            g = globals()
            g['sf'] = sf
            g['np'] = np
            g['ascii_lowercase'] = ascii_lowercase
            l = locals()
            try:
                post = eval(line, g, l)
                if post is not None:
                    print(post)
            except SyntaxError:
                exec(line, g, l)
            except MemoryError as e:
                traceback.print_exc(limit=-3)


def get_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
            description='Terminal animator',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            )
    # NOTE: when using --animate, CMD_PREFIX must be applied when the command is called
    p.add_argument('--animate',
            help='Name of class to display the animation.',
            )
    p.add_argument('--termtosvg',
            help='Name of class to termtosvg.',
            )
    p.add_argument('--asciinema',
            help='Name of class to asciinema.',
            )
    return p


if __name__ == '__main__':

    options = get_arg_parser().parse_args()
    line_gen = {cls.__name__: cls for cls in (
            LowMemoryOpsVerbose,
            DisplayConfig,
            LowMemoryQuilt,
            )}

    if options.animate:
        cls = line_gen[options.animate]
        Animator.main(cls.lines)

    elif options.termtosvg:
        cls = line_gen[options.termtosvg]

        cmd = ['termtosvg',
            '--template',
            'window_frame',
            '-g', '80x18',
            '--loop-delay', '2000',
            '--command',
            cls.get_animate_command(),
            '/tmp/term.svg',
            ]
        subprocess.run(cmd)

    elif options.asciinema:
        cls = line_gen[options.asciinema]

        cmd = ['asciinema',
            'rec',
            '--command',
            cls.get_animate_command(),
            '/tmp/term.cast',
            ]
        subprocess.run(cmd)
        # NOTE: can upload with asciinema upload /tmp/term.cast