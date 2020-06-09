

import typing as tp
import time
import sys
import traceback
import random

from static_frame.core.display_color import HexColor

#termtosvg --template window_frame -g 90x20 --command "prlimit --as=800000000 python3 doc/animate/animator.py" /tmp/term.svg

PAUSE_SHORT = object()
PAUSE_LONG = object()
PAUSE_FINAL = object()

class Comment:
    def __init__(self, message, color=0xaaaaaa):
        self.message = message
        self.color = color

    def __iter__(self):
        return HexColor.format_terminal(self.color, self.message).__iter__()

def relabel_concat_low_memory() -> tp.Iterator[str]:
    # return lines of code to execute

    yield Comment("# This example demonstrates one of the many benefits of StaticFrame's use of immutable data by simulating a low-memory environment with `prlimit`. Let us start by importing numpy, pandas, and static_frame")
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

    yield Comment('# We can create a Pandas DataFrame with that array.')
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


    yield Comment('# As StaticFrame is built on immutable arrays, we can relabel the Frame without a MemoryError as underlying data does not need to be copied.')
    yield PAUSE_SHORT

    yield 'f2 = f1.relabel(columns=lambda x: x.upper())'
    yield 'f2.shape'
    yield 'f2.columns'
    yield PAUSE_LONG


    yield Comment('# Similarly, while Pandas runs out of memory, StaticFrame can successfully concatenate the Frame without copying the underlying data.')
    yield PAUSE_SHORT

    yield 'f3 = sf.Frame.from_concat((f1, f2), axis=1)'
    yield 'f3.columns.values'
    yield 'f3.shape'
    yield PAUSE_LONG

class Runner:

    PREFIX = HexColor.format_terminal('lightgrey', '>>> ')
    CHAR_INTERVAL = 0.05 #0.07
    CHAR_JITTER = [x * .01 for x in range(5)]

    @classmethod
    def print_char(cls, char):
        print(char, end='')
        sys.stdout.flush()
        time.sleep(cls.CHAR_INTERVAL + random.choice(cls.CHAR_JITTER))

    @classmethod
    def pause(cls, interval):
        print(cls.PREFIX, end='')
        sys.stdout.flush()
        time.sleep(interval)
        print() # newline
        sys.stdout.flush()


    @classmethod
    def main(cls, func):

        for line in func():
            if line is PAUSE_SHORT:
                cls.pause(0.5)
                continue
            if line is PAUSE_LONG:
                cls.pause(2)
                continue
            if line is PAUSE_FINAL:
                cls.pause(5)
                continue

            print(cls.PREFIX, end='')
            for char in line:
                cls.print_char(char)
            cls.print_char('\n') # get new line

            if isinstance(line, Comment):
                continue

            try:
                post = eval(line)
                if post is not None:
                    print(post)
            except SyntaxError:
                exec(line)
            except MemoryError as e:
                traceback.print_exc(limit=-3)



if __name__ == '__main__':

    Runner.main(relabel_concat_low_memory)


