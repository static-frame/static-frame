


'''

Immutable Data & Immutable Data Frames


Benefits of immutable data

  Safety, or Protection against unintended side effects
    Functions mistakenly mutating their arguments
    Interfaces that hand out references to internally stored data

  Performance: Avoiding copying data and over-allocation
    Functions that make defensive copies of data
    All growable interfaces over-allocate for growth

'''

'''
Native immutable data structures

        String, tuple, frozen set
        namedtuple, frozen dataclass

    Proposals for a frozen dict:

        PEP 416 -- Add a frozendict builtin type | Python.org
        PEP 603 -- Adding a frozenmap type to collections | Python.org

'''
from dataclasses import dataclass

import typing as tp
import numpy as np
import static_frame as sf
import pandas as pd
# 'data'[2] = 20
frozenset

class Data1(tp.NamedTuple):
    square: float
    invert: float

@dataclass(frozen=True)
class Data2:
    square: float
    invert: float


'''

Making NumPy arrays immutable

    NumPy arrays are not growable but are mutable

'''



'''
Immutable DataFrames

    Cannot be done with Pandas

    Initializing a DataFrame with 2D array is perfect example of the dangers

    StaticFrame was motivated by this need
'''

a1 = np.arange(5)
s = pd.Series(a1)
a1[2:] = 2000

a1 = np.arange(20).reshape(4,5)
df = pd.DataFrame(a1, index=tuple('abcd'), columns=tuple('ABCDE'))
a1[:, 3] = 0


'''
Advantages of Immutable Data in StaticFrame

  Safety
    Interfaces can store hand hand out references without fear of mutation
    Initialization from arrays is always safe

  Performance
    Initialization from immutable arrays is no-copy
    Extending and concatenating on aligned indicies is no-copy
'''

def processor(
        functions: tp.Iterable[tp.Callable[[tp.Mapping[str, pd.Series]], pd.Series]],
        init: pd.Series
        ) -> tp.Dict[str, pd.Series]:
    results = {'init': init}

    for func in functions:
        results[func.__name__] = func(results.copy())

    return results


class DataInterface:

    def __init__(self, size: int):
        self._square = pd.Series(np.arange(size) ** 2)
        self._invert = -self._square

    @property
    def square(self) -> pd.Series:
        return self._square.copy()

    @property
    def invert(self) -> pd.Series:
        return self._invert.copy()




'''
Assignment like Moves

    If you have immutable container, how to make changes: create a new container
    With a Series, we allocate a new array
    With a Frame, we only mutate what needs to be mutated
'''

# s = sf.Series(range(6), index=tuple('abcdef'))
# s.assign['c'](300)
# s.assign['c':](300)
# s.assign[['a', 'd', 'f']]((-1, -2, -3))
# s.assign[['a', 'd', 'f']](sf.Series.from_dict(dict(f=-3, d=-2, a=-1)))
# s.assign[s.index.isin(('c', 'f'))](100)


# # Examples of defensive copies

# a1 = np.arange(20).reshape(4,5)
# f = sf.Frame(a1, index=tuple('abcd'), columns=tuple('ABCDE'))

# f.assign['C'](False)
# f.assign.loc['c', 'B'](-20)
# f.assign.loc['c', 'B':](-20)
# f.assign.loc['c':, 'B':](False)
# f.assign.loc['c':, 'B':].apply(lambda f: f*1000)






