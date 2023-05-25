import math

import numpy as np
import pandas as pd

# https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance


def get_weights(size: int, alpha: float, adjust: bool) -> np.array:
    # power of 0 causes start value of 1, then 1-alpha, then reductions
    assert 0 < alpha <= 1
    # get 1 in the rightmost position with exponent of 0
    a = np.power(1.0 - alpha, np.arange(size - 1, -1, -1))
    if adjust:
        return a
    a[1:] = a[1:] * alpha # keep the left-most value unchanged
    return a

def get_mean(s, alpha: float, adjust: bool, ignore_na: bool) -> float:
    if ignore_na: # the same as calling dropna before processing
        snona = s.dropna()
        w = get_weights(size=len(snona), alpha=alpha, adjust=adjust)
        return np.average(snona.values, weights=w)
    # if ignore_na is False, we calculate weights the same; simply remove na values from the final calculation
    notna = s.notna()
    w = get_weights(size=len(s), alpha=alpha, adjust=adjust)
    return np.average(s.values[notna], weights=w[notna])

LOG05 = math.log(0.5)

def get_series(s, **kwargs):
    if "span" in kwargs:
        alpha = 2 / (kwargs['span'] + 1)
    elif 'com' in kwargs: # center_of_mass
        alpha = 1 / (kwargs['com'] + 1)
    elif 'halflife' in kwargs: # center_of_mass
        alpha = 1 - math.exp(LOG05 / kwargs['halflife'])
    elif 'alpha' in kwargs:
        alpha = kwargs['alpha']

    ignore_na = kwargs.get('ignore_na', False) # false is default
    adjust = kwargs['adjust']
    array = np.empty(s.shape, dtype=float)
    for i in range(len(s)):
        array[i] = get_mean(s[:i+1], alpha=alpha, adjust=adjust, ignore_na=ignore_na)

    return pd.Series(array, index=s.index)



# Span corresponds to what is commonly called an “N-day EW moving average”.
# Center of mass has a more physical interpretation and can be thought of in terms of span.
# Half-life is the period of time for the exponential weight to reduce to one half.
# Alpha specifies the smoothing factor directly.

# https://pandas.pydata.org/docs/user_guide/window.html#window-exponentially-weighted

#  'agg',
#  'aggregate',

# https://pandas.pydata.org/docs/reference/window.html#api-functions-ewm``
#  'corr',
#  'cov',
#  'mean',
#  'std',
#  'var',
#  'sum',

#  'closed',
#  'exclusions',
#  'is_datetimelike',
#  'method',: single, table: only availael with numba
#  'times',: dt64 input
#  'win_type',
#  'window']

def pandas_no_na():
    s = pd.Series(np.arange(100))
    for kwargs in (
            {'alpha': .5, 'adjust': True},
            {'alpha': .5, 'adjust': False},
            {'span': 10, 'adjust': True},
            {'span': 10, 'adjust': False},
            {'com': 10, 'adjust': True},
            {'com': 10, 'adjust': False},
            {'halflife': 20, 'adjust': True},
            {'halflife': 20, 'adjust': False},
            ):

        print(kwargs)
        x = s.ewm(**kwargs).mean()
        print(x.tail(2))
        y = get_series(s, **kwargs)
        print(y.tail(2))
        assert (x.round(10) == y.round(10)).all()

def pandas_na():
    s = pd.Series(np.arange(100))
    s[(s + 2) % 10 == 0] = np.nan
    for kwargs in (
            {'alpha': .5, 'adjust': False, 'ignore_na': True},
            {'span': 10, 'adjust': False, 'ignore_na': True},
            {'alpha': .5, 'adjust': False, 'ignore_na': False},

            # {'com': 10, 'adjust': False, 'ignore_na': False},
            # {'halflife': 20, 'adjust': False, 'ignore_na': False},
            ):

        print(kwargs)
        x = s.ewm(**kwargs).mean()
        print(x.tail(2))
        y = get_series(s, **kwargs)
        print(y.tail(2))
        assert (x.round(10) == y.round(10)).all()


def focus():
    s = pd.Series([3, np.nan, 5])
    x1 = s.ewm(alpha=0.5, ignore_na=False).mean()
    y1 = get_series(s, adjust=True, alpha=0.5, ignore_na=False)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    # pandas_no_na()
    # pandas_na()
    focus()