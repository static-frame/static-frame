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

def get_mean(s, alpha: float, adjust: bool) -> float:
    w = get_weights(size=len(s), alpha=alpha, adjust=adjust)
    # print('weights', w)
    return np.average(s.values, weights=w)

def get_series(s, **kwargs):
    if "span" in kwargs:
        alpha = 2 / (kwargs['span'] + 1)
    elif 'alpha' in kwargs:
        alpha = kwargs['alpha']

    adjust = kwargs['adjust']

    array = np.empty(s.shape, dtype=float)
    for i in range(len(s)):
        array[i] = get_mean(s[:i+1], alpha=alpha, adjust=adjust)
    return pd.Series(array, index=s.index)


def pandas():

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

    s = pd.Series(np.arange(100))

    for kwargs in (
            {'alpha': .5, 'adjust': True},
            {'alpha': .5, 'adjust': False},
            {'span': 10, 'adjust': True},
            {'span': 10, 'adjust': False},
            ):

        print(kwargs)
        x = s.ewm(**kwargs).mean()
        print(x)
        y = get_series(s, **kwargs)
        print(y)
        assert (x.round(10) == y.round(10)).all()



if __name__ == '__main__':
    pandas()