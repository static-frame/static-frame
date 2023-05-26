import math

import numpy as np
import pandas as pd



def approx_equal(a1, a2):
    assert abs((a1 - a2).sum()) < 1e-10

# https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance


# As shown here, there is debate as to how the implementation of adjust=False, ignore_na=False; for this reason ignore_na=True should be the default and their should not be support for ignore_na=True
# https://github.com/pandas-dev/pandas/issues/31178
# https://github.com/pandas-dev/pandas/issues/31178#issuecomment-623051698


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


def get_weights(size: int, alpha: float, adjust: bool) -> np.array:
    # power of 0 causes start value of 1, then 1-alpha, then reductions
    assert 0 < alpha <= 1
    # get 1 in the rightmost position with exponent of 0
    a = np.power(1.0 - alpha, np.arange(size - 1, -1, -1))
    if adjust:
        return a
    # if adjust is false, this is "recursive"
    a[1:] = a[1:] * alpha # keep the left-most value unchanged
    return a

def get_mean(s, alpha: float, adjust: bool, ignore_na: bool) -> float:
    if ignore_na: # the same as calling dropna before processing
        snona = s.dropna()
        if not len(snona):
            return np.nan

        w = get_weights(size=len(snona), alpha=alpha, adjust=adjust)
        return np.average(snona.values, weights=w)

    # if ignore_na is False, we calculate weights the same; simply remove na values from the final calculation
    notna = s.notna()
    if not notna.any():
        return np.nan

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
        # this could return NaN
        array[i] = get_mean(s[:i+1], alpha=alpha, adjust=adjust, ignore_na=ignore_na)

    return pd.Series(array, index=s.index)

def validate_no_na():
    for s in (
        pd.Series(np.arange(100)),
        pd.Series((10, 10, 10, 10)),
        pd.Series((20, 4, 13, 27, 109, 235, 7)),
            ):
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
            print(x.tail(2).values.tolist())

            y = get_series(s, **kwargs)
            print(y.tail(2).values.tolist())

            assert (x.round(10) == y.round(10)).all()

def validate_na():
    s = pd.Series(np.arange(20))
    s[(s + 2) % 10 == 0] = np.nan

    # assume the default, ignore_na = False; we do not implemnt ignore_na = True
    for kwargs in (
            {'alpha': .5, 'adjust': True},
            {'alpha': .5, 'adjust': False},
            {'span': 10, 'adjust': True},
            # {'span': 10, 'adjust': False},
            {'com': 10, 'adjust': True},
            # {'com': 10, 'adjust': False},
            {'halflife': 20, 'adjust': True},
            # {'halflife': 20, 'adjust': False},
            ):
        kwargs['ignore_na'] = False
        print(kwargs)
        x = s.ewm(**kwargs).mean()
        y = get_series(s, **kwargs)
        try:
            assert x.fillna(-1).round(10).tolist() == y.fillna(-1).round(10).tolist()
        except:
            print(x.tail(2).values.tolist())
            print(y.tail(2).values.tolist())
            import ipdb; ipdb.set_trace()

def focus_na():

    for s in (
            # pd.Series([3, np.nan, np.nan, 5]), # Passes all four
            # pd.Series([2, np.nan, 1, 1]), # minimal case that fails
            # pd.Series([0, 1, 2, np.nan, 4]), # docs example
            pd.Series([1, np.nan, 5, 3]), # issue 31178 example

            # pd.Series([np.nan, np.nan, np.nan, 5]),
            # pd.Series([10, 20, np.nan, 30, 40, np.nan, 50, 60, np.nan, 20, np.nan]),
            ):
        print('=' * 10)
        print(s)
        # ignore_na False is the default: weights are spaced along full length
        # when ignore_na is True, only derive weights for the non-nan values
        kwargs = dict(alpha=0.5)
        for ignore_na in (True, False):
            for adjust in (False,):
                # print(f'{adjust=} {ignore_na=}')

                x1 = s.ewm(adjust=adjust, ignore_na=ignore_na, **kwargs).mean()
                y1 = get_series(s, adjust=adjust, ignore_na=ignore_na, **kwargs)
                try:
                    approx_equal(x1.fillna(-1).values, y1.fillna(-1).values)
                except:
                    print(f'---------- failed: {adjust=} {ignore_na=}')
                    print(x1.values.tolist())
                    print(y1.values.tolist())
                    import ipdb; ipdb.set_trace()


    # NOTE: observing deviation when adjust is False, ignore_na is False


if __name__ == '__main__':
    # validate_no_na()
    # validate_na()
    focus_na()


# adjust is False, ignore_na is False
# input 2, NaN, 1, 1


# In : s
# 0    2.0
# 1    NaN
# 2    1.0
# 3    1.0

# In : s.ewm(adjust=False, ignore_na=False, alpha=0.5).mean()
# 0    2.000000
# 1    2.000000
# 2    1.333333
# 3    1.166667
# dtype: float64


# When ignore_na is False, weights are based on absolute positions

# 2:
# (1-a)2, ...,  a

# In : ((2 * 0.25) + (1 * 0.5)) / (0.25 + 0.5)
# 1.3333333333333333

# 3:
# (1-a)3, (1-a)2 * a, (1-a) * a,  a
# 0.125,  ...     , 0.25       , .5

# In : ((2 * 0.125) + (1 * 0.25) + (1 * .5)) / (0.125 + 0.25 + 0.5)
# 1.1428571428571428



# When adjust=False, y[] is defined recursively.
# Without loss of generality, assume x[0] is not NaN.
# First, we set y[0] = x[0].

# Then, for t = 1, 2, ... we set y[t] = y[t-1] if x[t] is NaN, otherwise we set
# y[t] = ((1-alpha)^(t-u) * y[t-1]    +    alpha * x[t])    /    ((1-alpha)^(t-u) + alpha),

# where (i) if ignore_na=True, then u = t-1 (so t-u = 1);
# and (ii) if ignore_na=False, then u is the largest index <t for which x[u] is not NaN.
# Note that in the absence of NaNs, t-u = 1 regardless of whether ignore_na is True or False.

# input is 1, NaN, 5, 3

# y[0] = x[0] = 1.
# y[1] = y[0] = 1.
# y[2] = ((1-0.5)^(2-0) * y[0]   +   0.5 * x[2]) / ((1-0.5)^(2-0) + 0.5) = (0.25*1. + 0.5*5.) / (0.25 + 0.5) = 3.66666...
# y[3] = ((1-0.5)^(3-2) * y[2]   +   0.5 * x[3]) / ((1-0.5)^(3-2) + 0.5) = (0.5*3.66666... + 0.5*3.) / (0.5 + 0.5) = 3.33333...
