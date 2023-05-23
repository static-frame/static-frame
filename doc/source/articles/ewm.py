import numpy as np
import pandas as pd


# https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance


def get_weights(size: int, alpha: float, adjust: bool) -> np.array:
    # power of 0 causes start value of 1, then 1-alpha, then reductions
    assert alpha > 0
    a = np.power(1. - alpha, np.arange(size))
    if adjust:
        return a
    a[:size-1] = a[size-1] * alpha
    return a

def get_mean(s, alpha: float, adjust: bool) -> float:
    w = get_weights(size=len(s), alpha=alpha, adjust=adjust)
    return np.average(s.values, weights=w)

def pandas():

    # Exactly one of com, span, halflife, or alpha must be provided if times is not provided. If times is provided, halflife and one of com, span or alpha may be provided.

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

    # raise ValueError("comass, span, halflife, and alpha are mutually exclusive")
    post1 = s.ewm(alpha=.5).mean()
    print(post1)

    # post2 = s.ewm(halflife=12).mean()
    # print(post2)

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    pandas()