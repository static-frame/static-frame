import numpy as np
import pandas as pd


# https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance


def pandas():

    # Exactly one of com, span, halflife, or alpha must be provided if times is not provided. If times is provided, halflife and one of com, span or alpha may be provided.

    # Span corresponds to what is commonly called an “N-day EW moving average”.
    # Center of mass has a more physical interpretation and can be thought of in terms of span.
    # Half-life is the period of time for the exponential weight to reduce to one half.
    # Alpha specifies the smoothing factor directly.

    # https://pandas.pydata.org/docs/user_guide/window.html#window-exponentially-weighted

    s = pd.Series(np.arange(100))

    # raise ValueError("comass, span, halflife, and alpha are mutually exclusive")
    post1 = s.ewm(com=.5, min_periods=10).mean()
    print(post1)

    post2 = s.ewm(halflife=12, min_periods=10).mean()
    print(post2)

    # NOTE: can use var, std, mean,

    # import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    pandas()