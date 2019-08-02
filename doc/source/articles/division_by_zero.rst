

Division-by-zero in NumPy, Pandas, and StaticFrame
******************************************************

Division by zero of integers and floats in Python raises a ZeroDisionError. This leads programmers to explicitly handle zeros in advance of division, or expect the program to halt via an Exception on zero division. In some contexts, these two consequences are both desirable: the zero handling is made explicit in the code, and any unhandled cases are not handled silently.


https://stackoverflow.com/questions/3004095/division-by-zero-undefined-behavior-or-implementation-defined-in-c-and-or-c


However, for those coming from mathematics and numerical computing, the default assumption, as set in the IEEE Standard for Floating-Point Arithmetic (IEEE 754), is that division by zero is infinity, and division of zero by itself is NaN. As NumPy implements this standard, this is NumPy's default behavior.

https://en.wikipedia.org/wiki/IEEE_754


The difference between the two approaches, for programmers, is that in applications involving massive amounts of messy data full of missing and surprising values, when zero-division raises we are forced to always explicitly handle it. When zero-division silently becomes infinity, we delight in the convenience of not having to handle it, but might later regret lacking the ability to either isolate, observe, or handle such unexpected cases.

The latter is my experience, which is why I favor always raising on zero division. In the context of my work, zero division is not like missing data: to be expected and handled automatically (as NaN propagation). Zero division can mean an assumption of the data model, not just the data, is invalid. In cases where zeros really are expected as divisors, I want to their handling to be explicit: they could be removed from subsequent processing, set to NaN, or something else entirely.

Futher, zero division, as implemented along IEEE 754, produces values that can easily be produced through other means, masking their origin in zero division and potentially allowing downstream calculations to treat them in the larger (and maybe inappropriate) context of those values.

For example, the following dividend and divisor produce two NaNs and two infinity values, one each from zero-division, one each through another common means:

In [109]: np.array([0.34e126, 0, 8.0, 1483.0]) / np.array([np.nan, 0, 0, 1e-1000])
Out[109]: array([nan, nan, inf, inf])

Where there are certainly cases where this behavior is convenient, in my experience there are also cases where this behavior swallows up problematic, unhandled cases, leading zero division being treated as expected NaN and infinity values, even when that zero division was truely not expected.


For this reason, the authors of NumPy expose a way to configure the default behavior of zero division for all integer and floating-point data types, upon both scalar and with n-dimensional arrays.

https://docs.scipy.org/doc/numpy/reference/generated/numpy.seterr.html

(NumPy, through the same mechanism, provides the ability to similarly handle underflow and overflow issues, a case, similar to zero division, that might also be best treated with explicit handling. At this time, however, that check is only applied to scalar types (not arrays), making it less generally useful.)

https://github.com/numpy/numpy/issues/8987





In [103]: s = pd.Series([0.34e126, 0, 8.0, 1483.0])

In [104]: s / [np.nan, 0, 0, 1e-1000]
Out[104]:
0    NaN
1    NaN
2    inf
3    inf
dtype: float64













