
One Fill Value Is Not Enough: Preserving Columnar Types When Reindexing DataFrames
===========================================================================================

    Reindexing DataFrames in Pandas can lead to undesirable degradation of columnar types; StaticFrame offers alternatives that solve this problem.


When working with DataFrames, reindexing is common. When a DataFrame is reindexed, an old index (and its associated values) is conformed to a new index, potentially reordering, contracting, or expanding the rows or columns. When a reindex expands a DataFrame, new values are needed to fill the newly created rows or columns: these are "fill values."

When reindexing with Pandas, only a single value, via the ``fill_value`` parameter, is permitted. If that ``fill_value`` is a type incompatible with the type of one or more columns, the column will be re-cast into a different, likely undesirable, type.

For example, given a DataFrame with three columns typed object, integer, and Boolean, reindexing the index, by default, fills new rows with NaN, a float type that forces the integer column to be converted to float and the Boolean column to be converted to object.

>>> df = pd.DataFrame.from_records((('a', 1, True), ('b', 2, False)), columns=tuple('xyz'))
>>> df
   x  y      z
0  a  1   True
1  b  2  False

>>> df.dtypes.tolist()
[dtype('O'), dtype('int64'), dtype('bool')]

>>> df.reindex((1, 0, 2))
     x    y      z
1    b  2.0  False
0    a  1.0   True
2  NaN  NaN    NaN

>>> df.reindex((1, 0, 2)).dtypes.tolist()
[dtype('O'), dtype('float64'), dtype('O')]


Columnar type degradation is often detrimental. The pre-existing columnar type was probably appropriate for the data; having that type unnecessarily changed simply due to reindexing is often unexpected. Going from one C-level NumPy type to another, such as from int to float, might be tolerable. But when going from C-level NumPy types to arrays of Python objects (object dtypes), performance will be degraded. When reindexing with Pandas, there is no way to avoid this problem.

StaticFrame is an immutable DataFrame library that offers solutions to such problems. In StaticFrame, alternative fill value representations can be used to preserve columnar types in reindexing, shifting, and many other operations that require ``fill_value`` arguments. For operations on heterogeneously typed columnar data, one fill value is simply not enough.

StaticFrame supports providing ``fill_value`` as a single element, as a row-length list of values, as a mapping by column label, or as a ``FillValueAuto``, a novel object to define type-to-value mappings.

All examples use Pandas 1.4.3 and StaticFrame 0.9.6. Imports use the following convention:

>>> import pandas as pd
>>> import static_frame as sf

We can reproduce Pandas behavior in StaticFrame by reindexing the same DataFrame with a single fill value, NaN. This results in the same columnar types as Pandas. Notice that StaticFrame, by default, displays the dtype for each column, making columnar type degradation easily apparent.

>>> f = sf.Frame.from_records((('a', 1, True), ('b', 2, False)), columns=tuple('xyz'))
>>> f
<Frame>
<Index> x     y       z      <<U1>
<Index>
0       a     1       True
1       b     2       False
<int64> <<U1> <int64> <bool>

>>> f.reindex((1, 0, 2), fill_value=np.nan)
<Frame>
<Index> x        y         z        <<U1>
<Index>
1       b        2.0       False
0       a        1.0       True
2       nan      nan       nan
<int64> <object> <float64> <object>


One way to avoid type degradation in reindexing is to provide a fill value per column. With StaticFrame, fill values can be provide with a list, providing one value per column:

>>> f.reindex((1, 0, 2), fill_value=['', 0, False])
<Frame>
<Index> x     y       z      <<U1>
<Index>
1       b     2       False
0       a     1       True
2             0       False
<int64> <<U1> <int64> <bool>


Alternatively, a dictionary can be used to provide a mapping of column label to fill value. If a label is not provided, the default (NaN) will be provided.

>>> f.reindex((1, 0, 2), fill_value={'z':False, 'x':''})
<Frame>
<Index> x     y         z      <<U1>
<Index>
1       b     2.0       False
0       a     1.0       True
2             nan       False
<int64> <<U1> <float64> <bool>


The previous examples all require an explicit value per column, providing maximum specificity. In many cases (and in particular for larger DataFrames), a more general way of specifying fill values is necessary.

One option might be to map a fill value based on specific NumPy dtypes. Such an approach is rejected, as NumPy dtypes define a variable "itemsize" in bytes, leading to a very large number of possible NumPy dtypes. It is more likely that the same fill value would be used for families of dtypes independent of itemsize; for example, all sizes of integers (int8, int16, int32, and int64).

To identify size-independent type families, we can use dtype "kind". NumPy dtypes have a "kind" attribute independent of dtype itemsize: for example, int8, int16, int32, and int64 dtypes are all labeled kind "i". As shown below, there are eleven dtype kinds, each with a one-character label:

+-----------+---------+
|Kind Label |Type     |
+===========+=========+
|b          |bool     |
+-----------+---------+
|i          |int      |
+-----------+---------+
|u          |uint     |
+-----------+---------+
|f          |float    |
+-----------+---------+
|c          |complex  |
+-----------+---------+
|m          |timedelta|
+-----------+---------+
|M          |datetime |
+-----------+---------+
|O          |object   |
+-----------+---------+
|S          |bytes    |
+-----------+---------+
|U          |str      |
+-----------+---------+
|V          |void     |
+-----------+---------+


Specifying a fill value per dtype kind provides a convenient way to avoid columnar type coercions while not requiring a cumbersome specification per column. To do this, StaticFrame introduces a new object: ``FileValueAuto``.

Using the class ``FillValueAuto`` as a fill value provides type-coercion-free defaults for all dtype kinds. If a different mapping is desired, a ``FillValueAuto`` instance can be created, specifying a fill value per dtype kind.

Returning to the previous reindexing example, we see the convenience of using the ``FillValueAuto`` class and that all columnar types are preserved:

>>> f
<Frame>
<Index> x     y       z      <<U1>
<Index>
0       a     1       True
1       b     2       False
<int64> <<U1> <int64> <bool>

>>> f.reindex((1, 0, 2), fill_value=sf.FillValueAuto)
<Frame>
<Index> x     y       z      <<U1>
<Index>
1       b     2       False
0       a     1       True
2             0       False
<int64> <<U1> <int64> <bool>


If we need to deviate from the supplied ``FillValueAuto`` defaults, an instance can be created, specifying fill values per dtype kind. The key-word arguments of the initializer are the single-character dtype kind labels.

>>> f.reindex((1, 0, 2), fill_value=sf.FillValueAuto(U='x', i=-1, b=None))
<Frame>
<Index> x     y       z        <<U1>
<Index>
1       b     2       False
0       a     1       True
2       x     -1      None
<int64> <<U1> <int64> <object>


In StaticFrame, the same multitude of fill value types are accepted nearly everywhere fill values are needed. For example, in shifting data, fill values must be provided; but when shifting an entire DataFrame of heterogeneous types, one fill value is not enough. As shown below, the default ``fill_value``, NaN, forces all columnar types to either object or float.

>>> f = sf.Frame.from_records((('a', 1, True, 'p', 23.2), ('b', 2, False, 'q', 85.1), ('c', 3, True, 'r', 1.23)), columns=tuple('abcde'))

>>> f.shift(2)
<Frame>
<Index> a        b         c        d        e         <<U1>
<Index>
0       nan      nan       nan      nan      nan
1       nan      nan       nan      nan      nan
2       a        1.0       True     p        23.2
<int64> <object> <float64> <object> <object> <float64>


As before, using a ``FillValueAuto`` instance permits a general fill value specification that completely avoids columnar type degradation.

>>> f.shift(2, fill_value=sf.FillValueAuto(U='', b=False, f=0, i=0))
<Frame>
<Index> a     b       c      d     e         <<U1>
<Index>
0             0       False        0.0
1             0       False        0.0
2       a     1       True   p     23.2
<int64> <<U1> <int64> <bool> <<U1> <float64>


A fill value is also needed in many applications of binary operators. In general, binary operations on labelled data force operands to reindex to a union index, potentially introducing missing values. If the missing value is only NaN, the resulting columnar types might be recast.

For example, given two DataFrames, each with a float and an integer column, a binary operation will introduce NaN for reindexed values, coercing the integer column to floats. This can be avoided in StaticFrame by using ``FillValueAuto``.

As binary operators do not accept arguments, StaticFrame provides the ``via_fill_value`` interface to permit specification of a fill value to be used if reindexing is required in binary operations. This is similar to functionality provided by Pandas ``DataFrame.multiply()`` and related methods. With StaticFrame's ``via_fill_value``, we can continue to use expressions of arbitrary binary operators.

When multiplying two DataFrames, each with a column of floats and a column of integers, the introduction of NaNs due to reindexing forces all values to floats.

>>> f1 = sf.Frame.from_records(((10.2, 20), (2.4, 4)), index=('a', 'b'))
>>> f2 = sf.Frame.from_records(((3.4, 1), (8.2, 0)), index=('b', 'c'))

>>> f1 * f2
<Frame>
<Index> 0         1         <int64>
<Index>
a       nan       nan
b       8.16      4.0
c       nan       nan
<<U1>   <float64> <float64>


By using ``via_fill_value`` and ``FillValueAuto``, we can preserve columnar types, even when reindexing is required, and continue to use binary operators in expressions.

>>> f1.via_fill_value(sf.FillValueAuto) * f2
<Frame>
<Index> 0         1       <int64>
<Index>
a       nan       0
b       8.16      4
c       nan       0
<<U1>   <float64> <int64>


Examples with just a few columns, as used above, do not fully demonstrate the power of ``FillValueAuto``: when dealing with heterogeneously typed DataFrames of hundreds or thousands of columns, the generality of specification provides a concise and powerful tool.

The cost of inadvertent type coercion caused by reindexing or other transformations can lead to bugs or degraded performance. StaticFrame's flexible fill value types, as well as the new ``FillValueAuto``, provide solutions to these practical problems.

