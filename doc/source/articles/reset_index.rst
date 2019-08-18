
Boring Indices and Where to Find Them: The Auto-Incremented Integer Index in StaticFrame
==========================================================================================

This article is part of a series exploring the features and design of StaticFrame, a Python package that offers data structures similar to the Pandas DataFrame and Series, but with an immutable data model.

This article demonstrates how StaticFrame exposes functionality for creating the most boring index object: the auto-incremented integer index (AIII). While index objects that provide scrutable labels into data are a key feature of libraries like Pandas and StaticFrame, there are many situations where the simple, inscrutable AIII is needed, such as when data does not have a meaningful index, or in concatenation of data with redundant indices. Offering convenient and consistent approaches to creating these indices supports creating more maintainable code.

All examples use StaticFrame 0.3.9 or later (https://pypi.org/project/static-frame) and import with the following convention:


>>> import static_frame as sf


Reindexing
---------------

We will take a brief detour to consider how reindexing works in Pandas and StaticFrame.

The idea of reindexing a ``Series`` or ``Frame`` could be interpreted in at least two ways: (1) create a new container with a new index, supplying labels with values from the old container if those labels are in the old index (i.e., alignment based on index labels) or (2) create a new container with a new index, reusing the same values (alignment based on position).

Following the precedent of Pandas, StaticFrame implements ``Series.reindex()`` and ``Frame.reindex()`` with the former interpretation: alignment based on index labels. For example:


>>> s1 = sf.Series((x * 100 for x in range(1, 5)), index=tuple('wxyz'))
>>> s1
<Series>
<Index>
w        100
x        200
y        300
z        400
<<U1>    <int64>

>>> s1.reindex(tuple('stwx'), fill_value=0)
<Series>
<Index>
s        0
t        0
w        100
x        200
<<U1>    <int64>

To handle the latter interpretation, alignment based on position, Pandas offers at least two approaches: the mutable ``index`` attribute can be directly assigned, or the ``set_axis()`` function can be used.

One way to achieve setting a new index by position in StaticFrame is to create a new ``Series``, reusing the old ``values``. As NumPy arrays in StaticFrame are immutable, reusing ``values`` is practical and efficient: it is a no-copy operation.


>>> sf.Series(s1.values, index=tuple('abcd'))
<Series>
<Index>
a        100
b        200
c        300
d        400
<<U1>    <int64>


Setting an Auto-Incremented Integer Index
------------------------------------------------

One of the more common uses of index assignment based on position is "resetting" the index: replacing an existing index with an auto-incremented integer index (AIII). AIIIs are given to ``Series`` and ``Frame`` created without explicit index arguments; they are also useful when combining data that does not have a "natural" index along an axis.

While Pandas offers a discrete method for this operation, ``reset_index()``, that function is made complex due to the ``drop`` and ``inplace`` parameters. For example, ``reset_index()`` will produce, from a ``pd.Series``, a new ``pd.Series`` or ``pd.Frame`` depending on if ``drop`` is ``True`` or ``False``, and exposes a conflicting parameter configuration if ``drop`` is ``False`` and ``inplace`` is ``True``, raising "TypeError: Cannot reset_index inplace on a Series to create a DataFrame."

A key goal in StaticFrame's API design is to avoid, as much as possible, interfaces that permit conflicting, non-orthogonal arguments.

In addition to reindexing, there are other cases where an AIII might be desired. A common case is in concatenating numerous ``Series`` or ``Frame``. While one axis is aligned, it is common for the new, extended axis to need an AIII. Pandas supports this with a Boolean ``ignore_index`` parameter provided to the ``pd.concat()`` function.

Another goal of StaticFrame's API design is to support common interfaces wherever possible. Reusing, across diverse interfaces, the same mechanism for creating AIIIs is thus desirable.


The ``IndexAutoFactory`` Type
------------------------------------------------

Rather than specialized functions or arguments, AIIIs in StaticFrame can be created on ``Series`` or ``Frame`` by passing a special value, an ``IndexAutoFactory`` object, to index initializer arguments. This is presently supported for ``Series.reindex()``, ``Frame.reindex()``, ``Series.from_concat()``, and ``Frame.from_concat()``. ``Series`` and ``Frame`` initializers similarly can take an ``IndexAutoFactory``.

By using a special type that can be supplied to existing ``index`` or ``columns`` arguments, StaticFrame avoids non-orthogonal arguments and offers a consistent interface for producing AIIIs.


Resetting an Index when Reindexing
------------------------------------------------

By accepting an ``IndexAutoFactory`` argument, a ``reindex()`` method can be used to cover the functionality of the Pandas ``reset_index()`` method.

For example, the ``IndexAutoFactory`` class can be given as the ``index`` argument to ``Series.reindex()`` to produce a new ``Series`` with an AIII. As underlying NumPy arrays are immutable in StaticFrame, this is a no-copy operation.


>>> s1.reindex(sf.IndexAutoFactory)
<Series>
<Index>
0        100
1        200
2        300
3        400
<int64>  <int64>


The benefit of having a specific type, rather than using ``None``, to signify application of an AIII is made more clear in the context of ``Frame.reindex()``, where both a ``columns`` and ``index`` argument can be set independently. The example bellow demonstrates creating a ``Frame``, setting an AIII on both axis, and setting an AIII on ``columns`` while doing conventional reindexing on the ``index``.


>>> f1 = sf.Frame.from_dict(dict(a=(1,2), b=(True, False)), index=tuple('xy'))
>>> f1
<Frame>
<Index> a       b      <<U1>
<Index>
x       1       True
y       2       False
<<U1>   <int64> <bool>

>>> f1.reindex(index=sf.IndexAutoFactory, columns=sf.IndexAutoFactory)
<Frame>
<Index> 0       1      <int64>
<Index>
0       1       True
1       2       False
<int64> <int64> <bool>

>>> f1.reindex(index=tuple('xyz'), columns=sf.IndexAutoFactory)
<Frame>
<Index> 0         1        <int64>
<Index>
x       1.0       True
y       2.0       False
z       nan       nan
<<U1>   <float64> <object>


Resetting an Index when Concatenating
------------------------------------------------

Concatinating ``Series`` and ``Frame`` is a context where supplying a new index is often desirable along the extended axis. The ``IndexAutoFactory`` type can be used here to supply that index.

For example, when concatenating (vertically stacking) with ``Series.from_concat()``, we must supply a new index if the resulting index is not unique. Unlike Pandas, StaticFrame requires all indices to have unique values.


>>> s1
<Series>
<Index>
w        100
x        200
y        300
z        400
<<U1>    <int64>

>>> sf.Series.from_concat((s1, s1), index=tuple('abcdefgh'))
<Series>
<Index>
a        100
b        200
c        300
d        400
e        100
f        200
g        300
h        400
<<U1>    <int64>

However, if an AIII is needed, the ``IndexAutoFactory`` type can be used with the same interface:

>>> sf.Series.from_concat((s1, s1), index=sf.IndexAutoFactory)
<Series>
<Index>
0        100
1        200
2        300
3        400
4        100
5        200
6        300
7        400
<int64>  <int64>


The same approach is used with ``Frame.from_concat()``, where both ``columns`` and ``index`` arguments are exposed. For example, two ``Series`` can be horizontally "stacked" along axis 1 to produce a new ``Frame``. If the ``Series.name`` attributes are unique, they can be used to create the columns; otherwise, new columns can be supplied or an ``IndexAutoFactory`` value can be provided.


>>> s2 = s1 * .5
>>> sf.Frame.from_concat((s1, s2), axis=1, columns=sf.IndexAutoFactory)
<Frame>
<Index> 0       1         <int64>
<Index>
w       100     50.0
x       200     100.0
y       300     150.0
z       400     200.0
<<U1>   <int64> <float64>

Similarly, concatenating along axis 1 (horizontally stacking) the same ``Frame`` multiple times results in non-unique columns, which raises an ``Exception`` in StaticFrame. To avoid this, the ``IndexAutoFactory`` can be supplied.


>>> sf.Frame.from_concat((f1, f1), axis=1, columns=sf.IndexAutoFactory)
<Frame>
<Index> 0       1      2       3      <int64>
<Index>
x       1       True   1       True
y       2       False  2       False
<<U1>   <int64> <bool> <int64> <bool>



Consistent Interfaces for More Maintainable Code
------------------------------------------------

Resetting an index is not a complex operation. However, how to provide the option to create an AIII within diverse interfaces is not obvious. The approach taken with StaticFrame offers a consistent interface, leading to more maintainable code.

For more information about StaticFrame, see the documentation (http://static-frame.readthedocs.io) or project (https://github.com/InvestmentSystems/static-frame) sites. Feedback is encouraged.