




The idea of re-indexing a ``Series`` or ``Frame`` could be interpreted in at least two ways: (1) create a new container with the new index, suppling labels with values from the old container if those labels are in the old index, otherwise using a fill value (alignment based on index labels) or (2) create a new container with the new index, reusing the same values (alignment based on position).

Based on the well-defined precedent of Pandas, StaticFrame implements ``Series.reindex()`` and ``Frame.reindex()`` with the former interpretation: alignment based on index labels. For example:

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

To handle the latter interpretation, alignment based on position, Pandas offers at least two approaches: the mutable ``index`` attribute can be directly assigned to mutate the ``Series`` in place, or the ``set_axis()`` function can be used.

As ``Series`` are immutable in StaticFrame, one way to achieve this is to simply create a new ``Series``, reusing the old ``values``. (As NumPy arrays in StaticFrame are immutable, reusing ``values`` is pracitcal and efficient: it is a no-copy operation.)

>>> sf.Series(s1.values, index=tuple('abcd'))
<Series>
<Index>
a        100
b        200
c        300
d        400
<<U1>    <int64>


One of the more common uses of index assignment based on position is "resetting" the index: replacing an existing index with an auto-incremented integer index (AIII). This AIII is the same as an index given to a ``Series`` created without an explicit index argument.

While Pandas offers a discrete method for this operation, ``reset_index()``, that function is made complex due to the ``drop`` and ``inplace`` parameters. For example, ``reset_index()`` will produce, from a ``pd.Series``, a new ``pd.Series`` or ``pd.Frame`` depending on if ``drop`` is ``True`` or ``False``, and exposes a conflicting parameter configuration if ``drop`` is ``False`` and ``inplace`` is ``True``, raising "TypeError: Cannot reset_index inplace on a Series to create a DataFrame." A key goal in StaticFrame's API design is to avoid, as much as possible, interfaces that permit conflicting, non-orthogonal arguments.

In addition to reindexing, there are other cases where an AIII might be desired. Another common case is in catenating numerous ``Series`` or ``Frame``; while alignment on one axis is desired, for the new, extended axis, an AIII might be desirable. Pandas supports this with a Boolean ``ignore_index`` parameter provided to the ``pd.concat`` function.

Another goal of StaticFrame's API design is to support common interfaces wherever possible. To achieve this for AIII production, rather than additional functions or parameter arguments, a special object, ``IndexAutoFactory``, is introduced as a type that can be supplied to arguments that take index initializers. This is introduced in ``Series.reindex``, ``Series.from_concat``, ``Frame.reindex``, and ``Frame.from_concat``. ``Series`` and ``Frame`` initializers similarly can take an ``IndexAutoFactory``.



For ``Series.reindex``, the ``IndexAutoFactor`` class can be given as the ``index`` argument.

While None might be used to signify the same thing, in other contexts (such as ``Frame.reindex``) None already is used to signify no change in the Index.



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