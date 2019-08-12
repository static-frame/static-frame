




Given a ``Series``, reindexing is a well understood transformation: create a new container with that new index, keeping values that are found with labels in the new ``Index``, supplying values not found  in the new index with the ``fill_value`` argument.


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


In some situations, however, not a reindex, but an entirely new index is desired for existing data. In Pandas, the mutable ``index`` attribute can be directly assigned to mutate the ``Series`` in place. As ``Series`` are immutable in StaticFrame, one way to achieve this is to simply create a new ``Series``, reusing the old ``values``. (As NumPy arrays in StaticFrame are immutable, reusing ``values`` is pracitcal and efficient: it is a no-copy operation.)

>>> sf.Series(s1.values, index=tuple('abcd'))
<Series>
<Index>
a        100
b        200
c        300
d        400
<<U1>    <int64>


Perhaps the most common use of new index application is "resetting" an index: dropping an existing index and replacing it with an auto-incremented integer index. This auto-incremented integer index is the same as when creating a ``Series`` an index argument of ``None``.

In Pandas, there is a method for his operation: ``Series.reset_index``.


Instead, using the ``index`` argument is appropriate, as we are specifying what kind of Index is needed. Further, a common convention can be applied in many circumstances.

For ``Series.reindex``, the ``IndexAutoFactor`` class can be given as the ``index`` argument.

While None might be used to signify the same thing, in other contexts (such as ``Frame.reindex``) None already is used to signify no change in the Index.


