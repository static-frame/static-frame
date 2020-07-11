
Boring Indices & Where to Find Them: The Auto-Incremented Integer Index in StaticFrame
==========================================================================================

This article demonstrates how StaticFrame exposes functionality for creating the most boring index object: the auto-incremented integer index (AIII). An AIII makes an axis selectable with integers, just as a NumPy array; it makes ``loc`` selection mostly equivalent to ``iloc`` selection; and it is closely related to "auto increment" integer columns found in databases, such as in MySQL (the ``AUTO_INCREMENT`` keyword), SQLite (the ``AUTOINCREMENT`` keyword), or PostgreSQL (the ``SERIAL`` pseudo-type).

While index objects that provide scrutable labels into data are a key feature of libraries like Pandas and StaticFrame, there are many situations where the simple, inscrutable AIII is needed, such as when data does not have a meaningful index, or in concatenation of data with redundant indices. Offering convenient and consistent approaches to creating these indices supports creating more maintainable code.

All examples use StaticFrame 0.4.0 or later and import with the following convention:

>>> import static_frame as sf


Reindexing & Relabeling
-------------------------

We will take a brief detour to consider how reindexing and relabeling work in Pandas and StaticFrame.

Changing an index on a ``Series`` or ``Frame`` could be done in at least two ways: (1) create a new container with a new index of any size, supplying labels with values from the old container if those labels are in the old index (i.e., alignment based on index labels) or (2) create a new container with a new index of the same size, reusing the same values in the same position (alignment based on position).

Following the precedent of Pandas, StaticFrame implements ``Series.reindex()`` and ``Frame.reindex()`` with the former interpretation: alignment based on index labels. As shown in the example below, the new index only matches and retains two of the four previous values:

.. literalinclude:: ../../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_aiii_fig01
   :end-before: end_aiii_fig01

To handle the latter interpretation, alignment based on position, Pandas offers at least two approaches: the mutable ``index`` attribute can be directly assigned, or the ``set_axis()`` function can be used.

StaticFrame names all methods "relabel" that supply a new or transformed index of the same size, to be aligned by position. The ``Series.relabel()`` method can be used to create a new index by transforming old index labels (via a function or mapping), or by supplying an appropriately sized index initializer. As NumPy arrays in StaticFrame are immutable, relabeling is efficient: underlying data is never copied.

.. literalinclude:: ../../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_aiii_fig02
   :end-before: end_aiii_fig02



Setting an Auto-Incremented Integer Index
------------------------------------------------

A common use of index assignment based on position is "resetting" the index: replacing an existing index with an auto-incremented integer index (AIII). AIIIs are given to ``Series`` and ``Frame`` created without explicit index arguments; they are also useful when combining data that does not have a "natural" index along an axis.

While Pandas offers a discrete method for this operation, ``reset_index()``, that function is made complex due to the ``drop`` and ``inplace`` parameters. For example, ``reset_index()`` will produce, from a ``pd.Series``, a new ``pd.Series`` or a ``pd.Frame`` depending on if ``drop`` is ``True`` or ``False``, and exposes a conflicting parameter configuration if ``drop`` is ``False`` and ``inplace`` is ``True``, raising "TypeError: Cannot reset_index inplace on a Series to create a DataFrame."

A goal in StaticFrame's API design is to avoid, as much as possible, interfaces that permit conflicting, non-orthogonal arguments.

In addition to relabeling, another case where an AIII is frequently needed is in concatenating numerous ``Series`` or ``Frame``. For example, when concatenating a ``Frame``, one axis might be aligned while the other, extended axis requires an AIII. Deviating in naming from of the ``reset_index()`` method, Pandas supports this with a Boolean ``ignore_index`` parameter provided to the ``pd.concat()`` function.

Another goal of StaticFrame's API design is to support common interfaces wherever possible. Reusing, across diverse interfaces, the same mechanism for creating AIIIs is desirable.


The ``IndexAutoFactory`` Type
------------------------------------------------

Rather than specialized functions or arguments, AIIIs in StaticFrame can be created on ``Series`` or ``Frame`` by passing a special value, an ``IndexAutoFactory`` object, to index initializer arguments. This is presently supported for ``Series.relabel()``, ``Frame.relabel()``, ``Series.from_concat()``, and ``Frame.from_concat()``. ``Series`` and ``Frame`` initializers similarly can take an ``IndexAutoFactory``.

By using a special type that can be supplied to existing ``index`` or ``columns`` arguments, StaticFrame avoids non-orthogonal arguments and offers a consistent interface for producing AIIIs.


Resetting an Index when Relabeling
------------------------------------------------

By accepting an ``IndexAutoFactory`` argument, a ``relabel()`` method can be used to cover the functionality of the Pandas ``reset_index()`` method.

For example, the ``IndexAutoFactory`` class can be given as the ``index`` argument to ``Series.relabel()`` to produce a new ``Series`` with an AIII. As mentioned above, as underlying NumPy arrays are immutable in StaticFrame, this is a no-copy operation.


.. literalinclude:: ../../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_aiii_fig03
   :end-before: end_aiii_fig03


The benefit of having a specific type, rather than using ``None``, to signify application of an AIII is made more clear in the context of ``Frame.relabel()``, where both a ``columns`` and ``index`` argument can be set independently. The example bellow demonstrates creating a ``Frame``, setting an AIII on both axis, and setting an AIII on ``columns`` while doing relabeling on the ``index``.


.. literalinclude:: ../../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_aiii_fig04
   :end-before: end_aiii_fig04




Resetting an Index when Concatenating
------------------------------------------------

Concatinating ``Series`` and ``Frame`` is a context where supplying a new index is often desirable along the extended axis. The ``IndexAutoFactory`` type can be used here to supply that index.

For example, when concatenating (vertically stacking) with ``Series.from_concat()``, we must supply a new index if the resulting index is not unique. Unlike Pandas, StaticFrame requires all indices to have unique values.


.. literalinclude:: ../../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_aiii_fig05
   :end-before: end_aiii_fig05


If an AIII is needed, the ``IndexAutoFactory`` type can be used with the same interface:

.. literalinclude:: ../../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_aiii_fig06
   :end-before: end_aiii_fig06


The same approach is used with ``Frame.from_concat()``, where both ``columns`` and ``index`` arguments are exposed. For example, two ``Series`` can be horizontally "stacked" along axis 1 to produce a new ``Frame``. If the ``Series.name`` attributes are unique, they can be used to create the columns; otherwise, new columns can be supplied or an ``IndexAutoFactory`` value can be provided.


.. literalinclude:: ../../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_aiii_fig07
   :end-before: end_aiii_fig07


Similarly, concatenating along axis 1 (horizontally stacking) the same ``Frame`` multiple times results in non-unique columns, which raises an ``Exception`` in StaticFrame. To avoid this, the ``IndexAutoFactory`` can be supplied.


.. literalinclude:: ../../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_aiii_fig08
   :end-before: end_aiii_fig08



Consistent Interfaces for More Maintainable Code
------------------------------------------------

Resetting an index is not a complex operation. However, how to provide the option to create an AIII within diverse interfaces is not obvious. The approach taken with StaticFrame offers a consistent interface, leading to more maintainable code.

For more information about StaticFrame, see the documentation (http://static-frame.readthedocs.io) or project site (https://github.com/InvestmentSystems/static-frame).