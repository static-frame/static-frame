
.. role:: python(code)
   :language: python

API Overview
*******************************

This section provides an overview of StaticFrame components and features.

Throughout this section significant differences from Pandas will be called out. As Pandas has made numerous changes to its API over the years, no attempt will be made to isolate what versions of Pandas implement various functionality. Instead, reference will only be made to "versions" of Pandas.


Structures
===============================

Primary Containers
---------------------

The primary components of the StaticFrame library are the 1D :py:class:`Series` and the 2D :py:class:`Frame` and :py:class:`FrameGO`.

.. autoclass:: static_frame.Series

.. literalinclude:: overview.py
   :language: python
   :start-after: start_series_a
   :end-before: end_series_a


.. autoclass:: static_frame.Frame

.. literalinclude:: overview.py
   :language: python
   :start-after: start_frame_a
   :end-before: end_frame_a


.. autoclass:: static_frame.FrameGO

.. literalinclude:: overview.py
   :language: python
   :start-after: start_framego_a
   :end-before: end_framego_a


Index Mappings
---------------------

Index mapping classes are used to map labels to ordinal positions on the :py:class:`Series`, :py:class:`Frame`, and :py:class:`FrameGO`.


.. autoclass:: static_frame.Index

.. literalinclude:: overview.py
   :language: python
   :start-after: start_index_a
   :end-before: end_index_a


.. autoclass:: static_frame.IndexGO

.. literalinclude:: overview.py
   :language: python
   :start-after: start_indexgo_a
   :end-before: end_indexgo_a


.. admonition:: Deviations from Pandas
    :class: Warning

    :py:class:`Index` and :py:class:`IndexGO` require that all labels are unique. Duplicated labels will raise an error in all cases. This deviates form Pandas, where Index objects permit duplicate labels. This also makes options like the ``verify_integrity`` argument to ``pd.Series.set_index`` and ``pd.DataFrame.set_index`` unnecessary.



Utility Objects
---------------------

The following objects are generally only created by internal clients, and thus are not fully documented here.

.. autoclass:: static_frame.TypeBlocks

.. autoclass:: static_frame.IterNode

.. autoclass:: static_frame.IterNodeDelegate




Container Import and Creation
===============================

Both :py:class:`Series` and :py:class:`Frame` have ``from_items`` constructors that consume key/value pairs, such as returned by ``dict.items()`` and similar functions.


Series
---------

.. automethod:: static_frame.Series.from_items

.. literalinclude:: overview.py
   :language: python
   :start-after: start_series_from_items_a
   :end-before: end_series_from_items_a


Frame
---------

.. automethod:: static_frame.Frame.from_items

.. literalinclude:: overview.py
   :language: python
   :start-after: start_frame_from_items_a
   :end-before: end_frame_from_items_a


.. automethod:: static_frame.Frame.from_records

.. literalinclude:: overview.py
   :language: python
   :start-after: start_frame_from_records_a
   :end-before: end_frame_from_records_a


.. automethod:: static_frame.Frame.from_structured_array

.. literalinclude:: overview.py
   :language: python
   :start-after: start_frame_from_structured_array_a
   :end-before: end_frame_from_structured_array_a


.. automethod:: static_frame.Frame.from_concat

.. literalinclude:: overview.py
   :language: python
   :start-after: start_frame_from_concat_a
   :end-before: end_frame_from_concat_a


.. automethod:: static_frame.Frame.from_csv

.. literalinclude:: overview.py
   :language: python
   :start-after: start_frame_from_csv_a
   :end-before: end_frame_from_csv_a


.. automethod:: static_frame.Frame.from_tsv



Dictionary-Like Interface
===============================

:py:class:`Series` and :py:class:`Frame` provide dictionary-like interfaces.

For more flexible iteration of rows or columns, see Iterators, below.


Series
---------

.. automethod:: static_frame.Series.keys

.. automethod:: static_frame.Series.__iter__

.. automethod:: static_frame.Series.__contains__

.. py:attribute:: Series.values

    1D NumPy array of values

.. automethod:: static_frame.Series.items

.. automethod:: static_frame.Series.__len__

.. automethod:: static_frame.Series.get


Examples:
................

.. literalinclude:: overview.py
   :language: python
   :start-after: start_series_dict_like_a
   :end-before: end_series_dict_like_a


Frame
---------

.. automethod:: static_frame.Frame.keys

.. automethod:: static_frame.Frame.__iter__

.. automethod:: static_frame.Frame.__contains__

.. py:attribute:: Frame.values

    2D NumPy array of :py:class:`Frame` values

.. automethod:: static_frame.Frame.items

.. automethod:: static_frame.Frame.__len__

.. automethod:: static_frame.Frame.get


Examples:
................

.. literalinclude:: overview.py
   :language: python
   :start-after: start_frame_dict_like_a
   :end-before: end_frame_dict_like_a


.. admonition:: Deviations from Pandas
    :class: Warning

    For consistency, the iterator returned by :py:meth:`Series.keys` and :py:meth:`Frame.keys` is the same as the iterator returned by iterating the object itself. This deviates from Pandas, where iterating a Series iterates ``pd.Series.values`` while iterating a DataFrame iterates ``pd.DataFrame.keys()``.



Operators
=============================================

:py:class:`Series`, :py:class:`Frame`, and :py:class:`Index`, as well as their derived classes, provide support for the full range of operators available with NumPy. In addition, :py:class:`Series` and  :py:class:`Frame` feature index-alignment and automatic index expansion when both opperands are StaticFrame objects.


Index
---------

Index operators operate on the Index labels. In all cases, an immutable NumPy array is returned rather than a new Index instance.

Unary Operators
..................

.. jinja:: ctx

    {% for func, doc in index_operator_unary %}

    .. py:method:: Index.{{ func }}

        {{ doc }}

    {% endfor %}


Binary Operators
..................

.. jinja:: ctx

    {% for func, doc in index_operator_binary %}

    .. py:method:: Index.{{ func }}(other)

        {{ doc }}

    {% endfor %}



Series
---------

Series operators operate on the Series values. In all cases, a new Series is returned. Operations on two Series always return a new Series with a union Index.

Unary Operators
.................

.. jinja:: ctx

    {% for func, doc in series_operator_unary %}

    .. py:method:: Series.{{ func }}

        {{ doc }}

    {% endfor %}


Binary Operators
..................

.. jinja:: ctx

    {% for func, doc in series_operator_binary %}

    .. py:method:: Series.{{ func }}(other)

        {{ doc }}

    {% endfor %}


Examples
..................

.. literalinclude:: overview.py
   :language: python
   :start-after: start_series_operators_a
   :end-before: end_series_operators_a

.. literalinclude:: overview.py
   :language: python
   :start-after: start_series_operators_b
   :end-before: end_series_operators_b


Frame
---------

Frame operators operate on the Frame values. In all cases, a new Frame is returned.


Unary Operators
..................

.. jinja:: ctx

    {% for func, doc in frame_operator_unary %}

    .. py:method:: Frame.{{ func }}

        {{ doc }}

    {% endfor %}


Binary Operators
..................

.. jinja:: ctx

    {% for func, doc in frame_operator_binary %}

    .. py:method:: Frame.{{ func }}(other)

        {{ doc }}

    {% endfor %}


.. admonition:: Deviations from Pandas
    :class: Warning

    For consistency in operator application and to insure index alignment, all operators return an union index when both opperrands are StaticFrame containers. This deviates from Pandas, where in some versions equality operators did not align on a union index, and behaved differently than other operators.


Examples
..................

.. literalinclude:: overview.py
   :language: python
   :start-after: start_frame_operators_a
   :end-before: end_frame_operators_a




Mathematical / Logical / Statistical Utilities
====================================================

:py:class:`Series`, :py:class:`Frame`, and :py:class:`Index`, as well as their derived classes, provide support for common mathematical and statisticasl operations with NumPy.

Index
---------

Mathematical and statistical operations, when applied on an Index, apply to the index labels.

.. jinja:: ctx

    {% for func, doc, return_type in index_ufunc_axis %}

    .. py:method:: Index.{{ func }}(axis=0, skipna=True) -> {{ return_type }}

    {{ doc }}

    {% endfor %}


Series
---------

.. jinja:: ctx

    {% for func, doc, return_type in series_ufunc_axis %}

    .. py:method:: Series.{{ func }}(axis=0, skipna=True) -> {{ return_type }}

    {{ doc }}

    {% endfor %}


Frame
---------

.. jinja:: ctx

    {% for func, doc, return_type in frame_ufunc_axis %}

    .. py:method:: Frame.{{ func }}(axis=0, skipna=True) -> Series

    {{ doc }}

    {% endfor %}


Examples
..................

.. literalinclude:: overview.py
   :language: python
   :start-after: start_frame_math_logic_a
   :end-before: end_frame_math_logic_a



Index Manipulation
===============================


Index
---------

.. automethod:: static_frame.Index.relabel

.. literalinclude:: overview.py
   :language: python
   :start-after: start_index_relabel_a
   :end-before: end_index_relabel_a


Series
---------

.. automethod:: static_frame.Series.relabel

.. literalinclude:: overview.py
   :language: python
   :start-after: start_series_relabel_a
   :end-before: end_series_relabel_a


.. automethod:: static_frame.Series.reindex

.. literalinclude:: overview.py
   :language: python
   :start-after: start_series_reindex_a
   :end-before: end_series_reindex_a


Frame
---------

.. automethod:: static_frame.Frame.relabel

.. literalinclude:: overview.py
   :language: python
   :start-after: start_frame_relabel_a
   :end-before: end_frame_relabel_a


.. automethod:: static_frame.Frame.reindex

.. literalinclude:: overview.py
   :language: python
   :start-after: start_frame_reindex_a
   :end-before: end_frame_reindex_a


.. admonition:: Deviations from Pandas
    :class: Warning

    The functionality of the Pandas ``pd.DataFrame.rename()`` and ``pd.Series.rename()`` is available with :py:meth:`Frame.relabel` and :py:meth:`Series.relabel`, respectively.




Iterators
===============================

Both :py:class:`Series` and :py:class:`Frame` offer a variety of iterators (all generators) for flexible transversal of axis and values. In addition, all iterators have a family of apply methods for applying functions to the values iterated.


.. NOTE: these methods are are callable instances of IterNode and thus are manually documented as functions


Element Iterators
--------------------


Series
........

.. py:method:: Series.iter_element()

.. py:method:: Series.iter_element_items()


.. admonition:: Deviations from Pandas
    :class: Warning

    The functionality of Pandas ``pd.Series.map()`` and ``pd.Series.apply()`` can both be obtained with ``Series.iter_element().apply()``. When given a mapping, ``Series.iter_element().apply()`` will pass original values unchanged if they are not found in the mapping. This deviates from ``pd.Series.map()``, which fills unmapped values with NaN.


Frame
............

.. py:method:: Frame.iter_element()

.. py:method:: Frame.iter_element_items()



.. admonition:: Deviations from Pandas
    :class: Warning

    The functionality of Pandas ``pd.DataFrame.applymap()`` can be obtained with ``Frame.iter_element().apply()``, though the latter accepts both callables and mapping objects.


Axis Iterators
-----------------

.. py:method:: Frame.iter_array()

.. py:method:: Frame.iter_array_items()

.. py:method:: Frame.iter_tuple()

.. py:method:: Frame.iter_tuple_items()

.. py:method:: Frame.iter_series()

.. py:method:: Frame.iter_series_items()


.. admonition:: Deviations from Pandas
    :class: Warning

    The functionality of Pandas ``pd.DataFrame.itertuples()`` can be obtained with ``Frame.iter_tuple(axis=0)``. The functionality of Pandas ``pd.DataFrame.iterrows()`` can be obtained with ``Frame.iter_series(axis=0)``.  The functionality of Pandas ``pd.DataFrame.iteritems()`` can be obtained with ``Frame.iter_series_items(axis=1)``. The functionality of Pandas ``pd.DataFrame.apply(axis)`` can be obtained with ``Frame.iter_series(axis).apply()``.




Group Iterators
----------------------


Series
........

.. py:method:: Series.iter_group()

    Iterator of groups of unique values (in Series).

.. py:method:: Series.iter_group_items()

    Iterator of pairs of group and grouped values (in a Series).


Frame
............

.. py:method:: Frame.iter_group(key, axis=0)

    Iterate over groups (in Frames) based on unique values selected with key. If axis is 0, subgroups of rows are retuned and key selects columns; If axis is 1, subgroups of columns are returned and key selects rows.

.. py:method:: Frame.iter_group_items(key, axis=0)

    Iterator of pairs of group and grouped values (in a Frame).



Function Application to Iterators
=============================================

:py:class:`Frame` and :py:class:`Series` return :py:class:`IterNodeDelegate` instances when called. These instances are prepared for iteation via :py:meth:`IterNodeDelegate.__iter__`, and in addition, have a number of methods for function application.

.. automethod:: static_frame.IterNodeDelegate.__iter__

.. automethod:: static_frame.IterNodeDelegate.apply

.. automethod:: static_frame.IterNodeDelegate.apply_iter

.. automethod:: static_frame.IterNodeDelegate.apply_iter_items





Assignment
===============================

:py:class:`Series` and :py:class:`Frame` provide asign-to-copy interfaces, permitting assignment with slices similar to Pandas and Numpy.


Series
---------

.. py:method:: Series.assign[key](value)

    Given a key, replace the values specified by the key with value.

.. py:method:: Series.assign.loc[key](value)

    Given a loc key, replace the values specified by the key with value.

.. py:method:: Series.assign.iloc[key](value)

    Given a iloc key, replace the values specified by the key with value.


Frame
---------

.. py:method:: Frame.assign[key](value)

    Given a key, replace the values specified by the key with value.

.. py:method:: Frame.assign.loc[key](value)

    Given a loc key, replace the values specified by the key with value.

.. py:method:: Frame.assign.iloc[key](value)

    Given a iloc key, replace the values specified by the key with value.




Missing Value Handling
===============================

:py:class:`Series` and :py:class:`Frame` provide covnient funcions for finding, dropping, and replacing missing values. In the tradition of Pandas, NaN and None values are treated as both missing, regardless of the dtype in which they are contained.


Series
---------

.. automethod:: static_frame.Series.isna

.. automethod:: static_frame.Series.notna

.. automethod:: static_frame.Series.dropna

.. automethod:: static_frame.Series.fillna



Frame
---------

.. automethod:: static_frame.Frame.isna

.. automethod:: static_frame.Frame.notna

.. automethod:: static_frame.Frame.dropna

.. automethod:: static_frame.Frame.fillna


.. admonition:: Deviations from Pandas
    :class: Warning

    :func:`~static_frame.Frame.dropna` takes a ``condition`` argument, which is a NumPy ufunc that accepts an axis argument. This differs from Pandas ``how`` argument. A ``how`` of "all" is equivalent to a ``condition`` of ``np.all``; A ``how`` of "any" is equivalent to a ``condition`` of ``np.any``.



Sorting
===============================

:py:class:`Index`, :py:class:`Series` and :py:class:`Frame` provide sorting. In all cases, a new object is returned.

Index
---------

.. automethod:: static_frame.Index.sort


Series
---------

.. automethod:: static_frame.Series.sort_index

.. automethod:: static_frame.Series.sort_values


Frame
---------

.. automethod:: static_frame.Frame.sort_index

.. automethod:: static_frame.Frame.sort_columns

.. automethod:: static_frame.Frame.sort_values


.. admonition:: Deviations from Pandas
    :class: Warning

    The default sort kind, delegated to NumPy sorting routines, is merge sort, a stable sort. In some versions of Pandas the default sort kind is quicksort.






Transformations & Utilities
=============================================

The following utilites transform a container into a container of similar size.


Series
---------

.. automethod:: static_frame.Series.isin

.. automethod:: static_frame.Series.transpose

.. automethod:: static_frame.Series.unique

.. automethod:: static_frame.Series.duplicated

.. automethod:: static_frame.Series.drop_duplicated


Frame
---------

.. automethod:: static_frame.Frame.isin

.. automethod:: static_frame.Frame.transpose

.. automethod:: static_frame.Frame.unique

.. automethod:: static_frame.Frame.duplicated

.. automethod:: static_frame.Frame.drop_duplicated

.. automethod:: static_frame.Frame.set_index

.. automethod:: static_frame.Frame.head

.. automethod:: static_frame.Frame.tail


.. admonition:: Deviations from Pandas
    :class: Warning

    Pandas ``pd.DataFrame.duplicated()`` is equivalent to ``Frame.duplicated(exclude_first=True)``. Pandas ``pd.DataFrame.drop_duplicates()`` is equivalent to ``Frame.drop_duplicated(exclude_first=True)``.




Container Export
===============================

Methods for exporting alternative repesentations from :py:class:`Series` and :py:class:`Frame`.

Series
---------

.. automethod:: static_frame.Series.to_pairs

.. automethod:: static_frame.Series.to_pandas


Frame
---------

.. automethod:: static_frame.Frame.to_pairs

.. automethod:: static_frame.Frame.to_pandas

.. automethod:: static_frame.Frame.to_csv

.. automethod:: static_frame.Frame.to_tsv










