
Iterators
===============================

Both :py:class:`Series` and :py:class:`Frame` offer a variety of iterators (all generators) for flexible transversal of axis and values. In addition, all iterators have a family of apply methods for applying functions to the values iterated. In all cases, alternate "items" versions of iterators are provided that return pairs of (index, value).


.. NOTE: Iterator functionality is implemented with instances of :py:class:`IterNode` that, when called, return :py:class:`IterNodeDelegate` instances. See below for documentation of :py:class:`IterNodeDelegate` functions for function application on iterables.


Element Iterators
--------------------


Series
........

.. py:method:: Series.iter_element()

    Iterate over the values of the Series, or expose :py:class:`IterNodeDelegate` for function application.

.. literalinclude:: api.py
   :language: python
   :start-after: start_series_iter_element_a
   :end-before: end_series_iter_element_a


.. py:method:: Series.iter_element_items()

    Iterate over pairs of index and values of the Series, or expose :py:class:`IterNodeDelegate` for function application.

.. literalinclude:: api.py
   :language: python
   :start-after: start_series_iter_element_items_a
   :end-before: end_series_iter_element_items_a



.. admonition:: Deviations from Pandas
    :class: Warning

    The functionality of Pandas ``pd.Series.map()`` and ``pd.Series.apply()`` can both be obtained with ``Series.iter_element().apply()``. When given a mapping, ``Series.iter_element().apply()`` will pass original values unchanged if they are not found in the mapping. This deviates from ``pd.Series.map()``, which fills unmapped values with NaN.


Frame
............

.. py:method:: Frame.iter_element()

    Iterate over the values of the Frame, or expose :py:class:`IterNodeDelegate` for function application.

.. literalinclude:: api.py
   :language: python
   :start-after: start_frame_iter_element_a
   :end-before: end_frame_iter_element_a


.. py:method:: Frame.iter_element_items()

    Iterate over pairs of index / column coordinates and values of the Frame, or expose :py:class:`IterNodeDelegate` for function application.

.. literalinclude:: api.py
   :language: python
   :start-after: start_frame_iter_element_items_a
   :end-before: end_frame_iter_element_items_a


.. admonition:: Deviations from Pandas
    :class: Warning

    The functionality of Pandas ``pd.DataFrame.applymap()`` can be obtained with ``Frame.iter_element().apply()``, though the latter accepts both callables and mapping objects.



Axis Iterators
-----------------

Axis iterators are available on :py:class:`Frame` to support iterating on rows or columns as NumPy arrays, named tuples, or :py:class:`Series`. Alternative items functions are also available to pair values with the appropriate axis label (either columns or index).


.. py:method:: Frame.iter_array()

    Iterate over NumPy arrays of Frame axis, where axis 0 iterates column data and axis 1 iterates row data. The returned :py:class:`IterNodeDelegate` exposes interfaces for function application.

.. literalinclude:: api.py
   :language: python
   :start-after: start_frame_iter_array_a
   :end-before: end_frame_iter_array_a


.. py:method:: Frame.iter_array_items()

    Iterate over pairs of label, NumPy array, per Frame axis, where axis 0 iterates column data and axis 1 iterates row data. The returned :py:class:`IterNodeDelegate` exposes interfaces for function application.

.. literalinclude:: api.py
   :language: python
   :start-after: start_frame_iter_array_items_a
   :end-before: end_frame_iter_array_items_a


.. py:method:: Frame.iter_tuple()

    Iterate over NamedTuples of Frame axis, where axis 0 iterates column data and axis 1 iterates row data. The returned :py:class:`IterNodeDelegate` exposes interfaces for function application.

.. literalinclude:: api.py
   :language: python
   :start-after: start_frame_iter_tuple_a
   :end-before: end_frame_iter_tuple_a


.. py:method:: Frame.iter_tuple_items()

    Iterate over pairs of label, NamedTuple, per Frame axis, where axis 0 iterates column data and axis 1 iterates row data. The returned :py:class:`IterNodeDelegate` exposes interfaces for function application.

.. literalinclude:: api.py
   :language: python
   :start-after: start_frame_iter_tuple_items_a
   :end-before: end_frame_iter_tuple_items_a


.. py:method:: Frame.iter_series()

    Iterate over Series of Frame axis, where axis 0 iterates column data and axis 1 iterates row data. The returned :py:class:`IterNodeDelegate` exposes interfaces for function application.

.. literalinclude:: api.py
   :language: python
   :start-after: start_frame_iter_series_a
   :end-before: end_frame_iter_series_a


.. py:method:: Frame.iter_series_items()

    Iterate over pairs of label, Series, per Frame axis, where axis 0 iterates column data and axis 1 iterates row data. The returned :py:class:`IterNodeDelegate` exposes interfaces for function application.

.. literalinclude:: api.py
   :language: python
   :start-after: start_frame_iter_series_items_a
   :end-before: end_frame_iter_series_items_a


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




