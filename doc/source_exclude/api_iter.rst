
Iterators
===============================

Both :obj:`Series` and :obj:`Frame` offer a variety of iterators (all generators) for flexible transversal of axis and values. In addition, all iterators have a family of apply methods for applying functions to the values iterated. In all cases, alternate "items" versions of iterators are provided; these methods return pairs of (index, value).


.. NOTE: Iterator functionality is implemented with instances of :obj:`IterNode` that, when called, return :obj:`static_frame.IterNodeDelegate` instances. See below for documentation of :obj:`static_frame.IterNodeDelegate` functions for function application on iterables.


Element Iterators
--------------------


Series
........

.. py:method:: Series.iter_element()
.. py:method:: Series.iter_element().apply(func, dtype)
.. py:method:: Series.iter_element().apply_pool(func, dtype, max_workers, chunksize, use_threads)
.. py:method:: Series.iter_element().apply_iter(func)
.. py:method:: Series.iter_element().apply_iter_items(func)

    Iterate over the values of the Series, or expose :obj:`static_frame.IterNodeDelegate` for function application.

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_series_iter_element_a
   :end-before: end_series_iter_element_a


.. py:method:: Series.iter_element_items()
.. py:method:: Series.iter_element_items().apply(func)
.. py:method:: Series.iter_element_items().apply_pool(func, dtype, max_workers, chunksize, use_threads)
.. py:method:: Series.iter_element_items().apply_iter(func)
.. py:method:: Series.iter_element_items().apply_iter_items(func)

    Iterate over pairs of index and values of the Series, or expose :obj:`static_frame.IterNodeDelegate` for function application.

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_series_iter_element_items_a
   :end-before: end_series_iter_element_items_a



.. admonition:: Deviations from Pandas
    :class: Warning

    The functionality of Pandas ``pd.Series.map()`` and ``pd.Series.apply()`` can both be obtained with ``Series.iter_element().apply()``. When given a mapping, ``Series.iter_element().apply()`` will pass original values unchanged if they are not found in the mapping. This deviates from ``pd.Series.map()``, which fills unmapped values with NaN.


Frame
............

.. py:method:: Frame.iter_element()
.. py:method:: Frame.iter_element().apply(func)
.. py:method:: Frame.iter_element().apply_pool(func, dtype, max_workers, chunksize, use_threads)
.. py:method:: Frame.iter_element().apply_iter(func)
.. py:method:: Frame.iter_element().apply_iter_items(func)

    Iterate over the values of the Frame, or expose :obj:`static_frame.IterNodeDelegate` for function application.

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_frame_iter_element_a
   :end-before: end_frame_iter_element_a


.. py:method:: Frame.iter_element_items()
.. py:method:: Frame.iter_element_items().apply(func)
.. py:method:: Frame.iter_element_items().apply_pool(func, dtype, max_workers, chunksize, use_threads)
.. py:method:: Frame.iter_element_items().apply_iter(func)
.. py:method:: Frame.iter_element_items().apply_iter_items(func)

    Iterate over pairs of index / column coordinates and values of the Frame, or expose :obj:`static_frame.IterNodeDelegate` for function application.

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_frame_iter_element_items_a
   :end-before: end_frame_iter_element_items_a


.. admonition:: Deviations from Pandas
    :class: Warning

    The functionality of Pandas ``pd.DataFrame.applymap()`` can be obtained with ``Frame.iter_element().apply()``, though the latter accepts both callables and mapping objects.



Axis Iterators
-----------------

Axis iterators are available on :obj:`Frame` to support iterating on rows or columns as NumPy arrays, named tuples, or :obj:`Series`. Alternative items functions are also available to pair values with the appropriate axis label (either columns or index).


.. py:method:: Frame.iter_array(axis)
.. py:method:: Frame.iter_array(axis).apply(func)
.. py:method:: Frame.iter_array(axis).apply_pool(func, dtype, max_workers, chunksize, use_threads)
.. py:method:: Frame.iter_array(axis).apply_iter(func)
.. py:method:: Frame.iter_array(axis).apply_iter_items(func)

    Iterate over NumPy arrays of Frame axis, where axis 0 iterates column data and axis 1 iterates row data. The returned :obj:`static_frame.IterNodeDelegate` exposes interfaces for function application.

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_frame_iter_array_a
   :end-before: end_frame_iter_array_a


.. py:method:: Frame.iter_array_items(axis)
.. py:method:: Frame.iter_array_items(axis).apply(func)
.. py:method:: Frame.iter_array_items(axis).apply_pool(func, dtype, max_workers, chunksize, use_threads)
.. py:method:: Frame.iter_array_items(axis).apply_iter(func)
.. py:method:: Frame.iter_array_items(axis).apply_iter_items(func)

    Iterate over pairs of label, NumPy array, per Frame axis, where axis 0 iterates column data and axis 1 iterates row data. The returned :obj:`static_frame.IterNodeDelegate` exposes interfaces for function application.

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_frame_iter_array_items_a
   :end-before: end_frame_iter_array_items_a


.. py:method:: Frame.iter_tuple(axis)
.. py:method:: Frame.iter_tuple(axis).apply(func)
.. py:method:: Frame.iter_tuple(axis).apply_pool(func, dtype, max_workers, chunksize, use_threads)
.. py:method:: Frame.iter_tuple(axis).apply_iter(func)
.. py:method:: Frame.iter_tuple(axis).apply_iter_items(func)

    Iterate over NamedTuples of Frame axis, where axis 0 iterates column data and axis 1 iterates row data. The returned :obj:`static_frame.IterNodeDelegate` exposes interfaces for function application.

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_frame_iter_tuple_a
   :end-before: end_frame_iter_tuple_a


.. py:method:: Frame.iter_tuple_items(axis)
.. py:method:: Frame.iter_tuple_items(axis).apply(func)
.. py:method:: Frame.iter_tuple_items(axis).apply_pool(func, dtype, max_workers, chunksize, use_threads)
.. py:method:: Frame.iter_tuple_items(axis).apply_iter(func)
.. py:method:: Frame.iter_tuple_items(axis).apply_iter_items(func)

    Iterate over pairs of label, NamedTuple, per Frame axis, where axis 0 iterates column data and axis 1 iterates row data. The returned :obj:`static_frame.IterNodeDelegate` exposes interfaces for function application.

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_frame_iter_tuple_items_a
   :end-before: end_frame_iter_tuple_items_a


.. py:method:: Frame.iter_series(axis)
.. py:method:: Frame.iter_series(axis).apply(func)
.. py:method:: Frame.iter_series(axis).apply_pool(func, dtype, max_workers, chunksize, use_threads)
.. py:method:: Frame.iter_series(axis).apply_iter(func)
.. py:method:: Frame.iter_series(axis).apply_iter_items(func)

    Iterate over ``Series`` of ``Frame`` axis, where axis 0 iterates column data and axis 1 iterates row data. The returned :obj:`static_frame.IterNodeDelegate` exposes interfaces for function application.

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_frame_iter_series_a
   :end-before: end_frame_iter_series_a


.. py:method:: Frame.iter_series_items(axis)
.. py:method:: Frame.iter_series_items(axis).apply(func)
.. py:method:: Frame.iter_series_items(axis).apply_pool(func, dtype, max_workers, chunksize, use_threads)
.. py:method:: Frame.iter_series_items(axis).apply_iter(func)
.. py:method:: Frame.iter_series_items(axis).apply_iter_items(func)

    Iterate over pairs of label, ``Series``, per Frame axis, where axis 0 iterates column data and axis 1 iterates row data. The returned :obj:`static_frame.IterNodeDelegate` exposes interfaces for function application.

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
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
.. py:method:: Series.iter_group().apply(func)
.. py:method:: Series.iter_group().apply_pool(func, dtype, max_workers, chunksize, use_threads)
.. py:method:: Series.iter_group().apply_iter(func)
.. py:method:: Series.iter_group().apply_iter_items(func)

    Iterator of ``Series`` formed from groups of unique values in a Series.

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_series_iter_group_a
   :end-before: end_series_iter_group_a


.. py:method:: Series.iter_group_items()
.. py:method:: Series.iter_group_items().apply(func)
.. py:method:: Series.iter_group_items().apply_pool(func, dtype, max_workers, chunksize, use_threads)
.. py:method:: Series.iter_group_items().apply_iter(func)
.. py:method:: Series.iter_group_items().apply_iter_items(func)

    Iterator of pairs of group value and the ``Series`` formed from groups of unique values in a Series.

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_series_iter_group_items_a
   :end-before: end_series_iter_group_items_a



.. py:method:: Series.iter_group_labels(depth_level)
.. py:method:: Series.iter_group_labels(depth_level).apply(func)
.. py:method:: Series.iter_group_labels(depth_level).apply_pool(func, dtype, max_workers, chunksize, use_threads)
.. py:method:: Series.iter_group_labels(depth_level).apply_iter(func)
.. py:method:: Series.iter_group_labels(depth_level).apply_iter_items(func)

    Iterator of ``Series`` formed from groups of unique ``Index`` labels.

.. py:method:: Series.iter_group_labels_items(depth_level)
.. py:method:: Series.iter_group_labels_items(depth_level).apply(func)
.. py:method:: Series.iter_group_labels_items(depth_level).apply_pool(func, dtype, max_workers, chunksize, use_threads)
.. py:method:: Series.iter_group_labels_items(depth_level).apply_iter(func)
.. py:method:: Series.iter_group_labels_items(depth_level).apply_iter_items(func)

    Iterator of pairs of group value and ``Series`` formed from groups of unique ``Index`` labels.




Frame
............

.. py:method:: Frame.iter_group(key, axis)
.. py:method:: Frame.iter_group(key, axis).apply(func)
.. py:method:: Frame.iter_group(key, axis).apply_pool(func, dtype, max_workers, chunksize, use_threads)
.. py:method:: Frame.iter_group(key, axis).apply_iter(func)
.. py:method:: Frame.iter_group(key, axis).apply_iter_items(func)

    Iterate over groups (as Frames) based on unique values found in the column specified by ``key``. If axis is 0, subgroups of rows are retuned and key selects columns; If axis is 1, subgroups of columns are returned and key selects rows.

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_frame_iter_group_a
   :end-before: end_frame_iter_group_a


.. py:method:: Frame.iter_group_items(key, axis)
.. py:method:: Frame.iter_group_items(key, axis).apply(func)
.. py:method:: Frame.iter_group_items(key, axis).apply_pool(func, dtype, max_workers, chunksize, use_threads)
.. py:method:: Frame.iter_group_items(key, axis).apply_iter(func)
.. py:method:: Frame.iter_group_items(key, axis).apply_iter_items(func)

    Iterator over pairs of group value and groups (as ``Frame``) based on unique values found in the column specified by ``key``. If axis is 0, subgroups of rows are retuned and key selects columns; If axis is 1, subgroups of columns are returned and key selects rows.

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_frame_iter_group_items_a
   :end-before: end_frame_iter_group_items_a



.. py:method:: Frame.iter_group_labels(depth_level, axis)
.. py:method:: Frame.iter_group_labels(depth_level, axis).apply(func)
.. py:method:: Frame.iter_group_labels(depth_level, axis).apply_pool(func, dtype, max_workers, chunksize, use_threads)
.. py:method:: Frame.iter_group_labels(depth_level, axis).apply_iter(func)
.. py:method:: Frame.iter_group_labels(depth_level, axis).apply_iter_items(func)

    Iterate over groups (as ``Frame``) based on unique labels found in the index specified by ``depth_level``. If axis is 0, subgroups of rows are retuned and ``depth_level`` selects columns; If axis is 1, subgroups of columns are returned and ``depth_level`` selects rows.


.. py:method:: Frame.iter_group_labels_items(depth_level, axis)
.. py:method:: Frame.iter_group_labels_items(depth_level, axis).apply(func)
.. py:method:: Frame.iter_group_labels_items(depth_level, axis).apply_pool(func, dtype, max_workers, chunksize, use_threads)
.. py:method:: Frame.iter_group_labels_items(depth_level, axis).apply_iter(func)
.. py:method:: Frame.iter_group_labels_items(depth_level, axis).apply_iter_items(func)

    Iterator over pairs of group value and groups (as ``Frame``) based on unique labels found in the index specified by ``depth_level``. If axis is 0, subgroups of rows are retuned and ``depth_level`` selects columns; If axis is 1, subgroups of columns are returned and ``depth_level`` selects rows.





Function Application to Iterators
=============================================

:obj:`static_frame.Frame` and :obj:`static_frame.Series` :obj:`static_frame.IterNode` attributes return, when called,  :obj:`static_frame.IterNodeDelegate` instances. These instances are prepared for iteration via :py:meth:`static_frame.IterNodeDelegate.__iter__`, and expose a number of methods for function application.


.. autoclass:: static_frame.IterNode

.. autoclass:: static_frame.IterNodeDelegate

.. automethod:: static_frame.IterNodeDelegate.__iter__

.. automethod:: static_frame.IterNodeDelegate.apply

.. automethod:: static_frame.IterNodeDelegate.apply_pool

.. automethod:: static_frame.IterNodeDelegate.apply_iter

.. automethod:: static_frame.IterNodeDelegate.apply_iter_items


.. automethod:: static_frame.IterNodeDelegate.map_any

.. automethod:: static_frame.IterNodeDelegate.map_any_iter

.. automethod:: static_frame.IterNodeDelegate.map_any_iter_items


.. automethod:: static_frame.IterNodeDelegate.map_fill

.. automethod:: static_frame.IterNodeDelegate.map_fill_iter

.. automethod:: static_frame.IterNodeDelegate.map_fill_iter_items


.. automethod:: static_frame.IterNodeDelegate.map_all

.. automethod:: static_frame.IterNodeDelegate.map_all_iter

.. automethod:: static_frame.IterNodeDelegate.map_all_iter_items
