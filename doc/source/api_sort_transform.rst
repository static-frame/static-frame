
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


