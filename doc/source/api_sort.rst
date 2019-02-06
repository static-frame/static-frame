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




