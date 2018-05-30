
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

