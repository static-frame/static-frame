

Dictionary-Like Interface
===============================

:py:class:`Series` and :py:class:`Frame` provide dictionary-like interfaces.

For more flexible iteration of keys and values, see Iterators, below.


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


Examples
................

.. literalinclude:: api.py
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


Examples
................

.. literalinclude:: api.py
   :language: python
   :start-after: start_frame_dict_like_a
   :end-before: end_frame_dict_like_a


.. admonition:: Deviations from Pandas
    :class: Warning

    For consistency, the iterator returned by :py:meth:`Series.keys` and :py:meth:`Frame.keys` is the same as the iterator returned by iterating the object itself. This deviates from Pandas, where iterating a Series iterates ``pd.Series.values`` while iterating a DataFrame iterates ``pd.DataFrame.keys()``.

