Transformations & Utilities
=============================================

The following utilities transform a container into a container of similar size.


Index
--------

.. automethod:: static_frame.Index.isin

.. automethod:: static_frame.Index.roll

.. automethod:: static_frame.Index.head

.. automethod:: static_frame.Index.tail


Series
---------

.. automethod:: static_frame.Series.astype

.. automethod:: static_frame.Series.clip

.. automethod:: static_frame.Series.isin

.. automethod:: static_frame.Series.transpose

.. automethod:: static_frame.Series.unique

.. automethod:: static_frame.Series.duplicated

.. automethod:: static_frame.Series.drop_duplicated

.. automethod:: static_frame.Series.roll

.. automethod:: static_frame.Series.shift

.. automethod:: static_frame.Series.head

.. automethod:: static_frame.Series.tail


Frame
---------


.. py:method:: Series.astype(dtype)

    Replace the values specified by the key with values casted to the provided dtype.

.. py:method:: Series.astype[key](dtype)

    Given a column key (either a column label, list of column lables, slice of colum labels, or Boolean array), replace the values specified by the column key with values casted to the provided ``dtype``.

.. automethod:: static_frame.Frame.clip

.. automethod:: static_frame.Frame.isin


.. automethod:: static_frame.Frame.transpose

.. automethod:: static_frame.Frame.unique

.. automethod:: static_frame.Frame.duplicated

.. automethod:: static_frame.Frame.drop_duplicated

.. admonition:: Deviations from Pandas
    :class: Warning

    Pandas ``pd.DataFrame.duplicated()`` is equivalent to ``Frame.duplicated(exclude_first=True)``. Pandas ``pd.DataFrame.drop_duplicates()`` is equivalent to ``Frame.drop_duplicated(exclude_first=True)``.


.. automethod:: static_frame.Frame.roll

.. automethod:: static_frame.Frame.shift


.. automethod:: static_frame.Frame.head

.. automethod:: static_frame.Frame.tail


.. automethod:: static_frame.Frame.pivot



