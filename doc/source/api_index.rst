
Index Manipulation
===============================


Index
---------

.. automethod:: static_frame.Index.relabel

.. literalinclude:: api.py
   :language: python
   :start-after: start_index_relabel_a
   :end-before: end_index_relabel_a


Series
---------

.. automethod:: static_frame.Series.relabel

.. literalinclude:: api.py
   :language: python
   :start-after: start_series_relabel_a
   :end-before: end_series_relabel_a


.. automethod:: static_frame.Series.reindex

.. literalinclude:: api.py
   :language: python
   :start-after: start_series_reindex_a
   :end-before: end_series_reindex_a


Frame
---------

.. automethod:: static_frame.Frame.relabel

.. literalinclude:: api.py
   :language: python
   :start-after: start_frame_relabel_a
   :end-before: end_frame_relabel_a


.. automethod:: static_frame.Frame.reindex

.. literalinclude:: api.py
   :language: python
   :start-after: start_frame_reindex_a
   :end-before: end_frame_reindex_a


.. admonition:: Deviations from Pandas
    :class: Warning

    The functionality of the Pandas ``pd.DataFrame.rename()`` and ``pd.Series.rename()`` is available with :py:meth:`Frame.relabel` and :py:meth:`Series.relabel`, respectively.


