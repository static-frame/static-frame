
Index Manipulation
===============================


Index
---------

.. automethod:: static_frame.Index.relabel

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_index_relabel_a
   :end-before: end_index_relabel_a


Series
---------

.. automethod:: static_frame.Series.relabel

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_series_relabel_a
   :end-before: end_series_relabel_a


.. automethod:: static_frame.Series.reindex

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_series_reindex_a
   :end-before: end_series_reindex_a


.. automethod:: static_frame.Series.reindex_flat


.. automethod:: static_frame.Series.reindex_add_level


.. automethod:: static_frame.Series.reindex_drop_level



Frame
---------

.. automethod:: static_frame.Frame.relabel

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_frame_relabel_a
   :end-before: end_frame_relabel_a


.. automethod:: static_frame.Frame.reindex

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_frame_reindex_a
   :end-before: end_frame_reindex_a


.. admonition:: Deviations from Pandas
    :class: Warning

    The functionality of the Pandas ``pd.DataFrame.rename()`` and ``pd.Series.rename()`` is available with :py:meth:`Frame.relabel` and :py:meth:`Series.relabel`, respectively.


.. automethod:: static_frame.Frame.reindex_flat


.. automethod:: static_frame.Frame.reindex_add_level


.. automethod:: static_frame.Frame.reindex_drop_level

