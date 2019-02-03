

Structures
===============================


Primary Containers
---------------------

The primary components of the StaticFrame library are the one-dimensional :py:class:`static_frame.Series` and the two-dimensional :py:class:`static_frame.Frame` and :py:class:`static_frame.FrameGO`.

While `Frame` and `Series` are immutable, the `FrameGO` permits a type of grow-only mutation, the addition of columns.



.. autoclass:: static_frame.Series

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_series_a
   :end-before: end_series_a


.. autoclass:: static_frame.Frame

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_frame_a
   :end-before: end_frame_a


.. autoclass:: static_frame.FrameGO

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_framego_a
   :end-before: end_framego_a


Index Mappings
---------------------

Index mapping classes are used to map labels to ordinal positions on the :py:class:`Series`, :py:class:`Frame`, and :py:class:`FrameGO`.


.. autoclass:: static_frame.Index

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_index_a
   :end-before: end_index_a


.. autoclass:: static_frame.IndexGO

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_indexgo_a
   :end-before: end_indexgo_a


.. admonition:: Deviations from Pandas
    :class: Warning

    :py:class:`Index` and :py:class:`IndexGO` require that all labels are unique. Duplicated labels will raise an error in all cases. This deviates form Pandas, where Index objects permit duplicate labels. This also makes options like the ``verify_integrity`` argument to ``pd.Series.set_index`` and ``pd.DataFrame.set_index`` unnecessary.


.. autoclass:: static_frame.IndexHierarchy

.. autoclass:: static_frame.IndexHierarchyGO


Utility Objects
---------------------

The following objects are generally only created by internal clients, and thus are not fully documented here.

.. autoclass:: static_frame.TypeBlocks

