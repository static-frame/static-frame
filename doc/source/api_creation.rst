
Container Import & Creation
===============================

Rather than offering a complex default initializer with arguments that are sometimes irrelevant, :class:`Series` and :class:`Frame` containers offer numerous specialized constructors.

Both :class:`Series` and :class:`Frame` have ``from_items`` constructors that consume key/value pairs, such as those returned by ``dict.items()`` and similar functions.


Series
---------

.. automethod:: static_frame.Series.from_element

.. automethod:: static_frame.Series.from_items

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_series_from_items_a
   :end-before: end_series_from_items_a

.. automethod:: static_frame.Series.from_dict

.. automethod:: static_frame.Series.from_concat

.. automethod:: static_frame.Series.from_concat_items

.. automethod:: static_frame.Series.from_pandas


Frame
---------
.. automethod:: static_frame.Frame.from_element

.. automethod:: static_frame.Frame.from_elements

.. automethod:: static_frame.Frame.from_series

.. automethod:: static_frame.Frame.from_items

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_frame_from_items_a
   :end-before: end_frame_from_items_a


.. automethod:: static_frame.Frame.from_records

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_frame_from_records_a
   :end-before: end_frame_from_records_a

.. automethod:: static_frame.Frame.from_dict_records

.. automethod:: static_frame.Frame.from_records_items

.. automethod:: static_frame.Frame.from_dict_records_items

.. automethod:: static_frame.Frame.from_dict

.. automethod:: static_frame.Frame.from_concat

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_frame_from_concat_a
   :end-before: end_frame_from_concat_a


.. automethod:: static_frame.Series.from_concat_items

.. automethod:: static_frame.Frame.from_delimited

.. automethod:: static_frame.Frame.from_csv

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_frame_from_csv_a
   :end-before: end_frame_from_csv_a


.. automethod:: static_frame.Frame.from_tsv


.. automethod:: static_frame.Frame.from_xlsx


.. automethod:: static_frame.Frame.from_json


.. automethod:: static_frame.Frame.from_json_url


.. automethod:: static_frame.Frame.from_sql


.. automethod:: static_frame.Frame.from_structured_array

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_frame_from_structured_array_a
   :end-before: end_frame_from_structured_array_a


.. automethod:: static_frame.Frame.from_pandas

.. automethod:: static_frame.Frame.from_sqlite

.. automethod:: static_frame.Frame.from_hdf5

.. automethod:: static_frame.Frame.from_arrow

.. automethod:: static_frame.Frame.from_parquet



Index
-----------------

While indices are often specified with their data at container creation team, in some cases explicitly creating indices in advance of the data is practical.



.. automethod:: static_frame.Index.from_labels


.. automethod:: static_frame.IndexHierarchy.from_product

.. automethod:: static_frame.IndexHierarchy.from_tree

.. automethod:: static_frame.IndexHierarchy.from_labels
