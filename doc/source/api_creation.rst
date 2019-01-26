
Container Import and Creation
===============================

Both :py:class:`Series` and :py:class:`Frame` have ``from_items`` constructors that consume key/value pairs, such as those returned by ``dict.items()`` and similar functions.


Series
---------

.. automethod:: static_frame.Series.from_items

.. literalinclude:: api.py
   :language: python
   :start-after: start_series_from_items_a
   :end-before: end_series_from_items_a

.. automethod:: static_frame.Series.from_pandas


Frame
---------

.. automethod:: static_frame.Frame.from_items

.. literalinclude:: api.py
   :language: python
   :start-after: start_frame_from_items_a
   :end-before: end_frame_from_items_a


.. automethod:: static_frame.Frame.from_records

.. literalinclude:: api.py
   :language: python
   :start-after: start_frame_from_records_a
   :end-before: end_frame_from_records_a


.. automethod:: static_frame.Frame.from_structured_array

.. literalinclude:: api.py
   :language: python
   :start-after: start_frame_from_structured_array_a
   :end-before: end_frame_from_structured_array_a


.. automethod:: static_frame.Frame.from_concat

.. literalinclude:: api.py
   :language: python
   :start-after: start_frame_from_concat_a
   :end-before: end_frame_from_concat_a


.. automethod:: static_frame.Frame.from_csv

.. literalinclude:: api.py
   :language: python
   :start-after: start_frame_from_csv_a
   :end-before: end_frame_from_csv_a


.. automethod:: static_frame.Frame.from_tsv


.. automethod:: static_frame.Frame.from_json


.. automethod:: static_frame.Frame.from_json_url


.. automethod:: static_frame.Frame.from_pandas
