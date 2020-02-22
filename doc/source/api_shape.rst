

Size, Shape & Type
=======================================================


:obj:`static_frame.Series` and :obj:`static_frame.Frame` store underlying data in one or two-dimensional NumPy arrays. Attributes similar to those found on NumPy arrays are available to describe the characteristics of the container.




Series
---------

.. autoattribute:: static_frame.Series.shape

.. autoattribute:: static_frame.Series.ndim

.. autoattribute:: static_frame.Series.size

.. autoattribute:: static_frame.Series.nbytes

.. autoattribute:: static_frame.Series.dtype



Examples
................

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_series_shape_a
   :end-before: end_series_shape_a


Frame
---------

.. autoattribute:: static_frame.Frame.shape

.. autoattribute:: static_frame.Frame.ndim

.. autoattribute:: static_frame.Frame.size

.. autoattribute:: static_frame.Frame.nbytes

.. autoattribute:: static_frame.Frame.dtypes



Examples
................

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_frame_shape_a
   :end-before: end_frame_shape_a



