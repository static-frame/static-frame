
Mathematical / Logical / Statistical Utilities
====================================================

:py:class:`Series`, :py:class:`Frame`, and :py:class:`Index`, as well as their derived classes, provide support for common mathematical and statisticasl operations with NumPy.

Index
---------

Mathematical and statistical operations, when applied on an Index, apply to the index labels.

.. jinja:: ctx

    {% for func in Index_ufunc_axis %}

    .. automethod:: static_frame.Index.{{ func }}

    {% endfor %}


Series
---------

.. jinja:: ctx

    {% for func in Series_ufunc_axis %}

    .. automethod:: static_frame.Series.{{ func }}

    {% endfor %}


.. automethod:: static_frame.Series.loc_min

.. automethod:: static_frame.Series.loc_max


.. automethod:: static_frame.Series.iloc_min

.. automethod:: static_frame.Series.iloc_max



Frame
---------

.. jinja:: ctx

    {% for func in Frame_ufunc_axis %}

    .. automethod:: static_frame.Frame.{{ func }}

    {% endfor %}



.. automethod:: static_frame.Frame.loc_min

.. automethod:: static_frame.Frame.loc_max


.. automethod:: static_frame.Frame.iloc_min

.. automethod:: static_frame.Frame.iloc_max



Examples
..................

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_frame_math_logic_a
   :end-before: end_frame_math_logic_a

