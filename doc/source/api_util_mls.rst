
Mathematical / Logical / Statistical Utilities
====================================================

:py:class:`Series`, :py:class:`Frame`, and :py:class:`Index`, as well as their derived classes, provide support for common mathematical and statisticasl operations with NumPy.

Index
---------

Mathematical and statistical operations, when applied on an Index, apply to the index labels.

.. jinja:: ctx

    {% for func in index_ufunc_axis %}

    .. automethod:: static_frame.Index.{{ func }}

    {% endfor %}


Series
---------

.. jinja:: ctx

    {% for func in series_ufunc_axis %}

    .. automethod:: static_frame.Series.{{ func }}

    {% endfor %}


Frame
---------

.. jinja:: ctx

    {% for func in frame_ufunc_axis %}

    .. automethod:: static_frame.Frame.{{ func }}

    {% endfor %}


Examples
..................

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_frame_math_logic_a
   :end-before: end_frame_math_logic_a

