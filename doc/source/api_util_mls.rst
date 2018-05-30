
Mathematical / Logical / Statistical Utilities
====================================================

:py:class:`Series`, :py:class:`Frame`, and :py:class:`Index`, as well as their derived classes, provide support for common mathematical and statisticasl operations with NumPy.

Index
---------

Mathematical and statistical operations, when applied on an Index, apply to the index labels.

.. jinja:: ctx

    {% for func, doc, return_type in index_ufunc_axis %}

    .. py:method:: Index.{{ func }}(axis=0, skipna=True) -> {{ return_type }}

    {{ doc }}

    {% endfor %}


Series
---------

.. jinja:: ctx

    {% for func, doc, return_type in series_ufunc_axis %}

    .. py:method:: Series.{{ func }}(axis=0, skipna=True) -> {{ return_type }}

    {{ doc }}

    {% endfor %}


Frame
---------

.. jinja:: ctx

    {% for func, doc, return_type in frame_ufunc_axis %}

    .. py:method:: Frame.{{ func }}(axis=0, skipna=True) -> Series

    {{ doc }}

    {% endfor %}


Examples
..................

.. literalinclude:: api.py
   :language: python
   :start-after: start_frame_math_logic_a
   :end-before: end_frame_math_logic_a

