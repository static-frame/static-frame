
Operators
=============================================

:obj:`Series`, :obj:`Frame`, and :obj:`Index`, as well as their derived classes, provide support for the full range of operators available with NumPy. In addition, :obj:`Series` and  :obj:`Frame` feature index-alignment and automatic index expansion when both operands are StaticFrame objects.


Index
---------

Index operators operate on the Index labels. In all cases, an immutable NumPy array is returned rather than a new Index instance.

Unary Operators
..................

.. jinja:: ctx

    {% for func, doc in Index_operator_unary %}

    .. py:method:: Index.{{ func }}

        {{ doc }}

    {% endfor %}


Binary Operators
..................

.. jinja:: ctx

    {% for func, doc in Index_operator_binary %}

    .. py:method:: Index.{{ func }}(other)

        {{ doc }}

    {% endfor %}



Series
---------

Series operators operate on the Series values. In all cases, a new Series is returned. Operations on two Series always return a new Series with a union Index.

Unary Operators
.................

.. jinja:: ctx

    {% for func, doc in Series_operator_unary %}

    .. py:method:: Series.{{ func }}

        {{ doc }}

    {% endfor %}


Binary Operators
..................

.. jinja:: ctx

    {% for func, doc in Series_operator_binary %}

    .. py:method:: Series.{{ func }}(other)

        {{ doc }}

    {% endfor %}


Examples
..................

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_series_operators_a
   :end-before: end_series_operators_a

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_series_operators_b
   :end-before: end_series_operators_b


Frame
---------

Frame operators operate on the Frame values. In all cases, a new Frame is returned.


Unary Operators
..................

.. jinja:: ctx

    {% for func, doc in Frame_operator_unary %}

    .. py:method:: Frame.{{ func }}

        {{ doc }}

    {% endfor %}


Binary Operators
..................

.. jinja:: ctx

    {% for func, doc in Frame_operator_binary %}

    .. py:method:: Frame.{{ func }}(other)

        {{ doc }}

    {% endfor %}


.. admonition:: Deviations from Pandas
    :class: Warning

    For consistency in operator application and to insure index alignment, all operators return an union index when both opperrands are StaticFrame containers. This deviates from Pandas, where in some versions equality operators did not align on a union index, and behaved differently than other operators.


Examples
..................

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_frame_operators_a
   :end-before: end_frame_operators_a


