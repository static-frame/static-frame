.. _performance:

Performance
===============================

StaticFrame benchmarks its performance with one-to-one comparisons to functionally equivalent Pandas operations. Such measures of performance are poor indicators of real-world use-cases, but provide some insight into general run-time characteristics.


StaticFrame v. Pandas
--------------------------------------

The following tables illustrate absolute run-times ("pd", "sf", where smaller is better) and run-time ratios ("pd_faster", "sf_faster", where larger is better) for functionally equivalent operations. Where a value is given for a run-time ratio, that package is faster; e.g., where "sf_faster" is 4.7, that operation is 4.7 times faster in StaticFrame than it is in Pandas. For details and code for each operation, see :ref:`performance_comparison_code`.


StaticFrame 0.3.4 v. Pandas 0.23.4
.........................................


.. csv-table::
    :header-rows: 1
    :file: performance-0.3.4.txt


.. _performance_comparison_code:


Performance Comparison Test Code
--------------------------------------

Performance tests are based on the following implementations. Follow the source links to view the relevant code.


.. jinja:: ctx

    {% for name in performance_cls %}

    .. autoclass:: static_frame.performance.core.{{ name }}

    {% endfor %}
