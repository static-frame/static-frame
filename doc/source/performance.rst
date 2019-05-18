Performance
===============================

StaticFrame benchmarks its performance with one-to-one comparisons to functionally equivalent Pandas operations. Such measures of performance are only very general indicators of real-world use-cases.


StaticFrame v. Pandas
--------------------------------------

The following tables illustrate absolute run-times ("pd", "sf") and run-time ratios ("pd_outperform", "sf_outperform") for functionally equivalent operations. Where a value is given for a run-time ratio, that package is faster; e.g., where "sf_outperform" is 4.7, that operation is 4.7 times faster in StaticFrame than it is in Pandas. For details and code for each operation, see :ref:`performance_comparison_code`.


StaticFrame 0.3.4 v. Pandas 0.23.4
.........................................


.. csv-table::
    :header-rows: 1
    :file: performance-0.3.4.txt


.. _performance_comparison_code:

Performance Comparison TestCode
--------------------------------------

Performance tests are based on the following implementations. Follow the source links to view the relevant code.


.. jinja:: ctx

    {% for name in performance_cls %}

    .. autoclass:: static_frame.performance.core.{{ name }}

    {% endfor %}
