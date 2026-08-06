.. figure:: https://raw.githubusercontent.com/static-frame/static-frame/master/doc/images/sf-logo-web_icon-small.png
   :align: center


.. image:: https://img.shields.io/pypi/pyversions/static-frame.svg
  :target: https://pypi.org/project/static-frame

.. image:: https://img.shields.io/pypi/v/static-frame.svg
  :target: https://pypi.org/project/static-frame

.. image:: https://img.shields.io/conda/vn/conda-forge/static-frame.svg
  :target: https://anaconda.org/conda-forge/static-frame


.. image:: https://img.shields.io/codecov/c/github/static-frame/static-frame.svg
  :target: https://codecov.io/gh/static-frame/static-frame


.. image:: https://img.shields.io/github/actions/workflow/status/static-frame/static-frame/ci.yml?branch=master&label=test&logo=Github
  :target: https://github.com/static-frame/static-frame/actions/workflows/ci.yml

.. image:: https://img.shields.io/badge/hypothesis-tested-brightgreen.svg
  :target: https://hypothesis.readthedocs.io

.. image:: https://img.shields.io/pypi/status/static-frame.svg
  :target: https://pypi.org/project/static-frame



static-frame
=============

Immutable and statically-typeable DataFrames with runtime type and data validation.

Among the many Python DataFrame libraries, StaticFrame is an alternative that prioritizes correctness, maintainability, and reducing opportunities for error. Key features include:

* 🛡️ Immutable Data: Provides memory efficiency, excellent performance, and prohibits side effects.
* 🗜️ Static Typing: Use Python type-hints to statically type index, columns, and columnar types.
* 🚦 Runtime Validation: Use type hints and specialized validators for runtime type and data checks.
* 🧭 Consistent Interface: An easy-to-learn, hierarchical, and intuitive API that avoids the many inconsistencies of Pandas.
* 🧬 Comprehensive ``dtype`` Support: Full compatibility with all NumPy dtypes and datetime64 units.
* 🔗 Broad Interoperability: Translate between Pandas, Arrow, Parquet, CSV, TSV, JSON, Excel XLSX, SQLite, and NumPy; output to xarray, VisiData, HTML, RST, Markdown, LaTeX, and Jupyter notebooks.
* 🚀 Optimized Serialization & Memory Mapping: Fast disk I/O with custom NPZ and NPY encodings.
* 💼 Multi-Table Containers: The ``Bus`` and ``Yarn`` provide interfaces to collections of tables with lazy data loading, well-suited for large datasets.
* ⏳ Deferred Processing: The ``Batch`` provides a common interface for deferred processing of groups, windows, or any iterator.
* 🪶 Lean Dependencies: Core functionality relies only on NumPy and a team-maintained C-extension.
* 📚 Comprehensive Documentation: All API endpoints documented with thousands of easily runnable examples.


Code: https://github.com/static-frame/static-frame

Docs: http://static-frame.readthedocs.io

Packages: https://pypi.org/project/static-frame

API Search: https://staticframe.dev



Installation via ``pip``
-------------------------------

Install StaticFrame with ``pip``. Note that pre-built wheels are published for all supported Python versions and platforms (including Apple Silicon platforms)::

    pip install static-frame

To install optional dependencies for full support of input and output formats (such as XLSX and HDF5) via ``pip``::

    pip install static-frame [extras]


Installation via ``conda``
-------------------------------

StaticFrame can be installed via ``conda`` with the ``conda-forge`` channel. Note that pre-built wheels of StaticFrame and all compiled dependencies are available through ``pip`` and may offer more compatibility than a ``conda``-based installation ::

    conda install -c conda-forge static-frame


Installation via Pyodide
-------------------------------

StaticFrame can be run in the browser via Pyodide with the ``static_frame_pyodide`` package: https://github.com/static-frame/static-frame-pyodide


Dependencies
--------------

Core StaticFrame requires the following:

- Python>=3.10
- numpy>=1.24.3 (numpy>=2 is supported)
- arraykit==1.12.0
- typing-extensions>=4.12.0

For extended input and output, the following packages are required:

- pandas>=1.1.5
- xlsxwriter>=1.1.2
- openpyxl>=3.0.9
- xarray>=0.13.0
- pyarrow>=3.0.0
- visidata>=2.4


Performance: Faster than Pandas in Version 5
---------------------------------------------

*"Make it work, make it right, make it fast."* With version 5, StaticFrame is fast.

StaticFrame 5 replaces performance-critical pure-Python routines with ``O(n)`` C primitives (provided by the companion `ArrayKit <https://github.com/static-frame/arraykit>`__ package). On many core operations StaticFrame now *outperforms Pandas* — often by a wide margin — while preserving StaticFrame's immutable data model and its consistent, explicit interfaces.

The table below shows representative speed-ups (Pandas time ÷ StaticFrame time) measured on Python 3.14, NumPy 2.4, and Pandas 3.0. Every figure is reproducible with the self-contained benchmark that follows; your results will vary with hardware, data, and library versions.

.. list-table::
   :header-rows: 1
   :widths: 34 51 15

   * - Operation
     - StaticFrame interface
     - Speed-up
   * - Group-by aggregation
     - ``Frame.iter_group(...).reduce.from_label_map(...)``
     - ~2.6×
   * - Heterogeneous reduction
     - ``Frame.iter_group(...).reduce.from_label_map(...)``
     - ~2.7×
   * - Pivot table
     - ``Frame.pivot(...)``
     - ~2.2×
   * - Ranking (values with ties)
     - ``Series.rank_mean(...)``
     - ~6×
   * - Join (unique key)
     - ``Frame.join_left(...)``
     - ~1.5×
   * - Row-wise function application
     - ``Frame.iter_tuple(axis=1).apply(...)``
     - ~15×


To reproduce these results, first define a small timing helper and some shared data:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import timeit
    import static_frame as sf

    def compare(label, sf_call, pd_call, *, number):
        sf_call(); pd_call()  # warm-up
        st = timeit.timeit(sf_call, number=number) / number
        pt = timeit.timeit(pd_call, number=number) / number
        print(f'{label:10} StaticFrame {st*1e3:6.1f} ms | '
              f'Pandas {pt*1e3:6.1f} ms | {pt / st:.1f}x')

    rng = np.random.default_rng(42)
    N = 1_000_000

    # a low-cardinality string key with two numeric columns
    key = np.array([f'g{i:04d}' for i in range(1000)])[rng.integers(0, 1000, N)]
    f = sf.Frame.from_fields(
            (key, rng.random(N), rng.integers(0, 100, N)),
            columns=('key', 'a', 'b'))
    df = f.to_pandas()


**Group-by aggregation** and **heterogeneous reduction**. StaticFrame reduces groups with an ``O(n)`` C routine that maps a reducer over pre-grouped values, avoiding the per-group Python object construction that dominates a naive group-by:

.. code-block:: python

    compare('group-by',
        lambda: f.iter_group('key').reduce.from_label_map({'a': np.sum, 'b': np.sum}).to_frame(),
        lambda: df.groupby('key')[['a', 'b']].sum(),
        number=10)
    # group-by   StaticFrame    8.5 ms | Pandas   22.8 ms | 2.7x

    compare('reduce',
        lambda: f.iter_group('key').reduce.from_label_map({'a': np.sum, 'b': np.max}).to_frame(),
        lambda: df.groupby('key').agg({'a': 'sum', 'b': 'max'}),
        number=10)
    # reduce     StaticFrame    8.7 ms | Pandas   22.9 ms | 2.6x


**Pivot table**. The same fast grouping and reduction powers ``pivot``, which for low-cardinality keys hash-factorizes labels and reduces with ``np.bincount`` rather than sorting:

.. code-block:: python

    compare('pivot',
        lambda: f.pivot('key', data_fields='a', func=np.sum),
        lambda: df.pivot_table(index='key', values='a', aggfunc='sum'),
        number=10)
    # pivot      StaticFrame   10.0 ms | Pandas   21.6 ms | 2.2x


**Ranking**. Where values contain ties, StaticFrame ranks by hash-factorizing to the *k* unique values, avoiding an ``O(n log n)`` sort of all *n* values. (StaticFrame ranks are 0-based where Pandas ranks are 1-based, but the ordering is identical.)

.. code-block:: python

    s = sf.Series(rng.integers(0, 100_000, N))
    ps = s.to_pandas()
    compare('rank',
        lambda: s.rank_mean(),
        lambda: ps.rank(method='average'),
        number=20)
    # rank       StaticFrame   18.3 ms | Pandas  112.0 ms | 6.1x


**Join**. For a unique join key, StaticFrame maps the two indices with a single hash-join pass:

.. code-block:: python

    M = 500_000
    lkey = np.array([f'k{i}' for i in range(M)])
    rng.shuffle(lkey)
    left = sf.Frame.from_fields((rng.random(M),), columns=('lv',), index=lkey)
    right = sf.Frame.from_fields((rng.random(M),), columns=('rv',),
            index=np.array([f'k{i}' for i in range(M)]))
    pleft, pright = left.to_pandas(), right.to_pandas()
    compare('join',
        lambda: left.join_left(right, left_depth_level=0, right_depth_level=0),
        lambda: pleft.join(pright, how='left'),
        number=10)
    # join       StaticFrame   29.9 ms | Pandas   45.6 ms | 1.5x


**Row-wise function application**. This is where the difference is largest: StaticFrame assembles each row tuple in C and applies the function directly, where Pandas constructs a ``Series`` for every row:

.. code-block:: python

    g = sf.Frame.from_fields(
            (rng.random(100_000), rng.random(100_000), rng.random(100_000)),
            columns=('x', 'y', 'z'))
    dg = g.to_pandas()
    compare('row-apply',
        lambda: g.iter_tuple(axis=1).apply(lambda r: r.x * 2 + r.y - r.z),
        lambda: dg.apply(lambda r: r.x * 2 + r.y - r.z, axis=1),
        number=5)
    # row-apply  StaticFrame   35.7 ms | Pandas  528.0 ms | 14.8x


For the complete performance suite — dozens of comparisons across construction, selection, iteration, grouping, and reduction — run ``python -m static_frame.profile --performance``.

For a broader introduction to StaticFrame, including a worked classification example, articles, videos, and full documentation, see `here <https://static-frame.readthedocs.io/en/latest/intro.html>`__.
