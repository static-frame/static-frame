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
* 📚 Comprehensive Documentation: All API endpoints documented with thousands of easily runnable examples.


Code: https://github.com/static-frame/static-frame

Docs: http://static-frame.readthedocs.io

Packages: https://pypi.org/project/static-frame

API Search: https://staticframe.dev



Installation via ``pip``
-------------------------------

Install StaticFrame with ``pip``. Note that pre-built wheels are published for all supported Python versions and platforms (including Apple Silicon platforms)::

    pip install static-frame

To install optional dependencies for full support of input and output formats (such as XLSX and Parquet) via ``pip``::

    pip install static-frame [extras]


Installation via ``conda``
-------------------------------

StaticFrame can be installed via ``conda`` with the ``conda-forge`` channel. Note that pre-built wheels of StaticFrame and all compiled dependencies are available through ``pip`` and may offer more compatibility than a ``conda``-based installation ::

    conda install -c conda-forge static-frame


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


StaticFrame 5
---------------------------------------------------------------

*Make it work, make it right, make it fast*: after years of making it right, StaticFrame 5 makes it fast.

With further integration of performance-critical routines in C (provided by `ArrayKit <https://github.com/static-frame/arraykit>`__ ), many more operations in StaticFrame now outperform Pandas, all while preserving StaticFrame's immutable data model and its consistent, explicit interfaces.

The table below shows representative speed-ups measured on Python 3.14, NumPy 2.4, and Pandas 3.0.5. Examples are reproducible with the self-contained benchmarks that follow. Note that performance results can be highly variable based on specific data shapes and types, and any claim of "always faster" is dubious.

.. list-table::
   :header-rows: 1
   :widths: 38 47 15

   * - Operation
     - StaticFrame interface
     - Speed-up
   * - Rename axis
     - ``Frame.rename(...)``
     - ~70×
   * - Concatenate (axis 1)
     - ``Frame.from_concat(...)``
     - ~30×
   * - Row-wise function application
     - ``Frame.iter_tuple(axis=1).apply(...)``
     - ~15×
   * - Select columns
     - ``Frame[[...]]``
     - ~13×
   * - Set index
     - ``Frame.set_index(...)``
     - ~9×
   * - Ranking (with ties)
     - ``Series.rank_mean(...)``
     - ~6×
   * - Group-by reduction
     - ``Frame.iter_group(...).reduce.from_label_map(...)``
     - ~2×
   * - Pivot table
     - ``Frame.pivot(...)``
     - ~1.7×
   * - Join (unique key)
     - ``Frame.join_left(...)``
     - ~1.2×


All examples build their data with `frame_fixtures <https://github.com/static-frame/frame-fixtures>`__ (imported as ``ff``), and use ``compare()`` to time StaticFrame against an equivalent Pandas call:

.. code-block:: python

>>> import numpy as np
>>> import pandas as pd
>>> import timeit
>>> import static_frame as sf
>>> import frame_fixtures as ff

>>> def compare(label, sf_call, pd_call, *, number):
...    sf_call(); pd_call()  # warm-up
...    st = timeit.timeit(sf_call, number=number) / number
...    pt = timeit.timeit(pd_call, number=number) / number
...    scale, unit = (1e6, 'µs') if min(st, pt) < 1e-3 else (1e3, 'ms')
...    print(f'{label:16} StaticFrame {st*scale:7.1f} {unit} | Pandas {pt*scale:7.1f} {unit} | {pt / st:.1f}x')

No-Copy Operations on Immutable Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because all StaticFrame data is immutable, arrays can be safely shared between containers without defensive copies or complicated copy-on-write (CoW) management. Structural operations (relabeling, selecting columns, setting an index, concatenating) reuse the same underlying NumPy arrays and are often an order of magnitude (or more) faster than Pandas.

.. code-block:: python

    >>> f1 = ff.parse('s(10_000,1000)|v(int,int,str,float)')
    >>> f2 = ff.parse('s(10_000,1000)|v(int,bool,bool,float)')
    >>> df1, df2 = f1.to_pandas(), f2.to_pandas()
    >>> compare('rename', lambda: f1.rename(index='foo'), lambda: df1.rename_axis('foo'), number=10000)
    rename           StaticFrame     8.2 µs | Pandas   567.0 µs | 68.8x
    >>> compare('set index', lambda: f1.set_index(0), lambda: df1.set_index(0, drop=False), number=2000)
    set index        StaticFrame    64.7 µs | Pandas   596.1 µs | 9.2x
    >>> compare('select columns', lambda: f1[[10, 50, 100, 500]], lambda: df1[[10, 50, 100, 500]], number=10000)
    select columns   StaticFrame     5.5 µs | Pandas    69.0 µs | 12.6x
    >>> compare('concat (axis 1)', lambda: sf.Frame.from_concat((f1, f2), axis=1, columns=sf.IndexAutoFactory), lambda: pd.concat((df1, df2), axis=1), number=2000)
    concat (axis 1)  StaticFrame    62.3 µs | Pandas  1781.9 µs | 28.6x


Faster Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Version 5 extends this performance to commonly used group-by, reduce, pivot, join and related operations. A single one-million-row fixture serves these examples:

.. code-block:: python

    >>> f = (ff.parse('s(1_000_000,5)|v(int,int,float,float,float)').relabel(columns=('key', 'r', 'x', 'y', 'z')).assign['key'].apply(lambda s: 'g' + (s % 1000).astype('U4')).assign['r'].apply(lambda s: s % 100_000))
    >>> df = f.to_pandas()
    >>> compare('group-by', lambda: f.iter_group('key').reduce.from_label_map({'x': np.sum, 'y': np.sum}).to_frame(), lambda: df.groupby('key')[['x', 'y']].sum(), number=10)
    group-by         StaticFrame    11.7 ms | Pandas    23.7 ms | 2.0x
    >>> compare('reduce', lambda: f.iter_group('key').reduce.from_label_map({'x': np.sum, 'y': np.max}).to_frame(), lambda: df.groupby('key').agg({'x': 'sum', 'y': 'max'}), number=10)
    reduce           StaticFrame    11.6 ms | Pandas    23.8 ms | 2.0x
    >>> compare('pivot', lambda: f.pivot('key', data_fields='x', func=np.sum), lambda: df.pivot_table(index='key', values='x', aggfunc='sum'), number=10)
    pivot            StaticFrame    13.2 ms | Pandas    22.7 ms | 1.7x
    >>> compare('rank', lambda: f['r'].rank_mean(), lambda: df['r'].rank(method='average'), number=20)
    rank             StaticFrame    18.5 ms | Pandas   108.5 ms | 5.9x
    >>> compare('row-apply', lambda: f.iter_tuple(axis=1).apply(lambda t: t.x * 2 + t.y - t.z), lambda: df.apply(lambda t: t.x * 2 + t.y - t.z, axis=1), number=5)
    row-apply        StaticFrame   485.8 ms | Pandas  5348.0 ms | 11.0x

For join, two frames are built sharing the same 500,000 integer keys, relabeled as strings:

.. code-block:: python

    >>> left = ff.parse('s(500_000,1)|v(float)|i(I,int)').relabel(columns=('lv',))
    >>> right = ff.parse('s(500_000,1)|v(float)|i(I,int)').relabel(columns=('rv',))
    >>> keys = 'k' + left.index.values.astype('U12')
    >>> sf_left = left.relabel(index=keys).sort_index()
    >>> sf_right = right.relabel(index=keys)
    >>> df_left, df_right = sf_left.to_pandas(), sf_right.to_pandas()
    >>> compare('join', lambda: sf_left.join_left(sf_right, left_depth_level=0, right_depth_level=0), lambda: df_left.join(df_right, how='left'), number=10)
    join             StaticFrame    35.7 ms | Pandas    42.1 ms | 1.2x

For the complete performance suite (dozens of comparisons across construction, selection, iteration, grouping, and reduction) run ``python -m static_frame.profile --performance "*"``.
