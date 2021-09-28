.. figure:: https://raw.githubusercontent.com/InvestmentSystems/static-frame/master/doc/images/sf-logo-web_icon-small.png
   :align: center


.. image:: https://img.shields.io/pypi/pyversions/static-frame.svg
  :target: https://pypi.org/project/static-frame

.. image:: https://img.shields.io/pypi/v/static-frame.svg
  :target: https://pypi.org/project/static-frame

.. image:: https://img.shields.io/conda/vn/conda-forge/static-frame.svg
  :target: https://anaconda.org/conda-forge/static-frame


.. image:: https://img.shields.io/codecov/c/github/InvestmentSystems/static-frame.svg
  :target: https://codecov.io/gh/InvestmentSystems/static-frame



.. image:: https://img.shields.io/github/workflow/status/InvestmentSystems/static-frame/Test?label=test&logo=Github
  :target: https://github.com/InvestmentSystems/static-frame/actions?query=workflow%3ATest

.. image:: https://img.shields.io/github/workflow/status/InvestmentSystems/static-frame/TestForward?label=test-forward&logo=Github
  :target: https://github.com/InvestmentSystems/static-frame/actions?query=workflow%3ATestForward

.. image:: https://img.shields.io/github/workflow/status/InvestmentSystems/static-frame/Quality?label=quality&logo=Github
  :target: https://github.com/InvestmentSystems/static-frame/actions?query=workflow%3AQuality



.. image:: https://img.shields.io/readthedocs/static-frame.svg
  :target: https://static-frame.readthedocs.io/en/latest

.. image:: https://img.shields.io/badge/hypothesis-tested-brightgreen.svg
  :target: https://hypothesis.readthedocs.io

.. image:: https://img.shields.io/pypi/status/static-frame.svg
  :target: https://pypi.org/project/static-frame

.. image:: https://img.shields.io/badge/benchmarked%20by-asv-blue.svg
  :target: https://investmentsystems.github.io/static-frame-benchmark



static-frame
=============

A library of immutable and grow-only Pandas-like DataFrames with a more explicit and consistent interface. StaticFrame is suitable for applications in data science, data engineering, finance, scientific computing, and related fields where reducing opportunities for error by prohibiting in-place mutation is critical.

While many interfaces are similar to Pandas, StaticFrame deviates from Pandas in many ways: all data is immutable, and all indices are unique; the full range of NumPy data types is preserved, and date-time indices use discrete NumPy types; hierarchical indices are seamlessly integrated; and uniform approaches to element, row, and column iteration and function application are provided. Core StaticFrame depends only on NumPy and two C-extension packages (maintained by the StaticFrame team): Pandas is not a dependency.

A wide variety of table storage and representation formats are supported, including input from and output to CSV, TSV, JSON, MessagePack, Excel XLSX, SQLite, HDF5, NumPy, Pandas, Arrow, and Parquet; additionally, output to xarray, VisiData, HTML, RST, Markdown, and LaTeX is supported, as well as HTML representations in Jupyter notebooks.

StaticFrame features a family of multi-table containers: the Bus is a lazily-loaded container of tables, the Batch is a deferred processor of tables, the Yarn is virtual concatenation of many Buses, and the Quilt is a virtual concatenation of all tables within a single Bus or Yarn. All permit operating on large collections of tables with minimal memory overhead, as well as writing too and reading from zipped bundles of pickles, Parquet, or delimited files, as well as XLSX workbooks, SQLite, and HDF5.


Code: https://github.com/InvestmentSystems/static-frame

Docs: http://static-frame.readthedocs.io

Packages: https://pypi.org/project/static-frame

Benchmarks: https://investmentsystems.github.io/static-frame-benchmark

Context: `Ten Reasons to Use StaticFrame instead of Pandas <https://dev.to/flexatone/ten-reasons-to-use-staticframe-instead-of-pandas-4aad>`_


Why Immutable Data?
-------------------------------

The following example, executed in a low-memory environment (using ``prlimit``), shows how Pandas cannot re-label columns of a DataFrame or concatenate a DataFrame to itself without copying underlying data. By using immutable NumPy arrays, StaticFrame can perform these operations in the same low-memory environment. By reusing immutable arrays without copying, StaticFrame can achieve more efficient memory usage.

.. image:: https://raw.githubusercontent.com/InvestmentSystems/static-frame/master/doc/images/animate-low-memory-ops-verbose.svg
   :align: center


Installation
-------------------------------

Install StaticFrame via PIP::

    pip install static-frame

Or, install StaticFrame via conda::

    conda install -c conda-forge static-frame

To install full support of input and output routines via PIP::

    pip install static-frame [extras]


Dependencies
--------------

Core StaticFrame requires the following:

- Python >= 3.6
- NumPy >= 1.17.4
- automap >= 0.4.8
- arraykit >= 0.1.8

For extended input and output, the following packages are required:

- pandas >= 0.23.4
- xlsxwriter >= 1.1.2
- openpyxl >= 3.0.9
- xarray >= 0.13.0
- tables >= 3.6.1
- pyarrow >= 0.16.0
- visidata >= 2.4


Quick-Start Guide
---------------------

StaticFrame provides numerous methods for loading and creating data, either as a 1D ``Series`` or a 2D ``Frame``. All creation routines are exposed as alternate constructors on the desired class, such as ``Frame.from_records()``, ``Frame.from_csv()`` or ``Frame.from_pandas()``.

.. note::

    For a concise overview of all StaticFrame interfaces, see `API Overview <https://static-frame.readthedocs.io/en/latest/api_overview>`_.


For example, we can load JSON data from a URL using ``Frame.from_json_url()``, and then use ``Frame.head()`` to reduce the displayed output to just the first five rows. (Passing explicit ``dtypes`` is only necessary on Windows.)

>>> import numpy as np
>>> import static_frame as sf

>>> frame = sf.Frame.from_json_url('https://jsonplaceholder.typicode.com/photos', dtypes=dict(albumId=np.int64, id=np.int64))

>>> frame.head()
<Frame>
<Index> albumId id      title                url                  thumbnailUrl         <<U12>
<Index>
0       1       1       accusamus beatae ... https://via.place... https://via.place...
1       1       2       reprehenderit est... https://via.place... https://via.place...
2       1       3       officia porro iur... https://via.place... https://via.place...
3       1       4       culpa odio esse r... https://via.place... https://via.place...
4       1       5       natus nisi omnis ... https://via.place... https://via.place...
<int64> <int64> <int64> <<U86>               <<U38>               <<U38>


.. note::

    The Pandas CSV reader out-performs the NumPy-based reader in StaticFrame: thus, for now, using ``Frame.from_pandas(pd.read_csv(fp))`` is recommended for loading large CSV files.

    For more information on Frame constructors, see `Frame: Constructor <https://static-frame.readthedocs.io/en/latest/api_detail/frame.html#frame-constructor>`_.


As with a NumPy array, the ``Frame`` exposes common attributes of shape and size.

>>> frame.shape
(5000, 5)
>>> frame.size
25000
>>> frame.nbytes
3320000


Unlike a NumPy array, a Frame stores heterogeneous types, where each column is a single type. StaticFrame preserves the full range of NumPy types, including fixed-size character strings. Character strings can be converted to Python objects or other types as needed with the ``Frame.astype`` interface, which exposes a ``__getitem__`` style interface for selecting columns to convert. As with all similar functions, a new ``Frame`` is returned.

>>> frame.dtypes
<Series>
<Index>
albumId      int64
id           int64
title        <U86
url          <U38
thumbnailUrl <U38
<<U12>       <object>

>>> frame.astype['title':](object).dtypes
<Series>
<Index>
albumId      int64
id           int64
title        object
url          object
thumbnailUrl object
<<U12>       <object>


Utility functions common to Pandas users are available on ``Frame`` and ``Series``, such as ``Series.unique()``, ``Series.isna()``, and ``Series.any()``.

>>> frame['albumId'].unique().tolist()
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
>>> frame['id'].isna().any()
False

.. note::

    For more information on Frame utility functions, see `Frame: Method <https://static-frame.readthedocs.io/en/latest/api_detail/frame.html#frame-method>`_.

StaticFrame interfaces for extracting data will be familiar to Pandas users, though with a number of interface refinements to remove redundancies and increase consistency. On a ``Frame``, ``__getitem__`` is (exclusively) a column selector; ``loc`` and ``iloc`` are (with one argument) row selectors or (with two arguments) row and column selectors.

For example we can select a single column with ``__getitem__``:

>>> frame['albumId'].tail()
<Series: albumId>
<Index>
4995              100
4996              100
4997              100
4998              100
4999              100
<int64>           <int64>


Consistent with other ``__getitem__`` style selectors, a slice or a list can be used to select columns:

>>> frame['id':'title'].head()
<Frame>
<Index> id      title                <<U12>
<Index>
0       1       accusamus beatae ...
1       2       reprehenderit est...
2       3       officia porro iur...
3       4       culpa odio esse r...
4       5       natus nisi omnis ...
<int64> <int64> <<U86>


The ``loc`` interface, with one argument, returns a ``Series`` for the row found at the given index label.

>>> frame.loc[4]
<Series: 4>
<Index>
albumId      1
id           5
title        natus nisi omnis ...
url          https://via.place...
thumbnailUrl https://via.place...
<<U12>       <object>


With two arguments, ``loc`` can select both rows and columns at the same time:

>>> frame.loc[4:8, ['albumId', 'title']]
<Frame>
<Index> albumId title                <<U12>
<Index>
4       1       natus nisi omnis ...
5       1       accusamus ea aliq...
6       1       officia delectus ...
7       1       aut porro officii...
8       1       qui eius qui aute...
<int64> <int64> <<U86>


Where the ``loc`` interface uses index and column labels, the ``iloc`` interface uses integer offsets from zero, just as if the ``Frame`` were a NumPy array. For example, we can select the last row with ``-1``:

>>> frame.iloc[-1]
<Series: 4999>
<Index>
albumId        100
id             5000
title          error quasi sunt ...
url            https://via.place...
thumbnailUrl   https://via.place...
<<U12>         <object>


Or, using two arguments, we can select the first two columns of the last two rows:

>>> frame.iloc[-2:, 0:2]
<Frame>
<Index> albumId id      <<U12>
<Index>
4998    100     4999
4999    100     5000
<int64> <int64> <int64>


.. As providing both axis arguments at the same time is always more efficient than sequential selections, StaticFrame provides a selection wrapper, ``ILoc``, which permits including an ``iloc``-style seleciton in a ``loc`` selection:
.. Example here fails!
.. frame.loc[sf.ILoc[-1], ['id', 'title', 'url']]



Just as with Pandas, expressions can be used in ``__getitem__``, ``loc``, and ``iloc`` statements to create more narrow selections. For example, we can select all "albumId" greater than or equal to 98.

>>> frame.loc[frame['albumId'] >= 98, ['albumId', 'title']].head()
<Frame>
<Index> albumId title                <<U12>
<Index>
4850    98      aut aut nulla vol...
4851    98      ducimus neque del...
4852    98      fugit officiis su...
4853    98      pariatur temporib...
4854    98      qui inventore inc...
<int64> <int64> <<U86>


However, unlike Pandas, ``__getitem__``, ``loc``, and ``iloc`` cannot be used for assignment or in-place mutation on a ``Frame`` or ``Series``. Throughout StaticFrame, all underlying NumPy arrays, and all container attributes, are immutable. Making data and objects immutable reduces opportunities for coding errors and offers, in some situations, greater efficiency by avoiding defensive copies.

>>> frame.loc[4854, 'albumId']
98
>>> frame.loc[4854, 'albumId'] = 200
Traceback (most recent call last):
TypeError: 'InterfaceGetItem' object does not support item assignment
>>> frame.values[4854, 0] = 200
Traceback (most recent call last):
ValueError: assignment destination is read-only


.. note::

    For more information on Frame selection interfaces, see `Frame: Selector <https://static-frame.readthedocs.io/en/latest/api_detail/frame.html#frame-selector>`_.


Instead of in-place assignment, an ``assign`` interface object (similar to the ``Frame.astype`` interface shown above) is provided to expose ``__getitem__``, ``loc``, and ``iloc`` interfaces that, when called with an argument, return a new object with the desired changes. These interfaces expose the full range of expressive assignment-like idioms found in Pandas and NumPy. Arguments can be single values, or ``Series`` and ``Frame`` objects, where assignment will align on the Index.

>>> frame_new = frame.assign.loc[4854, 'albumId'](200)
>>> frame_new.loc[4854, 'albumId']
200


This pattern of specialized interfaces is used throughout StaticFrame, such as with the ``Frame.mask`` and ``Frame.drop`` interfaces. For example, ``Frame.mask`` can be used to create a Boolean ``Frame`` that sets rows to True if their "id" is even:

>>> frame.mask.loc[frame['id'] % 2 == 0].head()
<Frame>
<Index> albumId id     title  url    thumbnailUrl <<U12>
<Index>
0       False   False  False  False  False
1       True    True   True   True   True
2       False   False  False  False  False
3       True    True   True   True   True
4       False   False  False  False  False
<int64> <bool>  <bool> <bool> <bool> <bool>



Or, using the ``Frame.drop`` interface, a new ``Frame`` can be created by dropping rows with even "id" values and dropping URL columns specified in a list:

>>> frame.drop.loc[frame['id'] % 2 == 0, ['thumbnailUrl', 'url']].head()
<Frame>
<Index> albumId id      title                <<U12>
<Index>
0       1       1       accusamus beatae ...
2       1       3       officia porro iur...
4       1       5       natus nisi omnis ...
6       1       7       officia delectus ...
8       1       9       qui eius qui aute...
<int64> <int64> <int64> <<U86>


Iteration of rows, columns, and elements, as well as function application on those values, is unified under a family of generator interfaces. These interfaces are distinguished by the form of the data iterated (``Series``, ``namedtuple``, or ``array``) and whether key-value pairs (e.g., ``Frame.iter_series_items()``) or just values (e.g., ``Frame.iter_series()``) are yielded. For example, we can iterate over each row of a ``Frame`` and yield a corresponding ``Series``:

>>> next(iter(frame.iter_series(axis=1)))
<Series: 0>
<Index>
albumId      1
id           1
title        accusamus beatae ...
url          https://via.place...
thumbnailUrl https://via.place...
<<U12>       <object>

Or we can iterate over rows as named tuples, applying a function that matches a substring of the "title" or returns None, then drop those None records:

>>> frame.iter_tuple(axis=1).apply(lambda r: r.title if 'voluptatem' in r.title else None).dropna().head()
<Series>
<Index>
19       assumenda volupta...
27       non neque eligend...
29       odio enim volupta...
31       ad enim dignissim...
40       in voluptatem dol...
<int64>  <object>


Element iteration and function application works the same way as for rows or columns (though without an ``axis`` argument). For example, here each URL is processed with the same string transformation function:

>>> frame[['thumbnailUrl', 'url']].iter_element().apply(lambda c: c.replace('https://', '')).iloc[-4:]
<Frame>
<Index> thumbnailUrl         url                  <<U12>
<Index>
4996    via.placeholder.c... via.placeholder.c...
4997    via.placeholder.c... via.placeholder.c...
4998    via.placeholder.c... via.placeholder.c...
4999    via.placeholder.c... via.placeholder.c...
<int64> <<U30>               <<U30>


Group-by functionality is exposed in a similar manner with ``Frame.iter_group_items()`` and ``Frame.iter_group()``.

>>> next(iter(frame.iter_group('albumId', axis=0))).shape
(50, 5)


Function application to a group ``Frame`` can be used to produce a ``Series`` indexed by the group label. For example, a ``Series``, indexed by "albumId", can be produced to show the number of unique titles found per album.

>>> frame.iter_group('albumId', axis=0).apply(lambda g: len(g['title'].unique()), dtype=np.int64).head()
<Series>
<Index: albumId>
1                50
2                50
3                50
4                50
5                50
<int64>          <int64>

.. note::

    For more information on Frame iterators and tools for function application, see `Frame: Iterator <https://static-frame.readthedocs.io/en/latest/api_detail/frame.html#frame-iterator>`_.

If performing calculations on a ``Frame`` that result in a ``Series`` with a compatible ``Index``, a grow-only ``FrameGO`` can be used to add ``Series`` as new columns. This limited form of mutation, i.e., only the addition of columns, provides a convenient compromise between mutability and immutability. (Underlying NumPy array data always remains immutable.)

A ``FrameGO`` can be efficiently created from a ``Frame``, as underling NumPy arrays do not have to be copied:

>>> frame_go = frame.to_frame_go()


We can obtain a track number within each album, assuming the records are sorted, by creating the following generator expression pipe-line. Using a ``Frame`` grouped by "albumId", ``zip`` together as pairs the ``Frame.index`` and a contiguous integer sequence via ``range()``; ``chain`` all of those iterables, and then pass the resulting generator to ``Series.from_items()``. (As much as possible, StaticFrame supports generators as arguments wherever an ordered sequence is expected.)

>>> from itertools import chain
>>> index_to_track = chain.from_iterable(zip(g.index, range(len(g))) for g in frame_go.iter_group('albumId'))
>>> frame_go['track'] = sf.Series.from_items(index_to_track, dtype=np.int64) + 1

>>> frame_go.iloc[45:55]
<FrameGO>
<IndexGO> albumId id      title                url                  thumbnailUrl         track   <<U12>
<Index>
45        1       46      quidem maiores in... https://via.place... https://via.place... 46
46        1       47      et soluta est        https://via.place... https://via.place... 47
47        1       48      ut esse id           https://via.place... https://via.place... 48
48        1       49      quasi quae est mo... https://via.place... https://via.place... 49
49        1       50      et inventore quae... https://via.place... https://via.place... 50
50        2       51      non sunt voluptat... https://via.place... https://via.place... 1
51        2       52      eveniet pariatur ... https://via.place... https://via.place... 2
52        2       53      soluta et harum a... https://via.place... https://via.place... 3
53        2       54      ut ex quibusdam d... https://via.place... https://via.place... 4
54        2       55      voluptatem conseq... https://via.place... https://via.place... 5
<int64>   <int64> <int64> <<U86>               <<U38>               <<U38>               <int64>


Unlike with Pandas, StaticFrame ``Index`` objects always enforce uniqueness (there is no "verify_integrity" option: integrity is never optional). Thus, an index can never be set from non-unique data:

>>> frame_go.set_index('albumId')
Traceback (most recent call last):
static_frame.core.exception.ErrorInitIndexNonUnique: Labels have 4900 non-unique values, including 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.


For a data set such as the one used in this example, a hierarchical index, by "albumId" and "track", is practical. StaticFrame implements hierarchical indices as ``IndexHierarchy`` objects. The ``Frame.set_index_hierarchy()`` method, given columns in a ``Frame``, can be used to create a hierarchical index:


>>> frame_h = frame_go.set_index_hierarchy(['albumId', 'track'], drop=True)
>>> frame_h.head()
<FrameGO>
<IndexGO>                                    id      title                url                  thumbnailUrl         <<U12>
<IndexHierarchy: ('albumId', 'tra...
1                                    1       1       accusamus beatae ... https://via.place... https://via.place...
1                                    2       2       reprehenderit est... https://via.place... https://via.place...
1                                    3       3       officia porro iur... https://via.place... https://via.place...
1                                    4       4       culpa odio esse r... https://via.place... https://via.place...
1                                    5       5       natus nisi omnis ... https://via.place... https://via.place...
<int64>                              <int64> <int64> <<U86>               <<U38>               <<U38>




Hierarchical indices permit specifying selectors, per axis, at each hierarchical level. To distinguish hierarchical levels from axis arguments in a ``loc`` expression, the ``HLoc`` wrapper, exposing a ``__getitem__`` interface, can be used. For example, we can select, from all albums, the second and fifth track, and then only the "title" and "url" columns.

>>> frame_h.loc[sf.HLoc[:, [2,5]], ['title', 'url']].head()
<FrameGO>
<IndexGO>                                    title                url                  <<U12>
<IndexHierarchy: ('albumId', 'tra...
1                                    2       reprehenderit est... https://via.place...
1                                    5       natus nisi omnis ... https://via.place...
2                                    2       eveniet pariatur ... https://via.place...
2                                    5       voluptatem conseq... https://via.place...
3                                    2       eaque iste corpor... https://via.place...
<int64>                              <int64> <<U86>               <<U38>



Just as a hierarchical selection can reside in a ``loc`` expression with an ``HLoc`` wrapper, an integer index selection can reside in a ``loc`` expression with an ``ILoc`` wrapper. For example, the previous row selection is combined with the selection of the last column:

>>> frame_h.loc[sf.HLoc[:, [2,5]], sf.ILoc[-1]].head()
<Series: thumbnailUrl>
<IndexHierarchy: ('albumId', 'tra...
1                                    2       https://via.place...
1                                    5       https://via.place...
2                                    2       https://via.place...
2                                    5       https://via.place...
3                                    2       https://via.place...
<int64>                              <int64> <<U38>



.. note::

    For more information on IndexHierarchy, see `Index Hierarchy <https://static-frame.readthedocs.io/en/latest/api_detail/index_hierarchy.html>`_.

While StaticFrame offers many of the features of Pandas and similar data structures, exporting directly to NumPy arrays (via the ``.values`` attribute) or to Pandas is supported for functionality not found in StaticFrame or compatibility with other libraries. For example, a ``Frame`` can export to a Pandas ``DataFrame`` with ``Frame.to_pandas()``.

>>> df = frame_go.to_pandas()
