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


.. image:: https://img.shields.io/github/actions/workflow/status/static-frame/static-frame/test.yml?branch=master&label=test&logo=Github
  :target: https://github.com/static-frame/static-frame/actions/workflows/test.yml

.. image:: https://img.shields.io/github/actions/workflow/status/static-frame/static-frame/test_forward.yml?branch=master&label=test-forward&logo=Github
  :target: https://github.com/static-frame/static-frame/actions/workflows/test_forward.yml

.. image:: https://img.shields.io/github/actions/workflow/status/static-frame/static-frame/test_backward.yml?branch=master&label=test-backward&logo=Github
  :target: https://github.com/static-frame/static-frame/actions/workflows/test_backward.yml

.. image:: https://img.shields.io/github/actions/workflow/status/static-frame/static-frame/quality.yml?branch=master&label=quality&logo=Github
  :target: https://github.com/static-frame/static-frame/actions/workflows/quality.yml


.. image:: https://img.shields.io/readthedocs/static-frame.svg
  :target: https://static-frame.readthedocs.io/en/latest

.. image:: https://img.shields.io/badge/hypothesis-tested-brightgreen.svg
  :target: https://hypothesis.readthedocs.io

.. image:: https://img.shields.io/pypi/status/static-frame.svg
  :target: https://pypi.org/project/static-frame

.. image:: https://img.shields.io/badge/benchmarked%20by-asv-blue.svg
  :target: https://static-frame.github.io/static-frame-benchmark


.. image:: https://img.shields.io/badge/launch-binder-579ACA.svg?logo=data:image/png;base64,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
   :target: https://mybinder.org/v2/gh/static-frame/static-frame-ftgu/default?urlpath=tree/index.ipynb



static-frame
=============

A library of immutable and grow-only Pandas-like DataFrames with a more explicit and consistent interface. StaticFrame is suitable for applications in data science, data engineering, finance, scientific computing, and related fields where reducing opportunities for error by prohibiting in-place mutation is critical.

While many interfaces are similar to Pandas, StaticFrame deviates from Pandas in many ways: all data is immutable, and all indices are unique; the full range of NumPy data types is preserved, and date-time indices use discrete NumPy types; hierarchical indices are seamlessly integrated; and uniform approaches to element, row, and column iteration and function application are provided. Core StaticFrame depends only on NumPy and two C-extension packages (maintained by the StaticFrame team): Pandas is not a dependency.

A wide variety of table formats are supported, including input from and output to CSV, TSV, JSON, MessagePack, Excel XLSX, SQLite, HDF5, NumPy, Pandas, Arrow, and Parquet; additionally, output to xarray, VisiData, HTML, RST, Markdown, and LaTeX is supported, as well as HTML representations in Jupyter notebooks. Full serialization is also available via custom NPZ and NPY encodings, the latter supporting memory mapping.

StaticFrame features a family of multi-table containers: the Bus is a lazily-loaded container of tables, the Batch is a deferred processor of tables, the Yarn is virtual concatenation of many Buses, and the Quilt is a virtual concatenation of all tables within a single Bus or Yarn. All permit operating on large collections of tables with minimal memory overhead, as well as writing to and reading from zipped bundles of pickles, NPZ, Parquet, or delimited files, as well as XLSX workbooks, SQLite, and HDF5.


Code: https://github.com/static-frame/static-frame

Docs: http://static-frame.readthedocs.io

Packages: https://pypi.org/project/static-frame

Jupyter Notebook Tutorial: `Launch Binder <https://mybinder.org/v2/gh/static-frame/static-frame-ftgu/default?urlpath=tree/index.ipynb>`_

Context: `Ten Reasons to Use StaticFrame instead of Pandas <https://dev.to/flexatone/ten-reasons-to-use-staticframe-instead-of-pandas-4aad>`_


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

- Python>=3.7
- NumPy>=1.18.5
- automap==0.6.2
- arraykit==0.2.6

For extended input and output, the following packages are required:

- pandas>=0.24.2
- xlsxwriter>=1.1.2
- openpyxl>=3.0.9
- xarray>=0.13.0
- tables>=3.6.1
- pyarrow>=0.17.0
- visidata>=2.4


Quick-Start Guide
---------------------

To get startred quickly, lets download the classic iris (flower) characteristics data set and build a simple naive Bayes classifier that can predict species from iris petal characteristics.

While StaticFrame's API has over 7,000 endpoints, much will be familiar to users of Pandas or other DataFrame libraries. Rather than offering fewer interfaces with greater configurability, StaticFrame favors more numerous interfaces with more narrow parameters and functionality. This design makes for more maintainable code.

Lets get the data set from the UCI Machine Learning Repository and create a ``Frame``. StaticFrame exposes all constructors on the ``Frame`` or derived class. Here, we will use the ``from_csv()`` constructor. To download a resource and provide it to a constructor, we can use StaticFrame's ``WWW.from_file()`` interface.

>>> import static_frame as sf
>>> data = sf.Frame.from_csv(sf.WWW.from_file('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'), columns_depth=0)


We can use ``head()`` to view the first rows. Notice that StaticFrame's default display make it very clear what type of Frame, Index, and NumPy datatypes are present.

>>> data.head()
<Frame>
<Index> 0         1         2         3         4           <int64>
<Index>
0       5.1       3.5       1.4       0.2       Iris-setosa
1       4.9       3.0       1.4       0.2       Iris-setosa
2       4.7       3.2       1.3       0.2       Iris-setosa
3       4.6       3.1       1.5       0.2       Iris-setosa
4       5.0       3.6       1.4       0.2       Iris-setosa
<int64> <float64> <float64> <float64> <float64> <<U15>


StaticFrame supports reindexing (conforming existing axis labels to new labels, potentially changing the size and ordering) and relabeling (simply applying new labels without regard to existing labels, never changing size or ordering). To set new column labels, we will use the ``relabel()`` method. While we are creating a new ``Frame``, relabeling does not require us to copy the underlying NumPy data. As all data is immutable, we can reuse it in our new container.

>>> data = data.relabel(columns=('sepal_l', 'sepal_w', 'petal_l', 'petal_w', 'species'))
>>> data.head()
<Frame>
<Index> sepal_l   sepal_w   petal_l   petal_w   species     <<U7>
<Index>
0       5.1       3.5       1.4       0.2       Iris-setosa
1       4.9       3.0       1.4       0.2       Iris-setosa
2       4.7       3.2       1.3       0.2       Iris-setosa
3       4.6       3.1       1.5       0.2       Iris-setosa
4       5.0       3.6       1.4       0.2       Iris-setosa
<int64> <float64> <float64> <float64> <float64> <<U15>


We are going to use 80% of our data to train our classifier; the remaining 20% will be used to test the classifier. To divide our data into two groups, we will create a ``Series`` of contiguous integers and then extract a random selection of 80% of the values. The ``sample()`` method, given a count, samples that many values from the ``Series``. We can then use the ``drop`` interface to create a new ``Series`` that excludes the training group, leaving the testing group.

>>> sel = sf.Series(np.arange(len(data)))
>>> sel_train = sel.sample(round(len(data) * .8))
>>> sel_test = sel.drop[sel_train]
>>> sel_test.head()
<Series>
<Index>
8        8
15       15
18       18
23       23
26       26
<int64>  <int64>

Next, we will use ``loc`` on the ``Frame`` to select the training subset of the data.

>>> data_train = data.loc[sel_train]
>>> data_train.head()
<Frame>
<Index> sepal_l   sepal_w   petal_l   petal_w   species     <<U7>
<Index>
0       5.1       3.5       1.4       0.2       Iris-setosa
1       4.9       3.0       1.4       0.2       Iris-setosa
2       4.7       3.2       1.3       0.2       Iris-setosa
3       4.6       3.1       1.5       0.2       Iris-setosa
4       5.0       3.6       1.4       0.2       Iris-setosa
<int64> <float64> <float64> <float64> <float64> <<U15>

To get a ``Series`` of counts per species, I can select the species column and iterate over groups, based on the species name, and count the size of each group. In StaticFrame, this can be done by calling ``iter_group_items()`` to get an iterator of pairs of group label, group ``Series``.This iterator can be given to a ``Batch``, a chaining processor of ``Frame`` operations; once the ``Batch`` is created, we can call multiple methods on it. A container is only returned when a finalizer method is called such as ``to_series()``.


>>> counts = sf.Batch(data_train['species'].iter_group_items()).count().to_series()
>>> counts
<Series>
<Index>
Iris-setosa     41
Iris-versicolor 40
Iris-virginica  39
<<U15>          <int64>


We can calculate the "prior" by dividing counts by the size of the training data.

>>> prior = counts / len(data_train)
>>> prior
<Series>
<Index>
Iris-setosa     0.3416666666666667
Iris-versicolor 0.3333333333333333
Iris-virginica  0.325
<<U15>          <float64>



>>> mu = sf.Batch(data_train[['sepal_l', 'sepal_w', 'species']].iter_group_items('species', drop=True)).mean().to_frame()
>>> mu
<Frame>
<Index>         sepal_l           sepal_w            <<U7>
<Index>
Iris-setosa     5.021951219512196 3.426829268292683
Iris-versicolor 5.9               2.7924999999999995
Iris-virginica  6.587179487179487 2.9794871794871796
<<U15>          <float64>         <float64>

>>> sigma = sf.Batch(data_train[['sepal_l', 'sepal_w', 'species']].iter_group_items('species', drop=True)).std(ddof=1).to_frame()
>>> sigma
<Frame>
<Index>         sepal_l             sepal_w             <<U7>
<Index>
Iris-setosa     0.3588259990036614  0.3847235307619632
Iris-versicolor 0.48145079893471127 0.30330445058322136
Iris-virginica  0.6346070011305742  0.34654648596771576
<<U15>          <float64>           <float64>





>>> stats = sf.Frame.from_concat((mu.relabel_level_add('mu'), sigma.relabel_level_add('sigma')))
>>> round(stats, 2)
<Frame>
<Index>                          sepal_l   sepal_w   <<U7>
<IndexHierarchy>
mu               Iris-setosa     5.02      3.43
mu               Iris-versicolor 5.9       2.79
mu               Iris-virginica  6.59      2.98
sigma            Iris-setosa     0.36      0.38
sigma            Iris-versicolor 0.48      0.3
sigma            Iris-virginica  0.63      0.35
<<U5>            <<U15>          <float64> <float64>


>>> data_test = data.loc[sel_test.values, ['sepal_l', 'sepal_w']].to_frame_go()
>>> from scipy.stats import norm
>>> def fields():
...     for label in mu.index:
...             pdf = norm.pdf(data_test.values, mu.loc[label], sigma.loc[label])
...             yield np.log(pdf).sum(axis=1)


>>> likelihood = sf.Frame.from_fields(fields(), columns=mu.index, index=data_test.index)
>>> round(likelihood.head(), 2)
<Frame>
<Index> Iris-setosa Iris-versicolor Iris-virginica <<U15>
<Index>
1       -0.53       -2.31           -3.86
6       -0.55       -5.57           -5.96
12      -0.66       -2.76           -4.29
13      -2.5        -5.67           -6.82
17      0.1         -4.02           -4.2

>>> posterior = likelihood * prior
>>> data_test['predict'] = posterior.loc_max(axis=1)
>>> data_test['observed'] = data['species']
>>> data_test['correct'] = data_test['predict'] == data_test['observed']

>>> data_test.tail()
<FrameGO>
<IndexGO> sepal_l   sepal_w   predict         observed       correct <<U8>
<Index>
136       6.3       3.4       Iris-virginica  Iris-virginica True
137       6.4       3.1       Iris-virginica  Iris-virginica True
139       6.9       3.1       Iris-virginica  Iris-virginica True
146       6.3       2.5       Iris-versicolor Iris-virginica False
147       6.5       3.0       Iris-virginica  Iris-virginica True
<int64>   <float64> <float64> <<U15>          <<U15>         <bool>


>>> data_test["correct"].sum(), len(data_test)
(22, 30)


.. note::

    For a concise overview of all StaticFrame interfaces, see `API Overview <https://static-frame.readthedocs.io/en/latest/api_overview>`_.


.. note::

    For more information on Frame constructors, see `Frame: Constructor <https://static-frame.readthedocs.io/en/latest/api_detail/frame.html#frame-constructor>`_.


.. note::

    For more information on Frame utility functions, see `Frame: Method <https://static-frame.readthedocs.io/en/latest/api_detail/frame.html#frame-method>`_.


.. note::

    For more information on Frame selection interfaces, see `Frame: Selector <https://static-frame.readthedocs.io/en/latest/api_detail/frame.html#frame-selector>`_.


.. note::

    For more information on Frame iterators and tools for function application, see `Frame: Iterator <https://static-frame.readthedocs.io/en/latest/api_detail/frame.html#frame-iterator>`_.


.. note::

    For more information on IndexHierarchy, see `Index Hierarchy <https://static-frame.readthedocs.io/en/latest/api_detail/index_hierarchy.html>`_.

