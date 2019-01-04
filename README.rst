.. image:: https://travis-ci.org/InvestmentSystems/static-frame.svg?branch=master
    :target: https://travis-ci.org/InvestmentSystems/static-frame

.. image:: https://badge.fury.io/py/static-frame.svg
    :target: https://badge.fury.io/py/static-frame

static-frame
=============

The StaticFrame library consists of the Series and Frame, immutable data structures for one- and two-dimensional calculations with self-aligning, labelled axis. StaticFrame offers an alternative to Pandas. While many interfaces for data extraction and manipulation are similar to Pandas, StaticFrame deviates from Pandas in many ways: all data is immutable, and all indices must be unique; all vector processing uses NumPy, and the full range of NumPy data types is preserved; the implementation is concise and lightweight; consistent naming and interfaces are used throughout; and flexible approaches to iteration and function application, with built-in options for parallelization, are provided.

Code: https://github.com/InvestmentSystems/static-frame

Docs: http://static-frame.readthedocs.io

Packages: https://pypi.org/project/static-frame


Installation
-------------

Install StaticFrame via PIP::

    pip install static-frame


Dependencies
--------------

StaticFrame requires Python 3.5+ and NumPy 1.14.1+.


Quick-Start Guide
---------------------

StaticFrame provides numerous methods for reading in and creating data, either as a 1D ``Series`` or a 2D ``Frame``. All creation routines are exposed as alternate constructors on the desired class, such as ``Frame.from_csv()`` or ``Frame.from_records()``. For example, we can load JSON data from a URL using ``Frame.from_json_url()``, and then use ``Frame.head()`` to reduce the displayed output to just the first five rows.

>>> import static_frame as sf
>>> frame = sf.Frame.from_json_url('https://jsonplaceholder.typicode.com/photos')
>>> frame.head()
<Frame>
<Index> albumId id      thumbnailUrl         title                url                  <<U12>
<Index>
0       1       1       https://via.place... accusamus beatae ... https://via.place...
1       1       2       https://via.place... reprehenderit est... https://via.place...
2       1       3       https://via.place... officia porro iur... https://via.place...
3       1       4       https://via.place... culpa odio esse r... https://via.place...
4       1       5       https://via.place... natus nisi omnis ... https://via.place...
<int64> <int64> <int64> <<U38>               <<U86>               <<U38>


As with a NumPy array, the ``Frame`` exposes common attributes of shape and size.

>>> frame.shape
(5000, 5)
>>> frame.size
25000
>>> frame.nbytes
3320000


Unlike a NumPy array, a Frame stores heterogenous types per column. StaticFrame preserves the full range of NumPy types, including fixed-size character strings. Of course, character strings can be converted to Python objects or other types as needed with ``Frame.astype()``:

>>> frame.dtypes
<Index>      <Series>
albumId      int64
id           int64
thumbnailUrl <U38
title        <U86
url          <U38
<<U12>       <object>

>>> frame.astype['thumbnailUrl':](object).dtypes
<Index>      <Series>
albumId      int64
id           int64
thumbnailUrl object
title        object
url          object
<<U12>       <object>


StaticFrame interfaces for extracting data will be familiar to Pandas users, though with a number of refinements to remove redundancies and increase consistency. On a ``Frame``, ``__getitem__`` is (exclusively) a column selector; ``loc`` and ``iloc`` are (with one argument) row selectors or (with two arguments) row and column selectors. For example:

>>> frame['albumId'].tail()
<Index> <Series>
4995    100
4996    100
4997    100
4998    100
4999    100
<int64> <int64>

>>> frame['id':'title'].head()
<Frame>
<Index> id      thumbnailUrl         title                <<U12>
<Index>
0       1       https://via.place... accusamus beatae ...
1       2       https://via.place... reprehenderit est...
2       3       https://via.place... officia porro iur...
3       4       https://via.place... culpa odio esse r...
4       5       https://via.place... natus nisi omnis ...
<int64> <int64> <<U38>               <<U86>

>>> frame.loc[4]
<Index>      <Series>
albumId      1
id           5
thumbnailUrl https://via.place...
title        natus nisi omnis ...
url          https://via.place...
<<U12>       <object>

>>> frame.loc[4:8, ['albumId', 'title']]
<Frame>
<Index> albumId title                <<U12>
<Index>
4       1       natus nisi omnis ...
5       1       accusamus ea aliq...
6       1       officia delectus ...
7       1       aut porro officii...
<int64> <int64> <<U86>


Just as with Pandas, expressions can be used in ``__getitem__``, ``loc``, and ``iloc`` statements to create more narrow selections.

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
TypeError: 'GetItem' object does not support item assignment
>>> frame.values[4854, 0] = 200
Traceback (most recent call last):
ValueError: assignment destination is read-only


Instead of in-place assignment, an ``assign`` interface object is provided to expose ``__getitem__``, ``loc``, and ``iloc`` interfaces that, when called with an argument, return a new object with the desired changes. These interfaces expose the full range of expressive assignment-like idioms found in Pandas and NumPy. Arguments can be single values, or ``Series`` and ``Frame`` objects, where assignment will align on the Index.

>>> frame_new = frame.assign.loc[4854, 'albumId'](200)
>>> frame_new.loc[4854, 'albumId']
200


This pattern of specialized interfaces is used throughout StaticFrame, such as with the ``Frame.mask`` and ``Frame.drop`` interfaces.

>>> frame.mask.loc[frame['id'] % 2 == 0].head()
<Frame>
<Index> albumId id     thumbnailUrl title  url    <<U12>
<Index>
0       False   False  False        False  False
1       True    True   True         True   True
2       False   False  False        False  False
3       True    True   True         True   True
4       False   False  False        False  False
<int64> <bool>  <bool> <bool>       <bool> <bool>

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


Iteration of rows, columns, and elements, as well as function application on those values, is unified under a family of generator interfaces. These interfaces are distinguished by the form of the data iterated (``Series``, ``namedtuple``, or ``array``) and whether key-value pairs (e.g., ``Frame.iter_series_items()``) or just values (e.g., ``Frame.iter_array()``) are yielded.

>>> next(iter(frame.iter_series(axis=1)))
<Index>      <Series>
albumId      1
id           1
thumbnailUrl https://via.place...
title        accusamus beatae ...
url          https://via.place...
<<U12>       <object>

>>> frame.iter_tuple(axis=1).apply(lambda r: r.title if 'voluptatem' in r.title else None).dropna().head()
<Index> <Series>
19      assumenda volupta...
27      non neque eligend...
29      odio enim volupta...
31      ad enim dignissim...
40      in voluptatem dol...
<int64> <object>


Group-by functionality is exposed in a similar manner with ``Frame.iter_group_items()`` and ``Frame.iter_group()``.

>>> next(iter(frame.iter_group('albumId', axis=0))).shape
(50, 5)

>>> frame.iter_group('albumId', axis=0).apply(lambda g: len(g['title'].unique())).head()
<Index> <Series>
1       50
2       50
3       50
4       50
5       50
<int64> <int64>


Unlike with Pandas, StaticFrame `Index` objects always enforce uniqueness (there is no "verify_integrity" option: integrity is never optional). Thus, an index can never be set from non-unique data:

>>> frame.set_index('albumId')
Traceback (most recent call last):
KeyError: 'labels have non-unique values'


.. TODO: need to refine hierarchical indices as this test case gave errirs
.. StaticFrame's implementation of hierarchical indices deviates from Pandas' in many ways, but provides similar functionality
.. This does not work!
.. frame_h = frame.set_index_hierarchy(['albumId', 'id'], drop=True)


Utility functions common to Pandas users are available on ``Frame`` and ``Series``, such as ``Series.unqiue()``, ``Series.isna()``, and ``Series.any()``.

>>> frame['albumId'].unique()
array([  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,
        27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
        40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,
        53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
        66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,
        79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,
        92,  93,  94,  95,  96,  97,  98,  99, 100])
>>> frame['id'].isna().any()
False


If performing calculations on the ``Frame`` that result in a Series with a compatible ``Index``, a grow-only ``FrameGO`` can be used to add columns. This limited form of mutation, i.e., only the addition of columns, provides a convenient compromise between mutability and immutability. (Underlying NumPy array data remains immutable.)

>>> frame_go = frame.to_frame_go()
>>> tracks = frame.iter_group('albumId', axis=0).apply(lambda g: len(g))
>>> frame_go['tracks'] = frame['albumId'].iter_element().apply(tracks)


Finally, if functionality of Pandas is needed, StaticFrame can export a Pandas ``DataFrame`` from a ``Frame``.

>>> df = frame_go.to_pandas()




