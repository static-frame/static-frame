


Ten Tips for Transitioning from Pandas to StaticFrame
=============================================================

For those coming from Pandas, the manner in which StaticFrame offers a more consistent interface and reduces opportunities for error might be surprising. This article provides ten tips to get up to speed with StaticFrame.


Why StaticFrame
______________________

After years of using Pandas to develop production, back-end financial systems, it became clear to me that Pandas was not the right tool. Pandas's handling of labeled data and missing values, with performance close to NumPy, certainly accelerated my productivity. And yet, the numerous inconsistencies in Pandas's API led to hard to maintain code. Further, Pandas inconsistent approach to data ownership, and support for mutation and undesirable side effects, led to serious vulnerabilities and opportunities for error. So in May of 2017 I began implementing a library more suitable for critical production systems.

Now, after nearly three years of development and refinement, we are seeing excellent results in our production systems by replacing Pandas with StaticFrame.

What follows are ten tips to aid Pandas users in getting started with StaticFrame. While Pandas users will find many familiar idioms, there are significant differences to improve maintainability and reduce opportunities for error.


No. 1: Consistent and Discoverable Interfaces
____________________________________________________


An API can be consistent in where functions are located, how functions are named, and the name and types of arguments those functions accept. StaticFrame deviates from Pandas API to support greater consistency in all of these areas.

The desire for consistency is what leads StaticFrame to name its two primary containers ``Series`` and ``Frame``. The other consistent option would be ``DataSeries`` and ``DataFrame``, but since it is obvious these containers hold data, the "data" prefix was dropped.

To create ``Series`` and ``Frame``, you need constructors. Pandas places its ``pd.DataFrame`` constructors in at least two places: on the root namespace (``pd``, as commonly imported) and on the ``pd.DataFrame`` class.

For example, JSON data is loaded from a function on the ``pd`` namespace, while records are loaded from the ``DataFrame`` class.


>>> pd.read_json('[{"name":"muon", "mass":0.106},{"name":"tau", "mass":1.777}]')
    mass  name
0  0.106  muon
1  1.777   tau

>>> pd.DataFrame.from_records([{"name":"muon", "mass":0.106}, {"name":"tau", "mass":1.777}])
    mass  name
0  0.106  muon
1  1.777   tau


For the user, there is no benefit to this diversity. StaticFrame places all constructors on the class they construct. For example, ``from_json`` and ``from_dict_records`` are available on the ``Frame`` class.


>>> sf.Frame.from_json('[{"name":"muon", "mass":0.106}, {"name":"tau", "mass":1.777}]')
<Frame>
<Index> name  mass      <<U4>
<Index>
0       muon  0.106
1       tau   1.777
<int64> <<U4> <float64>

>>> sf.Frame.from_dict_records([{"name":"muon", "mass":0.106}, {"name":"tau", "mass":1.777}])
<Frame>
<Index> name  mass      <<U4>
<Index>
0       muon  0.106
1       tau   1.777
<int64> <<U4> <float64>


StaticFrame has both a specialized ``from_dict_records`` constructor (explicitly for handling dictionary records, where keys might not align) as well as a ``from_records`` constructor (for sequence types that are all the same size). Such explicit, specialized constructors are common in StaticFrame, and are easier to maintain than constructors that take wildly diverse inputs with parameters that are only needed for some input types.

For example, while Pandas has specialized ``DataFrame`` constructors (such as ``pd.DataFrame.from_records``, the default ``DataFrame`` constructor accepts a staggering diversity of inputs, including the same inputs as ``pd.DataFrame.from_records``.


>>> pd.DataFrame.from_records([{"name":"muon", "mass":0.106}, {"name":"tau", "mass":1.777}])
    mass  name
0  0.106  muon
1  1.777   tau

>>> pd.DataFrame([{"name":"muon", "mass":0.106}, {"name":"tau", "mass":1.777}])
    mass  name
0  0.106  muon
1  1.777   tau

>>> pd.DataFrame([[0.106, "muon"], [1.777, "tau"]], columns=('mass', 'name'))
    mass  name
0  0.106  muon
1  1.777   tau


Having multiple ways to do the same thing is undesirable. StaticFrame enforces the usage of explicit constructors, creating code that is easier to maintain because function signatures are more narowly defined.

Being explicit leads to lots of constructors. To help find what you are looking for, StaticFrame containers expose an ``interface`` attribute that lists the entire public interface of the class or instance.

>>> sf.Frame.interface.shape
(388, 3)

We can filter this table just to the constructors by using a ``loc`` selection.


>>> sf.Frame.interface.loc[sf.Frame.interface['group'] == 'Constructor']
<Frame: Frame>
<Index>                   cls   group       doc                  <<U5>
<Index: name>
__init__()                Frame Constructor
from_arrow()              Frame Constructor Convert an Arrow ...
from_concat()             Frame Constructor Concatenate multi...
from_concat_items()       Frame Constructor Produce a Frame w...
from_csv()                Frame Constructor Specialized versi...
from_delimited()          Frame Constructor Create a Frame fr...
from_dict()               Frame Constructor Create a Frame fr...
from_dict_records()       Frame Constructor Frame constructor...
from_dict_records_items() Frame Constructor Frame constructor...
from_element()            Frame Constructor Create a Frame fr...
from_element_iloc_items() Frame Constructor Given an iterable...
from_element_loc_items()  Frame Constructor This function is ...
from_elements()           Frame Constructor Create a Frame fr...
from_hdf5()               Frame Constructor Load Frame from t...
from_items()              Frame Constructor Frame constructor...
from_json()               Frame Constructor Frame constructor...
from_json_url()           Frame Constructor Frame constructor...
from_pandas()             Frame Constructor Given a Pandas Da...
from_parquet()            Frame Constructor Realize a Frame f...
from_records()            Frame Constructor Frame constructor...
from_records_items()      Frame Constructor Frame constructor...
from_series()             Frame Constructor Frame constructor...
from_sql()                Frame Constructor Frame constructor...
from_sqlite()             Frame Constructor Load Frame from t...
from_structured_array()   Frame Constructor Convert a NumPy s...
from_tsv()                Frame Constructor Specialized versi...
from_xlsx()               Frame Constructor Load Frame from t...
<<U51>                    <<U5> <<U15>      <<U53>




No. 2: Consistent and Colorful Display
___________________________________________


Pandas displays its containers in diverse, inconsistent ways. For example, a ``pd.Series`` is shown with its name and type, while a ``pd.DataFrame`` does not show either of those attributes. If you display a ``pd.Index`` or ``pd.MultiIndex``, you get a ``eval``-able string, but one that is unmanageable if large.

>>> df = pd.DataFrame.from_records([{'symbol':'c', 'mass':1.3}, {'symbol':'s', 'mass':0.1}], index=('charm', 'strange'))
>>> df
         mass symbol
charm     1.3      c
strange   0.1      s

>>> df['mass']
charm      1.3
strange    0.1
Name: mass, dtype: float64

>>> df.index
Index(['charm', 'strange'], dtype='object')


StaticFrame offers a consistent, configurable display for all conntainers. The display of ``Series``, ``Frame``, and ``Index`` share a common design. Under the hood, the display components are modular and reusable: the display of an ``IndexHierarchy`` is used to build the display of a ``Frame``.


>>> f = sf.Frame.from_dict_records_items((('charm', {'symbol':'c', 'mass':1.3}), ('strange', {'symbol':'s', 'mass':0.1})))
>>> f
<Frame>
<Index> symbol mass      <<U6>
<Index>
charm   c      1.3
strange s      0.1
<<U7>   <<U1>  <float64>

>>> f['mass']
<Series: mass>
<Index>
charm          1.3
strange        0.1
<<U7>          <float64>


As much time is spent looking at the contents of ``Frame`` and ``Series``, StaticFrame offers numerous configuration options for displaying containers, all exposed throught the ``DisplayConfig`` class. Specific types can be colored, type annotations can be removed entirely, and there are many other options.


>>> f.display(sf.DisplayConfig(type_color_str='lime', type_color_float='orange'))
<Frame>
<Index> symbol mass      <<U6>
<Index>
charm   c      1.3
strange s      0.1
<<U7>   <<U1>  <float64>



>>> f.display(sf.DisplayConfig(type_show=False))
        symbol mass
charm   c      1.3
strange s      0.1




No. 3: Immutable Data: Better Memory Management, No Defensive Copies
___________________________________________________________________________________


Pandas displays inconsistent behavior in regard to ownership of data inputs: sometimes we can mutate NumPy arrays "behind-the-back" of Pandas.

For example, if we give a 2D array as an input to a ``DataFrame``, the lingering reference of the array can be used to "remotely" change the values of ``DataFrame``. Counter-intuitively, the ``DataFrame`` is not protecting access to its data, serving simply as a wrapper of the shared array.

>>> a1 = np.array([[0.106, -1], [1.777, -1]])
>>> df = pd.DataFrame(a1, index=('muon', 'tau'), columns=('mass', 'charge'))
>>> df
       mass  charge
muon  0.106    -1.0
tau   1.777    -1.0

>>> a1[0, 0] = np.nan

>>> df
       mass  charge
muon    NaN    -1.0
tau   1.777    -1.0



There are other, similar cases. Sometimes (but not always), the arrays given from the ``values`` attribute of ``Series`` and ``DataFrame`` can be mutated, changing the values of the ``DataFrame`` from which they were extracted.


>>> a2 = df['charge'].values
>>> a2
array([-1., -1.])

>>> a2[1] = np.nan

>>> df
       mass  charge
muon    NaN    -1.0
tau   1.777     NaN



With StaticFrame, the inconsistency and vulnerability of "behind the back" mutation is never permitted, either from StaticFrame containers or from direct access to underlying arrays.


>>> f = sf.Frame.from_dict_records_items((('charm', {'symbol':'c', 'mass':1.3}), ('strange', {'symbol':'s', 'mass':0.1})))


>>> f.loc['charm', 'mass'] = np.nan
Traceback (most recent call last):
  File "<console>", line 1, in <module>
TypeError: 'InterfaceGetItem' object does not support item assignment

>>> f['mass'].values[1] = 100
Traceback (most recent call last):
  File "<console>", line 1, in <module>
ValueError: assignment destination is read-only


While immutable data reduces opportunities for error, it also offers performance advantages. For exmaple, when creating a new ``Frame`` when renaming or relabeling, underlying data is not copied. Such operations are thus fast and light-weight.

>>> f.rename('fermion')
<Frame: fermion>
<Index>          symbol mass      <<U6>
<Index>
charm            c      1.3
strange          s      0.1
<<U7>            <<U1>  <float64>



Similarly, some types of concatenation (horizontal, axis 1 concatenation on aligned indices) can be done without copying data. For example, concatenating a ``Series`` to this ``Frame`` does not require copying underlying data to the new ``Frame``.


>>> s = sf.Series.from_dict(dict(charm=0.666, strange=-0.333), name='charge')

>>> sf.Frame.from_concat((f, s), axis=1)
<Frame>
<Index> symbol mass      charge    <<U6>
<Index>
charm   c      1.3       0.666
strange s      0.1       -0.333
<<U7>   <<U1>  <float64> <float64>





No. 4: Assignment is a Function; Assignment Preserves Types
_____________________________________________________________


While Pandas permits arbitrary assignment, those assignments happen in-place, making getting the right derived type (when needed) difficult, and resulting in some undesirable bahavior. For example, a float assigned into an integer-typed `pd.Series` will simply have its floating-point components truncated.

>>> s = pd.Series((-1, -1), index=('tau', 'down'))
>>> s
tau    -1
down   -1
dtype: int64
>>> s['down'] = -0.333
>>> s
tau    -1
down    0
dtype: int64


With StaticFrame, assignment is a function that returns a new container. This permits evaluating the types to insure that the resultant array can completely contain the assigned value.


>>> s = sf.Series((-1, -1), index=('tau', 'down'))
>>> s
<Series>
<Index>
tau      -1
down     -1
<<U4>    <int64>

>>> s.assign['down'](-0.333)
<Series>
<Index>
tau      -1.0
down     -0.333
<<U4>    <float64>



Assignment on a ``Frame`` is similar. Further, as a data structure that contains heterogeneous types of columnar data, assignment on a ``Frame`` only mutates what needs to change, reusing columns without copying data.

For example, assigning to a single value in a ``Frame`` results in only one new array being created; the unmodified array is reused in the new ``Frame`` without copying data.


>>> f = sf.Frame.from_dict_records_items((('charm', {'charge':0.666, 'mass':1.3}), ('strange', {'charge':-0.333, 'mass':0.1})))

>>> f
<Frame>
<Index> charge    mass      <<U6>
<Index>
charm   0.666     1.3
strange -0.333    0.1
<<U7>   <float64> <float64>


>>> f.loc['charm', 'charge']
0.666

>>> f.assign.loc['charm', 'charge'](Fraction(2, 3))
<Frame>
<Index> charge   mass      <<U6>
<Index>
charm   2/3      1.3
strange -0.333   0.1
<<U7>   <object> <float64>




No. 5: Iterators are for Iterating and Function Application
________________________________________________________________


Pandas has separate functions for iterating and function application, even though function application requires iteration. For example, Pandas has ``DataFrame.iteritems``, ``DataFrame.iterrows``, ``DataFrame.itertuples``, ``DataFrame.groupby`` for iteration, and ``DataFrame.apply`` and ``DataFrame.applymap`` for function application.


StaticFrame avoids this redundancy and confusion by exposing, on all iterators (such as ``Frame.iter_array`` or ``Frame.iter_group_items``), an ``apply`` method, as well as functions for using mapping types (such as ``map_any`` and ``map_fill``). This means that once you you find how you want to iterate, function application is a just a method away.

For an example, we will create a ``Frame`` with ``Frame.from_records`` and then set an index.


>>> f = sf.Frame.from_records((('muon', 0.106, -1.0, 'lepton'), ('tau', 1.777, -1.0, 'lepton'), ('charm', 1.3, 0.666, 'quark'), ('strange', 0.1, -0.333, 'quark')), columns=('name', 'mass', 'charge', 'type'))
>>> f = f.set_index('name', drop=True)
>>> f
<Frame>
<Index>       mass      charge    type   <<U6>
<Index: name>
muon          0.106     -1.0      lepton
tau           1.777     -1.0      lepton
charm         1.3       0.666     quark
strange       0.1       -0.333    quark
<<U7>         <float64> <float64> <<U6>


Next, we can demonstrate one of the many StaticFrame iterators. We will iterate over elements in a ``Series`` with ``iter_element()``.

>>> tuple(f['type'].iter_element())
('lepton', 'lepton', 'quark', 'quark')


We can then use the same iterator to do function application, simply by using the ``apply`` method.

>>> f['type'].iter_element().apply(lambda e: e.upper())
<Series>
<Index>
muon     LEPTON
tau      LEPTON
charm    QUARK
strange  QUARK
<<U7>    <<U6>


This same approach is used for all iterators on all containers. For example, we can use ``iter_element()`` on ``Frame`` to apply string formating to each element.

>>> f.iter_element().apply(lambda e: str(e).rjust(8, '_'))
<Frame>
<Index>       mass     charge   type     <<U6>
<Index: name>
muon          ___0.106 ____-1.0 __lepton
tau           ___1.777 ____-1.0 __lepton
charm         _____1.3 ___0.666 ___quark
strange       _____0.1 __-0.333 ___quark
<<U7>         <object> <object> <object>


A family of methods for row or column iteration allows the user to choose the type of those iterated rows or columns, i.e, as an array, as a ``NamedTuple``, or as a ``Series`` (``iter_array()``, ``iter_tuple()``, ``iter_series()``). All such methods take an axis argument to determine whether we iterate by row or by column.

For example, to apply a function to columns, we can do the following.

>>> f[['mass', 'charge']].iter_array(axis=0).apply(np.sum)
<Series>
<Index>
mass     3.283
charge   -1.667
<<U6>    <float64>


If our ``apply`` function needs to process both key and value pairs, we can use the corresponding iterator that returns items-style pairs.


>>> f.iter_array_items(axis=0).apply(lambda k, v: v.sum() if k != 'type' else np.nan)
<Series>
<Index>
mass     3.283
charge   -1.667
type     nan
<<U6>    <float64>


Applying a function to a row simply requires changing the axis argument.

>>> f.iter_series(axis=1).apply(lambda s: s['mass'] > 1 and s['type'] == 'quark')
<Series>
<Index>
muon     False
tau      False
charm    True
strange  False
<<U7>    <bool>


Group iteration and function application in StaticFrame works exactly the same way.

>>> f.iter_group('type').apply(lambda f: f['mass'].mean())
<Series>
<Index>
lepton   0.9415
quark    0.7000000000000001
<<U6>    <float64>
>>>




No. 6: Strict, Grow-Only Frames
_____________________________________________

A common use of ``pd.DataFrame`` is to load initial data, then produce derived data by adding additional columns. ``StaticFrame`` makes this approach less vulnerable to error by offering a strictly grow-only version of a ``Frame`` called a ``FrameGO``.

For example, once a ``FrameGO`` is created, new columns can be added while exisiting columns cannot be mutated or reordered.


>>> f = sf.FrameGO.from_records(((0.106, -1.0, 'lepton'), (1.777, -1.0, 'lepton'), (1.3, 0.666, 'quark'), (0.1, -0.333, 'quark')), columns=('mass', 'charge', 'type'), index=('muon', 'tau', 'charm', 'strange'))

>>> f['positive'] = f['charge'] > 0

>>> f
<FrameGO>
<IndexGO> mass      charge    type   positive <<U8>
<Index>
muon      0.106     -1.0      lepton False
tau       1.777     -1.0      lepton False
charm     1.3       0.666     quark  True
strange   0.1       -0.333    quark  False


This limited, retricted form of mutation meets a practical need. Converting back and forth from a ``Frame`` to a ``FrameGO`` is a light-weight, no-copy operation using ``Frame.to_frame_go()`` and ``FrameGO.to_frame()`` (underlying immutable arrays can be safely reused and shared between ``Frame`` and ``FrameGO`` instances.



No 7: Everything is not a Nanosecond
__________________________________________________________________

Pandas models every date or timestamp as a NumPy nanosecond ``datetime64`` object, regardless of if nanosecond resolution is needed or practical. This has the amusing side effect of creating a "Y2262 problem": not permitting dates beyond 2262-04-11.

For exmaple, while I can create a ``pd.DatetimeIndex`` up to 2262-04-11, one day further and Pandas raises an error.

>>> pd.date_range('1980', '2262-04-11')
DatetimeIndex(['1980-01-01', '1980-01-02', '1980-01-03', '1980-01-04',
               '1980-01-05', '1980-01-06', '1980-01-07', '1980-01-08',
               '1980-01-09', '1980-01-10',
               ...
               '2262-04-02', '2262-04-03', '2262-04-04', '2262-04-05',
               '2262-04-06', '2262-04-07', '2262-04-08', '2262-04-09',
               '2262-04-10', '2262-04-11'],
              dtype='datetime64[ns]', length=103100, freq='D')


>>> pd.date_range('1980', '2262-04-12')
Traceback (most recent call last):
  File "<console>", line 1, in <module>
  File "/home/ariza/.env37/lib/python3.7/site-packages/pandas/core/indexes/datetimes.py", line 2749, in date_range
    closed=closed, **kwargs)
  File "/home/ariza/.env37/lib/python3.7/site-packages/pandas/core/indexes/datetimes.py", line 381, in __new__
    ambiguous=ambiguous)
  File "/home/ariza/.env37/lib/python3.7/site-packages/pandas/core/indexes/datetimes.py", line 479, in _generate
    end = Timestamp(end)
  File "pandas/_libs/tslibs/timestamps.pyx", line 644, in pandas._libs.tslibs.timestamps.Timestamp.__new__
  File "pandas/_libs/tslibs/conversion.pyx", line 275, in pandas._libs.tslibs.conversion.convert_to_tsobject
  File "pandas/_libs/tslibs/conversion.pyx", line 470, in pandas._libs.tslibs.conversion.convert_str_to_tsobject
  File "pandas/_libs/tslibs/conversion.pyx", line 439, in pandas._libs.tslibs.conversion.convert_str_to_tsobject
  File "pandas/_libs/tslibs/np_datetime.pyx", line 121, in pandas._libs.tslibs.np_datetime.check_dts_bounds
pandas._libs.tslibs.np_datetime.OutOfBoundsDatetime: Out of bounds nanosecond timestamp: 2262-04-12 00:00:00



As date/time indices are often used for things much larger than nanoseconds, such as years and dates, StaticFrame offers fixed, diverse, typed ``datetime64`` indices. This permits more explicit usage, and avoids the "Y2262 problem".

For example, getting StaticFrame indices with years or dates up to the end of the year 3000 is not a problem.

>>> sf.IndexYear.from_year_range(1980, 3000).tail()
<IndexYear>
2996
2997
2998
2999
3000
<datetime64[Y]>

>>> sf.IndexDate.from_year_range(1980, 3000).tail()
<IndexDate>
3000-12-27
3000-12-28
3000-12-29
3000-12-30
3000-12-31
<datetime64[D]>


No. 8: Well-behaved Hierarchical Indices
___________________________________________


>>> df = pd.DataFrame.from_records([('muon', 0.106, -1.0, 'lepton'), ('tau', 1.777, -1.0, 'lepton'), ('charm', 1.3, 0.666, 'quark'), ('strange', 0.1, -0.333, 'quark')], columns=('name', 'mass', 'charge', 'type'))
>>> df.set_index(['type', 'name'], inplace=True)
>>> df
                 mass  charge
type   name
lepton muon     0.106  -1.000
       tau      1.777  -1.000
quark  charm    1.300   0.666
       strange  0.100  -0.333


Pandas sometimes reduces the `pd.MultiIndex` to a normal Index, sometimes does not.

>>> df.loc['quark']
         mass  charge
name
charm     1.3   0.666
strange   0.1  -0.333

>>> df.iloc[2:]
               mass  charge
type  name
quark charm     1.3   0.666
      strange   0.1  -0.333


Note also that, even after selection, the index object surprisingly retains information from the original ``IndexMulti``.

>>> df.iloc[2:].index
MultiIndex(levels=[['lepton', 'quark'], ['charm', 'muon', 'strange', 'tau']],
           labels=[[1, 1], [0, 2]],
           names=['type', 'name'])





With an ``pd.IndexMulti``, Pandas sometimes uses the second argument in a `loc` selection to refer to the columns.

>>> df.loc['lepton', 'mass']
name
muon    0.106
tau     1.777
Name: mass, dtype: float64


But other times uses the second argument in a `loc` selection to refer to inner levels of the ``MultiIndex``.


>>> df.loc['lepton', 'tau']
mass      1.777
charge   -1.000
Name: (lepton, tau), dtype: float64





StaticFrame's ``IndexHierarchy`` are built for ``Index`` objects and offer more consistent behavior.



>>> f = sf.Frame.from_records((('muon', 0.106, -1.0, 'lepton'), ('tau', 1.777, -1.0, 'lepton'), ('charm', 1.3, 0.666, 'quark'), ('strange', 0.1, -0.333, 'quark')), columns=('name', 'mass', 'charge', 'type'))


>>> f = f.set_index_hierarchy(('type', 'name'), drop=True)
<Frame>
<Index>                                    mass      charge    <<U6>
<IndexHierarchy: ('type', 'name')>
lepton                             muon    0.106     -1.0
lepton                             tau     1.777     -1.0
quark                              charm   1.3       0.666
quark                              strange 0.1       -0.333
<<U7>                              <<U7>   <float64> <float64>





A selection never automatically reduces the ``IndexHierarchy`` to an ``Index``. If reduction is needed, the ``Frame.relabel_drop_level()`` can be used (without copying underlying data).


>>> f.loc[sf.HLoc['quark']]
<Frame>
<Index>                                    mass      charge    <<U6>
<IndexHierarchy: ('type', 'name')>
quark                              charm   1.3       0.666
quark                              strange 0.1       -0.333
<<U7>                              <<U7>   <float64> <float64>

>>> f.iloc[2:]
<Frame>
<Index>                                    mass      charge    <<U6>
<IndexHierarchy: ('type', 'name')>
quark                              charm   1.3       0.666
quark                              strange 0.1       -0.333
<<U7>                              <<U7>   <float64> <float64>

>>> f.iloc[2:].relabel_drop_level(1)
<Frame>
<Index> mass      charge    <<U6>
<Index>
charm   1.3       0.666
strange 0.1       -0.333
<<U7>   <float64> <float64>


Mixing Selection Types with HLoc and ILoc


StaticFrame is consistent in what ``loc`` arguments mean: the first argument is a row selector, the second argument is a column selector. For selection within an ``IndexHierarchy`` found on either or both rows and columns, the ``sf.HLoc`` selector modifier is used.



>>> f.loc[sf.HLoc['lepton'], 'mass']
<Series: mass>
<IndexHierarchy: ('type', 'name')>
lepton                             muon  0.106
lepton                             tau   1.777
<<U6>                              <<U6> <float64>


>>> f.loc[sf.HLoc['lepton', 'tau']]
<Series: ('lepton', 'tau')>
<Index>
mass                        1.777
charge                      -1.0
<<U6>                       <float64>





No. 9: Indices are Always Unique
_______________________________________________

It is natural to think of indices and columns on a ``pd.DataFrame`` like primary key columns in a database table: a label that uniquely identifies each record of data. Pandas indices, however, are not (by default) constrainted to unique values. Creating an index with duplicates means that, for some single-label selections, a ``pd.Series`` will be returned, but for other single-label selections, a ``pd.DataFrame`` will be returned.


>>> df = pd.DataFrame.from_records([('muon', 0.106, -1.0, 'lepton'), ('tau', 1.777, -1.0, 'lepton'), ('charm', 1.3, 0.666, 'quark'), ('strange', 0.1, -0.333, 'quark')], columns=('name', 'mass', 'charge', 'type'))

>>> df.set_index('charge', inplace=True)

>>> df.loc[-1.0]
        name   mass    type
charge
-1.0    muon  0.106  lepton
-1.0     tau  1.777  lepton

>>> df.loc[0.666]
name    charm
mass      1.3
type    quark
Name: 0.666, dtype: object


This feature makes client code more complicated by having to handle selection results that might sometimes return a ``pd.Series`, other times returns a ``pd.DataFrame``. Further, when conceived as primary-key-like label, validating that indicies are unique is often a simple and effective check of data coherancy.

Pandas provides an optional check of uniqueness, called `verify_integrity`. And yet, by default integrity is disabled.


>>> df.set_index('type', verify_integrity=True)
Traceback (most recent call last):
ValueError: Index has duplicate keys: Index(['lepton', 'quark'], dtype='object', name='type')


In StaticFrame, indices are always unique. Attempting to set a non-unique index will raise an exception.


>>> f = sf.Frame.from_records((('muon', 0.106, -1.0, 'lepton'), ('tau', 1.777, -1.0, 'lepton'), ('charm', 1.3, 0.666, 'quark'), ('strange', 0.1, -0.333, 'quark')), columns=('name', 'mass', 'charge', 'type'))
>>> f.set_index('type')
Traceback (most recent call last):
static_frame.core.exception.ErrorInitIndex: labels (4) have non-unique values (2)



No. 10: There and Back Again to Pandas
____________________________________________________

StaticFrame is designed to work in environments side-by-side with Pandas. Going back and forth is made possible with specialized constructors and exporters, such as ``Frame.from_pandas`` or ``Series.to_pandas``.


>>> df = pd.DataFrame.from_records([('muon', 0.106, -1.0, 'lepton'), ('tau', 1.777, -1.0, 'lepton'), ('charm', 1.3, 0.666, 'quark'), ('strange', 0.1, -0.333, 'quark')], columns=('name', 'mass', 'charge', 'type'))
>>> df
      name   mass  charge    type
0     muon  0.106  -1.000  lepton
1      tau  1.777  -1.000  lepton
2    charm  1.300   0.666   quark
3  strange  0.100  -0.333   quark

>>> sf.Frame.from_pandas(df)
<Frame>
<Index> name     mass      charge    type     <object>
<Index>
0       muon     0.106     -1.0      lepton
1       tau      1.777     -1.0      lepton
2       charm    1.3       0.666     quark
3       strange  0.1       -0.333    quark
<int64> <object> <float64> <float64> <object>



Conclusion
____________________________________________________


The concept of a DataFrame came long before Pandas, and today finds realization in a wide variety of languages and implementations. Pandas will continue to provide an excellent resource to a broad community of uses. However, for situations where correctness and code maintainability are critical, StaticFrame offers an alternative API that is more consistent and maintainable, and that reduces opportunities for error.