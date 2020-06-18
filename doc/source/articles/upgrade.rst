


Ten Tips for Transitioning from Pandas to StaticFrame
====================================================================

Pandas code can become complex, hard to maintain, and error prone. This happens because Pandas supports many ways to do the same thing, has inconsistencies in its interfaces, and broadly supports in-place mutation. For those coming from Pandas, StaticFrame offers a more consistent interface and reduces opportunities for error. This article provides ten tips to get up to speed with StaticFrame.


Why StaticFrame
______________________

After years of using Pandas to develop production, back-end financial systems, it became clear to me that Pandas was not the right tool. Pandas's handling of labeled data and missing values, with performance close to NumPy, certainly accelerated my productivity. And yet, the numerous inconsistencies in Pandas's API led to hard-to-maintain code. Further, Pandas's irregular approach to data ownership and support for mutation and undesirable side effects led to serious vulnerabilities and opportunities for error. So in May of 2017 I began implementing a library more suitable for critical production systems.

Now, after three years of development and refinement, we are seeing excellent results in our production systems by replacing Pandas with StaticFrame. While StaticFrame is not yet always as fast as Pandas for some operations, we often see StaticFrame out-performing Pandas on large-scale, real-world processes.

What follows are ten tips to aid Pandas users in transitioning to StaticFrame. While Pandas users will find many familiar idioms, there are significant differences.


No. 1: Consistent and Discoverable Interfaces
____________________________________________________


An API can be consistent in where functions are located, how functions are named, and the name and types of arguments those functions accept. StaticFrame deviates from Pandas's API to support greater consistency in all of these areas.

To create ``Series`` and ``Frame``, you need constructors. Pandas places its ``pd.DataFrame`` constructors in at least two places: on the root namespace (``pd``, as commonly imported) and on the ``pd.DataFrame`` class.

For example, JSON data is loaded from a function on the ``pd`` namespace, while records (Python sequences) are loaded from the ``pd.DataFrame`` class.


>>> pd.read_json('[{"name":"muon", "mass":0.106},{"name":"tau", "mass":1.777}]')
   name   mass
0  muon  0.106
1   tau  1.777

>>> pd.DataFrame.from_records([{"name":"muon", "mass":0.106}, {"name":"tau", "mass":1.777}])
   name   mass
0  muon  0.106
1   tau  1.777


Even though Pandas has specialized constructors, the default ``pd.DataFrame`` constructor accepts a staggering diversity of inputs, including many of the same inputs as ``pd.DataFrame.from_records``.

>>> pd.DataFrame.from_records([{"name":"muon", "mass":0.106}, {"name":"tau", "mass":1.777}])
   name   mass
0  muon  0.106
1   tau  1.777

>>> pd.DataFrame([{"name":"muon", "mass":0.106}, {"name":"tau", "mass":1.777}])
   name   mass
0  muon  0.106
1   tau  1.777


For the user, there is no benefit to this diversity and redundancy. StaticFrame places all constructors on the class they construct, and as much as possible, narrowly focuses their utility. As they are easier to maintain, explicit, specialized constructors are common in StaticFrame. For example, ``from_json`` and ``from_dict_records`` are available on the ``Frame`` class.

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


Being explicit leads to lots of constructors. To help find what you are looking for, StaticFrame containers expose an ``interface`` attribute that lists the entire public interface of the class or instance. We can filter this table to just the constructors by using a ``loc`` selection.

>>> sf.Frame.interface.loc[sf.Frame.interface['group'] == 'Constructor']
<Frame: Frame>
<Index>                              cls_name group       doc                  <<U18>
<Index: signature>
__init__(data, *, index, columns,... Frame    Constructor
from_arrow(value, *, index_depth,... Frame    Constructor Convert an Arrow ...
from_concat(frames, *, axis, unio... Frame    Constructor Concatenate multi...
from_concat_items(items, *, axis,... Frame    Constructor Produce a Frame w...
from_csv(fp, *, index_depth, inde... Frame    Constructor Specialized versi...
from_delimited(fp, *, delimiter, ... Frame    Constructor Create a Frame fr...
from_dict(mapping, *, index, fill... Frame    Constructor Create a Frame fr...
from_dict_records(records, *, ind... Frame    Constructor Frame constructor...
from_dict_records_items(items, *,... Frame    Constructor Frame constructor...
from_element(element, *, index, c... Frame    Constructor Create a Frame fr...
from_element_iloc_items(items, *,... Frame    Constructor Given an iterable...
from_element_loc_items(items, *, ... Frame    Constructor This function is ...
from_elements(elements, *, index,... Frame    Constructor Create a Frame fr...
from_hdf5(fp, *, label, index_dep... Frame    Constructor Load Frame from t...
from_items(pairs, *, index, fill_... Frame    Constructor Frame constructor...
from_json(json_data, *, dtypes, n... Frame    Constructor Frame constructor...
from_json_url(url, *, dtypes, nam... Frame    Constructor Frame constructor...
from_pandas(value, *, index_const... Frame    Constructor Given a Pandas Da...
from_parquet(fp, *, index_depth, ... Frame    Constructor Realize a Frame f...
from_records(records, *, index, c... Frame    Constructor Construct a Frame...
from_records_items(items, *, colu... Frame    Constructor Frame constructor...
from_series(series, *, name, colu... Frame    Constructor Frame constructor...
from_sql(query, *, connection, in... Frame    Constructor Frame constructor...
from_sqlite(fp, *, label, index_d... Frame    Constructor Load Frame from t...
from_structured_array(array, *, i... Frame    Constructor Convert a NumPy s...
from_tsv(fp, *, index_depth, inde... Frame    Constructor Specialized versi...
from_xlsx(fp, *, label, index_dep... Frame    Constructor Load Frame from t...
<<U94>                               <<U5>    <<U17>      <<U83>



No. 2: Consistent and Colorful Display
___________________________________________


Pandas displays its containers in diverse, inconsistent ways. For example, a ``pd.Series`` is shown with its name and type, while a ``pd.DataFrame`` shows neither of those attributes. If you display a ``pd.Index`` or ``pd.MultiIndex``, you get a third approach: an ``eval``-able string, but one that is unmanageable when large.

>>> df = pd.DataFrame.from_records([{'symbol':'c', 'mass':1.3}, {'symbol':'s', 'mass':0.1}], index=('charm', 'strange'))
>>> df
        symbol  mass
charm        c   1.3
strange      s   0.1

>>> df['mass']
charm      1.3
strange    0.1
Name: mass, dtype: float64

>>> df.index
Index(['charm', 'strange'], dtype='object')


StaticFrame offers a consistent, configurable display for all containers. The display of ``Series``, ``Frame``, ``Index``, and ``IndexHierarchy`` all share a common design. Under the hood, the display components are modular and reusable: the display components are composed to return the final display.

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

>>> f.columns
<Index>
symbol
mass
<<U6>


As much time is spent looking at the contents of ``Frame`` and ``Series``, StaticFrame offers numerous configuration options for displaying containers, all exposed through the ``DisplayConfig`` class. Specific types can be colored and type annotations can be removed entirely.


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


Pandas displays inconsistent behavior in regard to ownership of data inputs and data attached to containers. In some cases, it is possible to mutate NumPy arrays "behind-the-back" of Pandas.

For example, if we give an 2D array as an input to a ``pd.DataFrame``, the lingering reference of the array can be used to "remotely" change the values of ``pd.DataFrame``. In this case, the ``pd.DataFrame`` does not protect access to its data, serving only as a wrapper of a shared, mutable array.

>>> a1 = np.array([[0.106, -1], [1.777, -1]])
>>> df = pd.DataFrame(a1, index=('muon', 'tau'), columns=('mass', 'charge'))
>>> df
       mass  charge
muon  0.106    -1.0
tau   1.777    -1.0

>>> a1[0, 0] = np.nan # Mutating the original array.

>>> df # Mutation reflected in the DataFrame created from that array.
       mass  charge
muon    NaN    -1.0
tau   1.777    -1.0



Sometimes (but not always), NumPy arrays exposed from the ``values`` attribute of a ``pd.Series`` or a ``pd.DataFrame`` can be mutated, similarly changing the values of the ``DataFrame`` from which they were extracted.


>>> a2 = df['charge'].values
>>> a2
array([-1., -1.])

>>> a2[1] = np.nan # Mutating the array from .values.

>>> df # Mutation reflected in the DataFrame.
       mass  charge
muon    NaN    -1.0
tau   1.777     NaN


As StaticFrame manages immutable NumPy arrays, this vulnerability of "behind the back" mutation is never permitted, either from StaticFrame containers or from direct access to underlying arrays.


>>> f = sf.Frame.from_dict_records_items((('charm', {'symbol':'c', 'mass':1.3}), ('strange', {'symbol':'s', 'mass':0.1})))


>>> f.loc['charm', 'mass'] = np.nan
Traceback (most recent call last):
  File "<console>", line 1, in <module>
TypeError: 'InterfaceGetItem' object does not support item assignment

>>> f['mass'].values[1] = 100
Traceback (most recent call last):
  File "<console>", line 1, in <module>
ValueError: assignment destination is read-only


While immutable data reduces opportunities for error, it also offers performance advantages. For example, when creating a new ``Frame`` with a new ``name`` attribute, underlying data is not copied. Instead, references to the same immutable array are shared. Such "no-copy" operations are thus fast and light-weight.

>>> f.rename('fermion')
<Frame: fermion>
<Index>          symbol mass      <<U6>
<Index>
charm            c      1.3
strange          s      0.1
<<U7>            <<U1>  <float64>



Similarly, some types of concatenation (horizontal, axis 1 concatenation on aligned indices) can be done without copying data. Concatenating a ``Series`` to this ``Frame`` does not require copying underlying data to the new ``Frame``: it simply holds references to the data already allocated.

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


While Pandas permits arbitrary assignment, those assignments happen in-place, making getting the right derived type (when needed) difficult, and resulting in some undesirable behavior. For example, a float assigned into an integer `pd.Series` will have its floating-point components truncated without warning or error.

>>> s = pd.Series((-1, -1), index=('tau', 'down'))
>>> s
tau    -1
down   -1
dtype: int64

>>> s['down'] = -0.333 # Assigning a float.

>>> s # The -0.333 values was truncated to 0
tau    -1
down    0
dtype: int64


With StaticFrame's immutable data model, assignment is a function that returns a new container. This permits evaluating types to insure that the resultant array can completely contain the assigned value.


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



Assignment on a ``Frame`` is similar: ``Frame`` assignment only mutates what needs to change, reusing unchanged columns without copying data.

For example, assigning to a single value in a ``Frame`` results in only one new array being created; the unchanged arrays are reused in the new ``Frame``.


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

>>> f.assign.loc['charm', 'charge'](Fraction(2, 3)) # Assignment only affects one column.
<Frame>
<Index> charge   mass      <<U6>
<Index>
charm   2/3      1.3
strange -0.333   0.1
<<U7>   <object> <float64>




No. 5: Iterators are for Iterating and Function Application
________________________________________________________________


Pandas has separate functions for iterating and function application. For example, there is ``pd.DataFrame.iteritems``, ``pd.DataFrame.iterrows``, ``pd.DataFrame.itertuples``, ``pd.DataFrame.groupby`` for iteration, and ``pd.DataFrame.apply`` and ``pd.DataFrame.applymap`` for function application.

But since function application requires iteration, it is sensible for function application to build on iteration. StaticFrame organizes iteration and function application by providing families of iterators (such as ``Frame.iter_array`` or ``Frame.iter_group_items``)that can be extended to function application with an ``apply`` method. Functions for using mapping types (such as ``map_any`` and ``map_fill``) are also available on iterators. This means that once you find how you want to iterate, function application is a just a method away.

For an example, we will create a ``Frame`` with ``Frame.from_records``:


>>> f = sf.Frame.from_records(((0.106, -1.0, 'lepton'), (1.777, -1.0, 'lepton'), (1.3, 0.666, 'quark'), (0.1, -0.333, 'quark')), columns=('mass', 'charge', 'type'), index=('muon', 'tau', 'charm', 'strange'))
>>> f
<Frame>
<Index> mass      charge    type   <<U6>
<Index>
muon    0.106     -1.0      lepton
tau     1.777     -1.0      lepton
charm   1.3       0.666     quark
strange 0.1       -0.333    quark


Ee can iterate over elements in a ``Series`` with ``iter_element()``.

>>> tuple(f['type'].iter_element())
('lepton', 'lepton', 'quark', 'quark')


We can use the same iterator to do function application, simply by using the ``apply`` method.

>>> f['type'].iter_element().apply(lambda e: e.upper())
<Series>
<Index>
muon     LEPTON
tau      LEPTON
charm    QUARK
strange  QUARK
<<U7>    <<U6>


This same approach is used for all iterators on all containers. For example, we can use ``iter_element()`` on ``Frame`` to apply string formatting to each element.

>>> f.iter_element().apply(lambda e: str(e).rjust(8, '.'))
<Frame>
<Index> mass     charge   type     <<U6>
<Index>
muon    ...0.106 ....-1.0 ..lepton
tau     ...1.777 ....-1.0 ..lepton
charm   .....1.3 ...0.666 ...quark
strange .....0.1 ..-0.333 ...quark
<<U7>   <object> <object> <object>


For row or column iteration, a family of methods allows specifying the type of container to be used for the iterated rows or columns, i.e, as an array, as a ``NamedTuple``, or as a ``Series`` (``iter_array()``, ``iter_tuple()``, ``iter_series()``, respectively). These methods take an axis argument to determine whether iteration is by row or by column.

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


Applying a function to a row instead of a column simply requires changing the axis argument.

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

A practical use of ``pd.DataFrame`` is to load initial data, then produce derived data by adding additional columns. ``StaticFrame`` makes this approach less vulnerable to error by offering a strict, grow-only version of a ``Frame`` called a ``FrameGO``.

For example, once a ``FrameGO`` is created, new columns can be added while existing columns cannot be overwritten or mutated in-place.


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


This limited form of mutation meets a practical need. Further, converting back and forth from a ``Frame`` to a ``FrameGO`` (using ``Frame.to_frame_go()`` and ``FrameGO.to_frame()``) is a no-copy operation.



No 7: Everything is not a Nanosecond
__________________________________________________________________

Pandas models every date or timestamp as a NumPy nanosecond ``datetime64`` object, regardless of if nanosecond-level resolution is practical or appropriate. This has the amusing side effect of creating a "Y2262 problem" for Pandas: dates beyond 2262-04-11 cannot be expressed. While I can create a ``pd.DatetimeIndex`` up to 2262-04-11, one day further and Pandas raises an error.

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
pandas._libs.tslibs.np_datetime.OutOfBoundsDatetime: Out of bounds nanosecond timestamp: 2262-04-12 00:00:00


As indices are often used for date-time values much less granular than nanoseconds (such as dates, months, or years), StaticFrame offers the full range of NumPy typed ``datetime64`` indices. This permits exact date-time specification, and avoids the limits of nanosecond-based units.

While not possible with Pandas, creating an index of years or dates extending to the year 3000 is not a problem with StaticFrame.

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


Hierarchical indices permit fitting many dimensions into one. Using hierarchical indices, *n*-dimensional data can be encoded into ``Series`` or ``Frame``.

Pandas implementation of hierarchical indices, the `pd.MultiIndex`, while powerful, behaves inconsistently, again forcing client code to handle unnecessary variability. We can begin by creating a ``pd.DataFrame`` and setting a ``pd.MultiIndex``.


>>> df = pd.DataFrame.from_records([('muon', 0.106, -1.0, 'lepton'), ('tau', 1.777, -1.0, 'lepton'), ('charm', 1.3, 0.666, 'quark'), ('strange', 0.1, -0.333, 'quark')], columns=('name', 'mass', 'charge', 'type'))
>>> df.set_index(['type', 'name'], inplace=True)
>>> df
                 mass  charge
type   name
lepton muon     0.106  -1.000
       tau      1.777  -1.000
quark  charm    1.300   0.666
       strange  0.100  -0.333


When selecting subsets of data from the ``pd.MultiIndex``, whether or not Pandas returns a ``pd.MultiIndex`` or 1D index depends on how the selection is made. For example, implicitly selecting a single outer level reduces the ``pd.MultiIndex`` to a normal ``pd.Index``, yet an equivalent selection, using a slice, retains the ``pd.MultiIndex``.


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


Another inconsistency is in using ``loc`` with a ``pd.MultiIndex``. In common usage, when two arguments are given to ``loc``, the first argument is for row selection, the second argument is for column selection.

>>> df.loc['lepton', 'mass'] # Selects "lepton" from row, "mass" from columns
name
muon    0.106
tau     1.777
Name: mass, dtype: float64


In opposition to that behavior, Pandas will sometimes (depending on the value of the  argument) use the second argument in ``loc`` not as a column selection, but rather as an inner level selection on the rows.

>>> df.loc['lepton', 'tau'] # Selects lepton and tau from rows
mass      1.777
charge   -1.000
Name: (lepton, tau), dtype: float64


If a column selection is required, the more common behavior can be restored by wrapping the hierarchical row selection within a ``pd.IndexSlice`` selector.

>>> df.loc[pd.IndexSlice['lepton', 'tau'], 'charge']
-1.0

This inconsistency in the meaning of the arguments given to ``loc`` is unnecessary and difficult to maintain: what is intended from the usage of ``loc`` cannot be known without knowing what labels might be found in that index.

StaticFrame's ``IndexHierarchy`` are built from ``Index`` objects and offer more consistent behavior. We will create an equivalent ``Frame`` and set an ``IndexHierarchy``.

>>> f = sf.Frame.from_records((('muon', 0.106, -1.0, 'lepton'), ('tau', 1.777, -1.0, 'lepton'), ('charm', 1.3, 0.666, 'quark'), ('strange', 0.1, -0.333, 'quark')), columns=('name', 'mass', 'charge', 'type'))
>>> f = f.set_index_hierarchy(('type', 'name'), drop=True)
>>> f
<Frame>
<Index>                                    mass      charge    <<U6>
<IndexHierarchy: ('type', 'name')>
lepton                             muon    0.106     -1.0
lepton                             tau     1.777     -1.0
quark                              charm   1.3       0.666
quark                              strange 0.1       -0.333
<<U6>                              <<U7>   <float64> <float64>


Unlike Pandas, a selection never automatically reduces the ``IndexHierarchy`` to an ``Index``. If reduction is needed, the ``Frame.relabel_drop_level()`` can be used. This is a lightweight operation that does not copy underlying data. Notice also that an ``sf.HLoc`` selection modifier, similar to ``pd.IndexSlice`` is always required for partial selections within a hierarchical index.


>>> f.loc[sf.HLoc['quark']]
<Frame>
<Index>                                    mass      charge    <<U6>
<IndexHierarchy: ('type', 'name')>
quark                              charm   1.3       0.666
quark                              strange 0.1       -0.333
<<U5>                              <<U7>   <float64> <float64>

>>> f.iloc[2:]
<Frame>
<Index>                                    mass      charge    <<U6>
<IndexHierarchy: ('type', 'name')>
quark                              charm   1.3       0.666
quark                              strange 0.1       -0.333
<<U5>                              <<U7>   <float64> <float64>

>>> f.iloc[2:].relabel_drop_level(1)
<Frame>
<Index> mass      charge    <<U6>
<Index>
charm   1.3       0.666
strange 0.1       -0.333
<<U7>   <float64> <float64>


Further, unlike Pandas, StaticFrame is consistent in what ``loc`` arguments mean: the first argument is always a row selector, the second argument is always a column selector. For selection within an ``IndexHierarchy`` the ``sf.HLoc`` selection modifier is used to specify selection within the depths of the ``IndexHierarchy``.


>>> f.loc[sf.HLoc['lepton'], 'mass']
<Series: mass>
<IndexHierarchy: ('type', 'name')>
lepton                             muon  0.106
lepton                             tau   1.777
<<U6>                              <<U4> <float64>

>>> f.loc[sf.HLoc['lepton', 'tau'], 'charge']
-1.0





No. 9: Indices are Always Unique
_______________________________________________

It is natural to think of indices and columns on a ``pd.DataFrame`` as unique identifiers: their interfaces suggest they are like dictionaries, where keys are always unique. Pandas indices, however, are not constrained to unique values. Creating an index on a ``pd.Frame`` with duplicates means that, for some single-label selections, a ``pd.Series`` will be returned, but for other single-label selections, a ``pd.DataFrame`` will be returned.


>>> df = pd.DataFrame.from_records([('muon', 0.106, -1.0, 'lepton'), ('tau', 1.777, -1.0, 'lepton'), ('charm', 1.3, 0.666, 'quark'), ('strange', 0.1, -0.333, 'quark')], columns=('name', 'mass', 'charge', 'type'))
>>> df.set_index('charge', inplace=True)
>>> df
           name   mass    type
charge
-1.000     muon  0.106  lepton
-1.000      tau  1.777  lepton
 0.666    charm  1.300   quark
-0.333  strange  0.100   quark

>>> df.loc[-1.0] # Selecting a non-unique label results in a pd.DataFrame
        name   mass    type
charge
-1.0    muon  0.106  lepton
-1.0     tau  1.777  lepton

>>> df.loc[0.666] # Selecting a unique label results in a pd.Series
name    charm
mass      1.3
type    quark
Name: 0.666, dtype: object


Pandas support of non-unique indices makes client code more complicated by having to handle selections that might sometimes return a ``pd.Series` and other times returns a ``pd.DataFrame``. Further, uniqueness of indices is often a simple and effective check of data coherency.

In some interfaces Pandas provides an optional check of uniqueness, called `verify_integrity`. While it seems obvious that integrity it desirable, by default `verify_integrity` is disabled.


>>> df.set_index('type', verify_integrity=True)
Traceback (most recent call last):
ValueError: Index has duplicate keys: Index(['lepton', 'quark'], dtype='object', name='type')


In StaticFrame, indices are always unique. Attempting to set a non-unique index will always raise an exception.


>>> f = sf.Frame.from_records((('muon', 0.106, -1.0, 'lepton'), ('tau', 1.777, -1.0, 'lepton'), ('charm', 1.3, 0.666, 'quark'), ('strange', 0.1, -0.333, 'quark')), columns=('name', 'mass', 'charge', 'type'))
>>> f
<Frame>
<Index> name    mass      charge    type   <<U6>
<Index>
0       muon    0.106     -1.0      lepton
1       tau     1.777     -1.0      lepton
2       charm   1.3       0.666     quark
3       strange 0.1       -0.333    quark
<int64> <<U7>   <float64> <float64> <<U6>

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


The concept of a DataFrame came long before Pandas, and today finds realization in a wide variety of languages and implementations. Pandas will continue to provide an excellent resource to a broad community of users. However, for situations where correctness and code maintainability are critical, StaticFrame offers an alternative API designed to be more consistent and reduce opportunities for error.