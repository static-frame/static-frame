




Ten Reasons StaticFrame Might Lure You Away from Pandas

Ten Reasons to Try StaticFrame

Ten Reasons to Upgrade from Pandas to StaticFrame

Ten Ways StaticFrame can Improve your Pandas Code

Ten Reasons to Start Using StaticFrame for Your Next Pandas Project

Ten Tips for Transitioning from Pandas to StaticFrame



====================================================================

Complex Pandas applications can produce Python code that is hard to maintain and error prone. This happens because Pandas provides many ways to do the same thing, has inconsistent interfaces, and broadly supports in-place mutation. For those coming from Pandas, StaticFrame offers a more consistent interface and reduces opportunities for error. This article offers ten reasons to give StaticFrame a try.


Why StaticFrame
______________________

After years of using Pandas to develop back-end financial systems, it became clear to me that Pandas was not the right tool for the job. Pandas's handling of labeled data and missing values, with performance close to NumPy, certainly accelerated my productivity. And yet, the numerous inconsistencies in Pandas's API led to hard-to-maintain code. Further, Pandas's irregular approach to data ownership and support for in-place mutation led to serious opportunities for error. So in May of 2017 I began implementing a library more suitable for critical production systems.

Now, after three years of development and refinement, we are seeing excellent results in our production systems by replacing Pandas with StaticFrame. While, for some operations, StaticFrame is not yet as fast as Pandas, we often see StaticFrame out-perform Pandas in large-scale, real-world use cases.

What follows are ten reasons to try StaticFrame instead of Pandas for your next project. Of course, as the primary author of SaticFrame, I am biased: I think you might find some reasons to not use Pandas.

While Pandas users will find many familiar idioms, there are significant differences.

All examples use Pandas 1.0.3 and StaticFrame 0.6.20. Imports use the following convention:

>>> import pandas as pd
>>> import static_frame as sf


No. 1: Consistent and Discoverable Interfaces
____________________________________________________

An application programming interface (API) can be consistent in where functions are located, how functions are named, and the name and types of arguments those functions accept. StaticFrame deviates from Pandas's API to support greater consistency in all of these areas.

To create a ``Series`` or a ``Frame``, you need constructors. Pandas places its ``pd.DataFrame`` constructors in two places: on the root namespace (``pd``, as commonly imported) and on the ``pd.DataFrame`` class.

For example, JSON data is loaded from a function on the ``pd`` namespace, while record data (an iterable of Python sequences) is loaded from the ``pd.DataFrame`` class.


>>> pd.read_json('[{"name":"muon", "mass":0.106},{"name":"tau", "mass":1.777}]')
   name   mass
0  muon  0.106
1   tau  1.777
>>> pd.DataFrame.from_records([{"name":"muon", "mass":0.106}, {"name":"tau", "mass":1.777}])
   name   mass
0  muon  0.106
1   tau  1.777


Even though Pandas has specialized constructors, the default ``pd.DataFrame`` constructor accepts a staggering diversity of inputs, including many of the same inputs as ``pd.DataFrame.from_records()``.

>>> pd.DataFrame.from_records([{"name":"muon", "mass":0.106}, {"name":"tau", "mass":1.777}])
   name   mass
0  muon  0.106
1   tau  1.777
>>> pd.DataFrame([{"name":"muon", "mass":0.106}, {"name":"tau", "mass":1.777}])
   name   mass
0  muon  0.106
1   tau  1.777


For the user, there is no benefit to this diversity and redundancy. StaticFrame places all constructors on the class they construct, and as much as possible, narrowly focuses their functionality. As they are easier to maintain, explicit, specialized constructors are common in StaticFrame. For example, ``from_json()`` and ``from_dict_records()`` are available on the ``Frame`` class.

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


Being explicit leads to lots of constructors. To help you find what you are looking for, StaticFrame containers expose an ``interface`` attribute that provides the entire public interface of the calling class or instance as a ``Frame``. We can filter this table to show only constructors by using a ``loc[]`` selection.

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


Pandas displays its containers in diverse, inconsistent ways. For example, a ``pd.Series`` is shown with its name and type, while a ``pd.DataFrame`` shows neither of those attributes. If you display a ``pd.Index`` or ``pd.MultiIndex``, you get a third approach: a string suitable for ``eval()`` which is inscrutable when large.

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


StaticFrame offers a consistent, configurable display for all containers. The display of ``Series``, ``Frame``, ``Index``, and ``IndexHierarchy`` all share a common implementation and design. A priority of that design is to always make explicit container classes and underlying array types.

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


As much time is spent visually exploring the contents of ``Frame`` and ``Series`` containers, StaticFrame offers numerous display configuration options, all exposed through the ``DisplayConfig`` class.

For persistent changes, ``DisplayConfig`` instances can be passed to ``DisplayActive.set()``; for one-off changes, ``DisplayConfig`` instances can be passed to the container's ``display()`` method.

While ``pd.set_option`` can similarly be used to set Pandas display characteristics, such as maximum rows or columns, StaticFrame provides more extensive options for making data types apparent. As shown in the following terminal animation, specific types can be colored or type annotations can be removed entirely with ``DisplayConfig``.


.. image:: https://raw.githubusercontent.com/InvestmentSystems/static-frame/master/doc/images/animate-display-config.svg
   :align: center


No. 3: Immutable Data: Efficient Memory Management without Defensive Copies
___________________________________________________________________________________

Pandas displays inconsistent behavior in regard to ownership of data inputs and data exposed from within containers. In some cases, it is possible to mutate NumPy arrays "behind-the-back" of Pandas, exposing opportunities for confusion and bugs in Pandas code.

For example, if we give a 2D array as an input to a ``pd.DataFrame``, the original reference to the array can be used to "remotely" change the values within the ``pd.DataFrame``. In this case, the ``pd.DataFrame`` does not protect access to its data, serving only as a wrapper of a shared, mutable array.

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


Sometimes (but not always), NumPy arrays exposed from the ``values`` attribute of a ``pd.Series`` or a ``pd.DataFrame`` can be mutated, similarly changing the values within the ``DataFrame``.


>>> a2 = df['charge'].values
>>> a2
array([-1., -1.])
>>> a2[1] = np.nan # Mutating the array from .values.
>>> df # Mutation reflected in the DataFrame.
       mass  charge
muon    NaN    -1.0
tau   1.777     NaN


With StaticFrame, there is no vulnerability of "behind the back" mutation: as StaticFrame manages immutable NumPy arrays, references are only held to immutable arrays. If a mutable array is given at initialization, an immutable copy will be made. Immutable arrays cannot be mutated from containers or from direct access to underlying arrays.


>>> a1 = np.array([[0.106, -1], [1.777, -1]])
>>> f = sf.Frame(a1, index=('muon', 'tau'), columns=('mass', 'charge'))
>>> a1[0, 0] = np.nan # Mutating the original array has no affect on the Frame
>>> f
<Frame>
<Index> mass      charge    <<U6>
<Index>
muon    0.106     -1.0
tau     1.777     -1.0
<<U4>   <float64> <float64>
>>> f['charge'].values[1] = np.nan # An immutable array cannot be mutated
Traceback (most recent call last):
  File "<console>", line 1, in <module>
ValueError: assignment destination is read-only



While immutable data reduces opportunities for error, it also offers performance advantages. For example, when renaming an already-created ``Frame``, underlying data is not copied. Instead, references to the same immutable arrays are shared. Such "no-copy" operations are thus fast and light-weight.

>>> f.rename('fermion')
<Frame: fermion>
<Index>          symbol mass      <<U6>
<Index>
charm            c      1.3
strange          s      0.1
<<U7>            <<U1>  <float64>


Similarly, some types of concatenation (horizontal, axis-1 concatenation on aligned indices) can be done without copying data. Concatenating a ``Series`` to this ``Frame`` does not require copying underlying data to the new ``Frame``: it simply holds references to the already-allocated data.

>>> s = sf.Series.from_dict(dict(charm=0.666, strange=-0.333), name='charge')
>>> sf.Frame.from_concat((f, s), axis=1)
<Frame>
<Index> symbol mass      charge    <<U6>
<Index>
charm   c      1.3       0.666
strange s      0.1       -0.333
<<U7>   <<U1>  <float64> <float64>




No. 4: Assignment is a Function that Preserves Types
_____________________________________________________________


While Pandas permits in-place assignment and mutation, sometimes such operations cannot provide an appropriate derived type, resulting in undesirable behavior. For example, a float assigned into an integer ``pd.Series`` will have its floating-point components truncated without warning or error.

>>> s = pd.Series((-1, -1), index=('tau', 'down'))
>>> s
tau    -1
down   -1
dtype: int64
>>> s['down'] = -0.333 # Assigning a float.
>>> s # The -0.333 value was truncated to 0
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


Assignment on a ``Frame`` is similar: type compatibility is evaluated, and assignment only replaces what needs to change, reusing unchanged columns without copying data. For example, assigning a single value in a ``Frame`` results in only one new array being created; unchanged arrays are reused in the new ``Frame``.


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


Pandas has separate functions for iteration and function application. For iteration on a ``pd.DataFrame`` there is ``pd.DataFrame.iteritems()``, ``pd.DataFrame.iterrows()``, ``pd.DataFrame.itertuples()``, and ``pd.DataFrame.groupby()``; for function application on a ``pd.DataFrame`` there is ``pd.DataFrame.apply()`` and ``pd.DataFrame.applymap()``.

But since function application requires iteration, it is sensible for function application to be built on iteration. StaticFrame organizes iteration and function application by providing families of iterators (such as ``Frame.iter_array()`` or ``Frame.iter_group_items()``) that can be used for function application with an ``apply()`` method. Functions for applying mapping types (such as ``map_any()`` and ``map_fill()``) are also available on iterators. This means that once you know how you want to iterate, function application is a just a method away.

For example, we can create a ``Frame`` with ``Frame.from_records()``:


>>> f = sf.Frame.from_records(((0.106, -1.0, 'lepton'), (1.777, -1.0, 'lepton'), (1.3, 0.666, 'quark'), (0.1, -0.333, 'quark')), columns=('mass', 'charge', 'type'), index=('muon', 'tau', 'charm', 'strange'))
>>> f
<Frame>
<Index> mass      charge    type   <<U6>
<Index>
muon    0.106     -1.0      lepton
tau     1.777     -1.0      lepton
charm   1.3       0.666     quark
strange 0.1       -0.333    quark


We can iterate over elements in a ``Series`` with ``iter_element()``. We can use the same iterator to do function application, simply by using the ``apply()`` method.

>>> tuple(f['type'].iter_element())
('lepton', 'lepton', 'quark', 'quark')
>>> f['type'].iter_element().apply(lambda e: e.upper())
<Series>
<Index>
muon     LEPTON
tau      LEPTON
charm    QUARK
strange  QUARK
<<U7>    <<U6>


This approach is used for all iterators on all containers in StaticFrame. For example, we can use ``iter_element()`` and ``apply`` on a ``Frame``.

>>> f[['mass', 'charge']].iter_element().apply(lambda e: format(e, '.2e'))
<Frame>
<Index> mass     charge    <<U6>
<Index>
muon    1.06e-01 -1.00e+00
tau     1.78e+00 -1.00e+00
charm   1.30e+00 6.66e-01
strange 1.00e-01 -3.33e-01
<<U7>   <object> <object>


For row or column iteration, a family of methods allows specifying the type of container to be used for the iterated rows or columns, i.e, with an array, with a ``NamedTuple``, or with a ``Series`` (``iter_array()``, ``iter_tuple()``, ``iter_series()``, respectively). These methods take an axis argument to determine whether iteration is by row or by column, and similarly expose an ``apply()`` method for function application. To apply a function to columns, we can do the following.

>>> f[['mass', 'charge']].iter_array(axis=0).apply(np.sum)
<Series>
<Index>
mass     3.283
charge   -1.667
<<U6>    <float64>


If our ``apply()`` function needs to process both key and value pairs, we can use the corresponding items iterator, calling the provided function with both key and value.


>>> f.iter_array_items(axis=0).apply(lambda k, v: v.sum() if k != 'type' else np.nan)
<Series>
<Index>
mass     3.283
charge   -1.667
type     nan
<<U6>    <float64>


Applying a function to a row instead of a column simply requires changing the axis argument, and group iteration and function application follow the same pattern.

>>> f.iter_series(axis=1).apply(lambda s: s['mass'] > 1 and s['type'] == 'quark')
<Series>
<Index>
muon     False
tau      False
charm    True
strange  False
<<U7>    <bool>

>>> f.iter_group('type').apply(lambda f: f['mass'].mean())
<Series>
<Index>
lepton   0.9415
quark    0.7000000000000001
<<U6>    <float64>



No. 6: Strict, Grow-Only Frames
_____________________________________________

A practical and efficient use of a ``pd.DataFrame`` is to load initial data, then produce derived data by adding additional columns. This approach leverages the columnar organization of types and underlying arrays: adding new columns does not require re-allocating old columns. ``StaticFrame`` makes this approach less vulnerable to error by offering a strict, grow-only version of a ``Frame`` called a ``FrameGO``. For example, once a ``FrameGO`` is created, new columns can be added while existing columns cannot be overwritten or mutated in-place.


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


This limited form of mutation meets a practical need. Further, converting back and forth from a ``Frame`` to a ``FrameGO`` (using ``Frame.to_frame_go()`` and ``FrameGO.to_frame()``) is a no-copy operation: underlying immutable arrays can be shared between the two containers.



No. 7: Dates are not Nanoseconds
__________________________________________________________________

Pandas models all date or timestamp values as NumPy ``datetime64[ns]`` (nanosecond) arrays, regardless of if nanosecond-level resolution is practical or appropriate. This has the result of creating a "Y2262 problem" for Pandas: dates beyond 2262-04-11 cannot be expressed. While I can create a ``pd.DatetimeIndex`` up to 2262-04-11, one day further and Pandas raises an error.

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


As indices are often used for date-time values far less granular than nanoseconds (such as dates, months, or years), StaticFrame offers the full range of NumPy typed ``datetime64`` indices. This permits exact date-time type specification, and avoids the limits of nanosecond-based units.

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


No. 8: Well-Behaved Hierarchical Indices
___________________________________________


Hierarchical indices permit fitting many dimensions into one. Using hierarchical indices, *n*-dimensional data can be encoded into a single ``Series`` or ``Frame``.

Pandas's implementation of hierarchical indices, the ``pd.MultiIndex``, behaves inconsistently, forcing client code to handle unnecessary variability. We can see this by creating a ``pd.DataFrame`` and setting a ``pd.MultiIndex``.


>>> df = pd.DataFrame.from_records([('muon', 0.106, -1.0, 'lepton'), ('tau', 1.777, -1.0, 'lepton'), ('charm', 1.3, 0.666, 'quark'), ('strange', 0.1, -0.333, 'quark')], columns=('name', 'mass', 'charge', 'type'))
>>> df.set_index(['type', 'name'], inplace=True)
>>> df
                 mass  charge
type   name
lepton muon     0.106  -1.000
       tau      1.777  -1.000
quark  charm    1.300   0.666
       strange  0.100  -0.333


When selecting subsets of data from the ``pd.MultiIndex``, whether or not Pandas returns a ``pd.MultiIndex`` or a ``pd.Index`` depends on how the selection is made. For example, implicitly selecting a single outer level reduces the ``pd.MultiIndex`` to a normal ``pd.Index``, yet an equivalent selection, using a slice, retains the ``pd.MultiIndex``.


>>> df.loc['quark'] # Returned index is 1D
         mass  charge
name
charm     1.3   0.666
strange   0.1  -0.333
>>> df.iloc[2:] # Returned index is 2D
               mass  charge
type  name
quark charm     1.3   0.666
      strange   0.1  -0.333


The meaning of positional arguments in a ``loc[]`` selection with a ``pd.MultiIndex`` is similarly inconsistent. In general usage with a ``pd.DataFrame``, when two arguments are given to ``loc[]``, the first argument is a row selector, the second argument is a column selector.

>>> df.loc['lepton', 'mass'] # Selects "lepton" from row, "mass" from columns
name
muon    0.106
tau     1.777
Name: mass, dtype: float64


Yet, in violation of that expectation, sometimes Pandas will not use the second ``loc[]`` argument as a column selection, but instead as a row selection in an inner-depth of ``pd.MultiIndex``.

>>> df.loc['lepton', 'tau'] # Selects lepton and tau from rows
mass      1.777
charge   -1.000
Name: (lepton, tau), dtype: float64


If a column selection is required, the expected behavior can be restored by wrapping the hierarchical row selection within a ``pd.IndexSlice[]`` selection modifier.


>>> df.loc[pd.IndexSlice['lepton', 'tau'], 'charge']
-1.0


This inconsistency in the meaning of the positional arguments given to ``loc[]`` is unnecessary and makes Pandas code harder to maintain: what is intended from the usage of ``loc[]`` becomes ambiguous without a ``pd.IndexSlice[]``.

StaticFrame's ``IndexHierarchy`` offers more consistent behavior. We will create an equivalent ``Frame`` and set an ``IndexHierarchy``.


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


Unlike Pandas, a selection never automatically reduces the ``IndexHierarchy`` to an ``Index``. If reduction is needed, the ``Frame.relabel_drop_level()`` method can be used. This is a lightweight operation that does not copy underlying data. Notice also that an ``HLoc[]`` selection modifier, similar to ``pd.IndexSlice`` is always required for partial selections within a hierarchical index.


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


Unlike Pandas, StaticFrame is consistent in what positional ``loc[]`` arguments mean: the first argument is always a row selector, the second argument is always a column selector. For selection within an ``IndexHierarchy``, the ``HLoc[]`` selection modifier is required to specify selection at arbitrary depths within the hierarchy. This approach makes StaticFrame code easier to understand and maintain.

>>> f.loc[sf.HLoc['lepton']]
<Frame>
<Index>                                  mass      charge    <<U6>
<IndexHierarchy: ('type', 'name')>
lepton                             muon  0.106     -1.0
lepton                             tau   1.777     -1.0
<<U6>                              <<U4> <float64> <float64>
>>> f.loc[sf.HLoc[:, ['muon', 'strange']], 'mass']
<Series: mass>
<IndexHierarchy: ('type', 'name')>
lepton                             muon    0.106
quark                              strange 0.1
<<U6>                              <<U7>   <float64>




No. 9: Indices are Always Unique
_______________________________________________

It is natural to think index and column labels on a ``pd.DataFrame`` are unique identifiers: their interfaces suggest that they are like Python dictionaries, where keys are always unique. Pandas indices, however, are not constrained to unique values. Creating an index on a ``pd.Frame`` with duplicates means that, for some single-label selections, a ``pd.Series`` will be returned, but for other single-label selections, a ``pd.DataFrame`` will be returned.


>>> df = pd.DataFrame.from_records([('muon', 0.106, -1.0, 'lepton'), ('tau', 1.777, -1.0, 'lepton'), ('charm', 1.3, 0.666, 'quark'), ('strange', 0.1, -0.333, 'quark')], columns=('name', 'mass', 'charge', 'type'))
>>> df.set_index('charge', inplace=True) # Creating an index with duplicated labels
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


Pandas support of non-unique indices makes client code more complicated by having to handle selections that sometimes return a ``pd.Series`` and other times return a ``pd.DataFrame``. Further, uniqueness of indices is often a simple and effective check of data coherency.

Some Pandas interfaces, such as ``pd.concat()`` and ``pd.DataFrame.set_index()``, provide an optional check of uniqueness with a parameter named ``verify_integrity``. While it seems obvious that integrity is desirable, by default Pandas disables ``verify_integrity``.


>>> df.set_index('type', verify_integrity=True)
Traceback (most recent call last):
ValueError: Index has duplicate keys: Index(['lepton', 'quark'], dtype='object', name='type')


In StaticFrame, indices are always unique. Attempting to set a non-unique index will raise an exception. This constraint eliminates opportunities for mistakenly introducing duplicates in indices.


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

StaticFrame is designed to work in environments side-by-side with Pandas. Going back and forth is made possible with specialized constructors and exporters, such as ``Frame.from_pandas()`` or ``Series.to_pandas()``.


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


The concept of a "data frame" object came long before Pandas: the first implementation may have been as early as 1991 in the S language, a predecessor of R. Today, the data frame finds realization in a wide variety of languages and implementations. Pandas will continue to provide an excellent resource to a broad community of users. However, for situations where correctness and code maintainability are critical, StaticFrame offers an alternative designed to be more consistent and reduce opportunities for error.

For more information about StaticFrame, see the documentation (http://static-frame.readthedocs.io) or project site (https://github.com/InvestmentSystems/static-frame).

