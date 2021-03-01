

Ten Reasons to Use StaticFrame Instead of Pandas
====================================================================

If you work with data in Python, you probably use Pandas. Pandas provides nearly instant gratification: sophisticated data processing routines can be implemented in a few lines of code. However, if you have used Pandas on large projects over many years, you may have had some challenges. Complex Pandas applications can produce Python code that is hard to maintain and error-prone. This happens because Pandas provides many ways to do the same thing, has inconsistent interfaces, and broadly supports in-place mutation. For those coming from Pandas, StaticFrame offers a more consistent interface and reduces opportunities for error. This article demonstrates ten reasons you might use StaticFrame instead of Pandas.


Why StaticFrame
______________________

After years of using Pandas to develop back-end financial systems, it became clear to me that Pandas was not the right tool for the job. Pandas's handling of labeled data and missing values, with performance close to NumPy, certainly accelerated my productivity. And yet, the numerous inconsistencies in Pandas's API led to hard-to-maintain code. Further, Pandas's support for in-place mutation led to serious opportunities for error. So in May of 2017 I began implementing a library more suitable for critical production systems.

Now, after years of development and refinement, we are seeing excellent results in our production systems by replacing Pandas with StaticFrame. Libraries and applications written with StaticFrame are easier to maintain and test. And we often see StaticFrame out-perform Pandas in large-scale, real-world use cases, even though, for many isolated operations, StaticFrame is not yet as fast as Pandas.

What follows are ten reasons to favor using StaticFrame over Pandas. As the primary author of StaticFrame, I am certainly biased in this presentation. However, having worked with Pandas since 2013, I hope to have some perspective to share.

All examples use Pandas 1.0.3 and StaticFrame 0.6.20. Imports use the following convention:

>>> import pandas as pd
>>> import static_frame as sf


No. 1: Consistent and Discoverable Interfaces
____________________________________________________

An application programming interface (API) can be consistent in where functions are located, how functions are named, and the name and types of arguments those functions accept. StaticFrame deviates from Pandas's API to support greater consistency in all of these areas.

To create a ``sf.Series`` or a ``sf.Frame``, you need constructors. Pandas places its ``pd.DataFrame`` constructors in two places: on the root namespace (``pd``, as commonly imported) and on the ``pd.DataFrame`` class.

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


>>> pd.DataFrame([{"name":"muon", "mass":0.106}, {"name":"tau", "mass":1.777}])
   name   mass
0  muon  0.106
1   tau  1.777


For the user, there is little benefit to this diversity and redundancy. StaticFrame places all constructors on the class they construct, and as much as possible, narrowly focuses their functionality. As they are easier to maintain, explicit, specialized constructors are common in StaticFrame. For example, ``sf.Frame.from_json()`` and ``sf.Frame.from_dict_records()``:

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


Being explicit leads to lots of constructors. To help you find what you are looking for, StaticFrame containers expose an ``interface`` attribute that provides the entire public interface of the calling class or instance as a ``sf.Frame``. We can filter this table to show only constructors by using a ``sf.Frame.loc[]`` selection.

>>> sf.Frame.interface.loc[sf.Frame.interface['group'] == 'Constructor']
<Frame: Frame>
<Index>                              cls_name group       doc                  <<U18>
<Index: signature>
__init__(data, *, index, columns,... Frame    Constructor Initializer. Args...
from_arrow(value, *, index_depth,... Frame    Constructor Realize a Frame f...
from_clipboard(*, delimiter, inde... Frame    Constructor Create a Frame fr...
from_concat(frames, *, axis, unio... Frame    Constructor Concatenate multi...
from_concat_items(items, *, axis,... Frame    Constructor Produce a Frame w...
from_csv(fp, *, index_depth, inde... Frame    Constructor Specialized versi...
from_delimited(fp, *, delimiter, ... Frame    Constructor Create a Frame fr...
from_dict(mapping, *, index, fill... Frame    Constructor Create a Frame fr...
from_dict_records(records, *, ind... Frame    Constructor Frame constructor...
from_dict_records_items(items, *,... Frame    Constructor Frame constructor...
from_element(element, *, index, c... Frame    Constructor Create a Frame fr...
from_element_items(items, *, inde... Frame    Constructor Create a Frame fr...
from_elements(elements, *, index,... Frame    Constructor Create a Frame fr...
from_fields(fields, *, index, col... Frame    Constructor Frame constructor...
from_hdf5(fp, *, label, index_dep... Frame    Constructor Load Frame from t...
from_items(pairs, *, index, fill_... Frame    Constructor Frame constructor...
from_json(json_data, *, dtypes, n... Frame    Constructor Frame constructor...
from_json_url(url, *, dtypes, nam... Frame    Constructor Frame constructor...
from_msgpack(msgpack_data)           Frame    Constructor Frame constructor...
from_overlay(containers, *, index... Frame    Constructor Return a new Fram...
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
<<U94>                               <<U5>    <<U18>      <<U83>


No. 2: Consistent and Colorful Display
___________________________________________


Pandas displays its containers in diverse ways. For example, a ``pd.Series`` is shown with its name and type, while a ``pd.DataFrame`` shows neither of those attributes. If you display a ``pd.Index`` or ``pd.MultiIndex``, you get a third approach: a string suitable for ``eval()`` which is inscrutable when large.

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


StaticFrame offers a consistent, configurable display for all containers. The display of ``sf.Series``, ``sf.Frame``, ``sf.Index``, and ``sf.IndexHierarchy`` all share a common implementation and design. A priority of that design is to always make explicit container classes and underlying array types.

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


As much time is spent visually exploring the contents of these containers, StaticFrame offers numerous display configuration options, all exposed through the ``sf.DisplayConfig`` class. For persistent changes, ``sf.DisplayConfig`` instances can be passed to ``sf.DisplayActive.set()``; for one-off changes, ``sf.DisplayConfig`` instances can be passed to the container's ``display()`` method.

While ``pd.set_option()`` can similarly be used to set Pandas display characteristics, StaticFrame provides more extensive options for making types discoverable. As shown in the following terminal animation, specific types can be colored or type annotations can be removed entirely.


.. image:: https://raw.githubusercontent.com/InvestmentSystems/static-frame/master/doc/images/animate-display-config.svg
   :align: center


No. 3: Immutable Data: Efficient Memory Management without Defensive Copies
___________________________________________________________________________________

Pandas displays inconsistent behavior in regard to ownership of data inputs and data exposed from within containers. In some cases, it is possible to mutate NumPy arrays "behind-the-back" of Pandas, exposing opportunities for undesirable side-effects and coding errors.

For example, if we supply a 2D array to a ``pd.DataFrame``, the original reference to the array can be used to "remotely" change the values within the ``pd.DataFrame``. In this case, the ``pd.DataFrame`` does not protect access to its data, serving only as a wrapper of a shared, mutable array.

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


Similarly, sometimes NumPy arrays exposed from the ``values`` attribute of a ``pd.Series`` or a ``pd.DataFrame`` can be mutated, changing the values within the ``pd.DataFrame``.

>>> a2 = df['charge'].values
>>> a2
array([-1., -1.])

>>> a2[1] = np.nan # Mutating the array from .values.

>>> df # Mutation is reflected in the DataFrame.
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



While immutable data reduces opportunities for error, it also offers performance advantages. For example, when replacing column labels with ``sf.Frame.relabel()``, underlying data is not copied. Instead, references to the same immutable arrays are shared between the old and new containers. Such "no-copy" operations are thus fast and light-weight. This is in contrast to what happens when doing the same thing in Pandas: the corresponding Pandas method, ``df.DataFrame.rename()``, is forced to make a defensive copy of all underlying data.

>>> f.relabel(columns=lambda x: x.upper()) # Underlying arrays are not copied
<Frame>
<Index> MASS      CHARGE    <<U6>
<Index>
muon    0.106     -1.0
tau     1.777     -1.0
<<U4>   <float64> <float64>




No. 4: Assignment is a Function
_____________________________________________________________


While Pandas permits in-place assignment, sometimes such operations cannot provide an appropriate derived type, resulting in undesirable behavior. For example, a float assigned into an integer ``pd.Series`` will have its floating-point components truncated without warning or error.

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

>>> s.assign['down'](-0.333) # The float is assigned without truncation
<Series>
<Index>
tau      -1.0
down     -0.333
<<U4>    <float64>


StaticFrame uses a special ``assign`` interface for performing assignment function calls. On a ``sf.Frame``, this interface exposes a ``sf.Frame.assign.loc[]`` interface that can be used to select the target of assignment. Following this selection, the value to be assigned is passed through a function call.


>>> f = sf.Frame.from_dict_records_items((('charm', {'charge':0.666, 'mass':1.3}), ('strange', {'charge':-0.333, 'mass':0.1})))
>>> f
<Frame>
<Index> charge    mass      <<U6>
<Index>
charm   0.666     1.3
strange -0.333    0.1
<<U7>   <float64> <float64>

>>> f.assign.loc['charm', 'charge'](Fraction(2, 3)) # Assigning to a loc-style selection
<Frame>
<Index> charge   mass      <<U6>
<Index>
charm   2/3      1.3
strange -0.333   0.1
<<U7>   <object> <float64>



No. 5: Iterators are for Iterating and Function Application
________________________________________________________________


Pandas has separate functions for iteration and function application. For iteration on a ``pd.DataFrame`` there is ``pd.DataFrame.iteritems()``, ``pd.DataFrame.iterrows()``, ``pd.DataFrame.itertuples()``, and ``pd.DataFrame.groupby()``; for function application on a ``pd.DataFrame`` there is ``pd.DataFrame.apply()`` and ``pd.DataFrame.applymap()``.

But since function application requires iteration, it is sensible for function application to be built on iteration. StaticFrame organizes iteration and function application by providing families of iterators (such as ``Frame.iter_array()`` or ``Frame.iter_group_items()``) that, with a chained call to ``apply()``, can also be used for function application. Functions for applying mapping types (such as ``map_any()`` and ``map_fill()``) are also available on iterators. This means that once you know how you want to iterate, function application is a just a method away.

For example, we can create a ``sf.Frame`` with ``sf.Frame.from_records()``:


>>> f = sf.Frame.from_records(((0.106, -1.0, 'lepton'), (1.777, -1.0, 'lepton'), (1.3, 0.666, 'quark'), (0.1, -0.333, 'quark')), columns=('mass', 'charge', 'type'), index=('muon', 'tau', 'charm', 'strange'))
>>> f
<Frame>
<Index> mass      charge    type   <<U6>
<Index>
muon    0.106     -1.0      lepton
tau     1.777     -1.0      lepton
charm   1.3       0.666     quark
strange 0.1       -0.333    quark


We can iterate over a columns values with ``sf.Series.iter_element()``. We can use the same iterator to do function application by using the ``apply()`` method found on the object returned from ``sf.Series.iter_element()``. The same interface is found on both ``sf.Series`` and ``sf.Frame``.

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

>>> f[['mass', 'charge']].iter_element().apply(lambda e: format(e, '.2e'))
<Frame>
<Index> mass     charge    <<U6>
<Index>
muon    1.06e-01 -1.00e+00
tau     1.78e+00 -1.00e+00
charm   1.30e+00 6.66e-01
strange 1.00e-01 -3.33e-01
<<U7>   <object> <object>


For row or column iteration on a ``sf.Frame``, a family of methods allows specifying the type of container to be used for the iterated rows or columns, i.e, with an array, with a ``NamedTuple``, or with a ``sf.Series`` (``iter_array()``, ``iter_tuple()``, ``iter_series()``, respectively). These methods take an axis argument to determine whether iteration is by row or by column, and similarly expose an ``apply()`` method for function application. To apply a function to columns, we can do the following.

>>> f[['mass', 'charge']].iter_array(axis=0).apply(np.sum)
<Series>
<Index>
mass     3.283
charge   -1.667
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

Group-by operations are just another form of iteration, with an identical interface for iteration and function application.

>>> f.iter_group('type').apply(lambda f: f['mass'].mean())
<Series>
<Index>
lepton   0.9415
quark    0.7000000000000001
<<U6>    <float64>



No. 6: Strict, Grow-Only Frames
_____________________________________________

An efficient use of a ``pd.DataFrame`` is to load initial data, then produce derived data by adding additional columns. This approach leverages the columnar organization of types and underlying arrays: adding new columns does not require re-allocating old columns.

StaticFrame makes this approach less vulnerable to error by offering a strict, grow-only version of a ``sf.Frame`` called a ``sf.FrameGO``. For example, once a ``sf.FrameGO`` is created, new columns can be added while existing columns cannot be overwritten or mutated in-place.


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


This limited form of mutation meets a practical need. Further, converting back and forth from a ``sf.Frame`` to a ``sf.FrameGO`` (using ``Frame.to_frame_go()`` and ``FrameGO.to_frame()``) is a no-copy operation: underlying immutable arrays can be shared between the two containers.



No. 7: Dates are not Nanoseconds
__________________________________________________________________

Pandas models all date or timestamp values as NumPy ``datetime64[ns]`` (nanosecond) arrays, regardless of if nanosecond-level resolution is practical or appropriate. This creates a "Y2262 problem" for Pandas: dates beyond 2262-04-11 cannot be expressed. While I can create a ``pd.DatetimeIndex`` up to 2262-04-11, one day further and Pandas raises an error.

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

While not possible with Pandas, creating an index of years or dates extending to 3000 is simple with StaticFrame.

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


No. 8: Consistent Interfaces for Hierarchical Indices
___________________________________________________________________________


Hierarchical indices permit fitting many dimensions into one. Using hierarchical indices, *n*-dimensional data can be encoded into a single ``sf.Series`` or ``sf.Frame``.

A key feature of hierarchical indices is partial selection at arbitrary depths, whereby a selection can be composed from the intersection of selections at each depth level. Pandas offers numerous ways to express those inner depth selections.

One way is by overloading ``pd.DataFrame.loc[]``. When using Pandas's hierarchical index (``pd.MultiIndex``), the meaning of positional arguments in a ``pd.DataFrame.loc[]`` selection becomes dynamic. It is this that makes Pandas code using hierarchical indices hard to maintain. We can see this by creating a ``pd.DataFrame`` and setting a ``pd.MultiIndex``.

>>> df = pd.DataFrame.from_records([('muon', 0.106, -1.0, 'lepton'), ('tau', 1.777, -1.0, 'lepton'), ('charm', 1.3, 0.666, 'quark'), ('strange', 0.1, -0.333, 'quark')], columns=('name', 'mass', 'charge', 'type'))

>>> df.set_index(['type', 'name'], inplace=True)
>>> df
                 mass  charge
type   name
lepton muon     0.106  -1.000
       tau      1.777  -1.000
quark  charm    1.300   0.666
       strange  0.100  -0.333

Similar to 2D arrays in NumPy, when two arguments are given to ``pd.DataFrame.loc[]``, the first argument is a row selector, the second argument is a column selector.

>>> df.loc['lepton', 'mass'] # Selects "lepton" from row, "mass" from columns
name
muon    0.106
tau     1.777
Name: mass, dtype: float64


Yet, in violation of that expectation, sometimes Pandas will not use the second argument as a column selection, but instead as a row selection in an inner depth of the ``pd.MultiIndex``.

>>> df.loc['lepton', 'tau'] # Selects lepton and tau from rows
mass      1.777
charge   -1.000
Name: (lepton, tau), dtype: float64


To handle this ambiguity, Pandas offers two alternatives. If a row and a column selection is required, the expected behavior can be restored by wrapping the hierarchical row selection within a ``pd.IndexSlice[]`` selection modifier. Or, if an inner-depth selection is desired without using a ``pd.IndexSlice[]``, the ``pd.DataFrame.xs()`` method can be used.

>>> df.loc[pd.IndexSlice['lepton', 'tau'], 'charge']
-1.0
>>> df.xs(level=1, key='tau')
         mass  charge
type
lepton  1.777    -1.0

This inconsistency in the meaning of the positional arguments given to ``pd.DataFrame.loc[]`` is unnecessary and makes Pandas code harder to maintain: what is intended from the usage of ``pd.DataFrame.loc[]`` becomes ambiguous without a ``pd.IndexSlice[]``. Further, providing multiple ways to solve this problem is also a shortcoming, as it is preferable to have one obvious way to do things in Python.

StaticFrame's ``sf.IndexHierarchy`` offers more consistent behavior. We will create an equivalent ``sf.Frame`` and set a ``sf.IndexHierarchy``.


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


Unlike Pandas, StaticFrame is consistent in what positional ``sf.Frame.loc[]`` arguments mean: the first argument is always a row selector, the second argument is always a column selector. For selection within a ``sf.IndexHierarchy``, the ``sf.HLoc[]`` selection modifier is required to specify selection at arbitrary depths within the hierarchy. There is one obvious way to select inner depths. This approach makes StaticFrame code easier to understand and maintain.

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

It is natural to think index and column labels on a ``pd.DataFrame`` are unique identifiers: their interfaces suggest that they are like Python dictionaries, where keys are always unique. Pandas indices, however, are not constrained to unique values. Creating an index on a ``pd.DataFrame`` with duplicates means that, for some single-label selections, a ``pd.Series`` will be returned, but for other single-label selections, a ``pd.DataFrame`` will be returned.


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

Some Pandas interfaces, such as ``pd.concat()`` and ``pd.DataFrame.set_index()``, provide an optional check of uniqueness with a parameter named ``verify_integrity``. Surprisingly, by default Pandas disables ``verify_integrity``.


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



The concept of a "data frame" object came long before Pandas 0.1 release in 2009: the first implementation of a data frame may have been as early as 1991 in the S language, a predecessor of R. Today, the data frame finds realization in a wide variety of languages and implementations. Pandas will continue to provide an excellent resource to a broad community of users. However, for situations where correctness and code maintainability are critical, StaticFrame offers an alternative designed to be more consistent and reduce opportunities for error.

For more information about StaticFrame, see the documentation (http://static-frame.readthedocs.io) or project site (https://github.com/InvestmentSystems/static-frame).

