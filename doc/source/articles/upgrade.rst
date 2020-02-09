


Ten Tips for Transitioning from Pandas to StaticFrame
=============================================================

Now that Pandas 1.0 is out, it is the perfect time consider upgrading to an alternative that offers a more consistent interface and reduces opportunities for error: StaticFrame.


Why StaticFrame
______________________

After working with Pandas for years to develop production, back-end financial systems, it became clear to me that Pandas was not the right tool for that work. Sure, Pandas speed was similar to NumPy. And yet, Pandas interface inconsistency made Pandas code hard to read and maintain. In addition, Pandas inconsistent approach to data ownership, and support for mutation and undesirable side effects, led to serious vulnerabilities and opportunities for error. So, in May of 2017, I began implementation of a library more suitable for critical production systems.

Now, after a few years of development and refinement, we are observing excellent results replacing Pandas with StaticFrame in our production systems. As we approach a 1.0 release, we are looking for feedback. While we have a strong emphasis on test, with 99% coverage and batteries of property tests, real-world feedback is always valuable. Please report issues, feature requests, or discussion items on GitHub (https://github.com/InvestmentSystems/static-frame).

What follows are ten tips to aid in the transition from Pandas to StaticFrame. While many features are the same (e.g., there is a ``Frame`` and a ``Series`` that have ``loc`` and ``iloc`` selectors), much is different.


No. 1: Consistent Interfaces
______________________________________


An interface can be consistent in where functions are located, how functions are named, and the name and types of arguments those functions accept. StaticFrame deviates from the Pandas API in many ways to support greater consistency in all of these characteristics.


Pandas places its ``DataFrame`` constructors in at least two places: on the root namespace (`pd`, as commonly imported) and (as is more conventional) on the ``DataFrame`` class:


>>> pd.read_json('[{"name":"muon", "mass":0.106},{"name":"tau", "mass":1.777}]')
    mass  name
0  0.106  muon
1  1.777   tau

>>> pd.DataFrame.from_records([{"name":"muon", "mass":0.106}, {"name":"tau", "mass":1.777}])
    mass  name
0  0.106  muon
1  1.777   tau


As there is no benefit to this diversity, StaticFrame places all constructors on the class they construct.


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




Specialized, Explicit Constructors

Pandas default constructors accept a staggering diversity of inputs.


>>> pd.Series(np.nan, index=('up', 'charm', 'top'))
up      NaN
charm   NaN
top     NaN
dtype: float64

>>> pd.Series({'up': 0.002, 'charm': 1.3, 'top': 173})
up         0.002
charm      1.300
top      173.000
dtype: float64



>>> sf.Series.from_element(np.nan, index=('up', 'charm', 'top'))
<Series>
<Index>
up       nan
charm    nan
top      nan
<<U5>    <float64>

>>> sf.Series.from_dict({'up': 0.002, 'charm': 1.3, 'top': 173})
<Series>
<Index>
up       0.002
charm    1.3
top      173.0
<<U5>    <float64>



Having explicit constructors leads to lots of constructors. To help discover the constructors you are looking for, StaticFrame containers expose an ``interface`` attribute that lists the entire public interface of the class or instance.


>>> sf.Series.interface.shape
(264, 3)


>>> sf.Series.interface.loc[sf.Series.interface['group'] == 'Constructor']
<Frame: Series>
<Index>             cls    group       doc                  <<U5>
<Index: name>
__init__()          Series Constructor
from_concat()       Series Constructor Concatenate multi...
from_concat_items() Series Constructor Produce a Series ...
from_dict()         Series Constructor Series constructi...
from_element()      Series Constructor
from_items()        Series Constructor Series constructi...
from_pandas()       Series Constructor Given a Pandas Se...
<<U51>              <<U6>  <<U15>      <<U53>





No. 2: Consistent and Colorful Display
___________________________________________


Pandas default display is inconsistent. For example, ``pd.Series`` are shown with their name and type, while ``pd.DataFrame`` do not show their name and type. Further, if you display a ``pd.Index``, you get an entirely different display. In the case of ``pd.MultiIndex``, the display is often unmanageable.

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


Pandas displays inconsistent behavior in regard to ownership of data inputs.


We can mutate NumPy arrays "behind-the-back" of Pandas. We can do that arrays given as input, and we can sometimes do it with arrays given back to us from the `values` attribute.

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



>>> a2 = df['charge'].values
>>> a2
array([-1., -1.])
>>> a2[1] = np.nan


>>> df
       mass  charge
muon    NaN    -1.0
tau   1.777     NaN






With StaticFrame, mutation is never allowed, either via StaticFrame containers, or via direct access to underlying arrays.


>>> f = sf.Frame.from_dict_records_items((('charm', {'symbol':'c', 'mass':1.3}), ('strange', {'symbol':'s', 'mass':0.1})))


>>> f.loc['charm', 'mass'] = np.nan
Traceback (most recent call last):
  File "<console>", line 1, in <module>
TypeError: 'InterfaceGetItem' object does not support item assignment

>>> f['mass'].values[1] = 100
Traceback (most recent call last):
  File "<console>", line 1, in <module>
ValueError: assignment destination is read-only


Renaming, or relabeling, or similar operations do not have to copy underlying data, and are thus fast, light-weight operations.

>>> f.rename('fermion')
<Frame: fermion>
<Index>          symbol mass      <<U6>
<Index>
charm            c      1.3
strange          s      0.1
<<U7>            <<U1>  <float64>



Horizontal (axis 1) concatenation, if indices align, can be done without copying data.


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


While Pandas permits arbitrary assignment, it does not manage the types of mutated arrays, resulting in some undesirable bahavior, such as assigning a float into an integer `pd.Series`.

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



With StaticFrame, assignment is a function that returns a new object, and evaluates types to insure that the resultant array can contain the assigned value.


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



Assignment on a ``Frame`` works the same way. Yet, as data structure that contains heterogeneous types of columnar data, assignment only mutates what needs to change, reusing unchanged columns without copying data.


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

Pandas has separate functions for iterating and function application, even though function application requires iteration.

For example, Pandas has ``DataFrame.iteritems``, ``DataFrame.iterrows``, ``DataFrame.itertuples``, ``DataFrame.groupby`` for iteration, and ``DataFrame.apply`` and ``DataFrame.applymap`` for function application.

StaticFrame avoids this complexity by exposing, on all iterators, ``apply`` (for functions) and various functions for using mapping types (such as ``map_any`` and ``map_fill``).




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



So we can iterate over elements in a ``Series`` with ``iter_element()``.

>>> tuple(f['type'].iter_element())
('lepton', 'lepton', 'quark', 'quark')


We can reuse the same iterator to do function application, simply by using the ``apply`` method.

>>> f['type'].iter_element().apply(lambda e: e.upper())
<Series>
<Index>
muon     LEPTON
tau      LEPTON
charm    QUARK
strange  QUARK
<<U7>    <<U6>





The same design is applied to ``Frame``.


>>> f.iter_element().apply(lambda e: str(e).rjust(8, '_'))
<Frame>
<Index>       mass     charge   type     <<U6>
<Index: name>
muon          ___0.106 ____-1.0 __lepton
tau           ___1.777 ____-1.0 __lepton
charm         _____1.3 ___0.666 ___quark
strange       _____0.1 __-0.333 ___quark
<<U7>         <object> <object> <object>



For axis (row or column) iterators, we supply an axis argument to determine the inputs into the function. We can choose how to represent the axis values, either as an array, a ``NamedTuple``, or a ``Series``.

For example, to apply a function to columns, we can do the following.

>>> f[['mass', 'charge']].iter_array(axis=0).apply(np.sum)
<Series>
<Index>
mass     3.283
charge   -1.667
<<U6>    <float64>


If we need key, value pairs for each function application, we can use the corresponding iterator that returns items pairs.

>>> f.iter_array_items(axis=0).apply(lambda k, v: v.sum() if k != 'type' else np.nan)
<Series>
<Index>
mass     3.283
charge   -1.667
type     nan
<<U6>    <float64>


To apply a function to each row, we can do the following.

>>> f.iter_series(axis=1).apply(lambda s: s['mass'] > 1 and s['type'] == 'quark')
<Series>
<Index>
muon     False
tau      False
charm    True
strange  False
<<U7>    <bool>


Group iteration works exactly the same way.

>>> f.iter_group('type').apply(lambda f: f['mass'].mean())
<Series>
<Index>
lepton   0.9415
quark    0.7000000000000001
<<U6>    <float64>
>>>




No. 6: Strict, Grow-Only Frames
_____________________________________________

A common use of ``pd.DataFrame`` is to load initial data, then produce derived data by adding additional columns. ``StaticFrame`` makes this approach less vulnerable to error by using strict, grow-only tables called ``FrameGO``.


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





No 7: Typed Datetime Indices: Everything is not a Nanosecond
__________________________________________________________________

Pandas models every date or timestamp as a NumPy nanosecond ``datetime64`` object, regardless if nanosecond resolution is needed or practical. This has the amusing side effect of creating a "Y2262 problem": not permitting dates beyond 2262-04-11.


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



As date/time indices are often used for things much larger than nanoseconds, such as years and dates, StaticFrame offers fixed diverse, typed datetime indices. This permits more explicit usage, and avoids the "Y2262 problem".


>>> sf.IndexYear.from_year_range(1980, 3000)
<IndexYear>
1980
1981
1982
1983
1984
1985
1986
1987
1988
1989
1990
1991
1992
1993
1994
1995
...
2985
2986
2987
2988
2989
2990
2991
2992
2993
2994
2995
2996
2997
2998
2999
3000
<datetime64[Y]>



>>> sf.IndexDate.from_year_range(1980, 3000)
<IndexDate>
1980-01-01
1980-01-02
1980-01-03
1980-01-04
1980-01-05
1980-01-06
1980-01-07
1980-01-08
1980-01-09
1980-01-10
1980-01-11
1980-01-12
1980-01-13
1980-01-14
1980-01-15
1980-01-16
...
3000-12-16
3000-12-17
3000-12-18
3000-12-19
3000-12-20
3000-12-21
3000-12-22
3000-12-23
3000-12-24
3000-12-25
3000-12-26
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





No. 10: There and Back Again to Pandas
_____________________________

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



>>> s = sf.Series.from_dict({'up': 0.002, 'charm': 1.3, 'top': 173})
>>> s
<Series>
<Index>
up       0.002
charm    1.3
top      173.0
<<U5>    <float64>
>>> s.to_pandas()
up         0.002
charm      1.300
top      173.000
dtype: float64


