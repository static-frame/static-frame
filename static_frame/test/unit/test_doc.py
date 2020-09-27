import doctest
import os
import typing as tp


# useful constructors
# >>> f = sf.FrameGO.from_records((('Encke', 3.30, '2003-12-28'), ('Giacobini-Zinner', 6.52, '1998-11-21'), ('Tempel-Tuttle', 32.92, '1998-02-28'), ('Wild 2', 6.39, '2003-09-25')), columns=('Name', 'Orbital Period', 'Perihelion Date'))




api_example_str = '''


#-------------------------------------------------------------------------------
# import and setup

>>> import static_frame as sf
>>> _display_config_active = sf.DisplayActive.get()
>>> sf.DisplayActive.set(sf.DisplayConfig(type_color=False))
>>> import numpy as np
>>> import static_frame as sf

#-------------------------------------------------------------------------------
# documentation introduction

#start_immutability

>>> import static_frame as sf
>>> import numpy as np

>>> s = sf.Series((67, 62, 27, 14), index=('Jupiter', 'Saturn', 'Uranus', 'Neptune'), dtype=np.int64)
>>> s #doctest: +NORMALIZE_WHITESPACE
<Series>
<Index>
Jupiter  67
Saturn   62
Uranus   27
Neptune  14
<<U7>    <int64>
>>> s['Jupiter'] = 68
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'Series' object does not support item assignment
>>> s.iloc[0] = 68
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'InterfaceGetItem' object does not support item assignment
>>> s.values[0] = 68
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: assignment destination is read-only

#end_immutability

#start_assign
>>> s.assign['Jupiter'](69) #doctest: +NORMALIZE_WHITESPACE
<Series>
<Index>
Jupiter  69
Saturn   62
Uranus   27
Neptune  14
<<U7>    <int64>
>>> s.assign['Uranus':](s['Uranus':] - 2) #doctest: +NORMALIZE_WHITESPACE
<Series>
<Index>
Jupiter  67
Saturn   62
Uranus   25
Neptune  12
<<U7>   <int64>
>>> s.assign.iloc[[0, 3]]((68, 11)) #doctest: +NORMALIZE_WHITESPACE
<Series>
<Index>
Jupiter  68
Saturn   62
Uranus   27
Neptune  11
<<U7>    <int64>

#end_assign


#-------------------------------------------------------------------------------
# article: boring indices

#start_aiii_fig01
>>> s1 = sf.Series((x * 100 for x in range(1, 5)), index=tuple('wxyz'))
>>> s1
<Series>
<Index>
w        100
x        200
y        300
z        400
<<U1>    <int64>

>>> s1.reindex(tuple('stwx'), fill_value=0)
<Series>
<Index>
s        0
t        0
w        100
x        200
<<U1>    <int64>

#end_aiii_fig01


#start_aiii_fig02
>>> s1.relabel(tuple('abcd'))
<Series>
<Index>
a        100
b        200
c        300
d        400
<<U1>    <int64>

#end_aiii_fig02


#start_aiii_fig03
>>> s1.relabel(sf.IndexAutoFactory)
<Series>
<Index>
0        100
1        200
2        300
3        400
<int64>  <int64>

#end_aiii_fig03




#start_aiii_fig04
>>> f1 = sf.Frame.from_dict(dict(a=(1,2), b=(True, False)), index=tuple('xy'))
>>> f1
<Frame>
<Index> a       b      <<U1>
<Index>
x       1       True
y       2       False
<<U1>   <int64> <bool>

>>> f1.relabel(index=sf.IndexAutoFactory, columns=sf.IndexAutoFactory)
<Frame>
<Index> 0       1      <int64>
<Index>
0       1       True
1       2       False
<int64> <int64> <bool>

>>> f1.relabel(index=tuple('ab'), columns=sf.IndexAutoFactory)
<Frame>
<Index> 0       1      <int64>
<Index>
a       1       True
b       2       False
<<U1>   <int64> <bool>

#end_aiii_fig04


#start_aiii_fig05
>>> s1
<Series>
<Index>
w        100
x        200
y        300
z        400
<<U1>    <int64>

>>> sf.Series.from_concat((s1, s1), index=tuple('abcdefgh'))
<Series>
<Index>
a        100
b        200
c        300
d        400
e        100
f        200
g        300
h        400
<<U1>    <int64>

#end_aiii_fig05



#start_aiii_fig06
>>> sf.Series.from_concat((s1, s1), index=sf.IndexAutoFactory)
<Series>
<Index>
0        100
1        200
2        300
3        400
4        100
5        200
6        300
7        400
<int64>  <int64>

#end_aiii_fig06



#start_aiii_fig07
>>> s2 = s1 * .5
>>> sf.Frame.from_concat((s1, s2), axis=1, columns=sf.IndexAutoFactory)
<Frame>
<Index> 0       1         <int64>
<Index>
w       100     50.0
x       200     100.0
y       300     150.0
z       400     200.0
<<U1>   <int64> <float64>

#end_aiii_fig07



#start_aiii_fig08
>>> sf.Frame.from_concat((f1, f1), axis=1, columns=sf.IndexAutoFactory)
<Frame>
<Index> 0       1      2       3      <int64>
<Index>
x       1       True   1       True
y       2       False  2       False
<<U1>   <int64> <bool> <int64> <bool>

#end_aiii_fig08




#-------------------------------------------------------------------------------
# series

#start_Series-via_dt.year
>>> s = sf.Series.from_dict({'Halley': '1986-02-09', 'Encke': '2003-12-28', "d'Arrest": '2008-08-01', 'Tempel 1': '2005-07-05'}, name='Perihelion Date', dtype=np.datetime64)

>>> s
<Series: Perihelion Date>
<Index>
Halley                    1986-02-09
Encke                     2003-12-28
d'Arrest                  2008-08-01
Tempel 1                  2005-07-05
<<U8>                     <datetime64[D]>
>>> s.via_dt.year
<Series: Perihelion Date>
<Index>
Halley                    1986
Encke                     2003
d'Arrest                  2008
Tempel 1                  2005
<<U8>                     <int64>

#end_Series-via_dt.year


#start_Series-via_dt.month
>>> s = sf.Series.from_dict({'Halley': '1986-02-09', 'Encke': '2003-12-28', "d'Arrest": '2008-08-01', 'Tempel 1': '2005-07-05'}, name='Perihelion Date', dtype=np.datetime64)

>>> s
<Series: Perihelion Date>
<Index>
Halley                    1986-02-09
Encke                     2003-12-28
d'Arrest                  2008-08-01
Tempel 1                  2005-07-05
<<U8>                     <datetime64[D]>
>>> s.via_dt.month
<Series: Perihelion Date>
<Index>
Halley                    2
Encke                     12
d'Arrest                  8
Tempel 1                  7
<<U8>                     <int64>

#end_Series-via_dt.month


#start_Series-via_dt.day
>>> s = sf.Series.from_dict({'Halley': '1986-02-09', 'Encke': '2003-12-28', "d'Arrest": '2008-08-01', 'Tempel 1': '2005-07-05'}, name='Perihelion Date', dtype=np.datetime64)

>>> s
<Series: Perihelion Date>
<Index>
Halley                    1986-02-09
Encke                     2003-12-28
d'Arrest                  2008-08-01
Tempel 1                  2005-07-05
<<U8>                     <datetime64[D]>
>>> s.via_dt.day
<Series: Perihelion Date>
<Index>
Halley                    9
Encke                     28
d'Arrest                  1
Tempel 1                  5
<<U8>                     <int64>

#end_Series-via_dt.day

#start_Series-via_dt.weekday()
>>> s = sf.Series.from_dict({'Halley': '1986-02-09', 'Encke': '2003-12-28', "d'Arrest": '2008-08-01', 'Tempel 1': '2005-07-05'}, name='Perihelion Date', dtype=np.datetime64)
>>> s
<Series: Perihelion Date>
<Index>
Halley                    1986-02-09
Encke                     2003-12-28
d'Arrest                  2008-08-01
Tempel 1                  2005-07-05
<<U8>                     <datetime64[D]>
>>> s.via_dt.weekday()
<Series: Perihelion Date>
<Index>
Halley                    6
Encke                     6
d'Arrest                  4
Tempel 1                  1
<<U8>                     <int64>

#end_Series-via_dt.weekday()



#start_Series-via_dt.isoformat()
>>> s = sf.Series.from_dict({'Halley': '1986-02-09', 'Encke': '2003-12-28', "d'Arrest": '2008-08-01', 'Tempel 1': '2005-07-05'}, name='Perihelion Date', dtype=np.datetime64)
>>> s
<Series: Perihelion Date>
<Index>
Halley                    1986-02-09
Encke                     2003-12-28
d'Arrest                  2008-08-01
Tempel 1                  2005-07-05
<<U8>                     <datetime64[D]>
>>> s.via_dt.isoformat()
<Series: Perihelion Date>
<Index>
Halley                    1986-02-09
Encke                     2003-12-28
d'Arrest                  2008-08-01
Tempel 1                  2005-07-05
<<U8>                     <<U10>

#end_Series-via_dt.isoformat()


#start_Series-via_dt.strftime()
>>> s = sf.Series.from_dict({'Halley': '1986-02-09', 'Encke': '2003-12-28', "d'Arrest": '2008-08-01', 'Tempel 1': '2005-07-05'}, name='Perihelion Date', dtype=np.datetime64)
>>> s
<Series: Perihelion Date>
<Index>
Halley                    1986-02-09
Encke                     2003-12-28
d'Arrest                  2008-08-01
Tempel 1                  2005-07-05
<<U8>                     <datetime64[D]>
>>> s.via_dt.strftime('%m/%d/%y')
<Series: Perihelion Date>
<Index>
Halley                    02/09/86
Encke                     12/28/03
d'Arrest                  08/01/08
Tempel 1                  07/05/05
<<U8>                     <<U8>

#end_Series-via_dt.strftime()





#start_Series-via_str.capitalize()
>>> s = sf.Series(('lepton', 'lepton', 'quark'), index=('muon', 'tau', 'strange'))
>>> s.via_str.capitalize()
<Series>
<Index>
muon     Lepton
tau      Lepton
strange  Quark
<<U7>    <<U6>

#end_Series-via_str.capitalize()


#start_Series-via_str.center()
>>> s = sf.Series(('lepton', 'lepton', 'quark'), index=('muon', 'tau', 'strange'))
>>> s.via_str.center(20, '-')
<Series>
<Index>
muon     -------lepton-------
tau      -------lepton-------
strange  -------quark--------
<<U7>    <<U20>

#end_Series-via_str.center()


#start_Series-via_str.endswith()
>>> s = sf.Series(('lepton', 'lepton', 'quark'), index=('muon', 'tau', 'strange'))
>>> s.via_str.endswith('ton')
<Series>
<Index>
muon     True
tau      True
strange  False
<<U7>    <bool>

#end_Series-via_str.endswith()


#start_Series-via_str.isdigit()
>>> s = sf.Series(('lepton', 'lepton', 'quark'), index=('muon', 'tau', 'strange'))
>>> s.via_str.isdigit()
<Series>
<Index>
muon     False
tau      False
strange  False
<<U7>    <bool>

#end_Series-via_str.isdigit()


#start_Series-via_str.islower()
>>> s = sf.Series(('lepton', 'lepton', 'quark'), index=('muon', 'tau', 'strange'))
>>> s.via_str.islower()
<Series>
<Index>
muon     True
tau      True
strange  True
<<U7>    <bool>

#end_Series-via_str.islower()


#start_Series-via_str.ljust()
>>> s = sf.Series(('lepton', 'lepton', 'quark'), index=('muon', 'tau', 'strange'))
>>> s.via_str.ljust(10, '-')
<Series>
<Index>
muon     lepton----
tau      lepton----
strange  quark-----
<<U7>    <<U10>

#end_Series-via_str.ljust()


#start_Series-via_str.isupper()
>>> s = sf.Series(('lepton', 'lepton', 'quark'), index=('muon', 'tau', 'strange'))
>>> s.via_str.isupper()
<Series>
<Index>
muon     False
tau      False
strange  False
<<U7>    <bool>

#end_Series-via_str.isupper()


#start_Series-via_str.rjust()
>>> s = sf.Series(('lepton', 'lepton', 'quark'), index=('muon', 'tau', 'strange'))
>>> s.via_str.rjust(10, '-')
<Series>
<Index>
muon     ----lepton
tau      ----lepton
strange  -----quark
<<U7>    <<U10>

#end_Series-via_str.rjust()


#start_Series-via_str.startswith()
>>> s = sf.Series(('lepton', 'lepton', 'quark'), index=('muon', 'tau', 'strange'))
>>> s.via_str.startswith('lep')
<Series>
<Index>
muon     True
tau      True
strange  False
<<U7>    <bool>

#end_Series-via_str.startswith()


#start_Series-via_str.title()
>>> s.via_str.title()
<Series>
<Index>
muon     Lepton
tau      Lepton
strange  Quark
<<U7>    <<U6>

#end_Series-via_str.title()



#start_Series-via_str.upper()
>>> s = sf.Series(('lepton', 'lepton', 'quark'), index=('muon', 'tau', 'strange'))
>>> s.via_str.upper()
<Series>
<Index>
muon     LEPTON
tau      LEPTON
strange  QUARK
<<U7>    <<U6>

#end_Series-via_str.upper()




#start_Series-from_dict()
>>> sf.Series.from_dict(dict(Mercury=167, Neptune=-200), dtype=np.int64)
<Series>
<Index>
Mercury  167
Neptune  -200
<<U7>    <int64>

#end_Series-from_dict()


#start_Series-__init__()
>>> sf.Series((167, -200), index=('Mercury', 'Neptune'), dtype=np.int64)
<Series>
<Index>
Mercury  167
Neptune  -200
<<U7>    <int64>

#end_Series-__init__()


#start_Series-from_items()
>>> sf.Series.from_items(zip(('Mercury', 'Jupiter'), (4879, 12756)), dtype=np.int64)
<Series>
<Index>
Mercury  4879
Jupiter  12756
<<U7>    <int64>

#end_Series-from_items()


#start_Series-from_element()
>>> sf.Series.from_element('lepton', index=('electron', 'muon', 'tau'))
<Series>
<Index>
electron lepton
muon     lepton
tau      lepton
<<U8>    <<U6>

#end_Series-from_element()



#start_Series-items()
>>> s = sf.Series((1, 2, 67, 62, 27, 14), index=('Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'), dtype=np.int64)
>>> s
<Series>
<Index>
Earth    1
Mars     2
Jupiter  67
Saturn   62
Uranus   27
Neptune  14
<<U7>    <int64>
>>> [k for k, v in s.items() if v > 60]
['Jupiter', 'Saturn']

#end_Series-items()


#start_Series-get()
>>> s = sf.Series((1, 2, 67, 62, 27, 14), index=('Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'), dtype=np.int64)
>>> [s.get(k, None) for k in ('Mercury', 'Neptune', 'Pluto')]
[None, 14, None]

#end_Series-get()


#start_Series-__len__()
>>> s = sf.Series((1, 2, 67, 62, 27, 14), index=('Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'), dtype=np.int64)
>>> len(s)
6

#end_Series-__len__()


#start_Series-__sub__()
>>> s = sf.Series.from_items((('Venus', 108.2), ('Earth', 149.6), ('Saturn', 1433.5)))
>>> s
<Series>
<Index>
Venus    108.2
Earth    149.6
Saturn   1433.5
<<U6>    <float64>

>>> abs(s - s['Earth'])
<Series>
<Index>
Venus    41.39999999999999
Earth    0.0
Saturn   1283.9
<<U6>    <float64>

#end_Series-__sub__()


#start_Series-__gt__()
>>> s = sf.Series.from_items((('Venus', 108.2), ('Earth', 149.6), ('Saturn', 1433.5)))
>>> s > s['Earth']
<Series>
<Index>
Venus    False
Earth    False
Saturn   True
<<U6>    <bool>

#end_Series-__gt__()


#start_Series-__truediv__()
>>> s = sf.Series.from_items((('Venus', 108.2), ('Earth', 149.6), ('Saturn', 1433.5)))

>>> s / s['Earth']
<Series>
<Index>
Venus    0.7232620320855615
Earth    1.0
Saturn   9.582219251336898
<<U6>    <float64>

#end_Series-__truediv__()


#start_Series-__mul__()
>>> s1 = sf.Series((1, 2), index=('Earth', 'Mars'))
>>> s2 = sf.Series((2, 0), index=('Mars', 'Mercury'))
>>> s1 * s2
<Series>
<Index>
Earth    nan
Mars     4.0
Mercury  nan
<<U7>    <float64>

#end_Series-__mul__()


#start_Series-__eq__()
>>> s1 = sf.Series((1, 2), index=('Earth', 'Mars'))
>>> s2 = sf.Series((2, 0), index=('Mars', 'Mercury'))

>>> s1 == s2
<Series>
<Index>
Earth    False
Mars     True
Mercury  False
<<U7>    <bool>

#end_Series-__eq__()


#start_Series-relabel()
>>> s = sf.Series((0, 62, 13), index=('Venus', 'Saturn', 'Neptune'), dtype=np.int64)

>>> s.relabel({'Venus': 'Mercury'})
<Series>
<Index>
Mercury  0
Saturn   62
Neptune  13
<<U7>    <int64>

>>> s.relabel(lambda x: x[:2].upper())
<Series>
<Index>
VE       0
SA       62
NE       13
<<U2>    <int64>

#end_Series-relabel()


#start_Series-reindex()
>>> s = sf.Series((0, 62, 13), index=('Venus', 'Saturn', 'Neptune'))

>>> s.reindex(('Venus', 'Earth', 'Mars', 'Neptune'))
<Series>
<Index>
Venus    0.0
Earth    nan
Mars     nan
Neptune  13.0
<<U7>    <float64>

#end_Series-reindex()


#start_Series-shape
>>> s = sf.Series((1, 2, 67, 62, 27, 14), index=('Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'), dtype=np.int64)
>>> s
<Series>
<Index>
Earth    1
Mars     2
Jupiter  67
Saturn   62
Uranus   27
Neptune  14
<<U7>    <int64>
>>> s.shape
(6,)

#end_Series-shape


#start_Series-ndim
>>> s = sf.Series((1, 2, 67, 62, 27, 14), index=('Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'), dtype=np.int64)
>>> s.ndim
1

#end_Series-ndim


#start_Series-size
>>> s = sf.Series((1, 2, 67, 62, 27, 14), index=('Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'), dtype=np.int64)
>>> s.size
6

#end_Series-size


#start_Series-nbytes
>>> s = sf.Series((1, 2, 67, 62, 27, 14), index=('Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'), dtype=np.int64)
>>> s.nbytes
48

#end_Series-nbytes


#start_Series-dtype
>>> s = sf.Series((1, 2, 67, 62, 27, 14), index=('Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'), dtype=np.int64)
>>> s.dtype
dtype('int64')

#end_Series-dtype




#start_Series-interface
>>> sf.Series.interface.loc[sf.Series.interface.index.via_str.startswith('sort')]
<Frame: Series>
<Index>                         cls_name group  doc                  <<U18>
<Index: signature>
sort_index(*, ascending, kind)  Series   Method Return a new Seri...
sort_values(*, ascending, kind) Series   Method Return a new Seri...
<<U94>                          <<U6>    <<U17> <<U83>

#end_Series-interface


#start_Series-iter_element()
>>> s = sf.Series((1, 2, 67, 62, 27, 14), index=('Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'))
>>> [x for x in s.iter_element()]
[1, 2, 67, 62, 27, 14]

#end_Series-iter_element()


#start_Series-iter_element().apply()
>>> s = sf.Series((1, 2, 67, 62, 27, 14), index=('Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'))
>>> s.iter_element().apply(lambda v: v > 20)
<Series>
<Index>
Earth    False
Mars     False
Jupiter  True
Saturn   True
Uranus   True
Neptune  False
<<U7>    <bool>

#end_Series-iter_element().apply()


#start_Series-iter_element().apply_iter()
>>> s = sf.Series((1, 2, 67, 62, 27, 14), index=('Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'))
>>> [x for x in s.iter_element().apply_iter(lambda v: v > 20)]
[False, False, True, True, True, False]

#end_Series-iter_element().apply_iter()


#start_Series-iter_element_items()
>>> s = sf.Series((1, 2, 67, 62, 27, 14), index=('Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'))
>>> [x for x in s.iter_element_items()]
[('Earth', 1), ('Mars', 2), ('Jupiter', 67), ('Saturn', 62), ('Uranus', 27), ('Neptune', 14)]

#end_Series-iter_element_items()


#start_Series-iter_element_items().apply()
>>> s = sf.Series((1, 2, 67, 62, 27, 14), index=('Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'))
>>> s.iter_element_items().apply(lambda k, v: v if 'u' in k else None)
<Series>
<Index>
Earth    None
Mars     None
Jupiter  67
Saturn   62
Uranus   27
Neptune  14
<<U7>    <object>

#end_Series-iter_element_items().apply()


#start_Series-iter_element_items().apply_iter_items()
>>> s = sf.Series((1, 2, 67, 62, 27, 14), index=('Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'))

>>> [x for x in s.iter_element_items().apply_iter_items(lambda k, v: k.upper() if v > 20 else None)]
[('Earth', None), ('Mars', None), ('Jupiter', 'JUPITER'), ('Saturn', 'SATURN'), ('Uranus', 'URANUS'), ('Neptune', None)]


#end_Series-iter_element_items().apply_iter_items()


#start_Series-iter_group()
>>> s = sf.Series((0, 0, 1, 2), index=('Mercury', 'Venus', 'Earth', 'Mars'), dtype=np.int64)
>>> next(iter(s.iter_group()))
<Series>
<Index>
Mercury  0
Venus    0
<<U7>    <int64>
>>> [x.values.tolist() for x in s.iter_group()]
[[0, 0], [1], [2]]

#end_Series-iter_group()


#start_Series-iter_group_items()
>>> s = sf.Series((0, 0, 1, 2), index=('Mercury', 'Venus', 'Earth', 'Mars'))
>>> [(k, v.index.values.tolist()) for k, v in iter(s.iter_group_items()) if k > 0]
[(1, ['Earth']), (2, ['Mars'])]

#end_Series-iter_group_items()


#start_Series-assign[]()
>>> s = sf.Series.from_items((('Venus', 108.2), ('Earth', 149.6), ('Saturn', 1433.5)))
>>> s
<Series>
<Index>
Venus    108.2
Earth    149.6
Saturn   1433.5
<<U6>    <float64>
>>> s.assign['Earth'](150)
<Series>
<Index>
Venus    108.2
Earth    150.0
Saturn   1433.5
<<U6>    <float64>
>>> s.assign['Earth':](0)
<Series>
<Index>
Venus    108.2
Earth    0.0
Saturn   0.0
<<U6>    <float64>

#end_Series-assign[]()


#start_Series-assign.loc[]()
>>> s = sf.Series.from_items((('Venus', 108.2), ('Earth', 149.6), ('Saturn', 1433.5)))
>>> s.assign.loc[s < 150](0)
<Series>
<Index>
Venus    0.0
Earth    0.0
Saturn   1433.5
<<U6>    <float64>

#end_Series-assign.loc[]()


#start_Series-assign.iloc[]()
>>> s = sf.Series.from_items((('Venus', 108.2), ('Earth', 149.6), ('Saturn', 1433.5)))
>>> s.assign.iloc[-1](0)
<Series>
<Index>
Venus    108.2
Earth    149.6
Saturn   0.0
<<U6>    <float64>

#end_Series-assign.iloc[]()


#start_Series-drop[]
>>> s = sf.Series((0, 0, 1, 2), index=('Mercury', 'Venus', 'Earth', 'Mars'), dtype=np.int64)
>>> s
<Series>
<Index>
Mercury  0
Venus    0
Earth    1
Mars     2
<<U7>    <int64>
>>> s.drop[s < 1]
<Series>
<Index>
Earth    1
Mars     2
<<U7>    <int64>
>>> s.drop[['Mercury', 'Mars']]
<Series>
<Index>
Venus    0
Earth    1
<<U7>    <int64>

#end_Series-drop[]


#start_Series-drop.iloc[]
>>> s = sf.Series((0, 0, 1, 2), index=('Mercury', 'Venus', 'Earth', 'Mars'), dtype=np.int64)
>>> s.drop.iloc[-2:]
<Series>
<Index>
Mercury  0
Venus    0
<<U7>    <int64>

#end_Series-drop.iloc[]


#start_Series-[]
>>> s = sf.Series((1, 2, 67, 62, 27, 14), index=('Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'), dtype=np.int64)
>>> s
<Series>
<Index>
Earth    1
Mars     2
Jupiter  67
Saturn   62
Uranus   27
Neptune  14
<<U7>    <int64>

>>> s['Mars']
2
>>> s['Mars':]
<Series>
<Index>
Mars     2
Jupiter  67
Saturn   62
Uranus   27
Neptune  14
<<U7>    <int64>
>>> s[['Mars', 'Saturn']]
<Series>
<Index>
Mars     2
Saturn   62
<<U7>    <int64>
>>> s[s > 60]
<Series>
<Index>
Jupiter  67
Saturn   62
<<U7>    <int64>

#end_Series-[]


#start_Series-iloc[]
>>> s = sf.Series((1, 2, 67, 62, 27, 14), index=('Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'), dtype=np.int64)
>>> s.iloc[-2:]
<Series>
<Index>
Uranus   27
Neptune  14
<<U7>    <int64>

#end_Series-iloc[]


#-------------------------------------------------------------------------------
# Frame

#start_Frame-__init__()
>>> sf.Frame(np.array([[76.1, 0.967], [3.3, 0.847]]), columns=('Period', 'Eccentricity'), index=('Halley', 'Encke'), name='Orbits')
<Frame: Orbits>
<Index>         Period    Eccentricity <<U12>
<Index>
Halley          76.1      0.967
Encke           3.3       0.847
<<U6>           <float64> <float64>

#end_Frame-__init__()



#start_Frame-via_str.center()
>>> f = sf.Frame.from_records((('76.1 yrs.', '0.587 AU'), ('3.30 yrs.', '0.340 AU'), ('6.51 yrs.', '1.346 AU')), index=('Halley', 'Encke', "d'Arrest"), columns=('Orbital Period', 'Perihelion Distance'))
>>> f
<Frame>
<Index>  Orbital Period Perihelion Distance <<U19>
<Index>
Halley   76.1 yrs.      0.587 AU
Encke    3.30 yrs.      0.340 AU
d'Arrest 6.51 yrs.      1.346 AU
<<U8>    <<U9>          <<U8>
>>> f.via_str.center(18, '-')
<Frame>
<Index>  Orbital Period     Perihelion Distance <<U19>
<Index>
Halley   ----76.1 yrs.----- -----0.587 AU-----
Encke    ----3.30 yrs.----- -----0.340 AU-----
d'Arrest ----6.51 yrs.----- -----1.346 AU-----
<<U8>    <<U18>             <<U18>

#end_Frame-via_str.center()


#start_Frame-via_str.count()
>>> f = sf.Frame.from_records((('76.1 yrs.', '0.587 AU'), ('3.30 yrs.', '0.340 AU'), ('6.51 yrs.', '1.346 AU')), index=('Halley', 'Encke', "d'Arrest"), columns=('Orbital Period', 'Perihelion Distance'))
>>> f
<Frame>
<Index>  Orbital Period Perihelion Distance <<U19>
<Index>
Halley   76.1 yrs.      0.587 AU
Encke    3.30 yrs.      0.340 AU
d'Arrest 6.51 yrs.      1.346 AU
<<U8>    <<U9>          <<U8>
>>> f.via_str.count('3')
<Frame>
<Index>  Orbital Period Perihelion Distance <<U19>
<Index>
Halley   0              0
Encke    2              1
d'Arrest 0              1
<<U8>    <int64>        <int64>

#end_Frame-via_str.count()




#start_Frame-via_str.endswith()
>>> f = sf.Frame.from_records((('76.1 yrs.', '0.587 AU'), ('3.30 yrs.', '0.340 AU'), ('6.51 yrs.', '1.346 AU')), index=('Halley', 'Encke', "d'Arrest"), columns=('Orbital Period', 'Perihelion Distance'))
>>> f
<Frame>
<Index>  Orbital Period Perihelion Distance <<U19>
<Index>
Halley   76.1 yrs.      0.587 AU
Encke    3.30 yrs.      0.340 AU
d'Arrest 6.51 yrs.      1.346 AU
<<U8>    <<U9>          <<U8>
>>> f.via_str.endswith('AU')
<Frame>
<Index>  Orbital Period Perihelion Distance <<U19>
<Index>
Halley   False          True
Encke    False          True
d'Arrest False          True
<<U8>    <bool>         <bool>

#end_Frame-via_str.endswith()



#start_Frame-via_str.find()
>>> f = sf.Frame.from_records((('76.1 yrs.', '0.587 AU'), ('3.30 yrs.', '0.340 AU'), ('6.51 yrs.', '1.346 AU')), index=('Halley', 'Encke', "d'Arrest"), columns=('Orbital Period', 'Perihelion Distance'))
>>> f
<Frame>
<Index>  Orbital Period Perihelion Distance <<U19>
<Index>
Halley   76.1 yrs.      0.587 AU
Encke    3.30 yrs.      0.340 AU
d'Arrest 6.51 yrs.      1.346 AU
<<U8>    <<U9>          <<U8>
>>> f.via_str.find('.')
<Frame>
<Index>  Orbital Period Perihelion Distance <<U19>
<Index>
Halley   2              1
Encke    1              1
d'Arrest 1              1
<<U8>    <int64>        <int64>

#end_Frame-via_str.find()



#start_Frame-via_str.partition()
>>> f = sf.Frame.from_records((('76.1 yrs.', '0.587 AU'), ('3.30 yrs.', '0.340 AU'), ('6.51 yrs.', '1.346 AU')), index=('Halley', 'Encke', "d'Arrest"), columns=('Orbital Period', 'Perihelion Distance'))
>>> f
<Frame>
<Index>  Orbital Period Perihelion Distance <<U19>
<Index>
Halley   76.1 yrs.      0.587 AU
Encke    3.30 yrs.      0.340 AU
d'Arrest 6.51 yrs.      1.346 AU
<<U8>    <<U9>          <<U8>
>>> f.via_str.partition(' ').display_wide()
<Frame>
<Index>  Orbital Period        Perihelion Distance  <<U19>
<Index>
Halley   ('76.1', ' ', 'yrs.') ('0.587', ' ', 'AU')
Encke    ('3.30', ' ', 'yrs.') ('0.340', ' ', 'AU')
d'Arrest ('6.51', ' ', 'yrs.') ('1.346', ' ', 'AU')
<<U8>    <object>              <object>

#end_Frame-via_str.partition()


#start_Frame-via_str.rfind()
>>> f = sf.Frame.from_records((('76.1 yrs.', '0.587 AU'), ('3.30 yrs.', '0.340 AU'), ('6.51 yrs.', '1.346 AU')), index=('Halley', 'Encke', "d'Arrest"), columns=('Orbital Period', 'Perihelion Distance'))
>>> f
<Frame>
<Index>  Orbital Period Perihelion Distance <<U19>
<Index>
Halley   76.1 yrs.      0.587 AU
Encke    3.30 yrs.      0.340 AU
d'Arrest 6.51 yrs.      1.346 AU
<<U8>    <<U9>          <<U8>
>>> f.via_str.rfind('.')
<Frame>
<Index>  Orbital Period Perihelion Distance <<U19>
<Index>
Halley   8              1
Encke    8              1
d'Arrest 8              1
<<U8>    <int64>        <int64>

#end_Frame-via_str.rfind()


#start_Frame-via_str.ljust()
>>> f = sf.Frame.from_records((('76.1 yrs.', '0.587 AU'), ('3.30 yrs.', '0.340 AU'), ('6.51 yrs.', '1.346 AU')), index=('Halley', 'Encke', "d'Arrest"), columns=('Orbital Period', 'Perihelion Distance'))
>>> f
<Frame>
<Index>  Orbital Period Perihelion Distance <<U19>
<Index>
Halley   76.1 yrs.      0.587 AU
Encke    3.30 yrs.      0.340 AU
d'Arrest 6.51 yrs.      1.346 AU
<<U8>    <<U9>          <<U8>
>>> f.via_str.ljust(20, '.')
<Frame>
<Index>  Orbital Period       Perihelion Distance  <<U19>
<Index>
Halley   76.1 yrs............ 0.587 AU............
Encke    3.30 yrs............ 0.340 AU............
d'Arrest 6.51 yrs............ 1.346 AU............
<<U8>    <<U20>               <<U20>

#end_Frame-via_str.ljust()


#start_Frame-via_str.rjust()
>>> f = sf.Frame.from_records((('76.1 yrs.', '0.587 AU'), ('3.30 yrs.', '0.340 AU'), ('6.51 yrs.', '1.346 AU')), index=('Halley', 'Encke', "d'Arrest"), columns=('Orbital Period', 'Perihelion Distance'))
>>> f
<Frame>
<Index>  Orbital Period Perihelion Distance <<U19>
<Index>
Halley   76.1 yrs.      0.587 AU
Encke    3.30 yrs.      0.340 AU
d'Arrest 6.51 yrs.      1.346 AU
<<U8>    <<U9>          <<U8>
>>> f.via_str.rjust(20, '.')
<Frame>
<Index>  Orbital Period       Perihelion Distance  <<U19>
<Index>
Halley   ...........76.1 yrs. ............0.587 AU
Encke    ...........3.30 yrs. ............0.340 AU
d'Arrest ...........6.51 yrs. ............1.346 AU
<<U8>    <<U20>               <<U20>

#end_Frame-via_str.rjust()




#start_Frame-via_str.split()
>>> f = sf.Frame.from_records((('76.1 yrs.', '0.587 AU'), ('3.30 yrs.', '0.340 AU'), ('6.51 yrs.', '1.346 AU')), index=('Halley', 'Encke', "d'Arrest"), columns=('Orbital Period', 'Perihelion Distance'))
>>> f
<Frame>
<Index>  Orbital Period Perihelion Distance <<U19>
<Index>
Halley   76.1 yrs.      0.587 AU
Encke    3.30 yrs.      0.340 AU
d'Arrest 6.51 yrs.      1.346 AU
<<U8>    <<U9>          <<U8>
>>> f.via_str.split(' ')
<Frame>
<Index>  Orbital Period   Perihelion Distance <<U19>
<Index>
Halley   ('76.1', 'yrs.') ('0.587', 'AU')
Encke    ('3.30', 'yrs.') ('0.340', 'AU')
d'Arrest ('6.51', 'yrs.') ('1.346', 'AU')
<<U8>    <object>         <object>

#end_Frame-via_str.split()




#start_Frame-via_str.startswith()
>>> f = sf.Frame.from_records((('76.1 yrs.', '0.587 AU'), ('3.30 yrs.', '0.340 AU'), ('6.51 yrs.', '1.346 AU')), index=('Halley', 'Encke', "d'Arrest"), columns=('Orbital Period', 'Perihelion Distance'))
>>> f
<Frame>
<Index>  Orbital Period Perihelion Distance <<U19>
<Index>
Halley   76.1 yrs.      0.587 AU
Encke    3.30 yrs.      0.340 AU
d'Arrest 6.51 yrs.      1.346 AU
<<U8>    <<U9>          <<U8>
>>> f.via_str.startswith('0.')
<Frame>
<Index>  Orbital Period Perihelion Distance <<U19>
<Index>
Halley   False          True
Encke    False          True
d'Arrest False          False
<<U8>    <bool>         <bool>

#end_Frame-via_str.startswith()



#start_Frame-via_str.replace()
>>> f = sf.Frame.from_records((('76.1 yrs.', '0.587 AU'), ('3.30 yrs.', '0.340 AU'), ('6.51 yrs.', '1.346 AU')), index=('Halley', 'Encke', "d'Arrest"), columns=('Orbital Period', 'Perihelion Distance'))
>>> f
<Frame>
<Index>  Orbital Period Perihelion Distance <<U19>
<Index>
Halley   76.1 yrs.      0.587 AU
Encke    3.30 yrs.      0.340 AU
d'Arrest 6.51 yrs.      1.346 AU
<<U8>    <<U9>          <<U8>
>>> f.via_str.replace(' AU', '').via_str.replace(' yrs.', '').astype(float)
<Frame>
<Index>  Orbital Period Perihelion Distance <<U19>
<Index>
Halley   76.1           0.587
Encke    3.3            0.34
d'Arrest 6.51           1.346
<<U8>    <float64>      <float64>

#end_Frame-via_str.replace()


#start_Frame-interface
>>> sf.Frame.interface.loc[sf.Frame.interface.index.via_str.startswith('sort')]
<Frame: Frame>
<Index>                              cls_name group  doc                  <<U18>
<Index: signature>
sort_columns(*, ascending, kind)     Frame    Method Return a new Fram...
sort_index(*, ascending, kind)       Frame    Method Return a new Fram...
sort_values(key, *, ascending, ax... Frame    Method Return a new Fram...
<<U94>                               <<U5>    <<U17> <<U83>

#end_Frame-interface



#start_Frame-from_dict()
>>> sf.Frame.from_dict(dict(diameter=(12756, 142984, 120536), mass=(5.97, 1898, 568)), index=('Earth', 'Jupiter', 'Saturn'), dtypes=dict(diameter=np.int64))
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    5.97
Jupiter 142984   1898.0
Saturn  120536   568.0
<<U7>   <int64>  <float64>

#end_Frame-from_dict()




#start_Frame-from_records()
>>> index = ('Mercury', 'Venus', 'Earth', 'Mars')
>>> columns = ('diameter', 'gravity', 'temperature')
>>> records = ((4879, 3.7, 167), (12104, 8.9, 464), (12756, 9.8, 15), (6792, 3.7, -65))
>>> sf.Frame.from_records(records, index=index, columns=columns, dtypes=dict(diameter=np.int64, temperature=np.int64))
<Frame>
<Index> diameter gravity   temperature <<U11>
<Index>
Mercury 4879     3.7       167
Venus   12104    8.9       464
Earth   12756    9.8       15
Mars    6792     3.7       -65
<<U7>   <int64>  <float64> <int64>

#end_Frame-from_records()



#start_Frame-from_items()
>>> sf.Frame.from_items((('diameter', (12756, 142984, 120536)), ('mass', (5.97, 1898, 568))), index=('Earth', 'Jupiter', 'Saturn'), dtypes=dict(diameter=np.int64))
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    5.97
Jupiter 142984   1898.0
Saturn  120536   568.0
<<U7>   <int64>  <float64>

#end_Frame-from_items()



#start_Frame-from_concat()
>>> f1 = sf.Frame.from_dict(dict(diameter=(12756, 142984), mass=(5.97, 1898)), index=('Earth', 'Jupiter'))
>>> f2 = sf.Frame.from_dict(dict(mass=(0.642, 102), moons=(2, 14)), index=('Mars', 'Neptune'))
>>> sf.Frame.from_concat((f1, f2))
<Frame>
<Index> diameter  mass      moons     <<U8>
<Index>
Earth   12756.0   5.97      nan
Jupiter 142984.0  1898.0    nan
Mars    nan       0.642     2.0
Neptune nan       102.0     14.0
<<U7>   <float64> <float64> <float64>

>>> sf.Frame.from_concat((f1, f2), union=False)
<Frame>
<Index> mass      <<U8>
<Index>
Earth   5.97
Jupiter 1898.0
Mars    0.642
Neptune 102.0
<<U7>   <float64>

#end_Frame-from_concat()



#start_Frame-from_structured_array()
>>> a = np.array([('Venus', 4.87, 464), ('Neptune', 102, -200)], dtype=[('name', object), ('mass', 'f4'), ('temperature', 'i4')])
>>> sf.Frame.from_structured_array(a, index_depth=1)
<Frame>
<Index>  mass              temperature <<U11>
<Index>
Venus    4.869999885559082 464
Neptune  102.0             -200
<object> <float32>         <int32>

#end_Frame-from_structured_array()



#start_Frame-from_csv()
>>> from io import StringIO
>>> filelike = StringIO('name,mass,temperature\\nVenus,4.87,464\\nNeptune,102,-200')
>>> sf.Frame.from_csv(filelike, index_depth=1, dtypes=dict(temperature=np.int64))
<Frame>
<Index> mass      temperature <<U11>
<Index>
Venus   4.87      464
Neptune 102.0     -200
<<U7>   <float64> <int64>

#end_Frame-from_csv()



#start_Frame-items()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 142984, 120536), temperature=(15, -110, -140)), index=('Earth', 'Jupiter', 'Saturn'), dtypes=dict(temperature=np.int64, diameter=np.int64))
>>> f
<Frame>
<Index> diameter temperature <<U11>
<Index>
Earth   12756    15
Jupiter 142984   -110
Saturn  120536   -140
<<U7>   <int64>  <int64>
>>> len(f)
3
>>> [k for k, v in f.items() if (v < 0).any()]
['temperature']

#end_Frame-items()


#start_Frame-get()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 142984, 120536), temperature=(15, -110, -140)), index=('Earth', 'Jupiter', 'Saturn'), dtypes=dict(temperature=np.int64, diameter=np.int64))

>>> f.get('diameter')
<Series: diameter>
<Index>
Earth              12756
Jupiter            142984
Saturn             120536
<<U7>              <int64>

>>> f.get('mass', np.nan)
nan

#end_Frame-get()


#start_Frame-__contains__()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 142984, 120536), temperature=(15, -110, -140)), index=('Earth', 'Jupiter', 'Saturn'), dtypes=dict(temperature=np.int64, diameter=np.int64))

>>> 'temperature' in f
True

#end_Frame-__contains__()


#start_Frame-values
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 142984, 120536), temperature=(15, -110, -140)), index=('Earth', 'Jupiter', 'Saturn'), dtypes=dict(temperature=np.int64, diameter=np.int64))

>>> f.values.tolist()
[[12756, 15], [142984, -110], [120536, -140]]

#end_Frame-values


#start_Frame-__truediv__()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'), dtypes=dict(diameter=np.int64))
>>> f
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    5.97
Mars    6792     0.642
Jupiter 142984   1898.0
<<U7>   <int64>  <float64>

>>> f / f.loc['Earth']
<Frame>
<Index> diameter           mass                <<U8>
<Index>
Earth   1.0                1.0
Mars    0.5324553151458138 0.10753768844221107
Jupiter 11.209156475384132 317.92294807370183
<<U7>   <float64>          <float64>

#end_Frame-__truediv__()


#start_Frame-max()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 142984, 120536), mass=(5.97, 1898, 568)), index=('Earth', 'Jupiter', 'Saturn'), dtypes=dict(diameter=np.int64))
>>> f
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    5.97
Jupiter 142984   1898.0
Saturn  120536   568.0
<<U7>   <int64>  <float64>

>>> f.max()
<Series>
<Index>
diameter 142984.0
mass     1898.0
<<U8>    <float64>

#end_Frame-max()


#start_Frame-min()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 142984, 120536), mass=(5.97, 1898, 568)), index=('Earth', 'Jupiter', 'Saturn'), dtypes=dict(diameter=np.int64))

>>> f.min()
<Series>
<Index>
diameter 12756.0
mass     5.97
<<U8>    <float64>

#end_Frame-min()


#start_Frame-std()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 142984, 120536), mass=(5.97, 1898, 568)), index=('Earth', 'Jupiter', 'Saturn'), dtypes=dict(diameter=np.int64))

>>> f.std()
<Series>
<Index>
diameter 56842.64155250587
mass     793.344204533358
<<U8>    <float64>

#end_Frame-std()


#start_Frame-sum()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 142984, 120536), mass=(5.97, 1898, 568)), index=('Earth', 'Jupiter', 'Saturn'), dtypes=dict(diameter=np.int64))
>>> f
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    5.97
Jupiter 142984   1898.0
Saturn  120536   568.0
<<U7>   <int64>  <float64>

>>> f.sum()
<Series>
<Index>
diameter 276276.0
mass     2471.9700000000003
<<U8>    <float64>

#end_Frame-sum()


#start_Frame-mean()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 142984, 120536), mass=(5.97, 1898, 568)), index=('Earth', 'Jupiter', 'Saturn'), dtypes=dict(diameter=np.int64))

>>> f.mean()
<Series>
<Index>
diameter 92092.0
mass     823.9900000000001
<<U8>    <float64>

#end_Frame-mean()


#start_Frame-relabel()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'), dtypes=dict(diameter=np.int64))

>>> f.relabel(index=lambda x: x[:2].upper(), columns={'mass': 'mass(1e24kg)'})
<Frame>
<Index> diameter mass(1e24kg) <<U12>
<Index>
EA      12756    5.97
MA      6792     0.642
JU      142984   1898.0
<<U2>   <int64>  <float64>

#end_Frame-relabel()


#start_Frame-reindex()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'), dtypes=dict(diameter=np.int64))
>>> f
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    5.97
Mars    6792     0.642
Jupiter 142984   1898.0
<<U7>   <int64>  <float64>

>>> f.reindex(index=('Jupiter', 'Mars', 'Mercury'), columns=('density', 'mass'))
<Frame>
<Index> density   mass      <<U7>
<Index>
Jupiter nan       1898.0
Mars    nan       0.642
Mercury nan       nan
<<U7>   <float64> <float64>

#end_Frame-reindex()


#start_Frame-iter_element()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'), dtypes=dict(diameter=np.int64))

>>> [x for x in f.iter_element()]
[12756, 5.97, 6792, 0.642, 142984, 1898.0]

#end_Frame-iter_element()


#start_Frame-iter_element().apply()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'), dtypes=dict(diameter=np.int64))

>>> f.iter_element().apply(lambda x: x ** 2)
<Frame>
<Index> diameter    mass                <<U8>
<Index>
Earth   162715536   35.640899999999995
Mars    46131264    0.41216400000000003
Jupiter 20444424256 3602404.0
<<U7>   <object>    <object>

#end_Frame-iter_element().apply()


#start_Frame-iter_element_items()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'))

>>> [x for x in f.iter_element_items()]
[(('Earth', 'diameter'), 12756), (('Earth', 'mass'), 5.97), (('Mars', 'diameter'), 6792), (('Mars', 'mass'), 0.642), (('Jupiter', 'diameter'), 142984), (('Jupiter', 'mass'), 1898.0)]

#end_Frame-iter_element_items()


#start_Frame-iter_element_items().apply()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'))

>>> f.iter_element_items().apply(lambda k, v: v ** 2 if k[0] == 'Mars' else None)
<Frame>
<Index> diameter mass                <<U8>
<Index>
Earth   None     None
Mars    46131264 0.41216400000000003
Jupiter None     None
<<U7>   <object> <object>

#end_Frame-iter_element_items().apply()


#start_Frame-iter_array()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'), dtypes=dict(diameter=np.int64))
>>> f
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    5.97
Mars    6792     0.642
Jupiter 142984   1898.0
<<U7>   <int64>  <float64>

>>> [x.tolist() for x in f.iter_array(axis=0)]
[[12756, 6792, 142984], [5.97, 0.642, 1898.0]]

>>> [x.tolist() for x in f.iter_array(axis=1)]
[[12756.0, 5.97], [6792.0, 0.642], [142984.0, 1898.0]]

#end_Frame-iter_array()


#start_Frame-iter_array().apply()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'), dtypes=dict(diameter=np.int64))
>>> f
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    5.97
Mars    6792     0.642
Jupiter 142984   1898.0
<<U7>   <int64>  <float64>

>>> f.iter_array(axis=0).apply(np.sum)
<Series>
<Index>
diameter 162532.0
mass     1904.612
<<U8>    <float64>

#end_Frame-iter_array().apply()


#start_Frame-iter_array_items()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'))

>>> [x for x in f.iter_array_items(axis=0)]
[('diameter', array([ 12756,   6792, 142984])), ('mass', array([5.970e+00, 6.420e-01, 1.898e+03]))]

>>> [x for x in f.iter_array_items(axis=1)]
[('Earth', array([1.2756e+04, 5.9700e+00])), ('Mars', array([6.792e+03, 6.420e-01])), ('Jupiter', array([142984.,   1898.]))]

>>> f.iter_array_items(axis=1).apply(lambda k, v: v.sum() if k == 'Earth' else 0)
<Series>
<Index>
Earth    12761.97
Mars     0.0
Jupiter  0.0
<<U7>    <float64>

#end_Frame-iter_array_items()


#start_Frame-iter_array_items().apply()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'))

>>> f.iter_array_items(axis=1).apply(lambda k, v: v.sum() if k == 'Earth' else 0)
<Series>
<Index>
Earth    12761.97
Mars     0.0
Jupiter  0.0
<<U7>    <float64>

#end_Frame-iter_array_items().apply()


#start_Frame-iter_tuple()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'))

>>> [x for x in f.iter_tuple(axis=0)]
[Axis(Earth=12756, Mars=6792, Jupiter=142984), Axis(Earth=5.97, Mars=0.642, Jupiter=1898.0)]

>>> [x for x in f.iter_tuple(axis=1)]
[Axis(diameter=12756.0, mass=5.97), Axis(diameter=6792.0, mass=0.642), Axis(diameter=142984.0, mass=1898.0)]

#end_Frame-iter_tuple()


#start_Frame-iter_tuple().apply()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'))

>>> f.iter_tuple(axis=1).apply(lambda nt: nt.mass / (4 / 3 * np.pi * (nt.diameter * 0.5) ** 3))
<Series>
<Index>
Earth    5.49328558e-12
Mars     3.91330208e-12
Jupiter  1.24003876e-12
<<U7>    <float64>

#end_Frame-iter_tuple().apply()


#start_Frame-iter_tuple_items()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'))

>>> [x for x in f.iter_tuple_items(axis=0)]
[('diameter', Axis(Earth=12756, Mars=6792, Jupiter=142984)), ('mass', Axis(Earth=5.97, Mars=0.642, Jupiter=1898.0))]

>>> [x for x in f.iter_tuple_items(axis=1)]
[('Earth', Axis(diameter=12756.0, mass=5.97)), ('Mars', Axis(diameter=6792.0, mass=0.642)), ('Jupiter', Axis(diameter=142984.0, mass=1898.0))]

#end_Frame-iter_tuple_items()


#start_Frame-iter_tuple_items().apply()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'))

>>> f.iter_tuple_items(axis=1).apply(lambda k, v: v.diameter if k == 'Earth' else 0)
<Series>
<Index>
Earth    12756.0
Mars     0.0
Jupiter  0.0
<<U7>    <float64>

#end_Frame-iter_tuple_items().apply()


#start_Frame-iter_series()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'), dtypes=dict(diameter=np.int64))

>>> next(iter(f.iter_series(axis=0)))
<Series: diameter>
<Index>
Earth              12756
Mars               6792
Jupiter            142984
<<U7>              <int64>

>>> next(iter(f.iter_series(axis=1)))
<Series: Earth>
<Index>
diameter        12756.0
mass            5.97
<<U8>           <float64>

#end_Frame-iter_series()


#start_Frame-iter_series().apply()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'), dtypes=dict(diameter=np.int64))

>>> f.iter_series(axis=0).apply(lambda s: s.mean())
<Series>
<Index>
diameter 54177.333333333336
mass     634.8706666666667
<<U8>    <float64>

#end_Frame-iter_series().apply()


#start_Frame-iter_series_items()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'))

>>> [(k, v.mean()) for k, v in f.iter_series_items(axis=0)]
[('diameter', 54177.333333333336), ('mass', 634.8706666666667)]

>>> [(k, v.max()) for k, v in f.iter_series_items(axis=1)]
[('Earth', 12756.0), ('Mars', 6792.0), ('Jupiter', 142984.0)]

>>> f.iter_series_items(axis=0).apply(lambda k, v: v.mean() if k == 'diameter' else v.sum())
<Series>
<Index>
diameter 54177.333333333336
mass     1904.612
<<U8>    <float64>

#end_Frame-iter_series_items()


#start_Frame-iter_group()
>>> f = sf.Frame.from_dict(dict(mass=(0.33, 4.87, 5.97, 0.642), moons=(0, 0, 1, 2)), index=('Mercury', 'Venus', 'Earth', 'Mars'), dtypes=dict(moons=np.int64))
>>> next(iter(f.iter_group('moons')))
<Frame>
<Index> mass      moons   <<U5>
<Index>
Mercury 0.33      0
Venus   4.87      0
<<U7>   <float64> <int64>
>>> [x.shape for x in f.iter_group('moons')]
[(2, 2), (1, 2), (1, 2)]

#end_Frame-iter_group()


#start_Frame-iter_group_items()
>>> f = sf.Frame.from_dict(dict(mass=(0.33, 4.87, 5.97, 0.642), moons=(0, 0, 1, 2)), index=('Mercury', 'Venus', 'Earth', 'Mars'))
>>> [(k, v.index.values.tolist(), v['mass'].mean()) for k, v in f.iter_group_items('moons')]
[(0, ['Mercury', 'Venus'], 2.6), (1, ['Earth'], 5.97), (2, ['Mars'], 0.642)]

#end_Frame-iter_group_items()



#start_Frame-assign[]()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'), dtypes=dict(diameter=np.int64))
>>> f
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    5.97
Mars    6792     0.642
Jupiter 142984   1898.0
<<U7>   <int64>  <float64>

>>> f.assign['mass'](f['mass'] * .001)
<Frame>
<Index> diameter mass               <<U8>
<Index>
Earth   12756    0.00597
Mars    6792     0.000642
Jupiter 142984   1.8980000000000001
<<U7>   <int64>  <float64>

#end_Frame-assign[]()


#start_Frame-assign.loc[]()
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'), dtypes=dict(diameter=np.int64))

>>> f.assign.loc['Mars', 'mass'](0)
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    5.97
Mars    6792     0.0
Jupiter 142984   1898.0
<<U7>   <int64>  <float64>

>>> f.assign.loc['Mars':, 'diameter'](0)
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    5.97
Mars    0        0.642
Jupiter 0        1898.0
<<U7>   <int64>  <float64>

>>> f.assign.loc[f['diameter'] > 10000, 'mass'](0)
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    0.0
Mars    6792     0.642
Jupiter 142984   0.0
<<U7>   <int64>  <float64>

#end_Frame-assign.loc[]()


#start_Frame-drop[]
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 142984, 120536), temperature=(15, -110, -140)), index=('Earth', 'Jupiter', 'Saturn'), dtypes=dict(diameter=np.int64, temperature=np.int64))

>>> f
<Frame>
<Index> diameter temperature <<U11>
<Index>
Earth   12756    15
Jupiter 142984   -110
Saturn  120536   -140
<<U7>   <int64>  <int64>

>>> f.drop['diameter']
<Frame>
<Index> temperature <<U11>
<Index>
Earth   15
Jupiter -110
Saturn  -140
<<U7>   <int64>

#end_Frame-drop[]


#start_Frame-drop.loc[]
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 142984, 120536), temperature=(15, -110, -140)), index=('Earth', 'Jupiter', 'Saturn'), dtypes=dict(diameter=np.int64, temperature=np.int64))
>>> f
<Frame>
<Index> diameter temperature <<U11>
<Index>
Earth   12756    15
Jupiter 142984   -110
Saturn  120536   -140
<<U7>   <int64>  <int64>

>>> f.drop.loc[f['temperature'] < 0]
<Frame>
<Index> diameter temperature <<U11>
<Index>
Earth   12756    15
<<U7>   <int64>  <int64>

#end_Frame-drop.loc[]


#start_Frame-drop.iloc[]
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 142984, 120536), temperature=(15, -110, -140)), index=('Earth', 'Jupiter', 'Saturn'), dtypes=dict(diameter=np.int64, temperature=np.int64))
>>> f
<Frame>
<Index> diameter temperature <<U11>
<Index>
Earth   12756    15
Jupiter 142984   -110
Saturn  120536   -140
<<U7>   <int64>  <int64>

>>> f.drop.iloc[-1, -1]
<Frame>
<Index> diameter <<U11>
<Index>
Earth   12756
Jupiter 142984
<<U7>   <int64>

#end_Frame-drop.iloc[]


#start_Frame-shape
>>> f = sf.Frame.from_items((('diameter', (12756, 142984, 120536)), ('mass', (5.97, 1898, 568))), index=('Earth', 'Jupiter', 'Saturn'), dtypes=dict(diameter=np.int64))

>>> f
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    5.97
Jupiter 142984   1898.0
Saturn  120536   568.0
<<U7>   <int64>  <float64>

>>> f.shape
(3, 2)

#end_Frame-shape


#start_Frame-ndim
>>> f = sf.Frame.from_items((('diameter', (12756, 142984, 120536)), ('mass', (5.97, 1898, 568))), index=('Earth', 'Jupiter', 'Saturn'), dtypes=dict(diameter=np.int64))
>>> f
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    5.97
Jupiter 142984   1898.0
Saturn  120536   568.0
<<U7>   <int64>  <float64>

>>> f.ndim
2

#end_Frame-ndim


#start_Frame-size
>>> f = sf.Frame.from_items((('diameter', (12756, 142984, 120536)), ('mass', (5.97, 1898, 568))), index=('Earth', 'Jupiter', 'Saturn'), dtypes=dict(diameter=np.int64))
>>> f
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    5.97
Jupiter 142984   1898.0
Saturn  120536   568.0
<<U7>   <int64>  <float64>

>>> f.size
6

#end_Frame-size


#start_Frame-nbytes
>>> f = sf.Frame.from_items((('diameter', (12756, 142984, 120536)), ('mass', (5.97, 1898, 568))), index=('Earth', 'Jupiter', 'Saturn'), dtypes=dict(diameter=np.int64))
>>> f
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    5.97
Jupiter 142984   1898.0
Saturn  120536   568.0
<<U7>   <int64>  <float64>

>>> f.nbytes
48

#end_Frame-nbytes


#start_Frame-dtypes
>>> f = sf.Frame.from_items((('diameter', (12756, 142984, 120536)), ('mass', (5.97, 1898, 568))), index=('Earth', 'Jupiter', 'Saturn'), dtypes=dict(diameter=np.int64))

>>> f
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    5.97
Jupiter 142984   1898.0
Saturn  120536   568.0
<<U7>   <int64>  <float64>

>>> f.dtypes
<Series>
<Index>
diameter int64
mass     float64
<<U8>    <object>

#end_Frame-dtypes


#start_Frame-[]
>>> index = ('Mercury', 'Venus', 'Earth', 'Mars')
>>> columns = ('diameter', 'gravity', 'temperature')
>>> records = ((4879, 3.7, 167), (12104, 8.9, 464), (12756, 9.8, 15), (6792, 3.7, -65))
>>> f = sf.Frame.from_records(records, index=index, columns=columns, dtypes=dict(diameter=np.int64, temperature=np.int64))
>>> f
<Frame>
<Index> diameter gravity   temperature <<U11>
<Index>
Mercury 4879     3.7       167
Venus   12104    8.9       464
Earth   12756    9.8       15
Mars    6792     3.7       -65
<<U7>   <int64>  <float64> <int64>

>>> f['gravity']
<Series: gravity>
<Index>
Mercury           3.7
Venus             8.9
Earth             9.8
Mars              3.7
<<U7>             <float64>
>>> f['gravity':]
<Frame>
<Index> gravity   temperature <<U11>
<Index>
Mercury 3.7       167
Venus   8.9       464
Earth   9.8       15
Mars    3.7       -65
<<U7>   <float64> <int64>
>>> f[['diameter', 'temperature']]
<Frame>
<Index> diameter temperature <<U11>
<Index>
Mercury 4879     167
Venus   12104    464
Earth   12756    15
Mars    6792     -65
<<U7>   <int64>  <int64>

#end_Frame-[]


#start_Frame-loc[]
>>> index = ('Mercury', 'Venus', 'Earth', 'Mars')
>>> columns = ('diameter', 'gravity', 'temperature')
>>> records = ((4879, 3.7, 167), (12104, 8.9, 464), (12756, 9.8, 15), (6792, 3.7, -65))
>>> f = sf.Frame.from_records(records, index=index, columns=columns, dtypes=dict(diameter=np.int64, temperature=np.int64))
>>> f
<Frame>
<Index> diameter gravity   temperature <<U11>
<Index>
Mercury 4879     3.7       167
Venus   12104    8.9       464
Earth   12756    9.8       15
Mars    6792     3.7       -65
<<U7>   <int64>  <float64> <int64>

>>> f.loc['Earth', 'temperature']
15
>>> f.loc['Earth':, 'temperature']
<Series: temperature>
<Index>
Earth                 15
Mars                  -65
<<U7>                 <int64>
>>> f.loc[f['temperature'] > 100, 'diameter']
<Series: diameter>
<Index>
Mercury            4879
Venus              12104
<<U7>              <int64>
>>> f.loc[sf.ILoc[-1], ['gravity', 'temperature']]
<Series: Mars>
<Index>
gravity        3.7
temperature    -65.0
<<U11>         <float64>

#end_Frame-loc[]


#start_Frame-iloc[]
>>> index = ('Mercury', 'Venus', 'Earth', 'Mars')
>>> columns = ('diameter', 'gravity', 'temperature')
>>> records = ((4879, 3.7, 167), (12104, 8.9, 464), (12756, 9.8, 15), (6792, 3.7, -65))
>>> f = sf.Frame.from_records(records, index=index, columns=columns, dtypes=dict(diameter=np.int64, temperature=np.int64))
>>> f
<Frame>
<Index> diameter gravity   temperature <<U11>
<Index>
Mercury 4879     3.7       167
Venus   12104    8.9       464
Earth   12756    9.8       15
Mars    6792     3.7       -65
<<U7>   <int64>  <float64> <int64>

>>> f.iloc[-2:, -1]
<Series: temperature>
<Index>
Earth                 15
Mars                  -65
<<U7>                 <int64>

#end_Frame-iloc[]


#start_Frame-set_index_hierarchy()
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
>>> f.set_index_hierarchy(('type', 'name'), drop=True)
<Frame>
<Index>                                    mass      charge    <<U6>
<IndexHierarchy: ('type', 'name')>
lepton                             muon    0.106     -1.0
lepton                             tau     1.777     -1.0
quark                              charm   1.3       0.666
quark                              strange 0.1       -0.333
<<U6>                              <<U7>   <float64> <float64>

#end_Frame-set_index_hierarchy()


#start_Frame-relabel_flat()
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

>>> f.relabel_flat(index=True)
<Frame>
<Index>              mass      charge    <<U6>
<Index>
('lepton', 'muon')   0.106     -1.0
('lepton', 'tau')    1.777     -1.0
('quark', 'charm')   1.3       0.666
('quark', 'strange') 0.1       -0.333
<object>             <float64> <float64>

#end_Frame-relabel_flat()

#start_Frame-relabel_add_level()
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

>>> f.relabel_add_level(index='particle')
<Frame>
<Index>                                           mass      charge    <<U6>
<IndexHierarchy: ('type', 'name')>
particle                           lepton muon    0.106     -1.0
particle                           lepton tau     1.777     -1.0
particle                           quark  charm   1.3       0.666
particle                           quark  strange 0.1       -0.333
<<U8>                              <<U6>  <<U7>   <float64> <float64>

#end_Frame-relabel_add_level()


#start_Frame-relabel_drop_level()
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

>>> f.relabel_drop_level(index=1)
<Frame>
<Index> mass      charge    <<U6>
<Index>
muon    0.106     -1.0
tau     1.777     -1.0
charm   1.3       0.666
strange 0.1       -0.333
<<U7>   <float64> <float64>

#end_Frame-relabel_drop_level()


#start_Frame-pivot()
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

>>> f.pivot(index_fields='type', data_fields='mass', func={'mean':np.mean, 'max':np.max})
<Frame>
<IndexHierarchy: ('values', 'func')> mass               mass      <<U4>
                                     mean               max       <<U4>
<Index: type>
lepton                               0.9415             1.777
quark                                0.7000000000000001 1.3
<<U6>                                <float64>          <float64>

#end_Frame-pivot()


#start_Frame-pivot_stack()
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

>>> f.pivot_stack()
<Frame>
<Index>                         0         <int64>
<IndexHierarchy>
lepton           muon    mass   0.106
lepton           muon    charge -1.0
lepton           tau     mass   1.777
lepton           tau     charge -1.0
quark            charm   mass   1.3
quark            charm   charge 0.666
quark            strange mass   0.1
quark            strange charge -0.333
<<U6>            <<U7>   <<U6>  <float64>

#end_Frame-pivot_stack()

#start_Frame-pivot_unstack()
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

>>> f.pivot_unstack(0)
<Frame>
<IndexHierarchy> mass      mass      charge    charge    <<U6>
                 lepton    quark     lepton    quark     <<U6>
<Index>
muon             0.106     nan       -1.0      nan
tau              1.777     nan       -1.0      nan
charm            nan       1.3       nan       0.666
strange          nan       0.1       nan       -0.333
<<U7>            <float64> <float64> <float64> <float64>

>>> f.pivot_unstack(1)
<Frame>
<IndexHierarchy> mass      mass      mass      mass      charge    charge    charge    charge    <<U6>
                 muon      tau       charm     strange   muon      tau       charm     strange   <<U7>
<Index>
lepton           0.106     1.777     nan       nan       -1.0      -1.0      nan       nan
quark            nan       nan       1.3       0.1       nan       nan       0.666     -0.333
<<U6>            <float64> <float64> <float64> <float64> <float64> <float64> <float64> <float64>

>>> f.pivot_unstack([0, 1])
<Frame>
<IndexHierarchy> mass      mass      mass      mass      charge    charge    charge    charge    <<U6>
                 lepton    lepton    quark     quark     lepton    lepton    quark     quark     <<U6>
                 muon      tau       charm     strange   muon      tau       charm     strange   <<U7>
<Index>
0                0.106     1.777     1.3       0.1       -1.0      -1.0      0.666     -0.333
<int64>          <float64> <float64> <float64> <float64> <float64> <float64> <float64> <float64>

#end_Frame-pivot_unstack()



#-------------------------------------------------------------------------------
# FrameGO

#start_FrameGO-__init__()
>>> f = sf.FrameGO(np.array([[76.1, 0.967], [3.3, 0.847]]), columns=('Period', 'Eccentricity'), index=('Halley', 'Encke'), name='Orbits')
>>> f
<FrameGO: Orbits>
<IndexGO>         Period    Eccentricity <<U12>
<Index>
Halley            76.1      0.967
Encke             3.3       0.847
<<U6>             <float64> <float64>
>>> f['Inclination'] = (162.2, 11.8)
>>> f
<FrameGO: Orbits>
<IndexGO>         Period    Eccentricity Inclination <<U12>
<Index>
Halley            76.1      0.967        162.2
Encke             3.3       0.847        11.8
<<U6>             <float64> <float64>    <float64>

#end_FrameGO-__init__()


#start_FrameGO-interface
>>> sf.FrameGO.interface.loc[sf.FrameGO.interface.index.via_str.startswith('drop')]
<Frame: FrameGO>
<Index>                              cls_name group    doc                  <<U18>
<Index: signature>
drop_duplicated(*, axis, exclude_... FrameGO  Method   Return a Frame wi...
dropna(axis, condition)              FrameGO  Method   Return a new Fram...
drop[key]                            FrameGO  Selector Label-based selec...
drop.iloc[key]                       FrameGO  Selector
drop.loc[key]                        FrameGO  Selector
<<U94>                               <<U7>    <<U17>   <<U83>

#end_FrameGO-interface


#start_FrameGO-from_dict()
>>> f = sf.FrameGO.from_dict(dict(diameter=(12756, 142984, 120536), mass=(5.97, 1898, 568)), index=('Earth', 'Jupiter', 'Saturn'), dtypes=dict(diameter=np.int64))
>>> f['radius'] = f['diameter'] * 0.5
>>> f
<FrameGO>
<IndexGO> diameter mass      radius    <<U8>
<Index>
Earth     12756    5.97      6378.0
Jupiter   142984   1898.0    71492.0
Saturn    120536   568.0     60268.0
<<U7>     <int64>  <float64> <float64>

#end_FrameGO-from_dict()



#-------------------------------------------------------------------------------
# Bus

#start_Bus-interface
>>> sf.Bus.interface.loc[sf.Bus.interface['group'] == 'Exporter']
<Frame: Bus>
<Index>                    cls_name group    doc                  <<U18>
<Index: signature>
to_hdf5(fp, config)        Bus      Exporter Write the complet...
to_sqlite(fp, config)      Bus      Exporter Write the complet...
to_xlsx(fp, config)        Bus      Exporter Write the complet...
to_zip_csv(fp, config)     Bus      Exporter Write the complet...
to_zip_parquet(fp, config) Bus      Exporter Write the complet...
to_zip_pickle(fp, config)  Bus      Exporter Write the complet...
to_zip_tsv(fp, config)     Bus      Exporter Write the complet...
<<U50>                     <<U3>    <<U15>   <<U83>

#end_Bus-interface




#-------------------------------------------------------------------------------
# Index

#start_Index-interface
>>> sf.Index.interface.loc[sf.Index.interface.index.via_str.startswith('re')]
<Frame: Index>
<Index>            cls_name group  doc                  <<U18>
<Index: signature>
relabel(mapper)    Index    Method Return a new Inde...
rename(name)       Index    Method Return a new Fram...
<<U68>             <<U5>    <<U17> <<U83>

#end_Index-interface


#start_Index-__init__()
>>> sf.Index(('Mercury', 'Mars'), dtype=object)
<Index>
Mercury
Mars
<object>

>>> sf.Index(name[:2].upper() for name in ('Mercury', 'Mars'))
<Index>
ME
MA
<<U2>

#end_Index-__init__()


#start_Index-relabel()
>>> index = sf.Index(('Venus', 'Saturn', 'Neptune'))
>>> index.relabel({'Venus': 'Mars'})
<Index>
Mars
Saturn
Neptune
<<U7>

>>> index = sf.Index(('Venus', 'Saturn', 'Neptune'))
>>> index.relabel({'Neptune': 'Uranus'})
<Index>
Venus
Saturn
Uranus
<<U6>

>>> index.relabel(lambda x: x[:2].upper())
<Index>
VE
SA
NE
<<U2>

#end_Index-relabel()


#-------------------------------------------------------------------------------
# IndexGO

#start_IndexGO-interface
>>> sf.IndexGO.interface.loc[sf.IndexGO.interface.index.via_str.startswith('to_')]
<Frame: IndexGO>
<Index>                              cls_name group    doc                  <<U18>
<Index: signature>
to_html(config)                      IndexGO  Exporter Return an HTML ta...
to_html_datatables(fp, *, show, c... IndexGO  Exporter Return a complete...
to_pandas()                          IndexGO  Exporter Return a Pandas I...
to_series()                          IndexGO  Exporter Return a Series w...
<<U68>                               <<U7>    <<U17>   <<U83>

#end_IndexGO-interface


#start_IndexGO-append()
>>> a = sf.IndexGO(('Uranus', 'Neptune'))
>>> a.append('Pluto')
>>> a
<IndexGO>
Uranus
Neptune
Pluto
<<U7>

#end_IndexGO-append()

#-------------------------------------------------------------------------------
# IndexHierarchy

#start_IndexHierarchy-interface
>>> sf.IndexHierarchy.interface.loc[sf.IndexHierarchy.interface.index.via_str.startswith('from_')]
<Frame: IndexHierarchy>
<Index>                              cls_name       group       doc                  <<U18>
<Index: signature>
from_index_items(items, *, index_... IndexHierarchy Constructor Given an iterable...
from_labels(labels, *, name, reor... IndexHierarchy Constructor Construct an Inde...
from_labels_delimited(labels, *, ... IndexHierarchy Constructor Construct an Inde...
from_names(names)                    IndexHierarchy Constructor Construct a zero-...
from_pandas(value)                   IndexHierarchy Constructor Given a Pandas in...
from_product(*, name, *levels)       IndexHierarchy Constructor Given groups of i...
from_tree(tree, *, name)             IndexHierarchy Constructor Convert into a In...
<<U68>                               <<U14>         <<U17>      <<U83>

#end_IndexHierarchy-interface


#start_IndexHierarchy-from_labels()
>>> sf.IndexHierarchy.from_labels((('lepton', 'electron'), ('lepton', 'muon'), ('quark', 'up'), ('quark', 'down')))
<IndexHierarchy>
lepton           electron
lepton           muon
quark            up
quark            down
<<U6>            <<U8>

#end_IndexHierarchy-from_labels()


#-------------------------------------------------------------------------------
# IndexHierarchyGO

#start_IndexHierarchyGO-interface
>>> sf.IndexHierarchyGO.interface.loc[sf.IndexHierarchyGO.interface.index.via_str.startswith('re')]
<Frame: IndexHierarchyGO>
<Index>                   cls_name         group  doc                  <<U18>
<Index: signature>
rehierarch(depth_map)     IndexHierarchyGO Method Return a new Inde...
relabel(mapper)           IndexHierarchyGO Method Return a new Inde...
rename(name)              IndexHierarchyGO Method Return a new Fram...
<<U68>                    <<U16>           <<U17> <<U83>

#end_IndexHierarchyGO-interface



#start_IndexHierarchyGO-from_labels()
>>> ih = sf.IndexHierarchyGO.from_labels((('lepton', 'electron'), ('lepton', 'muon'), ('quark', 'up'), ('quark', 'down')))
>>> ih
<IndexHierarchyGO>
lepton             electron
lepton             muon
quark              up
quark              down
<<U6>              <<U8>

#end_IndexHierarchyGO-from_labels()



#start_IndexHierarchyGO-append()
>>> ih = sf.IndexHierarchyGO.from_labels((('lepton', 'electron'), ('lepton', 'muon'), ('quark', 'up'), ('quark', 'down')))
>>> ih
<IndexHierarchyGO>
lepton             electron
lepton             muon
quark              up
quark              down
<<U6>              <<U8>
>>> ih.append(('quark', 'strange'))
>>> ih
<IndexHierarchyGO>
lepton             electron
lepton             muon
quark              up
quark              down
quark              strange
<<U6>              <<U8>

#end_IndexHierarchyGO-append()


#-------------------------------------------------------------------------------
# IndexYear

#start_IndexYear-interface
>>> sf.IndexYear.interface.loc[sf.IndexYear.interface.index.via_str.startswith('from_')]
<Frame: IndexYear>
<Index>                              cls_name  group       doc                  <<U18>
<Index: signature>
from_date_range(start, stop, step... IndexYear Constructor Get an IndexYearM...
from_labels(labels, *, name)         IndexYear Constructor Construct an Inde...
from_pandas(value)                   IndexYear Constructor Given a Pandas in...
from_year_month_range(start, stop... IndexYear Constructor Get an IndexYearM...
from_year_range(start, stop, step... IndexYear Constructor Get an IndexDate ...
<<U68>                               <<U9>     <<U17>      <<U83>

#end_IndexYear-interface

#-------------------------------------------------------------------------------
# IndexYearGO

#start_IndexYearGO-interface
>>> sf.IndexYearGO.interface.loc[sf.IndexYearGO.interface.index.via_str.startswith('from_')]
<Frame: IndexYearGO>
<Index>                              cls_name    group       doc                  <<U18>
<Index: signature>
from_date_range(start, stop, step... IndexYearGO Constructor Get an IndexYearM...
from_labels(labels, *, name)         IndexYearGO Constructor Construct an Inde...
from_pandas(value)                   IndexYearGO Constructor Given a Pandas in...
from_year_month_range(start, stop... IndexYearGO Constructor Get an IndexYearM...
from_year_range(start, stop, step... IndexYearGO Constructor Get an IndexDate ...
<<U68>                               <<U11>      <<U17>      <<U83>

#end_IndexYearGO-interface


#-------------------------------------------------------------------------------
# IndexYearMonth

#start_IndexYearMonth-interface
>>> sf.IndexYearMonthGO.interface.loc[sf.IndexYearMonthGO.interface.index.via_str.startswith('from_')]
<Frame: IndexYearMonthGO>
<Index>                              cls_name         group       doc                  <<U18>
<Index: signature>
from_date_range(start, stop, step... IndexYearMonthGO Constructor Get an IndexYearM...
from_labels(labels, *, name)         IndexYearMonthGO Constructor Construct an Inde...
from_pandas(value)                   IndexYearMonthGO Constructor Given a Pandas in...
from_year_month_range(start, stop... IndexYearMonthGO Constructor Get an IndexYearM...
from_year_range(start, stop, step... IndexYearMonthGO Constructor Get an IndexYearM...
<<U68>                               <<U16>           <<U17>      <<U83>

#end_IndexYearMonth-interface


#-------------------------------------------------------------------------------
# IndexYearMonthGO

#start_IndexYearMonthGO-interface
>>> sf.IndexYearMonthGO.interface.loc[sf.IndexYearMonthGO.interface.index.via_str.startswith('from_')]
<Frame: IndexYearMonthGO>
<Index>                              cls_name         group       doc                  <<U18>
<Index: signature>
from_date_range(start, stop, step... IndexYearMonthGO Constructor Get an IndexYearM...
from_labels(labels, *, name)         IndexYearMonthGO Constructor Construct an Inde...
from_pandas(value)                   IndexYearMonthGO Constructor Given a Pandas in...
from_year_month_range(start, stop... IndexYearMonthGO Constructor Get an IndexYearM...
from_year_range(start, stop, step... IndexYearMonthGO Constructor Get an IndexYearM...
<<U68>                               <<U16>           <<U17>      <<U83>

#end_IndexYearMonthGO-interface


#-------------------------------------------------------------------------------
# IndexDate

#start_IndexDate-interface
>>> sf.IndexDate.interface.loc[sf.IndexDate.interface.index.via_str.startswith('from_')]
<Frame: IndexDate>
<Index>                              cls_name  group       doc                  <<U18>
<Index: signature>
from_date_range(start, stop, step... IndexDate Constructor Get an IndexDate ...
from_labels(labels, *, name)         IndexDate Constructor Construct an Inde...
from_pandas(value)                   IndexDate Constructor Given a Pandas in...
from_year_month_range(start, stop... IndexDate Constructor Get an IndexDate ...
from_year_range(start, stop, step... IndexDate Constructor Get an IndexDate ...
<<U68>                               <<U9>     <<U17>      <<U83>

#end_IndexDate-interface


#-------------------------------------------------------------------------------
# IndexDateGO

#start_IndexDateGO-interface
>>> sf.IndexDateGO.interface.loc[sf.IndexDateGO.interface.index.via_str.startswith('from_')]
<Frame: IndexDateGO>
<Index>                              cls_name    group       doc                  <<U18>
<Index: signature>
from_date_range(start, stop, step... IndexDateGO Constructor Get an IndexDate ...
from_labels(labels, *, name)         IndexDateGO Constructor Construct an Inde...
from_pandas(value)                   IndexDateGO Constructor Given a Pandas in...
from_year_month_range(start, stop... IndexDateGO Constructor Get an IndexDate ...
from_year_range(start, stop, step... IndexDateGO Constructor Get an IndexDate ...
<<U68>                               <<U11>      <<U17>      <<U83>

#end_IndexDateGO-interface




#-------------------------------------------------------------------------------
# restore initial configuration
>>> sf.DisplayActive.set(_display_config_active)


'''


from static_frame.test.test_case import TestCase
from static_frame.test.test_case import skip_win

@skip_win
class TestUnit(doctest.DocTestCase, TestCase):

    @staticmethod
    def get_readme_fp() -> str:

        target_fn = 'README.rst'

        fp = os.path.join(os.getcwd(), __file__)
        if not os.path.exists(fp):
            raise RuntimeError('got bad module path', fp)

        while len(fp) > len(os.sep):
            fp = os.path.dirname(fp)
            if target_fn in os.listdir(fp):
                return os.path.join(fp, target_fn)

        raise RuntimeError('could not find target fn', target_fn)

    @classmethod
    def get_readme_str(cls) -> str:
        # mutate the README
        fp_alt = cls.get_test_input('jph_photos.txt')

        readme_fp = cls.get_readme_fp()
        with open(readme_fp) as f:
            readme_str = f.read()

        # update display config to remove colors
        readme_str = '''
>>> _display_config = sf.DisplayActive.get()
>>> sf.DisplayActive.update(type_color=False)
>>>
        ''' + readme_str

        # inject content from local files
        src = ">>> frame = sf.Frame.from_json_url('https://jsonplaceholder.typicode.com/photos', dtypes=dict(albumId=np.int64, id=np.int64))"

        # using a raw string to avoid unicode decoding issues on windows
        dst = ">>> frame = sf.Frame.from_tsv(r'%s', dtypes=dict(albumId=np.int64, id=np.int64), encoding='utf-8')" % fp_alt

        if src not in readme_str:
            raise RuntimeError('did not find expected string')

        readme_str = readme_str.replace(src, dst)

        # restore active config
        readme_str = readme_str + '''
>>> sf.DisplayActive.set(_display_config)
        '''

        return readme_str

    @staticmethod
    def update_readme(source: object) -> None:
        target = "sf.Frame.from_json_url('https://jsonplaceholder.typicode.com/photos')"


    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:

        doctest_str = '\n'.join((api_example_str, self.get_readme_str()))

        sample = doctest.DocTestParser().get_doctest(
                doctest_str,
                globs={},
                name='test_doc',
                filename=None,
                lineno=None)

        super().__init__(sample, **kwargs)


if __name__ == "__main__":
    import unittest
    unittest.main()





# UNUSED


# #start_Frame-from_records()
# >>> sf.Frame.from_records(((-65, 227.9), (-200, 4495.1)), columns=('temperature', 'distance'), index=('Mars', 'Neptune'), dtypes=dict(temperature=np.int64))
# <Frame>
# <Index> temperature distance  <<U11>
# <Index>
# Mars    -65         227.9
# Neptune -200        4495.1
# <<U7>   <int64>     <float64>
# #end_Frame-from_records()
