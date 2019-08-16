import doctest
import os
import typing as tp

api_example_str = '''


#-------------------------------------------------------------------------------
# information

>>> import static_frame as sf
>>> _display_config_active = sf.DisplayActive.get()
>>> sf.DisplayActive.set(sf.DisplayConfig(type_color=False))


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
# API docs


#start_series_a
>>> import numpy as np
>>> import static_frame as sf

>>> sf.Series(dict(Mercury=167, Neptune=-200), dtype=np.int64)
<Series>
<Index>
Mercury  167
Neptune  -200
<<U7>    <int64>

>>> sf.Series((167, -200), index=('Mercury', 'Neptune'), dtype=np.int64)
<Series>
<Index>
Mercury  167
Neptune  -200
<<U7>    <int64>

#end_series_a


#start_frame_a
>>> sf.Frame(((-65, 227.9), (-200, 4495.1)), columns=('temperature', 'distance'), index=('Mars', 'Neptune'))
<Frame>
<Index> temperature distance  <<U11>
<Index>
Mars    -65.0       227.9
Neptune -200.0      4495.1
<<U7>   <float64>   <float64>

>>> sf.Frame.from_dict(dict(diameter=(12756, 142984, 120536), mass=(5.97, 1898, 568)), index=('Earth', 'Jupiter', 'Saturn'), dtypes=dict(diameter=np.int64))
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    5.97
Jupiter 142984   1898.0
Saturn  120536   568.0
<<U7>   <int64>  <float64>

#end_frame_a



#start_framego_a
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

#end_framego_a


#start_index_a
>>> sf.Index(('Mercury', 'Mars'), dtype=object)
<Index>
Mercury
Mars
<object>

>>> sf.Index(name[:2].upper() for name in ('Mercury', 'Mars'))
<Index>
ME
MA
<object>

#end_index_a


#start_indexgo_a
>>> a = sf.IndexGO(('Uranus', 'Neptune'))
>>> a.append('Pluto')
>>> a
<IndexGO>
Uranus
Neptune
Pluto
<<U7>

#end_indexgo_a


#start_series_from_items_a
>>> sf.Series.from_items(zip(('Mercury', 'Jupiter'), (4879, 12756)), dtype=np.int64)
<Series>
<Index>
Mercury  4879
Jupiter  12756
<<U7>    <int64>

#end_series_from_items_a






#start_frame_from_records_a
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

#end_frame_from_records_a




#start_frame_from_items_a
>>> sf.Frame.from_items((('diameter', (12756, 142984, 120536)), ('mass', (5.97, 1898, 568))), index=('Earth', 'Jupiter', 'Saturn'), dtypes=dict(diameter=np.int64))
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    5.97
Jupiter 142984   1898.0
Saturn  120536   568.0
<<U7>   <int64>  <float64>

#end_frame_from_items_a



#start_frame_from_concat_a
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

#end_frame_from_concat_a



#start_frame_from_structured_array_a
>>> a = np.array([('Venus', 4.87, 464), ('Neptune', 102, -200)], dtype=[('name', object), ('mass', 'f4'), ('temperature', 'i4')])
>>> sf.Frame.from_structured_array(a, index_column='name')
<Frame>
<Index>  mass              temperature <<U11>
<Index>
Venus    4.869999885559082 464
Neptune  102.0             -200
<object> <float32>         <int32>


#end_frame_from_structured_array_a



#start_frame_from_csv_a
>>> from io import StringIO
>>> filelike = StringIO('name,mass,temperature\\nVenus,4.87,464\\nNeptune,102,-200')
>>> sf.Frame.from_csv(filelike, index_column='name', dtypes=dict(temperature=np.int64))
<Frame>
<Index> mass      temperature <<U11>
<Index>
Venus   4.87      464
Neptune 102.0     -200
<<U7>   <float64> <int64>

#end_frame_from_csv_a


#start_series_dict_like_a
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
>>> len(s)
6
>>> [k for k, v in s.items() if v > 60]
['Jupiter', 'Saturn']
>>> [s.get(k, None) for k in ('Mercury', 'Neptune', 'Pluto')]
[None, 14, None]
>>> 'Pluto' in s
False
>>> s.values.flags
  C_CONTIGUOUS : True
  F_CONTIGUOUS : True
  OWNDATA : True
  WRITEABLE : False
  ALIGNED : True
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False

#end_series_dict_like_a




#start_frame_dict_like_a
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
>>> f.get('diameter')
<Series: diameter>
<Index>
Earth              12756
Jupiter            142984
Saturn             120536
<<U7>              <int64>
>>> f.get('mass', np.nan)
nan
>>> 'temperature' in f
True
>>> f.values.tolist()
[[12756, 15], [142984, -110], [120536, -140]]

>>> f.values.flags
  C_CONTIGUOUS : True
  F_CONTIGUOUS : False
  OWNDATA : True
  WRITEABLE : False
  ALIGNED : True
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False

#end_frame_dict_like_a









#start_series_operators_a
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

>>> s > s['Earth']
<Series>
<Index>
Venus    False
Earth    False
Saturn   True
<<U6>    <bool>

>>> s / s['Earth']
<Series>
<Index>
Venus    0.7232620320855615
Earth    1.0
Saturn   9.582219251336898
<<U6>    <float64>

#end_series_operators_a


#start_series_operators_b
>>> s1 = sf.Series((1, 2), index=('Earth', 'Mars'))
>>> s2 = sf.Series((2, 0), index=('Mars', 'Mercury'))
>>> s1 * s2
<Series>
<Index>
Earth    nan
Mars     4.0
Mercury  nan
<<U7>    <float64>

>>> s1 == s2
<Series>
<Index>
Earth    False
Mars     True
Mercury  False
<<U7>    <bool>

#end_series_operators_b





#start_frame_operators_a
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

#end_frame_operators_a




#start_frame_math_logic_a
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
>>> f.min()
<Series>
<Index>
diameter 12756.0
mass     5.97
<<U8>    <float64>
>>> f.std()
<Series>
<Index>
diameter 56842.64155250587
mass     793.344204533358
<<U8>    <float64>
>>> f.sum()
<Series>
<Index>
diameter 276276.0
mass     2471.9700000000003
<<U8>    <float64>
>>> f.mean()
<Series>
<Index>
diameter 92092.0
mass     823.9900000000001
<<U8>    <float64>

#end_frame_math_logic_a


#start_index_relabel_a
>>> index = sf.Index(('Venus', 'Saturn', 'Neptune'))
>>> index.relabel({'Venus': 'Mars'})
<Index>
Mars
Saturn
Neptune
<object>

>>> index = sf.Index(('Venus', 'Saturn', 'Neptune'))
>>> index.relabel({'Neptune': 'Uranus'})
<Index>
Venus
Saturn
Uranus
<object>

>>> index.relabel(lambda x: x[:2].upper())
<Index>
VE
SA
NE
<object>

#end_index_relabel_a


#start_series_relabel_a
>>> s = sf.Series((0, 62, 13), index=('Venus', 'Saturn', 'Neptune'), dtype=np.int64)

>>> s.relabel({'Venus': 'Mercury'})
<Series>
<Index>
Mercury  0
Saturn   62
Neptune  13
<object> <int64>
>>> s.relabel(lambda x: x[:2].upper())
<Series>
<Index>
VE       0
SA       62
NE       13
<object> <int64>

#end_series_relabel_a

#start_series_reindex_a
>>> s = sf.Series((0, 62, 13), index=('Venus', 'Saturn', 'Neptune'))

>>> s.reindex(('Venus', 'Earth', 'Mars', 'Neptune'))
<Series>
<Index>
Venus    0.0
Earth    nan
Mars     nan
Neptune  13.0
<<U7>    <float64>

#end_series_reindex_a



#start_frame_relabel_a
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'), dtypes=dict(diameter=np.int64))

>>> f.relabel(index=lambda x: x[:2].upper(), columns={'mass': 'mass(1e24kg)'})
<Frame>
<Index>  diameter mass(1e24kg) <object>
<Index>
EA       12756    5.97
MA       6792     0.642
JU       142984   1898.0
<object> <int64>  <float64>

#end_frame_relabel_a


#start_frame_reindex_a
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

#end_frame_reindex_a



#start_series_iter_element_a
>>> s = sf.Series((1, 2, 67, 62, 27, 14), index=('Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'))
>>> [x for x in s.iter_element()]
[1, 2, 67, 62, 27, 14]

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
>>> [x for x in s.iter_element().apply_iter(lambda v: v > 20)]
[False, False, True, True, True, False]

#end_series_iter_element_a


#start_series_iter_element_items_a
>>> s = sf.Series((1, 2, 67, 62, 27, 14), index=('Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'))

>>> [x for x in s.iter_element_items()]
[('Earth', 1), ('Mars', 2), ('Jupiter', 67), ('Saturn', 62), ('Uranus', 27), ('Neptune', 14)]

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

>>> [x for x in s.iter_element_items().apply_iter_items(lambda k, v: k.upper() if v > 20 else None)]
[('Earth', None), ('Mars', None), ('Jupiter', 'JUPITER'), ('Saturn', 'SATURN'), ('Uranus', 'URANUS'), ('Neptune', None)]

#end_series_iter_element_items_a




#start_frame_iter_element_a
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'), dtypes=dict(diameter=np.int64))

>>> [x for x in f.iter_element()]
[12756, 5.97, 6792, 0.642, 142984, 1898.0]

>>> f.iter_element().apply(lambda x: x ** 2)
<Frame>
<Index> diameter    mass                <<U8>
<Index>
Earth   162715536   35.640899999999995
Mars    46131264    0.41216400000000003
Jupiter 20444424256 3602404.0
<<U7>   <object>    <object>

#end_frame_iter_element_a



#start_frame_iter_element_items_a
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'))

>>> [x for x in f.iter_element_items()]
[(('Earth', 'diameter'), 12756), (('Earth', 'mass'), 5.97), (('Mars', 'diameter'), 6792), (('Mars', 'mass'), 0.642), (('Jupiter', 'diameter'), 142984), (('Jupiter', 'mass'), 1898.0)]

>>> f.iter_element_items().apply(lambda k, v: v ** 2 if k[0] == 'Mars' else None)
<Frame>
<Index> diameter mass                <<U8>
<Index>
Earth   None     None
Mars    46131264 0.41216400000000003
Jupiter None     None
<<U7>   <object> <object>

#end_frame_iter_element_items_a


#start_frame_iter_array_a
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

>>> f.iter_array(axis=0).apply(np.sum)
<Series>
<Index>
diameter 162532.0
mass     1904.612
<<U8>    <float64>

#end_frame_iter_array_a


#start_frame_iter_array_items_a
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

#end_frame_iter_array_items_a



#start_frame_iter_tuple_a
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'))

>>> [x for x in f.iter_tuple(axis=0)]
[Axis(Earth=12756, Mars=6792, Jupiter=142984), Axis(Earth=5.97, Mars=0.642, Jupiter=1898.0)]

>>> [x for x in f.iter_tuple(axis=1)]
[Axis(diameter=12756.0, mass=5.97), Axis(diameter=6792.0, mass=0.642), Axis(diameter=142984.0, mass=1898.0)]

>>> f.iter_tuple(1).apply(lambda nt: nt.mass / (4 / 3 * np.pi * (nt.diameter * 0.5) ** 3))
<Series>
<Index>
Earth    5.49328558e-12
Mars     3.91330208e-12
Jupiter  1.24003876e-12
<<U7>    <float64>

#end_frame_iter_tuple_a



#start_frame_iter_tuple_items_a
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'))

>>> [x for x in f.iter_tuple_items(axis=0)]
[('diameter', Axis(Earth=12756, Mars=6792, Jupiter=142984)), ('mass', Axis(Earth=5.97, Mars=0.642, Jupiter=1898.0))]

>>> [x for x in f.iter_tuple_items(axis=1)]
[('Earth', Axis(diameter=12756.0, mass=5.97)), ('Mars', Axis(diameter=6792.0, mass=0.642)), ('Jupiter', Axis(diameter=142984.0, mass=1898.0))]

>>> f.iter_tuple_items(axis=1).apply(lambda k, v: v.diameter if k == 'Earth' else 0)
<Series>
<Index>
Earth    12756.0
Mars     0.0
Jupiter  0.0
<<U7>    <float64>

#end_frame_iter_tuple_items_a




#start_frame_iter_series_a
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'), dtypes=dict(diameter=np.int64))

>>> next(iter(f.iter_series(axis=0)))
<Series>
<Index>
Earth    12756
Mars     6792
Jupiter  142984
<<U7>    <int64>

>>> next(iter(f.iter_series(axis=1)))
<Series>
<Index>
diameter 12756.0
mass     5.97
<<U8>    <float64>

>>> f.iter_series(0).apply(lambda s: s.mean())
<Series>
<Index>
diameter 54177.333333333336
mass     634.8706666666667
<<U8>    <float64>

#end_frame_iter_series_a



#start_frame_iter_series_items_a
>>> f = sf.Frame.from_dict(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'))

>>> [(k, v.mean()) for k, v in f.iter_series_items(0)]
[('diameter', 54177.333333333336), ('mass', 634.8706666666667)]

>>> [(k, v.max()) for k, v in f.iter_series_items(1)]
[('Earth', 12756.0), ('Mars', 6792.0), ('Jupiter', 142984.0)]

>>> f.iter_series_items(0).apply(lambda k, v: v.mean() if k == 'diameter' else v.sum())
<Series>
<Index>
diameter 54177.333333333336
mass     1904.612
<<U8>    <float64>

#end_frame_iter_series_items_a


#start_series_iter_group_a
>>> s = sf.Series((0, 0, 1, 2), index=('Mercury', 'Venus', 'Earth', 'Mars'), dtype=np.int64)
>>> next(iter(s.iter_group()))
<Series>
<Index>
Mercury  0
Venus    0
<<U7>    <int64>
>>> [x.values.tolist() for x in s.iter_group()]
[[0, 0], [1], [2]]

#end_series_iter_group_a


#start_series_iter_group_items_a
>>> s = sf.Series((0, 0, 1, 2), index=('Mercury', 'Venus', 'Earth', 'Mars'))
>>> [(k, v.index.values.tolist()) for k, v in iter(s.iter_group_items()) if k > 0]
[(1, ['Earth']), (2, ['Mars'])]

#end_series_iter_group_items_a


#start_frame_iter_group_a
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

#end_frame_iter_group_a


#start_frame_iter_group_items_a
>>> f = sf.Frame.from_dict(dict(mass=(0.33, 4.87, 5.97, 0.642), moons=(0, 0, 1, 2)), index=('Mercury', 'Venus', 'Earth', 'Mars'))
>>> [(k, v.index.values.tolist(), v['mass'].mean()) for k, v in f.iter_group_items('moons')]
[(0, ['Mercury', 'Venus'], 2.6), (1, ['Earth'], 5.97), (2, ['Mars'], 0.642)]

#end_frame_iter_group_items_a


#start_series_assign_a
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
>>> s.assign.loc[s < 150](0)
<Series>
<Index>
Venus    0.0
Earth    0.0
Saturn   1433.5
<<U6>    <float64>
>>> s.assign.iloc[-1](0)
<Series>
<Index>
Venus    108.2
Earth    149.6
Saturn   0.0
<<U6>    <float64>

#end_series_assign_a

#start_frame_assign_a
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

#end_frame_assign_a



#start_series_drop_a
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
>>> s.drop.iloc[-2:]
<Series>
<Index>
Mercury  0
Venus    0
<<U7>    <int64>

#end_series_drop_a

#start_frame_drop_a
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
>>> f.drop.loc[f['temperature'] < 0]
<Frame>
<Index> diameter temperature <<U11>
<Index>
Earth   12756    15
<<U7>   <int64>  <int64>
>>> f.drop.iloc[-1, -1]
<Frame>
<Index> diameter <<U11>
<Index>
Earth   12756
Jupiter 142984
<<U7>   <int64>

#end_frame_drop_a





#start_series_shape_a
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
>>> s.ndim
1
>>> s.size
6
>>> s.nbytes
48
>>> s.dtype
dtype('int64')

#end_series_shape_a


#start_frame_shape_a
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
>>> f.ndim
2
>>> f.size
6
>>> f.nbytes
48
>>> f.dtypes
<Series>
<Index>
diameter int64
mass     float64
<<U8>    <object>

#end_frame_shape_a



#start_series_selection_a
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

>>> s.iloc[-2:]
<Series>
<Index>
Uranus   27
Neptune  14
<<U7>    <int64>

#end_series_selection_a



#start_frame_selection_a
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

>>> f.iloc[-2:, -1]
<Series: temperature>
<Index>
Earth                 15
Mars                  -65
<<U7>                 <int64>

#end_frame_selection_a



# restore initial configuration
>>> sf.DisplayActive.set(_display_config_active)


'''


from static_frame.test.test_case import TestCase


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
