



'''
#start_series_a
>>> import numpy as np
>>> import static_frame as sf

>>> sf.Series(dict(Mercury=167, Neptune=-200))
<Index> <Series>
Mercury 167.0
Neptune -200.0
<<U7>   <float64>

>>> sf.Series((167, -200), index=('Mercury', 'Neptune'))
<Index> <Series>
Mercury 167
Neptune -200
<<U7>   <int64>

#end_series_a


#start_frame_a
>>> sf.Frame(((-65, 227.9), (-200, 4495.1)), columns=('temperature', 'distance'), index=('Mars', 'Neptune'))
<Frame>
<Index> temperature distance  <<U11>
<Index>
Mars    -65.0       227.9
Neptune -200.0      4495.1
<<U7>   <float64>   <float64>

>>> sf.Frame(dict(diameter=(12756, 142984, 120536), mass=(5.97, 1898, 568)), index=('Earth', 'Jupiter', 'Saturn'))
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    5.97
Jupiter 142984   1898.0
Saturn  120536   568.0
<<U7>   <int64>  <float64>

#end_frame_a



#start_framego_a
>>> f = sf.FrameGO(dict(diameter=(12756, 142984, 120536), mass=(5.97, 1898, 568)), index=('Earth', 'Jupiter', 'Saturn'))
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
>>> sf.Series.from_items(zip(('Mercury', 'Jupiter'), (4879, 12756)))
<Index> <Series>
Mercury 4879
Jupiter 12756
<<U7>   <int64>

#end_series_from_items_a






#start_frame_from_records_a
>>> index = ('Mercury', 'Venus', 'Earth', 'Mars')
>>> columns = ('diameter', 'gravity', 'temperature')
>>> records = ((4879, 3.7, 167), (12104, 8.9, 464), (12756, 9.8, 15), (6792, 3.7, -65))
>>> sf.Frame.from_records(records, index=index, columns=columns)
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
>>> sf.Frame.from_items((('diameter', (12756, 142984, 120536)), ('mass', (5.97, 1898, 568))), index=('Earth', 'Jupiter', 'Saturn'))
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    5.97
Jupiter 142984   1898.0
Saturn  120536   568.0
<<U7>   <int64>  <float64>

#end_frame_from_items_a



#start_frame_from_concat_a
>>> f1 = sf.Frame(dict(diameter=(12756, 142984), mass=(5.97, 1898)), index=('Earth', 'Jupiter'))
>>> f2 = sf.Frame(dict(mass=(0.642, 102), moons=(2, 14)), index=('Mars', 'Neptune'))
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
<Index>  mass      temperature <<U11>
<Index>
Venus    4.87      464
Neptune  102.0     -200
<object> <float32> <int32>

#end_frame_from_structured_array_a



#start_frame_from_csv_a
>>> from io import StringIO
>>> filelike = StringIO('name,mass,temperature\\nVenus,4.87,464\\nNeptune,102,-200')
>>> sf.Frame.from_csv(filelike, index_column='name')
<Frame>
<Index> mass      temperature <<U11>
<Index>
Venus   4.87      464
Neptune 102.0     -200
<<U7>   <float64> <int64>

#end_frame_from_csv_a


#start_series_dict_like_a
>>> s = sf.Series((1, 2, 67, 62, 27, 14), index=('Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'))
>>> s
<Index> <Series>
Earth   1
Mars    2
Jupiter 67
Saturn  62
Uranus  27
Neptune 14
<<U7>   <int64>
>>> len(s)
6
>>> [k for k, v in s.items() if v > 60]
['Jupiter', 'Saturn']
>>> [s.get(k, None) for k in ('Mercury', 'Neptune', 'Pluto')]
[None, 14, None]
>>> 'Pluto' in s
False
>>> s.values
array([ 1,  2, 67, 62, 27, 14])
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
>>> f = sf.Frame(dict(diameter=(12756, 142984, 120536), temperature=(15, -110, -140)), index=('Earth', 'Jupiter', 'Saturn'))
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
<Index> <Series>
Earth   12756
Jupiter 142984
Saturn  120536
<<U7>   <int64>
>>> f.get('mass', np.nan)
nan
>>> 'temperature' in f
True
>>> f.values
array([[ 12756,     15],
       [142984,   -110],
       [120536,   -140]])

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
<Index> <Series>
Venus   108.2
Earth   149.6
Saturn  1433.5
<<U6>   <float64>

>>> abs(s - s['Earth'])
<Index> <Series>
Venus   41.39999999999999
Earth   0.0
Saturn  1283.9
<<U6>   <float64>

>>> s > s['Earth']
<Index> <Series>
Venus   False
Earth   False
Saturn  True
<<U6>   <bool>

>>> s / s['Earth']
<Index> <Series>
Venus   0.7232620320855615
Earth   1.0
Saturn  9.582219251336898
<<U6>   <float64>

#end_series_operators_a


#start_series_operators_b
>>> s1 = sf.Series((1, 2), index=('Earth', 'Mars'))
>>> s2 = sf.Series((2, 0), index=('Mars', 'Mercury'))
>>> s1 * s2
<Index> <Series>
Earth   nan
Mars    4
Mercury nan
<<U7>   <object>

>>> s1 == s2
<Index> <Series>
Earth   False
Mars    True
Mercury False
<<U7>   <bool>

#end_series_operators_b





#start_frame_operators_a
>>> f = sf.Frame(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'))
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
>>> f = sf.Frame(dict(diameter=(12756, 142984, 120536), mass=(5.97, 1898, 568)), index=('Earth', 'Jupiter', 'Saturn'))
>>> f
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    5.97
Jupiter 142984   1898.0
Saturn  120536   568.0
<<U7>   <int64>  <float64>

>>> f.max()
<Index>  <Series>
diameter 142984.0
mass     1898.0
<<U8>    <float64>
>>> f.min()
<Index>  <Series>
diameter 12756.0
mass     5.97
<<U8>    <float64>
>>> f.std()
<Index>  <Series>
diameter 56842.64155250587
mass     793.344204533358
<<U8>    <float64>
>>> f.sum()
<Index>  <Series>
diameter 276276.0
mass     2471.9700000000003
<<U8>    <float64>
>>> f.mean()
<Index>  <Series>
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
>>> s = sf.Series((0, 62, 13), index=('Venus', 'Saturn', 'Neptune'))

>>> s.relabel({'Venus': 'Mercury'})
<Index>  <Series>
Mercury  0
Saturn   62
Neptune  13
<object> <int64>
>>> s.relabel(lambda x: x[:2].upper())
<Index>  <Series>
VE       0
SA       62
NE       13
<object> <int64>

#end_series_relabel_a

#start_series_reindex_a
>>> s = sf.Series((0, 62, 13), index=('Venus', 'Saturn', 'Neptune'))

>>> s.reindex(('Venus', 'Earth', 'Mars', 'Neptune'))
<Index> <Series>
Venus   0
Earth   nan
Mars    nan
Neptune 13
<<U7>   <object>

#end_series_reindex_a



#start_frame_relabel_a
>>> f = sf.Frame(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'))

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
>>> f = sf.Frame(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'))
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
<Index> <Series>
Earth   False
Mars    False
Jupiter True
Saturn  True
Uranus  True
Neptune False
<<U7>   <bool>
>>> [x for x in s.iter_element().apply_iter(lambda v: v > 20)]
[False, False, True, True, True, False]

#end_series_iter_element_a


#start_series_iter_element_items_a
>>> s = sf.Series((1, 2, 67, 62, 27, 14), index=('Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'))

>>> [x for x in s.iter_element_items()]
[('Earth', 1), ('Mars', 2), ('Jupiter', 67), ('Saturn', 62), ('Uranus', 27), ('Neptune', 14)]

>>> s.iter_element_items().apply(lambda k, v: v if 'u' in k else None)
<Index> <Series>
Earth   None
Mars    None
Jupiter 67
Saturn  62
Uranus  27
Neptune 14
<<U7>   <object>

>>> [x for x in s.iter_element_items().apply_iter_items(lambda k, v: k.upper() if v > 20 else None)]
[('Earth', None), ('Mars', None), ('Jupiter', 'JUPITER'), ('Saturn', 'SATURN'), ('Uranus', 'URANUS'), ('Neptune', None)]

#end_series_iter_element_items_a




#start_frame_iter_element_a
>>> f = sf.Frame(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'))

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
>>> f = sf.Frame(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'))

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
>>> f = sf.Frame(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'))
>>> f
<Frame>
<Index> diameter mass      <<U8>
<Index>
Earth   12756    5.97
Mars    6792     0.642
Jupiter 142984   1898.0
<<U7>   <int64>  <float64>

>>> [x for x in f.iter_array(axis=0)]
[array([ 12756,   6792, 142984]), array([5.970e+00, 6.420e-01, 1.898e+03])]
>>> [x for x in f.iter_array(axis=1)]

[array([1.2756e+04, 5.9700e+00]), array([6.792e+03, 6.420e-01]), array([142984.,   1898.])]

>>> f.iter_array(axis=0).apply(np.sum)
<Index>  <Series>
diameter 162532.0
mass     1904.612
<<U8>    <float64>

#end_frame_iter_array_a


#start_frame_iter_array_items_a
>>> f = sf.Frame(dict(diameter=(12756, 6792, 142984), mass=(5.97, 0.642, 1898)), index=('Earth', 'Mars', 'Jupiter'))

>>> [x for x in f.iter_array_items(axis=0)]
[('diameter', array([ 12756,   6792, 142984])), ('mass', array([5.970e+00, 6.420e-01, 1.898e+03]))]

>>> [x for x in f.iter_array_items(axis=1)]
[('Earth', array([1.2756e+04, 5.9700e+00])), ('Mars', array([6.792e+03, 6.420e-01])), ('Jupiter', array([142984.,   1898.]))]

>>> f.iter_array_items(axis=1).apply(lambda k, v: v.sum() if k == 'Earth' else 0)
<Index> <Series>
Earth   12761.97
Mars    0.0
Jupiter 0.0
<<U7>   <float64>

#end_frame_iter_array_items_a


'''


if __name__ == "__main__":
    import doctest
    doctest.testmod()
