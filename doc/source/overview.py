



'''
#start_series_a
>>> import numpy as np
>>> import static_frame as sf

>>> sf.Series(dict(Mercury=167, Neptune=-200)) #doctest: +NORMALIZE_WHITESPACE
<Index> <Series>
Mercury 167.0
Neptune -200.0
<<U7>   <float64>

>>> sf.Series((167, -200), index=('Mercury', 'Neptune')) #doctest: +NORMALIZE_WHITESPACE
<Index> <Series>
Mercury 167
Neptune -200
<<U7>   <int64>

#end_series_a


#start_frame_a
>>> sf.Frame(((-65, 227.9), (-200, 4495.1)), columns=('temperature', 'distance'), index=('Mars', 'Neptune')) #doctest: +NORMALIZE_WHITESPACE
<Frame>
<Index> temperature distance  <<U11>
<Index>
Mars    -65.0       227.9
Neptune -200.0      4495.1
<<U7>   <float64>   <float64>

>>> sf.Frame(dict(diameter=(12756, 142984, 120536), mass=(5.97, 1898, 568)), index=('Earth', 'Jupiter', 'Saturn')) #doctest: +NORMALIZE_WHITESPACE
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
>>> f #doctest: +NORMALIZE_WHITESPACE
<FrameGO>
<IndexGO> diameter mass      radius    <<U8>
<Index>
Earth     12756    5.97      6378.0
Jupiter   142984   1898.0    71492.0
Saturn    120536   568.0     60268.0
<<U7>     <int64>  <float64> <float64>

#end_framego_a


#start_index_a
>>> sf.Index(('Mercury', 'Mars'), dtype=object) #doctest: +NORMALIZE_WHITESPACE
<Index>
Mercury
Mars
<object>

>>> sf.Index(name[:2].upper() for name in ('Mercury', 'Mars')) #doctest: +NORMALIZE_WHITESPACE
<Index>
ME
MA
<object>

#end_index_a


#start_indexgo_a
>>> a = sf.IndexGO(('Uranus', 'Neptune')) #doctest: +NORMALIZE_WHITESPACE
>>> a.append('Pluto')
>>> a #doctest: +NORMALIZE_WHITESPACE
<IndexGO>
Uranus
Neptune
Pluto
<<U7>

#end_indexgo_a


#start_series_from_items_a
>>> sf.Series.from_items(zip(('Mercury', 'Jupiter'), (4879, 12756))) #doctest: +NORMALIZE_WHITESPACE
<Index> <Series>
Mercury 4879
Jupiter 12756
<<U7>   <int64>

#end_series_from_items_a






#start_frame_from_records_a
>>> index = ('Mercury', 'Venus', 'Earth', 'Mars')
>>> columns = ('diameter', 'gravity', 'temperature')
>>> records = ((4879, 3.7, 167), (12104, 8.9, 464), (12756, 9.8, 15), (6792, 3.7, -65))
>>> sf.Frame.from_records(records, index=index, columns=columns) #doctest: +NORMALIZE_WHITESPACE
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
>>> sf.Frame.from_items((('diameter', (12756, 142984, 120536)), ('mass', (5.97, 1898, 568))), index=('Earth', 'Jupiter', 'Saturn')) #doctest: +NORMALIZE_WHITESPACE
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
>>> sf.Frame.from_concat((f1, f2)) #doctest: +NORMALIZE_WHITESPACE
<Frame>
<Index> diameter  mass      moons     <<U8>
<Index>
Earth   12756.0   5.97      nan
Jupiter 142984.0  1898.0    nan
Mars    nan       0.642     2.0
Neptune nan       102.0     14.0
<<U7>   <float64> <float64> <float64>
>>> sf.Frame.from_concat((f1, f2), union=False) #doctest: +NORMALIZE_WHITESPACE
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
>>> sf.Frame.from_structured_array(a, index_column='name') #doctest: +NORMALIZE_WHITESPACE
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
>>> sf.Frame.from_csv(filelike, index_column='name') #doctest: +NORMALIZE_WHITESPACE
<Frame>
<Index> mass      temperature <<U11>
<Index>
Venus   4.87      464
Neptune 102.0     -200
<<U7>   <float64> <int64>

#end_frame_from_csv_a


#start_series_dict_like_a
>>> s = sf.Series((1, 2, 67, 62, 27, 14), index=('Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'))
>>> s #doctest: +NORMALIZE_WHITESPACE
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
>>> f #doctest: +NORMALIZE_WHITESPACE
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
>>> f.get('diameter') #doctest: +NORMALIZE_WHITESPACE
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
>>> s #doctest: +NORMALIZE_WHITESPACE
<Index> <Series>
Venus   108.2
Earth   149.6
Saturn  1433.5
<<U6>   <float64>

>>> abs(s - s['Earth']) #doctest: +NORMALIZE_WHITESPACE
<Index> <Series>
Venus   41.39999999999999
Earth   0.0
Saturn  1283.9
<<U6>   <float64>

>>> s > s['Earth'] #doctest: +NORMALIZE_WHITESPACE
<Index> <Series>
Venus   False
Earth   False
Saturn  True
<<U6>   <bool>

>>> s / s['Earth'] #doctest: +NORMALIZE_WHITESPACE
<Index> <Series>
Venus   0.7232620320855615
Earth   1.0
Saturn  9.582219251336898
<<U6>   <float64>

#end_series_operators_a









'''


if __name__ == "__main__":
    import doctest
    doctest.testmod()
