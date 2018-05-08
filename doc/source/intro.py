'''
#start_immutability
>>> import static_frame as sf
>>> s = sf.Series((67, 62, 27, 14), index=('Jupiter', 'Saturn', 'Uranus', 'Neptune'))
>>> s #doctest: +NORMALIZE_WHITESPACE
<Index> <Series>
Jupiter 67
Saturn  62
Uranus  27
Neptune 14
<<U7>   <int64>
>>> s['Jupiter'] = 68
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'Series' object does not support item assignment
>>> s.iloc[0] = 68
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'GetItem' object does not support item assignment
>>> s.values[0] = 68
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: assignment destination is read-only

#end_immutability

#start_assign
>>> s.assign['Jupiter'](69) #doctest: +NORMALIZE_WHITESPACE
<Index> <Series>
Jupiter 69
Saturn  62
Uranus  27
Neptune 14
<<U7>   <int64>
>>> s.assign['Uranus':](s['Uranus':] - 2) #doctest: +NORMALIZE_WHITESPACE
<Index> <Series>
Jupiter 67
Saturn  62
Uranus  25
Neptune 12
<<U7>   <int64>
>>> s.assign.iloc[[0, 3]]((68, 11)) #doctest: +NORMALIZE_WHITESPACE
<Index> <Series>
Jupiter 68
Saturn  62
Uranus  27
Neptune 11
<<U7>   <int64>

#end_assign

'''




if __name__ == "__main__":
    import doctest
    doctest.testmod()
