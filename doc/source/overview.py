



'''
#start_from_records_a
>>> import static_frame as sf
>>> index = ('Mercury', 'Venus', 'Earth', 'Mars')
>>> columns = ('Diameter (km)', 'Gravity (m/s2)', 'Temperature (C)')
>>> records = ((4879, 3.7, 167), (12104, 8.9, 464), (12756, 9.8, 15), (6792, 3.7, -65))
>>> sf.Frame.from_records(records, index=index, columns=columns) #doctest: +NORMALIZE_WHITESPACE
<Frame>
<Index> Diameter (km) Gravity (m/s2) Temperature (C) <<U15>
<Index>
Mercury 4879          3.7            167
Venus   12104         8.9            464
Earth   12756         9.8            15
Mars    6792          3.7            -65
<<U7>   <int64>       <float64>      <int64>

#end_from_records_a
'''




if __name__ == "__main__":
    import doctest
    doctest.testmod()
