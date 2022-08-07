import doctest
import os
import typing as tp

from static_frame.test.test_case import TestCase
from static_frame.test.test_case import skip_win

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
# restore initial configuration
>>> sf.DisplayActive.set(_display_config_active)

'''


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
