import string

import frame_fixtures as ff
import numpy as np

import static_frame as sf
from static_frame.core.frame import Frame
from static_frame.core.reduce import ReduceAxis
from static_frame.core.reduce import ReduceDispatchAligned
from static_frame.core.util import IterNodeType


def test_reduce_to_frame_a1():
    f = ff.parse('s(100,5)|v(int64, int64, int64, int64, int64)')
    f = f.assign[0].apply(lambda s: s % 10)
    f_iter = f.iter_group_array_items(0)
    ra = ReduceDispatchAligned(f_iter, f.columns, yield_type=IterNodeType.VALUES).from_label_map(
            {1: np.sum, 2: np.min, 3: np.max, 4: np.sum},
            )
    f2 = ra.to_frame()
    assert (f2.to_pairs() ==
            ((1, ((0, 543298), (1, 292181), (2, 347964), (3, 677008), (4, -644474), (5, 36734), (6, 292135), (7, 45330), (8, 318362), (9, 30307))), (2, ((0, -149082), (1, -56625), (2, 30628), (3, -159324), (4, -171231), (5, -168387), (6, -150573), (7, -170415), (8, -154686), (9, -110091))), (3, ((0, 194249), (1, 126025), (2, 146284), (3, 199490), (4, 191108), (5, 178267), (6, 89423), (7, 187478), (8, 195850), (9, 197228))), (4, ((0, 236989), (1, 777765), (2, 220650), (3, 579134), (4, 349298), (5, -170941), (6, 644531), (7, 318265), (8, -238911), (9, 211233))))
            )

def test_reduce_to_frame_a2():
    f = ff.parse('s(100,5)|v(int64, int64, int64, int64, int64)')
    f = f.assign[0].apply(lambda s: s % 10)
    f_iter = f.iter_group_array_items(0)

    def proc(l, a):
        if l % 2 == 0:
            return 0
        return np.sum(a)

    ra = ReduceDispatchAligned(f_iter, f.columns, yield_type=IterNodeType.ITEMS).from_label_map(
            {1: lambda l, a: np.sum(a),
             2: lambda l, a: np.min(a),
             3: lambda l, a: np.max(a),
             4: proc},
            )
    f2 = ra.to_frame()
    assert (f2.to_pairs() ==
            ((1, ((0, 543298), (1, 292181), (2, 347964), (3, 677008), (4, -644474), (5, 36734), (6, 292135), (7, 45330), (8, 318362), (9, 30307))), (2, ((0, -149082), (1, -56625), (2, 30628), (3, -159324), (4, -171231), (5, -168387), (6, -150573), (7, -170415), (8, -154686), (9, -110091))), (3, ((0, 194249), (1, 126025), (2, 146284), (3, 199490), (4, 191108), (5, 178267), (6, 89423), (7, 187478), (8, 195850), (9, 197228))), (4, ((0, 0), (1, 777765), (2, 0), (3, 579134), (4, 0), (5, -170941), (6, 0), (7, 318265), (8, 0), (9, 211233))))
            )


def test_reduce_to_frame_b1():
    f = ff.parse('s(100,5)|v(int64, int64, int64, int64, int64)')
    f = f.assign[0].apply(lambda s: s % 10)
    f_iter = f.iter_group_array_items(0)
    ra = ReduceDispatchAligned(f_iter, f.columns, yield_type=IterNodeType.VALUES).from_label_map(
        {1: np.sum, 2:np.min, 3: np.max, 4: np.sum},
        )
    f2 = ra.to_frame()
    assert (f2.to_pairs() ==
            ((1, ((0, 543298), (1, 292181), (2, 347964), (3, 677008), (4, -644474), (5, 36734), (6, 292135), (7, 45330), (8, 318362), (9, 30307))), (2, ((0, -149082), (1, -56625), (2, 30628), (3, -159324), (4, -171231), (5, -168387), (6, -150573), (7, -170415), (8, -154686), (9, -110091))), (3, ((0, 194249), (1, 126025), (2, 146284), (3, 199490), (4, 191108), (5, 178267), (6, 89423), (7, 187478), (8, 195850), (9, 197228))), (4, ((0, 236989), (1, 777765), (2, 220650), (3, 579134), (4, 349298), (5, -170941), (6, 644531), (7, 318265), (8, -238911), (9, 211233))))
            )


def test_reduce_to_frame_c():
    f = ff.parse('s(40,5)|v(int64, bool, int64, int64, int64)')
    f = f.assign[0].apply(lambda s: s % 4)
    f_iter = f.iter_group_items(0)
    rf = ReduceDispatchAligned(f_iter, f.columns, yield_type=IterNodeType.VALUES).from_label_map(
        {1: np.sum, 2:np.min, 3: np.max, 4: np.sum},
        )
    f2 = rf.to_frame()
    assert (f2.to_pairs() ==
            ((1, ((0, 5), (1, 5), (2, 6), (3, 2))), (2, ((0, -157437), (1, 6056), (2, -154686), (3, -3648))), (3, ((0, 195850), (1, 172142), (2, 170440), (3, 197228))), (4, ((0, 138242), (1, 31783), (2, 532783), (3, 1076588)))))



def test_reduce_to_frame_d1():
    f = ff.parse('s(40,5)|v(int64, bool, int64, int64, int64)')
    f = f.assign[0].apply(lambda s: s % 4)
    f_iter = f.iter_group_items(0)
    rf = ReduceDispatchAligned(f_iter, f.columns, yield_type=IterNodeType.VALUES).from_label_pair_map(
        {(1, 'a'): np.sum,
         (2, 'b'): np.sum,
         (1, 'c'): np.min,
         (3, 'd'): np.max,
         (4, 'e'): np.sum,
         (3, 'f'): np.max,
         },
        )
    f2 = rf.to_frame()
    assert (f2.to_pairs() ==
            (('a', ((0, 5), (1, 5), (2, 6), (3, 2))), ('b', ((0, 403578), (1, 692639), (2, 601237), (3, 1117328))), ('c', ((0, False), (1, False), (2, False), (3, False))), ('d', ((0, 195850), (1, 172142), (2, 170440), (3, 197228))), ('e', ((0, 138242), (1, 31783), (2, 532783), (3, 1076588))), ('f', ((0, 195850), (1, 172142), (2, 170440), (3, 197228)))))


def test_reduce_to_frame_e():

    f1 = Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    f2 = f1.iter_group_array('A').reduce.from_label_map({'B': np.sum, 'C': np.sum}).to_frame(consolidate_blocks=True)

    assert f2.consolidate.status.shape == (1, 8)

def test_reduce_to_frame_f():
    f1 = Frame(columns=('a', 'b'))
    post = f1.iter_group('a').reduce.from_func(np.sum).to_frame()
    assert post.shape == (0, 0)


def test_reduce_to_frame_g():

    f1 = Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    f2 = f1.iter_group_items('A').reduce.from_map_func(
        lambda l, f: np.sum(f),
        ).to_frame()

    assert (f2.to_pairs() == (('A', ((0, 0), (1, 5), (2, 10), (3, 15))), ('B', ((0, 205), (1, 230), (2, 255), (3, 280))), ('C', ((0, 210), (1, 235), (2, 260), (3, 285))), ('D', ((0, 215), (1, 240), (2, 265), (3, 290))), ('E', ((0, 220), (1, 245), (2, 270), (3, 295)))))


def test_reduce_to_frame_h1():
    f1 = sf.Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    f2 = f1.iter_group('A').reduce.from_label_map({'C': np.sum, 'D': np.min}).to_frame()
    assert (f2.to_pairs() ==
        (('C', ((0, 210), (1, 235), (2, 260), (3, 285))), ('D', ((0, 3), (1, 8), (2, 13), (3, 18)))))

def test_reduce_to_frame_h2():
    f1 = sf.Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    f3 = f1.iter_group('A').reduce.from_label_pair_map({('C', '2022-04'): np.sum, ('C', '2023-01'): np.min}).to_frame(columns_constructor=sf.IndexYearMonth)

    assert (f3.to_pairs() ==
        ((np.datetime64('2022-04'), ((0, 210), (1, 235), (2, 260), (3, 285))), (np.datetime64('2023-01'), ((0, 2), (1, 7), (2, 12), (3, 17)))))

def test_reduce_to_frame_h3():
    f1 = sf.Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    f4 = f1.iter_group('A').reduce.from_map_func(lambda s: s[-1]).to_frame()
    assert (f4.to_pairs() == (('A', ((0, 0), (1, 1), (2, 2), (3, 3))), ('B', ((0, 81), (1, 86), (2, 91), (3, 96))), ('C', ((0, 82), (1, 87), (2, 92), (3, 97))), ('D', ((0, 83), (1, 88), (2, 93), (3, 98))), ('E', ((0, 84), (1, 89), (2, 94), (3, 99)))))


def test_reduce_frame_i1():
    f1 = sf.Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    f5 = (sf.Batch(f1.iter_group('A').reduce.from_map_func(lambda s: s.iloc[-1]).items()) * 100).to_frame()
    assert f5.to_pairs() == (('A', ((0, 0), (1, 100), (2, 200), (3, 300))), ('B', ((0, 8100), (1, 8600), (2, 9100), (3, 9600))), ('C', ((0, 8200), (1, 8700), (2, 9200), (3, 9700))), ('D', ((0, 8300), (1, 8800), (2, 9300), (3, 9800))), ('E', ((0, 8400), (1, 8900), (2, 9400), (3, 9900))))

def test_reduce_frame_i2():
    f1 = sf.Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    f6 = f1.iter_window(size=10, step=3).reduce.from_label_map({'B': np.sum, 'C':np.min}).to_frame()
    assert f6.to_pairs() == (('B', (('j', 235), ('m', 385), ('p', 535), ('s', 685))), ('C', (('j', 2), ('m', 17), ('p', 32), ('s', 47))))

def test_reduce_frame_i3():
    f1 = sf.Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    f7 = (sf.Batch(f1.iter_window(size=10, step=3).reduce.from_label_map({'B': np.sum, 'C':np.min}).items()) * 10).to_frame()
    assert f7.to_pairs() == (('B', (('j', 2350), ('m', 3850), ('p', 5350), ('s', 6850))), ('C', (('j', 20), ('m', 170), ('p', 320), ('s', 470))))


#-------------------------------------------------------------------------------

def test_reduce_from_func_2d_a():

    f1 = Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    f2 = f1.iter_group('A').reduce.from_func(lambda f: f.iloc[2:, 2:]).to_frame()
    assert (f2.to_pairs() ==
            (('C', (('i', 42), ('m', 62), ('q', 82), ('j', 47), ('n', 67), ('r', 87), ('k', 52), ('o', 72), ('s', 92), ('l', 57), ('p', 77), ('t', 97))), ('D', (('i', 43), ('m', 63), ('q', 83), ('j', 48), ('n', 68), ('r', 88), ('k', 53), ('o', 73), ('s', 93), ('l', 58), ('p', 78), ('t', 98))), ('E', (('i', 44), ('m', 64), ('q', 84), ('j', 49), ('n', 69), ('r', 89), ('k', 54), ('o', 74), ('s', 94), ('l', 59), ('p', 79), ('t', 99))))
            )

def test_reduce_from_func_2d_b():

    f1 = Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    def proc(l, f):
        if l == 2:
            return f.iloc[1:, 1:]
        else:
            return f.iloc[2:, 2:]

    f2 = f1.iter_group_items('A').reduce.from_func(proc, fill_value=-1).to_frame()
    assert (f2.to_pairs() ==
            (('B', (('i', -1), ('m', -1), ('q', -1), ('j', -1), ('n', -1), ('r', -1), ('g', 31), ('k', 51), ('o', 71), ('s', 91), ('l', -1), ('p', -1), ('t', -1))), ('C', (('i', 42), ('m', 62), ('q', 82), ('j', 47), ('n', 67), ('r', 87), ('g', 32), ('k', 52), ('o', 72), ('s', 92), ('l', 57), ('p', 77), ('t', 97))), ('D', (('i', 43), ('m', 63), ('q', 83), ('j', 48), ('n', 68), ('r', 88), ('g', 33), ('k', 53), ('o', 73), ('s', 93), ('l', 58), ('p', 78), ('t', 98))), ('E', (('i', 44), ('m', 64), ('q', 84), ('j', 49), ('n', 69), ('r', 89), ('g', 34), ('k', 54), ('o', 74), ('s', 94), ('l', 59), ('p', 79), ('t', 99))))
            )

#-------------------------------------------------------------------------------

def test_reduce_iter_a():

    f1 = Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    it = iter(f1.iter_group('A').reduce.from_func(lambda f: f.iloc[2:, 2:]))
    assert next(it) == 0
    assert next(it) == 1

def test_reduce_iter_b():

    f1 = Frame()

    k, v = next(iter(f1.reduce.from_func(lambda f: f.iloc[2:, 2:]).items()))
    assert k is None
    assert v.shape == (0, 0)


def test_reduce_iter_c():

    f1 = Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    it = iter(f1.iter_group_array_items('A').reduce.from_map_func(lambda l, a: l).values())
    assert next(it).tolist() == [0, 0, 0, 0, 0]
    assert next(it).tolist() == [1, 1, 1, 1, 1]

#-------------------------------------------------------------------------------

def test_reduce_keys_a1():

    f1 = Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    assert list(f1.iter_group('A').reduce.from_map_func(lambda s: s.iloc[-1]).keys()) == [0, 1, 2, 3]

#-------------------------------------------------------------------------------

def test_reduce_items_a():
    f1 = Frame(columns=('a', 'b'))
    post = list(f1.iter_group('a').reduce.from_func(np.sum).items())
    assert not post # empty list

#-------------------------------------------------------------------------------
def test_reduce_values_a():
    f = ff.parse('s(40,5)|v(int64, bool, int64, int64, int64)')
    f = f.assign[0].apply(lambda s: s % 4)
    f_iter = f.iter_group_items(0)
    rf = ReduceDispatchAligned(f_iter, f.columns, yield_type=IterNodeType.VALUES).from_label_pair_map(
        {(1, 'a'): np.sum,
         (2, 'b'): np.sum,
         (1, 'c'): np.min,
         (3, 'd'): np.max,
         },
        )
    post = list(rf.values())
    assert [s.shape for s in post] ==[(4,), (4,), (4,), (4,)]

def test_reduce_values_b():

    f1 = Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    it = iter(f1.iter_group('A').reduce.from_map_func(lambda s: s.iloc[-1]).values())
    s1 = next(it)
    assert s1.to_pairs() == (('A', 0), ('B', 81), ('C', 82), ('D', 83), ('E', 84))
    s2 = next(it)
    assert s2.to_pairs() == (('A', 1), ('B', 86), ('C', 87), ('D', 88), ('E', 89))

def test_reduce_values_c():

    def proc(l, s):
        if l % 2 == 0:
            return 0
        return s.iloc[-1]

    f1 = Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    it = iter(f1.iter_group_items('A').reduce.from_map_func(proc).values())
    s1 = next(it)
    assert s1.to_pairs() == (('A', 0), ('B', 0), ('C', 0), ('D', 0), ('E', 0))
    s2 = next(it)
    assert s2.to_pairs() == (('A', 1), ('B', 86), ('C', 87), ('D', 88), ('E', 89))

def test_reduce_values_d():
    f1 = sf.Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    s1 = next(iter(f1.iter_group('A').reduce.from_map_func(lambda s: s.iloc[-1]).values()))
    assert s1.to_pairs() == (('A', 0), ('B', 81), ('C', 82), ('D', 83), ('E', 84))


def test_reduce_values_e1():

    f1 = Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    it = iter(f1.iter_group_array('A').reduce.from_map_func(lambda a: a[-1]).values())
    a1 = next(it)
    assert a1.tolist() == [0, 81, 82, 83, 84]

def test_reduce_values_e2():

    f1 = Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    it = iter(f1.iter_group_array('A').reduce.from_label_map({'B': np.sum, 'C': np.min}).values())
    a1 = next(it)
    assert a1.tolist() == [205, 2]

def test_reduce_values_e3():

    f1 = Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    def proc(l, a):
        if l % 2 == 0:
            return 0
        return a[-1]

    it = iter(f1.iter_group_array_items('A').reduce.from_label_map(
        {'B': lambda l, a: np.sum(a),
         'C': proc}).values())
    a1 = next(it)
    assert a1.tolist() == [205, 0]
    a2 = next(it)
    assert a2.tolist() == [230, 87]

#-------------------------------------------------------------------------------

def test_derive_row_dtype_array_a():

    assert (ReduceAxis._derive_row_dtype_array(
            np.array([0, 1], dtype=object),
            ((0, np.sum),)
            ) is None)

def test_derive_row_dtype_array_b():

    assert (ReduceAxis._derive_row_dtype_array(
            np.array([0, 1], dtype=np.int64),
            ((0, np.sum), (1, np.all))
            ) == np.dtype(object))


#-------------------------------------------------------------------------------

def test_reduce_iter_group_array_to_frame_a():

    f1 = Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    f2 = f1.iter_group_array('A').reduce.from_func(lambda a: a[4:, 2:]).to_frame()
    assert f2.to_pairs() == ((0, ((0, 82), (1, 87), (2, 92), (3, 97))), (1, ((0, 83), (1, 88), (2, 93), (3, 98))), (2, ((0, 84), (1, 89), (2, 94), (3, 99))))



