import string

import frame_fixtures as ff
import numpy as np

import static_frame as sf
from static_frame.core.frame import Frame
from static_frame.core.reduce import ReduceDispatchAligned


def test_reduce_to_frame_a():
    f = ff.parse('s(100,5)|v(int64, int64, int64, int64, int64)')
    f = f.assign[0].apply(lambda s: s % 10)
    f_iter = f.iter_group_array_items(0)
    ra = ReduceDispatchAligned(f_iter, f.columns).from_label_map(
            {1: np.sum, 2: np.min, 3: np.max, 4: np.sum},
            )
    f2 = ra.to_frame()
    assert (f2.to_pairs() ==
            ((1, ((0, 543298), (1, 292181), (2, 347964), (3, 677008), (4, -644474), (5, 36734), (6, 292135), (7, 45330), (8, 318362), (9, 30307))), (2, ((0, -149082), (1, -56625), (2, 30628), (3, -159324), (4, -171231), (5, -168387), (6, -150573), (7, -170415), (8, -154686), (9, -110091))), (3, ((0, 194249), (1, 126025), (2, 146284), (3, 199490), (4, 191108), (5, 178267), (6, 89423), (7, 187478), (8, 195850), (9, 197228))), (4, ((0, 236989), (1, 777765), (2, 220650), (3, 579134), (4, 349298), (5, -170941), (6, 644531), (7, 318265), (8, -238911), (9, 211233))))
            )

def test_reduce_to_frame_b():
    f = ff.parse('s(100,5)|v(int64, int64, int64, int64, int64)')
    f = f.assign[0].apply(lambda s: s % 10)
    f_iter = f.iter_group_array_items(0)
    ra = ReduceDispatchAligned(f_iter, f.columns).from_label_map(
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
    rf = ReduceDispatchAligned(f_iter, f.columns).from_label_map(
        {1: np.sum, 2:np.min, 3: np.max, 4: np.sum},
        )
    f2 = rf.to_frame()
    assert (f2.to_pairs() ==
            ((1, ((0, 5), (1, 5), (2, 6), (3, 2))), (2, ((0, -157437), (1, 6056), (2, -154686), (3, -3648))), (3, ((0, 195850), (1, 172142), (2, 170440), (3, 197228))), (4, ((0, 138242), (1, 31783), (2, 532783), (3, 1076588)))))



def test_reduce_to_frame_d():
    f = ff.parse('s(40,5)|v(int64, bool, int64, int64, int64)')
    f = f.assign[0].apply(lambda s: s % 4)
    f_iter = f.iter_group_items(0)
    rf = ReduceDispatchAligned(f_iter, f.columns).from_pair_map(
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


def test_reduce_frame_e1():
    import string

    f1 = Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    it = iter(f1.iter_group('A').reduce.from_func_0d(lambda s: s.iloc[-1]).values())
    s1 = next(it)
    assert s1.to_pairs() == (('A', 0), ('B', 81), ('C', 82), ('D', 83), ('E', 84))
    s2 = next(it)
    assert s2.to_pairs() == (('A', 1), ('B', 86), ('C', 87), ('D', 88), ('E', 89))

def test_reduce_frame_e2():
    import string

    f1 = Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    assert list(f1.iter_group('A').reduce.from_func_0d(lambda s: s.iloc[-1]).keys()) == [0, 1, 2, 3]


def test_reduce_frame_f1():
    f1 = sf.Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    f2 = f1.iter_group('A').reduce.from_label_map({'C': np.sum, 'D': np.min}).to_frame()
    assert (f2.to_pairs() ==
        (('C', ((0, 210), (1, 235), (2, 260), (3, 285))), ('D', ((0, 3), (1, 8), (2, 13), (3, 18)))))

def test_reduce_frame_f2():
    f1 = sf.Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    f3 = f1.iter_group('A').reduce.from_pair_map({('C', '2022-04'): np.sum, ('C', '2023-01'): np.min}).to_frame(columns_constructor=sf.IndexYearMonth)

    assert (f3.to_pairs() ==
        ((np.datetime64('2022-04'), ((0, 210), (1, 235), (2, 260), (3, 285))), (np.datetime64('2023-01'), ((0, 2), (1, 7), (2, 12), (3, 17)))))

def test_reduce_frame_f3():
    f1 = sf.Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    f4 = f1.iter_group('A').reduce.from_func_0d(lambda s: s[-1]).to_frame()
    assert (f4.to_pairs() == (('A', ((0, 0), (1, 1), (2, 2), (3, 3))), ('B', ((0, 81), (1, 86), (2, 91), (3, 96))), ('C', ((0, 82), (1, 87), (2, 92), (3, 97))), ('D', ((0, 83), (1, 88), (2, 93), (3, 98))), ('E', ((0, 84), (1, 89), (2, 94), (3, 99)))))

def test_reduce_frame_f4():
    f1 = sf.Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    s1 = next(iter(f1.iter_group('A').reduce.from_func_0d(lambda s: s.iloc[-1]).values()))
    assert s1.to_pairs() == (('A', 0), ('B', 81), ('C', 82), ('D', 83), ('E', 84))

def test_reduce_frame_f5():
    f1 = sf.Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    f5 = (sf.Batch(f1.iter_group('A').reduce.from_func_0d(lambda s: s.iloc[-1]).items()) * 100).to_frame()
    assert f5.to_pairs() == (('A', ((0, 0), (1, 100), (2, 200), (3, 300))), ('B', ((0, 8100), (1, 8600), (2, 9100), (3, 9600))), ('C', ((0, 8200), (1, 8700), (2, 9200), (3, 9700))), ('D', ((0, 8300), (1, 8800), (2, 9300), (3, 9800))), ('E', ((0, 8400), (1, 8900), (2, 9400), (3, 9900))))

def test_reduce_frame_f6():
    f1 = sf.Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    f6 = f1.iter_window(size=10, step=3).reduce.from_label_map({'B': np.sum, 'C':np.min}).to_frame()
    assert f6.to_pairs() == (('B', (('j', 235), ('m', 385), ('p', 535), ('s', 685))), ('C', (('j', 2), ('m', 17), ('p', 32), ('s', 47))))

def test_reduce_frame_f7():
    f1 = sf.Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    f7 = (sf.Batch(f1.iter_window(size=10, step=3).reduce.from_label_map({'B': np.sum, 'C':np.min}).items()) * 10).to_frame()
    assert f7.to_pairs() == (('B', (('j', 2350), ('m', 3850), ('p', 5350), ('s', 6850))), ('C', (('j', 20), ('m', 170), ('p', 320), ('s', 470))))


def test_reduce_frame_g1():
    import string

    f1 = Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    it = iter(f1.iter_group_array('A').reduce.from_func_0d(lambda a: a[-1]).values())
    a1 = next(it)
    assert a1.tolist() == [0, 81, 82, 83, 84]

def test_reduce_frame_g2():
    import string

    f1 = Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    it = iter(f1.iter_group_array('A').reduce.from_label_map({'B': np.sum, 'C': np.min}).values())
    a1 = next(it)
    assert a1.tolist() == [205, 2]



def test_reduce_from_func_2d_a():
    import string

    f1 = Frame(np.arange(100).reshape(20, 5), index=list(string.ascii_lowercase[:20]), columns=('A', 'B', 'C', 'D', 'E')).assign['A'].apply(lambda s: s % 4)

    f2 = f1.iter_group('A').reduce.from_func_2d(lambda f: f.iloc[2:, 2:]).to_frame()
    assert (f2.to_pairs() ==
            (('C', (('i', 42), ('m', 62), ('q', 82), ('j', 47), ('n', 67), ('r', 87), ('k', 52), ('o', 72), ('s', 92), ('l', 57), ('p', 77), ('t', 97))), ('D', (('i', 43), ('m', 63), ('q', 83), ('j', 48), ('n', 68), ('r', 88), ('k', 53), ('o', 73), ('s', 93), ('l', 58), ('p', 78), ('t', 98))), ('E', (('i', 44), ('m', 64), ('q', 84), ('j', 49), ('n', 69), ('r', 89), ('k', 54), ('o', 74), ('s', 94), ('l', 59), ('p', 79), ('t', 99))))
            )
