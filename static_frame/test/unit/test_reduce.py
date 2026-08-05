import string

import frame_fixtures as ff
import numpy as np

import static_frame as sf
from static_frame.core.frame import Frame
from static_frame.core.reduce import ReduceAxis, ReduceDispatchAligned
from static_frame.core.util import IterNodeType


def test_reduce_to_frame_a1():
    f = ff.parse('s(100,5)|v(int64, int64, int64, int64, int64)')
    f = f.assign[0].apply(lambda s: s % 10)
    f_iter = f.iter_group_array_items(0)
    ra = ReduceDispatchAligned(
        f_iter, f.columns, yield_type=IterNodeType.VALUES
    ).from_label_map(
        {1: np.sum, 2: np.min, 3: np.max, 4: np.sum},
    )
    f2 = ra.to_frame()
    assert f2.to_pairs() == (
        (
            1,
            (
                (0, 543298),
                (1, 292181),
                (2, 347964),
                (3, 677008),
                (4, -644474),
                (5, 36734),
                (6, 292135),
                (7, 45330),
                (8, 318362),
                (9, 30307),
            ),
        ),
        (
            2,
            (
                (0, -149082),
                (1, -56625),
                (2, 30628),
                (3, -159324),
                (4, -171231),
                (5, -168387),
                (6, -150573),
                (7, -170415),
                (8, -154686),
                (9, -110091),
            ),
        ),
        (
            3,
            (
                (0, 194249),
                (1, 126025),
                (2, 146284),
                (3, 199490),
                (4, 191108),
                (5, 178267),
                (6, 89423),
                (7, 187478),
                (8, 195850),
                (9, 197228),
            ),
        ),
        (
            4,
            (
                (0, 236989),
                (1, 777765),
                (2, 220650),
                (3, 579134),
                (4, 349298),
                (5, -170941),
                (6, 644531),
                (7, 318265),
                (8, -238911),
                (9, 211233),
            ),
        ),
    )


def test_reduce_to_frame_a2():
    f = ff.parse('s(100,5)|v(int64, int64, int64, int64, int64)')
    f = f.assign[0].apply(lambda s: s % 10)
    f_iter = f.iter_group_array_items(0)

    def proc(l, a):
        if l % 2 == 0:
            return 0
        return np.sum(a)

    ra = ReduceDispatchAligned(
        f_iter, f.columns, yield_type=IterNodeType.ITEMS
    ).from_label_map(
        {
            1: lambda l, a: np.sum(a),
            2: lambda l, a: np.min(a),
            3: lambda l, a: np.max(a),
            4: proc,
        },
    )
    f2 = ra.to_frame()
    assert f2.to_pairs() == (
        (
            1,
            (
                (0, 543298),
                (1, 292181),
                (2, 347964),
                (3, 677008),
                (4, -644474),
                (5, 36734),
                (6, 292135),
                (7, 45330),
                (8, 318362),
                (9, 30307),
            ),
        ),
        (
            2,
            (
                (0, -149082),
                (1, -56625),
                (2, 30628),
                (3, -159324),
                (4, -171231),
                (5, -168387),
                (6, -150573),
                (7, -170415),
                (8, -154686),
                (9, -110091),
            ),
        ),
        (
            3,
            (
                (0, 194249),
                (1, 126025),
                (2, 146284),
                (3, 199490),
                (4, 191108),
                (5, 178267),
                (6, 89423),
                (7, 187478),
                (8, 195850),
                (9, 197228),
            ),
        ),
        (
            4,
            (
                (0, 0),
                (1, 777765),
                (2, 0),
                (3, 579134),
                (4, 0),
                (5, -170941),
                (6, 0),
                (7, 318265),
                (8, 0),
                (9, 211233),
            ),
        ),
    )


def test_reduce_to_frame_b1():
    f = ff.parse('s(100,5)|v(int64, int64, int64, int64, int64)')
    f = f.assign[0].apply(lambda s: s % 10)
    f_iter = f.iter_group_array_items(0)
    ra = ReduceDispatchAligned(
        f_iter, f.columns, yield_type=IterNodeType.VALUES
    ).from_label_map(
        {1: np.sum, 2: np.min, 3: np.max, 4: np.sum},
    )
    f2 = ra.to_frame()
    assert f2.to_pairs() == (
        (
            1,
            (
                (0, 543298),
                (1, 292181),
                (2, 347964),
                (3, 677008),
                (4, -644474),
                (5, 36734),
                (6, 292135),
                (7, 45330),
                (8, 318362),
                (9, 30307),
            ),
        ),
        (
            2,
            (
                (0, -149082),
                (1, -56625),
                (2, 30628),
                (3, -159324),
                (4, -171231),
                (5, -168387),
                (6, -150573),
                (7, -170415),
                (8, -154686),
                (9, -110091),
            ),
        ),
        (
            3,
            (
                (0, 194249),
                (1, 126025),
                (2, 146284),
                (3, 199490),
                (4, 191108),
                (5, 178267),
                (6, 89423),
                (7, 187478),
                (8, 195850),
                (9, 197228),
            ),
        ),
        (
            4,
            (
                (0, 236989),
                (1, 777765),
                (2, 220650),
                (3, 579134),
                (4, 349298),
                (5, -170941),
                (6, 644531),
                (7, 318265),
                (8, -238911),
                (9, 211233),
            ),
        ),
    )


def test_reduce_to_frame_c():
    f = ff.parse('s(40,5)|v(int64, bool, int64, int64, int64)')
    f = f.assign[0].apply(lambda s: s % 4)
    f_iter = f.iter_group_items(0)
    rf = ReduceDispatchAligned(
        f_iter, f.columns, yield_type=IterNodeType.VALUES
    ).from_label_map(
        {1: np.sum, 2: np.min, 3: np.max, 4: np.sum},
    )
    f2 = rf.to_frame()
    assert f2.to_pairs() == (
        (1, ((0, 5), (1, 5), (2, 6), (3, 2))),
        (2, ((0, -157437), (1, 6056), (2, -154686), (3, -3648))),
        (3, ((0, 195850), (1, 172142), (2, 170440), (3, 197228))),
        (4, ((0, 138242), (1, 31783), (2, 532783), (3, 1076588))),
    )


def test_reduce_to_frame_d1():
    f = ff.parse('s(40,5)|v(int64, bool, int64, int64, int64)')
    f = f.assign[0].apply(lambda s: s % 4)
    f_iter = f.iter_group_items(0)
    rf = ReduceDispatchAligned(
        f_iter, f.columns, yield_type=IterNodeType.VALUES
    ).from_label_pair_map(
        {
            (1, 'a'): np.sum,
            (2, 'b'): np.sum,
            (1, 'c'): np.min,
            (3, 'd'): np.max,
            (4, 'e'): np.sum,
            (3, 'f'): np.max,
        },
    )
    f2 = rf.to_frame()
    assert f2.to_pairs() == (
        ('a', ((0, 5), (1, 5), (2, 6), (3, 2))),
        ('b', ((0, 403578), (1, 692639), (2, 601237), (3, 1117328))),
        ('c', ((0, False), (1, False), (2, False), (3, False))),
        ('d', ((0, 195850), (1, 172142), (2, 170440), (3, 197228))),
        ('e', ((0, 138242), (1, 31783), (2, 532783), (3, 1076588))),
        ('f', ((0, 195850), (1, 172142), (2, 170440), (3, 197228))),
    )


def test_reduce_to_frame_e():
    f1 = (
        Frame(
            np.arange(100).reshape(20, 5),
            index=list(string.ascii_lowercase[:20]),
            columns=('A', 'B', 'C', 'D', 'E'),
        )
        .assign['A']
        .apply(lambda s: s % 4)
    )

    f2 = (
        f1.iter_group_array('A')
        .reduce.from_label_map({'B': np.sum, 'C': np.sum})
        .to_frame(consolidate_blocks=True)
    )

    assert f2.consolidate.status.shape == (1, 8)


def test_reduce_to_frame_f():
    f1 = Frame(columns=('a', 'b'))
    post = f1.iter_group('a').reduce.from_func(np.sum).to_frame()
    assert post.shape == (0, 0)


def test_reduce_to_frame_g():
    f1 = (
        Frame(
            np.arange(100).reshape(20, 5),
            index=list(string.ascii_lowercase[:20]),
            columns=('A', 'B', 'C', 'D', 'E'),
        )
        .assign['A']
        .apply(lambda s: s % 4)
    )

    f2 = (
        f1.iter_group_items('A')
        .reduce.from_map_func(
            lambda l, f: np.sum(f),
        )
        .to_frame()
    )

    assert f2.to_pairs() == (
        ('A', ((0, 0), (1, 5), (2, 10), (3, 15))),
        ('B', ((0, 205), (1, 230), (2, 255), (3, 280))),
        ('C', ((0, 210), (1, 235), (2, 260), (3, 285))),
        ('D', ((0, 215), (1, 240), (2, 265), (3, 290))),
        ('E', ((0, 220), (1, 245), (2, 270), (3, 295))),
    )


def test_reduce_to_frame_h1():
    f1 = (
        sf.Frame(
            np.arange(100).reshape(20, 5),
            index=list(string.ascii_lowercase[:20]),
            columns=('A', 'B', 'C', 'D', 'E'),
        )
        .assign['A']
        .apply(lambda s: s % 4)
    )

    f2 = f1.iter_group('A').reduce.from_label_map({'C': np.sum, 'D': np.min}).to_frame()
    assert f2.to_pairs() == (
        ('C', ((0, 210), (1, 235), (2, 260), (3, 285))),
        ('D', ((0, 3), (1, 8), (2, 13), (3, 18))),
    )


def test_reduce_to_frame_h2():
    f1 = (
        sf.Frame(
            np.arange(100).reshape(20, 5),
            index=list(string.ascii_lowercase[:20]),
            columns=('A', 'B', 'C', 'D', 'E'),
        )
        .assign['A']
        .apply(lambda s: s % 4)
    )

    f3 = (
        f1.iter_group('A')
        .reduce.from_label_pair_map({('C', '2022-04'): np.sum, ('C', '2023-01'): np.min})
        .to_frame(columns_constructor=sf.IndexYearMonth)
    )

    assert f3.to_pairs() == (
        (np.datetime64('2022-04'), ((0, 210), (1, 235), (2, 260), (3, 285))),
        (np.datetime64('2023-01'), ((0, 2), (1, 7), (2, 12), (3, 17))),
    )


def test_reduce_to_frame_h3():
    f1 = (
        sf.Frame(
            np.arange(100).reshape(20, 5),
            index=list(string.ascii_lowercase[:20]),
            columns=('A', 'B', 'C', 'D', 'E'),
        )
        .assign['A']
        .apply(lambda s: s % 4)
    )

    f4 = f1.iter_group('A').reduce.from_map_func(lambda s: s[-1]).to_frame()
    assert f4.to_pairs() == (
        ('A', ((0, 0), (1, 1), (2, 2), (3, 3))),
        ('B', ((0, 81), (1, 86), (2, 91), (3, 96))),
        ('C', ((0, 82), (1, 87), (2, 92), (3, 97))),
        ('D', ((0, 83), (1, 88), (2, 93), (3, 98))),
        ('E', ((0, 84), (1, 89), (2, 94), (3, 99))),
    )


def test_reduce_frame_i1():
    f1 = (
        sf.Frame(
            np.arange(100).reshape(20, 5),
            index=list(string.ascii_lowercase[:20]),
            columns=('A', 'B', 'C', 'D', 'E'),
        )
        .assign['A']
        .apply(lambda s: s % 4)
    )

    f5 = (
        sf.Batch(f1.iter_group('A').reduce.from_map_func(lambda s: s.iloc[-1]).items())
        * 100
    ).to_frame()
    assert f5.to_pairs() == (
        ('A', ((0, 0), (1, 100), (2, 200), (3, 300))),
        ('B', ((0, 8100), (1, 8600), (2, 9100), (3, 9600))),
        ('C', ((0, 8200), (1, 8700), (2, 9200), (3, 9700))),
        ('D', ((0, 8300), (1, 8800), (2, 9300), (3, 9800))),
        ('E', ((0, 8400), (1, 8900), (2, 9400), (3, 9900))),
    )


def test_reduce_frame_i2():
    f1 = (
        sf.Frame(
            np.arange(100).reshape(20, 5),
            index=list(string.ascii_lowercase[:20]),
            columns=('A', 'B', 'C', 'D', 'E'),
        )
        .assign['A']
        .apply(lambda s: s % 4)
    )

    f6 = (
        f1.iter_window(size=10, step=3)
        .reduce.from_label_map({'B': np.sum, 'C': np.min})
        .to_frame()
    )
    assert f6.to_pairs() == (
        ('B', (('j', 235), ('m', 385), ('p', 535), ('s', 685))),
        ('C', (('j', 2), ('m', 17), ('p', 32), ('s', 47))),
    )


def test_reduce_frame_i3():
    f1 = (
        sf.Frame(
            np.arange(100).reshape(20, 5),
            index=list(string.ascii_lowercase[:20]),
            columns=('A', 'B', 'C', 'D', 'E'),
        )
        .assign['A']
        .apply(lambda s: s % 4)
    )

    f7 = (
        sf.Batch(
            f1.iter_window(size=10, step=3)
            .reduce.from_label_map({'B': np.sum, 'C': np.min})
            .items()
        )
        * 10
    ).to_frame()
    assert f7.to_pairs() == (
        ('B', (('j', 2350), ('m', 3850), ('p', 5350), ('s', 6850))),
        ('C', (('j', 20), ('m', 170), ('p', 320), ('s', 470))),
    )


# -------------------------------------------------------------------------------


def test_reduce_from_func_2d_a():
    f1 = (
        Frame(
            np.arange(100).reshape(20, 5),
            index=list(string.ascii_lowercase[:20]),
            columns=('A', 'B', 'C', 'D', 'E'),
        )
        .assign['A']
        .apply(lambda s: s % 4)
    )

    f2 = f1.iter_group('A').reduce.from_func(lambda f: f.iloc[2:, 2:]).to_frame()
    assert f2.to_pairs() == (
        (
            'C',
            (
                ('i', 42),
                ('m', 62),
                ('q', 82),
                ('j', 47),
                ('n', 67),
                ('r', 87),
                ('k', 52),
                ('o', 72),
                ('s', 92),
                ('l', 57),
                ('p', 77),
                ('t', 97),
            ),
        ),
        (
            'D',
            (
                ('i', 43),
                ('m', 63),
                ('q', 83),
                ('j', 48),
                ('n', 68),
                ('r', 88),
                ('k', 53),
                ('o', 73),
                ('s', 93),
                ('l', 58),
                ('p', 78),
                ('t', 98),
            ),
        ),
        (
            'E',
            (
                ('i', 44),
                ('m', 64),
                ('q', 84),
                ('j', 49),
                ('n', 69),
                ('r', 89),
                ('k', 54),
                ('o', 74),
                ('s', 94),
                ('l', 59),
                ('p', 79),
                ('t', 99),
            ),
        ),
    )


def test_reduce_from_func_2d_b():
    f1 = (
        Frame(
            np.arange(100).reshape(20, 5),
            index=list(string.ascii_lowercase[:20]),
            columns=('A', 'B', 'C', 'D', 'E'),
        )
        .assign['A']
        .apply(lambda s: s % 4)
    )

    def proc(l, f):
        if l == 2:
            return f.iloc[1:, 1:]
        else:
            return f.iloc[2:, 2:]

    f2 = f1.iter_group_items('A').reduce.from_func(proc, fill_value=-1).to_frame()
    assert f2.to_pairs() == (
        (
            'B',
            (
                ('i', -1),
                ('m', -1),
                ('q', -1),
                ('j', -1),
                ('n', -1),
                ('r', -1),
                ('g', 31),
                ('k', 51),
                ('o', 71),
                ('s', 91),
                ('l', -1),
                ('p', -1),
                ('t', -1),
            ),
        ),
        (
            'C',
            (
                ('i', 42),
                ('m', 62),
                ('q', 82),
                ('j', 47),
                ('n', 67),
                ('r', 87),
                ('g', 32),
                ('k', 52),
                ('o', 72),
                ('s', 92),
                ('l', 57),
                ('p', 77),
                ('t', 97),
            ),
        ),
        (
            'D',
            (
                ('i', 43),
                ('m', 63),
                ('q', 83),
                ('j', 48),
                ('n', 68),
                ('r', 88),
                ('g', 33),
                ('k', 53),
                ('o', 73),
                ('s', 93),
                ('l', 58),
                ('p', 78),
                ('t', 98),
            ),
        ),
        (
            'E',
            (
                ('i', 44),
                ('m', 64),
                ('q', 84),
                ('j', 49),
                ('n', 69),
                ('r', 89),
                ('g', 34),
                ('k', 54),
                ('o', 74),
                ('s', 94),
                ('l', 59),
                ('p', 79),
                ('t', 99),
            ),
        ),
    )


# -------------------------------------------------------------------------------


def test_reduce_iter_a():
    f1 = (
        Frame(
            np.arange(100).reshape(20, 5),
            index=list(string.ascii_lowercase[:20]),
            columns=('A', 'B', 'C', 'D', 'E'),
        )
        .assign['A']
        .apply(lambda s: s % 4)
    )

    it = iter(f1.iter_group('A').reduce.from_func(lambda f: f.iloc[2:, 2:]))
    assert next(it) == 0
    assert next(it) == 1


def test_reduce_iter_b():
    f1 = Frame()

    k, v = next(iter(f1.reduce.from_func(lambda f: f.iloc[2:, 2:]).items()))
    assert k is None
    assert v.shape == (0, 0)


def test_reduce_iter_c():
    f1 = (
        Frame(
            np.arange(100).reshape(20, 5),
            index=list(string.ascii_lowercase[:20]),
            columns=('A', 'B', 'C', 'D', 'E'),
        )
        .assign['A']
        .apply(lambda s: s % 4)
    )

    it = iter(
        f1.iter_group_array_items('A').reduce.from_map_func(lambda l, a: l).values()
    )
    assert next(it).tolist() == [0, 0, 0, 0, 0]
    assert next(it).tolist() == [1, 1, 1, 1, 1]


# -------------------------------------------------------------------------------


def test_reduce_keys_a1():
    f1 = (
        Frame(
            np.arange(100).reshape(20, 5),
            index=list(string.ascii_lowercase[:20]),
            columns=('A', 'B', 'C', 'D', 'E'),
        )
        .assign['A']
        .apply(lambda s: s % 4)
    )

    assert list(f1.iter_group('A').reduce.from_map_func(lambda s: s.iloc[-1]).keys()) == [
        0,
        1,
        2,
        3,
    ]


# -------------------------------------------------------------------------------


def test_reduce_items_a():
    f1 = Frame(columns=('a', 'b'))
    post = list(f1.iter_group('a').reduce.from_func(np.sum).items())
    assert not post  # empty list


# -------------------------------------------------------------------------------
def test_reduce_values_a():
    f = ff.parse('s(40,5)|v(int64, bool, int64, int64, int64)')
    f = f.assign[0].apply(lambda s: s % 4)
    f_iter = f.iter_group_items(0)
    rf = ReduceDispatchAligned(
        f_iter, f.columns, yield_type=IterNodeType.VALUES
    ).from_label_pair_map(
        {
            (1, 'a'): np.sum,
            (2, 'b'): np.sum,
            (1, 'c'): np.min,
            (3, 'd'): np.max,
        },
    )
    post = list(rf.values())
    assert [s.shape for s in post] == [(4,), (4,), (4,), (4,)]


def test_reduce_values_b():
    f1 = (
        Frame(
            np.arange(100).reshape(20, 5),
            index=list(string.ascii_lowercase[:20]),
            columns=('A', 'B', 'C', 'D', 'E'),
        )
        .assign['A']
        .apply(lambda s: s % 4)
    )

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

    f1 = (
        Frame(
            np.arange(100).reshape(20, 5),
            index=list(string.ascii_lowercase[:20]),
            columns=('A', 'B', 'C', 'D', 'E'),
        )
        .assign['A']
        .apply(lambda s: s % 4)
    )

    it = iter(f1.iter_group_items('A').reduce.from_map_func(proc).values())
    s1 = next(it)
    assert s1.to_pairs() == (('A', 0), ('B', 0), ('C', 0), ('D', 0), ('E', 0))
    s2 = next(it)
    assert s2.to_pairs() == (('A', 1), ('B', 86), ('C', 87), ('D', 88), ('E', 89))


def test_reduce_values_d():
    f1 = (
        sf.Frame(
            np.arange(100).reshape(20, 5),
            index=list(string.ascii_lowercase[:20]),
            columns=('A', 'B', 'C', 'D', 'E'),
        )
        .assign['A']
        .apply(lambda s: s % 4)
    )

    s1 = next(
        iter(f1.iter_group('A').reduce.from_map_func(lambda s: s.iloc[-1]).values())
    )
    assert s1.to_pairs() == (('A', 0), ('B', 81), ('C', 82), ('D', 83), ('E', 84))


def test_reduce_values_e1():
    f1 = (
        Frame(
            np.arange(100).reshape(20, 5),
            index=list(string.ascii_lowercase[:20]),
            columns=('A', 'B', 'C', 'D', 'E'),
        )
        .assign['A']
        .apply(lambda s: s % 4)
    )

    it = iter(f1.iter_group_array('A').reduce.from_map_func(lambda a: a[-1]).values())
    a1 = next(it)
    assert a1.tolist() == [0, 81, 82, 83, 84]


def test_reduce_values_e2():
    f1 = (
        Frame(
            np.arange(100).reshape(20, 5),
            index=list(string.ascii_lowercase[:20]),
            columns=('A', 'B', 'C', 'D', 'E'),
        )
        .assign['A']
        .apply(lambda s: s % 4)
    )

    it = iter(
        f1.iter_group_array('A')
        .reduce.from_label_map({'B': np.sum, 'C': np.min})
        .values()
    )
    a1 = next(it)
    assert a1.tolist() == [205, 2]


def test_reduce_values_e3():
    f1 = (
        Frame(
            np.arange(100).reshape(20, 5),
            index=list(string.ascii_lowercase[:20]),
            columns=('A', 'B', 'C', 'D', 'E'),
        )
        .assign['A']
        .apply(lambda s: s % 4)
    )

    def proc(l, a):
        if l % 2 == 0:
            return 0
        return a[-1]

    it = iter(
        f1.iter_group_array_items('A')
        .reduce.from_label_map({'B': lambda l, a: np.sum(a), 'C': proc})
        .values()
    )
    a1 = next(it)
    assert a1.tolist() == [205, 0]
    a2 = next(it)
    assert a2.tolist() == [230, 87]


# -------------------------------------------------------------------------------


def test_derive_row_dtype_array_a():
    assert (
        ReduceAxis._derive_row_dtype_array(np.array([0, 1], dtype=object), ((0, np.sum),))
        is None
    )


def test_derive_row_dtype_array_b():
    assert ReduceAxis._derive_row_dtype_array(
        np.array([0, 1], dtype=np.int64), ((0, np.sum), (1, np.all))
    ) == np.dtype(object)


# -------------------------------------------------------------------------------


def test_reduce_iter_group_array_to_frame_a():
    f1 = (
        Frame(
            np.arange(100).reshape(20, 5),
            index=list(string.ascii_lowercase[:20]),
            columns=('A', 'B', 'C', 'D', 'E'),
        )
        .assign['A']
        .apply(lambda s: s % 4)
    )

    f2 = f1.iter_group_array('A').reduce.from_func(lambda a: a[4:, 2:]).to_frame()
    assert f2.to_pairs() == (
        (0, ((0, 82), (1, 87), (2, 92), (3, 97))),
        (1, ((0, 83), (1, 88), (2, 93), (3, 98))),
        (2, ((0, 84), (1, 89), (2, 94), (3, 99))),
    )


# -------------------------------------------------------------------------------
# vectorized group-reduce fast path (factorize + arraykit.group_reduce)


def _reduce_fast_kwargs(columns):
    return dict(
        index=None,
        columns=columns,
        index_constructor=None,
        columns_constructor=None,
        name=None,
        consolidate_blocks=False,
    )


def test_reduce_group_fast_path_matches_loop():
    import warnings

    rng = np.random.RandomState(0)
    val_dtypes = [
        'int8',
        'int16',
        'int32',
        'int64',
        'uint8',
        'uint32',
        'uint64',
        'float16',
        'float32',
        'float64',
    ]
    ops = [np.sum, np.max, np.min, np.prod, len]

    def make_key(kind, n):
        if kind == 'int':
            return rng.randint(0, 6, n)
        if kind == 'str':
            return np.array([f'g{v}' for v in rng.randint(0, 6, n)])
        a = rng.randint(0, 6, n).astype(float)
        a[rng.rand(n) < 0.15] = np.nan  # NaN key groups together, sorted last
        return a

    def make_val(dt, n):
        if dt.startswith('float'):
            return (rng.rand(n) * 100).astype(dt)
        if dt.startswith('uint'):
            return rng.randint(0, 50, n).astype(dt)
        return rng.randint(-50, 50, n).astype(dt)

    fast_used = 0
    for _ in range(200):
        n = int(rng.randint(1, 40))
        kkind = ['int', 'str', 'floatnan'][rng.randint(0, 3)]
        d = {'k': make_key(kkind, n)}
        lm = {}
        for i in range(rng.randint(1, 4)):
            vd = val_dtypes[rng.randint(0, len(val_dtypes))]
            cn = f'v{i}'
            d[cn] = make_val(vd, n)
            lm[cn] = ops[rng.randint(0, len(ops))]
        f = Frame.from_dict(d)
        cols = list(lm.keys())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # ignore overflow-in-reduce
            r_fast = f.iter_group('k').reduce.from_label_map(lm)
            fast = r_fast.to_frame(columns=cols)
            r_loop = f.iter_group('k').reduce.from_label_map(lm)
            r_loop._group_source = None
            loop = r_loop.to_frame(columns=cols)
        if r_fast._to_frame_fast(**_reduce_fast_kwargs(cols)) is not None:
            fast_used += 1
        assert fast.index.equals(loop.index)
        assert fast.columns.equals(loop.columns)
        for c in cols:
            fv = fast[c].values
            lv = loop[c].values
            assert fv.dtype == lv.dtype, (c, lm[c].__name__)
            if fv.dtype.kind == 'f' and lm[c] in (np.sum, np.prod):
                # float sum/prod: ~1 ULP order difference is accepted
                assert np.allclose(fv, lv, equal_nan=True), (c, lm[c].__name__)
            elif fv.dtype.kind == 'f':  # float min/max: exact
                assert np.array_equal(fv, lv, equal_nan=True), (c, lm[c].__name__)
            else:  # all integer results are exact
                assert np.array_equal(fv, lv), (c, lm[c].__name__)
    assert fast_used > 50  # the fast path is actually exercised


def test_reduce_group_fast_path_nan_key():
    # a NaN key groups all NaNs together and sorts them last, like iter_group
    f = Frame.from_dict(
        dict(k=np.array([2.0, np.nan, 1.0, np.nan, 2.0]), v=(10, 20, 30, 40, 50))
    )
    r_fast = f.iter_group('k').reduce.from_label_map({'v': np.sum})
    assert r_fast._group_source is not None
    fast = r_fast.to_frame(columns=['v'])
    r_loop = f.iter_group('k').reduce.from_label_map({'v': np.sum})
    r_loop._group_source = None
    loop = r_loop.to_frame(columns=['v'])
    assert fast.equals(loop, compare_dtype=True)
    # k=1 -> 30; k=2 -> 10+50; nan -> 20+40
    assert fast['v'].values.tolist() == [30, 60, 60]


def test_reduce_group_fast_path_datetime_key():
    # the benchmark shape: a datetime key with mixed int/float value columns
    key = np.array(
        ['2020-01-01', '2020-01-02', '2020-01-01', '2020-01-03', '2020-01-02'],
        dtype='datetime64[D]',
    )
    f = Frame.from_dict(
        dict(k=key, c=np.array([1, 2, 3, 4, 5]), v=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    )
    lm = {'c': np.sum, 'v': np.max}
    r_fast = f.iter_group('k').reduce.from_label_map(lm)
    assert (
        r_fast._to_frame_fast(
            index=None,
            columns=['c', 'v'],
            index_constructor=sf.IndexDate,
            columns_constructor=None,
            name=None,
            consolidate_blocks=False,
        )
        is not None
    )
    fast = r_fast.to_frame(columns=['c', 'v'], index_constructor=sf.IndexDate)
    r_loop = f.iter_group('k').reduce.from_label_map(lm)
    r_loop._group_source = None
    loop = r_loop.to_frame(columns=['c', 'v'], index_constructor=sf.IndexDate)
    assert fast.equals(loop, compare_dtype=True)
    assert isinstance(fast.index, sf.IndexDate)
    assert fast['c'].dtype == np.dtype(np.int64)  # int sum stays int


def test_reduce_group_fast_path_fallbacks():
    f = Frame.from_dict(dict(k=(1, 1, 2, 2), a=(10, 20, 30, 40), b=(1.0, 2.0, 3.0, 4.0)))

    # a multi-column key is not a single-column iloc -> no fast path
    r_multi = f.iter_group(['k', 'a']).reduce.from_label_map({'b': np.sum})
    assert r_multi._group_source is None

    # an unrecognized reducer (np.mean) -> fast path declines, loop still correct
    r_mean = f.iter_group('k').reduce.from_label_map({'b': np.mean})
    assert r_mean._group_source is not None
    assert r_mean._to_frame_fast(**_reduce_fast_kwargs(['b'])) is None
    assert r_mean.to_frame(columns=['b'])['b'].values.tolist() == [1.5, 3.5]

    # float32 sum is unsupported by group_reduce (exactness) -> fast path declines
    f32 = Frame.from_dict(dict(k=(1, 1, 2), x=np.array([1, 2, 3], dtype=np.float32)))
    r32 = f32.iter_group('k').reduce.from_label_map({'x': np.sum})
    assert r32._to_frame_fast(**_reduce_fast_kwargs(['x'])) is None
    assert r32.to_frame(columns=['x'])['x'].dtype == np.dtype(np.float32)


def test_reduce_group_fast_path_iter_group_array_object():
    # iter_group_array on a mixed frame: the per-group 2D component is object, yet the
    # fast path reduces each numeric column at int64/float64 to match the loop's
    # value-inferred output (this is the benchmark shape: datetime key + int + float)
    key = np.array(
        ['2020-01-01', '2020-01-02', '2020-01-01', '2020-01-03', '2020-01-02'],
        dtype='datetime64[s]',
    )
    f = Frame.from_dict(
        dict(k=key, c=np.array([1, 2, 3, 4, 5]), v=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    )
    lm = {'c': np.sum, 'v': np.max}
    r_fast = f.iter_group_array('k').reduce.from_label_map(lm)
    assert r_fast._group_source is not None and r_fast._group_source[2] is True
    kwargs = dict(
        index=None,
        columns=['c', 'v'],
        index_constructor=sf.IndexSecond,
        columns_constructor=None,
        name=None,
        consolidate_blocks=False,
    )
    assert r_fast._to_frame_fast(**kwargs) is not None
    fast = r_fast.to_frame(columns=['c', 'v'], index_constructor=sf.IndexSecond)
    r_loop = f.iter_group_array('k').reduce.from_label_map(lm)
    r_loop._group_source = None
    loop = r_loop.to_frame(columns=['c', 'v'], index_constructor=sf.IndexSecond)
    assert fast.equals(loop, compare_dtype=True)
    # object component -> the loop infers the platform default int (int32 on Windows)
    assert fast['c'].dtype == np.dtype(np.int_)
    assert fast['v'].dtype == np.dtype(np.float64)


def test_reduce_group_fast_path_iter_group_array_numeric():
    # an all-numeric frame: the 2D component is a single numeric dtype (float64 here),
    # so each column is reduced cast to it -- matching the loop exactly
    f = Frame.from_dict(
        dict(
            k=(1, 1, 2, 2),
            a=np.array([1, 2, 3, 4], dtype=np.int8),
            b=np.array([1.5, 2.5, 3.5, 4.5]),
        )
    )
    lm = {'a': np.sum, 'b': np.min}
    r_fast = f.iter_group_array('k').reduce.from_label_map(lm)
    fast = r_fast.to_frame(columns=['a', 'b'])
    r_loop = f.iter_group_array('k').reduce.from_label_map(lm)
    r_loop._group_source = None
    loop = r_loop.to_frame(columns=['a', 'b'])
    assert fast.equals(loop, compare_dtype=True)
    # the int8 column unifies to float64 in the 2D array, so its sum is float64
    assert fast['a'].dtype == np.dtype(np.float64)


# -------------------------------------------------------------------------------
# reduce_pool concurrency


def _pool_heavy_col(col):  # module-level so a process pool can pickle it
    return float(np.sum(col)) * 2.0


def test_reduce_pool_threads_matches_sequential():
    # threads need no pickling: exercise every path with lambdas vs the sequential
    rng = np.random.RandomState(0)
    f = Frame.from_dict(
        dict(
            k=rng.randint(0, 8, 300),
            a=rng.rand(300),
            b=rng.randint(0, 50, 300),
            c=rng.rand(300),
        )
    )
    lm = {'a': lambda s: float(np.mean(s)), 'b': np.sum, 'c': lambda s: s.max() - s.min()}

    # from_label_map (ReduceAligned pooled block assembly)
    seq = f.iter_group('k').reduce.from_label_map(lm).to_frame(columns=['a', 'b', 'c'])
    pooled = (
        f.iter_group('k')
        .reduce_pool(use_threads=True, max_workers=4)
        .from_label_map(lm)
        .to_frame(columns=['a', 'b', 'c'])
    )
    assert seq.equals(pooled, compare_dtype=True)

    # iter_group_array pooled
    lm2 = {'a': np.sum, 'c': np.max}
    seq_a = (
        f.iter_group_array('k').reduce.from_label_map(lm2).to_frame(columns=['a', 'c'])
    )
    pool_a = (
        f.iter_group_array('k')
        .reduce_pool(use_threads=True)
        .from_label_map(lm2)
        .to_frame(columns=['a', 'c'])
    )
    assert seq_a.equals(pool_a, compare_dtype=True)

    # from_map_func pooled
    seq_m = f.iter_group('k').reduce.from_map_func(np.sum).to_frame()
    pool_m = (
        f.iter_group('k').reduce_pool(use_threads=True).from_map_func(np.sum).to_frame()
    )
    assert seq_m.equals(pool_m, compare_dtype=True)


def test_reduce_pool_from_func_threads():
    rng = np.random.RandomState(1)
    f = Frame.from_dict(dict(k=rng.randint(0, 6, 120), a=rng.rand(120), b=rng.rand(120)))
    # from_func returns a whole reduced Frame per group (ReduceComponent pooled)
    seq = f.iter_group('k').reduce.from_func(lambda fr: fr.iloc[:1]).to_frame()
    pooled = (
        f.iter_group('k')
        .reduce_pool(use_threads=True, max_workers=4)
        .from_func(lambda fr: fr.iloc[:1])
        .to_frame()
    )
    assert seq.equals(pooled, compare_dtype=True)

    # items() pooled matches sequential (ITEMS yield type)
    seq_items = dict(
        f.iter_group_items('k').reduce.from_func(lambda l, fr: fr.iloc[0]).items()
    )
    pool_items = dict(
        f.iter_group_items('k')
        .reduce_pool(use_threads=True)
        .from_func(lambda l, fr: fr.iloc[0])
        .items()
    )
    assert set(seq_items) == set(pool_items)
    assert all(seq_items[k].equals(pool_items[k]) for k in seq_items)


def test_reduce_pool_processes():
    # a process pool requires picklable funcs (module-level), unlike threads
    rng = np.random.RandomState(2)
    f = Frame.from_dict(dict(k=rng.randint(0, 4, 60), a=rng.rand(60)))
    seq = (
        f.iter_group('k')
        .reduce.from_label_map({'a': _pool_heavy_col})
        .to_frame(columns=['a'])
    )
    pooled = (
        f.iter_group('k')
        .reduce_pool(use_threads=False, max_workers=2)
        .from_label_map({'a': _pool_heavy_col})
        .to_frame(columns=['a'])
    )
    assert seq.equals(pooled, compare_dtype=True)


# -------------------------------------------------------------------------------
# fast-path / pooled-worker coverage edge cases


def test_reduce_column_plan_fallbacks():
    from static_frame.core.reduce import _reduce_column_plan
    from static_frame.core.util import DTYPE_OBJECT

    # native object column with a summing func: no derivable dtype -> fall back
    obj = np.array([1, 2, 3], dtype=object)
    assert _reduce_column_plan(np.sum, 'sum', obj, None) is None
    # object-unified component with a non-numeric (string) column -> fall back
    s = np.array(['a', 'b'], dtype='U1')
    assert _reduce_column_plan(np.max, 'max', s, DTYPE_OBJECT) is None
    # numeric-unified (datetime) where the func has no derivable dtype -> fall back
    dt = np.array(['2020-01-01'], dtype='datetime64[D]')
    assert _reduce_column_plan(np.sum, 'sum', dt, dt.dtype) is None


def test_reduce_pool_worker_paths():
    # reduce_pool with lambdas (fast path declines) exercises every _reduce_row_worker
    # branch: array/Frame components x VALUES/ITEMS yield types
    rng = np.random.RandomState(0)
    f = Frame.from_dict(dict(k=rng.randint(0, 5, 60), a=rng.rand(60), b=rng.rand(60)))

    # array component, VALUES
    lm_v = {'a': lambda a: float(a.sum())}
    seq = f.iter_group_array('k').reduce.from_label_map(lm_v).to_frame(columns=['a'])
    pool = (
        f.iter_group_array('k')
        .reduce_pool(use_threads=True)
        .from_label_map(lm_v)
        .to_frame(columns=['a'])
    )
    assert seq.equals(pool, compare_dtype=True)

    # array component, ITEMS (func takes label, values)
    lm_i = {'a': lambda l, a: float(a.sum())}
    seq = (
        f.iter_group_array_items('k').reduce.from_label_map(lm_i).to_frame(columns=['a'])
    )
    pool = (
        f.iter_group_array_items('k')
        .reduce_pool(use_threads=True)
        .from_label_map(lm_i)
        .to_frame(columns=['a'])
    )
    assert seq.equals(pool, compare_dtype=True)

    # Frame component, ITEMS
    lm_fi = {'a': lambda l, s: float(s.sum())}
    seq = f.iter_group_items('k').reduce.from_label_map(lm_fi).to_frame(columns=['a'])
    pool = (
        f.iter_group_items('k')
        .reduce_pool(use_threads=True)
        .from_label_map(lm_fi)
        .to_frame(columns=['a'])
    )
    assert seq.equals(pool, compare_dtype=True)


def _reduce_fast_kwargs2(columns):
    return dict(
        index=None,
        columns=columns,
        index_constructor=None,
        columns_constructor=None,
        name=None,
        consolidate_blocks=False,
    )


def test_reduce_group_fast_path_empty():
    # a zero-row frame -> no groups -> fast path declines (loop then raises)
    f = Frame.from_dict(
        dict(k=np.array([], dtype=np.int64), a=np.array([], dtype=np.float64))
    )
    r = f.iter_group('k').reduce.from_label_map({'a': np.sum})
    assert r._to_frame_fast(**_reduce_fast_kwargs2(['a'])) is None


def test_reduce_group_fast_path_unorderable_key():
    # a mixed-type object key cannot be sorted -> factorize(sort=True) raises -> fall back
    f = Frame.from_dict(dict(k=np.array([1, 'a', 2], dtype=object), a=(1.0, 2.0, 3.0)))
    r = f.iter_group('k').reduce.from_label_map({'a': np.sum})
    assert r._to_frame_fast(**_reduce_fast_kwargs2(['a'])) is None


def test_reduce_group_fast_path_object_value_column():
    # an object value column: the plan cannot reproduce it -> fast path declines,
    # the per-group loop still produces the correct result
    f = Frame.from_dict(dict(k=(1, 1, 2), a=np.array([10, 20, 30], dtype=object)))
    r = f.iter_group('k').reduce.from_label_map({'a': np.sum})
    assert r._to_frame_fast(**_reduce_fast_kwargs2(['a'])) is None
    post = r.to_frame(columns=['a'])  # loop path
    assert post['a'].values.tolist() == [30, 30]


def test_reduce_group_fast_path_sequence_axis_labels():
    # axis_labels as a plain sequence (not an Index) exercises the non-Index branch
    from static_frame.core.reduce import ReduceAligned
    from static_frame.core.util import IterNodeType

    f = Frame.from_dict(dict(k=(1, 1, 2), a=(10, 20, 30)))
    r = ReduceAligned(
        (),  # items unused by the fast path
        [(1, np.sum)],  # reduce column iloc 1 ('a')
        ['a'],  # a Sequence, not an IndexBase
        IterNodeType.VALUES,
        1,
        group_source=(f, 0, False),  # group by column 0 ('k')
    )
    post = r._to_frame_fast(**_reduce_fast_kwargs2(None))
    assert post is not None
    assert post.columns.values.tolist() == ['a']
    assert post['a'].values.tolist() == [30, 30]  # k=1: 10+20; k=2: 30


def test_reduce_group_consolidate_blocks():
    # the loop path (a lambda declines the fast path) with consolidate_blocks=True
    f = Frame.from_dict(dict(k=(1, 1, 2, 2), a=(1, 2, 3, 4), b=(5, 6, 7, 8)))
    post = (
        f.iter_group('k')
        .reduce.from_label_map({'a': lambda s: int(s.sum()), 'b': lambda s: int(s.sum())})
        .to_frame(columns=['a', 'b'], consolidate_blocks=True)
    )
    assert post.to_pairs() == (
        ('a', ((1, 3), (2, 7))),
        ('b', ((1, 11), (2, 15))),
    )
