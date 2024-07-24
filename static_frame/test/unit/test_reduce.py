import frame_fixtures as ff
import numpy as np

from static_frame.core.reduce import Reduce
from static_frame.core.reduce import ReduceDelegate


def test_reduce_to_frame_a():
    f = ff.parse('s(100,5)|v(int64, int64, int64, int64, int64)')
    f = f.assign[0].apply(lambda s: s % 10)
    f_iter = f.iter_group_array_items(0)
    ra = ReduceDelegate(f_iter, f.columns).from_label_map(
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
    ra = ReduceDelegate(f_iter, f.columns).from_label_map(
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
    rf = ReduceDelegate(f_iter, f.columns).from_label_map(
        {1: np.sum, 2:np.min, 3: np.max, 4: np.sum},
        )
    f2 = rf.to_frame()
    assert (f2.to_pairs() ==
            ((1, ((0, 5), (1, 5), (2, 6), (3, 2))), (2, ((0, -157437), (1, 6056), (2, -154686), (3, -3648))), (3, ((0, 195850), (1, 172142), (2, 170440), (3, 197228))), (4, ((0, 138242), (1, 31783), (2, 532783), (3, 1076588)))))



def test_reduce_to_frame_d():
    f = ff.parse('s(40,5)|v(int64, bool, int64, int64, int64)')
    f = f.assign[0].apply(lambda s: s % 4)
    f_iter = f.iter_group_items(0)
    rf = ReduceDelegate(f_iter, f.columns).from_pair_map(
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

