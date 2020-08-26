



import numpy as np

from static_frame.core.frame import Frame
from static_frame.core.batch import Batch








import time

def timer(f): # type: ignore
    def wraped(*args, **kwargs): # type: ignore
        t = time.time()
        post = f()
        print(time.time() - t, f)
        return post
    return wraped



def func_b(frame: Frame) -> Frame:
    for row in frame.iter_series():
        pass
        # if row[10] > 1000: # type: ignore
    return frame


def main() -> None:

    f1 = Frame(np.arange(100000000).reshape(1000000, 100), name='a')
    f2 = Frame(np.arange(100000000).reshape(1000000, 100), name='b')
    f3 = Frame(np.arange(100000000).reshape(1000000, 100), name='c')
    f4 = Frame(np.arange(100000000).reshape(1000000, 100), name='d')
    f5 = Frame(np.arange(100000000).reshape(1000000, 100), name='e')
    f6 = Frame(np.arange(100000000).reshape(1000000, 100), name='f')
    f7 = Frame(np.arange(100000000).reshape(1000000, 100), name='g')
    f8 = Frame(np.arange(100000000).reshape(1000000, 100), name='h')



    @timer #type: ignore
    def a1() -> None:
        batch1 = Batch.from_frames((f1, f2, f3, f4, f5, f6, f7, f8))
        batch2 = (batch1 * 100).sum()
        _ = tuple(batch2.items())

    @timer #type: ignore
    def a2() -> None:
        batch1 = Batch.from_frames((f1, f2, f3, f4, f5, f6, f7, f8), max_workers=6, use_threads=True)
        batch2 = (batch1 * 100).sum()
        post = dict(batch2.items())

    a1()
    a2()

    @timer #type: ignore
    def b1() -> None:
        batch1 = Batch.from_frames((f1, f2, f3, f4, f5, f6, f7, f8))
        batch2 = batch1.apply(func_b)
        _ = tuple(batch2.items())

    @timer #type: ignore
    def b2() -> None:
        batch1 = Batch.from_frames((f1, f2, f3, f4, f5, f6, f7, f8), max_workers=8, use_threads=False, chunksize=2)
        batch2 = batch1.apply(func_b)
        _ = tuple(batch2.items())

    # b1()
    # b2()







if __name__ == '__main__':
    main()