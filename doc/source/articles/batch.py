



import numpy as np

from static_frame.core.frame import Frame
from static_frame.core.batch import Batch








import time

def timer(f):
    def wraped(*args, **kwargs):
        t = time.time()
        post = f()
        print(time.time() - t, f)
        return post
    return wraped



def func_b(frame: Frame) -> Frame:
    for row in frame.iter_series():
        if row[10] > 10000:
            print(row)
    return frame


def main():

    f1 = Frame(np.arange(100000000).reshape(1000000, 100), name='a')
    f2 = Frame(np.arange(100000000).reshape(1000000, 100), name='b')
    f3 = Frame(np.arange(100000000).reshape(1000000, 100), name='c')
    f4 = Frame(np.arange(100000000).reshape(1000000, 100), name='d')
    f5 = Frame(np.arange(100000000).reshape(1000000, 100), name='e')
    f6 = Frame(np.arange(100000000).reshape(1000000, 100), name='f')
    f7 = Frame(np.arange(100000000).reshape(1000000, 100), name='g')
    f8 = Frame(np.arange(100000000).reshape(1000000, 100), name='h')



    @timer
    def a1():
        batch1 = Batch.from_frames((f1, f2, f3, f4, f5, f6, f7, f8))
        batch2 = (batch1 * 100).sum()
        return tuple(batch2.items())

    @timer
    def a2():
        batch1 = Batch.from_frames((f1, f2, f3, f4, f5, f6, f7, f8), max_workers=8, use_threads=True, chunksize=2)
        batch2 = (batch1 * 100).sum()
        return tuple(batch2.items())

    # post_a1 = a1()
    # post_a2 = a2()


    @timer
    def b1():
        batch1 = Batch.from_frames((f1, f2, f3, f4, f5, f6, f7, f8))
        batch2 = batch1.apply(func_b)
        return tuple(batch2.items())

    @timer
    def b2():
        batch1 = Batch.from_frames((f1, f2, f3, f4, f5, f6, f7, f8), max_workers=8, use_threads=False, chunksize=2)
        batch2 = batch1.apply(func_b)
        return tuple(batch2.items())

    post_b1 = b1()
    post_b2 = b2()







if __name__ == '__main__':
    main()