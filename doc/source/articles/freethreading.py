

import time

import numpy as np
import static_frame as sf
import sys

def main():
    # assert sys._is_gil_enabled() is False

    a1 = np.arange(100_000_000).reshape(100_000, 1000)
    f1 = sf.Frame(a1)

    t = time.time()
    y = f1.iter_series(axis=1).apply_pool(lambda s: ((s % 2) == 0).sum(), chunksize=100, use_threads=True)
    print('threaded', time.time() - t)

    t = time.time()
    x = f1.iter_series(axis=1).apply(lambda s: ((s % 2) == 0).sum())
    print('non-threaded', time.time() - t)



    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()