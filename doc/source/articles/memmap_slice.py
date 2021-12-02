import numpy as np

import static_frame as sf
import frame_fixtures as ff


if __name__ == '__main__':

    # prlimit --as=850000000 python3 doc/source/articles/memmap_slice.py

    fp = '/tmp/big_frame'


    a1 = np.arange(10_000_000).reshape(1_000_000, 10)
    columns = tuple('abcdefghij')
    f1 = sf.Frame(a1, columns=columns)

    print('to npy')
    f1.to_npy(fp)

    # # loading two of these fails
    # print('start from_npy f2')
    # f2 = sf.Frame.from_npy(fp)

    # print('start from_npy f3')
    # f3 = sf.Frame.from_npy(fp)

    # we can create two of these

    print('start from_npy f2 from memmap')
    f2 = sf.Frame.from_npy(fp, memory_map=True)

    print('start from_npy f3 from memmap')
    f3 = sf.Frame.from_npy(fp, memory_map=True)

    for label, col in f3.items():
        print(label, col.shape)

    # print('start from_npy f4 from memmap')
    # f4 = sf.Frame.from_npy(fp, memory_map=True)

