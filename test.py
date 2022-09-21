import numpy as np

def numpy_loop(arr):
    itr = iter(arr)
    s = []
    try:
        while True:
            s.append(next(itr))
            if len(s) > 5:
                    s.pop(0)
            yield s[-1]
    except:
        pass

a = np.array([np.array([None, None, i // 2]) for i in range(10)])

print('a', a)

print('a', set([id(el) for el in a]))
print('a numpy_loop', set([id(el) for el in numpy_loop(a)]))

# b = np.array([np.array([None, None, 1]), np.array([None, None, 2]), ['a', 'b', 'c']])
# print('b', set([id(el) for el in b]))

# c = np.array([np.array([None, None, 1]), np.array([None, None, 2]), ['a', 'b', 'c'], (2,3,4)])
# print('c', set([id(el) for el in c]))

# o1 = (2,3,4)
# print('o1:', id(o1))
# d = np.array([np.array([None, None, 1]), np.array([None, None, 2]), o1, ['a', 'b', 'c']])
# print('d', [id(el) for el in d])
# print('d', set([id(el) for el in d]))

# print(d)
# print(repr(d[-2]), repr(o1))
# print(d[-2] is o1)
# print(d[-2] == o1)
# print()

# e = np.array([np.array([None, None, 1]), np.array([None, None, 2]), ['a', 'b', 'c'], (2,3,4), 'abc'], dtype=np.object)
# print('e', [id(el) for el in e])
# print('e', set([id(el) for el in e]))

# f = np.array([np.array([None, None, 1]), np.array([None, None, 2]), ['a', 'b', 'c'], o1, 'abc'], dtype=np.object)
# print('f', [id(el) for el in f])
# print('f', set([id(el) for el in f]))
