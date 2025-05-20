import numpy as np
# import static_frame as sf


# def process1(x: np.ndarray[tuple[int], np.dtype[np.int16]]): ...

def process1(x: int): ...

# a1 = np.empty(100, dtype=np.int16)
# process1(a1) # mypy passes

# a2 = np.empty(100, dtype=np.float32)
# process1(a2) # mypy fails


process1(None)
print('done')

