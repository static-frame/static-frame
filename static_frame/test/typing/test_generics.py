import numpy as np

import static_frame as sf


def run_idx_int(idx: sf.Index[np.int64]) -> bool:
    return True

def run_bool(idx: bool) -> bool:
    return True


def test_generics_a() -> None:

    idx1: sf.Index[np.int64] = sf.Index((2, 3))
    idx2: sf.Index[np.unicode_] = sf.Index(('a', 'b'))

    x = run_idx_int(idx1)
    # y = run_idx_int(idx2) # this fails

    # x = run_bool(idx1)

    dt1: np.dtype[np.int64] = idx1.dtype
    # dt2: np.dtype[np.int64] = idx2.dtype # this fails



# def test_generics_index_hierarchy_a() -> None:
#     # NOTE: calling this with mypy --enable-incomplete-feature=TypeVarTuple --enable-incomplete-feature=Unpack '/home/ariza/src/static-frame/static_frame/test/typing/test_generics.py' results in an error

#     TIH = sf.IndexHierarchy[sf.Index[np.unicode_], sf.IndexDate]
#     ih1: TIH = sf.IndexHierarchy.from_product(('a', 'b'), ('2022-01-01', '1954-04-05'))

#     def proc(ih1: TIH) -> sf.Index[np.unicode_]:
#         return ih1.index_at_depth(0)

#     ih2 = proc(ih1)




