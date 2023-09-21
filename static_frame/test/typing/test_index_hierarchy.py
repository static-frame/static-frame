import numpy as np
import typing_extensions as tp

import static_frame as sf

IH = sf.IndexHierarchy
I = sf.Index

def test_hierarchy_a() -> None:

    ih1 = IH[I[np.int64], I[np.unicode_]].from_labels(((10, 'a'), (20, 'b')))
    assert len(ih1) == 2

    ih2 = IH[I[np.int64], I[np.int64]].from_labels(((10, 3), (20, 4)))
    assert len(ih1) == 2

    def proc1(ih: IH[I[np.int64], I[np.unicode_]]) -> int: # type: ignore[type-arg]
        return len(ih)

    def proc2(ih: IH[I[np.int64], I[np.int64]]) -> int: # type: ignore[type-arg]
        return len(ih)

    l1 = proc1(ih1)
    # l1 = proc2(ih1) # this  fails pyright
    l2 = proc2(ih2)


def test_hierarchy_b() -> None:
    ih1 = IH[I[np.int64], I[np.int64]].from_labels(((10, 1), (20, 2)))
    ih2 = IH[I[np.int64], I[np.int64], I[np.int64]].from_labels(((10, 1, 4), (20, 2, 7)))

    ih3 = IH[I[np.int64], I[np.str_], I[np.int64]].from_labels(((10, 'x', 4), (20, 'y', 7)))


    def proc1(ih: IH[tp.Unpack[tp.Tuple[I[np.int64], ...]]]) -> int: # type: ignore[type-arg]
        return ih.depth

    x1 = proc1(ih1)
    assert x1 == 2

    x2 = proc1(ih2)
    assert x2 == 3

    # x3 = proc1(ih3) # pyright fails

