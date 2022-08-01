import static_frame as sf
from static_frame.core.type_blocks import TypeBlocks
from operator import attrgetter
import numpy as np
import typing as tp
from static_frame.core.index import IndexConstructor
from static_frame.core.index import Index
from static_frame.core.util import DTYPE_UINT_DEFAULT, ufunc_unique1d_indexer
from static_frame.core.util import ufunc_unique1d
from static_frame.core.util import DTYPE_INT_DEFAULT

init = sf.IndexHierarchy.from_product(range(500), tuple("abcdefghijklmnopqrstuvwxyz"), [True, False, None, object()])
ih1 = init.iloc[:10000]
ih2 = init.iloc[10000:40000]
ih3 = init.iloc[40000:]


def construct_indices_and_indexers_from_column_arrays(
        *,
        column_iter: tp.Iterable[np.ndarray],
        index_constructors_iter: tp.Iterable[IndexConstructor],
        ) -> tp.Tuple[tp.List[Index], np.ndarray]:
    indices: tp.List[Index] = []
    indexers: tp.List[np.ndarray] = []

    for column, constructor in zip(column_iter, index_constructors_iter):
        # Alternative approach that retains order
        # positions, indexer = ufunc_unique1d_positions(column)
        # unsorted_unique = column[np.sort(positions)]
        # indexer_remap = ufunc_unique1d_indexer(unsorted_unique)[1]
        # indexer = indexer_remap[indexer]
        # unique_values = unsorted_unique

        unique_values, indexer = ufunc_unique1d_indexer(column)

        # we call the constructor on all lvl, even if it is already an Index
        indices.append(constructor(unique_values))
        indexers.append(indexer)

    indexers = np.array(indexers, dtype=DTYPE_INT_DEFAULT)
    indexers.flags.writeable = False # type: ignore

    return indices, indexers


def approach_1(*ihs):
    tb_blocks_getter = attrgetter("_blocks")

    tb = TypeBlocks.from_blocks(TypeBlocks.vstack_blocks_to_blocks(list(map(tb_blocks_getter, ihs))))
    size, depth = tb.shape

    def gen_columns():
        for i in range(depth):
            yield tb._extract_array_column(i).reshape(size)

    indices, indexers = construct_indices_and_indexers_from_column_arrays(
            column_iter=gen_columns(),
            index_constructors_iter=[sf.Index,sf.Index,sf.Index],
            )

    return sf.IndexHierarchy(indices=indices, indexers=indexers)



def approach_2(*ihs):
    index_groups = []

    for ih in ihs:
        for i, index in enumerate(ih._indices):
            if len(index_groups) == i:
                index_groups.append([])
            index_groups[i].append(index)

    indices = []
    indexers = []
    for depth, index_group in enumerate(index_groups):
        index = index_group[0]
        for new_index in index_group[1:]:
            index = index.union(new_index)

        indices.append(index)

        indexer_groups = []
        for ih in ihs:
            mapped = ih._indices[depth]._index_iloc_map(index)
            indexer_groups.append(mapped[ih._indexers[depth]])
        indexers.append(np.hstack(indexer_groups))

    indexers = np.array(indexers, dtype=DTYPE_INT_DEFAULT)
    indexers.flags.writeable = False

    return sf.IndexHierarchy(indices=indices, indexers=indexers)



"""
01:06:20  In [75] %timeit approach_1(ih1, ih2)
244 µs ± 18.1 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

01:06:25  In [76] %timeit approach_2(ih1, ih2)
305 µs ± 6.79 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

"""