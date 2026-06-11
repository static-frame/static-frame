from __future__ import annotations

from string import ascii_letters

from static_frame.core.index_base import IndexBase
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.index_hierarchy_set_utils import index_hierarchy_union
from static_frame.core.util import SortStatus


def setup() -> dict:
    ih = IndexHierarchy.from_product(tuple(ascii_letters), range(100), [True, False])

    size = len(ih) // 100
    half = size // 2

    indices = []
    for i in range(100):
        if i == 0:
            sl = slice(0, size * (i + 1) + half)
        elif i == 100 - 1:
            sl = slice(size * i - half, None)
        else:
            sl = slice(size * i - half, size * (i + 1) + half)

        indices.append(ih.iloc[sl])

    # Pre-compute the expected reference result once (outside the hot path).
    expected = IndexBase.union(*indices).sort()

    return {
        'ih': ih,
        'indices': indices,
        'expected': expected,
    }


def run(state: dict) -> None:
    indices = state['indices']
    actual = index_hierarchy_union(IndexHierarchy, *indices).sort()
    state['actual'] = actual


def verify(state: dict) -> None:
    expected = state['expected']
    actual = state['actual']
    assert expected._sort_status is SortStatus.ASC
    assert actual._sort_status is SortStatus.ASC
    assert actual.equals(expected), (
        expected.rename('expected'),
        actual.rename('actual'),
    )
