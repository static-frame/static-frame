"""Measurement harness extracted from:
static_frame/test/unit/test_type_blocks.py::TestUnit::test_type_blocks_group_d

Original test:
    def test_type_blocks_group_d(self) -> None:
        tb1 = ff.parse('s(6,2)|v(int)').assign[0].apply(lambda s: s % 4)._blocks
        post = tuple(tb1.group(axis=1, key=0))
        self.assertEqual([x.shape for _, _, x in post], [(6, 1), (6, 1)])

The operation groups columns of a TypeBlocks along axis=1 using column 0 as
the key. The original input is a 6x2 Frame; we scale it up by increasing the
row count and adding more per-column blocks, producing proportionally more
groups of shape (N, 1).
"""

import numpy as np
import static_frame as sf


# Scale factors for the measurement workload. Increasing the row count and the
# number of "extra" columns grows the amount of work performed by `group`.
N_ROWS = 2000
N_EXTRA_COLS = 20  # additional columns beyond the key column
KEY_MOD = 8  # number of unique values permitted in the key column


def setup() -> dict:
    """Build a deterministic Frame and extract its TypeBlocks for grouping.

    Mirrors the original construction:
        ff.parse('s(6,2)|v(int)').assign[0].apply(lambda s: s % 4)._blocks
    but at a larger scale. We use `consolidate_blocks=False` to ensure each
    column lives in its own block, which scales the grouping workload
    proportionally with the column count.
    """
    # Seed RNG so the operation under test sees a deterministic input.
    np.random.seed(42)

    # Key column: limited set of distinct values (mirrors `s % 4` in original).
    key = np.random.randint(0, KEY_MOD, size=N_ROWS)

    # Additional columns: wide range so each column is its own group identity.
    extra_cols = [
        np.random.randint(0, 1_000_000, size=N_ROWS)
        for _ in range(N_EXTRA_COLS)
    ]

    # Build Frame with per-column blocks via consolidate_blocks=False.
    items = [(0, key)] + [(i + 1, col) for i, col in enumerate(extra_cols)]
    f = sf.Frame.from_items(items, index=None, consolidate_blocks=False)

    # Apply the same transform as the original test on column 0.
    f2 = f.assign[0].apply(lambda s: s % KEY_MOD)

    tb1 = f2._blocks

    # Expected group count: 1 (the key column) + N_EXTRA_COLS (each extra
    # column forms its own group because its values are all unique).
    expected_group_count = 1 + N_EXTRA_COLS

    return {
        "tb1": tb1,
        "expected_group_count": expected_group_count,
        "expected_shape_rows": N_ROWS,
        "post": None,
    }


def run(state: dict) -> None:
    """Hot path: group TypeBlocks along axis=1 using column 0 as the key."""
    state["post"] = tuple(state["tb1"].group(axis=1, key=0))


def verify(state: dict) -> None:
    """Correctness check adapted from the original assertion."""
    post = state["post"]

    # Original assertion checked the shapes of each produced group.
    shapes = [x.shape for _, _, x in post]
    assert len(shapes) == state["expected_group_count"], (
        f"expected {state['expected_group_count']} groups, got {len(shapes)}: {shapes}"
    )
    expected_row_count = state["expected_shape_rows"]
    for shape in shapes:
        assert shape[0] == expected_row_count, f"unexpected row count: {shape}"
        assert shape[1] == 1, f"each group should be a single column, got {shape}"
