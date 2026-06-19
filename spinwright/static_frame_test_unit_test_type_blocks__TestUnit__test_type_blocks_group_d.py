"""Measurement harness for test_type_blocks_group_d."""
import numpy as np
from frame_fixtures import parse as ff_parse
from static_frame.core.type_blocks import TypeBlocks


def setup() -> dict:
    """Build inputs for the group operation."""
    # Create a larger TypeBlocks for meaningful measurement
    # Using the same pattern as the test but scaled up
    # The original test: s(6,2)|v(int) -> assign[0] apply(lambda s: s % 4)
    # We'll scale to 1000 rows, 10 columns to get multiple groups
    tb = ff_parse('s(1000,10)|v(int)').assign[0].apply(lambda s: s % 4)._blocks
    return {'tb': tb}


def run(state: dict) -> None:
    """Execute the group operation under measurement."""
    tb = state['tb']
    # Group by column 0 (axis=1 means group columns by key column)
    post = tuple(tb.group(axis=1, key=0))
    state['post'] = post


def verify(state: dict) -> None:
    """Verify correctness of the group operation."""
    post = state['post']
    shapes = [x.shape for _, _, x in post]
    
    # Original test expects: [(6, 1), (6, 1)]
    # For our scaled version (1000, 10), we expect 10 groups of (1000, 1)
    # because modulo 4 on column 0 creates values 0, 1, 2, 3
    # and there are 10 columns total
    expected_num_groups = state['tb'].shape[1]  # 10 columns
    expected_shape = (state['tb'].shape[0], 1)  # (1000, 1)
    
    assert len(post) == expected_num_groups, f"Expected {expected_num_groups} groups, got {len(post)}"
    assert all(s == expected_shape for s in shapes), f"Expected all shapes {expected_shape}, got {shapes}"