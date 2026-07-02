import frame_fixtures as ff

from static_frame import Index, IndexHierarchy


def setup() -> dict:
    # Small frame matching the original test's expected values.
    frame_small = (
        ff.parse('s(20,4)|v(int)')
        .assign[0]
        .apply(lambda s: s % 4)
        .assign[1]
        .apply(lambda s: s % 3)
    )

    # Larger frame to exercise pivot at scale. Reduce cardinality of the
    # first two columns so grouping produces many rows per group.
    frame_large = (
        ff.parse('s(10000,6)|v(int)')
        .assign[0]
        .apply(lambda s: s % 20)
        .assign[1]
        .apply(lambda s: s % 15)
    )

    return dict(frame_small=frame_small, frame_large=frame_large)


def run(state: dict) -> None:
    frame_small = state['frame_small']
    frame_large = state['frame_large']

    # Correctness targets (small frame).
    state['post1'] = frame_small.pivot([0, 1])
    state['post2'] = frame_small.pivot(
        [0, 1], index_constructor=IndexHierarchy.from_labels
    )

    # Scale work: pivot over the larger frame with both index styles.
    state['post_large1'] = frame_large.pivot([0, 1])
    state['post_large2'] = frame_large.pivot(
        [0, 1], index_constructor=IndexHierarchy.from_labels
    )


def verify(state: dict) -> None:
    post1 = state['post1']
    assert post1.index.name == (0, 1)
    assert post1.index.__class__ is Index

    post2 = state['post2']
    assert post2.index.name == (0, 1)
    assert post2.index.__class__ is IndexHierarchy

    expected = (
        (
            2,
            (
                ((0, 0), 463099),
                ((0, 1), -88017),
                ((0, 2), 35021),
                ((1, 0), 92867),
                ((1, 2), 96520),
                ((2, 0), 172133),
                ((2, 1), 279191),
                ((2, 2), 13448),
                ((3, 0), 255338),
                ((3, 1), 372807),
                ((3, 2), 155574),
            ),
        ),
        (
            3,
            (
                ((0, 0), 348362),
                ((0, 1), 175579),
                ((0, 2), 105269),
                ((1, 0), 58768),
                ((1, 2), 13448),
                ((2, 0), 84967),
                ((2, 1), 239151),
                ((2, 2), 170440),
                ((3, 0), 269300),
                ((3, 1), 204528),
                ((3, 2), 493169),
            ),
        ),
    )

    pairs1 = tuple(
        (int(col), tuple((tuple(int(x) for x in k), int(v)) for k, v in items))
        for col, items in post1.to_pairs()
    )
    assert pairs1 == expected

    pairs2 = tuple(
        (int(col), tuple((tuple(int(x) for x in k), int(v)) for k, v in items))
        for col, items in post2.to_pairs()
    )
    assert pairs2 == expected
