import static_frame as sf
from static_frame import Index, IndexHierarchy
from static_frame.core.frame import Frame
import static_frame.core.frame  # noqa
import frame_fixtures as ff


def setup() -> dict:
    frame = (
        ff.parse('s(20,4)|v(int)')
        .assign[0]
        .apply(lambda s: s % 4)
        .assign[1]
        .apply(lambda s: s % 3)
    )
    return {'frame': frame}


def run(state: dict) -> None:
    frame = state['frame']
    post1 = frame.pivot([0, 1])
    post2 = frame.pivot([0, 1], index_constructor=IndexHierarchy.from_labels)
    state['post1'] = post1
    state['post2'] = post2


def verify(state: dict) -> None:
    post1 = state['post1']
    post2 = state['post2']

    assert post1.index.name == (0, 1)
    assert post1.index.__class__ is Index

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
    # original test uses assertTrue on to_pairs() (truthy check), preserve that
    assert post1.to_pairs()
    assert expected

    assert post2.index.name == (0, 1)
    assert post2.index.__class__ is IndexHierarchy
    assert post2.to_pairs()
