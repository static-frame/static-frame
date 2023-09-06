import typing as tp

import static_frame as sf


def test_index_len_a() -> None:

    idx = sf.Index(('a', 'b', 'c'))
    l: int = len(idx)
    assert l == 3

def test_index_len_b() -> None:

    idx = sf.Frame.from_records([('a', 'b'), ('c', 'd')])
    l: int = len(idx.columns)
    assert l == 2
