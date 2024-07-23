import numpy as np

from static_frame.core.store_config import label_encode_tuple


def test_label_encode_tuple_a():
    assert label_encode_tuple(('a', 3)) == "('a', 3)"
    assert label_encode_tuple((2, 3)) == "(2, 3)"
    assert label_encode_tuple((2, 3, 'a', 'b')) == "(2, 3, 'a', 'b')"

def test_label_encode_tuple_b():
    assert label_encode_tuple(tuple(np.array([3, 2]))) == "(3, 2)"
    assert label_encode_tuple(tuple(np.array(['a', 'b']))) == "('a', 'b')"

def test_label_encode_tuple_c():
    assert label_encode_tuple(tuple(np.array([3, 2, None, 'b'], dtype=object))) == "(3, 2, None, 'b')"

