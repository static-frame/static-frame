from copy import copy

from static_frame.core.exception import GrowOnlyInvalid
from static_frame.core.exception import StoreFileMutation


def test_exception_a() -> None:
    e1 = StoreFileMutation()
    e2 = copy(e1)


def test_exception_b() -> None:
    e1 = GrowOnlyInvalid()
    e2 = copy(e1)



