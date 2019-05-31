# pylint: disable = all


import typing

import numpy  # type: ignore


def immutable_filter(__array: numpy.array) -> numpy.array: ...

def mloc(__array: numpy.array) -> int: ...

def name_filter(__name: typing.Hashable) -> typing.Hashable: ...

def resolve_dtype_iter(__dtypes: typing.Iterable[numpy.dtype]) -> numpy.dtype: ...

def _resolve_dtype(__d1: numpy.dtype, __d2: numpy.dtype) -> numpy.dtype: ...
