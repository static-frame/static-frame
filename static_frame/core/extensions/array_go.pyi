# pylint: disable = all


import typing

import numpy  # type: ignore


class ArrayGO:

    values: numpy.array

    def __init__(self, iterable: typing.Iterable[object], *, dtype: object = ..., own_iterable: bool = ...) -> None: ...

    def __iter__(self) -> typing.Iterator[typing.Any]: ...

    def __getitem__(self, __key: object) -> typing.Any: ...

    def __len__(self) -> int: ...

    def append(self, __value: object) -> None: ...

    def copy(self) -> 'ArrayGO': ...

    def extend(self, __values: typing.Iterable[object]) -> None: ...
