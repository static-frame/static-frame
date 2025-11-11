import dataclasses
import inspect
import typing as tp

import numpy as np
from pytest import mark

from static_frame.core.store import Store
from static_frame.core.store_config import (
    StoreConfig,
    label_encode_tuple,
)
from static_frame.core.store_sqlite import StoreSQLite
from static_frame.core.store_xlsx import StoreXLSX
from static_frame.core.store_zip import (
    StoreZipCSV,
    StoreZipNPY,
    StoreZipNPZ,
    StoreZipParquet,
    StoreZipPickle,
    StoreZipTSV,
)


def test_label_encode_tuple_a():
    assert label_encode_tuple(('a', 3)) == "('a', 3)"
    assert label_encode_tuple((2, 3)) == '(2, 3)'
    assert label_encode_tuple((2, 3, 'a', 'b')) == "(2, 3, 'a', 'b')"


def test_label_encode_tuple_b():
    assert label_encode_tuple(tuple(np.array([3, 2]))) == '(3, 2)'
    assert label_encode_tuple(tuple(np.array(['a', 'b']))) == "('a', 'b')"


def test_label_encode_tuple_c():
    assert (
        label_encode_tuple(tuple(np.array([3, 2, None, 'b'], dtype=object)))
        == "(3, 2, None, 'b')"
    )


@mark.parametrize(
    'cls',
    (
        StoreZipCSV,
        StoreZipNPZ,
        StoreZipParquet,
        StoreZipPickle,
        StoreZipTSV,
    ),
)
def test_store_config_subclasses(cls: type[Store[StoreConfig]]) -> None:
    skip = {field.name for field in dataclasses.fields(StoreConfig)}
    defined = {
        field.name
        for field in dataclasses.fields(cls._STORE_CONFIG_CLASS)
        if field.name not in skip
    }

    funcs = (cls._EXPORTER, cls._STORE_CONFIG_CLASS._CONSTRUCTOR)

    allowed = set()
    for func in funcs:
        if func is None:
            continue

        allowed |= {
            name
            for name, param in inspect.signature(func).parameters.items()
            if param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD)
        }

    if missing := defined - allowed:
        raise ValueError(
            f'The following fields are defined on {cls.__name__} but are '
            f"not part of {[f.__name__ for f in funcs]}'s signature: {missing}"
        )
