from __future__ import annotations

import importlib
import platform as platform_mod
import sys

import numpy as np
import typing_extensions as tp

import static_frame
from static_frame.core.display import Display
from static_frame.core.index import Index
from static_frame.core.series import Series


class Platform:

    @staticmethod
    def to_series() -> Series[Index[np.str_], tp.Any]:
        def items() -> tp.Iterator[tp.Tuple[str, tp.Any]]:
            yield 'platform', platform_mod.platform()
            yield 'sys.version', sys.version.replace('\n', '')
            yield 'static-frame', static_frame.__version__

            # NOTE: see requirements-extras.txt
            for package in (
                    'numpy',
                    'pandas',
                    'xlsxwriter',
                    'openpyxl',
                    'xarray',
                    'tables',
                    'pyarrow',
                    'msgpack',
                    'msgpack_numpy',
                    ):
                mod = None
                try:
                    mod = importlib.import_module(package)
                except ModuleNotFoundError: #pragma: no cover
                    yield package, ModuleNotFoundError #pragma: no cover
                    continue #pragma: no cover

                if hasattr(mod, '__version__'):
                    yield package, mod.__version__
                else:
                    yield package, None

        return Series.from_items(items(), name='platform')

    @classmethod
    def display(cls) -> Display:
        return cls.to_series().display_wide()


