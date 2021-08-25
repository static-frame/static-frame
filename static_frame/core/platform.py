import typing as tp
import importlib
import platform as platform_mod
import sys

from static_frame.core.series import Series
from static_frame.core.display import Display
import static_frame

class Platform:

    @staticmethod
    def to_series() -> Series:
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
                    yield package, mod.__version__ #type: ignore
                elif hasattr(mod, 'version'): # msgpack
                    yield package, mod.version #type: ignore
                else:
                    yield package, None

        return Series.from_items(items(), name='platform')

    @classmethod
    def display(cls) -> Display:
        return cls.to_series().display_wide() #type: ignore [no-any-return]


