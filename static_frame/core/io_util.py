import typing as tp
import os
from pathlib import Path

from static_frame.core import util


def read_file(p: Path):
    with open(p) as f:
        yield from f


def to_line_iter(x: util.PathSpecifierOrFileLikeOrIterator) -> tp.Iterator[str]:
    '''Unify all input types taken by StaticFrame's constructors into an iterable of lines.
    '''
    if isinstance(x, (str, os.PathLike)):
        return read_file(Path(x))
    return x
