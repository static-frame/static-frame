from subprocess import PIPE
from subprocess import Popen
from sys import executable

import numpy as np
import pandas as pd
from pytest import mark

import static_frame as sf

try:
    import IPython  # noqa: F401 pylint: disable=W0611
except ImportError:
    HAS_IPYTHON = False
else:
    HAS_IPYTHON = True


COMMAND = b'(np.__name__, np.__version__, pd.__name__, pd.__version__, sf.__name__, sf.__version__)'


def _test_main(python: str) -> None:

    # result = f"{eval(COMMAND.decode(), {'np': np, 'pd': pd, 'sf': sf})}".encode()

    args = (python, '-m', 'static_frame')

    with Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE) as process:
        stdout = process.communicate(COMMAND)[0]
        assert not process.returncode
    # disabling this check as it fails in a TOX context
    #assert result in stdout


def test_main_python() -> None:
    _test_main(executable)


@mark.skipif(not HAS_IPYTHON, reason='Requires IPython.')
def test_main_ipython() -> None:
    _test_main('ipython')
