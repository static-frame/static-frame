'''Drop into an interactive interpreter pre-loaded with sf, np, and (if installed) pd.

$ python -m static_frame
$ ipython -m static_frame
'''


from code import interact
from sys import platform
from sys import version

import numpy as np

import static_frame as sf
from static_frame.core.display_color import HexColor

imports = {'np': np, 'sf': sf}

try:
    import pandas as pd
except ImportError: #pragma: no cover
    pass #pragma: no cover
else:
    imports['pd'] = pd

try:
    import frame_fixtures as ff
except ImportError: #pragma: no cover
    pass #pragma: no cover
else:
    imports['ff'] = ff

commands = sorted(
        f'import {package.__name__} as {name} # {package.__version__}'
        for name, package in imports.items()
        )

try: # This lets us play nicely with IPython:
    from builtins import __IPYTHON__  #type: ignore
    from IPython import embed
    from IPython import get_ipython

except ImportError:
    is_ipython = False
else:
    is_ipython = __IPYTHON__


if __name__ == '__main__':

    if is_ipython:
        ipython = get_ipython()

        print() # Spacer.
        for command in commands:
            ipython.auto_rewrite_input(command)
        print() # Spacer.

        embed(user_ns=imports, colors='neutral')

    else:
        banner_head = f'Python {version} on {platform}\n'
        banner_body = '\n'.join(f'>>> {command}' for command in commands)
        interact(banner=(banner_head + HexColor.format_terminal(0x505050, banner_body)),
                local=imports,
                exitmsg=''
                )
