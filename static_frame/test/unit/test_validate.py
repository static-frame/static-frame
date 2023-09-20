import typing_extensions as tp
import numpy as np
import static_frame as sf
from static_frame.core.validate import validate_pair


def test_validate_pair_a():

    validate_pair(sf.IndexDate(('2022-01-01',)), sf.IndexDate)
    validate_pair(sf.IndexDate(('2022-01-01',)), tp.Any)

def test_validate_pair_b():
    v = sf.Series(('a', 'b'), index=sf.IndexDate(('2021-04-05', '2022-05-03')))
    h = sf.Series[sf.IndexDate, np.str_]

    validate_pair(v, h)
