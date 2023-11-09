from __future__ import annotations


import frame_fixtures as ff
import numpy as np
import typing_extensions as tp

import static_frame as sf
from static_frame.test.test_case import skip_pyle310


# NOTE: this test is a syntax error when run on le310
@skip_pyle310
def test_check_frame_h():
    f = sf.Frame.from_records(([3, '192004', 0.3], [3, '192005', -0.4]), columns=('permno', 'yyyymm', 'Mom3m'))

    cr = f.via_type_clinic(sf.Frame[sf.Index[np.int64], sf.TIndexAny, *tuple[tp.Any, ...]])
    assert len(cr) == 0


