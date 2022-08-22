# from io import StringIO
import typing as tp
from collections import defaultdict
import json
from pathlib import Path
from zipfile import ZipFile
# import os
# import sys
# import datetime

import numpy as np
import static_frame as sf
import pandas as pd

from static_frame.core.interface import InterfaceSummary
from static_frame.core.interface import InterfaceGroup
from static_frame.core.container_util import ContainerMap

from doc.source.conf import DOCUMENTED_COMPONENTS

def build(fp: Path) -> str:

    if fp.suffix != '.zip':
        raise RuntimeError('suffix must be .zip')

    sig_to_sig_full = {}
    # sig_full_to_key = {}
    sig_to_doc = {}
    method_to_sig = defaultdict(list) # one to many mapping from unqualified methods to keys
    sigs = []
    methods = set() # on

    for cls in DOCUMENTED_COMPONENTS:
        inter = InterfaceSummary.to_frame(cls, #type: ignore
                minimized=False,
                max_args=99,
                )
        for sig_full, row in inter.iter_series_items(axis=1):
            key = f'{row["cls_name"]}.{row["signature_no_args"]}'
            sigs.append(key)

            sig_full = f'{row["cls_name"]}.{sig_full}'
            sig_to_sig_full[key] = sig_full
            # sig_full_to_key[sig_full] = key

            sig_to_doc[key] = row['doc']

            method_to_sig[row["signature_no_args"]].append(key)
            methods.add(row["signature_no_args"])

    assert len(methods) == len(method_to_sig)
    assert len(sigs) == len(sig_to_sig_full)
    assert len(sigs) == len(sig_to_doc)
    import ipdb; ipdb.set_trace()

    with ZipFile(fp, mode='w', allowZip64=True) as archive:
        for name, bundle in (
                ('sig_to_sig_full', sig_to_sig_full),
                # ('sig_full_to_key', sig_full_to_key),
                ('sig_to_doc', sig_to_doc),
                ('method_to_sig', method_to_sig),
                ('sigs', sigs),
                ('methods', tuple(methods)),
                ):
            with archive.open(f'{name}.json', 'w', force_zip64=True) as f:
                f.write(json.dumps(bundle).encode('utf-8'))

    # import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    build(Path('/tmp/sf-api.zip'))