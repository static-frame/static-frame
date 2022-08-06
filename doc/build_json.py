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

    key_to_sig_full = {}
    sig_full_to_key = {}
    key_to_doc = {}
    method_to_keys = defaultdict(list)
    keys = []
    methods = set()

    for cls in DOCUMENTED_COMPONENTS:
        inter = InterfaceSummary.to_frame(cls, #type: ignore
                minimized=False,
                max_args=99,
                )
        for sig_full, row in inter.iter_series_items(axis=1):
            key = f'{row["cls_name"]}.{row["signature_no_args"]}'
            keys.append(key)

            sig_full = f'{row["cls_name"]}.{sig_full}'
            key_to_sig_full[key] = sig_full
            sig_full_to_key[sig_full] = key

            key_to_doc[key] = row['doc']

            method_to_keys[row["signature_no_args"]].append(key)
            methods.add(row["signature_no_args"])

    assert len(methods) == len(method_to_keys)
    assert len(keys) == len(key_to_sig_full)
    assert len(keys) == len(key_to_doc)
    assert len(keys) == len(sig_full_to_key)

    with ZipFile(fp, mode='w', allowZip64=True) as archive:
        for name, bundle in (
                ('key_to_sig_full', key_to_sig_full),
                ('sig_full_to_key', key_to_sig_full),
                ('key_to_dic', key_to_doc),
                ('method_to_keys', method_to_keys),
                ('keys', keys),
                ('methods', tuple(methods)),
                ):
            with archive.open(f'{name}.json', 'w', force_zip64=True) as f:
                f.write(json.dumps(bundle).encode('utf-8'))

    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    build(Path('/tmp/sf-api.zip'))