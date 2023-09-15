import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import typing_extensions as tp

from doc.build_example import to_json_bundle
# from doc.build_source import name_to_snake_case
from static_frame import __version__ as VERSION
from static_frame.core.interface import DOCUMENTED_COMPONENTS
from static_frame.core.interface import InterfaceSummary


def build(output: Path,
        write: bool = False,
        display: bool = False,
        component: tp.Optional[str] = None,
        zip_output: bool = False,
        ) -> tp.Optional[str]:
    if not write and not display:
        return

    sig_full_to_sig = {}
    sig_to_doc = {}
    sig_to_group = {}
    method_to_sig = defaultdict(list) # one to many mapping from unqualified methods to keys
    sigs = []
    # methods = set()

    for cls in DOCUMENTED_COMPONENTS:
        inter = InterfaceSummary.to_frame(cls, #type: ignore
                minimized=False,
                max_args=99,
                max_doc_chars=999_999,
                )
        for sig_full, row in inter.iter_series_items(axis=1):
            key = f'{row["cls_name"]}.{row["signature_no_args"]}'
            sigs.append(key)

            sig_full = f'{row["cls_name"]}.{sig_full}'
            sig_full_to_sig[sig_full] = key
            # sig_full_to_key[sig_full] = key

            sig_to_doc[key] = row['doc']
            sig_to_group[key] = row['group']

            method_to_sig[row["signature_no_args"]].append(key)
            # methods.add(row["signature_no_args"])

    sig_to_example = to_json_bundle()

    # assert len(methods) == len(method_to_sig)
    assert len(sigs) == len(sig_full_to_sig)
    assert len(sigs) == len(sig_to_doc)
    assert len(sigs) == len(sig_to_group)

    # as we want to create a Map in javascript, store values as array of pairs
#     itemize = lambda d: tuple(d.items())

    name_bundle = (
            ('sig_full_to_sig', sig_full_to_sig),
            ('sig_to_doc', sig_to_doc),
            ('sig_to_group', sig_to_group),
            ('method_to_sig', method_to_sig),
            ('sigs', sigs),
            # ('methods', tuple(sorted(methods))),
            ('sig_to_example', sig_to_example),
            ('metadata', {'version': VERSION})
            )

    if component:
        name_bundle = ((n, b) for n, b in name_bundle if n == component.lower())

    if display:
        for name, bundle in name_bundle:
            print(json.dumps(bundle, indent=4))

    if write and zip_output:
        fp = output / f'sf-api-{VERSION}.zip'
        if fp.exists():
            raise RuntimeError(f'output path exists: {fp}')
        with ZipFile(fp, mode='w', allowZip64=True) as archive:
            for name, bundle in name_bundle:
                with archive.open(f'{name}.json', 'w', force_zip64=True) as f:
                    f.write(json.dumps(bundle).encode('utf-8'))
        return str(fp)

    if write and not zip_output:
        for name, bundle in name_bundle:
            fp = output / f'{name}.json'
            with open(fp, 'w') as f:
                f.write(json.dumps(bundle))
            print(str(fp))

def get_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
            description='Build JSON',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            )
    p.add_argument('--print',
            help='Print output.',
            action='store_true',
            )
    p.add_argument('--write',
            help=f'Write output to --output.',
            action='store_true',
            )
    p.add_argument('--output',
            help=f'Directory to write output, or {tempfile.gettempdir()} by default.',
            default=Path(tempfile.gettempdir()),
            type=Path,
            )
    p.add_argument('--component',
            help='Name of JSON to process, else all.',
            default=None,
            )
    p.add_argument('--zip',
            help='Optionally zip output.',
            action='store_true',
            )
    return p


if __name__ == '__main__':

    options = get_arg_parser().parse_args()
    post = build(output=options.output,
            display=options.print,
            write=options.write,
            component=options.component,
            zip_output=options.zip,
            )
    if post:
        print(post)



