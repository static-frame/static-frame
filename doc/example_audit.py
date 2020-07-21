
import typing as tp
from static_frame.test.unit.test_doc import api_example_str

PREFIX_START = '#start_'
PREFIX_END = '#end_'

def get_defined() -> tp.Set[str]:

    defined = set()
    signature_start = ''
    signature_end = ''

    for line in api_example_str.split('\n'):
        if line.startswith(PREFIX_START):
            signature_start = line.replace(PREFIX_START, '').strip()
        elif line.startswith(PREFIX_END):
            signature_end = line.replace(PREFIX_END, '').strip()
            if signature_start == signature_end:
                if signature_start in defined:
                    raise RuntimeError(f'duplicate definition: {signature_start}')
                defined.add(signature_start)
                signature_start = ''
                signature_end = ''
            else:
                raise RuntimeError(f'mismatched: {signature_start}: {signature_end}')

    return defined