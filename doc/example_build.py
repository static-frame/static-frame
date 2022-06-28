from io import StringIO
import typing as tp

import numpy as np
import static_frame as sf
import pandas as pd

from static_frame.core.interface import INTERFACE_GROUP_ORDER
from static_frame.core.interface import InterfaceSummary
from static_frame.core.interface import InterfaceGroup
from static_frame.core.container_util import ContainerMap




SERIES_INIT_A = dict(values=(10, 2, 8), index=('a', 'b', 'c'))
SERIES_INIT_B = dict(values=(4, 3, 12), index=('d', 'e', 'f'))
SERIES_INIT_C = dict(values=(11, 1, None), index=('a', 'b', 'c'))
SERIES_INIT_D = dict(values=(2, 8, 19), index=('b', 'c', 'd'))

SERIES_INIT_DICT_A = dict(sf.Series(**SERIES_INIT_A))
SERIES_INIT_FROM_ELEMENT_A = dict(element=-1, index=('a', 'b', 'c'), name='x')
SERIES_INIT_FROM_ITEMS_A = dict(pairs=tuple(dict(sf.Series(**SERIES_INIT_A)).items()), name='x')


DC = sf.DisplayConfig(type_color=False)

def kwa(params, arg_first: bool = False):
    arg_only = set()
    if arg_first:
        arg_only.add(0)

    return ', '.join(
        f'{k}={repr(v)}' if i not in arg_only else f'{repr(v)}'
        for i, (k, v) in enumerate(params.items())
        )

def calls_to_msg(calls, row: sf.Series) -> tp.Iterator[str]:
    cls = ContainerMap.str_to_cls(row['cls_name'])

    yield f'#start_{cls.__name__}-{row["signature_no_args"]}'
    for call in calls:
        g = globals()
        g['sf'] = sf
        g['np'] = np
        g['pd'] = pd
        l = locals()

        print(call)
        try:
            yield f'>>> {call}'
            post = eval(call, g, l)
            if post is not None:
                yield from str(post).split('\n')
        except SyntaxError:
            exec(call, g, l)

    yield ''
    yield f'#end_{cls.__name__}-{row["signature_no_args"]}'
    yield ''


def series_constructors(row: sf.Series) -> tp.Iterator[str]:

    cls = ContainerMap.str_to_cls(row['cls_name'])
    icls = f'sf.{cls.__name__}' # interface cls
    attr = row['signature_no_args'][:-2] # drop paren
    iattr = f'{icls}.{attr}'

    if attr == '__init__':
        yield f's = {icls}({kwa(SERIES_INIT_A)})'
    elif attr == 'from_concat':
        yield f's1 = {icls}({kwa(SERIES_INIT_A)})'
        yield f's2 = {icls}({kwa(SERIES_INIT_B)})'
        yield f's = {iattr}((s1, s2))'
    elif attr == 'from_concat_items':
        yield f's1 = {icls}({kwa(SERIES_INIT_A)})'
        yield f's2 = {icls}({kwa(SERIES_INIT_B)})'
        yield f"s = {iattr}((('x', s1), ('y', s2)))"
    elif attr == 'from_dict':
        yield f's = {iattr}(dict({kwa(SERIES_INIT_DICT_A)}))'
    elif attr == 'from_element':
        yield f's = {iattr}({kwa(SERIES_INIT_FROM_ELEMENT_A)})'
    elif attr == 'from_items':
        yield f's = {iattr}({kwa(SERIES_INIT_FROM_ITEMS_A)})'
    elif attr == 'from_overlay':
        yield f's1 = {icls}({kwa(SERIES_INIT_C)})'
        yield f's1'
        yield f's2 = {icls}({kwa(SERIES_INIT_D)})'
        yield f"s = {iattr}((s1, s2))"
    elif attr == 'from_pandas':
        yield f'df = pd.Series({kwa(SERIES_INIT_A, arg_first=True)})'
        yield f's = {iattr}(df)'
    else:
        raise NotImplementedError(f'no handling for {attr}')
    yield f's'

def series_attribute(row: sf.Series) -> tp.Iterator[str]:

    cls = ContainerMap.str_to_cls(row['cls_name'])
    icls = f'sf.{cls.__name__}' # interface cls
    attr = row['signature_no_args'] # drop paren

    yield f's = {icls}({kwa(SERIES_INIT_A)})'
    yield f's.{attr}'

def series_dictlike(row: sf.Series) -> tp.Iterator[str]:

    cls = ContainerMap.str_to_cls(row['cls_name'])
    icls = f'sf.{cls.__name__}' # interface cls
    attr = row['signature_no_args']
    attr_func = row['signature_no_args'][:-2]

    yield f's = {icls}({kwa(SERIES_INIT_A)})'

    if attr == '__contains__()':
        yield f"s.{attr_func}('a')"
    elif attr == 'get()':
        yield f"s.{attr_func}('a')"
        yield f"s.{attr_func}('z', -1)"
    elif attr == 'values':
        yield f"s.{attr}"
    elif attr == 'items()':
        yield f"tuple(s.{attr_func}())"
    else:
        yield f's.{attr_func}()'


def gen_examples(target):

    sf.DisplayActive.set(DC)

    msg = []
    inter = InterfaceSummary.to_frame(target, #type: ignore
            minimized=False,
            max_args=99, # +inf, but keep as int
            )

    for row in inter.loc[inter['group'] == InterfaceGroup.Constructor].iter_series(axis=1):
        calls = series_constructors(row)
        msg.extend(calls_to_msg(calls, row))

    for row in inter.loc[inter['group'] == InterfaceGroup.Attribute].iter_series(axis=1):
        calls = series_attribute(row)
        msg.extend(calls_to_msg(calls, row))

    for row in inter.loc[inter['group'] == InterfaceGroup.DictLike].iter_series(axis=1):
        calls = series_dictlike(row)
        msg.extend(calls_to_msg(calls, row))


    for line in msg:
        print(line)




if __name__ == '__main__':
    gen_examples(sf.Series)
