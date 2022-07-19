from io import StringIO
import typing as tp
import os
import sys

import numpy as np
import static_frame as sf
import pandas as pd

from static_frame.core.interface import InterfaceSummary
from static_frame.core.interface import InterfaceGroup
from static_frame.core.container_util import ContainerMap

dt64 = np.datetime64

SERIES_INIT_A = dict(values=(10, 2, 8), index=('a', 'b', 'c'))
SERIES_INIT_B = dict(values=(4, 3, 12), index=('d', 'e', 'f'))
SERIES_INIT_C = dict(values=(11, 1, None), index=('a', 'b', 'c'))
SERIES_INIT_D = dict(values=(2, 8, 19), index=('b', 'c', 'd'))
SERIES_INIT_E = dict(values=(10.235, 2.124, 8.734), index=('a', 'b', 'c'))
SERIES_INIT_F = dict(values=(False, False, True), index=('a', 'b', 'c'))
SERIES_INIT_G = dict(values=(8, 5, None, 8), index=('a', 'b', 'c', 'd'))
SERIES_INIT_H = dict(values=('q', 'r', '', 's'), index=('a', 'b', 'c', 'd'))
SERIES_INIT_I = dict(values=('', '', 'r', 's'), index=('a', 'b', 'c', 'd'))
SERIES_INIT_J = dict(values=('p', 'q', '', ''), index=('a', 'b', 'c', 'd'))
SERIES_INIT_K = dict(values=(10.235, 2.124, np.nan, 8.734, np.nan),
        index=('a', 'b', 'c', 'd', 'e'))
SERIES_INIT_L = dict(values=(np.nan, np.nan, 10.235, 2.124, 8.734),
        index=('a', 'b', 'c', 'd', 'e'))
SERIES_INIT_M = dict(values=(10.235, 2.124, 8.734, np.nan, np.nan),
        index=('a', 'b', 'c', 'd', 'e'))
SERIES_INIT_N = dict(values=(2, 8, 19, 34, 54), index=('a', 'b', 'c', 'd', 'e'))
SERIES_INIT_O = dict(values=(2, '', 19, 0, None), index=('a', 'b', 'c', 'd', 'e'))
SERIES_INIT_P = dict(values=(8, 5, 0, 8), index=('a', 'b', 'c', 'd'))
SERIES_INIT_Q = dict(values=(8, 5, 0, 8), index=('d', 'b', 'a', 'c'))
SERIES_INIT_R = dict(values=(3, 2, 8, 7),
        index=b"sf.IndexHierarchy.from_product((1, 2), ('a', 'b'))")
SERIES_INIT_S = dict(values=(10, 2, 8), index=('a', 'b', 'c'), name='x')
SERIES_INIT_T = dict(values=(-2, 8, 19, -2, 8), index=('a', 'b', 'c', 'd', 'e'))
SERIES_INIT_U = dict(values=('1517-01-01', '1517-04-01', '1517-12-31', '1517-06-30', '1517-10-01'), index=('a', 'b', 'c', 'd', 'e'), dtype=b'np.datetime64')
SERIES_INIT_V = dict(values=('1/1/1517', '4/1/1517', '6/30/1517'), index=('a', 'b', 'c'))
SERIES_INIT_W = dict(values=('1517-01-01', '1517-04-01', '1517-12-31', '1517-06-30', '1517-10-01'), index=('a', 'b', 'c', 'd', 'e'))
SERIES_INIT_X = dict(values=('qrs ', 'XYZ', '123', ' wX '), index=('a', 'b', 'c', 'd'))
# SERIES_INIT_Y = dict(values=('qrs ', 'XYZ', '123', ' wX '), index=('a', 'b', 'c', 'd'), dtype=b'bytes')
SERIES_INIT_Z = dict(values=(False, False, True), index=('b', 'c', 'd'))

SERIES_INIT_DICT_A = dict(sf.Series(**SERIES_INIT_A))
SERIES_INIT_FROM_ELEMENT_A = dict(element=-1, index=('a', 'b', 'c'), name='x')
SERIES_INIT_FROM_ITEMS_A = dict(pairs=tuple(dict(sf.Series(**SERIES_INIT_A)).items()), name='x')


FRAME_INIT_A = dict(data=b'np.arange(6).reshape(3,2)', index=(('p', 'q', 'r')), columns=(('a', 'b')), name='x')

FRAME_INIT_B = dict(data=b'(np.arange(6).reshape(3,2) % 2).astype(bool)', index=(('p', 'q', 'r')), columns=(('c', 'd')), name='y')

FRAME_INIT_FROM_ELEMENT_A = dict(element=0, index=(('p', 'q', 'r')), columns=(('a', 'b')), name='x')

FRAME_INIT_FROM_ELEMENTS_A = dict(elements=(10, 2, 8, 3), index=(('p', 'q', 'r', 's')),columns=['a'], name='x')


FRAME_INIT_FROM_ELEMENT_ITEMS_A = dict(items=((('a',  0), -1), (('b',  0), 10), (('a',  1), 3), (('b', 'a'), 1)), columns=(0, 1), index= ('a', 'b'), name='x', axis=1)

FRAME_INIT_FROM_DICT_A = dict(mapping=b"dict(a=(10, 2, 8, 3), b=('1517-01-01', '1517-04-01', '1517-12-31', '1517-06-30'))", dtypes=b"dict(b=np.datetime64)", name='x')

FRAME_INIT_FROM_DICT_RECORDS_A = dict(records=b"(dict(a=10, b=False, c='1517-01-01'), dict(a=8, b=True, c='1517-04-01'))", index=('p', 'q'), dtypes=b"dict(c=np.datetime64)", name='x')

FRAME_INIT_FROM_DICT_RECORDS_ITEMS_A = dict(items=b"(('p', dict(a=10, b=False, c='1517-01-01')), ('q', dict(a=8, b=True, c='1517-04-01')))", dtypes=b"dict(c=np.datetime64)", name='x')

FRAME_INIT_FROM_RECORDS_A = dict(records=b"((10, False, '1517-01-01'), (8, True,'1517-04-01'))", index=('p', 'q'), columns=('a', 'b', 'c'), dtypes=b"dict(c=np.datetime64)", name='x')

FRAME_INIT_FROM_RECORDS_ITEMS_A = dict(items=b"(('p', (10, False, '1517-01-01')), ('q', (8, True,'1517-04-01')))", columns=('a', 'b', 'c'), dtypes=b"dict(c=np.datetime64)", name='x')


FRAME_INIT_FROM_FIELDS_A = dict(fields=((10, 2, 8, 3), (False, True, True, False), ('1517-01-01', '1517-04-01', '1517-12-31', '1517-06-30')), columns=('a', 'b', 'c'), dtypes=b"dict(c=np.datetime64)", name='x')
FRAME_INIT_FROM_FIELDS_B = dict(fields=((10, 2, 8, 3), ('qrs ', 'XYZ', '123', ' wX '), ('1517-01-01', '1517-04-01', '1517-12-31', '1517-06-30')), columns=('a', 'b', 'c'), dtypes=b"dict(c=np.datetime64)", name='x')
FRAME_INIT_FROM_FIELDS_C = dict(fields=((10, 2, 8, 3), ('qrs ', 'XYZ', '123', ' wX ')), columns=('a', 'b'), index=('p', 'q', 'r', 's'), name='x')

FRAME_INIT_FROM_ITEMS_A = dict(pairs=(('a', (10, 2, 8, 3)), ('b', ('qrs ', 'XYZ', '123', ' wX '))), index=('p', 'q', 'r', 's'), name='x')
FRAME_INIT_FROM_ITEMS_B = dict(pairs=(('a', (10, 2, np.nan, 3)), ('b', ('qrs ', 'XYZ', None, None))), index=('p', 'q', 'r', 's'), name='x')
FRAME_INIT_FROM_ITEMS_C = dict(pairs=(('a', (8, 3)), ('b', ('123', ' wX '))), index=('r', 's'), name='y')

FRAME_INIT_FROM_JSON_A = dict(json_data='[{"a": 10, "b": false, "c": "1517-01-01"}, {"a": 8, "b": true, "c": "1517-04-01"}]', dtypes=b"dict(c=np.datetime64)", name='x')



def repr_value(v) -> str:
    if isinstance(v, tuple):
        return f"({', '.join(repr_value(x) for x in v)})"
    if v is np.nan:
        # default string repr is not evalable
        return 'np.nan'
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, bytes):
        # use bytes to denote code string that should not be quoted
        return v.decode()
    return str(v)

def kwa(params, arg_first: bool = True):
    arg_only = set()
    if arg_first:
        arg_only.add(0)

    return ', '.join(
        f'{k}={repr_value(v)}' if i not in arg_only else f'{repr_value(v)}'
        for i, (k, v) in enumerate(params.items())
        )

def calls_to_msg(calls: tp.Iterator[str],
        row: sf.Series
        ) -> tp.Iterator[str]:
    cls = ContainerMap.str_to_cls(row['cls_name'])

    g = globals()
    g['sf'] = sf
    g['np'] = np
    g['pd'] = pd
    l = locals()

    i = -1
    for i, call in enumerate(calls):
        # enumerate to pass through empty calls without writing start / end boundaries
        if i == 0:
            yield f'#start_{cls.__name__}-{row["signature_no_args"]}'

        try:
            yield f'>>> {call}'
            post = eval(call, g, l)
            if post is not None:
                yield from str(post).split('\n')
        except SyntaxError:
            exec(call, g, l)
        except (ValueError, RuntimeError, NotImplementedError) as e:
            yield repr(e) # show this error

    if i >= 0:
        yield f'#end_{cls.__name__}-{row["signature_no_args"]}'
        yield ''

#-------------------------------------------------------------------------------
class ExGen:

    SIG_TO_OP_NUMERIC = {
        '__add__()': '+',
        '__eq__()': '==',
        '__floordiv__()': '//',
        '__ge__()': '>=',
        '__gt__()': '>',
        '__le__()': '<=',
        '__lt__()': '<',
        '__mod__()': '%',
        '__mul__()': '*',
        '__ne__()': '!=',
        '__pow__()': '**',
        '__sub__()': '-',
        '__truediv__()': '/',
        '__rfloordiv__()': '//',
        '__radd__()': '+',
        '__rmul__()': '*',
        '__rsub__()': '-',
        '__rtruediv__()': '/',
    }
    SIG_TO_OP_LOGIC = {
        '__and__()': '&',
        '__or__()': '|',
        '__xor__()': '^',
    }
    SIG_TO_OP_MATMUL = {
        '__matmul__()': '@',
        '__rmatmul__()': '@',
    }
    SIG_TO_OP_BIT = {
        '__rshift__()': '>>',
        '__lshift__()': '<<',
    }


    @classmethod
    def group_to_method(cls,
            ig: InterfaceGroup
            ) -> tp.Callable[[sf.Series], tp.Iterator[str]]:
        '''Derive the function name from the group label, then get the function from the cls.
        '''
        attr = str(ig).lower().replace(' ', '_').replace('-', '_')
        return getattr(cls, attr)

    @staticmethod
    def constructor(row: sf.Series) -> tp.Iterator[str]:
        return

    @staticmethod
    def exporter(row: sf.Series) -> tp.Iterator[str]:
        return

    @staticmethod
    def attribute(row: sf.Series,
            name: str,
            ctr_method: str,
            ctr_kwargs: str,
            ) -> tp.Iterator[str]:

        icls = f"sf.{ContainerMap.str_to_cls(row['cls_name']).__name__}" # interface cls
        attr = row['signature_no_args'] # drop paren
        ctr = f"{icls}{'.' if ctr_method else ''}{ctr_method}({kwa(ctr_kwargs)})"

        yield f'{name} = {ctr}'
        yield f'{name}.{attr}'


    @staticmethod
    def method(row: sf.Series) -> tp.Iterator[str]:
        return

    @staticmethod
    def dictionary_like(row: sf.Series,
            name: str,
            ctr_method: str,
            ctr_kwargs: str,
            ) -> tp.Iterator[str]:

        icls = f"sf.{ContainerMap.str_to_cls(row['cls_name']).__name__}" # interface cls
        attr = row['signature_no_args']
        attr_func = row['signature_no_args'][:-2]
        ctr = f"{icls}{'.' if ctr_method else ''}{ctr_method}({kwa(ctr_kwargs)})"

        yield f'{name} = {ctr}'

        if attr == '__contains__()':
            yield f"{name}.{attr_func}('a')"
        elif attr == 'get()':
            yield f"{name}.{attr_func}('a')"
            yield f"{name}.{attr_func}('z', -1)"
        elif attr == 'values':
            yield f"{name}.{attr}"
        elif attr in (
                'items()',
                '__reversed__()',
                '__iter__()'
                ):
            yield f"tuple({name}.{attr_func}())"
        else:
            yield f'{name}.{attr_func}()'

    @staticmethod
    def display(row: sf.Series,
            name: str,
            ctr_method: str,
            ctr_kwargs: str,
            ) -> tp.Iterator[str]:

        icls = f"sf.{ContainerMap.str_to_cls(row['cls_name']).__name__}" # interface cls
        attr = row['signature_no_args']
        attr_func = row['signature_no_args'][:-2]

        ctr = f"{icls}{'.' if ctr_method else ''}{ctr_method}({kwa(ctr_kwargs)})"

        if attr == 'interface':
            yield f'{name} = {ctr}'
            yield f"{name}.{attr}"
        elif attr in (
                'display()',
                'display_tall()',
                'display_wide()',
                ):
            yield f'{name} = {ctr}'
            yield f"{name}.{attr_func}()"
        elif attr == '__repr__()':
            yield f'{name} = {ctr}'
            yield f"repr({name})"
        elif attr == '__str__()':
            yield f'{name} = {ctr}'
            yield f"str({name})"
        else:
            raise NotImplementedError(f'no handling for {attr}')

    @staticmethod
    def assignment(row: sf.Series) -> tp.Iterator[str]:
        return

    @staticmethod
    def selector(row: sf.Series) -> tp.Iterator[str]:
        return

    @staticmethod
    def iterator(row: sf.Series) -> tp.Iterator[str]:
        return

    @classmethod
    def operator_binary(cls, row: sf.Series) -> tp.Iterator[str]:
        return

    @staticmethod
    def operator_unary(row: sf.Series) -> tp.Iterator[str]:
        return

    @staticmethod
    def accessor_datetime(row: sf.Series) -> tp.Iterator[str]:
        return

    @staticmethod
    def accessor_string(row: sf.Series,
            name: str,
            ctr_method: str,
            ctr_kwargs: str,
            ) -> tp.Iterator[str]:

        icls = f"sf.{ContainerMap.str_to_cls(row['cls_name']).__name__}" # interface cls
        attr = row['signature_no_args']
        attr_func = row['signature_no_args'][:-2]

        ctr = f"{icls}{'.' if ctr_method else ''}{ctr_method}({kwa(ctr_kwargs)})"

        if attr == 'via_str.__getitem__()':
            yield f'{name} = {ctr}'
            yield f'{name}'
            yield f'{name}.via_str[-1]'
        elif attr in (
                'via_str.center()',
                'via_str.ljust()',
                'via_str.rjust()',
                'via_str.zfill()',
                ):
            yield f'{name} = {ctr}'
            yield f'{name}'
            yield f'{name}.{attr_func}(8)'
        elif attr in (
                'via_str.count()',
                'via_str.find()',
                'via_str.index()',
                'via_str.partition()',
                'via_str.rpartition()',
                'via_str.rfind()',
                'via_str.rindex()',
                'via_str.rsplit()',
                'via_str.split()',
                'via_str.startswith()',
                ):
            yield f'{name} = {ctr}'
            yield f'{name}'
            yield f"{name}.{attr_func}('X')"
        elif attr == 'via_str.decode()':
            yield f'{name} = {ctr}.astype(bytes)'
            yield f'{name}'
            yield f"{name}.{attr_func}()"
        elif attr == 'via_str.endswith()':
            yield f'{name} = {ctr}'
            yield f'{name}'
            yield f"{name}.{attr_func}(' ')"
        elif attr == 'via_str.replace()':
            yield f'{name} = {ctr}'
            yield f'{name}'
            yield f"{name}.{attr_func}('X', '*')"
        else: # all other simple calls
            yield f'{name} = {ctr}'
            yield f'{name}'
            yield f'{name}.{attr}'

    @staticmethod
    def accessor_transpose(row: sf.Series) -> tp.Iterator[str]:
        return

    @classmethod
    def accessor_fill_value(cls, row: sf.Series) -> tp.Iterator[str]:
        return

    @staticmethod
    def accessor_regular_expression(row: sf.Series,
            name: str,
            ctr_method: str,
            ctr_kwargs: str,
            ) -> tp.Iterator[str]:

        icls = f"sf.{ContainerMap.str_to_cls(row['cls_name']).__name__}" # interface cls
        attr = row['signature_no_args']
        attr_funcs = [x.strip('.') for x in attr.split('()') if x]

        ctr = f"{icls}{'.' if ctr_method else ''}{ctr_method}({kwa(ctr_kwargs)})"

        yield f'{name} = {ctr}'
        yield f'{name}'

        if attr == 'via_re().sub()':
            yield f"{name}.via_re('[X123]').{attr_funcs[1]}('==')"
        elif attr == 'via_re().subn()':
            yield f"{name}.via_re('[X123]').{attr_funcs[1]}('==', 1)"
        elif attr == 'via_re().fullmatch()':
            yield f"{name}.via_re('123').{attr_funcs[1]}()"
        else:
            yield f"{name}.via_re('[X123]').{attr_funcs[1]}()"




class ExGenSeries(ExGen):

    @staticmethod
    def constructor(row: sf.Series) -> tp.Iterator[str]:

        icls = f"sf.{ContainerMap.str_to_cls(row['cls_name']).__name__}" # interface cls
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
            yield f's = {iattr}(dict({kwa(SERIES_INIT_DICT_A, arg_first=False)}))'
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
            yield f'df = pd.Series({kwa(SERIES_INIT_A)})'
            yield f's = {iattr}(df)'
        else:
            raise NotImplementedError(f'no handling for {attr}')
        yield f's'

    @staticmethod
    def exporter(row: sf.Series) -> tp.Iterator[str]:

        cls = ContainerMap.str_to_cls(row['cls_name'])
        icls = f'sf.{cls.__name__}' # interface cls
        attr = row['signature_no_args']
        attr_func = row['signature_no_args'][:-2]

        if attr in ('to_frame()',
                'to_frame_go()',
                'to_frame_he()',
                'to_pairs()',
                'to_pandas()',
                'to_series_he()',
                'to_series()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield f"s.{attr_func}()"
        elif attr in ('to_html()',
                'to_html_datatables()',
                'to_visidata()',
                ):
            pass
        else:
            raise NotImplementedError(f'no handling for {attr}')

    @staticmethod
    def attribute(row: sf.Series) -> tp.Iterator[str]:
        yield from ExGen.attribute(row, 's', '', SERIES_INIT_A)

    @staticmethod
    def method(row: sf.Series) -> tp.Iterator[str]:

        icls = f"sf.{ContainerMap.str_to_cls(row['cls_name']).__name__}" # interface cls
        attr = row['signature_no_args']
        attr_func = row['signature_no_args'][:-2]

        if attr in (
                '__array__()',
                'max()',
                'mean()',
                'median()',
                'min()',
                'prod()',
                'cumprod()',
                'cumsum()',
                'sum()',
                'std()',
                'var()',
                'transpose()',
                 ):
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield f"s.{attr_func}()"

        elif attr == '__array_ufunc__()':
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield f"np.array((0, 1, 0)) * s"
        elif attr == '__bool__()':
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield f"bool(s)"
        elif attr == '__deepcopy__()':
            yield 'import copy'
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield f"copy.deepcopy(s)"
        elif attr == '__len__()':
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield f"len(s)"
        elif attr == '__round__()':
            yield f's = {icls}({kwa(SERIES_INIT_E)})'
            yield 's'
            yield f"round(s, 1)"
        elif attr in (
                'all()',
                'any()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_F)})'
            yield f"s.{attr_func}()"
        elif attr == 'astype()':
            yield f's = {icls}({kwa(SERIES_INIT_C)})'
            yield 's'
            yield f"s.{attr_func}(float)"
        elif attr == 'clip()':
            yield f's = {icls}({kwa(SERIES_INIT_E)})'
            yield 's'
            yield f"s.{attr_func}(lower=2.5, upper=10.1)"
        elif attr == 'count()':
            yield f's = {icls}({kwa(SERIES_INIT_G)})'
            yield f"s.{attr_func}(skipna=True)"
            yield f"s.{attr_func}(unique=True)"

        elif attr in ('cov()',):
            yield f's1 = {icls}({kwa(SERIES_INIT_E)})'
            yield f's2 = {icls}({kwa(SERIES_INIT_A)})'
            yield f"s1.{attr_func}(s2)"
        elif attr in (
                'drop_duplicated()',
                'dropna()',
                'duplicated()',
                'unique()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_G)})'
            yield 's'
            yield f"s.{attr_func}()"

        elif attr == 'dropfalsy()':
            yield f's = {icls}({kwa(SERIES_INIT_H)})'
            yield 's'
            yield f"s.{attr_func}()"

        elif attr == 'equals()':
            yield f's1 = {icls}({kwa(SERIES_INIT_E)})'
            yield f's2 = {icls}({kwa(SERIES_INIT_A)})'
            yield f"s1.{attr_func}(s2)"
        elif attr == 'fillfalsy()':
            yield f's = {icls}({kwa(SERIES_INIT_H)})'
            yield 's'
            yield f"s.{attr_func}('missing')"
        elif attr == 'fillfalsy_backward()':
            yield f's = {icls}({kwa(SERIES_INIT_I)})'
            yield 's'
            yield f"s.{attr_func}()"
        elif attr == 'fillfalsy_forward()':
            yield f's = {icls}({kwa(SERIES_INIT_J)})'
            yield 's'
            yield f"s.{attr_func}()"
        elif attr == 'fillfalsy_leading()':
            yield f's = {icls}({kwa(SERIES_INIT_I)})'
            yield 's'
            yield f"s.{attr_func}('missing')"
        elif attr == 'fillfalsy_trailing()':
            yield f's = {icls}({kwa(SERIES_INIT_J)})'
            yield 's'
            yield f"s.{attr_func}('missing')"
        elif attr == 'fillna()':
            yield f's = {icls}({kwa(SERIES_INIT_K)})'
            yield 's'
            yield f"s.{attr_func}(0.0)"
        elif attr == 'fillna_backward()':
            yield f's = {icls}({kwa(SERIES_INIT_L)})'
            yield 's'
            yield f"s.{attr_func}()"
        elif attr == 'fillna_forward()':
            yield f's = {icls}({kwa(SERIES_INIT_M)})'
            yield 's'
            yield f"s.{attr_func}()"
        elif attr == 'fillna_leading()':
            yield f's = {icls}({kwa(SERIES_INIT_L)})'
            yield 's'
            yield f"s.{attr_func}(0.0)"
        elif attr == 'fillna_trailing()':
            yield f's = {icls}({kwa(SERIES_INIT_M)})'
            yield 's'
            yield f"s.{attr_func}(0.0)"
        elif attr in (
                'head()',
                'tail()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_K)})'
            yield 's'
            yield f"s.{attr_func}(2)"
        elif attr in (
                'iloc_max()',
                'iloc_min()',
                'loc_max()',
                'loc_min()',
                'isna()',
                'notna()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_K)})'
            yield 's'
            yield f"s.{attr_func}()"
        elif attr in (
                'iloc_searchsorted()',
                'loc_searchsorted()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield 's'
            yield f"s.{attr_func}(18)"
        elif attr in ('insert_before()', 'insert_after()'):
            yield f's1 = {icls}({kwa(SERIES_INIT_A)})'
            yield f's2 = {icls}({kwa(SERIES_INIT_B)})'
            yield f"s1.{attr_func}('b', s2)"
        elif attr in (
                'isfalsy()',
                'notfalsy()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_O)})'
            yield 's'
            yield f"s.{attr_func}()"
        elif attr == 'isin()':
            yield f's = {icls}({kwa(SERIES_INIT_O)})'
            yield f"s.{attr_func}((2, 19))"
        elif attr in (
                'rank_dense()',
                'rank_max()',
                'rank_min()',
                'rank_mean()',
                'rank_ordinal()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_P)})'
            yield 's'
            yield f"s.{attr_func}()"
        elif attr in (
                'sort_index()',
                'sort_values()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_P)})'
            yield 's'
            yield f"s.{attr_func}()"
            yield f"s.{attr_func}(ascending=False)"
        elif attr in (
                'shift()',
                'roll()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield 's'
            yield f"s.{attr_func}(2)" # could show fill value for shfit...
        elif attr == 'rehierarch()':
            yield f's = {icls}({kwa(SERIES_INIT_R)})'
            yield 's'
            yield f"s.{attr_func}((1, 0))"
        elif attr == 'reindex()':
            yield f's = {icls}({kwa(SERIES_INIT_P)})'
            yield 's'
            yield f"s.{attr_func}(('d', 'f', 'e', 'c'), fill_value=-1)"
        elif attr == 'relabel()':
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield 's'
            yield f"s.{attr_func}(('x', 'y', 'z'))"
            yield f"s.{attr_func}(dict(a='x', b='y'))"
            yield f"s.{attr_func}(lambda l: f'+{{l.upper()}}+')"
        elif attr == 'relabel_flat()':
            yield f's = {icls}({kwa(SERIES_INIT_R)})'
            yield 's'
            yield f"s.{attr_func}()"
        elif attr == 'relabel_level_add()':
            yield f's = {icls}({kwa(SERIES_INIT_R)})'
            yield 's'
            yield f"s.{attr_func}('x')"
        elif attr == 'relabel_level_drop()':
            yield f's = {icls}({kwa(SERIES_INIT_R)})'
            yield 's'
            yield f"s.iloc[:2].{attr_func}(1)"
        elif attr == 'rename()':
            yield f's = {icls}({kwa(SERIES_INIT_S)})'
            yield 's'
            yield f"s.{attr_func}('y')"
        elif attr == 'sample()':
            yield f's = {icls}({kwa(SERIES_INIT_K)})'
            yield 's'
            yield f"s.{attr_func}(2, seed=0)"
        else:
            raise NotImplementedError(f'no handling for {attr}')

    @staticmethod
    def dictionary_like(row: sf.Series) -> tp.Iterator[str]:
        yield from ExGen.dictionary_like(row, 's', '', SERIES_INIT_A)

    @staticmethod
    def display(row: sf.Series) -> tp.Iterator[str]:
        yield from ExGen.display(row, 's', '', SERIES_INIT_A)

    @staticmethod
    def assignment(row: sf.Series) -> tp.Iterator[str]:

        cls = ContainerMap.str_to_cls(row['cls_name'])
        icls = f'sf.{cls.__name__}' # interface cls
        attr = row['signature_no_args']
        # attr_func = row['signature_no_args'][:-2]

        if attr == 'assign[]()':
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"s.assign['c']('x')"
            yield f"s.assign['c':]('x')"
            yield f"s.assign[['a', 'd']](('x', 'y'))"
        elif attr == 'assign[].apply()':
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield 's'
            yield f"s.assign['c':].apply(lambda s: s / 100)"
        elif attr == 'assign.iloc[]()':
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"s.assign.iloc[2]('x')"
            yield f"s.assign.iloc[2:]('x')"
            yield f"s.assign.iloc[[0, 4]](('x', 'y'))"
        elif attr == 'assign.iloc[].apply()':
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield 's'
            yield f"s.assign.iloc[2:].apply(lambda s: s / 100)"
        elif attr == 'assign.loc[]()':
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"s.assign.loc['c']('x')"
            yield f"s.assign.loc['c':]('x')"
            yield f"s.assign.loc[['a', 'd']](('x', 'y'))"
        elif attr == 'assign.loc[].apply()':
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield 's'
            yield f"s.assign.loc['c':].apply(lambda s: s / 100)"
        else:
            raise NotImplementedError(f'no handling for {attr}')


    @staticmethod
    def selector(row: sf.Series) -> tp.Iterator[str]:

        icls = f"sf.{ContainerMap.str_to_cls(row['cls_name']).__name__}" # interface cls
        attr = row['signature_no_args']
        attr_sel = row['signature_no_args'][:-2]

        if attr in (
                'drop[]',
                'mask[]',
                'masked_array[]',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"s.{attr_sel}['c']"
            yield f"s.{attr_sel}['c':]"
            yield f"s.{attr_sel}[['a', 'd']]"
        elif attr in (
                'drop.iloc[]',
                'mask.iloc[]',
                'masked_array.iloc[]',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"s.{attr_sel}[2]"
            yield f"s.{attr_sel}[2:]"
            yield f"s.{attr_sel}[[0, 4]]"
        elif attr in (
                'drop.loc[]',
                'mask.loc[]',
                'masked_array.loc[]',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"s.{attr_sel}['c']"
            yield f"s.{attr_sel}['c':]"
            yield f"s.{attr_sel}[['a', 'd']]"
        elif attr == '[]':
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"s['c']"
            yield f"s['c':]"
            yield f"s[['a', 'd']]"
        elif attr == '[]':
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"s['c']"
            yield f"s['c':]"
            yield f"s[['a', 'd']]"
        elif attr == 'iloc[]':
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"s.iloc[2]"
            yield f"s.iloc[2:]"
            yield f"s.iloc[[0, 4]]"
        elif attr == 'loc[]':
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"s.loc['c']"
            yield f"s.loc['c':]"
            yield f"s.loc[['a', 'd']]"
        else:
            raise NotImplementedError(f'no handling for {attr}')

    @staticmethod
    def iterator(row: sf.Series) -> tp.Iterator[str]:

        icls = f"sf.{ContainerMap.str_to_cls(row['cls_name']).__name__}" # interface cls
        sig = row['signature_no_args']
        attr = sig
        attr_func = sig[:-2]

        if sig.count('()') == 2:
            # ['iter_element', 'apply']
            attr_funcs = [x.strip('.') for x in sig.split('()') if x]

        if attr in (
                'iter_element()',
                'iter_element_items()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"tuple(s.{attr_func}())"
        elif attr in (
                'iter_element().apply()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"s.{attr_func}(lambda e: e > 10)"
        elif attr in (
                'iter_element_items().apply()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"s.{attr_func}(lambda l, e: e > 10 if l != 'c' else 0)"
        elif attr in (
                'iter_element().apply_iter()',
                'iter_element().apply_iter_items()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"tuple(s.{attr_func}(lambda e: e > 10))"
        elif attr in (
                'iter_element().apply_pool()',
                ):
            yield 'def func(e): return e > 10'
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"s.{attr_func}(func, use_threads=True)"

        elif attr in (
                'iter_element().map_all()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield 's'
            yield f"s.{attr_func}({{2: 200, 10: -1, 8: 45}})"
        elif attr in (
                'iter_element().map_all_iter()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield 's'
            yield f"tuple(s.{attr_func}({{2: 200, 10: -1, 8: 45}}))"
        elif attr in (
                'iter_element().map_all_iter_items()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield 's'
            yield f"tuple(s.{attr_func}({{2: 200, 10: -1, 8: 45}}))"

        elif attr in (
                'iter_element().map_any()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield 's'
            yield f"s.{attr_func}({{10: -1, 8: 45}})"
        elif attr in (
                'iter_element().map_any_iter()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield 's'
            yield f"tuple(s.{attr_func}({{10: -1, 8: 45}}))"
        elif attr in (
                'iter_element().map_any_iter_items()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield 's'
            yield f"tuple(s.{attr_func}({{10: -1, 8: 45}}))"

        elif attr in (
                'iter_element().map_fill()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield 's'
            yield f"s.{attr_func}({{10: -1, 8: 45}}, fill_value=np.nan)"
        elif attr in (
                'iter_element().map_fill_iter()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield 's'
            yield f"tuple(s.{attr_func}({{10: -1, 8: 45}}, fill_value=np.nan))"
        elif attr in (
                'iter_element().map_fill_iter_items()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield 's'
            yield f"tuple(s.{attr_func}({{10: -1, 8: 45}}, fill_value=np.nan))"

        # iter_element_items
        elif attr in (
                'iter_element_items().apply_iter()',
                'iter_element_items().apply_iter_items()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"tuple(s.{attr_func}(lambda l, e: e > 10 and l != 'e'))"
        elif attr in (
                'iter_element_items().apply_pool()',
                ):
            yield "def func(pair): return pair[1] > 10 and pair[0] != 'e'"
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"s.{attr_func}(func, use_threads=True)"


        elif attr in (
                'iter_element_items().map_all()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield 's'
            yield f"s.{attr_func}({{('b', 2): 200, ('a', 10): -1, ('c', 8): 45}})"
        elif attr in (
                'iter_element_items().map_all_iter()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield 's'
            yield f"tuple(s.{attr_func}({{('b', 2): 200, ('a', 10): -1, ('c', 8): 45}}))"
        elif attr in (
                'iter_element_items().map_all_iter_items()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield 's'
            yield f"tuple(s.{attr_func}({{('b', 2): 200, ('a', 10): -1, ('c', 8): 45}}))"

        elif attr in (
                'iter_element_items().map_any()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield 's'
            yield f"s.{attr_func}({{('a', 10): -1, ('c', 8): 45}})"
        elif attr in (
                'iter_element_items().map_any_iter()',
                'iter_element_items().map_any_iter_items()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield 's'
            yield f"tuple(s.{attr_func}({{('a', 10): -1, ('c', 8): 45}}))"
        elif attr in (
                'iter_element_items().map_fill()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield 's'
            yield f"s.{attr_func}({{('a', 10): -1, ('c', 8): 45}}, fill_value=np.nan)"
        elif attr in (
                'iter_element_items().map_fill_iter()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield 's'
            yield f"tuple(s.{attr_func}({{('a', 10): -1, ('c', 8): 45}}, fill_value=np.nan))"
        elif attr in (
                'iter_element_items().map_fill_iter_items()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield 's'
            yield f"tuple(s.{attr_func}({{('a', 10): -1, ('c', 8): 45}}, fill_value=np.nan))"

        elif attr in (
                'iter_group()',
                'iter_group_array()',
                'iter_group_array_items()',
                'iter_group_items()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_T)})'
            yield f"tuple(s.{attr_func}())"
        elif attr in (
                'iter_group().apply()',
                'iter_group_labels().apply()',
                'iter_group_array().apply()',
                'iter_group_labels_array().apply()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_T)})'
            yield f"s.{attr_func}(lambda s: s.sum())"
        elif attr in (
                'iter_group().apply_iter()',
                'iter_group().apply_iter_items()',
                'iter_group_array().apply_iter()',
                'iter_group_array().apply_iter_items()',
                'iter_group_labels().apply_iter()',
                'iter_group_labels().apply_iter_items()',
                'iter_group_labels_array().apply_iter()',
                'iter_group_labels_array().apply_iter_items()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_T)})'
            yield f"tuple(s.{attr_func}(lambda s: s.sum()))"
        elif attr in (
                'iter_group().apply_pool()',
                'iter_group_array().apply_pool()',
                'iter_group_labels().apply_pool()',
                'iter_group_labels_array().apply_pool()',
                ):
            yield "def func(s): return s.sum()"
            yield f's = {icls}({kwa(SERIES_INIT_T)})'
            yield f"s.{attr_func}(func, use_threads=True)"
        elif attr in (
                'iter_group_items().apply_pool()',
                'iter_group_array_items().apply_pool()',
                'iter_group_labels_items().apply_pool()',
                'iter_group_labels_array_items().apply_pool()',
                ):
            # NOTE: check that this is delivering expected results
            yield "def func(pair): return pair[1].sum()"
            yield f's = {icls}({kwa(SERIES_INIT_T)})'
            yield f"s.{attr_func}(func, use_threads=True)"
        elif attr in (
                'iter_group_items().apply()',
                'iter_group_array_items().apply()',
                'iter_group_labels_items().apply()',
                'iter_group_labels_array_items().apply()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_T)})'
            yield f"s.{attr_func}(lambda l, s: s.sum() if l != 8 else s.shape)"
        elif attr in (
                'iter_group_items().apply_iter()',
                'iter_group_items().apply_iter_items()',
                'iter_group_array_items().apply_iter()',
                'iter_group_array_items().apply_iter_items()',
                'iter_group_labels_items().apply_iter()',
                'iter_group_labels_items().apply_iter_items()',
                'iter_group_labels_array_items().apply_iter()',
                'iter_group_labels_array_items().apply_iter_items()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_T)})'
            yield f"tuple(s.{attr_func}(lambda l, s: s.sum() if l != 8 else -1))"
        elif attr in (
                'iter_group_labels()',
                'iter_group_labels_array()',
                'iter_group_labels_items()',
                'iter_group_labels_array_items()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield f"tuple(s.{attr_func}())"
        elif attr in (
                'iter_window()',
                'iter_window_array()',
                'iter_window_array_items()',
                'iter_window_items()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"tuple(s.{attr_func}(size=3, step=1))"
        elif attr in (
                'iter_window().apply()',
                'iter_window_array().apply()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"s.{attr_funcs[0]}(size=3, step=1).{attr_funcs[1]}(lambda s: s.sum())"
        elif attr in (
                'iter_window().apply_iter()',
                'iter_window().apply_iter_items()',
                'iter_window_array().apply_iter()',
                'iter_window_array().apply_iter_items()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"tuple(s.{attr_funcs[0]}(size=3, step=1).{attr_funcs[1]}(lambda s: s.sum()))"
        elif attr in (
                'iter_window_items().apply()',
                'iter_window_array_items().apply()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"s.{attr_funcs[0]}(size=3, step=1).{attr_funcs[1]}(lambda l, s: s.sum() if l != 'd' else -1)"
        elif attr in (
                'iter_window_items().apply_iter()',
                'iter_window_items().apply_iter_items()',
                'iter_window_array_items().apply_iter()',
                'iter_window_array_items().apply_iter_items()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"tuple(s.{attr_funcs[0]}(size=3, step=1).{attr_funcs[1]}(lambda l, s: s.sum() if l != 'd' else -1))"
        elif attr in (
                'iter_window().apply_pool()',
                'iter_window_array().apply_pool()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"s.{attr_funcs[0]}(size=3, step=1).{attr_funcs[1]}(lambda s: s.sum(), use_threads=True)"
        elif attr in (
                'iter_window_items().apply_pool()',
                'iter_window_array_items().apply_pool()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_N)})'
            yield f"s.{attr_funcs[0]}(size=3, step=1).{attr_funcs[1]}(lambda pair: pair[1].sum(), use_threads=True)"
        else:
            raise NotImplementedError(f'no handling for {attr}')


    @classmethod
    def operator_binary(cls, row: sf.Series) -> tp.Iterator[str]:
        icls = f"sf.{ContainerMap.str_to_cls(row['cls_name']).__name__}" # interface cls
        attr = row['signature_no_args']

        if attr in cls.SIG_TO_OP_NUMERIC:
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            if attr.startswith('__r'):
                yield f'8 {cls.SIG_TO_OP_NUMERIC[attr]} s'
                # no need to show reverse on series
            else:
                yield f's {cls.SIG_TO_OP_NUMERIC[attr]} 8'
                yield f"s {cls.SIG_TO_OP_NUMERIC[attr]} s.reindex(('c', 'b'))"
        elif attr in cls.SIG_TO_OP_LOGIC:
            yield f's = {icls}({kwa(SERIES_INIT_F)})'
            yield f"s {cls.SIG_TO_OP_LOGIC[attr]} True"
            yield f"s {cls.SIG_TO_OP_LOGIC[attr]} (True, False, True)"
        elif attr in cls.SIG_TO_OP_MATMUL:
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield f"s {cls.SIG_TO_OP_MATMUL[attr]} (3, 0, 4)"
        elif attr in cls.SIG_TO_OP_BIT:
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield f"s {cls.SIG_TO_OP_BIT[attr]} 1"
        else:
            raise NotImplementedError(f'no handling for {attr}')

    @staticmethod
    def operator_unary(row: sf.Series) -> tp.Iterator[str]:
        icls = f"sf.{ContainerMap.str_to_cls(row['cls_name']).__name__}" # interface cls
        attr = row['signature_no_args']

        sig_to_op = {
            '__neg__()': '-',
            '__pos__()': '+',
        }
        if attr == '__abs__()':
            yield f's = {icls}({kwa(SERIES_INIT_T)})'
            yield f'abs(s)'
        elif attr == '__invert__()':
            yield f's = {icls}({kwa(SERIES_INIT_F)})'
            yield f'~s'
        elif attr in sig_to_op:
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield f"{sig_to_op[attr]}s"
        else:
            raise NotImplementedError(f'no handling for {attr}')

    @staticmethod
    def accessor_datetime(row: sf.Series) -> tp.Iterator[str]:
        icls = f"sf.{ContainerMap.str_to_cls(row['cls_name']).__name__}" # interface cls
        attr = row['signature_no_args']
        attr_func = row['signature_no_args'][:-2]

        if attr == 'via_dt.fromisoformat()':
            yield f's = {icls}({kwa(SERIES_INIT_W)})'
            yield f's.{attr}'
        elif attr == 'via_dt.strftime()':
            yield f's = {icls}({kwa(SERIES_INIT_U)})'
            yield f's.{attr_func}("%A | %B")'
        elif attr in (
                'via_dt.strptime()',
                'via_dt.strpdate()',
                ):
            yield f's = {icls}({kwa(SERIES_INIT_V)})'
            yield f's.{attr_func}("%m/%d/%Y")'
        else:
            yield f's = {icls}({kwa(SERIES_INIT_U)})'
            yield f's.{attr}'

    @staticmethod
    def accessor_string(row: sf.Series) -> tp.Iterator[str]:
        yield from ExGen.accessor_string(row, 's', '', SERIES_INIT_X)

    @classmethod
    def accessor_fill_value(cls, row: sf.Series) -> tp.Iterator[str]:
        icls = f"sf.{ContainerMap.str_to_cls(row['cls_name']).__name__}" # interface cls
        attr = row['signature_no_args']
        # attr_func = row['signature_no_args'][:-2]

        attr_op = attr.replace('via_fill_value().', '')

        if attr_op in cls.SIG_TO_OP_NUMERIC:
            yield f's1 = {icls}({kwa(SERIES_INIT_A)})'
            yield f's2 = {icls}({kwa(SERIES_INIT_D)})'
            if attr_op.startswith('__r'): # NOTE: these raise
                yield f's2 {cls.SIG_TO_OP_NUMERIC[attr_op]} s1.via_fill_value(0)'
            else:
                yield f's1.via_fill_value(0) {cls.SIG_TO_OP_NUMERIC[attr_op]} s2'
        elif attr_op in cls.SIG_TO_OP_LOGIC:
            yield f's1 = {icls}({kwa(SERIES_INIT_F)})'
            yield f's2 = {icls}({kwa(SERIES_INIT_Z)})'
            yield f"s1.via_fill_value(False) {cls.SIG_TO_OP_LOGIC[attr_op]} s2"
        elif attr_op in cls.SIG_TO_OP_MATMUL:
            yield f's1 = {icls}({kwa(SERIES_INIT_A)})'
            yield f's2 = {icls}({kwa(SERIES_INIT_D)})'
            yield f"s1.via_fill_value(1) {cls.SIG_TO_OP_MATMUL[attr_op]} s2"
        elif attr_op in cls.SIG_TO_OP_BIT:
            yield f's1 = {icls}({kwa(SERIES_INIT_A)})'
            yield f's2 = {icls}({kwa(SERIES_INIT_D)})'
            yield f"s1.via_fill_value(0) {cls.SIG_TO_OP_BIT[attr_op]} s2"
        elif attr == 'via_fill_value().loc':
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield f"s.via_fill_value(0).loc[['a', 'c', 'd', 'e']]"
        elif attr == 'via_fill_value().__getitem__()':
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield f"s.via_fill_value(0)[['a', 'c', 'd', 'e']]"
        elif attr == 'via_fill_value().via_T':
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield f's.{attr}'
        else:
            raise NotImplementedError(f'no handling for {attr}')

    @staticmethod
    def accessor_regular_expression(row: sf.Series) -> tp.Iterator[str]:
        yield from ExGen.accessor_regular_expression(row, 's', '', SERIES_INIT_A)



class ExGenFrame(ExGen):

    @staticmethod
    def constructor(row: sf.Series) -> tp.Iterator[str]:

        icls = f"sf.{ContainerMap.str_to_cls(row['cls_name']).__name__}" # interface cls
        attr = row['signature_no_args'][:-2] # drop paren
        iattr = f'{icls}.{attr}'

        if attr == '__init__':
            yield f'{icls}({kwa(FRAME_INIT_A)})'
        elif attr == 'from_arrow':
            yield f'f1 = {icls}({kwa(FRAME_INIT_A)})'
            yield f"aw = f1.to_arrow()"
            yield 'aw'
            yield f"{iattr}(aw, index_depth=1)"
        elif attr == 'from_clipboard':
            if sys.platform != 'darwin':
                yield f'f1 = {icls}({kwa(FRAME_INIT_A)})'
                yield f"f1.to_clipboard()"
                yield f"{iattr}(index_depth=1)"
        elif attr == 'from_concat':
            yield f'f1 = {icls}({kwa(FRAME_INIT_A)})'
            yield f'f2 = {icls}({kwa(FRAME_INIT_B)})'
            yield f'{iattr}((f1, f2), axis=1)'
            yield f"{iattr}((f1, f2.relabel(columns=('a', 'b'))), axis=0, index=sf.IndexAutoFactory)"
        elif attr == 'from_concat_items':
            yield f'f1 = {icls}({kwa(FRAME_INIT_A)})'
            yield f'f2 = {icls}({kwa(FRAME_INIT_B)})'
            yield f"{iattr}(((f1.name, f1), (f2.name, f2)), axis=1)"
            yield f"{iattr}(((f1.name, f1), (f2.name, f2.relabel(columns=('a', 'b')))), axis=0)"
        elif attr == 'from_csv':
            yield f'f1 = {icls}({kwa(FRAME_INIT_A)})'
            yield f"f1.to_csv('/tmp/f.csv')"
            yield "open('/tmp/f.csv').read()"
            yield f"{iattr}('/tmp/f.csv', index_depth=1)"
        elif attr == 'from_delimited':
            yield f'f1 = {icls}({kwa(FRAME_INIT_A)})'
            yield f"f1.to_delimited('/tmp/f.psv', delimiter='|')"
            yield "open('/tmp/f.psv').read()"
            yield f"{iattr}('/tmp/f.psv', delimiter='|', index_depth=1)"
        elif attr == 'from_dict':
            yield f'{iattr}({kwa(FRAME_INIT_FROM_DICT_A, arg_first=False)})'
        elif attr == 'from_dict_records':
            yield f'{iattr}({kwa(FRAME_INIT_FROM_DICT_RECORDS_A, arg_first=False)})'
        elif attr == 'from_dict_records_items':
            yield f'{iattr}({kwa(FRAME_INIT_FROM_DICT_RECORDS_ITEMS_A, arg_first=False)})'
        elif attr == 'from_element':
            yield f'{iattr}({kwa(FRAME_INIT_FROM_ELEMENT_A)})'
        elif attr == 'from_element_items':
            yield f'{iattr}({kwa(FRAME_INIT_FROM_ELEMENT_ITEMS_A)})'
        elif attr == 'from_elements':
            yield f'{iattr}({kwa(FRAME_INIT_FROM_ELEMENTS_A)})'
        elif attr == 'from_fields':
            yield f'{iattr}({kwa(FRAME_INIT_FROM_FIELDS_A)})'

        elif attr == 'from_hdf5':
            yield f'f1 = {icls}.from_fields({kwa(FRAME_INIT_FROM_FIELDS_C)})'
            yield f"f1.to_hdf5('/tmp/f.hdf5')"
            yield f"f1.from_hdf5('/tmp/f.hdf5', label='x', index_depth=1)"

        elif attr == 'from_items':
            yield f'{iattr}({kwa(FRAME_INIT_FROM_ITEMS_A)})'

        elif attr == 'from_json':
            yield f'{iattr}({kwa(FRAME_INIT_FROM_JSON_A)})'

        elif attr == 'from_json_url':
            pass

        elif attr == 'from_msgpack':
            yield f'f1 = {icls}.from_fields({kwa(FRAME_INIT_FROM_FIELDS_C)})'
            yield f'mb = f1.to_msgpack()'
            yield 'mb'
            yield f'{iattr}(mb)'

        elif attr == 'from_npy':
            yield f'f1 = {icls}.from_fields({kwa(FRAME_INIT_FROM_FIELDS_A)})'
            yield f"f1.to_npy('/tmp/f.npy')"
            yield f"{iattr}('/tmp/f.npy')"

        elif attr == 'from_npy_mmap':
            yield f'f1 = {icls}.from_fields({kwa(FRAME_INIT_FROM_FIELDS_A)})'
            yield f"f1.to_npy('/tmp/f.npy')"
            yield f"f2, closer = {iattr}('/tmp/f.npy')"
            yield 'f2'
            yield 'closer() # close mmaps after usage'

        elif attr == 'from_npz':
            yield f'f1 = {icls}.from_fields({kwa(FRAME_INIT_FROM_FIELDS_A)})'
            yield f"f1.to_npz('/tmp/f.npz')"
            yield f"{iattr}('/tmp/f.npz')"

        elif attr == 'from_overlay':
            yield f'f1 = {icls}.from_items({kwa(FRAME_INIT_FROM_ITEMS_B)})'
            yield 'f1'
            yield f'f2 = {icls}.from_items({kwa(FRAME_INIT_FROM_ITEMS_C)})'
            yield 'f2'
            yield f"{iattr}((f1, f2))"

        elif attr == 'from_pandas':
            yield f'f1 = {icls}.from_fields({kwa(FRAME_INIT_FROM_FIELDS_C)})'
            yield f'df = f1.to_pandas()'
            yield 'df'
            yield f'{iattr}(df, dtypes=dict(b=str))'

        elif attr == 'from_parquet':
            yield f'f1 = {icls}.from_fields({kwa(FRAME_INIT_FROM_FIELDS_C)})'
            yield f"f1.to_parquet('/tmp/f.parquet')"
            yield f"{iattr}('/tmp/f.parquet', index_depth=1)"

        elif attr == 'from_pickle':
            yield f'f1 = {icls}.from_fields({kwa(FRAME_INIT_FROM_FIELDS_A)})'
            yield f"f1.to_pickle('/tmp/f.pickle')"
            yield f"{iattr}('/tmp/f.pickle')"

        elif attr == 'from_records':
            yield f'{iattr}({kwa(FRAME_INIT_FROM_RECORDS_A)})'

        elif attr == 'from_records_items':
            yield f'{iattr}({kwa(FRAME_INIT_FROM_RECORDS_ITEMS_A)})'

        elif attr == 'from_series':
            yield f's = sf.Series({kwa(SERIES_INIT_S)})'
            yield f'{iattr}(s)'

        elif attr == 'from_sql':
            yield f'f1 = {icls}.from_fields({kwa(FRAME_INIT_FROM_FIELDS_A)})'
            yield "f1.to_sqlite('/tmp/f.db')"
            yield 'import sqlite3'
            yield "conn = sqlite3.connect('/tmp/f.db')"
            yield f'{iattr}("select * from x limit 2", connection=conn, index_depth=1)'

        elif attr == 'from_sqlite':
            yield f'f1 = {icls}.from_fields({kwa(FRAME_INIT_FROM_FIELDS_A)})'
            yield "f1.to_sqlite('/tmp/f.db')"
            yield f"{iattr}('/tmp/f.db', label=f1.name, index_depth=1)"

        elif attr == 'from_structured_array':
            yield "sa = np.array([(False, 8), (True, 19)], dtype=[('a', bool), ('b', int)])"
            yield 'sa'
            yield f"{iattr}(sa)"

        elif attr == 'from_tsv':
            yield f'f1 = {icls}({kwa(FRAME_INIT_A)})'
            yield f"f1.to_tsv('/tmp/f.tsv')"
            yield "open('/tmp/f.tsv').read()"
            yield f"{iattr}('/tmp/f.tsv', index_depth=1)"

        elif attr == 'from_xlsx':
            yield f'f1 = {icls}({kwa(FRAME_INIT_A)})'
            yield f"f1.to_xlsx('/tmp/f.xlsx')"
            yield f"{iattr}('/tmp/f.xlsx', index_depth=1)"

        else:
            raise NotImplementedError(f'no handling for {attr}')

    # @staticmethod
    # def exporter(row: sf.Series) -> tp.Iterator[str]:

    #     cls = ContainerMap.str_to_cls(row['cls_name'])
    #     icls = f'sf.{cls.__name__}' # interface cls
    #     attr = row['signature_no_args']
    #     attr_func = row['signature_no_args'][:-2]

    #     if attr in ('to_frame()',
    #             'to_frame_go()',
    #             'to_frame_he()',
    #             'to_pairs()',
    #             'to_pandas()',
    #             'to_series_he()',
    #             'to_series()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield f"s.{attr_func}()"
    #     elif attr in ('to_html()',
    #             'to_html_datatables()',
    #             'to_visidata()',
    #             ):
    #         pass
    #     else:
    #         raise NotImplementedError(f'no handling for {attr}')

    @staticmethod
    def attribute(row: sf.Series) -> tp.Iterator[str]:
        yield from ExGen.attribute(row, 'f', 'from_fields', FRAME_INIT_FROM_FIELDS_A)

    # @staticmethod
    # def method(row: sf.Series) -> tp.Iterator[str]:

    #     icls = f"sf.{ContainerMap.str_to_cls(row['cls_name']).__name__}" # interface cls
    #     attr = row['signature_no_args']
    #     attr_func = row['signature_no_args'][:-2]

    #     if attr in (
    #             '__array__()',
    #             'max()',
    #             'mean()',
    #             'median()',
    #             'min()',
    #             'prod()',
    #             'cumprod()',
    #             'cumsum()',
    #             'sum()',
    #             'std()',
    #             'var()',
    #             'transpose()',
    #              ):
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield f"s.{attr_func}()"

    #     elif attr == '__array_ufunc__()':
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield f"np.array((0, 1, 0)) * s"
    #     elif attr == '__bool__()':
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield f"bool(s)"
    #     elif attr == '__deepcopy__()':
    #         yield 'import copy'
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield f"copy.deepcopy(s)"
    #     elif attr == '__len__()':
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield f"len(s)"
    #     elif attr == '__round__()':
    #         yield f's = {icls}({kwa(SERIES_INIT_E)})'
    #         yield 's'
    #         yield f"round(s, 1)"
    #     elif attr in (
    #             'all()',
    #             'any()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_F)})'
    #         yield f"s.{attr_func}()"
    #     elif attr == 'astype()':
    #         yield f's = {icls}({kwa(SERIES_INIT_C)})'
    #         yield 's'
    #         yield f"s.{attr_func}(float)"
    #     elif attr == 'clip()':
    #         yield f's = {icls}({kwa(SERIES_INIT_E)})'
    #         yield 's'
    #         yield f"s.{attr_func}(lower=2.5, upper=10.1)"
    #     elif attr == 'count()':
    #         yield f's = {icls}({kwa(SERIES_INIT_G)})'
    #         yield f"s.{attr_func}(skipna=True)"
    #         yield f"s.{attr_func}(unique=True)"

    #     elif attr in ('cov()',):
    #         yield f's1 = {icls}({kwa(SERIES_INIT_E)})'
    #         yield f's2 = {icls}({kwa(SERIES_INIT_A)})'
    #         yield f"s1.{attr_func}(s2)"
    #     elif attr in (
    #             'drop_duplicated()',
    #             'dropna()',
    #             'duplicated()',
    #             'unique()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_G)})'
    #         yield 's'
    #         yield f"s.{attr_func}()"

    #     elif attr == 'dropfalsy()':
    #         yield f's = {icls}({kwa(SERIES_INIT_H)})'
    #         yield 's'
    #         yield f"s.{attr_func}()"

    #     elif attr == 'equals()':
    #         yield f's1 = {icls}({kwa(SERIES_INIT_E)})'
    #         yield f's2 = {icls}({kwa(SERIES_INIT_A)})'
    #         yield f"s1.{attr_func}(s2)"
    #     elif attr == 'fillfalsy()':
    #         yield f's = {icls}({kwa(SERIES_INIT_H)})'
    #         yield 's'
    #         yield f"s.{attr_func}('missing')"
    #     elif attr == 'fillfalsy_backward()':
    #         yield f's = {icls}({kwa(SERIES_INIT_I)})'
    #         yield 's'
    #         yield f"s.{attr_func}()"
    #     elif attr == 'fillfalsy_forward()':
    #         yield f's = {icls}({kwa(SERIES_INIT_J)})'
    #         yield 's'
    #         yield f"s.{attr_func}()"
    #     elif attr == 'fillfalsy_leading()':
    #         yield f's = {icls}({kwa(SERIES_INIT_I)})'
    #         yield 's'
    #         yield f"s.{attr_func}('missing')"
    #     elif attr == 'fillfalsy_trailing()':
    #         yield f's = {icls}({kwa(SERIES_INIT_J)})'
    #         yield 's'
    #         yield f"s.{attr_func}('missing')"
    #     elif attr == 'fillna()':
    #         yield f's = {icls}({kwa(SERIES_INIT_K)})'
    #         yield 's'
    #         yield f"s.{attr_func}(0.0)"
    #     elif attr == 'fillna_backward()':
    #         yield f's = {icls}({kwa(SERIES_INIT_L)})'
    #         yield 's'
    #         yield f"s.{attr_func}()"
    #     elif attr == 'fillna_forward()':
    #         yield f's = {icls}({kwa(SERIES_INIT_M)})'
    #         yield 's'
    #         yield f"s.{attr_func}()"
    #     elif attr == 'fillna_leading()':
    #         yield f's = {icls}({kwa(SERIES_INIT_L)})'
    #         yield 's'
    #         yield f"s.{attr_func}(0.0)"
    #     elif attr == 'fillna_trailing()':
    #         yield f's = {icls}({kwa(SERIES_INIT_M)})'
    #         yield 's'
    #         yield f"s.{attr_func}(0.0)"
    #     elif attr in (
    #             'head()',
    #             'tail()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_K)})'
    #         yield 's'
    #         yield f"s.{attr_func}(2)"
    #     elif attr in (
    #             'iloc_max()',
    #             'iloc_min()',
    #             'loc_max()',
    #             'loc_min()',
    #             'isna()',
    #             'notna()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_K)})'
    #         yield 's'
    #         yield f"s.{attr_func}()"
    #     elif attr in (
    #             'iloc_searchsorted()',
    #             'loc_searchsorted()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield 's'
    #         yield f"s.{attr_func}(18)"
    #     elif attr in ('insert_before()', 'insert_after()'):
    #         yield f's1 = {icls}({kwa(SERIES_INIT_A)})'
    #         yield f's2 = {icls}({kwa(SERIES_INIT_B)})'
    #         yield f"s1.{attr_func}('b', s2)"
    #     elif attr in (
    #             'isfalsy()',
    #             'notfalsy()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_O)})'
    #         yield 's'
    #         yield f"s.{attr_func}()"
    #     elif attr == 'isin()':
    #         yield f's = {icls}({kwa(SERIES_INIT_O)})'
    #         yield f"s.{attr_func}((2, 19))"
    #     elif attr in (
    #             'rank_dense()',
    #             'rank_max()',
    #             'rank_min()',
    #             'rank_mean()',
    #             'rank_ordinal()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_P)})'
    #         yield 's'
    #         yield f"s.{attr_func}()"
    #     elif attr in (
    #             'sort_index()',
    #             'sort_values()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_P)})'
    #         yield 's'
    #         yield f"s.{attr_func}()"
    #         yield f"s.{attr_func}(ascending=False)"
    #     elif attr in (
    #             'shift()',
    #             'roll()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield 's'
    #         yield f"s.{attr_func}(2)" # could show fill value for shfit...
    #     elif attr == 'rehierarch()':
    #         yield f's = {icls}({kwa(SERIES_INIT_R)})'
    #         yield 's'
    #         yield f"s.{attr_func}((1, 0))"
    #     elif attr == 'reindex()':
    #         yield f's = {icls}({kwa(SERIES_INIT_P)})'
    #         yield 's'
    #         yield f"s.{attr_func}(('d', 'f', 'e', 'c'), fill_value=-1)"
    #     elif attr == 'relabel()':
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield 's'
    #         yield f"s.{attr_func}(('x', 'y', 'z'))"
    #         yield f"s.{attr_func}(dict(a='x', b='y'))"
    #         yield f"s.{attr_func}(lambda l: f'+{{l.upper()}}+')"
    #     elif attr == 'relabel_flat()':
    #         yield f's = {icls}({kwa(SERIES_INIT_R)})'
    #         yield 's'
    #         yield f"s.{attr_func}()"
    #     elif attr == 'relabel_level_add()':
    #         yield f's = {icls}({kwa(SERIES_INIT_R)})'
    #         yield 's'
    #         yield f"s.{attr_func}('x')"
    #     elif attr == 'relabel_level_drop()':
    #         yield f's = {icls}({kwa(SERIES_INIT_R)})'
    #         yield 's'
    #         yield f"s.iloc[:2].{attr_func}(1)"
    #     elif attr == 'rename()':
    #         yield f's = {icls}({kwa(SERIES_INIT_S)})'
    #         yield 's'
    #         yield f"s.{attr_func}('y')"
    #     elif attr == 'sample()':
    #         yield f's = {icls}({kwa(SERIES_INIT_K)})'
    #         yield 's'
    #         yield f"s.{attr_func}(2, seed=0)"
    #     else:
    #         raise NotImplementedError(f'no handling for {attr}')

    @staticmethod
    def dictionary_like(row: sf.Series) -> tp.Iterator[str]:
        yield from ExGen.dictionary_like(row, 'f', 'from_fields', FRAME_INIT_FROM_FIELDS_A)

    @staticmethod
    def display(row: sf.Series) -> tp.Iterator[str]:
        yield from ExGen.display(row, 'f', 'from_fields', FRAME_INIT_FROM_FIELDS_A)


    # @staticmethod
    # def assignment(row: sf.Series) -> tp.Iterator[str]:

    #     cls = ContainerMap.str_to_cls(row['cls_name'])
    #     icls = f'sf.{cls.__name__}' # interface cls
    #     attr = row['signature_no_args']
    #     # attr_func = row['signature_no_args'][:-2]

    #     if attr == 'assign[]()':
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"s.assign['c']('x')"
    #         yield f"s.assign['c':]('x')"
    #         yield f"s.assign[['a', 'd']](('x', 'y'))"
    #     elif attr == 'assign[].apply()':
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield 's'
    #         yield f"s.assign['c':].apply(lambda s: s / 100)"
    #     elif attr == 'assign.iloc[]()':
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"s.assign.iloc[2]('x')"
    #         yield f"s.assign.iloc[2:]('x')"
    #         yield f"s.assign.iloc[[0, 4]](('x', 'y'))"
    #     elif attr == 'assign.iloc[].apply()':
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield 's'
    #         yield f"s.assign.iloc[2:].apply(lambda s: s / 100)"
    #     elif attr == 'assign.loc[]()':
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"s.assign.loc['c']('x')"
    #         yield f"s.assign.loc['c':]('x')"
    #         yield f"s.assign.loc[['a', 'd']](('x', 'y'))"
    #     elif attr == 'assign.loc[].apply()':
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield 's'
    #         yield f"s.assign.loc['c':].apply(lambda s: s / 100)"
    #     else:
    #         raise NotImplementedError(f'no handling for {attr}')


    # @staticmethod
    # def selector(row: sf.Series) -> tp.Iterator[str]:

    #     icls = f"sf.{ContainerMap.str_to_cls(row['cls_name']).__name__}" # interface cls
    #     attr = row['signature_no_args']
    #     attr_sel = row['signature_no_args'][:-2]

    #     if attr in (
    #             'drop[]',
    #             'mask[]',
    #             'masked_array[]',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"s.{attr_sel}['c']"
    #         yield f"s.{attr_sel}['c':]"
    #         yield f"s.{attr_sel}[['a', 'd']]"
    #     elif attr in (
    #             'drop.iloc[]',
    #             'mask.iloc[]',
    #             'masked_array.iloc[]',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"s.{attr_sel}[2]"
    #         yield f"s.{attr_sel}[2:]"
    #         yield f"s.{attr_sel}[[0, 4]]"
    #     elif attr in (
    #             'drop.loc[]',
    #             'mask.loc[]',
    #             'masked_array.loc[]',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"s.{attr_sel}['c']"
    #         yield f"s.{attr_sel}['c':]"
    #         yield f"s.{attr_sel}[['a', 'd']]"
    #     elif attr == '[]':
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"s['c']"
    #         yield f"s['c':]"
    #         yield f"s[['a', 'd']]"
    #     elif attr == '[]':
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"s['c']"
    #         yield f"s['c':]"
    #         yield f"s[['a', 'd']]"
    #     elif attr == 'iloc[]':
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"s.iloc[2]"
    #         yield f"s.iloc[2:]"
    #         yield f"s.iloc[[0, 4]]"
    #     elif attr == 'loc[]':
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"s.loc['c']"
    #         yield f"s.loc['c':]"
    #         yield f"s.loc[['a', 'd']]"
    #     else:
    #         raise NotImplementedError(f'no handling for {attr}')

    # @staticmethod
    # def iterator(row: sf.Series) -> tp.Iterator[str]:

    #     icls = f"sf.{ContainerMap.str_to_cls(row['cls_name']).__name__}" # interface cls
    #     sig = row['signature_no_args']
    #     attr = sig
    #     attr_func = sig[:-2]

    #     if sig.count('()') == 2:
    #         # ['iter_element', 'apply']
    #         attr_funcs = [x.strip('.') for x in sig.split('()') if x]

    #     if attr in (
    #             'iter_element()',
    #             'iter_element_items()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"tuple(s.{attr_func}())"
    #     elif attr in (
    #             'iter_element().apply()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"s.{attr_func}(lambda e: e > 10)"
    #     elif attr in (
    #             'iter_element_items().apply()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"s.{attr_func}(lambda l, e: e > 10 if l != 'c' else 0)"
    #     elif attr in (
    #             'iter_element().apply_iter()',
    #             'iter_element().apply_iter_items()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"tuple(s.{attr_func}(lambda e: e > 10))"
    #     elif attr in (
    #             'iter_element().apply_pool()',
    #             ):
    #         yield 'def func(e): return e > 10'
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"s.{attr_func}(func, use_threads=True)"

    #     elif attr in (
    #             'iter_element().map_all()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield 's'
    #         yield f"s.{attr_func}({{2: 200, 10: -1, 8: 45}})"
    #     elif attr in (
    #             'iter_element().map_all_iter()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield 's'
    #         yield f"tuple(s.{attr_func}({{2: 200, 10: -1, 8: 45}}))"
    #     elif attr in (
    #             'iter_element().map_all_iter_items()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield 's'
    #         yield f"tuple(s.{attr_func}({{2: 200, 10: -1, 8: 45}}))"

    #     elif attr in (
    #             'iter_element().map_any()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield 's'
    #         yield f"s.{attr_func}({{10: -1, 8: 45}})"
    #     elif attr in (
    #             'iter_element().map_any_iter()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield 's'
    #         yield f"tuple(s.{attr_func}({{10: -1, 8: 45}}))"
    #     elif attr in (
    #             'iter_element().map_any_iter_items()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield 's'
    #         yield f"tuple(s.{attr_func}({{10: -1, 8: 45}}))"

    #     elif attr in (
    #             'iter_element().map_fill()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield 's'
    #         yield f"s.{attr_func}({{10: -1, 8: 45}}, fill_value=np.nan)"
    #     elif attr in (
    #             'iter_element().map_fill_iter()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield 's'
    #         yield f"tuple(s.{attr_func}({{10: -1, 8: 45}}, fill_value=np.nan))"
    #     elif attr in (
    #             'iter_element().map_fill_iter_items()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield 's'
    #         yield f"tuple(s.{attr_func}({{10: -1, 8: 45}}, fill_value=np.nan))"

    #     # iter_element_items
    #     elif attr in (
    #             'iter_element_items().apply_iter()',
    #             'iter_element_items().apply_iter_items()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"tuple(s.{attr_func}(lambda l, e: e > 10 and l != 'e'))"
    #     elif attr in (
    #             'iter_element_items().apply_pool()',
    #             ):
    #         yield "def func(pair): return pair[1] > 10 and pair[0] != 'e'"
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"s.{attr_func}(func, use_threads=True)"


    #     elif attr in (
    #             'iter_element_items().map_all()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield 's'
    #         yield f"s.{attr_func}({{('b', 2): 200, ('a', 10): -1, ('c', 8): 45}})"
    #     elif attr in (
    #             'iter_element_items().map_all_iter()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield 's'
    #         yield f"tuple(s.{attr_func}({{('b', 2): 200, ('a', 10): -1, ('c', 8): 45}}))"
    #     elif attr in (
    #             'iter_element_items().map_all_iter_items()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield 's'
    #         yield f"tuple(s.{attr_func}({{('b', 2): 200, ('a', 10): -1, ('c', 8): 45}}))"

    #     elif attr in (
    #             'iter_element_items().map_any()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield 's'
    #         yield f"s.{attr_func}({{('a', 10): -1, ('c', 8): 45}})"
    #     elif attr in (
    #             'iter_element_items().map_any_iter()',
    #             'iter_element_items().map_any_iter_items()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield 's'
    #         yield f"tuple(s.{attr_func}({{('a', 10): -1, ('c', 8): 45}}))"
    #     elif attr in (
    #             'iter_element_items().map_fill()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield 's'
    #         yield f"s.{attr_func}({{('a', 10): -1, ('c', 8): 45}}, fill_value=np.nan)"
    #     elif attr in (
    #             'iter_element_items().map_fill_iter()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield 's'
    #         yield f"tuple(s.{attr_func}({{('a', 10): -1, ('c', 8): 45}}, fill_value=np.nan))"
    #     elif attr in (
    #             'iter_element_items().map_fill_iter_items()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield 's'
    #         yield f"tuple(s.{attr_func}({{('a', 10): -1, ('c', 8): 45}}, fill_value=np.nan))"

    #     elif attr in (
    #             'iter_group()',
    #             'iter_group_array()',
    #             'iter_group_array_items()',
    #             'iter_group_items()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_T)})'
    #         yield f"tuple(s.{attr_func}())"
    #     elif attr in (
    #             'iter_group().apply()',
    #             'iter_group_labels().apply()',
    #             'iter_group_array().apply()',
    #             'iter_group_labels_array().apply()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_T)})'
    #         yield f"s.{attr_func}(lambda s: s.sum())"
    #     elif attr in (
    #             'iter_group().apply_iter()',
    #             'iter_group().apply_iter_items()',
    #             'iter_group_array().apply_iter()',
    #             'iter_group_array().apply_iter_items()',
    #             'iter_group_labels().apply_iter()',
    #             'iter_group_labels().apply_iter_items()',
    #             'iter_group_labels_array().apply_iter()',
    #             'iter_group_labels_array().apply_iter_items()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_T)})'
    #         yield f"tuple(s.{attr_func}(lambda s: s.sum()))"
    #     elif attr in (
    #             'iter_group().apply_pool()',
    #             'iter_group_array().apply_pool()',
    #             'iter_group_labels().apply_pool()',
    #             'iter_group_labels_array().apply_pool()',
    #             ):
    #         yield "def func(s): return s.sum()"
    #         yield f's = {icls}({kwa(SERIES_INIT_T)})'
    #         yield f"s.{attr_func}(func, use_threads=True)"
    #     elif attr in (
    #             'iter_group_items().apply_pool()',
    #             'iter_group_array_items().apply_pool()',
    #             'iter_group_labels_items().apply_pool()',
    #             'iter_group_labels_array_items().apply_pool()',
    #             ):
    #         # NOTE: check that this is delivering expected results
    #         yield "def func(pair): return pair[1].sum()"
    #         yield f's = {icls}({kwa(SERIES_INIT_T)})'
    #         yield f"s.{attr_func}(func, use_threads=True)"
    #     elif attr in (
    #             'iter_group_items().apply()',
    #             'iter_group_array_items().apply()',
    #             'iter_group_labels_items().apply()',
    #             'iter_group_labels_array_items().apply()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_T)})'
    #         yield f"s.{attr_func}(lambda l, s: s.sum() if l != 8 else s.shape)"
    #     elif attr in (
    #             'iter_group_items().apply_iter()',
    #             'iter_group_items().apply_iter_items()',
    #             'iter_group_array_items().apply_iter()',
    #             'iter_group_array_items().apply_iter_items()',
    #             'iter_group_labels_items().apply_iter()',
    #             'iter_group_labels_items().apply_iter_items()',
    #             'iter_group_labels_array_items().apply_iter()',
    #             'iter_group_labels_array_items().apply_iter_items()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_T)})'
    #         yield f"tuple(s.{attr_func}(lambda l, s: s.sum() if l != 8 else -1))"
    #     elif attr in (
    #             'iter_group_labels()',
    #             'iter_group_labels_array()',
    #             'iter_group_labels_items()',
    #             'iter_group_labels_array_items()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield f"tuple(s.{attr_func}())"
    #     elif attr in (
    #             'iter_window()',
    #             'iter_window_array()',
    #             'iter_window_array_items()',
    #             'iter_window_items()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"tuple(s.{attr_func}(size=3, step=1))"
    #     elif attr in (
    #             'iter_window().apply()',
    #             'iter_window_array().apply()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"s.{attr_funcs[0]}(size=3, step=1).{attr_funcs[1]}(lambda s: s.sum())"
    #     elif attr in (
    #             'iter_window().apply_iter()',
    #             'iter_window().apply_iter_items()',
    #             'iter_window_array().apply_iter()',
    #             'iter_window_array().apply_iter_items()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"tuple(s.{attr_funcs[0]}(size=3, step=1).{attr_funcs[1]}(lambda s: s.sum()))"
    #     elif attr in (
    #             'iter_window_items().apply()',
    #             'iter_window_array_items().apply()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"s.{attr_funcs[0]}(size=3, step=1).{attr_funcs[1]}(lambda l, s: s.sum() if l != 'd' else -1)"
    #     elif attr in (
    #             'iter_window_items().apply_iter()',
    #             'iter_window_items().apply_iter_items()',
    #             'iter_window_array_items().apply_iter()',
    #             'iter_window_array_items().apply_iter_items()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"tuple(s.{attr_funcs[0]}(size=3, step=1).{attr_funcs[1]}(lambda l, s: s.sum() if l != 'd' else -1))"
    #     elif attr in (
    #             'iter_window().apply_pool()',
    #             'iter_window_array().apply_pool()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"s.{attr_funcs[0]}(size=3, step=1).{attr_funcs[1]}(lambda s: s.sum(), use_threads=True)"
    #     elif attr in (
    #             'iter_window_items().apply_pool()',
    #             'iter_window_array_items().apply_pool()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_N)})'
    #         yield f"s.{attr_funcs[0]}(size=3, step=1).{attr_funcs[1]}(lambda pair: pair[1].sum(), use_threads=True)"
    #     else:
    #         raise NotImplementedError(f'no handling for {attr}')


    # @classmethod
    # def operator_binary(cls, row: sf.Series) -> tp.Iterator[str]:
    #     icls = f"sf.{ContainerMap.str_to_cls(row['cls_name']).__name__}" # interface cls
    #     attr = row['signature_no_args']

    #     if attr in cls.SIG_TO_OP_NUMERIC:
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         if attr.startswith('__r'):
    #             yield f'8 {cls.SIG_TO_OP_NUMERIC[attr]} s'
    #             # no need to show reverse on series
    #         else:
    #             yield f's {cls.SIG_TO_OP_NUMERIC[attr]} 8'
    #             yield f"s {cls.SIG_TO_OP_NUMERIC[attr]} s.reindex(('c', 'b'))"
    #     elif attr in cls.SIG_TO_OP_LOGIC:
    #         yield f's = {icls}({kwa(SERIES_INIT_F)})'
    #         yield f"s {cls.SIG_TO_OP_LOGIC[attr]} True"
    #         yield f"s {cls.SIG_TO_OP_LOGIC[attr]} (True, False, True)"
    #     elif attr in cls.SIG_TO_OP_MATMUL:
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield f"s {cls.SIG_TO_OP_MATMUL[attr]} (3, 0, 4)"
    #     elif attr in cls.SIG_TO_OP_BIT:
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield f"s {cls.SIG_TO_OP_BIT[attr]} 1"
    #     else:
    #         raise NotImplementedError(f'no handling for {attr}')

    # @staticmethod
    # def operator_unary(row: sf.Series) -> tp.Iterator[str]:
    #     icls = f"sf.{ContainerMap.str_to_cls(row['cls_name']).__name__}" # interface cls
    #     attr = row['signature_no_args']

    #     sig_to_op = {
    #         '__neg__()': '-',
    #         '__pos__()': '+',
    #     }
    #     if attr == '__abs__()':
    #         yield f's = {icls}({kwa(SERIES_INIT_T)})'
    #         yield f'abs(s)'
    #     elif attr == '__invert__()':
    #         yield f's = {icls}({kwa(SERIES_INIT_F)})'
    #         yield f'~s'
    #     elif attr in sig_to_op:
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield f"{sig_to_op[attr]}s"
    #     else:
    #         raise NotImplementedError(f'no handling for {attr}')

    # @staticmethod
    # def accessor_datetime(row: sf.Series) -> tp.Iterator[str]:
    #     icls = f"sf.{ContainerMap.str_to_cls(row['cls_name']).__name__}" # interface cls
    #     attr = row['signature_no_args']
    #     attr_func = row['signature_no_args'][:-2]

    #     if attr == 'via_dt.fromisoformat()':
    #         yield f's = {icls}({kwa(SERIES_INIT_W)})'
    #         yield f's.{attr}'
    #     elif attr == 'via_dt.strftime()':
    #         yield f's = {icls}({kwa(SERIES_INIT_U)})'
    #         yield f's.{attr_func}("%A | %B")'
    #     elif attr in (
    #             'via_dt.strptime()',
    #             'via_dt.strpdate()',
    #             ):
    #         yield f's = {icls}({kwa(SERIES_INIT_V)})'
    #         yield f's.{attr_func}("%m/%d/%Y")'
    #     else:
    #         yield f's = {icls}({kwa(SERIES_INIT_U)})'
    #         yield f's.{attr}'

    @staticmethod
    def accessor_string(row: sf.Series) -> tp.Iterator[str]:
        yield from ExGen.accessor_string(row, 'f', 'from_fields', FRAME_INIT_FROM_FIELDS_C)

    # @staticmethod
    # def accessor_transpose(row: sf.Series) -> tp.Iterator[str]:
    #     cls = ContainerMap.str_to_cls(row['cls_name'])
    #     icls = f'sf.{cls.__name__}' # interface cls
    #     attr = row['signature_no_args']
    #     if attr in ():
    #         yield ''
    #     else:
    #         print('missing', attr)

    # @classmethod
    # def accessor_fill_value(cls, row: sf.Series) -> tp.Iterator[str]:
    #     icls = f"sf.{ContainerMap.str_to_cls(row['cls_name']).__name__}" # interface cls
    #     attr = row['signature_no_args']
    #     # attr_func = row['signature_no_args'][:-2]

    #     attr_op = attr.replace('via_fill_value().', '')

    #     if attr_op in cls.SIG_TO_OP_NUMERIC:
    #         yield f's1 = {icls}({kwa(SERIES_INIT_A)})'
    #         yield f's2 = {icls}({kwa(SERIES_INIT_D)})'
    #         if attr_op.startswith('__r'): # NOTE: these raise
    #             yield f's2 {cls.SIG_TO_OP_NUMERIC[attr_op]} s1.via_fill_value(0)'
    #         else:
    #             yield f's1.via_fill_value(0) {cls.SIG_TO_OP_NUMERIC[attr_op]} s2'
    #     elif attr_op in cls.SIG_TO_OP_LOGIC:
    #         yield f's1 = {icls}({kwa(SERIES_INIT_F)})'
    #         yield f's2 = {icls}({kwa(SERIES_INIT_Z)})'
    #         yield f"s1.via_fill_value(False) {cls.SIG_TO_OP_LOGIC[attr_op]} s2"
    #     elif attr_op in cls.SIG_TO_OP_MATMUL:
    #         yield f's1 = {icls}({kwa(SERIES_INIT_A)})'
    #         yield f's2 = {icls}({kwa(SERIES_INIT_D)})'
    #         yield f"s1.via_fill_value(1) {cls.SIG_TO_OP_MATMUL[attr_op]} s2"
    #     elif attr_op in cls.SIG_TO_OP_BIT:
    #         yield f's1 = {icls}({kwa(SERIES_INIT_A)})'
    #         yield f's2 = {icls}({kwa(SERIES_INIT_D)})'
    #         yield f"s1.via_fill_value(0) {cls.SIG_TO_OP_BIT[attr_op]} s2"
    #     elif attr == 'via_fill_value().loc':
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield f"s.via_fill_value(0).loc[['a', 'c', 'd', 'e']]"
    #     elif attr == 'via_fill_value().__getitem__()':
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield f"s.via_fill_value(0)[['a', 'c', 'd', 'e']]"
    #     elif attr == 'via_fill_value().via_T':
    #         yield f's = {icls}({kwa(SERIES_INIT_A)})'
    #         yield f's.{attr}'
    #     else:
    #         raise NotImplementedError(f'no handling for {attr}')

    @staticmethod
    def accessor_regular_expression(row: sf.Series) -> tp.Iterator[str]:
        yield from ExGen.accessor_regular_expression(row, 'f', 'from_fields', FRAME_INIT_FROM_FIELDS_B)




#-------------------------------------------------------------------------------
def gen_examples(target, exg: ExGen) -> tp.Iterator[str]:

    sf.DisplayActive.set(sf.DisplayConfig(type_color=False))

    inter = InterfaceSummary.to_frame(target, #type: ignore
            minimized=False,
            max_args=99, # +inf, but keep as int
            )

    for ig in (
            InterfaceGroup.Constructor,
            # InterfaceGroup.Exporter,
            # InterfaceGroup.Attribute, # sf
            # InterfaceGroup.Method,
            # InterfaceGroup.DictLike, # sf
            # InterfaceGroup.Display, # sf
            # InterfaceGroup.Assignment,
            # InterfaceGroup.Selector,
            # InterfaceGroup.Iterator,
            # InterfaceGroup.OperatorBinary,
            # InterfaceGroup.OperatorUnary,
            # InterfaceGroup.AccessorDatetime,
            # InterfaceGroup.AccessorString,
            # InterfaceGroup.AccessorTranspose,
            # InterfaceGroup.AccessorFillValue,
            # InterfaceGroup.AccessorRe, # sf

            ):
        func = exg.group_to_method(ig)
        # import ipdb; ipdb.set_trace()
        for row in inter.loc[inter['group'] == ig].iter_series(axis=1):
            calls = func(row)
            yield from calls_to_msg(calls, row)

def gen_all_examples() -> tp.Iterator[str]:
    # yield from gen_examples(sf.Series, ExGenSeries)
    # yield from gen_examples(sf.SeriesHE, ExGenSeries)
    yield from gen_examples(sf.Frame, ExGenFrame)
    # yield from gen_examples(sf.FrameHE, ExGenFrame)
    # yield from gen_examples(sf.FrameGO, ExGenFrame)


def write():
    doc_dir = os.path.abspath(os.path.dirname(__file__))
    fp = os.path.join(doc_dir, 'source', 'examples.txt')

    with open(fp, 'w') as f:
        for line in gen_all_examples():
            f.write(line)
            f.write('\n')


if __name__ == '__main__':
    for line in gen_all_examples():
        print(line)
        pass
    # write()


