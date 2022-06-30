from io import StringIO
import typing as tp

import numpy as np
import static_frame as sf
import pandas as pd

from static_frame.core.interface import InterfaceSummary
from static_frame.core.interface import InterfaceGroup
from static_frame.core.container_util import ContainerMap




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



SERIES_INIT_DICT_A = dict(sf.Series(**SERIES_INIT_A))
SERIES_INIT_FROM_ELEMENT_A = dict(element=-1, index=('a', 'b', 'c'), name='x')
SERIES_INIT_FROM_ITEMS_A = dict(pairs=tuple(dict(sf.Series(**SERIES_INIT_A)).items()), name='x')


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

    i = -1
    for i, call in enumerate(calls):
        # enumerate to pass through empty calls without writing start / end boundaries
        if i == 0:
            yield f'#start_{cls.__name__}-{row["signature_no_args"]}'

        g = globals()
        g['sf'] = sf
        g['np'] = np
        g['pd'] = pd
        l = locals()

        try:
            yield f'>>> {call}'
            post = eval(call, g, l)
            if post is not None:
                yield from str(post).split('\n')
        except SyntaxError:
            exec(call, g, l)
        except ValueError as e:
            yield repr(e) # show this error

    if i >= 0:
        yield ''
        yield f'#end_{cls.__name__}-{row["signature_no_args"]}'
        yield ''

#-------------------------------------------------------------------------------
class ExGen:

    @classmethod
    def group_to_method(cls,
            ig: InterfaceGroup
            ) -> tp.Callable[[sf.Series], tp.Iterator[str]]:
        attr = str(ig).lower().replace(' ', '_').replace('-', '_')
        return getattr(cls, attr)

    @staticmethod
    def constructor(row: sf.Series) -> tp.Iterator[str]:
        return

    @staticmethod
    def exporter(row: sf.Series) -> tp.Iterator[str]:
        return

    @staticmethod
    def attribute(row: sf.Series) -> tp.Iterator[str]:
        return

    @staticmethod
    def method(row: sf.Series) -> tp.Iterator[str]:
        return

    @staticmethod
    def dictionary_like(row: sf.Series) -> tp.Iterator[str]:
        return

    @staticmethod
    def display(row: sf.Series) -> tp.Iterator[str]:
        return

    @staticmethod
    def assignment(row: sf.Series) -> tp.Iterator[str]:
        return

    @staticmethod
    def selector(row: sf.Series) -> tp.Iterator[str]:
        return

    @staticmethod
    def iterator(row: sf.Series) -> tp.Iterator[str]:
        return

    @staticmethod
    def operator_binary(row: sf.Series) -> tp.Iterator[str]:
        return


class ExGenSeries(ExGen):

    @staticmethod
    def constructor(row: sf.Series) -> tp.Iterator[str]:

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

        cls = ContainerMap.str_to_cls(row['cls_name'])
        icls = f'sf.{cls.__name__}' # interface cls
        attr = row['signature_no_args'] # drop paren

        yield f's = {icls}({kwa(SERIES_INIT_A)})'
        yield f's.{attr}'

    @staticmethod
    def method(row: sf.Series) -> tp.Iterator[str]:

        cls = ContainerMap.str_to_cls(row['cls_name'])
        icls = f'sf.{cls.__name__}' # interface cls
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
        else:
            print(f'no handling for {attr}')
            # raise NotImplementedError(f'no handling for {attr}')

# no handling for relabel_flat()
# no handling for relabel_level_add()
# no handling for relabel_level_drop()
# no handling for rename()
# no handling for sample()

    @staticmethod
    def dictionary_like(row: sf.Series) -> tp.Iterator[str]:

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


#-------------------------------------------------------------------------------
def gen_examples(target, exg: ExGen):

    sf.DisplayActive.set(sf.DisplayConfig(type_color=False))

    msg = []
    inter = InterfaceSummary.to_frame(target, #type: ignore
            minimized=False,
            max_args=99, # +inf, but keep as int
            )

    for ig in (
            # InterfaceGroup.Constructor,
            # InterfaceGroup.Exporter,
            # InterfaceGroup.Attribute,
            InterfaceGroup.Method,
            # InterfaceGroup.DictLike,

            ):
        func = exg.group_to_method(ig)
        for row in inter.loc[inter['group'] == ig].iter_series(axis=1):
            calls = func(row)
            msg.extend(calls_to_msg(calls, row))

    for line in msg:
        print(line)




if __name__ == '__main__':
    gen_examples(sf.Series, ExGenSeries)
