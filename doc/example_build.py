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
SERIES_INIT_E = dict(values=(10.235, 2.124, 8.734), index=('a', 'b', 'c'))
SERIES_INIT_F = dict(values=(False, False, True), index=('a', 'b', 'c'))
SERIES_INIT_G = dict(values=(8, 5, None, 8), index=('a', 'b', 'c', 'd'))


SERIES_INIT_DICT_A = dict(sf.Series(**SERIES_INIT_A))
SERIES_INIT_FROM_ELEMENT_A = dict(element=-1, index=('a', 'b', 'c'), name='x')
SERIES_INIT_FROM_ITEMS_A = dict(pairs=tuple(dict(sf.Series(**SERIES_INIT_A)).items()), name='x')


DC = sf.DisplayConfig(type_color=False)

def kwa(params, arg_first: bool = True):
    arg_only = set()
    if arg_first:
        arg_only.add(0)

    return ', '.join(
        f'{k}={repr(v)}' if i not in arg_only else f'{repr(v)}'
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

        if attr in ('__array__()',):
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
        elif attr in ('all()', 'any()'):
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
        elif attr in ('cumprod()', 'cumsum()'):
            yield f's = {icls}({kwa(SERIES_INIT_A)})'
            yield f"s.{attr_func}()"

        else:
            print(f'no handling for {attr}')
            # raise NotImplementedError(f'no handling for {attr}')


# no handling for cumsum()
# no handling for drop_duplicated()
# no handling for dropfalsy()
# no handling for dropna()
# no handling for duplicated()
# no handling for equals()
# no handling for fillfalsy()
# no handling for fillfalsy_backward()
# no handling for fillfalsy_forward()
# no handling for fillfalsy_leading()
# no handling for fillfalsy_trailing()
# no handling for fillna()
# no handling for fillna_backward()
# no handling for fillna_forward()
# no handling for fillna_leading()
# no handling for fillna_trailing()
# no handling for head()
# no handling for iloc_max()
# no handling for iloc_min()
# no handling for iloc_searchsorted()
# no handling for insert_after()
# no handling for insert_before()
# no handling for isfalsy()
# no handling for isin()
# no handling for isna()
# no handling for loc_max()
# no handling for loc_min()
# no handling for loc_searchsorted()
# no handling for max()
# no handling for mean()
# no handling for median()
# no handling for min()
# no handling for notfalsy()
# no handling for notna()
# no handling for prod()
# no handling for rank_dense()
# no handling for rank_max()
# no handling for rank_mean()
# no handling for rank_min()
# no handling for rank_ordinal()
# no handling for rehierarch()
# no handling for reindex()
# no handling for relabel()
# no handling for relabel_flat()
# no handling for relabel_level_add()
# no handling for relabel_level_drop()
# no handling for rename()
# no handling for roll()
# no handling for sample()
# no handling for shift()
# no handling for sort_index()
# no handling for sort_values()
# no handling for std()
# no handling for sum()
# no handling for tail()
# no handling for transpose()
# no handling for unique()
# no handling for var()

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

    sf.DisplayActive.set(DC)

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
