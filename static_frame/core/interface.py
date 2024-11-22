'''
Tools for documenting the SF interface.
'''
from __future__ import annotations

import inspect
from collections import namedtuple
from collections.abc import Mapping
from itertools import chain

import numpy as np
import typing_extensions as tp

from static_frame.core.archive_npy import NPY
from static_frame.core.archive_npy import NPZ
from static_frame.core.batch import Batch
from static_frame.core.bus import Bus
from static_frame.core.container import ContainerBase
from static_frame.core.container import ContainerOperand
from static_frame.core.container import ContainerOperandSequence
from static_frame.core.display import Display
from static_frame.core.display import DisplayActive
from static_frame.core.display_config import DisplayConfig
from static_frame.core.fill_value_auto import FillValueAuto
from static_frame.core.frame import Frame
from static_frame.core.frame import FrameAsType
from static_frame.core.frame import FrameGO
from static_frame.core.frame import FrameHE
from static_frame.core.hloc import HLoc
from static_frame.core.index import ILoc
from static_frame.core.index import Index
from static_frame.core.index import IndexGO
from static_frame.core.index_auto import IndexAutoConstructorFactory
from static_frame.core.index_auto import IndexAutoFactory
from static_frame.core.index_auto import IndexDefaultConstructorFactory
from static_frame.core.index_base import IndexBase
from static_frame.core.index_datetime import IndexDate
from static_frame.core.index_datetime import IndexDateGO
from static_frame.core.index_datetime import IndexHour
from static_frame.core.index_datetime import IndexHourGO
from static_frame.core.index_datetime import IndexMicrosecond
from static_frame.core.index_datetime import IndexMicrosecondGO
from static_frame.core.index_datetime import IndexMillisecond
from static_frame.core.index_datetime import IndexMillisecondGO
from static_frame.core.index_datetime import IndexMinute
from static_frame.core.index_datetime import IndexMinuteGO
from static_frame.core.index_datetime import IndexNanosecond
from static_frame.core.index_datetime import IndexNanosecondGO
from static_frame.core.index_datetime import IndexSecond
from static_frame.core.index_datetime import IndexSecondGO
from static_frame.core.index_datetime import IndexYear
from static_frame.core.index_datetime import IndexYearGO
from static_frame.core.index_datetime import IndexYearMonth
from static_frame.core.index_datetime import IndexYearMonthGO
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.index_hierarchy import IndexHierarchyAsType
from static_frame.core.index_hierarchy import IndexHierarchyGO
from static_frame.core.memory_measure import MemoryDisplay
from static_frame.core.node_dt import InterfaceBatchDatetime
from static_frame.core.node_dt import InterfaceDatetime
from static_frame.core.node_fill_value import InterfaceBatchFillValue
from static_frame.core.node_fill_value import InterfaceFillValue
from static_frame.core.node_hashlib import InterfaceHashlib
from static_frame.core.node_re import InterfaceBatchRe
from static_frame.core.node_re import InterfaceRe
from static_frame.core.node_selector import Interface
from static_frame.core.node_selector import InterfaceAssignQuartet
from static_frame.core.node_selector import InterfaceAssignTrio
from static_frame.core.node_selector import InterfaceBatchAsType
from static_frame.core.node_selector import InterfaceConsolidate
from static_frame.core.node_selector import InterfaceFrameAsType
from static_frame.core.node_selector import InterfaceGetItemBLoc
from static_frame.core.node_selector import InterfaceIndexHierarchyAsType
from static_frame.core.node_selector import InterfaceSelectDuo
from static_frame.core.node_selector import InterfaceSelectTrio
from static_frame.core.node_selector import InterGetItemILoc
from static_frame.core.node_selector import InterGetItemILocCompound
from static_frame.core.node_selector import InterGetItemILocCompoundReduces
from static_frame.core.node_selector import InterGetItemILocReduces
from static_frame.core.node_selector import InterGetItemLoc
from static_frame.core.node_selector import InterGetItemLocCompound
from static_frame.core.node_selector import InterGetItemLocCompoundReduces
from static_frame.core.node_selector import InterGetItemLocReduces
from static_frame.core.node_selector import TVContainer_co
from static_frame.core.node_str import InterfaceBatchString
from static_frame.core.node_str import InterfaceString
from static_frame.core.node_transpose import InterfaceBatchTranspose
from static_frame.core.node_transpose import InterfaceTranspose
from static_frame.core.node_values import InterfaceBatchValues
from static_frame.core.node_values import InterfaceValues
from static_frame.core.platform import Platform
from static_frame.core.quilt import Quilt
from static_frame.core.reduce import InterfaceBatchReduceDispatch
from static_frame.core.reduce import Reduce
from static_frame.core.reduce import ReduceDispatch
from static_frame.core.series import Series
from static_frame.core.series import SeriesHE
from static_frame.core.store_config import StoreConfig
from static_frame.core.store_filter import StoreFilter
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.type_clinic import CallGuard
from static_frame.core.type_clinic import ClinicResult
from static_frame.core.type_clinic import Require
from static_frame.core.type_clinic import TypeClinic
from static_frame.core.util import DT64_S
from static_frame.core.util import EMPTY_ARRAY
from static_frame.core.util import TCallableAny
from static_frame.core.www import WWW
from static_frame.core.yarn import Yarn

#-------------------------------------------------------------------------------

TFrameAny = Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]]

DOCUMENTED_COMPONENTS = (
        Series,
        SeriesHE,
        Frame,
        FrameGO,
        FrameHE,
        Bus,
        Batch,
        Yarn,
        Quilt,
        Index,
        IndexGO,
        IndexHierarchy,
        IndexHierarchyGO,
        IndexYear,
        IndexYearGO,
        IndexYearMonth,
        IndexYearMonthGO,
        IndexDate,
        IndexDateGO,
        IndexMinute,
        IndexMinuteGO,
        IndexHour,
        IndexHourGO,
        IndexSecond,
        IndexSecondGO,
        IndexMillisecond,
        IndexMillisecondGO,
        IndexMicrosecond,
        IndexMicrosecondGO,
        IndexNanosecond,
        IndexNanosecondGO,
        HLoc,
        ILoc,
        TypeClinic,
        CallGuard,
        ClinicResult,
        Require,
        WWW,
        FillValueAuto,
        DisplayActive,
        DisplayConfig,
        StoreConfig,
        StoreFilter,
        IndexAutoFactory,
        IndexDefaultConstructorFactory,
        IndexAutoConstructorFactory,
        NPZ,
        NPY,
        MemoryDisplay,
        Platform,
        )


#-------------------------------------------------------------------------------

UFUNC_UNARY_OPERATORS = frozenset((
        '__pos__',
        '__neg__',
        '__abs__',
        '__invert__',
        ))

UFUNC_BINARY_OPERATORS = frozenset((
        '__add__',
        '__sub__',
        '__mul__',
        '__matmul__',
        '__truediv__',
        '__floordiv__',
        '__mod__',
        #'__divmod__', this returns two np.arrays when called on an np array
        '__pow__',
        '__lshift__',
        '__rshift__',
        '__and__',
        '__xor__',
        '__or__',
        '__lt__',
        '__le__',
        '__eq__',
        '__ne__',
        '__gt__',
        '__ge__',
        ))

RIGHT_OPERATOR_MAP = frozenset((
        '__radd__',
        '__rsub__',
        '__rmul__',
        '__rmatmul__',
        '__rtruediv__',
        '__rfloordiv__',
        ))

# reference attributes for ufunc interfa
UfuncSkipnaAttrs = namedtuple('UfuncSkipnaAttrs', ('ufunc', 'ufunc_skipna'))

UFUNC_AXIS_SKIPNA: tp.Dict[str, UfuncSkipnaAttrs] = {
        # 'all': UfuncSkipnaAttrs(ufunc_all, ufunc_nanall),
        # 'any': UfuncSkipnaAttrs(ufunc_any, ufunc_nanany),
        'sum': UfuncSkipnaAttrs(np.sum, np.nansum),
        'min': UfuncSkipnaAttrs(np.min, np.nanmin),
        'max': UfuncSkipnaAttrs(np.max, np.nanmax),
        'mean': UfuncSkipnaAttrs(np.mean, np.nanmean),
        'median': UfuncSkipnaAttrs(np.median, np.nanmedian),
        'std': UfuncSkipnaAttrs(np.std, np.nanstd),
        'var': UfuncSkipnaAttrs(np.var, np.nanvar),
        'prod': UfuncSkipnaAttrs(np.prod, np.nanprod),
        }

# ufuncs that retain the shape and dimensionality
UFUNC_SHAPE_SKIPNA: tp.Dict[str, UfuncSkipnaAttrs] = {
        'cumsum': UfuncSkipnaAttrs(np.cumsum, np.nancumsum),
        'cumprod': UfuncSkipnaAttrs(np.cumprod, np.nancumprod),
        }


INTERFACE_ATTRIBUTE_CLS = frozenset((
        InterfaceValues,
        InterfaceString,
        InterfaceDatetime,
        InterfaceTranspose,
        InterfaceHashlib,
        # Mapping,
        TypeClinic,

        InterfaceBatchValues,
        InterfaceBatchString,
        InterfaceBatchDatetime,
        InterfaceBatchTranspose,
        ))

#-------------------------------------------------------------------------------
# function inspection utilities

MAX_ARGS = 3
MAX_DOC_CHARS = 80

def _get_parameters(
        func: TCallableAny,
        is_getitem: bool = False,
        max_args: int = MAX_ARGS,
        ) -> str:
    # might need special handling for methods on built-ins
    try:
        sig = inspect.signature(func)
    except ValueError: #pragma: no cover
        # on Python 3.6, this error happens:
        # ValueError: no signature found for builtin <built-in function abs>
        return '[]' if is_getitem else '()' #pragma: no cover

    positional = []
    kwarg_only = ['*'] # preload
    var_positional = ''
    var_keyword = ''

    count = 0
    count_total = 0
    for p in sig.parameters.values():
        if count == 0 and p.name == 'self':
            continue # do not increment counts

        if count < max_args:
            if p.kind == p.KEYWORD_ONLY:
                kwarg_only.append(p.name)
            elif p.kind == p.VAR_POSITIONAL:
                var_positional = p.name
            elif p.kind == p.VAR_KEYWORD:
                var_keyword = p.name
            else:
                positional.append(p.name)
            count += 1
        count_total += 1

    suffix = '' if count >= count_total else f', {Display.ELLIPSIS}'

    # if truthy, update to a proper iterable
    if var_positional:
        var_positional = ('*' + var_positional,) #type: ignore
    if var_keyword:
        var_keyword = ('**' + var_keyword,)  #type: ignore

    if len(kwarg_only) > 1: # do not count the preload
        param_repr = ', '.join(chain(positional, kwarg_only, var_positional, var_keyword))
    else:
        param_repr = ', '.join(chain(positional, var_positional, var_keyword))

    if is_getitem:
        return f'[{param_repr}{suffix}]'
    return f'({param_repr}{suffix})'

def _get_signature_component(
        func: TCallableAny | None,
        name: str,
        max_args: int = MAX_ARGS,
        ) -> tp.Tuple[str, str]:
    '''Return a signature component, either a delegate or terminus. If named, a leading period will be included.
    '''
    if func:
        # sig will just be `()` (maybe with args) at this point
        sig = _get_parameters(func, max_args=max_args)
        if name and name != '__call__':
            # prefix with name if name is not __call__
            sig = f'.{name}{sig}'
            sig_no_args = f'.{name}()'
        else: # assume just function call
            sig_no_args = '()'
    else:
        sig = ''
        sig_no_args = ''
    return sig, sig_no_args

def _get_signatures(
        name: str,
        func: TCallableAny | None,
        *,
        is_getitem: bool = False,
        delegate_name: str = '',
        delegate_func: tp.Optional[TCallableAny] = None,
        delegate_namespace: str = '',
        max_args: int = MAX_ARGS,
        name_no_args: tp.Optional[str] = None,
        terminus_name: str = '',
        terminus_func: tp.Optional[TCallableAny] = None,
        ) -> tp.Tuple[str, str]:
    '''
    Utility to get two versions of ``func`` and ``delegate_func`` signatures

    Args:
        name_no_args: If this signature has a ``delegate_func``, the root name might need to be provided in a version with no arguments (if the root itself is a function).
    '''
    delegate, delegate_no_args = _get_signature_component(delegate_func, delegate_name, max_args=max_args)
    terminus, terminus_no_args = _get_signature_component(terminus_func, terminus_name, max_args=max_args)
    dns = f'.{delegate_namespace}' if delegate_namespace else ''

    name_args = '' if func is None else _get_parameters(func, is_getitem, max_args=max_args)
    signature = f'{name}{name_args}{dns}{delegate}{terminus}'

    name_no_args = name if not name_no_args else name_no_args
    if func is None: # name is a property
        signature_no_args = f'{name_no_args}{dns}{delegate_no_args}{terminus_no_args}'
    elif is_getitem:
        signature_no_args = f'{name_no_args}[]{dns}{delegate_no_args}{terminus_no_args}'
    else:
        signature_no_args = f'{name_no_args}(){dns}{delegate_no_args}{terminus_no_args}'

    return signature, signature_no_args


#-------------------------------------------------------------------------------
class Features:
    '''
    Core utilities need by both Interface and InterfaceSummary
    '''

    GETITEM = '__getitem__'
    CALL = '__call__'

    EXCLUDE_PRIVATE = {
        '__class__',
        '__class_getitem__',
        '__annotations__',
        '__doc__',
        '__del__',
        '__delattr__',
        '__dir__',
        '__dict__',
        '__format__',
        '__getattribute__',
        '__getstate__',
        '__hash__',
        '__init_subclass__',
        '__lshift__',
        '__module__',
        '__new__',
        '__orig_bases__',
        '__parameters__',
        '__setattr__',
        '__setstate__',
        '__setitem__',
        '__slots__',
        '__slotnames__',
        '__subclasshook__',
        '__weakref__',
        '__reduce__',
        '__reduce_ex__',
        '__sizeof__',
        '__firstlineno__',
        '__static_attributes__',
        }

    DICT_LIKE = {
        'get',
        'keys',
        'values',
        'items',
        '__contains__',
        '__iter__',
        '__reversed__'
        }

    DISPLAY = {
        'display',
        'display_tall',
        'display_wide',
        '__repr__',
        '__str__',
        'interface',
        }


    @classmethod
    def scrub_doc(cls,
            doc: tp.Optional[str],
            *,
            max_doc_chars: int = MAX_DOC_CHARS,
            remove_backticks: bool = True,
            ) -> str:
        if not doc:
            return ''
        if remove_backticks:
            doc = doc.replace('`', '')
        doc = doc.replace(':py:meth:', '')
        doc = doc.replace(':obj:', '')
        doc = doc.replace('static_frame.', '')

        # split and join removes contiguous whitespace
        msg = ' '.join(doc.split())
        if len(msg) <= max_doc_chars:
            return msg
        return msg[:max_doc_chars].strip() + Display.ELLIPSIS


#-------------------------------------------------------------------------------

class InterfaceGroup:
    Constructor = 'Constructor'
    Exporter = 'Exporter'
    Attribute = 'Attribute'
    Method = 'Method'
    DictLike = 'Dictionary-Like'
    Display = 'Display'
    Assignment = 'Assignment'
    Selector = 'Selector'
    Iterator = 'Iterator'
    OperatorBinary = 'Operator Binary'
    OperatorUnary = 'Operator Unary'
    AccessorValues = 'Accessor Values'
    AccessorDatetime = 'Accessor Datetime'
    AccessorString = 'Accessor String'
    AccessorTranspose = 'Accessor Transpose'
    AccessorFillValue = 'Accessor Fill Value'
    AccessorRe = 'Accessor Regular Expression'
    AccessorHashlib = 'Accessor Hashlib'
    AccessorTypeClinic = 'Accessor Type Clinic'
    AccessorReduce = 'Accessor Reduce'
    AccessorMapping = 'Accessor Mapping'

# NOTE: order from definition retained
INTERFACE_GROUP_ORDER = tuple(v for k, v in vars(InterfaceGroup).items()
        if not k.startswith('_'))

# NOTE: Used in conf.py to provide interface group documentation on class API TOC pages.
INTERFACE_GROUP_DOC = {
    'Constructor': 'Alternative constructors for creating instances.',
    'Exporter': 'Methods for transforming, exporting, or serializing objects.',
    'Attribute': 'Attributes for retrieving basic characteristics.',
    'Method': 'Methods for general functionality.',
    'Dictionary-Like': 'All dictionary-like methods and iterators.',
    'Display': 'Methods for providing a text representation of the object.',
    'Assignment': 'Interfaces for creating new containers with assignment-like specification.',
    'Selector': 'Interfaces for selecting by position, label or Boolean.',
    'Iterator': 'Interfaces for iterating (and applying functions to) elements, axis, groups, or windows.',
    'Operator Binary': 'Underlying (magic) methods for binary operator implementation.',
    'Operator Unary': 'Underlying (magic) methods for unary operator implementation.',
    'Accessor Values': 'Interface for using NumPy functions on conatainers.',
    'Accessor Datetime': 'Interface for extracting date and datetime characteristics on elements.',
    'Accessor String': 'Interface for employing string methods on container elements.',
    'Accessor Transpose': 'Interface representing a virtual transposition, permiting application of binary operators with Series along columns instead of rows.',
    'Accessor Fill Value': 'Interface that permits supplying a fill value to be used when binary operator application forces reindexing.',
    'Accessor Regular Expression': 'Interface exposing regular expression application on container elements.',
    'Accessor Hashlib': 'Interface exposing cryptographic hashing via hashlib interfaces.',
    'Accessor Type Clinic': 'Interface for providing a type hint from a container or validating a container against a type hint.',
    'Accessor Reduce': 'Interface for providing function application to columns or containers that result in new `Frame`.',
    }

class InterfaceRecord(tp.NamedTuple):

    cls_name: str
    group: str # should be InterfaceGroup
    signature: str
    doc: str
    reference: str = '' # a qualified name as a string for doc gen
    use_signature: bool = False
    is_attr: bool = False
    delegate_reference: str = ''
    delegate_is_attr: bool = False
    signature_no_args: str = ''

    @classmethod
    def gen_from_dict_like(cls, *,
            cls_name: str,
            cls_target: tp.Type[ContainerBase],
            name: str,
            obj: TCallableAny,
            reference: str,
            doc: str,
            max_args: int,
            max_doc_chars: int,
            ) -> tp.Iterator[InterfaceRecord]:
        if name == 'values':
            signature = signature_no_args = name
        else:
            signature, signature_no_args = _get_signatures(
                    name,
                    obj,
                    is_getitem=False,
                    max_args=max_args,
                    )
        yield cls(cls_name,
                InterfaceGroup.DictLike,
                signature,
                doc,
                reference,
                signature_no_args=signature_no_args
                )

    @classmethod
    def gen_from_display(cls, *,
            cls_name: str,
            cls_target: tp.Type[ContainerBase],
            name: str,
            obj: TCallableAny,
            reference: str,
            doc: str,
            max_args: int,
            max_doc_chars: int,
            ) -> tp.Iterator[InterfaceRecord]:
        if name != 'interface':
            # signature = f'{name}()'
            signature, signature_no_args = _get_signatures(
                    name,
                    obj,
                    is_getitem=False,
                    max_args=max_args,
                    )
            yield cls(cls_name,
                    InterfaceGroup.Display,
                    signature,
                    doc,
                    reference,
                    signature_no_args=signature_no_args
                    )
        else: # interface attr
            yield cls(cls_name,
                    InterfaceGroup.Display,
                    name,
                    doc,
                    use_signature=True,
                    signature_no_args=name
                    )

    @classmethod
    def gen_from_astype(cls, *,
            cls_name: str,
            cls_target: tp.Type[ContainerBase],
            name: str,
            obj: tp.Any,
            reference: str,
            doc: str,
            max_args: int,
            max_doc_chars: int,
            ) -> tp.Iterator[InterfaceRecord]:
        if isinstance(obj, (InterfaceFrameAsType,
                InterfaceIndexHierarchyAsType,
                InterfaceBatchAsType,
                )):
            for field in obj._INTERFACE:
                delegate_obj = getattr(obj, field)
                delegate_reference = f'{obj.__class__.__name__}.{field}'
                if field == Features.GETITEM:
                    cls_returned = FrameAsType if isinstance(obj, InterfaceFrameAsType) else IndexHierarchyAsType
                    signature, signature_no_args = _get_signatures(
                            name,
                            delegate_obj,
                            is_getitem=True,
                            delegate_func=cls_returned.__call__,
                            max_args=max_args,
                            )
                else:
                    signature, signature_no_args = _get_signatures(
                            name,
                            delegate_obj,
                            is_getitem=False,
                            max_args=max_args,
                            )
                doc = Features.scrub_doc(
                        getattr(obj.__class__, field).__doc__,
                        max_doc_chars=max_doc_chars,
                        )
                yield cls(cls_name,
                        InterfaceGroup.Method,
                        signature,
                        doc,
                        reference,
                        use_signature=True,
                        is_attr=True,
                        delegate_reference=delegate_reference,
                        signature_no_args=signature_no_args
                        )
        else: # Series, Index, astype is just a method
            signature, signature_no_args = _get_signatures(name, obj, max_args=max_args)
            yield cls(cls_name,
                    InterfaceGroup.Method,
                    signature,
                    doc,
                    reference,
                    signature_no_args=signature_no_args
                    )

    @classmethod
    def gen_from_consolidate(cls, *,
            cls_name: str,
            cls_target: tp.Type[ContainerBase],
            name: str,
            obj: tp.Any,
            reference: str,
            doc: str,
            max_args: int,
            max_doc_chars: int,
            ) -> tp.Iterator[InterfaceRecord]:
        '''Interfaces that are not full selectors or via but define an _INTERFACE component.
        '''
        if isinstance(obj, InterfaceConsolidate):
            for field in obj._INTERFACE:
                doc = Features.scrub_doc(
                        getattr(obj.__class__, field).__doc__,
                        max_doc_chars=max_doc_chars,
                        )
                delegate_reference = f'{obj.__class__.__name__}.{field}'
                delegate_obj = getattr(obj, field)

                if field == 'status': # a property
                    signature = f'{name}.{field}' # manual construct signature
                    yield cls(cls_name,
                            InterfaceGroup.Method,
                            signature,
                            doc,
                            reference,
                            delegate_reference=delegate_reference,
                            delegate_is_attr=True,
                            is_attr=True,
                            signature_no_args=signature,
                            use_signature=True,
                            )
                else:
                    signature, signature_no_args = _get_signatures(
                            name,
                            delegate_obj,
                            is_getitem=field == Features.GETITEM,
                            max_args=max_args,
                            )
                    yield cls(cls_name,
                            InterfaceGroup.Method,
                            signature,
                            doc,
                            reference,
                            delegate_reference=delegate_reference,
                            delegate_is_attr=False,
                            is_attr=True,
                            signature_no_args=signature_no_args,
                            use_signature=True,
                            )
        else:
            # TypeBlocks has a consolidate method
            signature, signature_no_args = _get_signatures(
                    name,
                    obj,
                    is_getitem=False,
                    max_args=max_args,
                    )
            yield cls(cls_name,
                    InterfaceGroup.Method,
                    signature,
                    doc,
                    reference,
                    signature_no_args=signature_no_args
                    )

    @classmethod
    def gen_from_constructor(cls, *,
            cls_name: str,
            cls_target: tp.Type[ContainerBase],
            name: str,
            obj: TCallableAny,
            reference: str,
            doc: str,
            max_args: int,
            max_doc_chars: int,
            ) -> tp.Iterator[InterfaceRecord]:

        signature, signature_no_args = _get_signatures(
                name,
                obj,
                is_getitem=False,
                max_args=max_args,
                )
        yield cls(cls_name,
                InterfaceGroup.Constructor,
                signature,
                doc,
                reference,
                signature_no_args=signature_no_args
                )

    @classmethod
    def gen_from_exporter(cls, *,
            cls_name: str,
            cls_target: tp.Type[ContainerBase],
            name: str,
            obj: TCallableAny,
            reference: str,
            doc: str,
            max_args: int,
            max_doc_chars: int,
            ) -> tp.Iterator[InterfaceRecord]:

        signature, signature_no_args = _get_signatures(
                name,
                obj,
                is_getitem=False,
                max_args=max_args,
                )
        yield cls(cls_name,
                InterfaceGroup.Exporter,
                signature,
                doc,
                reference,
                signature_no_args=signature_no_args
                )

    @classmethod
    def gen_from_iterator(cls, *,
            cls_name: str,
            cls_target: tp.Type[ContainerBase],
            name: str,
            obj: TCallableAny,
            reference: str,
            doc: str,
            max_args: int,
            max_doc_chars: int,
            ) -> tp.Iterator[InterfaceRecord]:

        signature, signature_no_args = _get_signatures(
                name,
                obj.__call__, #type: ignore
                is_getitem=False,
                max_args=max_args,
                )

        yield cls(cls_name,
                InterfaceGroup.Iterator,
                signature,
                doc,
                reference,
                use_signature=True,
                is_attr=True, # doc as attr so sphinx does not add parens to sig
                signature_no_args=signature_no_args,
                )
        # TypeBlocks as iter_* methods that are just functions
        if hasattr(obj, 'CLS_DELEGATE'):
            cls_interface = obj.CLS_DELEGATE
            # IterNodeDelegate or IterNodeDelegateMapable
            for field in cls_interface._INTERFACE: # apply, map, etc
                if field == 'reduce':
                    # need to create an instance of obj in order to get to instance returned from property
                    # delegate_obj = getattr(obj(), field)
                    delegate_obj = ReduceDispatch
                    for field_sub in delegate_obj._INTERFACE:
                        delegate_sub_obj = getattr(delegate_obj, field_sub)
                        doc = Features.scrub_doc(
                                delegate_sub_obj.__doc__,
                                max_doc_chars=max_doc_chars,
                            )
                        terminus_obj = Reduce # cls_interface.CLS_DELEGATE
                        for terminus_name in terminus_obj._INTERFACE:
                            terminus_func = getattr(terminus_obj, terminus_name)
                            signature, signature_no_args = _get_signatures(
                                    name,
                                    obj.__call__, #type: ignore
                                    is_getitem=False,
                                    delegate_func=delegate_sub_obj,
                                    delegate_name=field_sub,
                                    delegate_namespace='reduce',
                                    max_args=max_args,
                                    terminus_name=terminus_name,
                                    terminus_func=terminus_func,
                                    )
                            delegate_reference = f'{delegate_obj.__name__}.{field_sub}'
                            yield cls(cls_name,
                                    InterfaceGroup.Iterator,
                                    signature,
                                    doc,
                                    reference,
                                    use_signature=True,
                                    is_attr=True,
                                    delegate_reference=delegate_reference,
                                    signature_no_args=signature_no_args
                                    )
                else:
                    delegate_obj = getattr(cls_interface, field)
                    delegate_reference = f'{cls_interface.__name__}.{field}'
                    doc = Features.scrub_doc(
                            delegate_obj.__doc__,
                            max_doc_chars=max_doc_chars,
                            )
                    signature, signature_no_args = _get_signatures(
                            name,
                            obj.__call__, #type: ignore
                            is_getitem=False,
                            delegate_func=delegate_obj,
                            delegate_name=field,
                            max_args=max_args,
                            )
                    yield cls(cls_name,
                            InterfaceGroup.Iterator,
                            signature,
                            doc,
                            reference,
                            use_signature=True,
                            is_attr=True,
                            delegate_reference=delegate_reference,
                            signature_no_args=signature_no_args
                            )

    @classmethod
    def gen_from_accessor(cls, *,
            cls_name: str,
            cls_target: tp.Type[ContainerBase],
            name: str,
            obj: TCallableAny,
            reference: str,
            doc: str,
            cls_interface: tp.Type[Interface],
            max_args: int,
            max_doc_chars: int,
            ) -> tp.Iterator[InterfaceRecord]:

        if cls_interface is InterfaceValues or cls_interface is InterfaceBatchValues: # type: ignore[comparison-overlap]
            group = InterfaceGroup.AccessorValues
        elif cls_interface is InterfaceString or cls_interface is InterfaceBatchString: # type: ignore[comparison-overlap]
            group = InterfaceGroup.AccessorString
        elif cls_interface is InterfaceDatetime or cls_interface is InterfaceBatchDatetime: # type: ignore[comparison-overlap]
            group = InterfaceGroup.AccessorDatetime
        elif cls_interface is InterfaceTranspose or cls_interface is InterfaceBatchTranspose: # type: ignore[comparison-overlap]
            group = InterfaceGroup.AccessorTranspose
        elif cls_interface is InterfaceFillValue or cls_interface is InterfaceBatchFillValue: # type: ignore[comparison-overlap]
            group = InterfaceGroup.AccessorFillValue
        elif cls_interface is InterfaceRe or cls_interface is InterfaceBatchRe: # type: ignore[comparison-overlap]
            group = InterfaceGroup.AccessorRe
        elif cls_interface is InterfaceHashlib: # type: ignore[comparison-overlap]
            group = InterfaceGroup.AccessorHashlib
        elif cls_interface is TypeClinic: # type: ignore[comparison-overlap]
            group = InterfaceGroup.AccessorTypeClinic
        elif issubclass(cls_interface, ReduceDispatch) or cls_interface is InterfaceBatchReduceDispatch: # type: ignore[comparison-overlap]
            group = InterfaceGroup.AccessorReduce
        elif issubclass(cls_interface, Mapping):
            group = InterfaceGroup.AccessorMapping
        else:
            raise NotImplementedError(cls_interface) #pragma: no cover

        terminus_name_no_args: tp.Optional[str]

        for field in cls_interface._INTERFACE: # apply, map, etc
            delegate_obj = getattr(cls_interface, field)
            delegate_reference = f'{cls_interface.__name__}.{field}'
            doc = Features.scrub_doc(
                    delegate_obj.__doc__,
                    max_doc_chars=max_doc_chars,
                    )
            if issubclass(cls_interface, ReduceDispatch):
                # NOTE: we do not want to match InterfaceBatchReduceDispatch
                # delegate_obj is ReduceDispatch.from_func, etc
                terminus_obj = cls_interface.CLS_DELEGATE
                for terminus_name in terminus_obj._INTERFACE:
                    terminus_func = getattr(terminus_obj, terminus_name)
                    signature, signature_no_args = _get_signatures(
                            name,
                            None, # force name as a property
                            delegate_func=delegate_obj,
                            delegate_name=field,
                            max_args=max_args,
                            terminus_name=terminus_name,
                            terminus_func=terminus_func,
                            )
                    yield cls(cls_name,
                            group,
                            signature,
                            doc,
                            reference,
                            is_attr=True,
                            use_signature=True,
                            delegate_reference=delegate_reference,
                            delegate_is_attr=False,
                            signature_no_args=signature_no_args
                            )

            else:
                if cls_interface in (InterfaceFillValue, InterfaceRe, InterfaceHashlib):
                    terminus_sig, terminus_sig_no_args = _get_signatures(
                            name,
                            obj,
                            max_args=max_args,
                            )
                    terminus_name = f'{terminus_sig}.{field}'
                    terminus_name_no_args = f'{terminus_sig_no_args}.{field}'
                else:
                    terminus_name = f'{name}.{field}'
                    # NOTE: not certain that that no arg form is always right
                    terminus_name_no_args = f'{name}.{field}'

                if isinstance(delegate_obj, property):
                    # some date tools are properties
                    yield cls(cls_name,
                            group,
                            terminus_name,
                            doc,
                            reference,
                            is_attr=True,
                            use_signature=True,
                            delegate_reference=delegate_reference,
                            delegate_is_attr=True,
                            signature_no_args=terminus_name_no_args
                            )
                else:
                    signature, signature_no_args = _get_signatures(
                            terminus_name,
                            delegate_obj,
                            max_args=max_args,
                            name_no_args=terminus_name_no_args,
                            )
                    yield cls(cls_name,
                            group,
                            signature,
                            doc,
                            reference,
                            is_attr=True,
                            use_signature=True,
                            delegate_reference=delegate_reference,
                            signature_no_args=signature_no_args
                            )

    @classmethod
    def gen_from_getitem(cls, *,
            cls_name: str,
            cls_target: tp.Type[ContainerBase],
            name: str,
            obj: TCallableAny,
            reference: str,
            doc: str,
            max_args: int,
            max_doc_chars: int,
            ) -> tp.Iterator[InterfaceRecord]:
        '''
        For root __getitem__ methods, as well as __getitem__ on InterGetItemLocReduces objects.
        '''
        if name != Features.GETITEM:
            target = obj.__getitem__ #type: ignore
        else:
            target = obj
            name = ''

        signature, signature_no_args = _get_signatures(
                name,
                target,
                is_getitem=True,
                max_args=max_args,
                )

        yield InterfaceRecord(cls_name,
                InterfaceGroup.Selector,
                signature,
                doc,
                reference,
                use_signature=True,
                is_attr=True,
                signature_no_args=signature_no_args
                )


    @classmethod
    def gen_from_selection(cls, *,
            cls_name: str,
            cls_target: tp.Type[ContainerBase],
            name: str,
            obj: TCallableAny,
            reference: str,
            doc: str,
            cls_interface: tp.Type[Interface],
            max_args: int,
            max_doc_chars: int,
            ) -> tp.Iterator[InterfaceRecord]:

        for field in cls_interface._INTERFACE:
            # get from object, not class
            delegate_obj = getattr(obj, field)
            delegate_reference = f'{cls_interface.__name__}.{field}'
            doc = Features.scrub_doc(
                    delegate_obj.__doc__,
                    max_doc_chars=max_doc_chars,
                    )

            if field != Features.GETITEM:
                delegate_is_attr = True
                signature, signature_no_args = _get_signatures(
                        f'{name}.{field}', # make compound interface
                        delegate_obj.__getitem__,
                        is_getitem=True,
                        max_args=max_args,
                        )
            else: # is getitem
                delegate_is_attr = False
                signature, signature_no_args = _get_signatures(
                        name, # on the root, no change necessary
                        delegate_obj,
                        is_getitem=True,
                        max_args=max_args,
                        )

            yield InterfaceRecord(cls_name,
                    InterfaceGroup.Selector,
                    signature,
                    doc,
                    reference,
                    use_signature=True,
                    is_attr=True,
                    delegate_reference=delegate_reference,
                    delegate_is_attr=delegate_is_attr,
                    signature_no_args=signature_no_args
                    )


    @classmethod
    def gen_from_assignment(cls, *,
            cls_name: str,
            cls_target: tp.Type[ContainerBase],
            name: str,
            obj: tp.Union[InterfaceAssignTrio[TVContainer_co],
                    InterfaceAssignQuartet[TVContainer_co]],
            reference: str,
            doc: str,
            cls_interface: tp.Type[Interface],
            max_args: int,
            max_doc_chars: int,
            ) -> tp.Iterator[InterfaceRecord]:

        for field in cls_interface._INTERFACE:

            # get from object, not class
            delegate_obj = getattr(obj, field)
            delegate_reference = f'{cls_interface.__name__}.{field}'
            delegate_doc = Features.scrub_doc(
                    delegate_obj.__doc__,
                    max_doc_chars=max_doc_chars,
                    )

            # will be either SeriesAssign or FrameAssign
            for field_terminus in obj.delegate._INTERFACE:
                terminus_obj = getattr(obj.delegate, field_terminus)
                terminus_reference = f'{obj.delegate.__name__}.{field_terminus}'
                terminus_doc = Features.scrub_doc(
                        terminus_obj.__doc__,
                        max_doc_chars=max_doc_chars,
                        )

                # use the delegate to get the root signature, as the root is just a property that returns an InterfaceAssignTrio or similar
                if field != Features.GETITEM:
                    signature, signature_no_args = _get_signatures(
                            f'{name}.{field}', # make compound interface
                            delegate_obj.__getitem__,
                            is_getitem=True,
                            delegate_func=terminus_obj,
                            delegate_name=field_terminus,
                            max_args=max_args,
                            )
                else: # is getitem
                    signature, signature_no_args = _get_signatures(
                            name, # on the root, no change necessary
                            delegate_obj,
                            is_getitem=True,
                            delegate_func=terminus_obj,
                            delegate_name=field_terminus,
                            max_args=max_args,
                            )

                yield InterfaceRecord(cls_name,
                        InterfaceGroup.Assignment,
                        signature,
                        terminus_doc,
                        reference,
                        use_signature=True,
                        is_attr=True,
                        delegate_reference=terminus_reference,
                        signature_no_args=signature_no_args
                        )

    @classmethod
    def gen_from_method(cls, *,
            cls_name: str,
            cls_target: tp.Type[ContainerBase],
            name: str,
            obj: TCallableAny,
            reference: str,
            doc: str,
            max_args: int,
            max_doc_chars: int,
            ) -> tp.Iterator[InterfaceRecord]:

        signature, signature_no_args = _get_signatures(name, obj, max_args=max_args)

        if name in UFUNC_UNARY_OPERATORS:
            yield InterfaceRecord(cls_name,
                    InterfaceGroup.OperatorUnary,
                    signature,
                    doc,
                    reference,
                    signature_no_args=signature_no_args
                    )
        elif name in UFUNC_BINARY_OPERATORS or name in RIGHT_OPERATOR_MAP:
            # NOTE: as all classes have certain binary operators by default, we need to only show binary operators for ContainerOperand subclasses
            if issubclass(cls_target, ContainerOperandSequence):
                yield InterfaceRecord(cls_name,
                        InterfaceGroup.OperatorBinary,
                        signature,
                        doc,
                        reference,
                        signature_no_args=signature_no_args
                        )
        else:
            yield InterfaceRecord(cls_name,
                    InterfaceGroup.Method,
                    signature,
                    doc,
                    reference,
                    signature_no_args=signature_no_args
                    )


    @classmethod
    def gen_from_class(cls, *,
            cls_name: str,
            cls_target: tp.Type[ContainerBase],
            name: str,
            obj: TCallableAny,
            reference: str,
            doc: str,
            max_args: int,
            max_doc_chars: int,
            ) -> tp.Iterator[InterfaceRecord]:
        '''For classes defined on outer classes.
        '''
        signature, signature_no_args = _get_signatures(name, obj, max_args=max_args)
        yield InterfaceRecord(cls_name,
                InterfaceGroup.Constructor,
                signature,
                doc,
                reference,
                signature_no_args=signature_no_args,
                use_signature=True,
                )

#-------------------------------------------------------------------------------

class InterfaceSummary(Features):

    _CLS_TO_INSTANCE_CACHE: tp.Dict[tp.Type[ContainerBase], ContainerBase] = {}
    _CLS_ONLY = frozenset((
            WWW,
            CallGuard,
            Require,
            ))
    _CLS_INIT_SIMPLE = frozenset((
            ContainerOperandSequence,
            ContainerOperand,
            ContainerBase,
            IndexBase,
            DisplayConfig,
            StoreFilter,
            StoreConfig,
            DisplayActive,
            Platform,
            WWW,
            CallGuard,
            )) | _CLS_ONLY

    _SELECTORS = ('__getitem__', 'iloc', 'loc')

    @classmethod
    def is_public(cls, field: str) -> bool:
        if field.startswith('_') and not field.startswith('__'):
            return False
        if field in cls.EXCLUDE_PRIVATE:
            return False
        return True

    @classmethod
    def get_instance(cls, target: tp.Any) -> ContainerBase:
        '''
        Get a sample instance from any ContainerBase; cache to only create one per life of process.
        '''
        f: TFrameAny
        if target not in cls._CLS_TO_INSTANCE_CACHE:
            if target is TypeBlocks:
                instance = target.from_blocks(np.array((0,)))
            elif target is Bus:
                f = Frame.from_elements((0,), name='frame')
                instance = target.from_frames((f,))
            elif target is Yarn:
                f = Frame.from_elements((0,), name='frame')
                instance = Yarn.from_buses(
                    (Bus.from_frames((f,), name='bus'),),
                    retain_labels=False,
                    )
            elif target is Quilt:
                f = Frame.from_elements((0,), name='frame')
                bus = Bus.from_frames((f,))
                instance = target(bus, retain_labels=False)
            elif target is Batch:
                instance = Batch(iter(()))
            elif target is NPY or target is NPZ:
                instance = target
            elif issubclass(target, IndexHierarchy):
                instance = target.from_labels(((0,0),))
            elif issubclass(target, (IndexYearMonth, IndexYear, IndexDate)):
                instance = target(np.array((0,), dtype=DT64_S))
            elif issubclass(target, Frame):
                instance = target.from_elements((0,))
            elif target in cls._CLS_INIT_SIMPLE:
                instance = target()
            elif target is MemoryDisplay:
                f = Frame(EMPTY_ARRAY)
                instance = target.from_any(f)
            else:
                instance = target((0,))
            cls._CLS_TO_INSTANCE_CACHE[target] = instance
        return cls._CLS_TO_INSTANCE_CACHE[target]

    @classmethod
    def name_obj_iter(cls,
            target: tp.Type[ContainerBase],
            ) -> tp.Iterator[tp.Tuple[str, tp.Any, tp.Any]]:
        instance = cls.get_instance(target=target)

        if hasattr(target.__class__, 'interface'):
            yield 'interface', None, ContainerBase.__class__.interface #type: ignore

        # force these to be ordered at the bottom
        selectors_found = set()

        for name_attr in sorted(dir(target)):
            if name_attr == 'interface':
                continue # skip, provided by metaclass
            if not cls.is_public(name_attr):
                continue
            if target in cls._CLS_ONLY and (
                    name_attr == '__init__' or name_attr in Features.DISPLAY):
                continue

            if name_attr in cls._SELECTORS:
                selectors_found.add(name_attr)
                continue
            try:
                yield name_attr, getattr(instance, name_attr), getattr(target, name_attr)
            except NotImplementedError: # base class properties that are not implemented
                pass

        for name_attr in cls._SELECTORS:
            if name_attr in selectors_found:
                yield name_attr, getattr(instance, name_attr), getattr(target, name_attr)


    #---------------------------------------------------------------------------
    @classmethod
    def interrogate(cls,
            target: tp.Type[ContainerBase],
            *,
            max_args: int = MAX_ARGS,
            max_doc_chars: int = MAX_DOC_CHARS,
            ) -> tp.Iterator[InterfaceRecord]:
        for name_attr, obj, obj_cls in cls.name_obj_iter(target):
            doc = ''

            if isinstance(obj_cls, property):
                doc = cls.scrub_doc(obj_cls.__doc__,
                        max_doc_chars=max_doc_chars,
                        )
            elif hasattr(obj, '__doc__'):
                doc = cls.scrub_doc(obj.__doc__,
                        max_doc_chars=max_doc_chars,
                        )

            if name_attr == 'values':
                name = name_attr # on Batch this is generator that has an generic name
            elif hasattr(obj, '__name__'):
                name = obj.__name__
            else: # some attributes yield objects like arrays, Series, or Frame
                name = name_attr

            cls_name = target.__name__
            reference = f'{cls_name}.{name}' # check if this is still necessary

            kwargs = dict(
                    cls_name=cls_name,
                    cls_target=target,
                    name=name,
                    obj=obj,
                    reference=reference,
                    doc=doc,
                    max_args=max_args,
                    max_doc_chars=max_doc_chars,
                    )

            if name in cls.DICT_LIKE:
                yield from InterfaceRecord.gen_from_dict_like(**kwargs) # pyright: ignore
            elif name in cls.DISPLAY:
                yield from InterfaceRecord.gen_from_display(**kwargs) # pyright: ignore
            elif name == 'astype':
                yield from InterfaceRecord.gen_from_astype(**kwargs) # pyright: ignore
            elif name == 'consolidate':
                yield from InterfaceRecord.gen_from_consolidate(**kwargs) # pyright: ignore
            elif callable(obj) and name.startswith('from_') or name == '__init__':
                yield from InterfaceRecord.gen_from_constructor(**kwargs) # pyright: ignore
            elif callable(obj) and name.startswith('to_'):
                yield from InterfaceRecord.gen_from_exporter(**kwargs) # pyright: ignore
            elif name.startswith('iter_'):
                yield from InterfaceRecord.gen_from_iterator(**kwargs) # pyright: ignore
            elif isinstance(obj, (
                    InterGetItemLoc,
                    InterGetItemLocReduces,
                    InterGetItemLocCompound,
                    InterGetItemLocCompoundReduces,
                    InterGetItemILoc,
                    InterGetItemILocReduces,
                    InterGetItemILocCompound,
                    InterGetItemILocCompoundReduces,
                    InterfaceGetItemBLoc,
                    )) or name == cls.GETITEM:
                yield from InterfaceRecord.gen_from_getitem(**kwargs) # pyright: ignore

            elif obj.__class__ in INTERFACE_ATTRIBUTE_CLS:
                yield from InterfaceRecord.gen_from_accessor(
                        cls_interface=obj.__class__,
                        **kwargs, # pyright: ignore
                        )
            elif obj.__class__ in (InterfaceSelectDuo, InterfaceSelectTrio):
                yield from InterfaceRecord.gen_from_selection(
                        cls_interface=obj.__class__,
                        **kwargs, # pyright: ignore
                        )
            elif obj.__class__ in (InterfaceAssignTrio, InterfaceAssignQuartet):
                yield from InterfaceRecord.gen_from_assignment(
                        cls_interface=obj.__class__,
                        **kwargs, # pyright: ignore
                        )
            # as InterfaceFillValue, InterfaceRe are methods, must match on name, not INTERFACE_ATTRIBUTE_CLS
            elif name == 'via_fill_value':
                yield from InterfaceRecord.gen_from_accessor(
                        cls_interface=InterfaceFillValue,
                        **kwargs, # pyright: ignore
                        )
            elif name == 'via_re':
                yield from InterfaceRecord.gen_from_accessor(
                        cls_interface=InterfaceRe,
                        **kwargs, # pyright: ignore
                        )
            elif name == 'via_mapping':
                yield from InterfaceRecord.gen_from_accessor(
                        cls_interface=obj.__class__,
                        **kwargs, # pyright: ignore
                        )
            elif name == 'reduce': # subclasses not in INTERFACE_ATTRIBUTE_CLS
                yield from InterfaceRecord.gen_from_accessor(
                        cls_interface=obj.__class__,
                        **kwargs, # pyright: ignore
                        )
            elif callable(obj):
                if obj.__class__ == type: # a class defined on this class
                    yield from InterfaceRecord.gen_from_class(**kwargs) # pyright: ignore
                else: # general methods
                    yield from InterfaceRecord.gen_from_method(**kwargs) # pyright: ignore
            else:
                yield InterfaceRecord(cls_name,
                        InterfaceGroup.Attribute,
                        name,
                        doc,
                        reference,
                        signature_no_args=name
                        )

    @classmethod
    def to_frame(cls,
            target: tp.Type[ContainerBase],
            *,
            minimized: bool = True,
            max_args: int = MAX_ARGS,
            max_doc_chars: int = MAX_DOC_CHARS,
            ) -> TFrameAny:
        '''
        Reduce to key fields.
        '''
        f: TFrameAny = Frame.from_records(
                cls.interrogate(target,
                        max_args=max_args,
                        max_doc_chars=max_doc_chars,
                        ),
                )
        # order be group order
        f = Frame.from_concat(
                (f.loc[f['group'] == g] for g in INTERFACE_GROUP_ORDER),
                name=target.__name__
                )
        f = f.set_index('signature', drop=True)

        # derive Sphinx-RST compatible (case insensitive) label that handles single character case-senstive attrs in FillValueAuto
        sna_label = f['signature_no_args'].iter_element().apply(
                lambda e: e if len(e) > 1 else f'{e}_' if e.isupper() else e
                ).via_str.lower().rename('sna_label')

        assert len(sna_label.unique()) == len(f)
        f = Frame.from_concat((f, sna_label), axis=1)

        if minimized:
            return f[['cls_name', 'group', 'doc']]
        return f


