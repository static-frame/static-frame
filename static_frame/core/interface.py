"""
Tools for documenting the SF interface.
"""

from __future__ import annotations

import inspect
from collections import defaultdict, namedtuple
from collections.abc import Mapping
from itertools import chain

import numpy as np
import typing_extensions as tp

from static_frame.core.archive_npy import NPY, NPZ
from static_frame.core.batch import Batch
from static_frame.core.bus import Bus
from static_frame.core.container import (
    ContainerBase,
    ContainerOperand,
    ContainerOperandSequence,
)
from static_frame.core.display import Display, DisplayActive
from static_frame.core.display_config import DisplayConfig
from static_frame.core.fill_value_auto import FillValueAuto
from static_frame.core.frame import Frame, FrameAsType, FrameGO, FrameHE
from static_frame.core.hloc import HLoc
from static_frame.core.index import ILoc, Index, IndexGO
from static_frame.core.index_auto import (
    IndexAutoConstructorFactory,
    IndexAutoFactory,
    IndexDefaultConstructorFactory,
)
from static_frame.core.index_base import IndexBase
from static_frame.core.index_datetime import (
    IndexDate,
    IndexDateGO,
    IndexHour,
    IndexHourGO,
    IndexMicrosecond,
    IndexMicrosecondGO,
    IndexMillisecond,
    IndexMillisecondGO,
    IndexMinute,
    IndexMinuteGO,
    IndexNanosecond,
    IndexNanosecondGO,
    IndexSecond,
    IndexSecondGO,
    IndexYear,
    IndexYearGO,
    IndexYearMonth,
    IndexYearMonthGO,
)
from static_frame.core.index_hierarchy import (
    IndexHierarchy,
    IndexHierarchyAsType,
    IndexHierarchyGO,
)
from static_frame.core.memory_measure import MemoryDisplay
from static_frame.core.node_dt import InterfaceBatchDatetime, InterfaceDatetime
from static_frame.core.node_fill_value import InterfaceBatchFillValue, InterfaceFillValue
from static_frame.core.node_hashlib import InterfaceHashlib
from static_frame.core.node_re import InterfaceBatchRe, InterfaceRe
from static_frame.core.node_selector import (
    Interface,
    InterfaceAssignQuartet,
    InterfaceAssignTrio,
    InterfaceBatchAsType,
    InterfaceConsolidate,
    InterfaceFrameAsType,
    InterfaceGetItemBLoc,
    InterfaceIndexHierarchyAsType,
    InterfacePersist,
    InterfaceSelectDuo,
    InterfaceSelectTrio,
    InterGetItemILoc,
    InterGetItemILocCompound,
    InterGetItemILocCompoundReduces,
    InterGetItemILocInPlace,
    InterGetItemILocReduces,
    InterGetItemLoc,
    InterGetItemLocCompound,
    InterGetItemLocCompoundReduces,
    InterGetItemLocInPlace,
    InterGetItemLocReduces,
    TVContainer_co,
)
from static_frame.core.node_str import InterfaceBatchString, InterfaceString
from static_frame.core.node_transpose import InterfaceBatchTranspose, InterfaceTranspose
from static_frame.core.node_values import InterfaceBatchValues, InterfaceValues
from static_frame.core.platform import Platform
from static_frame.core.quilt import Quilt
from static_frame.core.reduce import InterfaceBatchReduceDispatch, Reduce, ReduceDispatch
from static_frame.core.series import Series, SeriesHE
from static_frame.core.store_config import StoreConfig
from static_frame.core.store_filter import StoreFilter
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.type_clinic import CallGuard, ClinicResult, Require, TypeClinic
from static_frame.core.util import DT64_S, EMPTY_ARRAY, TCallableAny
from static_frame.core.www import WWW
from static_frame.core.yarn import Yarn

# -------------------------------------------------------------------------------

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


# -------------------------------------------------------------------------------

UFUNC_UNARY_OPERATORS = frozenset(
    (
        'abs',
        '__pos__',
        '__neg__',
        '__abs__',
        '__invert__',
    )
)

UFUNC_BINARY_OPERATORS = frozenset(
    (
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
    )
)

RIGHT_OPERATOR_MAP = frozenset(
    (
        '__radd__',
        '__rsub__',
        '__rmul__',
        '__rmatmul__',
        '__rtruediv__',
        '__rfloordiv__',
    )
)

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


INTERFACE_ATTRIBUTE_CLS = frozenset(
    (
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
    )
)

# -------------------------------------------------------------------------------
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
    except ValueError:  # pragma: no cover
        # on Python 3.6, this error happens:
        # ValueError: no signature found for builtin <built-in function abs>
        return '[]' if is_getitem else '()'  # pragma: no cover
    pos_only = []
    pos_or_kwarg = []
    kwarg_only = ['*']  # preload

    # these only ever have one
    var_args = ''
    var_kwargs = ''

    count = 0
    count_total = 0
    for p in sig.parameters.values():
        if count == 0 and p.name == 'self':
            continue  # do not increment counts

        if count < max_args:
            if p.kind == p.POSITIONAL_ONLY:
                pos_only.append(p.name)
            elif p.kind == p.POSITIONAL_OR_KEYWORD:
                pos_or_kwarg.append(p.name)
            elif p.kind == p.KEYWORD_ONLY:
                kwarg_only.append(p.name)
            elif p.kind == p.VAR_POSITIONAL:
                var_args = p.name
            elif p.kind == p.VAR_KEYWORD:
                var_kwargs = p.name
            else:
                raise RuntimeError(f'unknown parameter kind {p}')  # pragma: no cover
            count += 1
        count_total += 1

    suffix = '' if count >= count_total else f', {Display.ELLIPSIS}'

    # if truthy, update to a proper iterable
    if var_args:
        var_args = ('*' + var_args,)  # type: ignore
    if var_kwargs:
        var_kwargs = ('**' + var_kwargs,)  # type: ignore

    if pos_only:
        pos_only.append('/')

    if len(kwarg_only) > 1:  # do not count the preload
        param_repr = ', '.join(
            chain(pos_only, pos_or_kwarg, kwarg_only, var_args, var_kwargs)
        )
    else:
        param_repr = ', '.join(chain(pos_only, pos_or_kwarg, var_args, var_kwargs))

    if is_getitem:
        return f'[{param_repr}{suffix}]'
    return f'({param_repr}{suffix})'


def _get_signature_component(
    func: TCallableAny | None,
    name: str,
    max_args: int = MAX_ARGS,
) -> tp.Tuple[str, str]:
    """Return a signature component, either a delegate or terminus. If named, a leading period will be included."""
    if func:
        # sig will just be `()` (maybe with args) at this point
        sig = _get_parameters(func, max_args=max_args)
        if name and name != '__call__':
            # prefix with name if name is not __call__
            sig = f'.{name}{sig}'
            sig_no_args = f'.{name}()'
        else:  # assume just function call
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
    """
    Utility to get two versions of ``func`` and ``delegate_func`` signatures

    Args:
        name_no_args: If this signature has a ``delegate_func``, the root name might need to be provided in a version with no arguments (if the root itself is a function).
    """
    delegate, delegate_no_args = _get_signature_component(
        delegate_func, delegate_name, max_args=max_args
    )
    terminus, terminus_no_args = _get_signature_component(
        terminus_func, terminus_name, max_args=max_args
    )
    dns = f'.{delegate_namespace}' if delegate_namespace else ''

    name_args = (
        '' if func is None else _get_parameters(func, is_getitem, max_args=max_args)
    )
    signature = f'{name}{name_args}{dns}{delegate}{terminus}'

    name_no_args = name if not name_no_args else name_no_args
    if func is None:  # name is a property
        signature_no_args = f'{name_no_args}{dns}{delegate_no_args}{terminus_no_args}'
    elif is_getitem:
        signature_no_args = f'{name_no_args}[]{dns}{delegate_no_args}{terminus_no_args}'
    else:
        signature_no_args = f'{name_no_args}(){dns}{delegate_no_args}{terminus_no_args}'

    return signature, signature_no_args


def valid_argument_types(
    func: TCallableAny,
    # cls: tp.Type[ContainerBase] | None = None,
) -> None:
    if not hasattr(func, '__name__'):
        return  # filter out classes

    is_method = inspect.ismethod(func)  # is a class method
    sig = inspect.signature(func)
    params: dict[inspect._ParameterKind, int] = defaultdict(int)
    for i, (name, p) in enumerate(sig.parameters.items()):
        # if an instance method, ignore self; when not instantiated, only class methods are is_method
        if i == 0 and not is_method and name == 'self':
            continue
        params[p.kind] += 1
    if not params:
        return

    if p.POSITIONAL_OR_KEYWORD in params:
        if pos_or_kw := params.pop(p.POSITIONAL_OR_KEYWORD):
            raise RuntimeError(
                f'Invalid interface ({func.__name__}): {pos_or_kw} positional-or-keyward arguments.'
            )

    if p.POSITIONAL_ONLY in params:
        if (pos_only := params.pop(p.POSITIONAL_ONLY)) > 2:
            raise RuntimeError(
                f'Invalid interface ({func.__name__}): {pos_only} positional only is more than 2.'
            )  # pragma: no cover

    if p.KEYWORD_ONLY in params:
        params.pop(p.KEYWORD_ONLY)
    if p.VAR_POSITIONAL in params:  # *args
        params.pop(p.VAR_POSITIONAL)

    if params:
        raise RuntimeError(
            f'Invalid interface ({func.__name__}): unexpected argument type: {params}'
        )  # pragma: no cover


# -------------------------------------------------------------------------------
class Features:
    """
    Core utilities need by both Interface and InterfaceSummary
    """

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
        '__reversed__',
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
    def scrub_doc(
        cls,
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


# -------------------------------------------------------------------------------


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
INTERFACE_GROUP_ORDER = tuple(
    v for k, v in vars(InterfaceGroup).items() if not k.startswith('_')
)

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
    'Accessor Mapping': 'Return a wrapper around Series data that fully implements the Python Mapping interface.',
}


class InterfaceRecord(tp.NamedTuple):
    cls_name: str
    group: str  # should be InterfaceGroup
    signature: str
    doc: str
    reference: str = ''  # a qualified name as a string for doc gen
    use_signature: bool = False
    is_attr: bool = False
    delegate_reference: str = ''
    delegate_is_attr: bool = False
    signature_no_args: str = ''

    @classmethod
    def gen_from_dict_like(
        cls,
        *,
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
        yield cls(
            cls_name,
            InterfaceGroup.DictLike,
            signature,
            doc,
            reference,
            signature_no_args=signature_no_args,
        )

    @classmethod
    def gen_from_display(
        cls,
        *,
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
            yield cls(
                cls_name,
                InterfaceGroup.Display,
                signature,
                doc,
                reference,
                signature_no_args=signature_no_args,
            )
        else:  # interface attr
            yield cls(
                cls_name,
                InterfaceGroup.Display,
                name,
                doc,
                use_signature=True,
                signature_no_args=name,
            )

    @classmethod
    def gen_from_astype(
        cls,
        *,
        cls_name: str,
        cls_target: tp.Type[ContainerBase],
        name: str,
        obj: tp.Any,
        reference: str,
        doc: str,
        max_args: int,
        max_doc_chars: int,
    ) -> tp.Iterator[InterfaceRecord]:
        if isinstance(
            obj,
            (
                InterfaceFrameAsType,
                InterfaceIndexHierarchyAsType,
                InterfaceBatchAsType,
            ),
        ):
            for field in obj._INTERFACE:
                delegate_obj = getattr(obj, field)
                delegate_reference = f'{obj.__class__.__name__}.{field}'
                if field == Features.GETITEM:
                    cls_returned = (
                        FrameAsType
                        if isinstance(obj, InterfaceFrameAsType)
                        else IndexHierarchyAsType
                    )
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
                yield cls(
                    cls_name,
                    InterfaceGroup.Method,
                    signature,
                    doc,
                    reference,
                    use_signature=True,
                    is_attr=True,
                    delegate_reference=delegate_reference,
                    signature_no_args=signature_no_args,
                )
        else:  # Series, Index, astype is just a method
            signature, signature_no_args = _get_signatures(name, obj, max_args=max_args)
            yield cls(
                cls_name,
                InterfaceGroup.Method,
                signature,
                doc,
                reference,
                signature_no_args=signature_no_args,
            )

    @classmethod
    def gen_from_persist(
        cls,
        *,
        cls_name: str,
        cls_target: tp.Type[ContainerBase],
        name: str,
        obj: tp.Any,
        reference: str,
        doc: str,
        max_args: int,
        max_doc_chars: int,
    ) -> tp.Iterator[InterfaceRecord]:
        """Interfaces that are not full selectors or via but define an _INTERFACE component."""
        assert isinstance(obj, InterfacePersist)
        for field in obj._INTERFACE:
            doc = Features.scrub_doc(
                getattr(obj.__class__, field).__doc__,
                max_doc_chars=max_doc_chars,
            )
            delegate_reference = f'{obj.__class__.__name__}.{field}'
            delegate_obj = getattr(obj, field)

            if field in ('iloc', 'loc'):
                field = f'{name}.{field}'
                delegate_obj = delegate_obj.__getitem__
                is_getitem = True
                delegate_name = field
            elif field == '__call__':
                field = name
                is_getitem = False
                delegate_name = ''
            else:  # getitem
                field = name  # will get brackets added
                is_getitem = True
                delegate_name = ''

            signature, signature_no_args = _get_signatures(
                field,
                delegate_obj,
                is_getitem=is_getitem,
                delegate_name=delegate_name,
                max_args=max_args,
            )
            yield cls(
                cls_name,
                InterfaceGroup.Method,
                signature,
                doc,
                reference,
                delegate_reference=delegate_reference,
                delegate_is_attr=is_getitem,
                is_attr=True,
                signature_no_args=signature_no_args,
                use_signature=True,
            )

    @classmethod
    def gen_from_consolidate(
        cls,
        *,
        cls_name: str,
        cls_target: tp.Type[ContainerBase],
        name: str,
        obj: tp.Any,
        reference: str,
        doc: str,
        max_args: int,
        max_doc_chars: int,
    ) -> tp.Iterator[InterfaceRecord]:
        """Interfaces that are not full selectors or via but define an _INTERFACE component."""
        if isinstance(obj, InterfaceConsolidate):
            for field in obj._INTERFACE:
                doc = Features.scrub_doc(
                    getattr(obj.__class__, field).__doc__,
                    max_doc_chars=max_doc_chars,
                )
                delegate_reference = f'{obj.__class__.__name__}.{field}'
                delegate_obj = getattr(obj, field)

                if field == 'status':  # a property
                    signature = f'{name}.{field}'  # manual construct signature
                    yield cls(
                        cls_name,
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
                    yield cls(
                        cls_name,
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
            yield cls(
                cls_name,
                InterfaceGroup.Method,
                signature,
                doc,
                reference,
                signature_no_args=signature_no_args,
            )

    @classmethod
    def gen_from_constructor(
        cls,
        *,
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
        yield cls(
            cls_name,
            InterfaceGroup.Constructor,
            signature,
            doc,
            reference,
            signature_no_args=signature_no_args,
        )

    @classmethod
    def gen_from_exporter(
        cls,
        *,
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
        yield cls(
            cls_name,
            InterfaceGroup.Exporter,
            signature,
            doc,
            reference,
            signature_no_args=signature_no_args,
        )

    @classmethod
    def gen_from_iterator(
        cls,
        *,
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
            obj.__call__,  # type: ignore
            is_getitem=False,
            max_args=max_args,
        )

        yield cls(
            cls_name,
            InterfaceGroup.Iterator,
            signature,
            doc,
            reference,
            use_signature=True,
            is_attr=True,  # doc as attr so sphinx does not add parens to sig
            signature_no_args=signature_no_args,
        )
        # TypeBlocks as iter_* methods that are just functions
        if hasattr(obj, 'CLS_DELEGATE'):
            cls_interface = obj.CLS_DELEGATE
            # IterNodeDelegate or IterNodeDelegateMapable
            for field in cls_interface._INTERFACE:  # apply, map, etc
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
                        terminus_obj = Reduce  # cls_interface.CLS_DELEGATE
                        for terminus_name in terminus_obj._INTERFACE:
                            terminus_func = getattr(terminus_obj, terminus_name)
                            signature, signature_no_args = _get_signatures(
                                name,
                                obj.__call__,  # type: ignore
                                is_getitem=False,
                                delegate_func=delegate_sub_obj,
                                delegate_name=field_sub,
                                delegate_namespace='reduce',
                                max_args=max_args,
                                terminus_name=terminus_name,
                                terminus_func=terminus_func,
                            )
                            delegate_reference = f'{delegate_obj.__name__}.{field_sub}'
                            yield cls(
                                cls_name,
                                InterfaceGroup.Iterator,
                                signature,
                                doc,
                                reference,
                                use_signature=True,
                                is_attr=True,
                                delegate_reference=delegate_reference,
                                signature_no_args=signature_no_args,
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
                        obj.__call__,  # type: ignore
                        is_getitem=False,
                        delegate_func=delegate_obj,
                        delegate_name=field,
                        max_args=max_args,
                    )
                    yield cls(
                        cls_name,
                        InterfaceGroup.Iterator,
                        signature,
                        doc,
                        reference,
                        use_signature=True,
                        is_attr=True,
                        delegate_reference=delegate_reference,
                        signature_no_args=signature_no_args,
                    )

    @classmethod
    def gen_from_accessor(
        cls,
        *,
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
        if cls_interface is InterfaceValues or cls_interface is InterfaceBatchValues:  # type: ignore[comparison-overlap]
            group = InterfaceGroup.AccessorValues
        elif cls_interface is InterfaceString or cls_interface is InterfaceBatchString:  # type: ignore[comparison-overlap]
            group = InterfaceGroup.AccessorString
        elif (
            cls_interface is InterfaceDatetime or cls_interface is InterfaceBatchDatetime  # type: ignore[comparison-overlap]
        ):
            group = InterfaceGroup.AccessorDatetime
        elif (
            cls_interface is InterfaceTranspose
            or cls_interface is InterfaceBatchTranspose  # type: ignore[comparison-overlap]
        ):
            group = InterfaceGroup.AccessorTranspose
        elif (
            cls_interface is InterfaceFillValue
            or cls_interface is InterfaceBatchFillValue  # type: ignore[comparison-overlap]
        ):
            group = InterfaceGroup.AccessorFillValue
        elif cls_interface is InterfaceRe or cls_interface is InterfaceBatchRe:  # type: ignore[comparison-overlap]
            group = InterfaceGroup.AccessorRe
        elif cls_interface is InterfaceHashlib:  # type: ignore[comparison-overlap]
            group = InterfaceGroup.AccessorHashlib
        elif cls_interface is TypeClinic:  # type: ignore[comparison-overlap]
            group = InterfaceGroup.AccessorTypeClinic
        elif (
            issubclass(cls_interface, ReduceDispatch)
            or cls_interface is InterfaceBatchReduceDispatch  # type: ignore[comparison-overlap]
        ):
            group = InterfaceGroup.AccessorReduce
        elif issubclass(cls_interface, Mapping):
            group = InterfaceGroup.AccessorMapping
        else:
            raise NotImplementedError(cls_interface)  # pragma: no cover

        terminus_name_no_args: tp.Optional[str]

        for field in cls_interface._INTERFACE:  # apply, map, etc
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
                        None,  # force name as a property
                        delegate_func=delegate_obj,
                        delegate_name=field,
                        max_args=max_args,
                        terminus_name=terminus_name,
                        terminus_func=terminus_func,
                    )
                    yield cls(
                        cls_name,
                        group,
                        signature,
                        doc,
                        reference,
                        is_attr=True,
                        use_signature=True,
                        delegate_reference=delegate_reference,
                        delegate_is_attr=False,
                        signature_no_args=signature_no_args,
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
                    yield cls(
                        cls_name,
                        group,
                        terminus_name,
                        doc,
                        reference,
                        is_attr=True,
                        use_signature=True,
                        delegate_reference=delegate_reference,
                        delegate_is_attr=True,
                        signature_no_args=terminus_name_no_args,
                    )
                else:
                    signature, signature_no_args = _get_signatures(
                        terminus_name,
                        delegate_obj,
                        max_args=max_args,
                        name_no_args=terminus_name_no_args,
                    )
                    yield cls(
                        cls_name,
                        group,
                        signature,
                        doc,
                        reference,
                        is_attr=True,
                        use_signature=True,
                        delegate_reference=delegate_reference,
                        signature_no_args=signature_no_args,
                    )

    @classmethod
    def gen_from_getitem(
        cls,
        *,
        cls_name: str,
        cls_target: tp.Type[ContainerBase],
        name: str,
        obj: TCallableAny,
        reference: str,
        doc: str,
        max_args: int,
        max_doc_chars: int,
    ) -> tp.Iterator[InterfaceRecord]:
        """
        For root __getitem__ methods, as well as __getitem__ on InterGetItemLocReduces objects.
        """
        if name != Features.GETITEM:
            target = obj.__getitem__  # type: ignore
        else:
            target = obj
            name = ''

        signature, signature_no_args = _get_signatures(
            name,
            target,
            is_getitem=True,
            max_args=max_args,
        )

        yield InterfaceRecord(
            cls_name,
            InterfaceGroup.Selector,
            signature,
            doc,
            reference,
            use_signature=True,
            is_attr=True,
            signature_no_args=signature_no_args,
        )

    @classmethod
    def gen_from_selection(
        cls,
        *,
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
                    f'{name}.{field}',  # make compound interface
                    delegate_obj.__getitem__,
                    is_getitem=True,
                    max_args=max_args,
                )
            else:  # is getitem
                delegate_is_attr = False
                signature, signature_no_args = _get_signatures(
                    name,  # on the root, no change necessary
                    delegate_obj,
                    is_getitem=True,
                    max_args=max_args,
                )

            yield InterfaceRecord(
                cls_name,
                InterfaceGroup.Selector,
                signature,
                doc,
                reference,
                use_signature=True,
                is_attr=True,
                delegate_reference=delegate_reference,
                delegate_is_attr=delegate_is_attr,
                signature_no_args=signature_no_args,
            )

    @classmethod
    def gen_from_assignment(
        cls,
        *,
        cls_name: str,
        cls_target: tp.Type[ContainerBase],
        name: str,
        obj: tp.Union[
            InterfaceAssignTrio[TVContainer_co], InterfaceAssignQuartet[TVContainer_co]
        ],
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
                        f'{name}.{field}',  # make compound interface
                        delegate_obj.__getitem__,
                        is_getitem=True,
                        delegate_func=terminus_obj,
                        delegate_name=field_terminus,
                        max_args=max_args,
                    )
                else:  # is getitem
                    signature, signature_no_args = _get_signatures(
                        name,  # on the root, no change necessary
                        delegate_obj,
                        is_getitem=True,
                        delegate_func=terminus_obj,
                        delegate_name=field_terminus,
                        max_args=max_args,
                    )

                yield InterfaceRecord(
                    cls_name,
                    InterfaceGroup.Assignment,
                    signature,
                    terminus_doc,
                    reference,
                    use_signature=True,
                    is_attr=True,
                    delegate_reference=terminus_reference,
                    signature_no_args=signature_no_args,
                )

    @classmethod
    def gen_from_method(
        cls,
        *,
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
            yield InterfaceRecord(
                cls_name,
                InterfaceGroup.OperatorUnary,
                signature,
                doc,
                reference,
                signature_no_args=signature_no_args,
            )
        elif name in UFUNC_BINARY_OPERATORS or name in RIGHT_OPERATOR_MAP:
            # NOTE: as all classes have certain binary operators by default, we need to only show binary operators for ContainerOperand subclasses
            if issubclass(cls_target, ContainerOperandSequence):
                yield InterfaceRecord(
                    cls_name,
                    InterfaceGroup.OperatorBinary,
                    signature,
                    doc,
                    reference,
                    signature_no_args=signature_no_args,
                )
        else:
            yield InterfaceRecord(
                cls_name,
                InterfaceGroup.Method,
                signature,
                doc,
                reference,
                signature_no_args=signature_no_args,
            )

    @classmethod
    def gen_from_class(
        cls,
        *,
        cls_name: str,
        cls_target: tp.Type[ContainerBase],
        name: str,
        obj: TCallableAny,
        reference: str,
        doc: str,
        max_args: int,
        max_doc_chars: int,
    ) -> tp.Iterator[InterfaceRecord]:
        """For classes defined on outer classes."""
        signature, signature_no_args = _get_signatures(name, obj, max_args=max_args)
        yield InterfaceRecord(
            cls_name,
            InterfaceGroup.Constructor,
            signature,
            doc,
            reference,
            signature_no_args=signature_no_args,
            use_signature=True,
        )


# -------------------------------------------------------------------------------


class InterfaceSummary(Features):
    _CLS_TO_INSTANCE_CACHE: tp.Dict[tp.Type[ContainerBase], ContainerBase] = {}
    _CLS_ONLY = frozenset(
        (
            WWW,
            CallGuard,
            Require,
        )
    )
    _CLS_INIT_SIMPLE = (
        frozenset(
            (
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
            )
        )
        | _CLS_ONLY
    )

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
        """
        Get a sample instance from any ContainerBase; cache to only create one per life of process.
        """
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
                instance = target.from_labels(((0, 0),))
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
    def name_obj_iter(
        cls,
        target: tp.Type[ContainerBase],
    ) -> tp.Iterator[tp.Tuple[str, tp.Any, tp.Any]]:
        instance = cls.get_instance(target=target)

        if hasattr(target.__class__, 'interface'):
            yield 'interface', None, ContainerBase.__class__.interface  # type: ignore

        # force these to be ordered at the bottom
        selectors_found = set()

        for name_attr in sorted(dir(target)):
            if name_attr == 'interface':
                continue  # skip, provided by metaclass
            if not cls.is_public(name_attr):
                continue
            if target in cls._CLS_ONLY and (
                name_attr == '__init__' or name_attr in Features.DISPLAY
            ):
                continue

            if name_attr in cls._SELECTORS:
                selectors_found.add(name_attr)
                continue
            try:
                yield (
                    name_attr,
                    getattr(instance, name_attr),
                    getattr(target, name_attr),
                )
            except NotImplementedError:  # base class properties that are not implemented
                pass

        for name_attr in cls._SELECTORS:
            if name_attr in selectors_found:
                yield (
                    name_attr,
                    getattr(instance, name_attr),
                    getattr(target, name_attr),
                )

    # ---------------------------------------------------------------------------
    @classmethod
    def interrogate(
        cls,
        target: tp.Type[ContainerBase],
        *,
        max_args: int = MAX_ARGS,
        max_doc_chars: int = MAX_DOC_CHARS,
    ) -> tp.Iterator[InterfaceRecord]:
        for name_attr, obj, obj_cls in cls.name_obj_iter(target):
            doc = ''

            if isinstance(obj_cls, property):
                doc = cls.scrub_doc(
                    obj_cls.__doc__,
                    max_doc_chars=max_doc_chars,
                )
            elif hasattr(obj, '__doc__'):
                doc = cls.scrub_doc(
                    obj.__doc__,
                    max_doc_chars=max_doc_chars,
                )

            if name_attr == 'values':
                name = name_attr  # on Batch this is generator that has an generic name
            elif hasattr(obj, '__name__'):
                name = obj.__name__
            else:  # some attributes yield objects like arrays, Series, or Frame
                name = name_attr

            cls_name = target.__name__
            reference = f'{cls_name}.{name}'  # check if this is still necessary

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
                yield from InterfaceRecord.gen_from_dict_like(**kwargs)  # pyright: ignore
            elif name in cls.DISPLAY:
                yield from InterfaceRecord.gen_from_display(**kwargs)  # pyright: ignore
            elif name == 'astype':
                yield from InterfaceRecord.gen_from_astype(**kwargs)  # pyright: ignore
            elif name == 'persist':
                yield from InterfaceRecord.gen_from_persist(**kwargs)  # pyright: ignore
            elif name == 'consolidate':
                yield from InterfaceRecord.gen_from_consolidate(**kwargs)  # pyright: ignore
            elif (callable(obj) and name.startswith('from_')) or name == '__init__':
                yield from InterfaceRecord.gen_from_constructor(**kwargs)  # pyright: ignore
            elif callable(obj) and name.startswith('to_'):
                yield from InterfaceRecord.gen_from_exporter(**kwargs)  # pyright: ignore
            elif name.startswith('iter_'):
                yield from InterfaceRecord.gen_from_iterator(**kwargs)  # pyright: ignore
            elif (
                isinstance(
                    obj,
                    (
                        InterGetItemLoc,
                        InterGetItemLocInPlace,
                        InterGetItemLocReduces,
                        InterGetItemLocCompound,
                        InterGetItemLocCompoundReduces,
                        InterGetItemILoc,
                        InterGetItemILocInPlace,
                        InterGetItemILocReduces,
                        InterGetItemILocCompound,
                        InterGetItemILocCompoundReduces,
                        InterfaceGetItemBLoc,
                    ),
                )
                or name == cls.GETITEM
            ):
                yield from InterfaceRecord.gen_from_getitem(**kwargs)  # pyright: ignore

            elif obj.__class__ in INTERFACE_ATTRIBUTE_CLS:
                yield from InterfaceRecord.gen_from_accessor(
                    cls_interface=obj.__class__,
                    **kwargs,  # pyright: ignore
                )
            elif obj.__class__ in (InterfaceSelectDuo, InterfaceSelectTrio):
                yield from InterfaceRecord.gen_from_selection(
                    cls_interface=obj.__class__,
                    **kwargs,  # pyright: ignore
                )
            elif obj.__class__ in (InterfaceAssignTrio, InterfaceAssignQuartet):
                yield from InterfaceRecord.gen_from_assignment(
                    cls_interface=obj.__class__,
                    **kwargs,  # pyright: ignore
                )
            # as InterfaceFillValue, InterfaceRe are methods, must match on name, not INTERFACE_ATTRIBUTE_CLS
            elif name == 'via_fill_value':
                yield from InterfaceRecord.gen_from_accessor(
                    cls_interface=InterfaceFillValue,
                    **kwargs,  # pyright: ignore
                )
            elif name == 'via_re':
                yield from InterfaceRecord.gen_from_accessor(
                    cls_interface=InterfaceRe,
                    **kwargs,  # pyright: ignore
                )
            elif name == 'via_mapping':
                yield from InterfaceRecord.gen_from_accessor(
                    cls_interface=obj.__class__,
                    **kwargs,  # pyright: ignore
                )
            elif name == 'reduce':  # subclasses not in INTERFACE_ATTRIBUTE_CLS
                yield from InterfaceRecord.gen_from_accessor(
                    cls_interface=obj.__class__,
                    **kwargs,  # pyright: ignore
                )
            elif callable(obj):
                if obj.__class__ == type:  # a class defined on this class
                    yield from InterfaceRecord.gen_from_class(**kwargs)  # pyright: ignore
                else:  # general methods
                    yield from InterfaceRecord.gen_from_method(**kwargs)  # pyright: ignore
            else:
                yield InterfaceRecord(
                    cls_name,
                    InterfaceGroup.Attribute,
                    name,
                    doc,
                    reference,
                    signature_no_args=name,
                )

    @classmethod
    def to_frame(
        cls,
        target: tp.Type[ContainerBase],
        *,
        minimized: bool = True,
        max_args: int = MAX_ARGS,
        max_doc_chars: int = MAX_DOC_CHARS,
    ) -> TFrameAny:
        """
        Reduce to key fields.
        """
        f: TFrameAny = Frame.from_records(
            cls.interrogate(
                target,
                max_args=max_args,
                max_doc_chars=max_doc_chars,
            ),
        )
        # order be group order
        f = Frame.from_concat(
            (f.loc[f['group'] == g] for g in INTERFACE_GROUP_ORDER),
            name=target.__name__,
        )
        f = f.set_index('signature', drop=True)

        # derive Sphinx-RST compatible (case insensitive) label that handles single character case-senstive attrs in FillValueAuto
        sna_label = (
            f['signature_no_args']
            .iter_element()
            .apply(lambda e: e if len(e) > 1 else f'{e}_' if e.isupper() else e)
            .via_str.lower()
            .rename('sna_label')
        )

        assert len(sna_label.unique()) == len(f)
        f = Frame.from_concat((f, sna_label), axis=1)

        if minimized:
            return f[['cls_name', 'group', 'doc']]
        return f
