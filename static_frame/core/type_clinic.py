from __future__ import annotations

import re
import types
import typing
import warnings
from collections import deque
from collections.abc import MutableMapping
from collections.abc import Sequence
from enum import Enum
from functools import partial
from functools import wraps
from inspect import BoundArguments
from inspect import Signature
from itertools import chain
from itertools import repeat

import numpy as np
import typing_extensions as tp
from numpy.typing import NBitBase

from static_frame.core.bus import Bus
from static_frame.core.frame import Frame
from static_frame.core.index import Index
from static_frame.core.index_base import IndexBase
from static_frame.core.index_datetime import IndexDatetime
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.series import Series
from static_frame.core.util import DTYPE_COMPLEX_KIND
from static_frame.core.util import INT_TYPES
from static_frame.core.util import TLabel
from static_frame.core.yarn import Yarn

TFrameAny = Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]]

TValidator = tp.Callable[..., bool]
TLabelMatchSpecifier = tp.Union[TLabel, tp.Pattern[tp.Any], tp.Set[TLabel]]

if tp.TYPE_CHECKING:
    from types import EllipsisType  # pragma: no cover
    TDtypeAny = np.dtype[tp.Any] #pragma: no cover
    TShapeComponent = tp.Union[int, EllipsisType] #pragma: no cover # pyright: ignore
    TShapeSpecifier = tp.Tuple[TShapeComponent, ...] #pragma: no cover

def _iter_generic_classes() -> tp.Iterable[tp.Type[tp.Any]]:
    if t := getattr(types, 'GenericAlias', None):
        yield t
    if t := getattr(typing, '_GenericAlias', None):
        yield t # pyright: ignore

GENERIC_TYPES = tuple(_iter_generic_classes())

def _iter_union_classes() -> tp.Iterable[tp.Type[tp.Any]]:
    if t := getattr(types, 'UnionType', None):
        yield t
    if t := getattr(typing, '_UnionGenericAlias', None):
        yield t # pyright: ignore

UNION_TYPES = tuple(_iter_union_classes())

def _iter_unpack_classes() -> tp.Iterable[tp.Type[tp.Any]]:
    # NOTE: type extensions Unpack is not equal to typing.Unpack
    if t := getattr(tp, 'Unpack', None):
        yield t
    if t := getattr(typing, 'Unpack', None):
        yield t # pyright: ignore

UNPACK_TYPES = tuple(_iter_unpack_classes())

def is_generic(hint: tp.Any) -> bool:
    return isinstance(hint, GENERIC_TYPES)

def is_union(hint: tp.Any) -> bool:
    if UNION_TYPES:
        return isinstance(hint, UNION_TYPES)
    # this might only be possible pre 3.9
    elif isinstance(hint, GENERIC_TYPES): #pragma: no cover
        return tp.get_origin(hint) is tp.Union #pragma: no cover
    return False #pragma: no cover

def is_unpack(origin: tp.Any, generic_alias: tp.Any) -> bool:
    # NOTE: the * syntax on *tuple does not create an Unpack instances
    # tp.get_origin(*tuple[tp.Any, ...]) is tuple
    # NOTE: cannot use isinstance or issubclass with Unpack
    if origin in UNPACK_TYPES:
        return True
    if getattr(generic_alias, '__unpacked__', False):
        # if hint.__unpacked__ is True, this is a *tuple that does not need to be unpacked
        return True
    return False

def get_args_unpack(hint: tp.Any) -> tp.Any:
    '''Normalize the heterogeneity of dealing with *tuple[tp.Any, ...] and tp.Unpack[tp.Tuple[tp.Any, *]]; always return the contained tuple generic alias
    '''
    if getattr(hint, '__unpacked__', False):
        # if hint.__unpacked__ is True, this is a *tuple that does not need to be unpacked
        tga = hint
    else:
        # unpack from tp.Unpack to return the tuple generic alias
        [tga] = tp.get_args(hint)
    assert issubclass(tuple, tp.get_origin(tga))
    return (tga,) # clients expect a tuple of size 1

#-------------------------------------------------------------------------------

TParent = tp.Tuple[tp.Any, ...]
# A validation record can be used to queue checks or report errors
TValidation = tp.Tuple[tp.Any, tp.Any, TParent, TParent]

#-------------------------------------------------------------------------------
# error reporting, presentation

ERROR_MESSAGE_TYPE = object()

def to_name(v: tp.Any,
        func_to_str: tp.Callable[..., str] = str,
        ) -> str:
    if is_generic(v):
        if hasattr(v, '__name__'):
            name = v.__name__
        else:
            # for older Python, not all generics have __name__
            origin = tp.get_origin(v)
            if hasattr(origin, '__name__'):
                name = origin.__name__
            elif is_unpack(origin, v): # needed for backwards compat
                name = 'Unpack'
            else:
                name = str(origin)
        s = f'{name}[{", ".join(to_name(q) for q in tp.get_args(v))}]'
    elif isinstance(v, tp.TypeVar):
        # str() gets tilde, __name__ does not have tilde
        if v.__bound__:
            s = f'{v}: {to_name(v.__bound__)}'
        elif v.__constraints__:
            s = f'{v}: {to_name(v.__constraints__)}'
        else:
            s = str(v)
    elif hasattr(v, '__name__'):
        s = v.__name__
    elif v is ...:
        s = '...'
    elif isinstance(v, tuple):
        s = f"({', '.join(to_name(w) for w in v)})"
    else:
        s = func_to_str(v)
    return s

def to_signature(
        sig: BoundArguments,
        hints: tp.Mapping[str, tp.Any]) -> str:
    msg = []
    for k in sig.arguments:
        msg.append(f'{k}: {to_name(hints.get(k, tp.Any))}')
    r = to_name(hints.get('return', tp.Any))
    return f'({", ".join(msg)}) -> {r}'



class ClinicResult:
    '''A ``ClinicResult`` instance stores zero or more error messages resulting from a check.
    '''
    __slots__ = ('_log',)

    _LINE = '─'
    _CORNER = '└'
    _WIDTH = 4

    def __init__(self, log: tp.Sequence[TValidation]) -> None:
        self._log = log

    def __iter__(self) -> tp.Iterator[TValidation]:
        return self._log.__iter__()

    def __len__(self) -> int:
        return len(self._log)

    def __bool__(self) -> bool:
        '''Return True if there are validation issues.
        '''
        return bool(self._log)

    @property
    def validated(self) -> bool:
        return not bool(self._log)

    @classmethod
    def _get_indent(cls, size: int) -> str:
        if size < 1:
            return ''
        c_width = cls._WIDTH - 2
        l_width = cls._WIDTH * (size - 1)
        return f'{" " * l_width}{cls._CORNER}{cls._LINE * c_width} '

    def to_str(self) -> str:
        '''Return error messages as a formatted string with line breaks and indentation.
        '''
        msg = []
        for v, h, ph, pv in self._log:
            if ph:
                path_components = []
                for i, pc in enumerate(ph):
                    path_components.append(f'{self._get_indent(i)}{to_name(pc)}')
                path = '\n'.join(path_components)
                i_next = i + 1
            else:
                path = ''
                i_next = 1

            if v is ERROR_MESSAGE_TYPE: # in this case, do not use the value
                error_msg = f'{h}'
            else:
                error_msg = f'Expected {to_name(h)}, provided {to_name(type(v))} invalid'

            if not path:
                msg.append(f'\n{error_msg}')
            else:
                msg.append(f'\nIn {path}\n{self._get_indent(i_next)}{error_msg}')

        return ''.join(msg)

    def __repr__(self) -> str:
        log_len = len(self)
        return f'<ClinicResult: {log_len} {"errors" if log_len != 1 else "error"}>'


class ClinicError(TypeError):
    '''A TypeError subclass for exposing check errors.
    '''
    def __init__(self, cr: ClinicResult) -> None:
        TypeError.__init__(self, cr.to_str())


#-------------------------------------------------------------------------------

class Validator:
    '''Base class of all run-time constraints, deployed in ``Annotated`` generics.
    '''
    __slots__: tp.Tuple[str, ...] = ()

    def _iter_errors(self,
            value: tp.Any,
            hint: tp.Any,
            parent_hints: TParent,
            parent_values: TParent,
            ) -> tp.Iterator[TValidation]:
        raise NotImplementedError() #pragma: no cover

    def __repr__(self) -> str:
        args = ', '.join(to_name(getattr(self, v)) for v in self.__slots__)
        return f'{self.__class__.__name__}({args})'


class Require:
    '''A collection of classes to be used in ``Annotated`` generics to perform run-time data validations.
    '''
    __slots__ = ()

    class Name(Validator):
        '''Validate the name of a container.

        Args:
            name: The name to validate against.
            /
        '''
        __slots__ = ('_name',)

        def __init__(self, name: TLabel, /):
            self._name = name

        def _iter_errors(self,
                value: tp.Any,
                hint: tp.Any,
                parent_hints: TParent,
                parent_values: TParent,
                ) -> tp.Iterator[TValidation]:
            # returning anything is an error
            if (n := value.name) != self._name:
                yield (ERROR_MESSAGE_TYPE,
                        f'Expected name {self._name!r}, provided name {n!r}',
                        parent_hints,
                        parent_values,
                        )

    class Len(Validator):
        '''Validate the length of a container.

        Args:
            len: The length to validate against.
            /
        '''

        __slots__ = ('_len',)

        def __init__(self, len: int, /):
            self._len = len

        def _iter_errors(self,
                value: tp.Any,
                hint: tp.Any,
                parent_hints: TParent,
                parent_values: TParent,
                ) -> tp.Iterator[TValidation]:
            if (vl := len(value)) != self._len:
                yield (ERROR_MESSAGE_TYPE,
                        f'Expected length {self._len}, provided length {vl}',
                        parent_hints,
                        parent_values,
                        )

    class Shape(Validator):
        '''Validate the length of a container.

        Args:
            shape: A tuple of one or two values, where values are either an integer or an `...`, specifying any value for that position. The size of the shape always species the dimensionality.
            /
        '''

        __slots__ = ('_shape',)

        @staticmethod
        def _validate_shape_component(ss: TShapeSpecifier) -> TShapeSpecifier:
            for c in ss:
                if c is not ... and not isinstance(c, INT_TYPES):
                    raise TypeError(f'Components must be either `...` or an integer, not {c!r}.')
            return ss

        def __init__(self, /, *shape: TShapeComponent):
            self._shape = self._validate_shape_component(shape)

        def _iter_errors(self,
                value: tp.Any,
                hint: tp.Any,
                parent_hints: TParent,
                parent_values: TParent,
                ) -> tp.Iterator[TValidation]:

            # same for both 1d and 2d
            if len(self._shape) != len(value.shape):
                yield (ERROR_MESSAGE_TYPE,
                        f'Expected shape ({self._shape}), provided shape {value.shape}',
                        parent_hints,
                        parent_values,
                        )
            elif self._shape[0] is not ... and self._shape[0] != value.shape[0]:
                yield (ERROR_MESSAGE_TYPE,
                        f'Expected shape ({self._shape}), provided shape {value.shape}',
                        parent_hints,
                        parent_values,
                        )
            elif len(self._shape) == 2 and self._shape[1] is not ... and self._shape[1] != value.shape[1]:
                yield (ERROR_MESSAGE_TYPE,
                        f'Expected shape ({self._shape}), provided shape {value.shape}',
                        parent_hints,
                        parent_values,
                        )

    class _LabelsValidator(Validator):
        __slots__ = ('_labels',)

        def __repr__(self) -> str:
            msg = []
            for v in self._labels: # type: ignore
                if isinstance(v, list):
                    parts = [to_name(p, func_to_str=repr) for p in v]
                    msg.append(f'[{", ".join(parts)}]')
                else:
                    msg.append(to_name(v, func_to_str=repr))
            return f'{self.__class__.__name__}({", ".join(msg)})'

        @staticmethod
        def _find_parent_frame(parent_values: TParent) -> TFrameAny | None:
            for v in reversed(parent_values):
                if isinstance(v, Frame):
                    return v
            return None

        @staticmethod
        def _split_validators(
                label: tp.Any,
                ) -> tp.Tuple[TLabel | tp.Set[TLabel], tp.Sequence[TValidator]]:
            '''Given an object that might be a label, or a list of label and validators, split into two and return label, validators
            '''
            label_e: TLabel
            label_validators: tp.Sequence[TValidator] | None

            # evaluate that all post-label values are callables?
            if isinstance(label, list):
                label_e = label[0] # must be first
                label_validators = label[1:]
            else:
                label_e = label
                label_validators = ()

            return label_e, label_validators


        @staticmethod
        def _iter_validator_results(*,
                frame: TFrameAny | None,
                labels: IndexBase,
                label: TLabel,
                validators: tp.Sequence[TValidator],
                parent_hints: TParent,
                parent_values: TParent,
                ) -> tp.Iterator[TValidation]:
            # be a no-op when no validators are present
            if validators:
                if frame is None:
                    raise RuntimeError('Provided label validators in a context without a discoverable Frame.')
                # NOTE: the same index instance might be on both axis, though this is unlikely
                if labels is frame.index:
                    s = frame.loc[label]
                elif labels is frame.columns:
                    s = frame[label]
                else:
                    raise RuntimeError('Labels associated with an index that is not a member of the parent Frame')
                for validator in validators:
                    if not validator(s):
                        yield (ERROR_MESSAGE_TYPE,
                                f'Validation failed of label {label!r} with {to_name(validator)}',
                                parent_hints,
                                parent_values,
                                )

    class LabelsOrder(_LabelsValidator):
        r'''Validate the ordering of labels.

        Args:
            \*labels: Provide labels as args. Use ... for regions of zero or more undefined labels.
        '''
        __slots__ = ()

        def __init__(self, *labels: tp.Sequence[TLabel]):
            self._labels: tp.Sequence[TLabel | tp.List[TLabel | TValidator]] = labels

        @staticmethod
        def _repr_remainder(
                labels: tp.Sequence[TLabel | tp.List[TLabel | TValidator]],
                ) -> str:
            # always drop leading or trailing ellipses
            if labels[0] is ...:
                labels = labels[1:]
            if labels[-1] is ...:
                labels = labels[:-1]
            return ', '.join((repr(l) if l is not ... else '...') for l in labels)

        def _provided_is_expected(self,
                label_e: tp.Any,
                label_p: TLabel,
                iloc_p: int,
                index: IndexBase,
                ) -> bool:
            if label_e == label_p:
                return True
            try:
                # expected is defined in the hint; see if we can use it as a lookup, permitting string to date combination in IndexDate
                return index.loc_to_iloc(label_e) == iloc_p # type: ignore
            except KeyError:
                pass
            return False

        def _iter_errors(self,
                value: tp.Any, # an index object
                hint: tp.Any,
                parent_hints: TParent,
                parent_values: TParent,
                ) -> tp.Iterator[TValidation]:

            if not isinstance(value, IndexBase):
                yield (ERROR_MESSAGE_TYPE,
                        f'Expected {self} to be used on Index or IndexHierarchy, not provided {to_name(type(value))}',
                        parent_hints,
                        parent_values,
                        )
            else:
                pf = self._find_parent_frame(parent_values)
                pos_e = 0 # position in expected
                len_e = len(self._labels)

                for iloc_p, label_p in enumerate(value): # iterate provided index
                    # print('pos_p:', iloc_p, repr(label_p), '| pos_e:', pos_e, repr(self._labels[pos_e]))
                    if pos_e >= len_e:
                        yield (ERROR_MESSAGE_TYPE,
                                f'Expected labels exhausted at provided {label_p!r}',
                                parent_hints,
                                parent_values,
                                )
                        break

                    label_e, label_validators = self._split_validators(self._labels[pos_e])
                    # partial processor, but defer calling until after label eval
                    iter_validator_results = partial(
                                self._iter_validator_results,
                                frame=pf,
                                labels=value,
                                parent_hints=parent_hints,
                                parent_values=parent_values,)

                    if label_e is not ...:
                        if not self._provided_is_expected(
                                label_e,
                                label_p,
                                iloc_p,
                                value,
                                ):
                            yield (ERROR_MESSAGE_TYPE,
                                    f'Expected {label_e!r}, provided {label_p!r}',
                                    parent_hints,
                                    parent_values,
                                    )
                            break
                        for log in iter_validator_results(
                                label=label_p,
                                validators=label_validators,
                                ):
                            yield log
                            break
                        pos_e += 1
                    # expected is an Ellipses; either find next as match or continue with Ellipses
                    elif pos_e + 1 == len_e: # last expected is an ellipses
                        # do not need to look ahead, evaluate validators
                        for log in iter_validator_results(
                                label=label_p,
                                validators=label_validators,
                                ):
                            yield log
                            break
                    elif pos_e + 1 < len_e: # more expected labels available
                        # look ahead to see if the next expected hint matches the current label, if so, we use those validators on this column
                        label_next_e, label_next_validators = self._split_validators(
                                self._labels[pos_e + 1],
                                )
                        if label_next_e is ...:
                            yield (ERROR_MESSAGE_TYPE,
                                    'Expected cannot be defined with adjacent ellipses',
                                    parent_hints,
                                    parent_values,
                                    )
                            break
                        if self._provided_is_expected(
                                label_next_e,
                                label_p,
                                iloc_p,
                                value,
                                ):
                            # NOTE: if current expected is an ellipses, and the current provided label is equal to the next expected, we know we are done with the ellipses region and must compare to the next expected value; we then skip that value for subsequent evaluation
                            for log in iter_validator_results(
                                    label=label_next_e, # type: ignore
                                    validators=label_next_validators,
                                    ):
                                yield log
                                break
                            pos_e += 2 # skip the compared value, prepare to get next
                        else:
                            # if the lookahead label is not this label, run these validators
                            for log in iter_validator_results(
                                    label=label_p,
                                    validators=label_validators,
                                    ):
                                yield log
                                break

                else: # no break, exhausted all labels; evaluate final conditions
                    # NOTE: must re-split validators to handle all scenarios
                    if pos_e + 1 == len_e and self._split_validators(self._labels[pos_e])[0] is ...:
                        pass # expected ending on an ellipses
                    elif pos_e < len_e: # if we have unevaluated expected
                        remainder = self._repr_remainder(self._labels[pos_e:])
                        yield (ERROR_MESSAGE_TYPE,
                                f'Expected has unmatched labels {remainder}',
                                parent_hints,
                                parent_values,
                                )


    class LabelsMatch(_LabelsValidator):
        r'''Validate the presence of one or more labels, specified with the value, a pattern, or set of values. Order of labels is not relevant.

        Args:
            \*labels: Provide labels matchers as args. A label matcher can be a label, a set of labels (of which at least one contained label must match), or a compiled regular expression (with which the `search` method is used to determine a match of string labels). Each label matcher provided must find at least one match, otherwise an error is returned.
        '''
        __slots__ = (
                '_match_labels',
                '_match_res',
                '_match_sets',
                '_match_to_validators',
                )

        def __init__(self, *labels: tp.Sequence[TLabelMatchSpecifier]):
            self._labels: tp.Sequence[TLabelMatchSpecifier | tp.List[TLabelMatchSpecifier | TValidator]] = labels

            self._match_labels: tp.Set[TLabel] = set()
            self._match_res: tp.List[tp.Pattern[str]] = []
            self._match_sets: tp.List[tp.FrozenSet[TLabel]] = []
            self._match_to_validators: tp.Dict[
                    tp.Union[TLabel, tp.Pattern[str], tp.FrozenSet[TLabel]],
                    tp.Sequence[TValidator]
                    ] = {}

            for l in labels:
                m, validators = self._split_validators(l)
                if isinstance(m, re.Pattern):
                    self._match_res.append(m)
                elif isinstance(m, set):
                    m = frozenset(m)
                    self._match_sets.append(m)
                else:
                    self._match_labels.add(m)

                # validator might be empty tuple
                self._match_to_validators[m] = validators

        def _iter_errors(self,
                value: tp.Any, # an index object
                hint: tp.Any,
                parent_hints: TParent,
                parent_values: TParent,
                ) -> tp.Iterator[TValidation]:

            if not isinstance(value, IndexBase):
                yield (ERROR_MESSAGE_TYPE,
                        f'Expected {self} to be used on Index or IndexHierarchy, not provided {to_name(type(value))}',
                        parent_hints,
                        parent_values,
                        )
            else:
                # count number of times each condition is met
                score = {k: 0 for k in chain(self._match_labels, self._match_res, self._match_sets)}

                pf = self._find_parent_frame(parent_values)
                iter_validator_results = partial(
                            self._iter_validator_results,
                            frame=pf,
                            labels=value,
                            parent_hints=parent_hints,
                            parent_values=parent_values,)

                # NOTE: a label_p will be tested to match each of the possible three scenarios, as different validators might be assigned to different groups, and all should be tested if validators exist

                for label_p in value: # iterate provided index
                    matched = False

                    if label_p in self._match_labels:
                        score[label_p] += 1
                        for log in iter_validator_results(label=label_p,
                                validators=self._match_to_validators[label_p],
                                ):
                            yield log
                            break
                        # continue

                    if isinstance(label_p, str): # NOTE: could coerce all values to strings
                        for match_re in self._match_res:
                            if match_re.search(label_p):
                                score[match_re] += 1
                                matched = True
                                for log in iter_validator_results(label=label_p,
                                        validators=self._match_to_validators[match_re],
                                        ):
                                    yield log
                                    break
                            if matched:
                                break
                        # if matched:
                        #     continue

                    for match_set in self._match_sets:
                        if label_p in match_set:
                            score[match_set] += 1
                            matched = True
                            for log in iter_validator_results(label=label_p,
                                    validators=self._match_to_validators[match_set],
                                    ):
                                yield log
                                break
                        if matched:
                            break

                for k, count in score.items():
                    if count == 0:
                        yield (ERROR_MESSAGE_TYPE,
                                f'Expected label to match {k!r}, no provided match',
                                parent_hints,
                                parent_values,
                                )

    class Apply(Validator):
        '''Apply a function to a container with an arbitrary function. The validation passes if the function returns True (or a truthy value).

        Args:
            func: A function that takes a container and returns a Boolean.
        '''

        __slots__ = ('_func',)

        def __init__(self, func: tp.Callable[..., bool], /):
            self._func: tp.Callable[..., bool] = func

        @staticmethod
        def _prepare_callable(func: tp.Callable[..., bool]) -> str:
            return func.__name__

        def _iter_errors(self,
                value: tp.Any,
                hint: tp.Any,
                parent_hints: TParent,
                parent_values: TParent,
                ) -> tp.Iterator[TValidation]:
            post = self._func(value)
            if not bool(post):
                yield (ERROR_MESSAGE_TYPE,
                        f'{to_name(type(value))} failed validation with {self._prepare_callable(self._func)}', parent_hints,
                        parent_values,
                        )

#-------------------------------------------------------------------------------

def _value_to_hint(value: tp.Any) -> tp.Any: # tp._GenericAlias
    if isinstance(value, type):
        return tp.Type[value]

    if isinstance(value, tuple):
        return value.__class__.__class_getitem__(tuple(_value_to_hint(v) for v in value))

    if isinstance(value, Sequence) and not isinstance(value, str):
        if not len(value):
            return value.__class__.__class_getitem__(tp.Any) # type: ignore[attr-defined]

        # as classes may not be hashable, we key to string name to from a set; this is imperfect
        ut = {v.__class__.__name__: v.__class__ for v in value}
        if len(ut) == 1:
            return value.__class__.__class_getitem__(ut[next(iter(ut.keys()))]) # type: ignore[attr-defined]

        hu = tp.Union.__getitem__(tuple(ut.values())) # pyright: ignore
        return value.__class__.__class_getitem__(hu) # type: ignore[attr-defined]

    if isinstance(value, MutableMapping):
        if not len(value):
            return value.__class__.__class_getitem__((tp.Any, tp.Any)) # type: ignore[attr-defined]

        keys_ut = {k.__class__.__name__: k.__class__ for k in value.keys()}
        values_ut = {v.__class__.__name__: v.__class__ for v in value.values()}

        if len(keys_ut) == 1:
            kt = keys_ut[next(iter(keys_ut.keys()))]
        else:
            kt = tp.Union.__getitem__(tuple(keys_ut.values())) # pyright: ignore

        if len(values_ut) == 1:
            vt = values_ut[next(iter(values_ut.keys()))]
        else:
            vt = tp.Union.__getitem__(tuple(values_ut.values())) # pyright: ignore

        return value.__class__.__class_getitem__((kt, vt)) # type: ignore[attr-defined]

    # --------------------------------------------------------------------------
    # SF containers

    if isinstance(value, Frame):
        hints = [_value_to_hint(value.index), _value_to_hint(value.columns)]
        hints.extend(dt.type().__class__ for dt in value._blocks._iter_dtypes())
        return value.__class__.__class_getitem__(tuple(hints)) # type: ignore

    if isinstance(value, Series):
        return value.__class__[_value_to_hint(value.index), value.dtype.type().__class__] # type: ignore

    # must come before index
    if isinstance(value, IndexDatetime):
        return value.__class__

    if isinstance(value, Index):
        return value.__class__[value.dtype.type().__class__] # type: ignore

    if isinstance(value, IndexHierarchy):
        hints = list(_value_to_hint(value.index_at_depth(i)) for i in range(value.depth))
        return value.__class__.__class_getitem__(tuple(hints)) # type: ignore

    if isinstance(value, (Bus, Yarn)):
        return value.__class__[_value_to_hint(value.index)] # type: ignore


    if isinstance(value, np.dtype):
        return np.dtype.__class_getitem__(value.type().__class__)

    if isinstance(value, np.ndarray):
        return value.__class__.__class_getitem__(_value_to_hint(value.dtype))

    return value.__class__

#-------------------------------------------------------------------------------
# handlers for getting components out of generics

def iter_sequence_checks(
        value: tp.Any,
        hint: tp.Any,
        parent_hints: TParent,
        parent_values: TParent,
        ) -> tp.Iterable[TValidation]:

    [h_component] = tp.get_args(hint)

    pv_next = parent_values + (value,)
    for v in value:
        yield v, h_component, parent_hints, pv_next

def iter_tuple_checks(
        value: tp.Any,
        hint: tp.Any,
        parent_hints: TParent,
        parent_values: TParent,
        ) -> tp.Iterable[TValidation]:

    h_components = tp.get_args(hint)
    if h_components[-1] is ...:
        if len(h_components) != 2 or h_components[0] is ...:
            yield ERROR_MESSAGE_TYPE, 'Invalid ellipses usage', parent_hints, parent_values
        else: # support any number of values using the same hint
            h = h_components[0]
            pv_next = parent_values + (value,)
            for v in value:
                yield v, h, parent_hints, pv_next
    else:
        if (h_len := len(h_components)) != len(value):
            msg = f'Expected tuple length of {h_len}, provided tuple length of {len(value)}'
            yield ERROR_MESSAGE_TYPE, msg, parent_hints, parent_values

        pv_next = parent_values + (value,)
        for v, h in zip(value, h_components):
            yield v, h, parent_hints, pv_next

def iter_mapping_checks(
        value: tp.Any,
        hint: tp.Any,
        parent_hints: TParent,
        parent_values: TParent,
        ) -> tp.Iterable[TValidation]:

    [h_keys, h_values] = tp.get_args(hint)
    pv_next = parent_values + (value,)

    for v, h in chain(
            zip(value.keys(), repeat(h_keys)),
            zip(value.values(), repeat(h_values)),
            ):
        yield v, h, parent_hints, pv_next


def iter_typeddict_checks(
        value: tp.Any,
        hint: tp.Any,
        parent_hints: TParent,
        parent_values: TParent,
        ) -> tp.Iterable[TValidation]:

    # hint is a typedict class; returns dict of key name and value
    hints = tp.get_type_hints(hint, include_extras=True)
    total = hint.__total__

    pv_next = parent_values + (value,)

    for k, hint_key in hints.items(): # iterating hints retains order
        k_found = k in value
        if not total or (is_generic(hint_key) and tp.get_origin(hint_key) is tp.NotRequired):
            required = False
        else:
            required = True

        if k_found:
            yield (value[k],
                    hint_key,
                    parent_hints + (f'Key {k!r}',),
                    pv_next,
                    )
        elif not k_found and required:
            yield (ERROR_MESSAGE_TYPE,
                    'Expected key not provided',
                    parent_hints + (f'Key {k!r}',),
                    pv_next,
                    )

    # get over-specified keys
    if keys_os := value.keys() - hints.keys():
        yield (ERROR_MESSAGE_TYPE,
                f"Keys provided not expected: {', '.join(f'{k!r}' for k in sorted(keys_os))}",
                parent_hints,
                pv_next,
                )

# NOTE: For SF containers, we create an instance of dtype.type() so as to not modify h_generic, as it might be Union or other generic that cannot be wrapped in a tp.Type. This returns a "sample" instance of the type that can be used for testing. Caching this value does not seem a benefit.

def iter_series_checks(
        value: tp.Any,
        hint: tp.Any,
        parent_hints: TParent,
        parent_values: TParent,
        ) -> tp.Iterable[TValidation]:

    h_index, h_generic = tp.get_args(hint) # there must be two
    pv_next = parent_values + (value,)

    yield value.index, h_index, parent_hints, pv_next
    yield value.dtype.type(), h_generic, parent_hints, pv_next

def iter_index_checks(
        value: tp.Any,
        hint: tp.Any,
        parent_hints: TParent,
        parent_values: TParent,
        ) -> tp.Iterable[TValidation]:

    [h_generic] = tp.get_args(hint)
    pv_next = parent_values + (value,)
    yield value.dtype.type(), h_generic, parent_hints, pv_next

def iter_index_hierarchy_checks(
        value: tp.Any,
        hint: tp.Any,
        parent_hints: TParent,
        parent_values: TParent,
        ) -> tp.Iterable[TValidation]:

    h_generics = tp.get_args(hint)
    h_len = len(h_generics)
    pv_next = parent_values + (value,)

    unpack_pos = -1
    for i, h in enumerate(h_generics):
        # must get_origin to id unpack; origin of simplet types is None
        if is_unpack(tp.get_origin(h), h):
            unpack_pos = i
            break

    if h_len == 1 and unpack_pos == 0:
        [h_tuple] = get_args_unpack(h_generics[0])
        # to support usage of Ellipses, treat this as a hint of a corresponding tuple of types
        yield (tuple(value.index_at_depth(i) for i in range(value.depth)),
                h_tuple,
                parent_hints,
                pv_next,
                )
    else:
        col_count = value.shape[1]

        if unpack_pos == -1: # no unpack
            if h_len != col_count:
                # if no unpack and lengths are not equal
                yield (ERROR_MESSAGE_TYPE,
                        f'Expected IndexHierarchy has {h_len} depth, provided IndexHierarchy has {col_count} depth',
                        parent_hints,
                        pv_next,
                        )
            else:
                yield from zip(
                        (value.index_at_depth(i) for i in range(value.depth)),
                        h_generics,
                        repeat(parent_hints),
                        repeat(pv_next),
                        )

        else: # unpack found
            if h_len > col_count + 1:
                # if 1 unpack, there cannot be more than width h_generics + 1 for the Unpack
                yield (ERROR_MESSAGE_TYPE,
                        f'Expected IndexHierarchy has {h_len - 1} depth (excluding Unpack), provided IndexHierarchy has {col_count} depth',
                        parent_hints,
                        pv_next,
                        )
            else:
                h_types_post = h_len - unpack_pos - 1
                depth_post_unpack = col_count - h_types_post

                indexes = tuple(value.index_at_depth(i) for i in range(value.depth))

                index_pre = indexes[:unpack_pos]
                index_unpack = indexes[unpack_pos: depth_post_unpack]
                index_post = indexes[depth_post_unpack:]

                h_pre = h_generics[:unpack_pos]
                h_unpack = h_generics[unpack_pos]
                h_post = h_generics[unpack_pos + 1:]

                # if len(index_pre) != len(h_pre):
                #     yield ERROR_MESSAGE_TYPE, f'Expected IndexHierarchy has {len(h_pre)} depth before Unpack, provided IndexHierarchy has {len(index_pre)} alignable depth', parent
                # if len(index_post) != len(h_post):
                #     yield ERROR_MESSAGE_TYPE, f'Expected IndexHierarchy has {len(h_post)} depth after Unpack, provided IndexHierarchy has {len(index_post)} alignable depth', parent
                # else:

                col_pos = 0
                for index, h in zip(index_pre, h_pre):
                    yield index, h, parent_hints + (f'Depth {col_pos}',), parent_values
                    col_pos += 1

                if index_unpack: # we may not have dtypes to compare if fewer
                    [h_tuple] = get_args_unpack(h_unpack)
                    yield (tuple(index_unpack),
                            h_tuple,
                            parent_hints + (f'Depths {col_pos} to {depth_post_unpack - 1}',),
                            pv_next,
                            )

                col_pos = depth_post_unpack
                for index, h in zip(index_post, h_post):
                    yield (index,
                            h,
                            parent_hints + (f'Depth {col_pos}',),
                            pv_next,
                            )
                    col_pos += 1


def iter_frame_checks(
        value: tp.Any,
        hint: tp.Any,
        parent_hints: TParent,
        parent_values: TParent,
        ) -> tp.Iterable[TValidation]:

    # NOTE: note: at runtime TypeVarTuple defaults do not return anything
    h_index, h_columns, *h_types = tp.get_args(hint)
    h_types_len = len(h_types)
    pv_next = parent_values + (value,)

    yield value.index, h_index, parent_hints, pv_next
    yield value.columns, h_columns, parent_hints, pv_next

    unpack_pos = -1
    for i, h in enumerate(h_types):
        # must get_origin to id unpack; origin of simplet types is None
        if is_unpack(tp.get_origin(h), h):
            unpack_pos = i
            break

    if h_types_len == 1 and unpack_pos == 0:
        [h_tuple] = get_args_unpack(h_types[0])
        # to support usage of Ellipses, treat this as a hint of a corresponding tuple of types
        yield (tuple(d.type() for d in value._blocks._iter_dtypes()),
                h_tuple,
                parent_hints,
                pv_next,
                )
    else:
        col_count = value.shape[1]
        if unpack_pos == -1: # no unpack
            if h_types_len != col_count:
                # if no unpack and lengths are not equal
                yield (ERROR_MESSAGE_TYPE,
                        f'Expected Frame has {h_types_len} dtype, provided Frame has {col_count} dtype',
                        parent_hints,
                        pv_next,
                        )
            else:
                for dt, h in zip(value._blocks._iter_dtypes(), h_types):
                    yield dt.type(), h, parent_hints, pv_next

        else: # unpack found
            if h_types_len > col_count + 1:
                # if 1 unpack, there cannot be more than width h_types + 1 for the Unpack
                yield (ERROR_MESSAGE_TYPE,
                        f'Expected Frame has {h_types_len - 1} dtype (excluding Unpack), provided Frame has {col_count} dtype',
                        parent_hints,
                        pv_next,
                        )
            else:
                # in terms of column position, we need to find the first column position after the unpack
                # if unpack pos is 0, and 5 types, 4 are post
                # if unpack pos is 3, and 5 types, 1 is post
                h_types_post = h_types_len - unpack_pos - 1
                col_post_unpack = col_count - h_types_post

                dts = tuple(value._blocks._iter_dtypes())

                dt_pre = dts[:unpack_pos]
                dt_unpack = dts[unpack_pos: col_post_unpack]
                dt_post = dts[col_post_unpack:]

                h_pre = h_types[:unpack_pos]
                h_unpack = h_types[unpack_pos]
                h_post = h_types[unpack_pos + 1:]

                # if len(dt_pre) != len(h_pre):
                #     yield ERROR_MESSAGE_TYPE, f'Expected Frame has {len(h_pre)} dtype before Unpack, provided Frame has {len(dt_pre)} alignable dtype', parent
                # elif len(dt_post) != len(h_post):
                #     yield ERROR_MESSAGE_TYPE, f'Expected Frame has {len(h_post)} dtype after Unpack, provided Frame has {len(dt_post)} alignable dtype', parent
                # else:

                col_pos = 0
                for dt, h in zip(dt_pre, h_pre):
                    yield (dt.type(),
                            h,
                            parent_hints + (f'Field {col_pos}',),
                            pv_next,
                            )
                    col_pos += 1

                if dt_unpack: # we may not have dtypes to compare if fewer
                    [h_tuple] = get_args_unpack(h_unpack)
                    yield (tuple(d.type() for d in dt_unpack),
                            h_tuple,
                            parent_hints + (f'Fields {col_pos} to {col_post_unpack - 1}',),
                            pv_next,
                            )

                col_pos = col_post_unpack
                for dt, h in zip(dt_post, h_post):
                    yield (dt.type(),
                            h,
                            parent_hints + (f'Field {col_pos}',),
                            pv_next,
                            )
                    col_pos += 1


def iter_bus_checks(
        value: tp.Any,
        hint: tp.Any,
        parent_hints: TParent,
        parent_values: TParent,
        ) -> tp.Iterable[TValidation]:

    [h_index] = tp.get_args(hint) # there must be one
    pv_next = parent_values + (value,)
    yield value.index, h_index, parent_hints, pv_next

def iter_yarn_checks(
        value: tp.Any,
        hint: tp.Any,
        parent_hints: TParent,
        parent_values: TParent,
        ) -> tp.Iterable[TValidation]:

    [h_index] = tp.get_args(hint) # there must be one
    pv_next = parent_values + (value,)
    yield value.index, h_index, parent_hints, pv_next

def iter_ndarray_checks(
        value: tp.Any,
        hint: tp.Any,
        parent_hints: TParent,
        parent_values: TParent,
        ) -> tp.Iterable[TValidation]:

    h_shape, h_dtype = tp.get_args(hint)
    pv_next = parent_values + (value,)
    yield value.dtype, h_dtype, parent_hints, pv_next

def iter_dtype_checks(
        value: tp.Any,
        hint: tp.Any,
        parent_hints: TParent,
        parent_values: TParent,
        ) -> tp.Iterable[TValidation]:

    [h_generic] = tp.get_args(hint)
    pv_next = parent_values + (value,)
    yield value.type(), h_generic, parent_hints, pv_next

def iter_np_generic_checks(
        value: tp.Any,
        hint: tp.Any,
        parent_hints: TParent,
        parent_values: TParent,
        ) -> tp.Iterable[TValidation]:
    # we have already confirmed that value is an instance of the origin type
    h_components = tp.get_args(hint)
    pv_next = parent_values + (value,)

    # there are two components we have a complexfloating
    for h_component in h_components:
        yield value, h_component, parent_hints, pv_next

BIT_TO_LITERAL = {
    8: tp.Literal[8],
    16: tp.Literal[16],
    32: tp.Literal[32],
    64: tp.Literal[64],
    80: tp.Literal[80],
    96: tp.Literal[96],
    128: tp.Literal[128],
    256: tp.Literal[256],
    # 512: tp.Literal[512], # not defined in numpy/_typing/__init__.py
}

def iter_np_nbit_checks(
        value: tp.Any,
        hint: tp.Any,
        parent_hints: TParent,
        parent_values: TParent,
        ) -> tp.Iterable[TValidation]:
    if not isinstance(value, np.generic):
        pv_next = parent_values + (value,)
        yield (ERROR_MESSAGE_TYPE,
                f'Expected {hint.__name__}, provided value is not an np.generic',
                parent_hints,
                pv_next,
                )
    else:
        v_bits = value.dtype.itemsize * 8
        pv_next = parent_values + (v_bits,)
        # NumPy uses __init_subclass__ to limit class names to those with a the bit number in the name
        h_bits = int(''.join(c for c in hint.__name__ if c.isdecimal()))

        if value.dtype.kind == DTYPE_COMPLEX_KIND:
            # complex is represented with two hints, each half of the whole itemsize; adjust value bits and let each side check independently
            yield v_bits // 2, BIT_TO_LITERAL[h_bits], parent_hints, pv_next
        else:
            yield v_bits, BIT_TO_LITERAL[h_bits], parent_hints, pv_next

#-------------------------------------------------------------------------------
class TypeVarState:
    __slots__ = (
            '_var',
            '_value',
            '_bound',
            '_bound_unset',
            '_is_bound_union',
            )

    def __init__(self, var: tp.Any, value: tp.Any):
        '''Provide `TypeVar` and and the first-encountered value for that `TypeVar`
        '''
        self._var = var
        self._value = value
        # as bounds might have unions that need specialization, we store the current bound type here and update it if needed
        self._bound = var.__bound__ # might be None
        if self._bound is not None and is_union(self._bound):
            self._is_bound_union = True
            self._bound_unset = set(range(len(tp.get_args(self._bound))))
        else:
            self._is_bound_union = False

    @property
    def is_bound_union(self) -> bool:
        return self._is_bound_union

    def _specialize_bound_union(self,
            value: tp.Any,
            ) -> None:
        '''If `hint` is a Union, given value that is assigned to the type var, find value in the Union and replace that hint with the hint of the value.
        '''
        components = list(tp.get_args(self._bound))
        # NOTE: this refence to a set is mutated inplace
        for i, hint in enumerate(components):
            if i in self._bound_unset and _check(value, hint).validated:
                components[i] = _value_to_hint(value)
                self._bound_unset.discard(i)
        self._bound = tp.Union.__getitem__(tuple(components)) # pyright: ignore


    @property
    def constraints(self) -> tp.Any:
        return self._var.__constraints__

    def get_hint(self, value: tp.Any) -> tp.Any:
        '''Return a hint from derived from the stored value. If this is a bound union, the value is used to specialize the union.
        '''
        if self.is_bound_union:
            self._specialize_bound_union(value)
            return self._bound

        return _value_to_hint(self._value) # this could be stored on init


class TypeVarRegistry:
    __slots__ = (
            '_id_to_var',
            )
    _id_to_var: tp.Dict[tp.TypeVar, TypeVarState]

    def __init__(self) -> None:
        self._id_to_var = dict()

    def iter_checks(self,
            value: tp.Any,
            var: tp.TypeVar,
            parent_hints: TParent,
            parent_values: TParent,
            ) -> tp.Iterator[TValidation]:

        pv_next = parent_values + (value,)

        if var not in self._id_to_var:
            tvs = TypeVarState(var, value)
            self._id_to_var[var] = tvs
            if hints := tvs.constraints:
                # with constratings we select one option and use it for the life of the Typevar; on the first value, check that the value meets the constraints (recast as a uion); subsequent checks will be based on the stored value
                yield value, tp.Union.__getitem__(hints), parent_hints, pv_next # pyright: ignore
        else:
            tvs = self._id_to_var[var]

        yield value, tvs.get_hint(value), parent_hints, pv_next

#-------------------------------------------------------------------------------

def _check(
        value: tp.Any,
        hint: tp.Any,
        tvr: tp.Optional[TypeVarRegistry] = None,
        parent_hints: TParent = (),
        parent_values: TParent = (),
        fail_fast: bool = False,
        ) -> ClinicResult:

    # Check queue: queue all checks
    q = deque(((value, hint, parent_hints, parent_values),))

    # Error log: any entry is considered an error
    e_log: tp.List[TValidation] = []

    def tee_error_or_check(records: tp.Iterable[TValidation]) -> None:
        for record in records:
            if record[0] is ERROR_MESSAGE_TYPE:
                e_log.append(record)
            else:
                q.append(record)

    while q:
        if fail_fast and e_log:
            return ClinicResult(e_log)

        v, h, ph, pv = q.popleft()
        # an ERROR_MESSAGE_TYPE should only be used as a place holder in error logs, not queued checks
        assert v is not ERROR_MESSAGE_TYPE
        # print(v, h, ph, pv)

        if h is tp.Any:
            continue

        ph_next = ph + (h,)
        if is_union(h):
            # NOTE: must check union types first as tp.Union matches as generic type
            u_log: tp.List[TValidation] = []
            for c_hint in tp.get_args(h): # get components
                # handing one pair at a time with a secondary call will allow nested types in the union to be evaluated on their own
                c_log = _check(v, c_hint, tvr, ph_next, pv, fail_fast)
                if not c_log: # no error found, can exit
                    break
                u_log.extend(c_log)
            else: # no breaks, so no matches within union
                e_log.extend(u_log)

        elif isinstance(h, Validator):
            e_log.extend(h._iter_errors(v, h, ph_next, pv))

        elif isinstance(h, tp.TypeVar):
            if tvr is not None:
                tee_error_or_check(tvr.iter_checks(v, h, ph_next, pv))
            else: # ignore typevar
                continue

        elif is_generic(h):
            origin = tp.get_origin(h)

            if origin is type: # a tp.Type[x] generic
                [t] = tp.get_args(h)
                try: # the v should be a subclass of t
                    t_check = issubclass(t, v)
                except TypeError:
                    t_check = False
                if t_check:
                    continue
                e_log.append((v, h, ph, pv))

            elif origin == tp.Annotated: # NOTE: cannot use `is` due backwards compat
                h_type, *h_annotations = tp.get_args(h)
                # perform the un-annotated check
                q.append((v, h_type, ph_next, pv))
                for h_annotation in h_annotations:
                    if isinstance(h_annotation, Validator):
                        q.append((v, h_annotation, ph_next, pv))

            elif origin == tp.Literal: # NOTE: cannot use `is` due backwards compat
                l_log: tp.List[TValidation] = []
                for l_hint in tp.get_args(h): # get components
                    c_log = _check(v, l_hint, tvr, ph_next, pv, fail_fast)
                    if not c_log: # no error found, can exit
                        break
                    l_log.extend(c_log)
                else: # no breaks, so no matches within literal
                    e_log.extend(l_log)

            elif origin == tp.Required or origin == tp.NotRequired:
                # semantics handled elsewhere, just unpack
                [h_type] = tp.get_args(h)
                q.append((v, h_type, ph_next, pv))

            else:
                # NOTE: many generic containers require recursing into component values. It is in these functions below that parent_values is updated and yielded back into the queue. There are many other cases where parent_values does not need to be updated (and for efficiency is not).
                if not isinstance(v, origin):
                    e_log.append((v, origin, ph, pv))
                    continue

                if isinstance(v, tuple):
                    tee_error_or_check(iter_tuple_checks(v, h, ph_next, pv))
                elif isinstance(v, Sequence):
                    tee_error_or_check(iter_sequence_checks(v, h, ph_next, pv))
                elif isinstance(v, MutableMapping):
                    tee_error_or_check(iter_mapping_checks(v, h, ph_next, pv))

                elif isinstance(v, Index):
                    tee_error_or_check(iter_index_checks(v, h, ph_next, pv))
                elif isinstance(v, IndexHierarchy):
                    tee_error_or_check(iter_index_hierarchy_checks(v, h, ph_next, pv))
                elif isinstance(v, Series):
                    tee_error_or_check(iter_series_checks(v, h, ph_next, pv))
                elif isinstance(v, Frame):
                    tee_error_or_check(iter_frame_checks(v, h, ph_next, pv))
                elif isinstance(v, Bus):
                    tee_error_or_check(iter_bus_checks(v, h, ph_next, pv))
                elif isinstance(v, Yarn):
                    tee_error_or_check(iter_yarn_checks(v, h, ph_next, pv))

                elif isinstance(v, np.ndarray):
                    tee_error_or_check(iter_ndarray_checks(v, h, ph_next, pv))
                elif isinstance(v, np.dtype):
                    tee_error_or_check(iter_dtype_checks(v, h, ph_next, pv))
                elif isinstance(v, np.generic):
                    tee_error_or_check(iter_np_generic_checks(v, h, ph_next, pv))
                else:
                    raise NotImplementedError(f'no handling for generic {origin}') #pragma: no cover
        elif tp.is_typeddict(h):
            tee_error_or_check(iter_typeddict_checks(v, h, ph_next, pv))

        elif not isinstance(h, type): # h is a value from a literal
            # must check type: https://peps.python.org/pep-0586/#equivalence-of-two-literals
            if type(v) != type(h) or v != h: # pylint: disable=C0123
                e_log.append((v, h, ph, pv))

        # h is a class
        elif issubclass(h, NBitBase):
            tee_error_or_check(iter_np_nbit_checks(v, h, ph_next, pv))

        else: # h is non-generic type, must continue if valid
            if v.__class__ is bool:
                if h is bool:
                    continue
            elif h is np.object_ and v is None:
                # when comparing object dtypes our test value is None, which is not an instance of np.object_
                continue
            # general case
            elif isinstance(v, h):
                continue
            e_log.append((v, h, ph, pv))

    return ClinicResult(e_log)

#-------------------------------------------------------------------------------
# public interfaces

class TypeClinic:
    '''A ``TypeClinic`` instance, created from (almost) any object, can be used to derive a type hint (or type hint string), or test the object against a provided hint.
    '''

    __slots__ = ('_value',)

    _INTERFACE = (
        'to_hint',
        'check',
        'warn',
        '__call__',
        '__repr__',
        )

    def __init__(self, value: tp.Any, /):
        self._value = value

    def to_hint(self) -> tp.Any:
        '''Return the type hint (the type and/or generic aliases necessary) to represent the object given at initialization.
        '''
        # NOTE: this can cache as value assumed immutable
        return _value_to_hint(self._value)

    def __repr__(self) -> str:
        '''Return a compact string representation of the type hint (the type and/or generic aliases necessary) to represent the object given at initialization.
        '''
        return to_name(self.to_hint())

    def check(self,
            hint: tp.Any,
            /, *,
            fail_fast: bool = False,
            ) -> None:
        '''Given a hint (a type and/or generic alias), raise a ``ClinicError`` exception describing the result of the check if an error is found.

        Args:
            fail_fast: If True, return on first failure. If False, all failures are discovered and reported.
        '''
        if cr := self(hint, fail_fast=fail_fast):
            raise ClinicError(cr)

    def warn(self,
            hint: tp.Any,
            /, *,
            fail_fast: bool = False,
            category: tp.Type[Warning] = UserWarning,
            ) -> None:
        '''Given a hint (a type and/or generic alias), issue a warning describing the result of the check if an error is found.

        Args:
            fail_fast: If True, return on first failure. If False, all failures are discovered and reported.
            category: The ``Warning`` subclass to be used for issueing the warning.
        '''
        if cr := self(hint, fail_fast=fail_fast):
            warnings.warn(cr.to_str(), category)


    def __call__(self,
            hint: tp.Any,
            /, *,
            fail_fast: bool = False,
            ) -> ClinicResult:
        '''Given a hint (a type and/or generic alias), return a ``ClinicResult`` object describing the result of the check.

        Args:
            fail_fast: If True, return on first failure. If False, all failures are discovered and reported.
        '''
        tvr = TypeVarRegistry()
        post = _check(self._value, hint, tvr, fail_fast=fail_fast)
        return post


class ErrorAction(Enum):
    RAISE = 0
    WARN = 1
    RETURN = 2

def _check_interface(
        func: tp.Callable[..., tp.Any],
        args: tp.Any,
        kwargs: tp.Any,
        fail_fast: bool,
        error_action: ErrorAction,
        category: tp.Type[Warning] = UserWarning,
        ) -> tp.Any:
    # include_extras insures that Annotated generics are returned
    hints = tp.get_type_hints(func, include_extras=True)

    sig = Signature.from_callable(func)
    sig_bound = sig.bind(*args, **kwargs)
    sig_bound.apply_defaults()
    sig_str = to_signature(sig_bound, hints)
    parent_hints = (f'args of {sig_str}',)
    parent_values = (func,)

    # NOTE: we create one TV registry per check, so state associated with typevars will be bound by the context of one function
    tvr = TypeVarRegistry()

    for k, v in sig_bound.arguments.items():
        if h_p := hints.get(k, None):
            arg_hints = parent_hints + (f'In arg {k}',)
            if cr := _check(v, h_p, tvr, arg_hints, parent_values, fail_fast=fail_fast):
                if error_action is ErrorAction.RAISE:
                    raise ClinicError(cr)
                elif error_action is ErrorAction.WARN:
                    warnings.warn(cr.to_str(), category)
                elif error_action is ErrorAction.RETURN:
                    return cr

    post = func(*args, **kwargs)

    if h_return := hints.get('return', None):
        if cr := _check(post,
                h_return,
                tvr,
                (f'return of {sig_str}',),
                parent_values,
                fail_fast=fail_fast,
                ):
            if error_action is ErrorAction.RAISE:
                raise ClinicError(cr)
            elif error_action is ErrorAction.WARN:
                warnings.warn(cr.to_str(), category)
            elif error_action is ErrorAction.RETURN:
                return cr

    return post


TVFunc = tp.TypeVar('TVFunc', bound=tp.Callable[..., tp.Any])

class CallGuard:
    '''A family of decorators for run-time type checking and data validation of functions.
    '''

    @tp.overload
    @staticmethod
    def check(func: TVFunc, /) -> TVFunc: ...

    @tp.overload
    @staticmethod
    def check(*, fail_fast: bool) -> tp.Callable[[TVFunc], TVFunc]: ...

    @tp.overload
    @staticmethod
    def check(func: None, /, *, fail_fast: bool) -> tp.Callable[[TVFunc], TVFunc]: ...

    @staticmethod
    def check(
            func: TVFunc | None = None,
            /, *,
            fail_fast: bool = False,
            ) -> tp.Any:
        '''A function decorator to perform run-time checking of function arguments and return values based on the function type annotations, including type hints and ``Require``-provided validators. Raises ``ClinicError`` on failure.
        '''
        def decorator(func: TVFunc) -> TVFunc:
            @wraps(func)
            def wrapper(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
                return _check_interface(func,
                        args,
                        kwargs,
                        fail_fast,
                        ErrorAction.RAISE,
                        )
            return tp.cast(TVFunc, wrapper)

        if func is not None:
            return decorator(func)
        return decorator


    @tp.overload
    @staticmethod
    def warn(func: TVFunc, /) -> TVFunc: ...

    @tp.overload
    @staticmethod
    def warn(*, fail_fast: bool, category: tp.Type[Warning]) -> tp.Callable[[TVFunc], TVFunc]: ...

    @tp.overload
    @staticmethod
    def warn(func: None, /, *, fail_fast: bool, category: tp.Type[Warning]) -> tp.Callable[[TVFunc], TVFunc]: ...

    @staticmethod
    def warn(
            func: TVFunc | None = None,
            /, *,
            fail_fast: bool = False,
            category: tp.Type[Warning] = UserWarning,
            ) -> tp.Any:
        '''A function decorator to perform run-time checking of function arguments and return values based on the function type annotations, including type hints and ``Require``-provided validators. Issues a warning on failure.
        '''
        def decorator(func: TVFunc) -> TVFunc:
            @wraps(func)
            def wrapper(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
                return _check_interface(func,
                        args,
                        kwargs,
                        fail_fast,
                        ErrorAction.WARN,
                        category,
                        )
            return tp.cast(TVFunc, wrapper)

        if func is not None:
            return decorator(func)
        return decorator

