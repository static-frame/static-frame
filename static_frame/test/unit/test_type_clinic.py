# pylint: disable=C0321
# from __future__ import annotations # cannot use with TypeVars!

import re
import warnings
from functools import partial

import frame_fixtures as ff
import numpy as np
import pytest
import typing_extensions as tp
from numpy.typing import NBitBase

import static_frame as sf
from static_frame.core.type_clinic import CallGuard
from static_frame.core.type_clinic import ClinicResult
from static_frame.core.type_clinic import ErrorAction
from static_frame.core.type_clinic import Require
from static_frame.core.type_clinic import TValidation
from static_frame.core.type_clinic import TypeClinic
from static_frame.core.type_clinic import TypeVarRegistry
from static_frame.core.type_clinic import _check
from static_frame.core.type_clinic import _check_interface
from static_frame.core.type_clinic import is_union
from static_frame.core.type_clinic import is_unpack
from static_frame.test.test_case import IS_NP2
from static_frame.test.test_case import skip_np_no_float128
from static_frame.test.test_case import skip_pyle310
from static_frame.test.test_case import skip_win

#-------------------------------------------------------------------------------
# recreate private nbit types for testing

class _256Bit(NBitBase):  # type: ignore[misc] # pylint: disable=W0240
    pass
class _128Bit(_256Bit):  # type: ignore[misc]
    pass
class _96Bit(_128Bit):  # type: ignore[misc]
    pass
class _80Bit(_96Bit):  # type: ignore[misc]
    pass
class _64Bit(_80Bit):  # type: ignore[misc]
    pass
class _32Bit(_64Bit):  # type: ignore[misc]
    pass
class _16Bit(_32Bit):  # type: ignore[misc]
    pass
class _8Bit(_16Bit):  # type: ignore[misc]
    pass

# complex partitions are always balanced

# complex160 = np.complexfloating[_80Bit, _80Bit]
# complex192 = np.complexfloating[_96Bit, _96Bit]
# complex256 = np.complexfloating[_128Bit, _128Bit]
# complex512 = np.complexfloating[_256Bit, _256Bit]

#-------------------------------------------------------------------------------
def test_check_a():
    cr = _check([3, 'a'], tp.List[tp.Union[str, int]])
    assert cr.validated

def test_check_b():
    T = tp.TypeVar('T')
    tvr = TypeVarRegistry()
    cr1 = _check((3, 'a'), tp.Tuple[T, T], tvr=tvr)
    assert not cr1.validated
    assert scrub_str(cr1.to_str()) == 'In Tuple[~T, ~T] ~T Expected int, provided str invalid'

    # we can ignore TypeVars
    cr2 = _check((3, 'a'), tp.Tuple[T, T], tvr=None)
    assert cr2.validated

#-------------------------------------------------------------------------------

def test_check_type_a():

    TypeClinic(sf.IndexDate(('2022-01-01',))).check(sf.IndexDate)
    TypeClinic(sf.IndexDate(('2022-01-01',))).check(tp.Any)

    with pytest.raises(TypeError):
        TypeClinic(sf.IndexDate(('2022-01-01',))).check(sf.IndexSecond)


def test_check_type_b():

    TypeClinic(3).check(int)
    TypeClinic('foo').check(str)
    TypeClinic(False).check(bool)

    with pytest.raises(TypeError):
        TypeClinic(3).check(str)

    with pytest.raises(TypeError):
        TypeClinic(True).check(int)


def test_check_type_c():
    TypeClinic(['a', 'b']).check(tp.List[str])
    TypeClinic([{3: 'a'}, {4: 'x'}]).check(tp.List[tp.Dict[int, str]])
    TypeClinic([(3, ['a', 'b']), (4, ['x', 'y'])]).check(tp.List[tp.Tuple[int, tp.List[str]]])

    tc = TypeClinic([(3, ['a', 'b']), (4, ['x', 'y'])])
    with pytest.raises(TypeError):
        tc.check(tp.List[tp.Tuple[int, tp.List[bool]]])

    cr = tc(tp.List[tp.Tuple[int, tp.List[bool]]])
    assert len(cr) == 4


def test_check_type_d1():

    class Record1(tp.TypedDict):
        a: int
        b: float
        c: str

    class Record2(tp.TypedDict):
        a: int
        b: float
        c: bool

    tc = TypeClinic(dict(a=3, b=10.5, c='foo'))
    tc.check(Record1)

    cr = tc(Record2)
    assert not cr.validated

    assert scrub_str(cr.to_str()) == "In Record2 Key 'c' Expected bool, provided str invalid"

def test_check_type_d2():

    class Record1(tp.TypedDict, total=False):
        a: int
        b: float
        c: str

    class Record2(tp.TypedDict):
        a: int
        b: float
        c: str

    tc1 = TypeClinic(dict(a=3, b=10.5))
    assert tc1(Record1).validated

    tc2 = TypeClinic(dict(a=3, b=10.5, c='foo'))
    assert tc2(Record2).validated

    # tc3 = TypeClinic(dict(a=3, b=10.5))
    assert not tc1(Record2).validated
    assert scrub_str(tc1(Record2).to_str()) == "In Record2 Key 'c' Expected key not provided"

def test_check_type_d3():
    class Record1(tp.TypedDict, total=False):
        a: int
        b: float

    tc1 = TypeClinic(dict(a=3, b=10.5, c='foo', d=False))
    cr = tc1(Record1)
    assert scrub_str(cr.to_str()) == "In Record1 Keys provided not expected: 'c', 'd'"


def test_check_type_d4():
    Record1 = tp.TypedDict('Record1', dict(a=int, b=float, c=str))
    Record2 = tp.TypedDict('Record2', dict(a=int, b=float, c=bool))

    tc = TypeClinic(dict(a=3, b=10.5, c='foo'))
    tc.check(Record1)

    cr = tc(Record2)
    assert not cr.validated
    assert scrub_str(cr.to_str()) == "In Record2 Key 'c' Expected bool, provided str invalid"


def test_check_type_d5():

    class Record1(tp.TypedDict):
        a: tp.Required[int]
        b: tp.Required[float]
        c: tp.NotRequired[str]

    class Record2(tp.TypedDict):
        a: int
        b: float
        c: tp.NotRequired[str]

    tc1 = TypeClinic(dict(a=3, b=10.5))
    assert tc1(Record1).validated
    assert tc1(Record2).validated

    tc2 = TypeClinic(dict(a=3, b=10.5, c='foo'))
    assert tc2(Record1).validated
    assert tc2(Record2).validated

    # tc3 = TypeClinic(dict(b=10.5, c='foo'))
    # assert not tc3.check(Record2).validated
    # assert scrub_str(tc1.check(Record2).to_str()) == "In Record2 Key 'c' Expected key not provided"


#-------------------------------------------------------------------------------

def test_is_unpack_a():
    assert is_unpack(tp.Unpack, None)
    assert not is_unpack(None, None)

def test_is_union_a():
    assert is_union(tp.Union[int, str])
    assert not is_union(tp.Tuple[str, str])
    assert not is_union(str)

#-------------------------------------------------------------------------------

def test_check_result_a():
    assert ClinicResult([]).validated

def test_check_result_b():
    try:
        post = TypeClinic((3, 'x'))(tp.Tuple[int, str, ...]).to_str()
        assert scrub_str(post) == 'In Tuple[int, str, ...] Invalid ellipses usage'
    except TypeError:
        pass

def test_check_result_c():
    post = TypeClinic(sf.Index(('a', 'b')))(tp.Annotated[sf.Index[np.str_], Require.Len(1)]).to_str()
    assert scrub_str(post) == 'In Annotated[Index[str_], Len(1)] Len(1) Expected length 1, provided length 2'


#-------------------------------------------------------------------------------

def test_check_type_union_a():

    TypeClinic(3).check(tp.Union[int, str])

    with pytest.raises(TypeError):
        TypeClinic('x').check(tp.Union[int, float])

    TypeClinic('x').check(tp.Union[str, bytes])
    TypeClinic('x').check(tp.Union[int, str])


#-------------------------------------------------------------------------------

def test_check_type_type_a():

    TypeClinic(sf.Series).check(tp.Type[sf.Series])

    with pytest.raises(TypeError):
        TypeClinic(sf.Series).check(tp.Type[sf.Index])

@skip_pyle310
def test_check_type_type_b():
    try:
        TypeClinic(3).check(tp.Type[sf.Series])
    except TypeError as e:
        assert str(e).replace('\n', '') == 'Expected Type[Series], provided int invalid'

#-------------------------------------------------------------------------------

@skip_pyle310
def test_validate_numpy_a():
    v = np.array([False, True, False])
    h1 = np.ndarray[tp.Any, np.dtype[np.bool_]]
    h2 = np.ndarray[tp.Any, np.dtype[np.str_]]

    TypeClinic(v).check(h1)
    with pytest.raises(TypeError):
        TypeClinic(v).check(h2)

#-------------------------------------------------------------------------------

def test_check_type_numpy_a():
    a = np.array([2, 4], dtype=np.int32)
    h1 = np.ndarray[tp.Any, np.dtype[np.signedinteger[tp.Any]]]
    TypeClinic(a).check(h1)

    h2 = np.ndarray[tp.Any, np.dtype[np.unsignedinteger[tp.Any]]]
    with pytest.raises(TypeError):
        TypeClinic(a).check(h2)

def test_check_type_numpy_b():
    a = np.array([2, 4], dtype=np.int32)
    h1 = np.ndarray[tp.Any, np.dtype[np.signedinteger[np.integer[tp.Any]]]]
    TypeClinic(a).check(h1)

    h2 = np.ndarray[tp.Any, np.dtype[np.floating[tp.Any]]]
    with pytest.raises(TypeError):
        TypeClinic(a).check(h2)

@skip_pyle310
def test_check_type_numpy_c():
    a = np.array([2, 4], dtype=np.int32)
    # NOTE: need np.typing.NBitBase
    h1 = np.ndarray[tp.Any, np.dtype[np.signedinteger[np.integer[_32Bit]]]]
    TypeClinic(a).check(h1)

    h2 = np.ndarray[tp.Any, np.dtype[np.signedinteger[np.integer[_16Bit]]]]
    try:
        TypeClinic(a).check(h2)
    except TypeError as e:
        assert str(e).replace('\n', '') == 'In ndarray[Any, dtype[signedinteger[integer[_16Bit]]]]└── dtype[signedinteger[integer[_16Bit]]]    └── signedinteger[integer[_16Bit]]        └── integer[_16Bit]            └── _16Bit                └── Literal[16]                    └── Expected 16, provided int invalid'

def test_check_type_numpy_d():
    a = np.array([2, 4], dtype=np.int32)

    h1 = np.ndarray[tp.Any, np.dtype[np.signedinteger[np.integer[np.number[tp.Union[_16Bit, _32Bit]]]]]]
    TypeClinic(a).check(h1)

    h2 = np.ndarray[tp.Any, np.dtype[np.signedinteger[np.integer[np.number[tp.Union[_16Bit, _64Bit]]]]]]
    with pytest.raises(TypeError):
        TypeClinic(a).check(h2)


def test_check_type_numpy_e():
    a = np.array([2.2, 4.2], dtype=np.float64)
    h1 = np.ndarray[tp.Any, np.dtype[np.inexact[tp.Any]]]
    TypeClinic(a).check(h1)

    h2 = np.ndarray[tp.Any, np.dtype[np.floating[tp.Any]]]
    TypeClinic(a).check(h2)

    h3 = np.ndarray[tp.Any, np.dtype[np.complexfloating[tp.Any, tp.Any]]]
    with pytest.raises(TypeError):
        TypeClinic(a).check(h3)

def test_check_type_numpy_f():
    a = np.array([2.2, 4.2], dtype=np.float64)
    h1 = np.ndarray[tp.Any, np.dtype[np.floating[np.inexact[np.number[_64Bit]]]]]
    TypeClinic(a).check(h1)

    h2 = np.ndarray[tp.Any, np.dtype[np.floating[np.inexact[np.number[_32Bit]]]]]
    with pytest.raises(TypeError):
        TypeClinic(a).check(h2)

@skip_np_no_float128
def test_check_type_numpy_g1():
    a = np.array([2.2, 4.2], dtype=np.complex256)
    h1 = np.ndarray[tp.Any, np.dtype[np.complexfloating[_128Bit, _128Bit]]]
    TypeClinic(a).check(h1)

    h2 = np.ndarray[tp.Any, np.dtype[np.complexfloating[_64Bit, _128Bit]]]
    with pytest.raises(TypeError):
        TypeClinic(a).check(h2)

    h3 = np.ndarray[tp.Any, np.dtype[np.complexfloating[_128Bit, _64Bit]]]
    with pytest.raises(TypeError):
        TypeClinic(a).check(h3)

def test_check_type_numpy_g2():
    a = np.array([2.2, 4.2], dtype=np.complex128)
    h1 = np.ndarray[tp.Any, np.dtype[np.complexfloating[_64Bit, _64Bit]]]
    TypeClinic(a).check(h1)

    h2 = np.ndarray[tp.Any, np.dtype[np.complexfloating[_32Bit, _64Bit]]]
    with pytest.raises(TypeError):
        TypeClinic(a).check(h2)

    h3 = np.ndarray[tp.Any, np.dtype[np.complexfloating[_64Bit, _32Bit]]]
    with pytest.raises(TypeError):
        TypeClinic(a).check(h3)

def test_check_type_numpy_h():
    a = np.array([2.2, 4.2], dtype=np.complex128)
    h1 = np.ndarray[tp.Any, np.dtype[np.inexact[tp.Any]]]
    TypeClinic(a).check(h1)

    h2 = np.ndarray[tp.Any, np.dtype[np.signedinteger[tp.Any]]]
    with pytest.raises(TypeError):
        TypeClinic(a).check(h2)

@skip_np_no_float128
def test_check_type_numpy_i():
    a = np.array([2.2, 4.2], dtype=np.complex256)
    h1 = np.ndarray[tp.Any, np.dtype[np.complex256]]
    TypeClinic(a).check(h1)

    h2 = np.ndarray[tp.Any, np.dtype[np.complex128]]
    with pytest.raises(TypeError):
        TypeClinic(a).check(h2)



#-------------------------------------------------------------------------------

def test_check_type_nbit_a():
    TypeClinic(np.int64()).check(_64Bit) # pylint: disable=E1120
    with pytest.raises(TypeError):
        TypeClinic(np.int8()).check(_32Bit) # pylint: disable=E1120
    with pytest.raises(TypeError):
        TypeClinic(np.int32()).check(_8Bit) # pylint: disable=E1120


def test_check_type_nbit_b():
    with pytest.raises(TypeError):
        TypeClinic(0).check(_64Bit)

#-------------------------------------------------------------------------------

def test_check_type_containers_a():
    v = sf.Series(('a', 'b'), index=sf.IndexDate(('2021-04-05', '2022-05-03')))
    h1 = sf.SeriesHE[sf.IndexDate, np.str_]
    h2 = sf.Index[np.str_]
    h3 = sf.Series[sf.IndexDate, np.str_]

    with pytest.raises(TypeError):
        TypeClinic(v).check(h2)

    with pytest.raises(TypeError):
        TypeClinic(v).check(h1)

def test_check_type_containers_b():
    v = sf.Series(('a', 'b'), index=sf.IndexDate(('2021-04-05', '2022-05-03')))
    h1 = sf.Series[sf.IndexDate, np.str_]
    h2 = sf.Series[sf.IndexDate, np.int64]

    TypeClinic(v).check(h1)
    with pytest.raises(TypeError):
        TypeClinic(v).check(h2)


def test_check_type_containers_c():
    v = sf.Series(('a', 'b'), index=sf.Index((10, 20), dtype=np.int64))
    h1 = sf.Series[sf.Index[np.str_], np.str_]
    h2 = sf.Series[sf.IndexDate, np.int64]
    h3 = sf.Series[sf.Index[np.str_], np.int64]
    h4 = sf.Series[sf.Index[np.int64], np.str_]

    with pytest.raises(TypeError):
        TypeClinic(v,).check(h1)
    with pytest.raises(TypeError):
        TypeClinic(v,).check(h2)
    with pytest.raises(TypeError):
        TypeClinic(v,).check(h3)

    TypeClinic(v,).check(h4)


def test_check_type_containers_d():
    v1 = sf.Series(('a', 'b'), index=sf.Index((10, 20), dtype=np.int64))
    v2 = sf.Series(('a', 'b'), index=sf.Index((10, 20), dtype=np.str_))
    v3 = sf.Series(('a', 'b'), index=sf.Index((1, 0), dtype=np.bool_))

    h1 = sf.Series[sf.Index[tp.Union[np.int64, np.str_]], np.str_]

    TypeClinic(v1).check(h1)
    TypeClinic(v2).check(h1)
    with pytest.raises(TypeError):
        TypeClinic(v3).check(h1)


def test_check_type_containers_e():
    v1 = sf.Series(('a', 'b'), index=sf.IndexDate(('2021-04-05', '2022-05-03')))
    v2 = sf.Series(('a', 'b'), index=sf.IndexSecond(('2021-04-05', '2022-05-03')))
    v3 = sf.Series(('a', 'b'), index=sf.Index(('x', 'y')))

    h1 = sf.Series[tp.Union[sf.IndexDate, sf.IndexSecond], np.str_]

    TypeClinic(v1).check(h1)
    TypeClinic(v2).check(h1)

    with pytest.raises(TypeError):
        TypeClinic(v3).check(h1)


#-------------------------------------------------------------------------------

def scrub_str(s: str) -> str:
    s = s.replace('\n', ' '
            ).replace(ClinicResult._LINE, ''
            ).replace(ClinicResult._CORNER, ''
            ).replace('tuple[', 'Tuple[') # normalize tuple presentation
    while '  ' in s:
        s = s.replace('  ', ' ')
    return s.strip()

@skip_pyle310
def test_check_type_fail_fast_a():
    v = sf.Series(('a', 'b'), index=sf.Index(('x', 'y'), dtype=np.str_))
    h = sf.Series[sf.Index[np.int64], np.int64]


    with pytest.raises(TypeError):
        TypeClinic(v).check(h, fail_fast=True)
    try:
        TypeClinic(v).check(h, fail_fast=True)
    except TypeError as e:
        assert scrub_str(str(e)) == 'In Series[Index[int64], int64] Expected int64, provided str_ invalid'

    with pytest.raises(TypeError):
        TypeClinic(v).check(h, fail_fast=False)
    try:
        TypeClinic(v).check(h, fail_fast=False)
    except TypeError as e:
        assert scrub_str(str(e)) == 'In Series[Index[int64], int64] Expected int64, provided str_ invalid In Series[Index[int64], int64] Index[int64] Expected int64, provided str_ invalid'

#-------------------------------------------------------------------------------

def test_check_type_sequence_a():
    TypeClinic([3, 4]).check(tp.List[int])

    with pytest.raises(TypeError):
        TypeClinic([3, 4, 'a']).check(tp.List[int])


    TypeClinic([3, 4, 'a']).check(tp.List[tp.Union[int, str]])

    TypeClinic(['c', 'b', 'a']).check(tp.Union[tp.List[int], tp.List[str]])

    with pytest.raises(TypeError):
        TypeClinic([3, 4, 'a', True]).check(tp.List[tp.Union[int, str]])

def test_check_type_sequence_b():

    TypeClinic([3, 4]).check(tp.Sequence[int])

    with pytest.raises(TypeError):
        TypeClinic([3, 4, 'a']).check(tp.Sequence[int])


    TypeClinic([3, 4, 'a']).check(tp.Sequence[tp.Union[int, str]])

    TypeClinic(['c', 'b', 'a']).check(tp.Union[tp.Sequence[int], tp.Sequence[str]])

    with pytest.raises(TypeError):
        TypeClinic([3, 4, 'a', True]).check(tp.Sequence[tp.Union[int, str]])


#-------------------------------------------------------------------------------

def test_check_type_tuple_a():

    with pytest.raises(TypeError):
        TypeClinic([3, 4]).check(tp.Tuple[int, bool])

    with pytest.raises(TypeError):
        TypeClinic((3, False, 'foo')).check(tp.Tuple[int, ...])

    TypeClinic((3, 4, 5)).check(tp.Tuple[int, ...])
    TypeClinic((3, 4, 5, 3, 20)).check(tp.Tuple[int, ...])
    TypeClinic((3,)).check(tp.Tuple[int, ...])

def test_check_type_tuple_b():

    TypeClinic((3, 4, False)).check(tp.Tuple[int, int, bool])
    TypeClinic((3, 4.1, False)).check(tp.Tuple[int, float, bool])

@skip_pyle310
def test_check_type_tuple_c():

    cr = TypeClinic((3, 4))(tp.Tuple[int, int, int])
    assert [r[1] for r in cr] == ['Expected tuple length of 3, provided tuple length of 2']

@skip_pyle310
def test_check_type_tuple_d():

    cr = TypeClinic((3, 4, 5))(tp.Tuple[..., int, ...])
    assert [r[1] for r in cr] == ['Invalid ellipses usage']


#-------------------------------------------------------------------------------

@skip_pyle310
def test_check_type_literal_a():
    TypeClinic(42)(tp.Literal[42])
    TypeClinic(42)(tp.Literal[-1, 42])

    cr = TypeClinic(42)(tp.Literal['a', 'b'])
    assert list(cr) == [(42, 'a', (tp.Literal['a', 'b'],), ()),
                        (42, 'b', (tp.Literal['a', 'b'],), ())]

#-------------------------------------------------------------------------------

def test_check_type_dict_a():
    TypeClinic({'a': 3}).check(tp.Dict[str, int])
    TypeClinic({'b': 20}).check(tp.Dict[str, int])

    with pytest.raises(TypeError):
        TypeClinic({'a': 20, 'b': 18, 'c': False}).check(tp.Dict[str, int])

    with pytest.raises(TypeError):
        TypeClinic({'a': 20, 'b': 18, 20: 3}).check(tp.Dict[str, int])


#-------------------------------------------------------------------------------
def test_check_interface_a():

    @CallGuard.check(fail_fast=False)
    def proc1(a: int, b: int) -> int:
        return a * b

    assert proc1(2, 3) == 6

def test_check_interface_b():

    @CallGuard.check(fail_fast=False)
    def proc1(a: int, b: int) -> bool:
        return a * b
    try:
        assert proc1(2, 3) == 6
    except TypeError as e:
        assert scrub_str(str(e)) == 'In return of (a: int, b: int) -> bool Expected bool, provided int invalid'

    try:
        assert proc1(2, 'foo') == 6
    except TypeError as e:
        assert scrub_str(str(e)) == 'In args of (a: int, b: int) -> bool In arg b Expected int, provided str invalid'

def test_check_interface_c1():

    @CallGuard.check(fail_fast=False)
    def proc1(a: int, b) -> int:
        return a * b

    assert proc1(2, False) == 0
    assert proc1(2, 1) == 2

    with pytest.raises(TypeError):
        assert proc1('foo', 1) == 2

def test_check_interface_c2():

    @CallGuard.check(fail_fast=False)
    def proc1(a: int, b) -> int:
        return a * b

    assert proc1(2, False) == 0
    assert proc1(2, 1) == 2

    with pytest.raises(TypeError):
        assert proc1('foo', 1) == 2

def test_check_interface_d():

    @CallGuard.check
    def proc1(a: int, b: int) -> int:
        return a * b

    assert proc1(2, 0) == 0
    assert proc1(2, 1) == 2



def test_check_interface_e():

    @CallGuard.check
    def proc1(a: tp.Annotated[int, 'foo'], b: tp.Annotated[int, 'bar']) -> int:
        return a * b

    assert proc1(2, 0) == 0
    assert proc1(2, 1) == 2


def test_check_interface_f1():

    @CallGuard.check
    def proc1(idx: tp.Annotated[sf.Index[np.str_], Require.Len(3), Require.Name('foo')]) -> int:
        return len(idx)

    idx1 = sf.Index(('a', 'b', 'c'), name='foo')
    assert proc1(idx1) == 3

    idx2 = sf.Index(('a', 'b', 'c'), name='fab')
    with pytest.raises(TypeError):
        _ = proc1(idx2)

    idx3 = sf.Index(('a', 'c'), name='fab')
    with pytest.raises(TypeError):
        _ = proc1(idx3)


def test_check_interface_f2():

    @CallGuard.warn(category=DeprecationWarning)
    def proc1(idx: tp.Annotated[sf.Index[np.str_], Require.Len(3), Require.Name('foo')]) -> int:
        return len(idx)

    idx1 = sf.Index(('a', 'b', 'c'), name='foo')
    assert proc1(idx1) == 3

    idx2 = sf.Index(('a', 'b', 'c'), name='fab')

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        _ = proc1(idx2)
        assert 'Annotated[Index[str_], Len(3), Name(foo)]' in str(w[0])


def test_check_interface_f3():

    @CallGuard.warn
    def proc1(idx: tp.Annotated[sf.Index[np.str_], Require.Len(3), Require.Name('foo')]) -> int:
        return len(idx)

    idx1 = sf.Index(('a', 'b', 'c'), name='foo')
    assert proc1(idx1) == 3

    idx2 = sf.Index(('a', 'b', 'c'), name='fab')

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        _ = proc1(idx2)
        assert 'Annotated[Index[str_], Len(3), Name(foo)]' in str(w[0])



def test_check_interface_g1():
    def proc1(x: int) -> int:
        return x

    assert _check_interface(proc1, (1,), {}, False, ErrorAction.RAISE) == 1

    cr1 = _check_interface(proc1, (False,), {}, False, ErrorAction.RETURN)
    assert scrub_str(cr1.to_str()) == 'In args of (x: int) -> int In arg x Expected int, provided bool invalid'

    def proc2(x: int) -> int:
        return 'foo'

    cr2 = _check_interface(proc2, (3,), {}, False, ErrorAction.RETURN)
    assert scrub_str(cr2.to_str()) == 'In return of (x: int) -> int Expected int, provided str invalid'


@skip_pyle310
def test_check_interface_g2():
    def proc1(x: int) -> int | None:
        return None

    assert _check_interface(proc1, (1,), {}, False, ErrorAction.RAISE) is None

    def proc2(x: int) -> int | None:
        return 'foo'

    cr1 = _check_interface(proc2, (3,), {}, False, ErrorAction.RETURN)
    assert scrub_str(cr1.to_str()) == "In return of (x: int) -> int | None int | None Expected int, provided str invalid In return of (x: int) -> int | None int | None Expected NoneType, provided str invalid"


def test_check_interface_g3():
    def proc1(x: int) -> int:
        return x

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        _check_interface(proc1, (False,), {}, False, ErrorAction.WARN)
        # two warnings, one for input, one for output
        assert len(w) == 2
        assert 'Expected int, provided bool invalid' in str(w[0])
        assert 'Expected int, provided bool invalid' in str(w[1])

#-------------------------------------------------------------------------------

def test_check_annotated_a():

    TypeClinic(3)(tp.Annotated[int, 'foo'])

def test_check_annotated_b():

    v1 = sf.Series(('a', 'b'), index=sf.Index((10, 20)))
    h1 = tp.Annotated[
            sf.Series[sf.Index[np.int_], np.str_],
            Require.Name('foo'),
    ]
    with pytest.raises(TypeError):
        TypeClinic(v1).check(h1)

    v1 = sf.Series(('a', 'b'), index=sf.Index((10, 20)), name='foo')
    TypeClinic(v1).check(h1)

def test_check_annotated_c():

    v1 = sf.Series(('a', 'b'), index=sf.Index((10, 20)))
    h1 = tp.Annotated[
            sf.Series[sf.Index[np.int_], np.str_],
            Require.Len(1),
            ]
    h2 = tp.Annotated[
            sf.Series[sf.Index[np.int_], np.str_],
            Require.Len(2),
            ]

    with pytest.raises(TypeError):
        TypeClinic(v1).check(h1)

    TypeClinic(v1).check(h2)


def test_check_index_a():
    idx = sf.Index((None, 'A', 1024, True))
    idx.via_type_clinic.check(sf.Index[np.object_])

#-------------------------------------------------------------------------------

def test_check_index_hierarchy_a():

    v1 = sf.IndexHierarchy.from_product(('a', 'b'), (1, 2))
    h1 = tp.Annotated[
            sf.IndexHierarchy[sf.Index[np.str_], sf.Index[np.integer]],
            Require.Len(4),
            ]
    TypeClinic(v1).check(h1)

    h1 = sf.IndexHierarchy[sf.Index[np.str_], sf.Index[np.integer], sf.IndexDate]
    with pytest.raises(TypeError):
        TypeClinic(v1).check(h1)

def test_check_index_hierarchy_b():

    v1 = sf.IndexHierarchy.from_labels([(1, 100), (1, 200), (2, 100)])
    v2 = sf.IndexHierarchy.from_labels([(1, 100, 3), (1, 200, 3), (2, 100, 3)])

    h1 = sf.IndexHierarchy[tp.Unpack[tp.Tuple[sf.Index[np.int_], ...]]]
    h2 = sf.IndexHierarchy[tp.Unpack[tp.Tuple[sf.Index[np.str_], ...]]]

    TypeClinic(v1)(h1)
    assert TypeClinic(v1)(h1).validated

    TypeClinic(v2)(h1)
    assert not TypeClinic(v1)(h2).validated

def test_check_index_hierarchy_c():

    v1 = sf.IndexHierarchy.from_labels([(1, 'a', False), (1, 'b', False), (2, 'c', True)])

    h1 = sf.IndexHierarchy[sf.Index[np.int_], sf.Index[np.str_], sf.Index[np.bool_]]
    h2 = sf.IndexHierarchy[tp.Unpack[tp.Tuple[sf.Index[np.int_], sf.Index[np.str_], sf.Index[np.bool_]]]]

    h3 = sf.IndexHierarchy[sf.Index[np.int_], sf.Index[np.bool_], sf.Index[np.str_]]

    TypeClinic(v1).check(h1)
    TypeClinic(v1).check(h2)

def test_check_index_hierarchy_d1():

    v1 = sf.IndexHierarchy.from_labels([(1, 'a', False), (1, 'b', False), (2, 'c', True)])
    h1 = sf.IndexHierarchy[
            sf.Index[np.int_],
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[sf.Index[np.bool_], ...]],
            ]
    assert v1.via_type_clinic(h1).validated

    v2 = sf.IndexHierarchy.from_labels([(1, 'a',), (1, 'b',), (2, 'c',)])
    assert v2.via_type_clinic(h1).validated

    v3 = sf.IndexHierarchy.from_labels([
            (1, 'a', False, True),
            (1, 'b', False, True),
            (2, 'c', True, False),
            ])
    assert v3.via_type_clinic(h1).validated

def test_check_index_hierarchy_d2():

    v1 = sf.IndexHierarchy.from_labels(
            [(1,  False), (3,  False), (2,  True)],
            index_constructors=(partial(sf.Index, dtype=np.int64), sf.Index),
            )
    h1 = sf.IndexHierarchy[
            sf.Index[np.int64],
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[sf.Index[np.bool_], ...]],
            ]
    assert not v1.via_type_clinic(h1).validated
    if IS_NP2:
        assert scrub_str(v1.via_type_clinic(h1).to_str()) == 'In IndexHierarchy[Index[int64], Index[str_], Unpack[Tuple[Index[bool], ...]]] Depth 1 Index[str_] Expected str_, provided bool invalid'
    else:
        assert scrub_str(v1.via_type_clinic(h1).to_str()) == 'In IndexHierarchy[Index[int64], Index[str_], Unpack[Tuple[Index[bool_], ...]]] Depth 1 Index[str_] Expected str_, provided bool_ invalid'

def test_check_index_hierarchy_e1():

    v1 = sf.IndexHierarchy.from_labels([(1,  3), (3,  2), (2,  3)])
    h1 = sf.IndexHierarchy[
            tp.Unpack[tp.Tuple[sf.Index[np.integer], ...]],
            ]
    assert v1.via_type_clinic(h1).validated

    v2 = sf.IndexHierarchy.from_labels([(1,  3, 5), (3,  2, 2), (2,  3, 7)])
    assert v2.via_type_clinic(h1).validated

def test_check_index_hierarchy_e2():

    v1 = sf.IndexHierarchy.from_labels([(1,  'a'), (3,  'b'), (2,  'c')])
    h1 = sf.IndexHierarchy[
            tp.Unpack[tp.Tuple[sf.Index[np.integer], ...]],
            ]
    assert not v1.via_type_clinic(h1).validated
    assert scrub_str(v1.via_type_clinic(h1).to_str()) == 'In IndexHierarchy[Unpack[Tuple[Index[integer], ...]]] Tuple[Index[integer], ...] Index[integer] Expected integer, provided str_ invalid'


def test_check_index_hierarchy_f():

    v1 = sf.IndexHierarchy.from_labels([(1,  'a'), (3,  'b'), (2,  'c')])
    h1 = sf.IndexHierarchy[sf.Index[np.integer], sf.IndexDate, sf.IndexDate]

    assert not v1.via_type_clinic(h1).validated
    assert scrub_str(v1.via_type_clinic(h1).to_str()) == 'In IndexHierarchy[Index[integer], IndexDate, IndexDate] Expected IndexHierarchy has 3 depth, provided IndexHierarchy has 2 depth'


def test_check_index_hierarchy_g():

    v1 = sf.IndexHierarchy.from_labels([
            (1, 'a', False, 'a',),
            (3, 'b', True, 'c',),
            (2, 'c', False, 'd',),
            ],
            index_constructors=(
            partial(sf.Index, dtype=np.int64),
            sf.Index,
            sf.Index,
            sf.Index,
            ))
    h1 = sf.IndexHierarchy[
            sf.Index[np.int64],
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[sf.Index[np.bool_], ...]],
            sf.Index[np.str_],
            sf.Index[np.int64],
            sf.Index[np.int64],
            ]

    assert not v1.via_type_clinic(h1).validated

    if IS_NP2:
        assert scrub_str(v1.via_type_clinic(h1).to_str()) == 'In IndexHierarchy[Index[int64], Index[str_], Unpack[Tuple[Index[bool], ...]], Index[str_], Index[int64], Index[int64]] Expected IndexHierarchy has 5 depth (excluding Unpack), provided IndexHierarchy has 4 depth'
    else:
        assert scrub_str(v1.via_type_clinic(h1).to_str()) == 'In IndexHierarchy[Index[int64], Index[str_], Unpack[Tuple[Index[bool_], ...]], Index[str_], Index[int64], Index[int64]] Expected IndexHierarchy has 5 depth (excluding Unpack), provided IndexHierarchy has 4 depth'


def test_check_index_hierarchy_h1():

    v1 = sf.IndexHierarchy.from_labels([
            (1, 'a', False, 'a', 3, 2),
            (3, 'b', True, 'c', 10, 12),
            (2, 'c', False, 'd', 20, 3),
            ])
    h1 = sf.IndexHierarchy[
            sf.Index[np.int_],
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[sf.Index[np.bool_], ...]],
            sf.Index[np.str_],
            sf.Index[np.int_],
            sf.Index[np.int_],
            ]

    assert v1.via_type_clinic(h1).validated

def test_check_index_hierarchy_h2():

    v1 = sf.IndexHierarchy.from_labels([
            (1, 'a', False, True, 'a', 3, 2),
            (3, 'b', True, False, 'c', 10, 12),
            (2, 'c', False, True, 'd', 20, 3),
            ])
    h1 = sf.IndexHierarchy[
            sf.Index[np.int_],
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[sf.Index[np.bool_], ...]],
            sf.Index[np.str_],
            sf.Index[np.int_],
            sf.Index[np.int_],
            ]

    assert v1.via_type_clinic(h1).validated


def test_check_index_hierarchy_h3():

    v1 = sf.IndexHierarchy.from_labels([
            (1, 'a', False, True, 'a', 3, 2),
            (3, 'b', True, False, 'c', 10, 12),
            (2, 'c', False, True, 'd', 20, 3),
            ])
    h1 = sf.IndexHierarchy[
            sf.Index[np.int_],
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[sf.Index[np.bool_], ...]],
            sf.Index[np.str_],
            sf.Index[np.int_],
            sf.Index[np.int_],
            sf.Index[np.int_],
            ]

    assert not v1.via_type_clinic(h1).validated

#-------------------------------------------------------------------------------

@skip_pyle310
def test_check_frame_a1():
    records = ((1, 3, True), (3, 8, True),)

    # NOTE: the default is any number of any columns
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],]

    index = sf.IndexDate(('2022-01-03', '2018-04-02'))
    f: h1 = sf.Frame.from_records(records,
            columns=('a', 'b', 'c'),
            index=index,
            )
    cr = TypeClinic(f)(h1)
    assert not get_hints(cr)

def test_check_frame_a2():
    records = ((1, 3, True), (3, 8, True),)

    # NOTE: the default is any number of any columns
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            np.bool_]

    index = sf.IndexDate(('2022-01-03', '2018-04-02'))
    f: h1 = sf.Frame.from_records(records,
            columns=('a', 'b', 'c'),
            index=index,
            )
    cr = TypeClinic(f)(h1)
    assert get_hints(cr) == ('Expected Frame has 1 dtype, provided Frame has 3 dtype',)

def test_check_frame_b():
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[tp.Any, ...]],
            ]

    records = ((1, 3, True), (3, 8, True),)
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))
    f1: h1 = sf.Frame.from_records(records,
            columns=('a', 'b', 'c'),
            index=index,
            )
    assert TypeClinic(f1)(h1).validated

    records = ((1, 3, True, False), (3, 8, True, True),)
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))
    f2: h1 = sf.Frame.from_records(records,
            columns=('a', 'b', 'c', 'd'),
            index=index,
            )
    assert TypeClinic(f2)(h1).validated

def test_check_frame_c():
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[np.float64, ...]],
            ]
    records = ((1.8, 3.1), (3.2, 8.1),)
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))
    f1: h1 = sf.Frame.from_records(records,
            columns=('a', 'b'),
            index=index,
            )
    assert TypeClinic(f1)(h1).validated

    records = ((1.8, 3.1, 5.4), (3.2, 8.1, 4.7),)
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))
    f2: h1 = sf.Frame.from_records(records,
            columns=('a', 'b', 'c'),
            index=index,
            )
    assert TypeClinic(f2)(h1).validated


    records = ((1.8, 3.1, False), (3.2, 8.1, True),)
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))
    f3: h1 = sf.Frame.from_records(records,
            columns=('a', 'b', 'c'),
            index=index,
            )
    if IS_NP2:
        assert scrub_str(TypeClinic(f3)(h1).to_str()) == 'In Frame[IndexDate, Index[str_], Unpack[Tuple[float64, ...]]] Tuple[float64, ...] Expected float64, provided bool invalid'
    else:
        assert scrub_str(TypeClinic(f3)(h1).to_str()) == 'In Frame[IndexDate, Index[str_], Unpack[Tuple[float64, ...]]] Tuple[float64, ...] Expected float64, provided bool_ invalid'

def test_check_frame_d():
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            np.bool_,
            tp.Unpack[tp.Tuple[np.float64, ...]],
            np.str_,
            np.str_
            ]
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))

    records1 = ((True, 1.8, 3.1, 'x', 'y'), (False, 3.2, 8.1, 'a', 'b'),)
    f1 = sf.Frame.from_records(records1,
            columns=('a', 'b', 'c', 'd', 'e'),
            index=index,
            )
    assert TypeClinic(f1)(h1).validated

    records2 = ((True, 1.8, 3.1, 1.2, 'x', 'y'), (False, 3.2, 8.1, 3.5, 'a', 'b'),)
    f2 = sf.Frame.from_records(records2,
            columns=('a', 'b', 'c', 'd', 'e', 'f'),
            index=index,
            )
    assert TypeClinic(f2)(h1).validated

    records3 = ((1.8, 3.1, 1.2, 'x', 'y'), (3.2, 8.1, 3.5, 'a', 'b'),)
    f3 = sf.Frame.from_records(records3,
            columns=('a', 'b', 'c', 'd', 'e'),
            index=index,
            )
    assert not TypeClinic(f3)(h1).validated

    if IS_NP2:
        assert scrub_str(TypeClinic(f3)(h1).to_str()) == 'In Frame[IndexDate, Index[str_], bool, Unpack[Tuple[float64, ...]], str_, str_] Field 0 Expected bool, provided float64 invalid'
    else:
        assert scrub_str(TypeClinic(f3)(h1).to_str()) == 'In Frame[IndexDate, Index[str_], bool_, Unpack[Tuple[float64, ...]], str_, str_] Field 0 Expected bool_, provided float64 invalid'

    records4 = ((True, 1.8, 'x'), (False, 3.2, 'a'),)
    f4 = sf.Frame.from_records(records4,
            columns=('a', 'b', 'c'),
            index=index,
            )
    assert not TypeClinic(f4)(h1).validated

    if IS_NP2:
        assert scrub_str(TypeClinic(f4)(h1).to_str()) == 'In Frame[IndexDate, Index[str_], bool, Unpack[Tuple[float64, ...]], str_, str_] Field 1 Expected str_, provided float64 invalid'
    else:
        assert scrub_str(TypeClinic(f4)(h1).to_str()) == 'In Frame[IndexDate, Index[str_], bool_, Unpack[Tuple[float64, ...]], str_, str_] Field 1 Expected str_, provided float64 invalid'


def test_check_frame_e1():
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[np.float64, ...]],
            np.str_,
            np.str_
            ]
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))

    records1 = ((3.1, 'x', 'y'), (8.1, 'a', 'b'),)
    f1 = sf.Frame.from_records(records1,
            columns=('a', 'b', 'c'),
            index=index,
            )
    assert TypeClinic(f1)(h1).validated

    records2 = ((3.1, 3.2, 5.2, 'x', 'y'), (8.1, 1.5, 5.2, 'a', 'b'),)
    f2 = sf.Frame.from_records(records2,
            columns=('a', 'b', 'c', 'd', 'e'),
            index=index,
            )
    assert TypeClinic(f2)(h1).validated

    records3 = ((3.1, False, 5.2, 'x', 'y'), (8.1, True, 5.2, 'a', 'b'),)
    f3 = sf.Frame.from_records(records3,
            columns=('a', 'b', 'c', 'd', 'e'),
            index=index,
            )
    assert not TypeClinic(f3)(h1).validated

    if IS_NP2:
        assert scrub_str(TypeClinic(f3)(h1).to_str()) == 'In Frame[IndexDate, Index[str_], Unpack[Tuple[float64, ...]], str_, str_] Fields 0 to 2 Tuple[float64, ...] Expected float64, provided bool invalid'
    else:
        assert scrub_str(TypeClinic(f3)(h1).to_str()) == 'In Frame[IndexDate, Index[str_], Unpack[Tuple[float64, ...]], str_, str_] Fields 0 to 2 Tuple[float64, ...] Expected float64, provided bool_ invalid'


def test_check_frame_e2():
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[np.float64, ...]],
            np.str_,
            np.str_
            ]
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))

    records1 = ((3.1, 'x'), (8.1, 'a'),)
    f1 = sf.Frame.from_records(records1,
            columns=('a', 'b'),
            index=index,
            )

    assert not f1.via_type_clinic(h1).validated
    assert scrub_str(f1.via_type_clinic(h1).to_str()) == 'In Frame[IndexDate, Index[str_], Unpack[Tuple[float64, ...]], str_, str_] Field 0 Expected str_, provided float64 invalid'

    records2 = ((3.1, 'x', 'p'), (8.1, 'a', 'q'),)
    f2 = sf.Frame.from_records(records2,
            columns=('a', 'b', 'c'),
            index=index,
            )

    assert f2.via_type_clinic(h1).validated


def test_check_frame_e3():
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            tp.Unpack[tp.Tuple[np.float64, ...]],
            np.str_,
            np.str_
            ]
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))

    records1 = (('a', 'x'), ('b', 'a'),)
    f1 = sf.Frame.from_records(records1,
            columns=('a', 'b'),
            index=index,
            )

    assert f1.via_type_clinic(h1).validated

    records2 = ((1.2, 'a', 'x'), (3.4, 'b', 'a'),)
    f2 = sf.Frame.from_records(records2,
            columns=('a', 'b', 'c'),
            index=index,
            )
    assert f2.via_type_clinic(h1).validated


def test_check_frame_f1():
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            np.str_,
            np.str_,
            tp.Unpack[tp.Tuple[np.float64, ...]],
            ]
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))

    records1 = (('a', 'x'), ('b', 'a'),)
    f1 = sf.Frame.from_records(records1,
            columns=('a', 'b'),
            index=index,
            )

    assert f1.via_type_clinic(h1).validated

    records2 = (('a', 'x', 1.2), ('b', 'a', 5.4),)
    f2 = sf.Frame.from_records(records2,
            columns=('a', 'b', 'c'),
            index=index,
            )

    assert f2.via_type_clinic(h1).validated

    records3 = (('a', 'x', 1.2, 5.3, 5.4), ('b', 'a', 5.4, 1.2, 1.4),)
    f3 = sf.Frame.from_records(records3,
            columns=('a', 'b', 'c', 'd', 'e'),
            index=index,
            )

    assert f3.via_type_clinic(h1).validated


def test_check_frame_f2():
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            np.str_,
            np.str_,
            tp.Unpack[tp.Tuple[np.float64, ...]],
            ]
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))

    records1 = (('a', 'x', 1.3, 'q'), ('b', 'a', 1.5, 'x'),)
    f1 = sf.Frame.from_records(records1,
            columns=('a', 'b', 'c', 'd'),
            index=index,
            )
    assert not f1.via_type_clinic(h1).validated
    assert scrub_str(f1.via_type_clinic(h1).to_str()) == 'In Frame[IndexDate, Index[str_], str_, str_, Unpack[Tuple[float64, ...]]] Fields 2 to 3 Tuple[float64, ...] Expected float64, provided str_ invalid'

def test_check_frame_g():
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            np.str_,
            np.str_,
            np.str_,
            tp.Unpack[tp.Tuple[np.float64, ...]],
            ]
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))

    records1 = (('a', 'x'), ('b', 'a'),)
    f1 = sf.Frame.from_records(records1,
            columns=('a', 'b'),
            index=index,
            )
    assert not f1.via_type_clinic(h1).validated
    assert scrub_str(f1.via_type_clinic(h1).to_str()) == 'In Frame[IndexDate, Index[str_], str_, str_, str_, Unpack[Tuple[float64, ...]]] Expected Frame has 3 dtype (excluding Unpack), provided Frame has 2 dtype'


#-------------------------------------------------------------------------------

def test_check_bus_a():
    f1 = ff.parse('s(2,2)|c(I,str)|v(int)')
    f2 = ff.parse('s(2,2)|c(I,str)|v(bool)')
    b1 = sf.Bus((f1, f2), index=('a', 'b'))

    cr1 = b1.via_type_clinic(sf.Bus[sf.Index[np.str_]])
    assert cr1.validated is True

    cr2 = b1.via_type_clinic(sf.Bus[sf.Index[np.int64]])
    assert cr2.validated is False

def test_check_bus_b():
    f1 = ff.parse('s(2,2)|c(I,str)|v(int)')
    f2 = ff.parse('s(2,2)|c(I,str)|v(bool)')
    b1 = sf.Bus((f1, f2), index=('a', 'b'))

    assert b1.via_type_clinic.to_hint() == sf.Bus[sf.Index[np.str_]]

def test_check_yarn_a():

    f1 = ff.parse('s(4,4)|v(int,float)').rename('f1')
    f2 = ff.parse('s(4,4)|v(str)').rename('f2')
    f3 = ff.parse('s(4,4)|v(bool)').rename('f3')
    b1 = sf.Bus.from_frames((f1, f2, f3))

    f4 = ff.parse('s(4,4)|v(int,float)').rename('f4')
    f5 = ff.parse('s(4,4)|v(str)').rename('f5')
    b2 = sf.Bus.from_frames((f4, f5))

    y1 = sf.Yarn((b1, b2), index=tuple('abcde'))
    cr1 = y1.via_type_clinic(sf.Yarn[sf.Index[np.str_]])
    assert cr1.validated is True

    cr2 = y1.via_type_clinic(sf.Yarn[sf.Index[np.int64]])
    assert cr2.validated is False


#-------------------------------------------------------------------------------

def get_hints(records: tp.Union[tp.Iterable[TValidation], ClinicResult]) -> tp.Tuple[str]:
    return tuple(r[1] for r in records)

def test_validate_labels_order_a1():
    idx1 = sf.Index(('a', 'b', 'c'))
    v = Require.LabelsOrder('a', 'b', 'c')
    assert not get_hints(v._iter_errors(idx1, None, (), ()))

def test_validate_labels_order_a2():
    idx1 = sf.Index(('a', 'x', 'c'))
    v = Require.LabelsOrder('a', 'b', 'c')
    if IS_NP2:
        assert get_hints(v._iter_errors(idx1, None, (), ())) == ("Expected 'b', provided np.str_('x')",)
    else:
        assert get_hints(v._iter_errors(idx1, None, (), ())) == ("Expected 'b', provided 'x'",)

def test_validate_labels_order_a3():
    idx1 = sf.Index(('a', 'x', 'z'))
    v = Require.LabelsOrder('a', 'b', 'c')
    if IS_NP2:
        assert get_hints(v._iter_errors(idx1, None, (), ())) == ("Expected 'b', provided np.str_('x')",)
    else:
        assert get_hints(v._iter_errors(idx1, None, (), ())) == ("Expected 'b', provided 'x'",)

def test_validate_labels_order_b1():
    idx1 = sf.Index(('a', 'b', 'c', 'd'))
    v = Require.LabelsOrder('a', ..., 'd')
    assert not get_hints(v._iter_errors(idx1, None, (), ()))

def test_validate_labels_order_b2():
    idx1 = sf.Index(('a', 'b'))
    v = Require.LabelsOrder('a', ..., 'b')
    assert not get_hints(v._iter_errors(idx1, None, (), ()))


def test_validate_labels_order_c():
    idx1 = sf.Index(('a', 'b', 'c', 'd'))
    v = Require.LabelsOrder(..., 'd')
    assert not get_hints(v._iter_errors(idx1, None, (), ()))

def test_validate_labels_order_d1():
    idx1 = sf.Index(('a', 'b', 'c', 'd'))
    v = Require.LabelsOrder('a', 'b', ...)
    assert not get_hints(v._iter_errors(idx1, None, (), ()))

def test_validate_labels_order_d2():
    idx1 = sf.Index(('a', 'b', 'c', 'd'))
    v = Require.LabelsOrder('a', 'b', ..., 'e')
    assert get_hints(v._iter_errors(idx1, None, (), ())) == ("Expected has unmatched labels 'e'",)

def test_validate_labels_order_e1():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e'))
    v = Require.LabelsOrder('a', ..., 'c', ..., 'd')
    if IS_NP2:
        assert get_hints(v._iter_errors(idx1, None, (), ())) == ("Expected labels exhausted at provided np.str_('e')",)
    else:
        assert get_hints(v._iter_errors(idx1, None, (), ())) == ("Expected labels exhausted at provided 'e'",)

def test_validate_labels_order_e2():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e'))
    v = Require.LabelsOrder('a', ..., 'c', ..., 'e')
    assert not get_hints(v._iter_errors(idx1, None, (), ()))

def test_validate_labels_order_e3():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e'))
    v = Require.LabelsOrder('a', ..., 'c', ...)
    assert not get_hints(v._iter_errors(idx1, None, (), ()))

def test_validate_labels_order_e4():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e'))
    v = Require.LabelsOrder(..., 'c', ...)
    assert not get_hints(v._iter_errors(idx1, None, (), ()))

def test_validate_labels_order_e5():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e'))
    v = Require.LabelsOrder(..., 'b', 'c', ...)
    assert not get_hints(v._iter_errors(idx1, None, (), ()))

def test_validate_labels_order_e6():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e'))
    v = Require.LabelsOrder(..., 'b', ..., 'd', 'e')
    assert not get_hints(v._iter_errors(idx1, None, (), ()))

def test_validate_labels_order_e7():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e'))
    v = Require.LabelsOrder('a', 'b', ..., 'd', 'e')
    assert not get_hints(v._iter_errors(idx1, None, (), ()))

def test_validate_labels_order_e8():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e'))
    v = Require.LabelsOrder('a', 'b', ..., 'f', ...)
    assert get_hints(v._iter_errors(idx1, None, (), ())) == ("Expected has unmatched labels 'f'",)

def test_validate_labels_order_e9():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e'))
    v = Require.LabelsOrder(..., 'x', ..., 'y', ...)
    assert get_hints(v._iter_errors(idx1, None, (), ())) == ("Expected has unmatched labels 'x', ..., 'y'",)

def test_validate_labels_order_e10():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e'))
    v = Require.LabelsOrder(..., 'a', ..., ...)
    assert get_hints(v._iter_errors(idx1, None, (), ())) == ("Expected cannot be defined with adjacent ellipses",)

def test_validate_labels_order_e11():
    idx1 = sf.Series(('a',))
    v = Require.LabelsOrder('a', ...)
    assert get_hints(v._iter_errors(idx1, None, (), ())) == ("Expected LabelsOrder('a', ...) to be used on Index or IndexHierarchy, not provided Series",)

def test_validate_labels_order_e12():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e'))
    v = Require.LabelsOrder('a', 'b', ..., 'd', 'e')
    assert not get_hints(v._iter_errors(idx1, None, (), ()))


def test_validate_labels_order_f1():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e'))
    v = Require.LabelsOrder('a', 'b', 'c', 'd', 'e', ...)
    assert not get_hints(v._iter_errors(idx1, None, (), ()))

def test_validate_labels_order_f2():
    idx1 = sf.Index(('a', 'b', 'c', 'd', 'e', 'f', 'g'))
    v = Require.LabelsOrder('a', 'b', 'c', 'd', 'e', ...)
    assert not get_hints(v._iter_errors(idx1, None, (), ()))

def test_validate_labels_order_f3():
    idx1 = sf.Index(('a', 'b', 'c', 'd'))
    v = Require.LabelsOrder('a', 'b', 'c', 'd', 'e', ...)
    assert get_hints(v._iter_errors(idx1, None, (), ())) == ("Expected has unmatched labels 'e'", )


#-------------------------------------------------------------------------------

def test_validate_labels_order_g():
    records = (
            (1, 3, True),
            (4, 100, False),
            (3, 8, True),
            )
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            tp.Annotated[sf.Index[np.str_],
                    sf.Require.LabelsOrder(
                            ['a', lambda s: (s < 0).all()],
                            ...,
                            'c',
                            )
                    ],
            np.int64,
            np.int64,
            np.bool_]

    index = sf.IndexDate(('2022-01-03', '2022-02-05', '2018-04-02'))
    f: h1 = sf.Frame.from_records(records,
            columns=('a', 'b', 'c'),
            index=index,
            dtypes=(np.int64, np.int64, np.bool_)
            )

    cr = TypeClinic(f)(h1)
    if IS_NP2:
        assert (scrub_str(cr.to_str()) ==
                "In Frame[IndexDate, Annotated[Index[str_], LabelsOrder(['a', <lambda>], ..., 'c')], int64, int64, bool] Annotated[Index[str_], LabelsOrder(['a', <lambda>], ..., 'c')] LabelsOrder(['a', <lambda>], ..., 'c') Validation failed of label np.str_('a') with <lambda>"
                )
    else:
        assert (scrub_str(cr.to_str()) ==
                "In Frame[IndexDate, Annotated[Index[str_], LabelsOrder(['a', <lambda>], ..., 'c')], int64, int64, bool_] Annotated[Index[str_], LabelsOrder(['a', <lambda>], ..., 'c')] LabelsOrder(['a', <lambda>], ..., 'c') Validation failed of label 'a' with <lambda>"
                )

    h2 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            tp.Annotated[sf.Index[np.str_],
                    sf.Require.LabelsOrder(
                            ['a', lambda s: (s > 0).all()],
                            ...,
                            ['c', lambda s: s.sum() == 3],
                            )
                    ],
            np.int64,
            np.int64,
            np.bool_]

    cr = TypeClinic(f)(h2)

    if IS_NP2:
        assert scrub_str(cr.to_str()) == "In Frame[IndexDate, Annotated[Index[str_], LabelsOrder(['a', <lambda>], ..., ['c', <lambda>])], int64, int64, bool] Annotated[Index[str_], LabelsOrder(['a', <lambda>], ..., ['c', <lambda>])] LabelsOrder(['a', <lambda>], ..., ['c', <lambda>]) Validation failed of label 'c' with <lambda>"
    else:
        assert scrub_str(cr.to_str()) == "In Frame[IndexDate, Annotated[Index[str_], LabelsOrder(['a', <lambda>], ..., ['c', <lambda>])], int64, int64, bool_] Annotated[Index[str_], LabelsOrder(['a', <lambda>], ..., ['c', <lambda>])] LabelsOrder(['a', <lambda>], ..., ['c', <lambda>]) Validation failed of label 'c' with <lambda>"



#-------------------------------------------------------------------------------

def test_validate_labels_order_h1():
    idx1 = sf.Index(('a', 'b', 'c'))
    records = (
            (1, 3, True),
            (4, 100, False),
            (3, 8, True),
            )

    index = sf.IndexDate(('2022-01-03', '2022-02-05', '2018-04-02'))
    f = sf.Frame.from_records(records,
            columns=('a', 'b', 'c'),
            index=index,
            dtypes=(np.int64, np.int64, np.bool_)
            )

    v = Require.LabelsOrder('a', 'b', ['c', lambda s: s.dtype == bool])

    with pytest.raises(RuntimeError):
        # Provided label validators in a context without a discoverable Frame.
        tuple(v._iter_errors(idx1, None, (), ()))

    with pytest.raises(RuntimeError):
        # Labels associated with an index that is not a member of the parent Frame
        tuple(v._iter_errors(idx1, None, (), (f,)))


def test_validate_labels_order_h2():
    records = (
            (1, 3, True, 'y'),
            (4, 100, False, 'x'),
            (3, 8, True, 'q'),
            )

    index = sf.IndexDate(('2022-01-03', '2022-02-05', '2018-04-02'))
    f = sf.Frame.from_records(records,
            columns=('a', 'b', 'c', 'd'),
            index=index,
            dtypes=(np.int64, np.int64, np.bool_, np.str_)
            )

    v = Require.LabelsOrder('2022-01-03', ...)
    assert not tuple(v._iter_errors(f.index, None, (), (f,)))

    v = Require.LabelsOrder(..., '2018-04-02')
    assert not tuple(v._iter_errors(f.index, None, (), (f,)))

    v = Require.LabelsOrder(..., '2022-02-05', '2018-04-02')
    assert not tuple(v._iter_errors(f.index, None, (), (f,)))

    v = Require.LabelsOrder('2021-01-03', ...)
    assert tuple(v._iter_errors(f.index, None, (), (f,)))

    v = Require.LabelsOrder(..., '2021-01-03')
    assert tuple(v._iter_errors(f.index, None, (), (f,)))


def test_validate_labels_order_h3():
    records = (
            (1, 3, True, 'y'),
            (4, 100, False, 'x'),
            (3, 8, True, 'q'),
            )

    index = sf.IndexDate(('2022-01-03', '2022-02-05', '2018-04-02'))
    f = sf.Frame.from_records(records,
            columns=('a', 'b', 'c', 'd'),
            index=index,
            dtypes=(np.int64, np.int64, np.bool_, np.str_)
            )

    v = Require.LabelsOrder(['2022-01-03', lambda s: 'y' in s.values], ...)
    assert not tuple(v._iter_errors(f.index, None, (), (f,)))

    v = Require.LabelsOrder(..., ['2018-04-02', lambda s: 'q' in s.values])
    assert not tuple(v._iter_errors(f.index, None, (), (f,)))


def test_validate_labels_order_h4():
    records = (
            (-3, True, 'y'),
            (-100, False, 'x'),
            (8, True, 'z'),
            )

    index = sf.IndexDate(('2022-01-03', '2022-02-05', '2023-04-02'))
    f = sf.Frame.from_records(records,
            columns=('a', 'b', 'c'),
            index=index,
            dtypes=(np.int64, np.bool_, np.str_)
            )

    f.via_type_clinic.check(sf.Frame[sf.IndexDate, sf.Index[np.str_], np.int64, np.bool_, np.str_])

    f.via_type_clinic.check(sf.Frame[tp.Annotated[sf.IndexDate, sf.Require.LabelsOrder('2022-01-03', ..., '2023-04-02')], sf.Index[np.str_], np.int64, np.bool_, np.str_])

    f.via_type_clinic.check(sf.Frame[tp.Annotated[sf.IndexDate, sf.Require.LabelsOrder('2022-01-03', ..., '2023-04-02')], tp.Annotated[sf.Index[np.str_], sf.Require.LabelsOrder(..., 'b', 'c')], np.int64, np.bool_, np.str_])

    f.via_type_clinic.check(sf.Frame[tp.Annotated[sf.IndexDate, sf.Require.LabelsOrder('2022-01-03', ..., '2023-04-02')], tp.Annotated[sf.Index[np.str_], sf.Require.LabelsOrder(..., 'b', ['c', lambda s: s.isin(('x', 'y', 'z')).all()])], np.int64, np.bool_, np.str_])

    with pytest.raises(TypeError):
        f.via_type_clinic.check(sf.Frame[tp.Annotated[sf.IndexDate, sf.Require.LabelsOrder('2022-01-03', ..., '2023-04-02')], tp.Annotated[sf.Index[np.str_], sf.Require.LabelsOrder(..., ['b', lambda s: s.all()], ['c', lambda s: s.isin(('x', 'y', 'z')).all()])], np.int64, np.bool_, np.str_])

    @sf.CallGuard.check
    def proc1(f: sf.Frame[tp.Annotated[sf.IndexDate, sf.Require.LabelsOrder('2022-01-03', ..., '2023-04-02')], tp.Annotated[sf.Index[np.str_], sf.Require.LabelsOrder(..., 'b', ['c', lambda s: s.isin(('x', 'y', 'z')).all()])], np.int64, np.bool_, np.str_]) -> np.int_:
        return f.loc[f['b'], 'c'].isin(('y', 'x')).sum()

    assert proc1(f) == 1

    @sf.CallGuard.check
    def proc2(f: sf.Frame[tp.Annotated[sf.IndexDate, sf.Require.LabelsOrder('2022-01-03', ..., '2023-04-02')], tp.Annotated[sf.Index[np.str_], sf.Require.LabelsOrder(..., ['b', lambda s: s.all()], ['c', lambda s: s.isin(('x', 'y', 'z')).all()])], np.int64, np.bool_, np.str_]) -> np.int_:
        return f.loc[f['b'], 'c'].isin(('y', 'x')).sum()

    with pytest.raises(TypeError):
        assert proc2(f) == 1


def test_validate_labels_order_i1():
    records = (
            (1, 3, True, 'y'),
            (4, 100, False, 'x'),
            (3, 8, True, 'q'),
            )

    f = sf.Frame.from_records(records,
            columns=('a', 'b', 'c', 'd'),
            dtypes=(np.int64, np.int64, np.bool_, np.str_)
            )

    v = Require.LabelsOrder([..., lambda s: (s > 0).all()], 'c', 'd')
    assert not tuple(v._iter_errors(f.columns, None, (), (f,)))

    v = Require.LabelsOrder([..., lambda s: (s < 0).all()], 'c', 'd')
    assert tuple(v._iter_errors(f.columns, None, (), (f,)))


def test_validate_labels_order_i2():
    records = (
            (1, 3, True, 'y'),
            (4, 100, False, 'x'),
            (3, 8, True, 'q'),
            )

    f = sf.Frame.from_records(records,
            columns=('a', 'b', 'c', 'd'),
            dtypes=(np.int64, np.int64, np.bool_, np.str_)
            )

    v = Require.LabelsOrder([..., lambda s: s.dtype.kind == 'i'], ['c', lambda s: s.dtype == bool], 'd')
    assert not tuple(v._iter_errors(f.columns, None, (), (f,)))

    v = Require.LabelsOrder([..., lambda s: s.dtype.kind == 'i'], ['c', lambda s: s.dtype == float], 'd')
    assert tuple(v._iter_errors(f.columns, None, (), (f,)))


def test_validate_labels_order_i3():
    records = (
            (1, 3, True, 'y'),
            (4, 100, False, 'x'),
            (3, 8, True, 'q'),
            )

    f = sf.Frame.from_records(records,
            columns=('a', 'b', 'c', 'd'),
            dtypes=(np.int64, np.int64, np.bool_, np.str_)
            )

    v = Require.LabelsOrder([..., lambda s: s.dtype.kind == 'i'], 'c', ['d', lambda s: s.dtype.kind == 'U'])
    assert not tuple(v._iter_errors(f.columns, None, (), (f,)))



def test_validate_labels_order_i4():
    records = (
            (1, 3, True, 'y'),
            (4, 100, False, 'x'),
            (3, 8, True, 'q'),
            )

    f = sf.Frame.from_records(records,
            columns=('a', 'b', 'c', 'd'),
            dtypes=(np.int64, np.int64, np.bool_, np.str_)
            )

    v1 = Require.LabelsOrder('a', 'b', 'c', ...)
    assert not tuple(v1._iter_errors(f.columns, None, (), (f,)))

    v2 = Require.LabelsOrder('a', 'b', 'c', [..., lambda s: s.dtype.kind == 'U'])
    assert not tuple(v2._iter_errors(f.columns, None, (), (f,)))

    v3 = Require.LabelsOrder('a', 'b', 'c', [..., lambda s: s.dtype.kind == 'i'])
    assert tuple(v3._iter_errors(f.columns, None, (), (f,)))



#-------------------------------------------------------------------------------


def test_validate_labels_match_a1():
    idx1 = sf.Index(('a', 'b', 'c', 'd'))
    v = Require.LabelsMatch('c', 'b', 'a')
    assert not get_hints(v._iter_errors(idx1, None, (), ()))

def test_validate_labels_match_a2():
    idx1 = sf.Index(('a', 'b', 'd'))
    v = Require.LabelsMatch('a', 'b', 'c')
    assert get_hints(v._iter_errors(idx1, None, (), ()))

def test_validate_labels_match_a3():
    idx1 = sf.Index((10, 20, 30))
    v = Require.LabelsMatch(10, 30)
    assert not get_hints(v._iter_errors(idx1, None, (), ()))

def test_validate_labels_match_a4():
    idx1 = sf.Index((10, 'a', 'c', 40))
    v = Require.LabelsMatch(10, 'c')
    assert not get_hints(v._iter_errors(idx1, None, (), ()))

    idx2 = sf.Index(('a', 40))
    assert get_hints(v._iter_errors(idx2, None, (), ()))


def test_validate_labels_match_b1():
    idx1 = sf.Index(('a', 'b', 'c', 'd'))
    v1 = Require.LabelsMatch('c', {'b', 'a'})
    assert not get_hints(v1._iter_errors(idx1, None, (), ()))

    v2 = Require.LabelsMatch('c', {'b', 'x'})
    assert not get_hints(v2._iter_errors(idx1, None, (), ()))

    v3 = Require.LabelsMatch('c', {'y', 'x'})
    assert get_hints(v3._iter_errors(idx1, None, (), ()))


def test_validate_labels_match_b2():
    idx1 = sf.Index(('a', 'b', 'c', 'd'))
    v1 = Require.LabelsMatch({'c', 'd'}, {'b', 'a'})
    assert not get_hints(v1._iter_errors(idx1, None, (), ()))

    idx2 = sf.Index(('a', 'c', 'd'))
    assert not get_hints(v1._iter_errors(idx2, None, (), ()))

    idx3 = sf.Index(('y', 'x'))
    assert len(get_hints(v1._iter_errors(idx3, None, (), ()))) == 2



def test_validate_labels_match_c1():
    idx1 = sf.Index(('aaa', 'bbb', 'ccc', 'dcc'))
    v1 = Require.LabelsMatch('bbb', re.compile('cc') )
    assert not get_hints(v1._iter_errors(idx1, None, (), ()))

    idx2 = sf.Index(('aaa', 'bbb', 'dcc'))
    assert not get_hints(v1._iter_errors(idx2, None, (), ()))

    idx3 = sf.Index(('aaa', 'bbb', 'c_c'))
    assert get_hints(v1._iter_errors(idx3, None, (), ()))

def test_validate_labels_match_c2():
    idx1 = sf.Index(('aaa', 'bbb', 'ccc'))
    v1 = Require.LabelsMatch('bbb', re.compile('^cc') )
    assert not get_hints(v1._iter_errors(idx1, None, (), ()))

    idx2 = sf.Index(('aaa', 'bbb', 'dcc'))
    assert get_hints(v1._iter_errors(idx2, None, (), ()))


def test_validate_labels_match_d1():
    s1 = sf.Series(('a', 'b', 'c', 'd'))
    v = Require.LabelsMatch('c', 'b', 'a')
    # returns error for Series
    assert get_hints(v._iter_errors(s1, None, (), ()))



def test_validate_labels_match_h1():
    records = (
            (1, 3, True, 'y'),
            (4, 100, False, 'x'),
            (3, 8, True, 'q'),
            )

    f = sf.Frame.from_records(records,
            columns=('a', 'b', 'c', 'd'),
            dtypes=(np.int64, np.int64, np.bool_, np.str_)
            )

    v1 = Require.LabelsMatch('c', 'd')
    assert not tuple(v1._iter_errors(f.columns, None, (), (f,)))

    v2 = Require.LabelsMatch('c', 'd', 'a', 'b')
    assert not tuple(v2._iter_errors(f.columns, None, (), (f,)))

    v3 = Require.LabelsMatch(['b', lambda s: s.dtype.kind == 'i'], ['c', lambda s: s.dtype.kind == 'b'], 'd')
    assert not tuple(v3._iter_errors(f.columns, None, (), (f,)))

    # fails validation
    v4 = Require.LabelsMatch(['b', lambda s: s.dtype.kind == 'f'], 'd')
    assert tuple(v4._iter_errors(f.columns, None, (), (f,)))


def test_validate_labels_match_h2():
    records = (
            (1, 3, True, 'y'),
            (4, 100, False, 'x'),
            (3, 8, True, 'q'),
            )

    f = sf.Frame.from_records(records,
            columns=('a', 'b', 'c', 'd'),
            dtypes=(np.int64, np.int64, np.bool_, np.str_)
            )

    v1 = Require.LabelsMatch({'a', 'b'}, {'c', 'd'})
    assert not tuple(v1._iter_errors(f.columns, None, (), (f,)))

    v2 = Require.LabelsMatch([{'a', 'b'}, lambda s: s.dtype.kind == 'i'],)
    assert not tuple(v2._iter_errors(f.columns, None, (), (f,)))


def test_validate_labels_match_h3():
    records = (
            (1.5, 3.2, True, 'y'),
            (4.4, 100, False, 'x'),
            (3.2, 8.1, True, 'q'),
            )

    f = sf.Frame.from_records(records,
            columns=('a', 'b', 'c', 'd'),
            dtypes=(np.float64, np.float64, np.bool_, np.str_)
            )

    v2 = Require.LabelsMatch([{'a', 'b'}, lambda s: s.dtype.kind == 'i'],)
    assert len(tuple(v2._iter_errors(f.columns, None, (), (f,)))) == 2


def test_validate_labels_match_h4():
    records = (
            (1, 3.2, True, 'y'),
            (4, 100, False, 'x'),
            (3, 8.1, True, 'q'),
            )

    f = sf.Frame.from_records(records,
            columns=('a', 'b', 'c', 'd'),
            dtypes=(np.int64, np.float64, np.bool_, np.str_)
            )

    v2 = Require.LabelsMatch([{'a', 'b'}, lambda s: s.dtype.kind == 'i'],)
    assert len(tuple(v2._iter_errors(f.columns, None, (), (f,)))) == 1


def test_validate_labels_match_i1():
    records = (
            (1, 3.2, True, 'y'),
            (4, 100, False, 'x'),
            (3, 8.1, True, 'q'),
            )

    f = sf.Frame.from_records(records,
            columns=('aa', 'ab', 'ac', 'ad'),
            dtypes=(np.int64, np.float64, np.bool_, np.str_)
            )

    v1 = Require.LabelsMatch([re.compile('a'), lambda s: s.dtype.kind == 'M'],)
    assert len(tuple(v1._iter_errors(f.columns, None, (), (f,)))) == 4

    v2 = Require.LabelsMatch([re.compile('a'), lambda s: s.dtype.kind == 'b'],)
    assert len(tuple(v2._iter_errors(f.columns, None, (), (f,)))) == 3

    v3 = Require.LabelsMatch([re.compile('a'), lambda s: len(s) == 3],)
    assert not tuple(v3._iter_errors(f.columns, None, (), (f,)))



def test_validate_labels_match_j1():
    records = (
            (1.1, 3.2, True, 'y'),
            (4.1, 100, False, 'x'),
            (3.1, 8.1, True, 'q'),
            )

    f = sf.Frame.from_records(records,
            columns=('aa', 'ab', 'ac', 'qq'),
            dtypes=(np.float64, np.float64, np.bool_, np.str_)
            )

    v1 = Require.LabelsMatch(
            [{'aa', 'ab'}, lambda s: (s > 0).all()],
            )
    assert len(tuple(v1._iter_errors(f.columns, None, (), (f,)))) == 0

    v1 = Require.LabelsMatch(
            [{'aa', 'ab'}, lambda s: (s > 0).all()],
            [re.compile('a'), lambda s: s.dtype.kind == 'f'],
            )
    assert len(tuple(v1._iter_errors(f.columns, None, (), (f,)))) == 1

    v1 = Require.LabelsMatch(
            [{'aa', 'ab'}, lambda s: (s > 0).all()],
            [re.compile('a'), lambda s: s.dtype.kind == 'f'],
            ['aa', lambda s: s.sum() < 8],
            )
    assert len(tuple(v1._iter_errors(f.columns, None, (), (f,)))) == 2

#-------------------------------------------------------------------------------

def test_validate_shape_a():
    records = (
            (1, 3, True, 'y'),
            (4, 100, False, 'x'),
            (3, 8, True, 'q'),
            )

    f = sf.Frame.from_records(records,
            columns=('a', 'b', 'c', 'd'),
            dtypes=(np.int64, np.int64, np.bool_, np.str_)
            )
    v1 = Require.Shape(..., 4)
    assert not tuple(v1._iter_errors(f, None, (), ()))

    v2 = Require.Shape(..., 3)
    assert tuple(v2._iter_errors(f, None, (), ()))

    v3 = Require.Shape(3, 3)
    assert tuple(v3._iter_errors(f, None, (), ()))

    v4 = Require.Shape(3, 4)
    assert not tuple(v4._iter_errors(f, None, (), ()))

    v5 = Require.Shape(3, ...)
    assert not tuple(v5._iter_errors(f, None, (), ()))

    v6 = Require.Shape(3) # this specifies a 1D shape
    assert tuple(v6._iter_errors(f, None, (), ()))

    v7 = Require.Shape(5, ...)
    assert tuple(v7._iter_errors(f, None, (), ()))

def test_validate_shape_b():
    with pytest.raises(TypeError):
        v1 = Require.Shape(None, 4)


#-------------------------------------------------------------------------------

def test_validate_apply_a():
    idx1 = sf.Index(('a', 'b', 'c'))
    v1 = Require.Apply(lambda i: 'b' in i)
    assert not get_hints(v1._iter_errors(idx1, None, (), ()))

    v2 = Require.Apply(lambda i: 'q' in i)
    assert get_hints(v2._iter_errors(idx1, None, (), ())) == ("Index failed validation with <lambda>",)



#-------------------------------------------------------------------------------
@skip_win
def test_check_error_display_a():

    records = (
            (1, 3, True),
            (4, 100, False),
            (3, 8, True),
            )
    h1 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.str_],
            np.int_,
            np.int_,
            np.bool_]

    index = sf.IndexDate(('2022-01-03', '2022-02-05', '2018-04-02'))
    f: h1 = sf.Frame.from_records(records,
            columns=('a', 'b', 'c'),
            index=index,
            )

    h2 = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.Index[np.int_],
            np.int_,
            np.int_,
            np.str_]

    with pytest.raises(TypeError):
        TypeClinic(f).check(h2)
    try:
        TypeClinic(f).check(h2)
    except TypeError as e:
        if IS_NP2:
            assert scrub_str(str(e)) == 'In Frame[IndexDate, Index[int64], int64, int64, str_] Expected str_, provided bool invalid In Frame[IndexDate, Index[int64], int64, int64, str_] Index[int64] Expected int64, provided str_ invalid'
        else:
            assert scrub_str(str(e)) == 'In Frame[IndexDate, Index[int64], int64, int64, str_] Expected str_, provided bool_ invalid In Frame[IndexDate, Index[int64], int64, int64, str_] Index[int64] Expected int64, provided str_ invalid'


#-------------------------------------------------------------------------------
def test_type_clinic_a():
    records = (
            (1, True, 20, True),
            (30, False, 100, False),
            (54, False, 8, True),
            )
    index = sf.IndexDate(('2022-01-03', '2022-02-05', '2018-04-02'))
    columns = sf.IndexHierarchy.from_product(('a', 'b'), (True, False))
    f = sf.Frame.from_records(records, columns=columns, index=index, dtypes=(np.int64, np.bool_, np.int64, np.bool_))

    h = sf.Frame[sf.IndexDate, # type: ignore[type-arg]
            sf.IndexHierarchy[sf.Index[np.str_], sf.Index[np.int64]],
            np.int64,
            np.bool_,
            np.int64,
            np.int64,
            ]

    assert str(TypeClinic(f)(h)) == '<ClinicResult: 2 errors>'
    post = TypeClinic(f)(h).to_str()

    if IS_NP2:
        assert post == '\nIn Frame[IndexDate, IndexHierarchy[Index[str_], Index[int64]], int64, bool, int64, int64]\n└── Expected int64, provided bool invalid\nIn Frame[IndexDate, IndexHierarchy[Index[str_], Index[int64]], int64, bool, int64, int64]\n└── IndexHierarchy[Index[str_], Index[int64]]\n    └── Index[int64]\n        └── Expected int64, provided bool invalid'
    else:
        assert post == '\nIn Frame[IndexDate, IndexHierarchy[Index[str_], Index[int64]], int64, bool_, int64, int64]\n└── Expected int64, provided bool_ invalid\nIn Frame[IndexDate, IndexHierarchy[Index[str_], Index[int64]], int64, bool_, int64, int64]\n└── IndexHierarchy[Index[str_], Index[int64]]\n    └── Index[int64]\n        └── Expected int64, provided bool_ invalid'


def test_type_clinic_to_hint_a():
    s = sf.Series((3, 2), index=sf.Index(('a', 'b')), dtype=np.int64)
    assert TypeClinic(s).to_hint() == sf.Series[sf.Index[np.str_], np.int64]

def test_type_clinic_to_hint_b():
    s = sf.Index(('a', 'b'))
    assert TypeClinic(s).to_hint() == sf.Index[np.str_]

def test_type_clinic_to_hint_c():
    s = sf.IndexHierarchy.from_product(('a', 'b'), (True, False))
    assert TypeClinic(s).to_hint() == sf.IndexHierarchy[sf.Index[np.str_], sf.Index[np.bool_]]


def test_type_clinic_to_hint_d():
    records = ((1, 3, True), (3, 8, True),)
    index = sf.IndexDate(('2022-01-03', '2018-04-02'))
    f = sf.Frame.from_records(records,
            columns=('a', 'b', 'c'),
            index=index,
            dtypes=(np.int64, np.int64, np.bool_)
            )
    h = TypeClinic(f).to_hint()
    assert h == sf.Frame[sf.IndexDate, sf.Index[np.str_], np.int64, np.int64, np.bool_]

def test_type_clinic_to_hint_e():
    assert TypeClinic(3).to_hint() == int
    assert TypeClinic('foo').to_hint() == str
    assert TypeClinic(str).to_hint() == tp.Type[str]
    assert TypeClinic(sf.Frame).to_hint() == tp.Type[sf.Frame]

@skip_pyle310
def test_type_clinic_to_hint_f():
    assert TypeClinic(np.dtype(np.float64)).to_hint() == np.dtype[np.float64]
    assert TypeClinic(np.array([False, True])).to_hint() == np.ndarray[np.dtype[np.bool_]]

def test_type_clinic_to_hint_g1():
    assert TypeClinic((3, 'foo', False)).to_hint() == tuple[int, str, bool]

def test_type_clinic_to_hint_h1():
    assert TypeClinic([3, 1, 2]).to_hint() == list[int]
    assert TypeClinic([]).to_hint() == list[tp.Any]
    assert TypeClinic([False, True, False]).to_hint() == list[bool]
    assert TypeClinic([False, True, 1, 2]).to_hint() == list[tp.Union[bool, int]]

def test_type_clinic_to_hint_j1():
    assert TypeClinic({}).to_hint() == dict[tp.Any, tp.Any]

    assert TypeClinic({3: 'a'}).to_hint() == dict[int, str]
    assert TypeClinic({3: 'a', 42: 'x'}).to_hint() == dict[int, str]

    assert TypeClinic({'a': 3}).to_hint() == dict[str, int]
    assert TypeClinic({'a': 3, 'x': 30}).to_hint() == dict[str, int]

def test_type_clinic_to_hint_j2():

    assert TypeClinic({3: 'a', 42: 'x', 1.2: 'y'}).to_hint() == dict[tp.Union[int, float], str]

    assert TypeClinic({'a': 3, 'x': 30, 'z': 10.5}).to_hint() == dict[str, tp.Union[int, float]]

def test_type_clinic_to_hint_j3():

    assert TypeClinic({3: 'a', 42: 'x', 1.2: 'y', 30: b'q'}).to_hint() == dict[tp.Union[int, float], tp.Union[str, bytes]]

    assert TypeClinic({'a': 3, 'x': 30, b'z': 10.5}).to_hint() == dict[tp.Union[str, bytes], tp.Union[int, float]]

#-------------------------------------------------------------------------------
def test_type_clinic_warn_a():
    idx = sf.IndexDate(('2022-01-03', '2018-04-02'))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        TypeClinic(idx).warn(sf.IndexSecond, category=DeprecationWarning)
        assert 'Expected IndexSecond, provided IndexDate invalid' in str(w[0])


#-------------------------------------------------------------------------------
def test_via_type_clinic_a():
    s = sf.Series(('a', 'b'), index=(('x', 'y')))
    assert str(s.via_type_clinic) == 'Series[Index[str_], str_]'
    assert s.via_type_clinic(s.via_type_clinic.to_hint()).validated

def test_via_type_clinic_b():
    s = sf.Series(('a', 'b'), index=(('x', 'y')))

    with pytest.raises(TypeError):
        s.via_type_clinic.check(sf.Series[sf.IndexDate, np.str_])


#-------------------------------------------------------------------------------
def test_type_clinic_typevar_a():

    T = tp.TypeVar('T', bound=np.generic)
    h1 = sf.Frame[sf.Index[T],
            sf.Index[T],
            tp.Unpack[tp.Tuple[tp.Any, ...]],
            ]

    records = ((1, 3, True), (3, 8, True),)
    f1 = sf.Frame.from_records(records,
            columns=('a', 'b', 'c'),
            index=np.array((1, 2), np.int64),
            )
    # NOTE: this is valid if we interpret the bound as simply meaning that the any of the values must independently by subclasses of the bound; the observed values does not set the type
    assert scrub_str(f1.via_type_clinic(h1).to_str()) == 'In Frame[Index[~T: generic], Index[~T: generic], Unpack[Tuple[Any, ...]]] Index[~T: generic] ~T: generic Expected int64, provided str_ invalid'

def test_type_clinic_typevar_b1():

    T = tp.TypeVar('T', np.int64, np.float64)
    h1 = sf.Frame[sf.Index[T],
            sf.Index[T],
            tp.Unpack[tp.Tuple[tp.Any, ...]],
            ]

    records = ((1, 3, True), (3, 8, True),)
    f1 = sf.Frame.from_records(records,
            columns=sf.Index((10, 20, 30), dtype=np.int64),
            index=sf.Index((1, 2), dtype=np.int64),
            )
    assert TypeClinic(f1)(h1).validated

def test_type_clinic_typevar_b2():

    T = tp.TypeVar('T', np.int64, np.float64)
    h1 = sf.Frame[sf.Index[T],
            sf.Index[T],
            tp.Unpack[tp.Tuple[tp.Any, ...]],
            ]

    records = ((1, 3, True), (3, 8, True),)
    f1 = sf.Frame.from_records(records,
            columns=sf.Index((10, 20, 30), dtype=np.int64),
            index=sf.Index((1, 2), dtype=np.float64),
            )
    cr = TypeClinic(f1)(h1)
    assert scrub_str(cr.to_str()) == 'In Frame[Index[~T: (int64, float64)], Index[~T: (int64, float64)], Unpack[Tuple[Any, ...]]] Index[~T: (int64, float64)] ~T: (int64, float64) Expected float64, provided int64 invalid'

def test_type_clinic_typevar_c():

    T = tp.TypeVar('T')

    h1 = sf.Frame[sf.Index[T],
            sf.Index[T],
            tp.Unpack[tp.Tuple[tp.Any, ...]],
            ]

    records = ((1, 3, True), (3, 8, True),)
    f1 = sf.Frame.from_records(records,
            columns=sf.Index((10, 20, 30), dtype=np.int64),
            index=('a', 'b'),
            )

    cr = TypeClinic(f1)(h1)
    assert scrub_str(cr.to_str()) == 'In Frame[Index[~T], Index[~T], Unpack[Tuple[Any, ...]]] Index[~T] ~T Expected str_, provided int64 invalid'

@skip_pyle310
def test_type_clinic_typevar_d1():

    class A: ...
    class A1(A): ...
    class A2(A): ...
    class B: ...
    class B1(B): ...
    class B2(B): ...
    class C: ...
    class C1(C): ...
    class C2(C): ...

    T = tp.TypeVar('T', bound=tp.Union[A, B, C])
    h = tp.Tuple[T, T, T]

    v1 = (A2(), C1(), B2())
    cr = TypeClinic(v1)(h)
    assert cr.validated

    v2 = (A2(), C1(), A1()) # we specialized the Union to A1, not A2
    cr = TypeClinic(v2)(h)
    assert scrub_str(cr.to_str()) == 'In Tuple[~T: Union[A, B, C], ~T: Union[A, B, C], ~T: Union[A, B, C]] ~T: Union[A, B, C] Union[A2, B, C1] Expected A2, provided A1 invalid In Tuple[~T: Union[A, B, C], ~T: Union[A, B, C], ~T: Union[A, B, C]] ~T: Union[A, B, C] Union[A2, B, C1] Expected B, provided A1 invalid In Tuple[~T: Union[A, B, C], ~T: Union[A, B, C], ~T: Union[A, B, C]] ~T: Union[A, B, C] Union[A2, B, C1] Expected C1, provided A1 invalid'

    v3 = (A2(), A2(), A1()) # we specialized the Union to A1, not A2
    cr = TypeClinic(v3)(h)
    assert len(cr) == 3

    v4 = (C2(), A2(), C1()) # we specialized the Union to A1, not A2
    cr = TypeClinic(v4)(h)
    assert len(cr) == 3

def test_type_clinic_typevar_d2():

    class A: ...
    class A1(A): ...
    class A2(A): ...
    class B: ...
    class B1(B): ...
    class B2(B): ...
    class C: ...
    class C1(C): ...
    class C2(C): ...

    T = tp.TypeVar('T', bound=tp.Union[A, B, C])
    h = tp.Tuple[T, T, T, T, T, T]

    v1 = (A2(), C1(), B2(), A2(), C1(), B2())
    cr = TypeClinic(v1)(h)
    assert cr.validated

    v2 = (A2(), C2(), C2(), B1(), C2(), B1())
    cr = TypeClinic(v2)(h)
    assert cr.validated

@skip_pyle310
def test_type_clinic_typevar_d3():

    class A: ...
    class A1(A): ...
    class A2(A): ...
    class B: ...
    class B1(B): ...
    class B2(B): ...
    class C: ...
    class C1(C): ...
    class C2(C): ...

    T = tp.TypeVar('T', bound=tp.Union[A, B, C])
    h = tp.Tuple[T, T, T, T, T, T]

    v = (A2(), C2(), C2(), B1(), C1(), B1())
    cr = TypeClinic(v)(h)
    assert not cr.validated
    assert scrub_str(cr.to_str()) == 'In Tuple[~T: Union[A, B, C], ~T: Union[A, B, C], ~T: Union[A, B, C], ~T: Union[A, B, C], ~T: Union[A, B, C], ~T: Union[A, B, C]] ~T: Union[A, B, C] Union[A2, B1, C2] Expected A2, provided C1 invalid In Tuple[~T: Union[A, B, C], ~T: Union[A, B, C], ~T: Union[A, B, C], ~T: Union[A, B, C], ~T: Union[A, B, C], ~T: Union[A, B, C]] ~T: Union[A, B, C] Union[A2, B1, C2] Expected B1, provided C1 invalid In Tuple[~T: Union[A, B, C], ~T: Union[A, B, C], ~T: Union[A, B, C], ~T: Union[A, B, C], ~T: Union[A, B, C], ~T: Union[A, B, C]] ~T: Union[A, B, C] Union[A2, B1, C2] Expected C2, provided C1 invalid'


#-------------------------------------------------------------------------------

def test_call_guard_typevar_a():

    T1 = tp.TypeVar('T1')
    T2 = tp.TypeVar('T2')

    @CallGuard.check
    def process1(a: sf.Series[sf.Index[T1], T2]) -> sf.Series[sf.Index[T1], T2]:
        return a * 2

    _ = process1(sf.Series((1.2, 5.4), index=sf.Index(('a', 'b'))))
    _ = process1(sf.Series(('a', 'b'), index=sf.Index((1.2, 5.3))))

    T3 = tp.TypeVar('T3')
    T4 = tp.TypeVar('T4')

    @CallGuard.warn
    def process2(a: sf.Series[sf.Index[T3], T4]) -> sf.Series[sf.Index[T3], T4]:
        return sf.Series(('a', 'b'), index=sf.Index(('a', 'b')))

    with warnings.catch_warnings(record=True) as w:
        # expected to return float values (based on input) but returned string values
        _ = process2(sf.Series((1.2, 5.4), index=sf.Index(('a', 'b'))))
        assert scrub_str(str(w[0].message)) == 'In return of (a: Series[Index[~T3], ~T4]) -> Series[Index[~T3], ~T4] Series[Index[~T3], ~T4] ~T4 Expected float64, provided str_ invalid'

    with warnings.catch_warnings(record=True) as w:
        # expected to return float values (based on input) but returned string values
        _ = process2(sf.Series((False, True), index=sf.Index(('a', 'b'))))
        if IS_NP2:
            assert scrub_str(str(w[0].message)) == 'In return of (a: Series[Index[~T3], ~T4]) -> Series[Index[~T3], ~T4] Series[Index[~T3], ~T4] ~T4 Expected bool, provided str_ invalid'
        else:
            assert scrub_str(str(w[0].message)) == 'In return of (a: Series[Index[~T3], ~T4]) -> Series[Index[~T3], ~T4] Series[Index[~T3], ~T4] ~T4 Expected bool_, provided str_ invalid'


def test_call_guard_typevar_b():

    T = tp.TypeVar('T')

    @sf.CallGuard.warn
    def process1(
            a: sf.Series[sf.Index[T], np.number[tp.Any]],
            b: sf.Series[sf.Index[T], np.number[tp.Any]],
            ) -> sf.Series[sf.Index[T], np.number[tp.Any]]:
        return a + b

    with warnings.catch_warnings(record=True) as w:
        _ = process1(sf.Series((1.2, 5.4), index=('a', 'b')), sf.Series((4, 5), index=np.array((30, 10), dtype=np.int64), dtype=np.int64))
        assert scrub_str(str(w[0].message)) == 'In args of (a: Series[Index[~T], number[Any]], b: Series[Index[~T], number[Any]]) -> Series[Index[~T], number[Any]] In arg b Series[Index[~T], number[Any]] Index[~T] ~T Expected str_, provided int64 invalid'


def test_call_guard_typevar_c1():
    # based on examples here: https://stackoverflow.com/a/59937840

    T1 = tp.TypeVar('T1', bound=tp.Union[int, str])

    @sf.CallGuard.warn
    def concat1(x: tp.Iterable[T1], y: tp.Iterable[T1]) -> tp.List[T1]:
        out = list(x)
        out.extend(y)
        return out

    mix1: tp.List[tp.Union[int, str]] = [1, "a", 3]
    mix2: tp.List[tp.Union[int, str]] = [4, "x", "y"]
    all_ints = [1, 2, 3]
    all_strs = ["a", "b", "c"]

    _ = concat1(mix1, mix2) # does not error
    _ = concat1(all_ints, all_strs) # does not error
    _ = concat1(all_strs, all_strs) # does not error

@skip_pyle310
def test_call_guard_typevar_c2():
    # based on examples here: https://stackoverflow.com/a/59937840

    T1 = tp.TypeVar('T1', int, str)

    @sf.CallGuard.warn(fail_fast=True)
    def concat1(x: tp.Iterable[T1], y: tp.Iterable[T1]) -> tp.List[T1]:
        out = list(x)
        out.extend(y)
        return out

    mix1: tp.List[tp.Union[int, str]] = [1, "a", 3]
    mix2: tp.List[tp.Union[int, str]] = [4, "x", "y"]
    all_ints = [1, 2, 3]
    all_strs = ["a", "b", "c"]

    with warnings.catch_warnings(record=True) as w:
        _ = concat1(mix1, mix2) # fails
        assert scrub_str(str(w[0].message)) == "In args of (x: Iterable[~T1: (int, str)], y: Iterable[~T1: (int, str)]) -> List[~T1: (int, str)] In arg x Iterable[~T1: (int, str)] ~T1: (int, str) Expected int, provided str invalid"

    _ = concat1(all_ints, all_ints) # does not error
    _ = concat1(all_strs, all_strs) # does not error

    with warnings.catch_warnings(record=True) as w:
        _ = concat1(all_ints, all_strs) # fails
        assert scrub_str(str(w[0].message)) == "In args of (x: Iterable[~T1: (int, str)], y: Iterable[~T1: (int, str)]) -> List[~T1: (int, str)] In arg y Iterable[~T1: (int, str)] ~T1: (int, str) Expected int, provided str invalid"


def test_call_guard_typevar_d():
    T = tp.TypeVar('T', np.uint16, np.int8)

    @sf.CallGuard.warn
    def process1(
            a: sf.Series[sf.Index[str], T],
            b: sf.Series[sf.Index[str], T],
            ) -> sf.Series[sf.Index[str], T]:
        return a + b

    _ = process1(sf.Series((1.2, 5.4), index=('a', 'b'), dtype=np.int8), sf.Series((4, 5), index=('a', 'b'), dtype=np.int8))

    _ = process1(sf.Series((1.2, 5.4), index=('a', 'b'), dtype=np.uint16), sf.Series((4, 5), index=('a', 'b'), dtype=np.uint16))

    with warnings.catch_warnings(record=True) as w:
        _ = process1(sf.Series((1.2, 5.4), index=('a', 'b'), dtype=np.uint16), sf.Series((4, 5), index=('a', 'b'), dtype=np.int8))
        assert scrub_str(str(w[0].message)) == 'In args of (a: Series[Index[str], ~T: (uint16, int8)], b: Series[Index[str], ~T: (uint16, int8)]) -> Series[Index[str], ~T: (uint16, int8)] In arg b Series[Index[str], ~T: (uint16, int8)] ~T: (uint16, int8) Expected uint16, provided int8 invalid'

def test_call_guard_typevar_e():

    T1 = tp.TypeVar('T1', bound=np.number[tp.Any])
    T2 = tp.TypeVar('T2', bound=np.number[tp.Any])

    @sf.CallGuard.warn
    def process2(
            a: sf.Series[sf.Index[T1], T2],
            b: sf.Series[sf.Index[T1], T2],
            ) -> sf.Series[sf.Index[T1], T2]:
        return a + b

    _ = process2(sf.Series((1.2, 5.4), index=('a', 'b')), sf.Series((4.3, 5.1), index=('a', 'c')))


