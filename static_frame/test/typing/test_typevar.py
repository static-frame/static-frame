# pylint: disable=C0321

import typing_extensions as tp

class A: ...
class B1(A): ...
class B2(A): ...
class C: ...
class D1(C): ...
class D2(C): ...

def test_typevar_a1() -> None:

    T = tp.TypeVar('T', bound=A)

    def process(x: T, y: T) -> T:
        return x

    post: B1 = process(B1(), B1())
    # NOTE: this shows that the first-encountered type sets the type; it seems that Pyright uses the last-encountered value...
    # post: B1 = process(B1(), B2()) # fails: "B2" is incompatible with "B1"
    # post: B2 = process(B2(), B1()) # fails: "B1" is incompatible with "B2"

def test_typevar_a2() -> None:

    T = tp.TypeVar('T', bound=A)

    def process(x: T, y: T) -> T:
        return x

    post: B2 = process(B2(), B2())


def test_typevar_b1() -> None:

    T = tp.TypeVar('T', bound=tp.Union[B1, D1])

    def process(x: T, y: T) -> T:
        return x

    post: B1 = process(B1(), B1())
    # post: B1 = process(B1(), D1()) # fails: "D1" is incompatible with "B1"

def test_typevar_b2() -> None:

    T = tp.TypeVar('T', bound=tp.Union[B1, D1])

    def process(x: T, y: T) -> T:
        return x

    post: D1 = process(D1(), D1())
    # post: D1 = process(D1(), B1()) # fails: "B1" is incompatible with "D1"
    # this shows the first observed value sets the type

def test_typevar_c1() -> None:
    # https://stackoverflow.com/questions/59933946/difference-between-typevart-a-b-and-typevart-bound-uniona-b
    T1 = tp.TypeVar('T1', bound=tp.Union[int, str])

    def concat1(x: tp.Iterable[T1], y: tp.Iterable[T1]) -> tp.List[T1]:
        out: tp.List[T1] = []
        out.extend(x)
        out.extend(y)
        return out

    mix1: tp.List[tp.Union[int, str]] = [1, "a", 3]
    mix2: tp.List[tp.Union[int, str]] = [4, "x", "y"]
    all_ints = [1, 2, 3]
    all_strs = ["a", "b", "c"]

    a = concat1(mix1, mix2) # passes
    b: tp.List[tp.Union[int, str]] = concat1(all_ints, all_strs)
    c = concat1(all_strs, all_strs)


def test_typevar_c2() -> None:
    T1 = tp.TypeVar('T1', bound=tp.Union[A, C])

    def concat1(x: tp.Iterable[T1], y: tp.Iterable[T1]) -> tp.List[T1]:
        out: tp.List[T1] = []
        out.extend(x)
        out.extend(y)
        return out

    mix1: tp.List[tp.Union[B1, D1]] = [B1(), D1()]
    mix2: tp.List[tp.Union[B1, D1]] = [B1(), D1()]
    mix3: tp.List[tp.Union[B2, D2]] = [B2(), D2()]
    mix4: tp.List[tp.Union[B1, D2]] = [B1(), D2()]

    # all_ints = [1, 2, 3]
    # all_strs = ["a", "b", "c"]

    a = concat1(mix1, mix2) # passes
    # a = concat1(mix3, mix3) # fails as we have already assigned type var to b1, d1
    # a = concat1(mix1, mix4) # Argument 2 to "concat1" has incompatible type "list[B1 | D2]"; expected "Iterable[B1 | D1]"  [arg-type]


def test_typevar_c3() -> None:
    T1 = tp.TypeVar('T1', bound=tp.Union[A, C])

    def concat1(x: tp.Iterable[T1], y: tp.Iterable[T1]) -> tp.List[T1]:
        out: tp.List[T1] = []
        out.extend(x)
        out.extend(y)
        return out

    mix1: tp.List[B1] = [B1(), B1()]
    mix2: tp.List[D1] = [D1(), D1()]
    mix3: tp.List[D2] = [D2(), D2()]
    mix4: tp.List[tp.Union[B1, D1]] = [B1(), D1()]
    mix5: tp.List[tp.Union[B2, D2]] = [B2(), D2()]

    a = concat1(mix1, mix2) # passes
    b = concat1(mix1, mix3) # passes, which is surprising as we used D1 in mix2 above
    c = concat1(mix4, mix4)
    d = concat1(mix4, mix5)

def test_type_clinic_typevar_d1() -> None:

    class A: ...
    class A1(A): ...
    class A2(A): ...
    class B: ...
    class B1(B): ...
    class B2(B): ...
    class C: ...
    class C1(C): ...
    class C2(C): ...

    T = tp.TypeVar('T', bound=tp.Union[A, C])
    def process(a: tp.Tuple[T, T, T, T, T, T]) -> T:
        return a[0]

    x = process((A2(), C2(), C2(), A1(), C1(), C1())) # passes
    # this shows that the union is not specialized, nor not set on first encuontered

def test_type_clinic_typevar_d2() -> None:

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
    def process(a: T, b: T, c: T, d: T) -> T:
        return a

    x = process(A2(), C2(), B2(), C2()) # this passes
    x = process(A1(), C2(), A2(), C1()) # this passes
    # this shows that a union tyupe as bound is not specialized by subclass; the bound remains the requirements

def test_type_clinic_typevar_d3() -> None:

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
    def process(a: T, b: T, c: T, d: T) -> T:
        return a

    x = process(A2(), C2(), B2(), C2()) # this passes
    x = process(A1(), C2(), A2(), C1()) # this passes
    # this shows that a union type as bound is not specialized by subclass; the bound remains the requirements

def test_type_clinic_typevar_d4() -> None:

    class A: ...
    class A1(A): ...
    class A2(A): ...

    T = tp.TypeVar('T', bound=A)
    def process(a: T, b: T) -> T:
        return a

    x = process(A1(), A1()) # this passes
    y = process(A2(), A2()) # this passes
    z = process(A2(), A1()) # this passes

    # this shows that a union type as bound is not specialized by subclass; the bound remains the requirement

def test_type_clinic_typevar_d5() -> None:

    class A: ...
    class A1(A): ...
    class A2(A): ...

    T = tp.TypeVar('T')
    def process(a: T, b: T) -> T:
        return a

    x = process(A1(), A1()) # this passes
    y = process(A2(), A2()) # this passes
    z = process(A2(), A1()) # this passes

def test_type_clinic_typevar_d6() -> None:

    T = tp.TypeVar('T')
    def process(a: T, b: T) -> T:
        return b

    x = process('a', 'a') # this passes
    y = process(0, 0) # this passes
    z = process('a', 0) # this passes, which does not make sense!
    a = process(0, 'q') # this passes, which does not make sense!
