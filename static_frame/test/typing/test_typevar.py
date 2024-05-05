# pylint: disable=C0321

import typing_extensions as tp




def test_typevar_a1() -> None:
    class A: ...
    class A1(A): ...
    class A2(A): ...

    T = tp.TypeVar('T', bound=A)

    def process(x: T, y: T) -> T:
        return x

    a1: A1 = process(A1(), A1())
    a2: A2 = process(A2(), A2())

    # NOTE: this shows that the first-encountered type sets the type; it must be a subtype of the bound, but once set, other subtypes cannot be used
    # post: A1 = process(A1(), A2()) # fails: "A2" is incompatible with "A1"
    # post: A2 = process(A2(), A1()) # fails: "A1" is incompatible with "A2"

def test_typevar_b1() -> None:
    class A: ...
    class A1(A): ...
    class B: ...
    class B1(B): ...

    T = tp.TypeVar('T', bound=tp.Union[A1, B1])

    def process(x: T, y: T) -> T:
        return x

    a1: A1 = process(A1(), A1())

    # a2: A2 = process(A2(), A2())
    # Type "A2" cannot be assigned to type "A1 | B1"
    #       Type "A2" cannot be assigned to type "A1 | B1"
    #         "A2" is incompatible with "A1"
    #         "A2" is incompatible with "B1" (reportGeneralTypeIssues)

    b1: B1 = process(B1(), B1())

    ab1: tp.Union[A1, B1] = process(A1(), B1())

    # ab2: tp.Union[A1, B1] = process(A1(), A2())
    # error: Argument of type "A2" cannot be assigned to parameter "y" of type "T@process" in function "process"
    #     "A2" is incompatible with "A1" (reportGeneralTypeIssues)

def test_typevar_b2() -> None:
    class A: ...
    class B1(A): ...
    class B2(A): ...
    class C: ...
    class D1(C): ...
    class D2(C): ...

    T = tp.TypeVar('T', bound=tp.Union[B1, D1])

    def process(x: T, y: T) -> T:
        return x

    post: D1 = process(D1(), D1())
    # post: D1 = process(D1(), B1()) # fails: "B1" is incompatible with "D1"
    # this shows the first observed value sets the type

def test_typevar_c1() -> None:
    class A: ...
    class B1(A): ...
    class B2(A): ...
    class C: ...
    class D1(C): ...
    class D2(C): ...

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
    class A: ...
    class B1(A): ...
    class B2(A): ...
    class C: ...
    class D1(C): ...
    class D2(C): ...

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
    class A: ...
    class A1(A): ...
    class A2(A): ...
    class B: ...
    class B1(B): ...
    class B2(B): ...

    T1 = tp.TypeVar('T1', bound=tp.Union[A, B])

    def concat1(x: tp.Iterable[T1], y: tp.Iterable[T1]) -> tp.List[T1]:
        out: tp.List[T1] = []
        out.extend(x)
        out.extend(y)
        return out

    mix1: tp.List[A1] = [A1(), A1()]
    mix2: tp.List[B1] = [B1(), B1()]
    mix3: tp.List[B2] = [B2(), B2()]
    mix4: tp.List[tp.Union[A1, B1]] = [A1(), B1()]
    mix5: tp.List[tp.Union[A2, B2]] = [A2(), B2()]

    a = concat1(mix1, mix2) # passes
    b: tp.List[tp.Union[A1, B2]] = concat1(mix1, mix3) # passes
    c: tp.List[tp.Union[A1, B1]] = concat1(mix4, mix4)
    # d: tp.List[tp.Union[A1, B2]] = concat1(mix4, mix5) # fails

    # error: Argument of type "list[A1 | B1]" cannot be assigned to parameter "x" of type "Iterable[T1@concat1]" in function "concat1"
    #     "list[A1 | B1]" is incompatible with "Iterable[A1 | B2]"
    #       Type parameter "_T_co@Iterable" is covariant, but "A1 | B1" is not a subtype of "A1 | B2" (reportGeneralTypeIssues)
    # error: Argument of type "list[A2 | B2]" cannot be assigned to parameter "y" of type "Iterable[T1@concat1]" in function "concat1"
    #     "list[A2 | B2]" is incompatible with "Iterable[A1 | B2]"
    #       Type parameter "_T_co@Iterable" is covariant, but "A2 | B2" is not a subtype of "A1 | B2" (reportGeneralTypeIssues)

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

    # y: A1 = process(A1(), A1(), A2(), A1()) # "A2" is incompatible with "A1" (reportGeneralTypeIssues)

    # x: C2 = process(A2(), C2(), B2(), C2()) # "A2" is incompatible with "C2" (reportGeneralTypeIssues) "B2" is incompatible with "C2" (reportGeneralTypeIssues)

def test_type_clinic_typevar_d4() -> None:

    class A: ...
    class A1(A): ...
    class A2(A): ...

    T = tp.TypeVar('T', bound=A)
    def process(a: T, b: T) -> T:
        return a

    x: A1 = process(A1(), A1()) # this passes
    y: A2 = process(A2(), A2()) # this passes
    # z: A1 = process(A2(), A1()) # fails: "A2" is incompatible with "A1" (reportGeneralTypeIssues)
    # w: A2 = process(A2(), A1()) # fails: "A1" is incompatible with "A2" (reportGeneralTypeIssues)

    # this shows that a type parameter must be a subclass of the bound, but once set, all type var usage must be the same subclass

def test_type_clinic_typevar_d5() -> None:

    class A: ...
    class A1(A): ...
    class A2(A): ...

    T = tp.TypeVar('T')
    def process(a: T, b: T) -> T:
        return a

    x = process(A1(), A1()) # this passes
    y = process(A2(), A2()) # this passes

    # z: A1 = process(A2(), A1()) # fails: A2" is incompatible with "A1" (reportGeneralTypeIssues)

def test_type_clinic_typevar_d6() -> None:

    T = tp.TypeVar('T')
    def process(a: T, b: T) -> T:
        return b

    x = process('a', 'a') # this passes
    y = process(0, 0) # this passes
    # z: int = process('a', 0) # this fails if we assign the return type
    # error: Argument of type "Literal['a']" cannot be assigned to parameter "a" of type "T@process" in function "process"
    # "Literal['a']" is incompatible with "int" (reportGeneralTypeIssues)