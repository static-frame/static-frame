import typing_extensions as tp



class A:
    pass

class B1(A):
    pass

class B2(A):
    pass

class C:
    pass

class D1(C):
    pass

class D2(C):
    pass

def test_typevar_a1() -> None:

    T = tp.TypeVar('T', bound=A)

    def process(x: T, y: T) -> T:
        return x

    post: B1 = process(B1(), B1())
    # post: B = process(B(), C()) # fails: "C" is incompatible with "B"
    # post: C = process(C(), C()) # fails: "C" is incompatible with "A"

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


# def test_typevar_c1() -> None:

#     pass