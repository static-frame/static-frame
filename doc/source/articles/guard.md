

# Improve Code Quality with Array and DataFrame Type Hints

#### How complete generic specification permits powerful static and runtime validation

<!--
Type-Hinting Arrays and DataFrames for Static Analysis and Runtime Validation
How Type Annotations of Arrays and DataFrames Improve Code
Type-Hinting Generic Arrays and DataFrames
How Type-Hinting Arrays and DataFrames Improve Your Code
-->

As tools for Python type annotations have evolved, more complex data structures can be typed, providing improved readability and static analysis. Arrays and DataFrames, as complex containers, have only recently supported complete type annotations in Python. NumPy 1.20 introduced generic specification of arrays and dtypes. Building on NumPy's foundation, StaticFrame 2.0 introduced complete type specification of DataFrames, employing NumPy primitives and variadic generics. This article demonstrates practical approaches to fully type-hinting arrays and DataFrames, and shows how the same annotations can improve code quality with both static analysis and run-time validation.


## Type Hints Improve Code Quality

Type hints improve code quality in a number of ways. Instead of using variable names or comments to communicate types, Python-object-based type annotations provide maintainable and expressive tools for type specification. These type annotations can be tested with type checkers such as ``mypy`` or ``pyright``.

Further, the same annotations can be used for run-time validation. While reliance on duck-typing over run-time validation is common in Python, run-time validation is often needed with complex data structures such as arrays and DataFrames.

Many important typing utilities are only available with the most-recent versions of Python. Fortunately, the `typing-extensions` package back-ports standard-library utilities for older versions of Python. A related challenge is that type checkers can take time to implement full support for new features: many of the examples shown here require ``mypy`` 1.9.0, released just a few months before ago.

## Elemental Type Annotations

Without type annotations, a Python function signature gives no indication of the expected types.

```python
def process0(v, q): ... # no type information
```

By adding type hints, the signature informs readers of the expected types. With modern Python, user-defined and built-in classes can be used to specify types, with additional resources (such as ``Any``, ``Iterator``, and ``Annotated``) found in the ``typing`` module.

```python
def process0(v: int, q: bool) -> list[float]: ...
```

When used with a type checker like `mypy`, code that violates the specifications of the type annotations will raise an error during static analysis (shown as comments, below). For example, providing an integer when a bool is required is an error:

```python
x = process0(v=5, q=20)
# tp_basic.py: error: Argument "q" to "process0"
# has incompatible type "int"; expected "bool"  [arg-type]
```

Static analysis can only validate known (statically defined) types. The full range of run-time inputs and outputs is often more diverse, requiring some form run-time validation. The best-of-both-worlds is possible by re-using type annotations for run-time validation. While there are a few libraries that do this (``typeguard`` and ``beartype``), StaticFrame offers ``CallGuard``, a run-time validator specialized handling array and DataFrame validations.

A Python decorator is the perfect tool for leveraging annotation for run-time validation. ``CallGuard`` offers two: ``@CallGuard.check``, which raises an information ``Exception``on error, or ``@CallGuard.warn``, which issues a warning.

Extending the `process0` function above with ``@CallGuard.check``, the same type annotations can be used at run time to raise an ``Exception`` at run-time (shown again as comments, below):

```python
import static_frame as sf

@sf.CallGuard.check
def process0(v: int, q: bool) -> list[float]:
    return [x * (0.5 if q else 0.25) for x in range(v)]

z = process0(v=5, q=20)
# static_frame.core.type_clinic.ClinicError:
# In args of (v: int, q: bool) -> list[float]
# └── Expected bool, provided int invalid
```

While type annotations must be valid Python, they are irrelevant at run-time and can be wrong: it is possible to have correctly verified types that do not reflect run-time reality. Reusing type annotations for run-time checks ensure that they reflect run-time reality.


## Array Type Annotations

Python classes that permit component type specification are "generic"; component types are specified with positional "type variables". A list of integers, for example, is annotated with ``list[int]``; a dictionary of floats keyed by tuples of integers and strings is annotated ``dict[tuple[int, str], float]``.

With NumPy 1.20, ``ndarray`` and ``dtype`` types become generic. The generic ``ndarray`` requires two arguments, a shape and a ``dtype``. As the usage of the first argument is still under development, ``Any`` is used. The second argument, `dtype`, is itself a generic that requires a type variable for a NumPy type such as ``np.int64``. NumPy also offers more general generic types such as ``np.integer[tp.Any]``.

For example, an array of Booleans is annotated ``np.ndarray[tp.Any, np.dtype[np.bool_]]``; an array of any type of integer is annotated ``np.ndarray[tp.Any, np.dtype[np.integer[tp.Any]]]``.

As generic annotations with component type specifications can become verbose, it is practical to store them as type aliases (here prefixed with the letter T). The following function specifies such aliases and then uses them with a new, array based implementation.

```python
TNDArrayInt8 = np.ndarray[tp.Any, np.dtype[np.int8]]
TNDArrayBool = np.ndarray[tp.Any, np.dtype[np.bool_]]
TNDArrayFloat64 = np.ndarray[tp.Any, np.dtype[np.float64]]

def process1(
        v: TNDArrayInt8,
        q: TNDArrayBool,
        ) -> TNDArrayFloat64:
    s: TNDArrayFloat64 = np.where(q, 0.5, 0.25)
    return v * s
```

As before, when used with ``mypy``, code that violates the specifications of the type annotations will raise an error during static analysis. For example, providing an integer when a bool is required is an error:

```python
v1: TNDArrayInt8 = np.arange(20, dtype=np.int8)
x = process1(v1, v1)
# tp_np.py: error: Argument 2 to "process1" has incompatible type
# "ndarray[Any, dtype[floating[_64Bit]]]"; expected "ndarray[Any, dtype[bool_]]"  [arg-type]
```

The interface requires 8-bit signed integers (`np.int8`); attempting to use a different sized integer is also an error:

```python
TNDArrayInt64 = np.ndarray[tp.Any, np.dtype[np.int64]]
v2: TNDArrayInt64 = np.arange(20, dtype=np.int64)
q: TNDArrayBool = np.arange(20) % 3 == 0
x = process1(v2, q)
# tp_np.py: error: Argument 1 to "process1" has incompatible type
# "ndarray[Any, dtype[signedinteger[_64Bit]]]"; expected "ndarray[Any, dtype[signedinteger[_8Bit]]]"  [arg-type]
```

While some interfaces might benefit from such narrow numeric type specifications, broader specification is possible with NumPy's generic types such as ``np.integer[tp.Any]``, ``np.signedinteger[Any]``, ``np.float[Any]``, etc. For example, we can define and test a new function that accepts any size signed integer:

```python
TNDArrayIntAny = np.ndarray[tp.Any, np.dtype[np.signedinteger[tp.Any]]]
def process2(
        v: TNDArrayIntAny, # a more flexible interface
        q: TNDArrayBool,
        ) -> TNDArrayFloat64:
    s: TNDArrayFloat64 = np.where(q, 0.5, 0.25)
    return v * s

x = process2(v1, q) # no mypy error
x = process2(v2, q) # no mypy error
```

Just as shown above with elements, these generically specified NumPy arrays can be validated at runtime if decorated with ``CallGuard.check``:


```python
@sf.CallGuard.check
def process3(v: TNDArrayIntAny, q: TNDArrayBool) -> TNDArrayFloat64:
    s: TNDArrayFloat64 = np.where(q, 0.5, 0.25)
    return v * s

x = process3(v1, q) # no error, same as mypy
x = process3(v2, q) # no error, same as mypy
v3: TNDArrayFloat64 = np.arange(20, dtype=np.float64) * 0.5
x = process3(v3, q) # error
# static_frame.core.type_clinic.ClinicError:
# In args of (v: ndarray[Any, dtype[signedinteger[Any]]],
# q: ndarray[Any, dtype[bool_]]) -> ndarray[Any, dtype[float64]]
# └── ndarray[Any, dtype[signedinteger[Any]]]
#     └── dtype[signedinteger[Any]]
#         └── Expected signedinteger, provided float64 invalid
```

StaticFrame provides utilities to extend run-time validation beyond type checking. Using the ``typing`` module's ``Annotated`` class, we can extend the type specification with one or more StaticFrame ``Require`` objects. For example, to validate that an array has a 1D shape of `(24,)`, we can replace ``TNDArrayIntAny`` with ``Annotated[TNDArrayIntAny, sf.Require.Shape(24)]``. To validate that a float array has no NaNs, we can replace ``TNDArrayFloat64`` with ``Annotated[TNDArrayFloat64, sf.Require.Apply(lambda a: ~a.insna().any())]``

Implementing a new function, we can require that all input and output arrays have the shape `(24,)`. Calling this function with the previously created shape `(20,)` arrays raises an error:

```python
@sf.CallGuard.check
def process4(
        v: tp.Annotated[TNDArrayIntAny, sf.Require.Shape(24)],
        q: tp.Annotated[TNDArrayBool, sf.Require.Shape(24)],
        ) -> tp.Annotated[TNDArrayFloat64, sf.Require.Shape(24)]:
    s: TNDArrayFloat64 = np.where(q, 0.5, 0.25)
    return v * s

x = process4(v1, q) # types pass, but Require.Shape fails
# static_frame.core.type_clinic.ClinicError:
# In args of (v: Annotated[ndarray[Any, dtype[int8]], Shape((24,))], q: Annotated[ndarray[Any, dtype[bool_]], Shape((24,))]) -> Annotated[ndarray[Any, dtype[float64]], Shape((24,))]
# └── Annotated[ndarray[Any, dtype[int8]], Shape((24,))]
#     └── Shape((24,))
#         └── Expected shape ((24,)), provided shape (20,)
```


## DataFrame Type Annotations

Just like a dictionary or an array, a DataFrame is complex data structure composed of many component types: the type of the index, the type of the columns, and the types of columns.

A challenge of generically specifying a DataFrame is that a DataFrame has a variable number of columns, where each column might be a different type. The Python ``TypeVarTuple`` variadic generic specifier, first released in Python 3.11, permits defining a variable numbers of column types.

With StaticFrame 2.0, ``Frame``, ``Series``, ``Index`` and related types become generic. Support for variable column type definitions is provided by ``TypeVarTuple``, back-ported with the implementation in ``typing-extensions``. StaticFrame might be the first DataFrame library to implement a completely generic DataFrame. Pandas does not support generic type specification.

A generic ``Frame`` requires two or more type variables: the type of the index, the type of the columns, and zero or more specifications of columnar types specified with NumPy types. A generic ``Series`` requires two type variables: the type of the index and a NumPy type for the values. The ``Index`` is itself generic, also requiring a NumPy type as a type variable.

With generic specification, a ``Series`` of floats, indexed by dates, can be annotated with ``sf.Series[sf.IndexDate, np.float64]``. A ``Frame`` with dates as index labels, strings as column labels, and columns of integers and floats, can be annotated with ``sf.Frame[sf.IndexDate, sf.Index[str_], np.int64, np.float64]``.

Given a complex ``Frame``, deriving the annotation might be difficult. StaticFrame offers the ``via_type_clinic`` interface to provide a complete generic specification for any run-time ``Frame``:

```python
>>> v4 = sf.Frame.from_fields([range(5), np.arange(3, 8) * 0.5],
columns=('a', 'b'), index=sf.IndexDate.from_date_range('2021-12-30', '2022-01-03'))
>>> v4
<Frame>
<Index>         a       b         <<U1>
<IndexDate>
2021-12-30      0       1.5
2021-12-31      1       2.0
2022-01-01      2       2.5
2022-01-02      3       3.0
2022-01-03      4       3.5
<datetime64[D]> <int64> <float64>

# get a string representation of the annotation
>>> v4.via_type_clinic
Frame[IndexDate, Index[str_], int64, float64]
```

As shown above with arrays, storing annotations as type aliases permits reuse and more concise function signatures. Below, a new function is defined with generic ``Frame`` and ``Series`` arguments fully annotated.

```python
TFrameDateInts = sf.Frame[sf.IndexDate, sf.Index[np.str_], np.int64, np.int64]
TSeriesYMBool = sf.Series[sf.IndexYearMonth, np.bool_]
TSeriesDFloat = sf.Series[sf.IndexDate, np.float64]

def process5(v: TFrameDateInts, q: TSeriesYMBool) -> TSeriesDFloat:
    t = v.index.iter_label().apply(lambda l: q[l.astype('datetime64[M]')]) # type: ignore
    s = np.where(t, 0.5, 0.25)
    return tp.cast(TSeriesDFloat, (v.via_T * s).mean(axis=1))
```

As before, ``mypy`` can be used to validate annotated interfaces. Below, a ``Frame`` without the expected column type is passed, causing ``mypy`` to error (shown as comments, below).

```python
TFrameDateIntFloat = sf.Frame[sf.IndexDate, sf.Index[np.str_], np.int64, np.float64]
v5: TFrameDateIntFloat = sf.Frame.from_fields([range(5), np.arange(3, 8) * 0.5],
columns=('a', 'b'), index=sf.IndexDate.from_date_range('2021-12-30', '2022-01-03'))

q: TSeriesYMBool = sf.Series([True, False],
index=sf.IndexYearMonth.from_date_range('2021-12', '2022-01'))

x = process5(v5, q)
# tpst_frame.py: error: Argument 1 to "process5" has incompatible type
# "Frame[IndexDate, Index[str_], signedinteger[_64Bit], floating[_64Bit]]"; expected
# "Frame[IndexDate, Index[str_], signedinteger[_64Bit], signedinteger[_64Bit]]"  [arg-type]
```

To use the same type hints for run-time validation, the ``sf.CallGuard.check`` decorator can be applied. Below, a ``Frame`` of two three integer columns is provided where a ``Frame`` of two columns is expected.

```python
# a Frame of three columns of integers
TFrameDateIntIntInt = sf.Frame[sf.IndexDate, sf.Index[np.str_], np.int64, np.int64, np.int64]
v6: TFrameDateIntIntInt = sf.Frame.from_fields([range(5), range(3, 8), range(1, 6)],
columns=('a', 'b', 'c'), index=sf.IndexDate.from_date_range('2021-12-30', '2022-01-03'))

x = process5(v6, q)
# static_frame.core.type_clinic.ClinicError:
# In args of (v: Frame[IndexDate, Index[str_], signedinteger[_64Bit], signedinteger[_64Bit]],
# q: Series[IndexYearMonth, bool_]) -> Series[IndexDate, float64]
# └── Frame[IndexDate, Index[str_], signedinteger[_64Bit], signedinteger[_64Bit]]
#     └── Expected Frame has 2 dtype, provided Frame has 3 dtype
```

It might not be practical to explicitly annotate every column of every ``Frame``: it is common for interfaces to permit usage with ``Frame`` of variable numbers of columns. The Python 3.11 ``TypeVarTuple`` supports this through the usage of ``*tuple`` expressions (back-ported with the ``Unpack`` annotation). For example, the function above could be defined to take any number of integer columns with that annotation ``Frame[IndexDate, Index[str_], *tuple[np.int64, ...]]``, where ``*tuple[np.int64, ...]]`` means zero or more integer columns.

The same implementation can be annotated with a far more general specificaion of columnar types. Below, we type the columns as ``np.number[tp.Any]``, permitting any type of numeric NumPy type, and use a ``*tuple`` expression to permit any number of numeric columns: ``*tuple[np.number[tp.Any], ...]``.

```python
TFrameDateNums = sf.Frame[sf.IndexDate, sf.Index[np.str_], *tuple[np.number[tp.Any], ...]]

@sf.CallGuard.check
def process6(v: TFrameDateNums, q: TSeriesYMBool) -> TSeriesDFloat:
    t = v.index.iter_label().apply(lambda l: q[l.astype('datetime64[M]')]) # type: ignore
    s = np.where(t, 0.5, 0.25)
    return tp.cast(TSeriesDFloat, (v.via_T * s).mean(axis=1))

x = process6(v5, q) # a Frame with integer, float columns passes
x = process6(v6, q) # a Frame with three integer columns passes
```

## Conclusion

Python type annotaitons can make static analysis of types a valuable check of code quality. Up until recently, only the the outer-most type could be defined: an interface might take an array or a DataFrame, but no specification of the types contained in those containers was possible. More recently, complete specification of component types is possible, permitting much more powerful static analysis of types.

Providing correct type annotations is an investment. Reusing those annotations for run-time checks provides the best of both worlds. The StaticFrame ``CallGuard`` run-time type checker is specialized to correctly evaluate fully specified generic NumPy types, as well as all generic StaticFrame containers and DataFrames.











