

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

While Python's typing resources are still evolving, their usage already improves code quality. Instead of using variable names or comments to communicate types, code-object annotations provide readable, maintainable, and powerful tools for type specification. These type annotations can be used for static analysis with `mypy` or `pyright` type checkers.

Providing additional benefits, the same annotations can be used for run-time validation. While reliance on duck-typing over run-time validation is common in Python, run-time validation is often needed when working with arrays and DataFrames.

Many important typing utilities are only available with the most-recent versions of Python. Fortunately, the `typing-extensions` package back-ports standard library utilities for older versions of Python. A related challenge is that type checkers can take significant time to implement full support for new features: many of the examples shown here require `mypy` 1.9.0, released just a few months before ago.

## Elemental Type Annotations

Without type annotations, a Python function signature gives no indication of the expected argument or return types.

```python
def process(v, q): ... # no type information
```

By adding type hints, the signature clearly informs readers of the expected types. With modern Python, user-defined and built-in classes can be used to specify types, with additional resources (such as `Any`) found in the `typing` module.

```python
def process(v: int, q: bool) -> list[float]: ...
```

While type annotations require valid syntax and objects, they are irrelevant at run-time and can be wrong: it is possible to have correctly verified types that do not reflect run-time reality.

When used with a type checker like `mypy`, code that violates the specifications of the type annotations will raise an error during static analysis (shown as a comment, below). For example, providing an integer when a bool is required is an error:

```python
x = process(v=5, q=20)
# tp_basic.py: error: Argument "q" to "process"
# has incompatible type "int"; expected "bool"  [arg-type]
```

Similarly, assigning a return value defined as `list[float]` to ``list[int]` value is an error:

```python
y: list[int] = process(v=5, q=False)
# tp_basic.py: error: Incompatible types in assignment
# (expression has type "list[float]", variable has type
# "list[int]")  [assignment]
```

Static analysis can only work with known (statically defined) inputs. In many contexts, it is necessary to perform run-time validation of all inputs. The best-of-both-worlds is possible by re-using type annotations for run-time validation. While there are a few libraries that do this (``typeguard`` and ``beartype`` being fine offerings), StaticFrame offers ``CallGuard``, a run-time validator specialized handling array and DataFrame validations.

A Python decorator is the perfect tool for leveraging annotation for run-time validation.











The NumPy interface module:
https://github.com/numpy/numpy/blob/main/numpy/__init__.pyi

Introduction to typing in StaticFrame:
https://static-frame.readthedocs.io/en/latest/articles/ftyping.html

PEP 646 on Variadic Generics
https://peps.python.org/pep-0646/

Example of a Part Presentation (SciPy 2023)
https://www.youtube.com/watch?v=i4IqWD1zBuo

Example of a Past Presentation (PyCon 2023)
https://www.youtube.com/watch?v=ppPXPVV4rDc



For one example, by defining overloaded function definitions using `Literal` arguments to encode integer `axis` values, axis-specific return types can be specified. For another example, `TypeVarTuple` permits defining contiguous regions of uniform types, permitting expressive specifcation of a wide range of types.



