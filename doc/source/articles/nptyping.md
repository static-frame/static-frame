# Do More with NumPy Array Type Annotations
# Using NumPy Array Type Hints to the Fullest


<!--
Static Type Numpy Arrays
-->

The NumPy array object can take many concrete forms. It might be a one-dimensional array of Booleans, or a three-dimensional array of 8-bit unsigned integers. Simple `isinstance()` checks match all arrays as instances of `np.ndarray`, regardless of shape or `dtype`. Similarly, many type-annotated interfaces only specify `np.ndarray`:


```python {all}
import numpy as np

def process(
    x: np.ndarray,
    y: np.ndarray,
    ) -> np.ndarray: ...
```

Such type annotations are insufficient: most interfaces have strong expectations of the shape or `dtype` of passed arrays. Most code wil fail if a three-dimensional array is passed where a one dimensional array is expected, or an array of dates is passed where an array of floats is expected.

Taking full advantage of the generic `np.ndarray`, such issues can now be found in static analysis with type checkers like `mypy` and `pyright`. For example, the same interfaces as above can be made far more explicit by specifying shape and `dtype` for type parameters.

```python
import numpy as np

def process(
    x: np.ndarray[tuple[int], np.dtype[np.bool_]],
    y: np.ndarray[tuple[int, int, int], np.dtype[np.uint8]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]: ...
```

With such detail, static analyis can find issues before code is even run; further, run-time validators specialized for NumPy, like StaticFrame's `CallGuard`, can re-use the same notations to provide run-time validation of passed values.



## Generic Types in Python

Generic containers such as `list` and `tuple` can be made concrete by specifying, for each interface, the contained types. A function can declare it takes a `list` of `str` with `list[str]`; or a `tuple` of zero or more floats can be specified as `tuple[float, ...]`.


## The Generic `np.ndarray`

The NumPy array is more complex. A concrete `np.ndarray` is an N-dimensional array of a single NumPy value type (or `dytpe`). The `np.ndarray` generic takes two type parameters: the first defines the shape with a `tuple`, the second defines the value types with the generic `np.dtype`.

While the `np.ndarray` generic has taken two type parameters for some time, not until NumPy 2.1 has the definition of those parameters been settled. Full typing of NumPy arrays is now possible.


### The Shape Type Parameter

When creating an array with interfaces like `np.empty` or `np.full`, a shape argument is given as a tuple. The length of the tuple defines its dimensionality; the magnitude of each position defines the size of that dimension. Thus a shape `(10,)` is a one-dimensional array of 10 elements; a shape `(10, 100, 1000)` is a three dimensional array of size 10 by 100 by 1000.

While in the future it might be possible to type-check an `np.ndarray` with specific magnitudes per dimension (using `Literal`), for now only specifying dimensionality is practical. A `tuple[int]` can specify a one-dimensional array; a `tuple[int, int, int]` can specify a three-dimensional array; a `tuple[int, ...]` can denote a truly N-dimensional array.


### The `dtype` Type Parameter

The type of values contained in a NumPy array is defined by the `dtype` object. The `dtype` itself is generic, taking a NumPy "generic" type as a type parameter. The most narrow types specify specifc characteristics and sizes: `np.uint8`, `np.float64`, or `np.bool_`. Beyond these narrow types, NumPy provides more general types, such as `np.integer`, `np.floating`, or `np.number`.


## Making `np.ndarray` Concrete

The following examples illustrate concrete `np.ndarrayt`:

A one-dimensional array of Booleans:

```python
np.ndarray[tuple[int], np.dtype[np.bool_]]
```

A three dimensional array of unisnged 8-bit integers:

```python
np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]
```

A two dimensional array of `np.datetime64` dates:

```python
np.ndarray[tuple[int, int], np.dtype[np.datetime64['D']]]
```

A one dimensional array of any numeric type:

```python
np.ndarray[tuple[int], np.dtype[np.number]]
```



## Static Type Checking with `mypy`



```python
def process1(x: np.ndarray[tuple[int], np.dtype[np.signedinteger]]): ...

a1 = np.empty(100, dtype=np.int16)
process1(a1) # mypy passes

a2 = np.empty(100, dtype=np.uint8)
process1(a2) # mypy fails
# error: Argument 1 to "process1" has incompatible type
# "ndarray[tuple[int], dtype[unsignedinteger[_8Bit]]]";
# expected "ndarray[tuple[int], dtype[signedinteger[Any]]]"

a3 = np.empty((100, 100, 100), dtype=np.int64)
process1(a3) # mypy fails
# error: Argument 1 to "process1" has incompatible type
# "ndarray[tuple[int, int, int], dtype[signedinteger[_64Bit]]]";
# expected "ndarray[tuple[int], dtype[signedinteger[Any]]]"
```



## Runtime Validation of `np.ndarray` Types


```python
import static_frame as sf
@sf.CallGuard.check
def process2(x: np.ndarray[tuple[int], np.dtype[np.signedinteger]]): ...

a2 = np.empty(100, dtype=np.uint8)
process2(a2)
# static_frame.core.type_clinic.ClinicError:
# In args of (x: ndarray[tuple[int], dtype[signedinteger]]) -> Any
# └── In arg x
#     └── ndarray[tuple[int], dtype[signedinteger]]
#         └── dtype[signedinteger]
#             └── Expected signedinteger, provided uint8 invalid
```



```python {1-3|1-12}
import static_frame as sf
@sf.CallGuard.check
def process2(x: np.ndarray[tuple[int], np.dtype[np.signedinteger]]): ...

a3 = np.empty((100, 100, 100), dtype=np.int64)
process2(a3)
# static_frame.core.type_clinic.ClinicError:
# In args of (x: ndarray[tuple[int], dtype[signedinteger]]) -> Any
# └── In arg x
#     └── ndarray[tuple[int], dtype[signedinteger]]
# TODO: update!
```
</Transform>



## Conclusion

Support for `datetime64` units.


Make your NumPy arrays concrete

Type check with `mypy`

Runtime validation with `TypeClinic`
