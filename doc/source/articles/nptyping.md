# Do More with NumPy Array Type Annotations

<!--
Static Type Numpy Arrays
-->

The NumPy array, as single type of Python Object, can take many concrete forms. It might be a one-dimensionaly array of Booleans, or a three-dimensional array of 8-bit unsigned integers. And yet, up until recently, a type-annotated function might only specify that the arguments are NumPy arrays:


```python {all}
import numpy as np

def process(
    x: np.ndarray,
    y: np.ndarray,
    ) -> np.ndarray: ...
```

These type annotations lack sufficient detail. Much better would be to narrowly define the shape and type expectations. For example, specifying argument shape and type expectations permits static analyis or runtime-validation to catch errors, such as passing a two-dimensional array when a one-dimensional array is expected. Deep type annotations make arguemnts expectations unambiguous:

```python
import numpy as np

def process(
    x: np.ndarray[tuple[int], np.dtype[np.bool_]],
    y: np.ndarray[tuple[int, int, int], np.dtype[np.uint8]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]: ...
```



## Generic Type in Python


With type annotations in Python, interfaces can precisely specify expected arguments types. Using static analysis tools like `mypy` and `pyright`, code that calls these interfaces can be type checked, alerting developers to errors before the code is ever run. Further, many tool exist to re-use those type annotations for run-time validation, providing another layer of checks beyond what is possible in static analysis.

Generic containers such as `list` and `tuple` can be made concrete by specifying, for each interface, the expected types contained in the container. A function can declare it takes of `list` of `str` with `list[str]`; or a `tuple` of floats can be specified as `tuple[float, ...]`.

The NumPy array, however, is more complex than these containers. The `np.ndarray` object represents an N-dimensional array of a single NumPy `dytpe`. While the `np.ndarray` is a generic type that takes two parameters, up until NumPy 2.1 the definition of those parameters has been incomplete. Finally, NumPy has settled on a definition, and full typing of NumPy arrays is now possible.

## The Generic `np.ndarray`


The `np.ndarray` generic is made conrete with two type parameters. The first defines the shape with a `tuple`. The second defines the value types with the generic `np.dtype`.


### The Shape Type Parameter

When creating an array with interfaces like `np.empty` or `np.full`, the length of the tuple defines the arrays dimensionality, and the magnitude of each value defines the size of that dimension. Thus a shape `(10,)` is one-dimensional array of 10 elements; a shape `(10, 100, 1000)` is a three dimensional array of size 10 by 100 by 1000.

While in the future it might be practical to concretize an `np.ndarray` with specific maganitudes per dimension (using `Literal[]`), for now, we can at least specify dimensionality. A `tuple[int]` can specify any one-dimensional array; a `tuple[int, int, int]` can specify a three-dimensional array; and, if needed, a `tuple[int, ...]` can denote a truly N-dimensional array.


```python
tuple[int, ...]  # ND
tuple[int]  # 1D
tuple[int, int] # 2D
tuple[Literal[20], int] # 2D, 20 rows
```

### The `dtype` Type Parameter

The type of values in a NumPy array are defined by the `dtype` object. The `dtype` object itself is generica, and takes a NumPy "generic" type as type parameter. The most narrow types are likely well known to NumPy users: `np.uint8`, `np.float64`, or `np.bool_`. Beyond these narrow types, NumPy provides more general types, such as `np.integer`, `np.floating`, or `np.number`.


```python
import numpy as np

np.dtype[np.int8]
np.dtype[np.uint16]
np.dtype[np.integer]
np.dtype[np.floating]
np.dtype[np.number]

```


## Making `np.ndarray` Concrete

After explaining the type variables of `np.ndarray`, some examples combining the two are useful.

A one-dimensional array of 8 bit integers:

```python
np.ndarray[tuple[int], np.dtype[np.int8]]
```


```python
np.ndarray[tuple[int, int], np.dtype[np.integer]
```


```python
np.ndarray[tuple[int, int], np.dtype[np.datetime64['D']]]
```

While like not common, a truly N-dimensional array of floats can be specified as follows


```python
np.ndarray[tuple[int, ...], np.dtype[np.float64]]
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


Make your NumPy arrays concrete

Type check with `mypy`

Runtime validation with `TypeClinic`
