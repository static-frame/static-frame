Do More with NumPy Array Type Hints: Annotate & Validate Shape & Dtype
================================================================================

.. Improve static analysis and run-time validation with full generic specification


The NumPy array object can take many concrete forms. It might be a one-dimensional (1D) array of Booleans, or a three-dimensional (3D) array of 8-bit unsigned integers. As the built-in function ``isinstance()`` will show, every array is an instance of ``np.ndarray``, regardless of shape or the type of elements stored in the array, i.e., the ``dtype``. Similarly, many type-annotated interfaces still only specify ``np.ndarray``:

.. code-block:: python

    import numpy as np

    def process(
        x: np.ndarray,
        y: np.ndarray,
        ) -> np.ndarray: ...


Such type annotations are insufficient: most interfaces have strong expectations of the shape or ``dtype`` of passed arrays. Most code will fail if a 3D array is passed where a 1D array is expected, or an array of dates is passed where an array of floats is expected.

Taking full advantage of the generic ``np.ndarray``, array shape and ``dtype`` characteristics can now be fully specified:

.. code-block:: python

    def process(
        x: np.ndarray[tuple[int], np.dtype[np.bool_]],
        y: np.ndarray[tuple[int, int, int], np.dtype[np.uint8]],
        ) -> np.ndarray[tuple[int], np.dtype[np.float64]]: ...


With such detail, recent versions of static analysis tools like ``mypy`` and ``pyright`` can find issues before code is even run. Further, run-time validators specialized for NumPy, like `StaticFrame <https://github.com/static-frame/static-frame>`_'s ``sf.CallGuard``, can re-use the same annotations for run-time validation.



Generic Types in Python
---------------------------------

Generic built-in containers such as ``list`` and ``dict`` can be made concrete by specifying, for each interface, the contained types. A function can declare it takes a ``list`` of ``str`` with ``list[str]``; or a ``dict`` of ``str`` to ``bool`` can be specified with ``dict[str, bool]``.


The Generic ``np.ndarray``
--------------------------------

An ``np.ndarray`` is an N-dimensional array of a single element type (or ``dtype``). The ``np.ndarray`` generic takes two type parameters: the first defines the shape with a ``tuple``, the second defines the element type with the generic ``np.dtype``. While ``np.ndarray`` has taken two type parameters for some time, the definition of the first parameter, shape, was not full specified until NumPy 2.1.


The Shape Type Parameter
..................................

When creating an array with interfaces like ``np.empty`` or ``np.full``, a shape argument is given as a tuple. The length of the tuple defines the array's dimensionality; the magnitude of each position defines the size of that dimension. Thus a shape ``(10,)`` is a 1D array of 10 elements; a shape ``(10, 100, 1000)`` is a three dimensional array of size 10 by 100 by 1000.

When using a ``tuple`` to define shape in the ``np.ndarray`` generic, at present only the number of dimensions can generally be used for type checking. Thus, a ``tuple[int]`` can specify a 1D array; a ``tuple[int, int, int]`` can specify a 3D array; a ``tuple[int, ...]``, specifying a tuple of zero or more integers, denotes an N-dimensional array. It might be possible in the future to type-check an ``np.ndarray`` with specific magnitudes per dimension (using ``Literal``), but this is not yet broadly supported.


The ``dtype`` Type Parameter
..........................................

The NumPy ``dtype`` object defines element types and, for some types, other characteristics such as size (for Unicode and string types) or unit (for ``np.datetime64`` types). The ``dtype`` itself is generic, taking a NumPy "generic" type as a type parameter. The most narrow types specify specific element characteristics, for example ``np.uint8``, ``np.float64``, or ``np.bool_``. Beyond these narrow types, NumPy provides more general types, such as ``np.integer``, ``np.inexact``, or ``np.number``.


Making ``np.ndarray`` Concrete
----------------------------------------

The following examples illustrate concrete ``np.ndarray`` definitions:

A 1D array of Booleans:

.. code-block:: python

    np.ndarray[tuple[int], np.dtype[np.bool_]]


A 3D array of unsigned 8-bit integers:

.. code-block:: python

    np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]


A two-dimensional (2D) array of Unicode strings:

.. code-block:: python

    np.ndarray[tuple[int, int], np.dtype[np.str_]]


A 1D array of any numeric type:

.. code-block:: python

    np.ndarray[tuple[int], np.dtype[np.number]]



Static Type Checking with Mypy
----------------------------------------------

Once the generic ``np.ndarray`` is made concrete, ``mypy`` or similar type checkers can, for some code paths, identify values that are incompatible with an interface.

For example, the function below requires a 1D array of signed integers. As shown below, unsigned integers, or dimensionalities other than one, fail ``mypy`` checks.

.. code-block:: python

    def process1(x: np.ndarray[tuple[int], np.dtype[np.signedinteger]]): ...

    a1 = np.empty(100, dtype=np.int16)
    process1(a1) # mypy passes

    a2 = np.empty(100, dtype=np.uint8)
    process1(a2) # mypy fails
    # error: Argument 1 to "process1" has incompatible type
    # "ndarray[tuple[int], dtype[unsignedinteger[_8Bit]]]";
    # expected "ndarray[tuple[int], dtype[signedinteger[Any]]]"  [arg-type]

    a3 = np.empty((100, 100, 100), dtype=np.int64)
    process1(a3) # mypy fails
    # error: Argument 1 to "process1" has incompatible type
    # "ndarray[tuple[int, int, int], dtype[signedinteger[_64Bit]]]";
    # expected "ndarray[tuple[int], dtype[signedinteger[Any]]]"



Runtime Validation with ``sf.CallGuard``
--------------------------------------------------


Not all array operations can statically define the shape or ``dtype`` of a resulting array. For this reason, static analysis will not catch all mismatched interfaces. Better than creating redundant validation code across many functions, type annotations can be re-used for run-time validation with tools specialized for NumPy types.

The `StaticFrame <https://github.com/static-frame/static-frame>`_ ``CallGuard`` interface offers two decorators, ``check`` and ``warn``, which raise exceptions or warnings, respectively, on validation errors. These decorators will validate type-annotations against the characteristics of run-time objects.

For example, by adding ``sf.CallGuard.check`` to the function below, the arrays fail validation with expressive ``CallGuard`` exceptions:

.. code-block:: python

    import static_frame as sf

    @sf.CallGuard.check
    def process2(x: np.ndarray[tuple[int], np.dtype[np.signedinteger]]): ...

    b1 = np.empty(100, dtype=np.uint8)
    process2(b1)
    # static_frame.core.type_clinic.ClinicError:
    # In args of (x: ndarray[tuple[int], dtype[signedinteger]]) -> Any
    # └── In arg x
    #     └── ndarray[tuple[int], dtype[signedinteger]]
    #         └── dtype[signedinteger]
    #             └── Expected signedinteger, provided uint8 invalid

    b2 = np.empty((10, 100), dtype=np.int8)
    process2(b2)
    # static_frame.core.type_clinic.ClinicError:
    # In args of (x: ndarray[tuple[int], dtype[signedinteger]]) -> Any
    # └── In arg x
    #     └── ndarray[tuple[int], dtype[signedinteger]]
    #         └── tuple[int]
    #             └── Expected tuple length of 1, provided tuple length of 2



Conclusion
-----------------

More can be done to improve NumPy typing. For example, the ``np.object_`` type could be made generic such that Python types contained in an object array could be defined. For example, a 1D object array of pairs of integers could be annotated as:


.. code-block:: python

    np.ndarray[tuple[int], np.dtype[np.object_[tuple[int, int]]]]


Further, units of ``np.datetime64`` cannot yet be statically specified. For example, date units could be distinguished from nanosecond units with annotations like ``np.dtype[np.datetime64[Literal['D']]]`` or ``np.dtype[np.datetime64[Literal['ns']]]``.

Even with limitations, fully-specified NumPy type annotations catch errors and improve code quality. As shown, static analysis can identify mismatched shape or ``dtype``, and validation with ``sf.CallGuard`` can provide strong run-time guarantees.


