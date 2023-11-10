

# Type-Hinting DataFrames for Static Analysis and Runtime Validation


Since the advent of type hints in Python 3.5, statically typing a DataFrame has generally been limited to specifying just the type.

```python
def process(f: DataFrame) -> Series: ...
```

This is inadequate, as it ignores the types contained within the container. A DataFrame might have string column labels and three columns of integer, string, and floating-point values: these characteristics define the type. A function argument with such type hints would provide developers, static analyzers, and runtime checkers the information to fully understand the expectations of the interface. StaticFrame 2 now permits this:

```python
from typing import Any
from static_frame import Frame, Index, TSeriesAny

def process(f: Frame[   # the type of the container
        Any,            # the type of the index labels
        Index[np.str_], # the type of the column labels
        np.int_,        # the type of the first column
        np.str_,        # the type of the second column
        np.float64,     # the type of the third column
        ]) -> TSeriesAny: ...
```

All core StaticFrame containers now support generic specification. While statically checkable, a new decorator, ``@CallGuard.check``, permits runtime validation of these type hints on function interfaces. Using ``Annotated`` generics, the new ``Require`` class defines a family of powerful runtime validators, permitting per-column or per-row analysis as well as many other validations. Finally, each container exposes a new ``via_type_clinic`` interface to derive and validate type hints. Together, these tools offer a cohesive approach to type-hinting and validating DataFrames.


## The Path to a Generic DataFrame

Python's built-in generic types (e.g.,``tuple`` or ``dict``) require specification of component types (e.g., ``tuple[int, str, bool]`` or ``dict[str, int]``). Defining component types permits more accurate static analysis. While the same is true for DataFrames, there have been few attempts to comprehensively define type hints for DataFrames.

Pandas, even with the ``pandas-stubs`` package, does not permit specifying the types of a DataFrame's components. The Pandas DataFrame, permitting extensive in-place mutation, may not even be sensible to statically type. Fortunately, immutable DataFrames are available in StaticFrame.

Further, Python's tools for defining generics, until recently, have not been well-suited for DataFrames. That a DataFrame has a variable number heterogenous columnar types poses a challenge for generic specification. Typing such a structure became easier with the new ``TypeVarTuple``, introduced in Python 3.11 (and back-ported in the ``typing_extensions`` package).

A ``TypeVarTuple`` permits defining generics that accept a variable number of types. (See PEP 646 https://peps.python.org/pep-0646 for details.) With this new type variable, StaticFrame can defines a generic ``Frame`` with a ``TypeVar`` for the index, a ``TypeVar`` for the columns, and a ``TypeVarTuple`` for zero or more columnar types. A generic ``Series`` is defined as a ``TypeVar`` for the index and a ``TypeVar`` for the values. The StaticFrame ``Index`` and ``IndexHierarchy`` are also generic, the latter again taking advantage of ``TypeVarTuple`` to define a variable number of component ``Index`` for each depth level.

For defining the columnar types of a ``Frame``, or the values of a ``Series`` or ``Index``, NumPy generic types are used. This permits narrowly specifying sized numerical types, such as ``np.uint8`` or ``np.complex128``; or broadly specifying categories of types, such as ``np.integer`` or ``np.inexact``. As StaticFrame supports all NumPy types, the correspondence is direct.


## Interfaces Defined with Generic DataFrames

Extending the example above, the function interface below shows a ``Frame`` with three columns being transformed into a dictionary of ``Series``. With so much more information provided by component type hints, the function's purpose is almost obvious.

```python
from typing import Any
from static_frame import Frame, Series, Index, IndexYearMonth

def process(f: Frame[
        Any,
        Index[np.str_],
        np.int_,
        np.str_,
        np.float64,
        ]) -> dict[
                int,
                Series[IndexYearMonth, np.float64],
                ]: ...
```

This function processes a signal table from an Open Source Asset Pricing (OSAP https://www.openassetpricing.com) dataset (Firm Level Characteristics / Individual / Predictors). Each table has three columns: security identifier (labelled "permno"), year and month (labelled "yyyymm"), and the signal (with a name specific to the signal).

The function ignores the index (typed as ``Any``) and creates groups defined by the first column "permno" ``np.int_`` values. A dictionary keyed by "permno" is returned, where each value is a ``Series`` of ``np.float64`` values for that "permno"; the index is an ``IndexYearMonth`` created from the ``np.str_`` "yyyymm" column. (StaticFrame uses NumPy ``datetime64`` values to define unit-typed indices: ``IndexYearMonth`` stores ``datetime64[M]`` labels.)

Rather than returning a ``dict``, the function below returns a ``Series`` with a hierarchical index. The ``IndexHierarchy`` generic specifies a component ``Index`` for each depth level; here, the outer depth is an ``Index[np.int_]`` (derived from the "permno" column), the inner depth an ``IndexYearMonth`` (derived from the "yyyymm" column).

```python
from typing import Any
from static_frame import Frame, Series, Index, IndexYearMonth, IndexHierarchy

def process(f: Frame[
        Any,
        Index[np.str_],
        np.int_,
        np.str_,
        np.float64,
        ]) -> Series[
                IndexHierarchy[Index[np.int_], IndexYearMonth],
                np.float64,
                ]: ...
```

Combined with a better function name (e.g., ``partition_by_permno``), such rich type-hints provide a self-documenting interface that makes functionality explicit.

Even better, these type hints can be used for static analysis with Pyright (now) and MyPy (pending full ``TypeVarTuple`` support). Calling this function with a ``Frame`` of two columns of ``np.float64``, for example, will fail a static analysis type check or deliver a warning in an editor.


## Runtime Type Validation

Static type checking may not be enough: runtime evaluation provides even stronger constraints, particularly for dynamic or incompletely (or incorrectly) type-hinted values.

Building on a new runtime type checker named ``TypeClinic``, StaticFrame 2 introduces ``@CallGuard.check``, a decorator for runtime validation of type-hinted interfaces. All StaticFrame and NumPy generics are supported, as well as support for most built-in Python types, even when deeply nested. The function below adds the ``@CallGuard.check`` decorator.

```python
from typing import Any
from static_frame import Frame, Series, Index, IndexYearMonth, IndexHierarchy, CallGuard

@CallGuard.check
def process(f: Frame[
        Any,
        Index[np.str_],
        np.int_,
        np.str_,
        np.float64,
        ]) -> Series[
                IndexHierarchy[Index[np.int_], IndexYearMonth],
                np.float64,
                ]: ...

```

Now decorated with ``@CallGuard.check``, if the function above is called with an unlabelled ``Frame`` of two columns of ``np.float64``, a ``ClinicError`` exception will be raised, illustrating that, where three columns were expected, two were provided, and where string column labels were expected, integer labels were provided.

<!-- f = Frame(np.random.rand(20).reshape(10,2)) -->

```
ClinicError:
In args of (f: Frame[Any, Index[str_], int64, str_, float64]) -> Series[IndexHierarchy[Index[int64], IndexYearMonth], float64]
└── Frame[Any, Index[str_], int64, str_, float64]
    └── Expected Frame has 3 dtype, provided Frame has 2 dtype
In args of (f: Frame[Any, Index[str_], int64, str_, float64]) -> Series[IndexHierarchy[Index[int64], IndexYearMonth], float64]
└── Frame[Any, Index[str_], int64, str_, float64]
    └── Index[str_]
        └── Expected str_, provided int64 invalid
```

To issue warnings instead of raising exceptions, the ``@CallGuard.warn`` decorator may be used.


## Runtime Data Validation

There are other characteristics that might be validated at runtime. For example, the ``shape`` or ``name`` attributes, or the sequence of labels on the index or columns. The StaticFrame ``Require`` class provides a family of configurable validators.

* ``Require.Name``: Validate the ``name`` attribute of the container.
* ``Require.Len``: Validate the length of the container.
* ``Require.Shape``: Validate the ``shape`` attribute of the container.
* ``Require.LabelsOrder``: Validate the ordering of the labels.
* ``Require.LabelsMatch``: Validate inclusion of labels independent of order.
* ``Require.Apply``: Apply a Boolean-returning function to the container.

Aligning with a growing trend, these objects are provided in type hints as one or more additional arguments to an ``Annotated`` generic. (See PEP 593 https://peps.python.org/pep-0593 for details).

Extending the example of processing an OSAP signal table, we might validate our expectation of column labels. The ``Require.LabelsOrder`` validator can define a sequence of labels, optionally using ``...`` for contiguous regions of zero or more unspecified labels. To specify that the first two columns of the table are labelled "permno" and "yyyymm", while the third label is variable (depending on the signal), the following ``Require.LabelsOrder`` can be defined with an ``Annotated`` generic:


```python
from typing import Any, Annotated
from static_frame import Frame, Series, Index, IndexYearMonth, IndexHierarchy, CallGuard, Require

@CallGuard.check
def process(f: Frame[
        Any,
        Annotated[
                Index[np.str_],
                Require.LabelsOrder('permno', 'yyyymm', ...),
                ],
        np.int_,
        np.str_,
        np.float64,
        ]) -> Series[
                IndexHierarchy[Index[np.int_], IndexYearMonth],
                np.float64,
                ]: ...

```

If the interface expects a small collection of OSAP signal tables, we can validate the third column with the ``Require.LabelsMatch`` validator. This validator can specify required labels, sets of labels (from which at least one must match), and regular expression patterns. If tables from only three files are expected (i.e. Mom12m.csv, Mom6m.csv, and LRreversal.csv), we can validate the names of the third column by defining a set:

```python
@CallGuard.check
def process(f: Frame[
        Any,
        Annotated[
                Index[np.str_],
                Require.LabelsOrder('permno', 'yyyymm', ...),
                Require.LabelsMatch({'Mom12m', 'Mom6m', 'LRreversal'}),
                ],
        np.int_,
        np.str_,
        np.float64,
        ]) -> Series[
                IndexHierarchy[Index[np.int_], IndexYearMonth],
                np.float64,
                ]: ...

```

To perform data value validation, both ``Require.LabelsOrder`` and ``Require.LabelsMatch`` can associate functions with label specifiers. If the validator is applied to column labels, a ``Series`` of column values will be provided to the function; if the validator is applied to index labels, a ``Series`` of row values will be provided to the function.

Similar to usage of ``Annotated``, the label is replaced with a list, where the first item is the label specifier and the remaining items are row- or column-processing functions that return a Boolean.

To extend the example above, we might validate all "permno" values are greater than zero and that all signal values ("Mom12m", "Mom6m", "LRreversal") are greater than -1.


```python
from typing import Any, Annotated
from static_frame import Frame, Series, Index, IndexYearMonth, IndexHierarchy, CallGuard, Require

@CallGuard.check
def process(f: Frame[
        Any,
        Annotated[
                Index[np.str_],
                Require.LabelsOrder(
                        ['permno', lambda s: (s > 0).all()],
                        'yyyymm',
                        ...,
                        ),
                Require.LabelsMatch(
                        [{'Mom12m', 'Mom6m', 'LRreversal'}, lambda s: (s > -1).all()],
                        ),
                ],
        np.int_,
        np.str_,
        np.float64,
        ]) -> Series[
                IndexHierarchy[Index[np.int_], IndexYearMonth],
                np.float64,
                ]: ...

```

If a type or data validation fails, ``@CallGuard.check`` will raise an exception. For example, if the above function is called with a ``Frame`` that has an unexpected third-column label, the following exception will be raised:

<!-- >>> f = sf.Frame.from_records(([3, '192004', 1.0], [3, '192005', -2.0]), columns=('permno', 'yyyymm', 'Mom3m')) -->

```
ClinicError:
In args of (f: Frame[Any, Annotated[Index[str_], LabelsOrder(['permno', <lambda>], 'yyyymm', ...), LabelsMatch([{'Mom12m', 'LRreversal', 'Mom6m'}, <lambda>])], int64, str_, float64]) -> Series[IndexHierarchy[Index[int64], IndexYearMonth], float64]
└── Frame[Any, Annotated[Index[str_], LabelsOrder(['permno', <lambda>], 'yyyymm', ...), LabelsMatch([{'Mom12m', 'LRreversal', 'Mom6m'}, <lambda>])], int64, str_, float64]
    └── Annotated[Index[str_], LabelsOrder(['permno', <lambda>], 'yyyymm', ...), LabelsMatch([{'Mom12m', 'LRreversal', 'Mom6m'}, <lambda>])]
        └── LabelsMatch([{'Mom12m', 'LRreversal', 'Mom6m'}, <lambda>])
            └── Expected label to match frozenset({'Mom12m', 'LRreversal', 'Mom6m'}), no provided match
```


## The Expressive Power of ``TypeVarTuple``

As shown above, ``TypeVarTuple`` permits specifying ``Frame`` with zero or more heterogenous columnar types. For example, we can provide type hints for a ``Frame`` of two float or six mixed columnar types:

```python
from typing import Any
from static_frame import Frame, Index

>>> f1: sf.Frame[Any, Any, np.float64, np.float64]
>>> f2: sf.Frame[Any, Any, np.bool_, np.float64, np.int8, np.int8, np.str_, np.datetime64]
```

While this accommodates diverse DataFrames, type-hinting wide DataFrames, such as those with hundreds of columns, would be unwieldy. Python 3.11 introduces a new syntax for providing variable ranges of types in ``TypeVarTuple`` generics: star expressions of ``tuple`` generic aliases. For example, to type-hint a ``Frame`` with a date index, string column labels, and any configuration of columnar types, we can star-unpack a ``tuple`` of zero or more ``All``:

```python
from typing import Any
from static_frame import Frame, Index

>>> f: sf.Frame[Index[np.datetime64], Index[np.str_], *tuple[All, ...]]
```

The ``tuple`` star expression can go anywhere in a list of types, but there can be only one. For example, the type hint below defines a ``Frame`` that must start with Boolean and string columns, but has a flexible requirement for any number of subsequent ``np.float64`` columns.

```python
from typing import Any
from static_frame import Frame

>>> f: sf.Frame[Any, Any, np.bool_, np.str_, *tuple[np.float64, ...]]
```

## Utilities for Type Hinting

Working with such detailed type hints can be challenging. StaticFrame provides convenient utilities for runtime type hinting and checking. All StaticFrame 2 containers feature a ``via_type_clinic`` interface, permitting access to ``TypeClinic`` functionality.

First, utilities are provided to translate a container, such as a complete ``Frame``, into a type hint. The default string representation of ``via_type_clinic`` provides a string representation of the type hint. The ``to_hint()`` method returns a complete generic alias object.

```python
>>> import static_frame as sf
>>> f = sf.Frame.from_records(([3, '192004', 0.3], [3, '192005', -0.4]), columns=('permno', 'yyyymm', 'Mom3m'))

>>> f.via_type_clinic
Frame[Index[int64], Index[str_], int64, str_, float64]

>>> f.via_type_clinic.to_hint()
static_frame.core.frame.Frame[static_frame.core.index.Index[numpy.int64], static_frame.core.index.Index[numpy.str_], numpy.int64, numpy.str_, numpy.float64]
```

Second, utilities are provided for runtime type hint testing. The ``via_type_clinic.check()`` function permits validating the container against a provided type-hint.

```python
>>> f.via_type_clinic.check(sf.Frame[sf.Index[np.str_], sf.TIndexAny, *tuple[tp.Any, ...]])
ClinicError:
In Frame[Index[str_], Index[Any], Unpack[Tuple[Any, ...]]]
└── Index[str_]
    └── Expected str_, provided int64 invalid
```

To support gradual typing, StaticFrame defines several generic aliases configured with ``Any`` for every component type. For example, ``TFrameAny`` can be used for any ``Frame``, and ``TSeriesAny`` for any ``Series``. As expected, ``TFrameAny`` will validate the ``Frame`` created above without error.

```python
>>> f.via_type_clinic.check(sf.TFrameAny)
```


## Conclusion

Better type hinting for DataFrames is overdue. With modern Python typing tools and a DataFrame built on an immutable data model, StaticFrame 2 meets this need, providing powerful resources for engineers prioritizing maintainability and verifiability.


