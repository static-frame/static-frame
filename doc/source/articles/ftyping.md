

# Type-Hinting DataFrames for Static Analysis and Runtime Validation


Since the advent of type hints in Python 3.5, statically typing a DataFrame has generally been limited to specifying just the type:

```python
def process(f: DataFrame) -> Series: ...
```

This is insufficient, as it ignores the types contained within the container. A DataFrame might have string column labels and three columns of integer, string, and floating-point values: these characteristics are necessary to define the type. A function argument with such a type hint would provide developers, static analyzers, and run-time checkers the information to fully understand the expectations of the interface. A full generic specification, now available in StaticFrame 2, defines the type of the index, the type of the column labels, and the types of the columnar values:

```python
def process(f: Frame[
        Any,            # the type of the index labels
        Index[np.str_], # the type of the column labels
        np.int_,        # the type of the first column
        np.str_,        # the type of the second column
        np.float64,     # the type of the third column
        ]) -> None: ...
```

All core StaticFrame containers now support generic specification. While statically checkable, a new decorator, ``@CallGuard.check``, permits run-time validation of these type hints on function interfaces. Using ``Annotated`` generics, the new ``Require`` class defines a family of powerful run-time validators, permitting per column or per row analysis in addition to a host of other validations. Finally, each container exposes a new ``via_type_clinic`` interface to derive type hints at run time or validate a type hint. Together, these tools offer one of the first cohesive and integrated approaches to typing DataFrames.


## The Path to a Generic DataFrame

Python's built-in generic types (e.g.,``tuple`` or ``dict``) require specification of component types. Defining component types for such containers permits more accurate static analysis. While the same is true for DataFrames, there have been few attempts to comprehensively type hint DataFrames.

Pandas, even with the ``pandas-stubs`` package, does not permit specifying the types of a DataFrame's components. The Pandas DataFrame, permitting extensive in-place mutation, may not even be sensible to statically type. Fortunately, immutable DataFrames are available in StaticFrame.

Beyond mutability, Python's tools for defining generics have limited fully type hinting DataFrames. A key aspect of a DataFrame is that it has a variable number of columns of heterogenous types. Typing such a structure became much easier with the new ``TypeVarTuple``, introduced in Python 3.11

A ``TypeVarTuple`` permits creating generics that accept a variable number of types. (See PEP 646 https://peps.python.org/pep-0646). StaticFrame can then define a generic ``Frame`` with a ``TypeVar`` for the index, a ``TypeVar`` for the columns, and a ``TypeVarTuple`` for zero or more columnar types. A generic ``Series`` can be defined as a ``TypeVar`` for the index and a ``TypeVar`` for the values. The StaticFrame ``Index`` and ``IndexHierarchy`` are also generic, the latter taking advantage of ``TypeVarTuple`` to define a component ``Index`` per depth level.

For defining columnar types of a ``Frame``, or the values of a ``Series`` or ``Index``, NumPy generic types are used. For example, ``np.int_`` can be used to type a platform-default integer size, while ``np.int64`` can be used to type an explicit, 64-bit integer.


## Interfaces Defined with Generic DataFrames

Using generic specifications, the same interface from above can be annotated to show a ``Frame`` with three columns being transformed into a dictionary of ``Series``.

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

The type hints in this interface offer so much more information we might intuit what the function does. This function processes a signal table from an Open Source Asset Pricing (OSAP https://www.openassetpricing.com) dataset (Firm Level Characteristics / Full Sets / Predictors / PredictorsIndiv). The table has three columns: security identifier ("permno"), year and month ("yyyymm"), and signal name.

The function ignores the index (typed as ``Any``) and creates groups defined by the first column "permno" ``np.int_`` values. For each group, a ``Series`` is returned of the ``np.float64`` values, the index an ``IndexYearMonth`` created from the ``np.str_`` "yyyymm" column.

Rather than returning a ``dict``, the function above might be revised to return a ``Series`` with a hierarchical index. The ``IndexHierarchy`` generic specifies a component ``Index`` for each depth level; as is clear, the outer depth is an ``Index[np.int_]``, the inner depth an ``IndexYearMonth``.

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

Combined with a better function name (e.g., ``partition_by_permno``), such rich type-hints provide self-documenting interface that makes the functionality explicit.

Even better, these type hints can be used for static analysis with Pyright (now) and MyPy (pending full ``TypeVarTuple`` support). Calling such a function with a ``Frame`` of two columns of ``np.float64``, for example, will fail a static analysis type check or deliver a warning in your editor.


## Runtime Type Validation

Static type checking might not be enough: runtime evaluation provides even stronger constraints, particularly for dynamic values or incompletely (or incorrectly) type-hinted values.

Building on a new run-time type checker, ``TypeClinic``, StaticFrame 2 introduces ``@CallGuard.check``, a decorator for run-time validation of type-hinted interfaces. All StaticFrame and NumPy generics are supported, as well as support for most built-in types, even when deeply nested.

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

Now decorated with ``@CallGuard.check``, if the function above is called with un labelled ``Frame`` of two columns of ``np.float64`` (for example), a ``ClinicError`` exception will be raised, reporting that, where three columns were expected, only two were provided, and where string column labels were expected, integer labels were provided.

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

There are other characteristics that might be validated at run-time. For example, the ``shape`` or ``name`` attributes, or the sequence of labels on the index or columns. The StaticFrame ``Require`` class provides a family of configurable validators. Aligning with a trend seen in numerous packages, these objects are provided as one or more additional arguments to an ``Annotated`` generic.

* ``Require.Name``: Validate the ``name`` attribute of the container.
* ``Require.Len``: Validate the length of the container.
* ``Require.Shape``: Validate the ``shape`` attribute of the container.
* ``Require.LabelsOrder``: Validate the ordering of the labels.
* ``Require.LabelsMatch``: Validate inclusion of labels independent of order.
* ``Require.Apply``: Apply a Boolean-returning function to the container.


Extending the example of processing an OSAP signal table, we might validate our expectation of column labels. The ``Require.LabelsOrder`` validator can define a sequence of labels, optionally using ``...`` for contiguous regions of zero or more unspecified labels. To specify that the first two columns of the table are labelled "permno" and "yyyymm", while the third label is variable (depending on the specific OSAP file in this data set), ``Require.LabelsOrder`` can be used as such:


```python
from typing import Any, Annotated
from static_frame import Frame, Series, Index, IndexYearMonth, IndexHierarchy, CallGuard, Require

@CallGuard.check
def process(f: Frame[
        Any,
        Annotated[
                Index[np.str_],
                Require.LabelsOrder('permo', 'yyyymm', ...),
                ],
        np.int_,
        np.str_,
        np.float64,
        ]) -> Series[
                IndexHierarchy[Index[np.int_], IndexYearMonth],
                np.float64,
                ]: ...

```

If the interface expects a small collection of OSAP signal tables, we can validate the third column with the ``Require.LabelsMatch`` validator. This validator can specify required labels, sets of labels (from which at least one must match), and regular expression patterns. If we are only using the files Mom12m.csv, Mom6m.csv, and LRreversal.csv, we can validate the expected names of the third column:


```python
from typing import Any, Annotated
from static_frame import Frame, Series, Index, IndexYearMonth, IndexHierarchy, CallGuard, Require

@CallGuard.check
def process(f: Frame[
        Any,
        Annotated[
                Index[np.str_],
                Require.LabelsOrder('permo', 'yyyymm', ...),
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

To perform data validation, both ``Require.LabelsOrder`` and ``Require.LabelsMatch`` can be further extended by associating functions with labels. If the validator is associated with the column labels, a ``Series`` of column values will be provided to the function; if the validator is associated with index labels, a ``Series`` of row values will be provided to the function. Similar to usage of ``Annotated``, the label is replaced with a list (which itself cannot be a label), where the first item is the label specifier and remain arguments are fucntions.

To extend the example above, we might validate all "permo" values are great than zero and that all signal values ("Mom12m", "Mom6m", "LRreversal") are greater than -1. As shown, any label specifier, even ``...`` can be associated with functions.


```python
from typing import Any, Annotated
from static_frame import Frame, Series, Index, IndexYearMonth, IndexHierarchy, CallGuard, Require

@CallGuard.check
def process(f: Frame[
        Any,
        Annotated[
                Index[np.str_],
                Require.LabelsOrder(
                        ['permo', lambda s: (s > 0).all()],
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


## The Expressive Powers of ``TypeVarTuple``



## Typing Utilities

Comprehensively typing a ``Frame`` can be time consuming. StaticFrame includes a number of generic aliases to define just the outermost container types. For example, ``TFrameAny`` can be used for any ``Frame``, and ``TSeries`` for any ``Series``.



## Alternative Approaches

The Pandera library permits specifying columnar schema that can be used as a type-hint stand-in for the Pandas DataFrame type.


## Conclusion

Given the extensive use of DataFrames in the Python ecosystem, as well as the growing interest in static typing, better typing for DataFrames is overdue. With modern Python typing tools and a DataFrame built on an immutable data model, StaticFrame 2 meets this need, providing tooling for engineers prioritizing quality, correctness, and verifiability.



