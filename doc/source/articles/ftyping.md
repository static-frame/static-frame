

# Type-Hinting DataFrames for Static Analysis and Runtime Validation


Since the advent of type hints in Python 3.5, statically typing a DataFrame has generally been limited to specifying just the type:

```python
def process(f: DataFrame) -> Series: ...
```

This is insufficient, as it ignores the types contained within the container. A DataFrame might have string column labels and three columns of integer, string, and floating-point values: these characteristics define the type. A function argument with such a type hint would provide developers, static analyzers, and run-time checkers the information to fully understand the expectations of the interface. StaticFrame 2 now permits this:

```python
from typing import Any
from static_frame import Frame, Index

def process(f: Frame[
        Any,            # the type of the index labels
        Index[np.str_], # the type of the column labels
        np.int_,        # the type of the first column
        np.str_,        # the type of the second column
        np.float64,     # the type of the third column
        ]) -> None: ...
```

All core StaticFrame containers now support generic specification. While statically checkable, a new decorator, ``@CallGuard.check``, permits run-time validation of these type hints on function interfaces. Using ``Annotated`` generics, the new ``Require`` class defines a family of powerful run-time validators, permitting per-column or per-row analysis in addition many other validations. Finally, each container exposes a new ``via_type_clinic`` interface to derive or validate type hints at run time. Together, these tools offer a cohesive and integrated approach to type-hinting and validating DataFrames.


## The Path to a Generic DataFrame

Python's built-in generic types (e.g.,``tuple`` or ``dict``) require specification of component types (e.g., ``tuple[int, str, bool]`` or ``dict[str, int]``). Defining component types permits more accurate static analysis. While the same is true for DataFrames, there have been few attempts to comprehensively type hint DataFrames.

Pandas, even with the ``pandas-stubs`` package, does not permit specifying the types of a DataFrame's components. The Pandas DataFrame, permitting extensive in-place mutation, may not even be sensible to statically type. Fortunately, immutable DataFrames are available in StaticFrame.

Until recently, Python's tools for defining generics have also limited type-hinting DataFrames. A key aspect of a DataFrame is that it has a variable number of columns of heterogenous types. Typing such a structure became much easier with the new ``TypeVarTuple``, introduced in Python 3.11 (and available in the ``typing_extensions`` package).

A ``TypeVarTuple`` permits creating generics that accept a variable number of types. (See PEP 646 https://peps.python.org/pep-0646). With this new type of type variable, StaticFrame can defines a generic ``Frame`` with a ``TypeVar`` for the index, a ``TypeVar`` for the columns, and a ``TypeVarTuple`` for zero or more columnar types. A generic ``Series`` is defined as a ``TypeVar`` for the index and a ``TypeVar`` for the values. The StaticFrame ``Index`` and ``IndexHierarchy`` are also generic, the latter again taking advantage of ``TypeVarTuple`` to define a variable number of component ``Index`` for each depth level.

For defining the columnar types of a ``Frame``, or the values of a ``Series`` or ``Index``, NumPy generic types are used. For example, ``np.int_`` is used to type a platform-default integer size, while ``np.int64`` is used to type an explicit, 64-bit integer. As StaticFrame supports all NumPy types, the correspondence is direct.


## Interfaces Defined with Generic DataFrames

Extending the example above, this interface shows a ``Frame`` with three columns being transformed into a dictionary of ``Series``. With so much more information provided by component type hints, the function's purpose is almost obvious.

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

This function processes a signal table from an Open Source Asset Pricing (OSAP https://www.openassetpricing.com) dataset (Firm Level Characteristics / Full Sets / Predictors / PredictorsIndiv). The table has three columns: security identifier ("permno"), year and month ("yyyymm"), and a signal name.

The function ignores the index (typed as ``Any``) and creates groups defined by the first column "permno" ``np.int_`` values. A dictionary keyed by "permno" is returned, where each value is a ``Series`` of the ``np.float64`` values for that "permno", the index an ``IndexYearMonth`` created from the ``np.str_`` "yyyymm" column. (StaticFrame uses NumPy ``datetime64`` values to define unit-typed indices.)

Rather than returning a ``dict``, the function below returns a ``Series`` with a hierarchical index. The ``IndexHierarchy`` generic specifies a component ``Index`` for each depth level; as shown, the outer depth is an ``Index[np.int_]`` (for the "permno" column), the inner depth an ``IndexYearMonth`` (from the "yyyymm" column).

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

Combined with a better function name (e.g., ``partition_by_permno``), such rich type-hints provide a self-documenting interface that makes the functionality explicit.

Even better, these type hints can be used for static analysis with Pyright (now) and MyPy (pending full ``TypeVarTuple`` support). Calling such a function with a ``Frame`` of two columns of ``np.float64``, for example, will fail a static analysis type check or deliver a warning in your editor.


## Runtime Type Validation

Static type checking might not be enough: runtime evaluation provides even stronger constraints, particularly for dynamic or incompletely (or incorrectly) type-hinted values.

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

If the interface expects a small collection of OSAP signal tables, we can validate the third column with the ``Require.LabelsMatch`` validator. This validator can specify required labels, sets of labels (from which at least one must match), and regular expression patterns. If we are only processing tables from the files Mom12m.csv, Mom6m.csv, and LRreversal.csv, we can validate the expected names of the third column:

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

To perform data validation, both ``Require.LabelsOrder`` and ``Require.LabelsMatch`` can be extended by associating functions with label specifiers. If the validator is associated with the column labels, a ``Series`` of column values will be provided to the function; if the validator is associated with index labels, a ``Series`` of row values will be provided to the function. Similar to usage of ``Annotated``, the label is replaced with a list, where the first item is the label specifier and the remain items are row- or column-processing functions that return a Boolean.

To extend the example above, we might validate all "permno" values are great than zero and that all signal values ("Mom12m", "Mom6m", "LRreversal") are greater than -1. As shown, any label specifier, even ``...`` can be associated with functions.


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

As shown above, ``TypeVarTuple`` permits specifying ``Frame`` of zero or more types. For example, we can provide type hints for a ``Frame`` of two float or six mixed column values, all while accepting any index or columns type:

```python
from typing import Any
from static_frame import Frame, Index

>>> f1: sf.Frame[Any, Any, np.float64, np.float64]

>>> f2: sf.Frame[Any, Any, np.bool_, np.float64, np.int8, np.int8, np.str_, np.datetime64]
```

Python 3.11 introduces a new syntax for ``TypeVarTuple`` values: star-expandable ``tuple`` generic aliases. For example, to type-hint a ``Frame`` with a date index, string column labels, and any configuration of columns, we can start-expand a ``tuple`` of zero or more ``All``:

```python
from typing import Any
from static_frame import Frame, Index

>>> f: sf.Frame[Index[np.datetime64], Index[np.str_], *tuple[All, ...]]
```

The star-expandable ``tuple`` can go anywhere in a list of types. For example, the type hint below defines a ``Frame`` that starts with Boolean and floating-point columns, but has no requirement for any number of following columns.

```python
from typing import Any
from static_frame import Frame

>>> f: sf.Frame[Any, Any, np.bool_, np.float64, *tuple[All, ...]]
```

While monly one ``TypeVarTuple`` can be used in sequence of columnar types, this tool permits a range of options, letting the definition make requirements about leading or trailing columns as needed.


## Typing Utilities

All StaticFrame containers feature a ``via_type_clinic`` interface to permit access to a ``TypeClinic`` instantiated with the container. A utility provided by this interface is, given a container, the complete tyep hint for that container. The default string representation of ``via_type_clinic`` provides a string representation of the hint; the ``to_hint`` method returns the a complete generic alias object:

```python
>>> import static_frame as sf
>>> f = sf.Frame.from_records(([3, '192004', 0.3], [3, '192005', -0.4]), columns=('permno', 'yyyymm', 'Mom3m'))
>>> f.via_type_clinic
Frame[Index[int64], Index[str_], int64, str_, float64]

>>> f.via_type_clinic.to_hint()
static_frame.core.frame.Frame[static_frame.core.index.Index[numpy.int64], static_frame.core.index.Index[numpy.str_], numpy.int64, numpy.str_, numpy.float64]
```

The ``via_type_clinic.check()`` fun permits validating the container against type-hint at run-time.

```python
>>> f.via_type_clinic.check(sf.Frame[sf.Index[np.str_], sf.TIndexAny, *tuple[tp.Any, ...]])
ClinicError:
In Frame[Index[str_], Index[Any], Unpack[Tuple[Any, ...]]]
└── Index[str_]
    └── Expected str_, provided int64 invalid
```

Comprehensively typing a ``Frame`` can be time consuming. StaticFrame includes a number of generic aliases to define just the outermost container types. For example, ``TFrameAny`` can be used for any ``Frame``, and ``TSeries`` for any ``Series``.

```python
>>> f.via_type_clinic.check(sf.TFrameAny)
```


## Conclusion

Given the extensive use of DataFrames in the Python ecosystem, as well as the growing interest in static typing, better typing for DataFrames is overdue. With modern Python typing tools and a DataFrame built on an immutable data model, StaticFrame 2 meets this need, providing resources for engineers prioritizing quality, maintainability, and verifiability.



