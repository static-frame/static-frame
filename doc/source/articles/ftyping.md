

# Type-Hinting DataFrames for Static Analysis and Runtime Validation


Since the advent of type hints in Python 3.5, statically typing a DataFrame has generally been limited to specifying just the type:

```python
def process(f: DataFrame) -> Series: ...
```

This is insufficient, as it ignores the types contained within the container. A DataFrame might have an integer index, string column labels, and two floating-point columns: these characteristics are necessary to define the type.

StaticFrame 2 offers complete generic specification of a DataFrame, including the index, columns, and a variable number of columnar types. All core StaticFrame containers now support generic specification. While statically checkable, a new decorator, ``@CallGuard.check``, permits run-time validation of these type hints on function interfaces. Using ``Annotated`` generics, the new ``Require`` class defines a family of powerful run-time validators, permitting per column or per row analysis in addition to a host of other validations. Finally, each container exposes a new ``via_type_clinic`` interface to derive type hints at run time or validate a type hint. Together, these tools offer one of the first cohesive and integrated approaches to typing DataFrames.


## The Path to a Generic DataFrame

Python's built-in generic types (e.g.,``tuple`` or ``dict``) require specification of component types. Defining component types for such containers permits more accurate static analysis. While the same is true for DataFrames, there have been few attempts to comprehensively type hint DataFrames.

Pandas, even with the ``pandas-stubs`` package, does not permit specifying the types of a DataFrame's components. The Pandas DataFrame, permitting extensive in-place mutation, may not even be sensible to statically type. Fortunately, immutable DataFrames are available in StaticFrame.

Beyond mutability, Python's tools for defining generics have limited fully type hinting DataFrames. A key aspect of a DataFrame is that it has a variable number of columns of heterogenous types. Typing such a structure became much easier with the new ``TypeVarTuple``, introduced in Python 3.11

A ``TypeVarTuple`` permits creating generics that accept a variable number of types. (See PEP 646 https://peps.python.org/pep-0646). StaticFrame can then define a generic ``Frame`` with a ``TypeVar`` for the index, a ``TypeVar`` for the columns, and a ``TypeVarTuple`` for zero or more columnar types. A generic ``Series`` can be defined as a ``TypeVar`` for the index and a ``TypeVar`` for the values. The StaticFrame ``Index`` and ``IndexHierarchy`` are also generic, the latter taking advantage of ``TypeVarTuple`` to define a component ``Index`` per depth level.

For defining columnar types of a ``Frame``, or the values of a ``Series`` or ``Index``, NumPy generic types are used. For example, ``np.int_`` can be used to type a platform-default integer size, while ``np.int64`` can be used to type an explicit, 64-bit integer.


## Interfaces Defined with DataFrame Generics

Using generic specifications, the same interface from above can be annotated to show a ``Frame`` with three columns being transformed into a dictionary of ``Series``.

```python
def process(f: Frame[
        Any,
        Index[np.str_],
        np.int_,
        np.str_,
        np.float64,
        ]) -> dict[int, Series[IndexYearMonth, np.float64]]: ...
```

The type hints in this interface offer so much more information we might intuit what the function does. This function processes a signal table from an Open Source Asset Pricing (https://www.openassetpricing.com) dataset (Firm Level Characteristics / Full Sets / Predictors / PredictorsIndiv). The table has three columns: security identifier ("permno"), year and month ("yyyymm"), and signal name.

The function ignores the index (typed as ``Any``) and creates groups defined by the first column "permno" ``np.int_`` values. For each group, a ``Series`` is returned of the ``np.float64`` values, the index an ``IndexYearMonth`` created from the ``np.str_`` "yyyymm" column.

Rather than returning a ``dict``, the function above might be revised to return a ``Series`` with a hierarchical index. The ``IndexHierarchy`` generic specifies a component ``Index`` for each depth level; as is clear, the outer depth is an ``Index[np.int_]``, the inner depth an ``IndexYearMonth``.

```python
def process(f: Frame[
        Any,
        Index[np.str_],
        np.int_,
        np.str_,
        np.float64,
        ]) -> Series[IndexHierarchy[Index[np.int_], IndexYearMonth], np.float64]: ...
```

Combined with a better function name (e.g., ``partition_by_permno``), such rich type-hints provide an interface that makes the functionality explicit.

Even better, these type hints can be used for static analysis with Pyright (now) and MyPy (pending full ``TypeVarTuple`` support). Calling such a function with a ``Frame`` of two columns of ``np.float64``, for example, will fail a static analysis type check or deliver a warning in your editor.


## Runtime Validation

Static type checking might not be enough: runtime evaluation provides even stronger constraints, even for incomplately typed values.

StaticFrame 2 introduces a built-in decorator, ``@CallGuard.check``, that uses the same type hints for runtime validation. Extended with one or more validators enclosed in an `Annotated` generic, additional run-time validation of shape, labels, or arbitrary functional checks can be included.
