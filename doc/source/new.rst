What is New in StaticFrame
===============================


2.11-dev
-----------

All ``via_hashlib`` interfaces will now raise if an object ``dtype`` is encountered.

Improved identification of mappings by using ``collections.abc.Mapping``.


2.10.3
-----------

Corrected issue in package publishing.


2.10.2
-----------

Corrected issue in ``Frame.pivot`` whereby, under some scenarios, ``fill_value`` is not applied.


2.10.1
-----------

Updated ``arraykit`` to 0.8.2.


2.10.0
-----------

Support for NumPy 2.

Updated ``arraykit`` to 0.8.0.

Updated ``arraymap`` to 0.3.0.

Updated ``frame-fixtures`` to 1.1.0.


2.9.0
-----------

Added preliminary support for DuckDB with ``Frame.to_duckdb()``, ``Frame.from_duckdb()``, ``Bus.to_duckdb()``, ``Bus.from_duckdb()``, ``Quilt.to_duckdb()``, ``Quilt.from_duckdb()``, ``Batch.to_duckdb()``, ``Batch.from_duckdb()`` and ``Yarn.to_duckdb()``.

Added ``name`` parameter to ``Frame.from_xlsx()``, ``Frame.from_sqlite()``, and ``Frame.from_hdf5()```

Optimized routines for column iteration and ``Frame.transpose()``.


2.8.2
-----------

Updated ``arraykit`` to 0.7.2. Integrated enhancements in all 2D array to tuple routines.


2.8.1
-----------

Updated ``arraykit`` to 0.6.3.


2.8.0
-----------

New implementation, improved performance, and corrected issues in ``Frame.join_left``, ``Frame.join_right``, ``Frame.join_inner``, and ``Frame.join_outer``.

Updated ``arraykit`` to 0.6.2.

Updated ``typing-extensions`` to 4.12.0.


2.7.0
-----------

``CalllGuard`` and other ``TypeClinic`` interfaces now support run-time usage of ``TypeVar``.


2.6.0
-----------

Internal redesign of ``Yarn`` to permit arbitrary reording of ``Frame`` in contained ``Bus``.

Added ``reindex()``, ``sort_index()``, ``sort_values()``, ``roll()``, and ``shift()`` methods to ``Yarn``.

Improved bytes representation used by ``Yarn.via_hashlib`` to exclude internal implementation configuration.


2.5.2
-----------

Improvements to ``Yarn`` to always support unique labels after ``retain_labels`` application.


2.5.1
-----------

Extended ``TypeClinic`` and ``CallGuard`` support for generic NumPy ``generic``.


2.5.0
-----------

Removed support for Python 3.8.

Set minimum ``numpy`` to 1.22.4.

Set minimum ``mypy`` to 1.9.0.

Removed support for pre 1.0 Pandas.

Updated minimum versions of optional packages.


2.4.0
-----------

Added ``isna()``, ``notna()``, ``isfalsy()``, and ``notfalsy()`` methods to all ``IndexBase`` subclasses.


2.3.0
-----------

All ``prod()`` and ``sum()`` methods now have an ``allna`` parameter to, when ``fillna`` is ``True``, specify a value if all values are NA.

Improved type hints for interfaces that accept ``IO`` objects.

URLs given to the ``WWW`` interface now escape spaces in path components.

Performance enhancements to low-level ``TypeBlocks`` routines.


2.2.3
-----------

Corrected issue with ``Bus`` with ``IndexHierarchy`` and ``max_persist`` greater than one.


2.2.2
-----------

Corrected handling of ZIP64 extensions in ``zip_namelist``.


2.2.1
-----------

Corrected issue with non-sequential ``Bus`` selections with ``max_persist`` greater than one, resulting in incorrectly sized generators.

Performance enhancements to ``Bus`` loading routines with ``max_persist``.

Corrected handling of ZIP64 extensions in ``ZipFileRO``.


2.2.0
-----------

Performance enhancements to reading NPZ files and ZIP NPZ archives.


2.1.2
-----------

``ContainerBase`` added to the public namespace.


2.1.1
-----------

Updated release automation in CI.


2.1.0
-----------

Added ``TSeriesAny``, ``TSeriesHEAny``, ``TBusAny`` generic aliases.

Added ``Frame.to_json_typed()`` and ``Frame.from_json_typed()``, supporting optimized, complete round-trip ``Frame`` serialization.

Set ``arraykit`` to 0.5.1, now with functional wheels for Mac OS ``arm64`` / Apple Silicon.

Set ``arraymap`` to 0.2.2, now with functional wheels for Mac OS ``arm64`` / Apple Silicon.

Added support for Python 3.12.


2.0.1
-----------

Corrected import in ``setup.py``.


2.0.0
-----------

Nearly all containers are now generic, permitting specification of component types for static analysis and runtime validation.

Added ``CallGuard``, exposing decorators for performing run-time function interface type check and data validation.

Added ``Require``, a collection of classes to be used for run-time validation in ``tp.Annotated`` generics.

Added ``TypeClinic``, ``ClinicResult``, and ``ClinicError``, providing type to hint conversion and checks against arbitrary hints.

Added ``via_type_clinic`` interfaces to all containers, exposing a pre-configured ``TypeClinic`` interface.

Now performing static-analysis with both MyPy (1.6.1) and Pyright (1.1.331).


1.6.5
-----------

Implemented fall-back behavior for Parquet files written with  ``pyarrow`` less than 1.0.


1.6.4
-----------

Improvements to typing through usage of ``overload`` signature definitions.

Set ``arraykit`` to 0.4.10.


1.6.3
-----------

Added ``label_missing_skips`` to all ``iter_window`` and related interfaces.


1.6.2
-----------

Added ``label_missing_raises`` to all ``iter_window`` and related interfaces.

Set ``typing-extensions`` to greater than or equal to 4.7.1


1.6.1
-----------

Corrected exception raised in ``IndexHierarchy.level_drop()`` when provided count is 0.

Added ``typing-extensions`` as a dependency for back-ported typing features.

Updated ``mypy`` configuration for broader checks.

Numerous typing improvements and enhancements.


1.6.0
-----------

Comprehensive improvements and expansions to type hints using Python 3.11 and NumPy 1.25.


1.5.0
-----------

Removed support for Python 3.7.

Set minimum ``numpy`` to 1.19.5.

Improved type hints for ``loc`` and ``iloc`` interfaces on 2D containers.

Usage of ``from __future__ import annotations`` in all modules.

Normalized exceptions raised from invalid selections to raise ``KeyError``.


1.4.6
-----------

Set ``arraykit`` to 0.4.9.


1.4.5
-----------

Imports of ``ThreadPoolExecutor`` and ``ProcessPoolExecutor`` are now deferred until used.

Multiprocessing interfaces that use ``ProcessPoolExecutor`` now accept the ``mp_context`` parameter.


1.4.4
-----------

Set ``arraymap`` to 0.1.9.

Updated ``TypeBlocks`` to use ``BlockIndex.iter_block()``.


1.4.3
-----------

Integrated ``arraykit.BlockIndex`` in ``TypeBlocks``, offering significant ``Frame`` performance improvements.

``via_dt`` will now raise if an NA is encountered and a ``fill_value`` is not provided.

Corrected issue in ``Quilt`` construction when component ``Frame`` have ``IndexHierarchy``.

Corrected issue when using an ``Index`` as a selector in ``via_fill_value().loc[]``.

Added ``ErrorInitColumns`` to better indicated failures in column ``Index`` construction.


1.4.2
-----------

Improved performance when assigning an ``Index`` with ``FrameGO.__setitem__()``.

Improved typing and input handling for ``StoreFilter`` values.

Set ``arraymap`` to 0.1.8.

Removed deprecated JSON interfaces: ``Frame.from_json()``, ``Frame.from_json_url()``.


1.4.1
-----------

Optimizations to reduce ``TypeBlocks`` size and initialization time.


1.4.0
-----------

Replaces ``automap`` with ``arraymap`` 0.1.7, offering optimized ``Index`` performance.


1.3.2
-----------

Corrected issue with ``Frame.iter_element().apply()`` whereby ``columns.name`` are not preserved.

Corrected issue when selecting a single row from a ``Frame`` that has object columns that contain sequences.

Set ``arraykit`` version to 0.3.4.


1.3.1
-----------

Set ``arraykit`` version to 0.3.1.


1.3.0
-----------

The ``via_dt`` interface now supports usage of a ``fill_value`` argument for handling missing values.

Added ``unique_enumerated()`` method to ``Series`` and ``Frame``.


1.2.7
-----------

Set ``arraykit`` version to 0.3.0: replaced usage of ``np.nonzero()`` with ``first_true_1d()`` and ``first_true_2d()``.

Specifiers of dtypes given to ``astype()`` and related interfaces will now permit "object" to be used (in addition to ``object``).


1.2.6
-----------

Corrected issue when selecting ``IndexHierarchy`` labels that contain tuples.

``IndexHiearchy.astype()`` now properly delegates the ``name`` attribute.

Better usage of ``np.concatenate`` in ``TypeBlocks.transpose()`` and ``HierarchicalLocMap.build_offsets_and_overflow()``.


1.2.5
-----------

Specifiers of dtypes given to ``astype()`` and related interfaces will now raise if NumPy will implicitly convert the argument to ``object``.

``Batch.to_frame()`` now exposes ``index_constructor``, ``columns_constructor`` arguments.

Improvements to usage of ``index_constructor`` and ``columns_constructor`` arguments in ``Frame.from_concat`` when ``Series`` are provided as components.

When using a ``Batch`` to write to an archive, non-unique labels will now raise a ``StoreLabelNonUnique`` exception.


1.2.4
-----------

``IndexHierarchy.__setstate__`` now properly sets indexers to be immutable.

``Bus`` with associated ``Store`` instances are now pickleable after removing ``_weak_cache`` in ``Store.__getstate__()``.


1.2.3
-----------

``Series.isna()``, ``Series.dropna()``, and related functions now properly handle arrays that contain objects such as ``Frame`` or ``np.ndarray`` that raise for usage of ``__bool__()``.

Set ``arraykit`` version to 0.2.9: ``isna_element`` now identifies ``pd.Timestamp('nat')`` as a NA value, and invalid ``datetime64`` strings given to ``Frame.from_delimited`` and related interfaces now properly raise exceptions.


1.2.2
-----------

Corrected issue in ``IndexHierarchy.label_widths_at_depth()`` which caused incorrect cell merging in XLSX output when ``merge_hierarchical_labels`` is True.


1.2.1
-----------

Corrected issue in ``Series.dropna()`` whereby full drops would not retain the index class.

Performance enhancement to ``TypeBlocks.equals()`` and related routines using ``arrays_equal()``.


1.2.0
-----------

Significant performance optimizations to ``IndexHierarchy`` set operations, as well as optimized pathways for determining ``TypeBlocks`` equality.

JSON metadata in NPY and NPZ encodings of ``Frame`` data now properly encodes and decodes ``np.datetime64`` and ``datetime.date`` objects.

Corrected issue on Python 3.11 in the creation of ``memory`` displays due to usage of ``Enum``.

Corrected issue in ``Frame.relabel_shift_out()`` where ``index_constructors`` are not assigned to subset index.

Extended ``Frame.iter_tuple()`` ``constructor`` argument to support ``dataclass``-created classes.


1.1.1
-----------

Corrected handling of 0-sized containers in ``Frame.insert_before()`` and ``Frame.insert_after()``.

Corrected issue with some ``IndexHierarchy`` formations when using slices in an ``HLoc`` with more than one depth selection.


1.1.0
-----------

Added ``Frame.consolidate`` interface, including ``Frame.consolidate[]`` and ``Frame.consolidate.status``.

Added ``Quilt.bus`` property.

``IndexHierarchy.rehierarch()`` and related routines now correctly reorder index constructors by default.

Added ``index_constructors`` arguments to ``IndexHierarchy.rehierarch()``, ``Series.rehierarch()``, ``Bus.rehierarch()``, ``Yarn.rehierarch()``.

Added ``index_constructors``, ``columns_constructors`` arguments to ``Frame.rehierarch()``.


1.0.1
-----------

Parameters ``dtypes``, ``fill_value``, and ``format`` (given to ``via_str.format``) now properly work with ``defaultdict``, infinite iterators, and mappings indexed by position (when columns are not defined or created with ``IndexAutoFactory``).


1.0.0
----------

API change: ``IndexHierarchy`` numerical and statistical methods, such as ``sum()``, ``var()``, ``std()``, ``cumprod()``, ``cumsum()``, now raise ``NotImplementedError``.

API change: ``Frame.astype[]`` calls now set ``consolidate_blocks`` to ``False`` by default.

API change: ``composite_index`` and ``composite_index_fill_value`` parameters removed from ``Frame.join_left``, ``Frame.join_right``, ``Frame.join_inner``, and ``Frame.join_outer``; added ``include_index`` parameter.

API change: ``IndexDefaultFactory`` renamed ``IndexDefaultConstructorFactory``.

Added ``via_dt.year_month``.


0.9.23
----------

Added ``via_hashlib`` interface to all containers.

Added ``Frame.iter_group_other()``, ``Frame.iter_group_other_items()``, ``Frame.iter_group_other_array()``, ``Frame.iter_group_other_array_items()``.

Added ``Series.iter_group_other()``, ``Series.iter_group_other_items()``, ``Series.iter_group_other_array()``, ``Series.iter_group_other_array_items()``.

Set ``arraykit`` version to 0.2.6.


0.9.22
----------

``IndexYear`` now accepts selection by integers for years.

Added ``Series.loc_notna_first()``, ``Series.loc_notna_last()``, ``Series.loc_notfalsy_first()``, ``Series.loc_notfalsy_last()``.

Added ``Frame.loc_notna_first()``, ``Frame.loc_notna_last()``, ``Frame.loc_notfalsy_first()``, ``Frame.loc_notfalsy_last()``.

Set ``arraykit`` version to 0.2.4.


0.9.21
----------

Set ``arraykit`` version to 0.2.3.

Set ``automap`` version to 0.6.2.

Improvements to delimited file parsing when ``index_depth`` is greater than zero and quoted fields or escaped characters are used.

Corrected issue where Pandas ``MutliIndex`` are not converted to ``IndexHierarchy`` in ``Frame.from_pandas()`` or ``Series.from_pandas()``.


0.9.20
----------

Added ``via_str.format()`` for applying Python formatting mini-language strings to elements.

Added ``skip_initial_space``, ``quoting``, ``quote_double``, and ``escape_char`` parameters to ``Frame.from_delimited()`` and related interfaces.

Added ``Frame.from_json_index()``, ``Frame.from_json_columns``, ``Frame.from_json_split()``, ``Frame.from_json_records()``, ``Frame.from_json_values()``.

Added ``Frame.to_json_index()``, ``Frame.to_json_columns``, ``Frame.to_json_split()``, ``Frame.to_json_records()``, ``Frame.to_json_values``.

Added ``axis`` argument to ``Series.count()`` for compatibility with ``Frame`` interface.


0.9.19
----------

``Frame.from_delimited`` now powered by ``arraykit.delimited_to_arrays``, offering vastly improved performance for all delimited file reading.

Added ``Series.from_delimited``.

Added ``WWW`` interface for downloading network resources to provide to constructors.

Added ``columns_select``, ``skip_initial_space``, ``quoting``, ``quote_double``, ``escape_char``, ``thousands_char``, and ``decimal_char`` parameters to ``Frame.from_delimited`` and derived interfaces.

Corrected issue in converting to ``IndexHierarchy`` from Pandas ``MultiIndex`` when the ``MultiIndex`` is bloated.


0.9.18
----------

Improved layout of default ``memory`` display.

Corrected issue where a non matching ``Series`` assignment or reindex might use ``fill_value`` to determine the type of the returned values.

Certain invalid ``Frame`` selections now properly raise ``KeyError``.


0.9.17
----------

Unified implementation of ``IndexBase`` set and concatenation operations.

Added ``IndexHierarchy.from_values_per_depth()`` constructor.

Set ``automap`` version to 0.6.1.

Extended ``Batch.astype`` interface to fully cover ``Frame.astype`` interface.


0.9.16
----------

Added ``memory`` property to display memory usage via ``MemoryDisplay`` interfaces.

Implemented the DataFrame Interchange Protocol export via the ``Frame.__dataframe__()`` interface.

Implemented ``__repr__()`` for ``ILoc``.

Updated ``Batch.__repr__()``.

Added ``name`` parameter to ``Frame.to_frame()``, ``Series.to_frame()``, and related methods.

Improved error reporting for invalid ``IndexHierarchy``.

``IndexHierarchy.from_index_items()`` now supports items of ``IndexHierarchy``.


0.9.15
----------

Added ``__repr__()`` for ``HLoc``.

Added ``IndexHierarchy.index_at_depth()``

Added ``IndexHierarchy.indexer_at_depth()``

Added ``depth_level`` and ``order_by_occurrence`` parameters to ``unique()`` on ``IndexBase`` subclasses.

Corrected issue calling ``Frame.to_pandas()`` with an empty ``Frame``.

Implemented argument checking on all ``IndexHierarchy`` depth selection parameters.

``IndexHierarchy.astype()`` now accepts a ``dtypes`` argument to assign dtypes by tuple or mapping.

Corrected return type of ``via_str.contains()`` to return Booleans.


0.9.14
----------

Added ``Frame.corr()``, ``Series.corr()``, and ``Batch.corr()``.

Added ``compression`` argument to ``StoreClientMixin`` exporters that write ZIPs.

Corrected issue with selection on zero-sized ``IndexHierarchy``.


0.9.13
----------

Added ``to_zip_npy()`` and ``from_zip_npy()`` interfaces to ``Bus``, ``Batch``, ``Yarn``, and ``Quilt``.

Added ``test_example_gen.py`` to test and enforce automatic example generation.

Improved usage of ``__slots__`` throughout.

Continuous integration quality checks now using pylint 2.15.0 and isort 5.10.1

Minimum pyarrow set to 0.17.0


0.9.12
----------

``ErrorNPYEncode`` exceptions raised during authoring NPZ files or NPY directories now remove those files or directories.

``ErrorNPYEncode`` exceptions raised during ``to_zip_npz()`` now remove the archive.

Authoring NPYs to a pre-existing directory will now raise a ``RuntimeError``.


0.9.11
----------

Supplying an ``IndexDatetime`` subclass as an ``explicit_constructor`` to an ``IndexAutoFactory`` now raises.

Corrected issue with ``fillna`` and ``fillfalsy`` functions when non-elemental fill values are supplied.

Updated ``arraykit`` to 0.1.13.


0.9.10
----------

Corrected single depth selection issue with ``IndexHierarchy``.


0.9.9
----------

Updated ``arraykit`` to 0.1.12.

Set minimum ``numpy`` to 1.18.5.

Added ``index_constructors`` parameter to ``Frame.relabel_shift_in()``.

Corrected issue when ``Frame.astype()`` called with an empty ``Frame``.

Extended ``via_values`` property to take optional consolidation arguments via ``__call__()`` constructor; usage of ``via_values`` instance returns same-typed, same-sized container.


0.9.8
----------

Added ``via_values`` property to ``Series``, ``Index``, ``Frame``, ``IndexHierarchy`` and ``Batch``; permits applying functions to complete containers with ``apply()`` and supports usage as arguments in arbitrary NumPy functions with ``__array_ufunc__()``.

Corrected usage of ``IterNodeDelegate`` with iterator endpoints that do not iterate hashables; added ``IterNodeDelegateMapable`` for usage with iterators of hashables.

Improved type and dtype preservation in concatenation and set operations on ``IndexHierarchy``.

Normalized ordering of results from ``Frame.bloc[]`` selections to row-major ordering without sorting labels.

Added ``via_str.contains()``.

Corrected issue in ``ArchiveZIP`` when ``__del__`` is called when no archive is set.

``Frame.to_sqlite()`` now requires a named ``Frame`` or an explicit ``label``; ``Frame.from_sqlite()``, ``Frame.from_hdf5()`` now make ``label`` a required argument.

API Documentation re-organized, now using procedurally generated code examples.


0.9.7
----------

Corrected issue in ``Series.from_overlay()`` that prematurely aborted processing all ``Series``

Normalized ordering of results from ``Frame.bloc[]`` selections.


0.9.6
----------

Corrected issue in ``Quilt`` creation when given a ``Bus`` with ``Frame`` with ``datetime64`` indices.

Extended ``IndexAutoConstructorFactory`` to evaluate dtype from arbitrary iterables, not just arrays.

Improvements to consistency and performance of ``loc_to_iloc``.

Implemented ``max`` and ``min`` methods on ``IndexHierarchy``; related statistical methods now raise.


0.9.5
----------

Updated AutoMap to 0.5.1

Removed "performance" package from ``setup.py``.


0.9.4
----------

Enhanced support for ``fill_value`` as a ``FillValueAuto``, a mapping, or a sequence of fill values (per column) where appropriate.

Added ``Index.dropna()``, ``Index.dropfalsy()``, ``IndexHierarchy.dropna()``, ``IndexHierarchy.dropfalsy()``.

Added ``Index.fillfalsy()``, ``IndexHierarchy.fillfalsy()``.

Performance improvements for ``Frame.iter_group_labels()``, ``Frame.iter_group_labels_items()``, ``Frame.iter_group_labels_array()``, ``Frame.iter_group_labels_array_items()``.

Fixed usage of ``dtypes`` argument when encountering zero-sized data in ``Frame.from_records()`` and ``Frame.from_pandas()``.

Improved ``Frame.iter_tuple`` to not coerce types through arrays.

Added ``Frame.set_columns()``, ``Frame.set_columns_hierarchy()``, and ``Frame.unset_columns()``.


0.9.3
----------

Added ``apply_element()`` and ``apply_element_items()`` methods to ``FrameAssign`` and ``SeriesAssign`` interfaces.

Added implementation of ``__array__()`` and ``__array_ufunc__()`` to all containers for better support with NumPy objects and binary operators.

Added ``Series.iter_group_array()``, ``Series.iter_group_array_items()``, ``Series.iter_group_labels_array()``, ``Series.iter_group_labels_array_items()``.

Added ``Frame.iter_group_array()``, ``Frame.iter_group_array_items()``, ``Frame.iter_group_labels_array()``, ``Frame.iter_group_labels_array_items()``.

Corrected issue when using binary operators with a ``FrameGO`` and a ``Series``.

Corrected issue and performance of ``name`` assignment when extracting ``Series`` from ``Frame`` with an ``IndexHierarchy``.

Added ``IndexAutoConstructorFactory`` for automatic constructor selection based on NumPy dtype.


0.9.2
----------

Corrected more issues when calling ``IndexHierarchy.loc[]`` with another ``IndexHierarchy``, or when calling ``Frame.assign.apply`` when that frame has ``IndexHierarchy`` columns.

Corrected undesirable type coercion from happening in single-row selections from ``IndexHierarchy``.


0.9.1
----------

Corrected issue when calling ``IndexDatetime.loc[]`` with an empty list.

Corrected issue when calling ``IndexHierarchy.loc[]`` with another ``IndexHierarchy``


0.9.0
----------

API change: ``Bus`` no longer accepts a ``Series`` on initialization; use ``Bus.from_series()``.

API change: ``Batch`` no longer normalizes containers after each step in processing; use ``Batch.via_container`` to force elements or arrays to ``Frame`` or ``Series``.

API change: ``Index`` objects can no longer be created with ``np.datetime64`` arrays; such labels must use an ``IndexDatetime`` subclass instead. If this is happening implicitly with an operation, that operation should expose a parameter for ``index_constructor`` or ``index_constructors``.

API change: ``IndexAutoFactory`` is no longer accepted as an ``index_constructor`` argument in ``Series.from_pandas()`` and ``Frame.from_pandas()``; ``IndexAutoFactory`` should be passed as an ``index`` or ``columns`` argument instead.

Minimum Python version is now 3.7

New implementation of ``IndexHierarchy``, offering significantly improved performance and removal of the requirement of tree hierarchies.

Added ``Batch.to_series()``.

Fixed issue when using ``Frame.from_npz`` with an NPZ created with a ``FrameGO``.

Fixed issue when supplying overspecified mappings to ``Frame.astype``.


0.8.38
----------

Further improved handling of binary equality operators with ``IndexDatetime`` subclasses.


0.8.37
----------

Improved handling of binary equality operators with ``IndexDatetime`` subclasses.


0.8.36
----------

Extended interface of ``Batch`` to include all methods for handling missing values, as well as all ``via_*`` interfaces.

Silenced all NumPy warnings where the issue raised in the warning is being explicitly handled in the code.


0.8.35
----------

Performance enhancements to ``Frame.pivot()``, ``Frame.iter_group()``, and ``Frame.iter_group_items()``.

``Frame.pivot()`` ``func`` parameter can now be set to ``None`` to perform no aggregation.

Extended ``Series.from_overlay()`` and ``Frame.from_overlay()`` to support ``func`` and ``fill_value`` arguments; ``func`` can be used to optionally specify what elements are available for assignment in overlay.

Extended ``via_fill_value()`` interfaces to implement ``__getitem__`` and ``loc`` selection interfaces on ``Series`` and ``Frame`` for selections that potentially contain new labels filled with the fill value.


0.8.34
----------

Added ``NPY`` and ``NPZ`` interfaces for creating NPY and NPZ archvies from arrays and ``Frame`` components.

Added ``index_constructors`` argument to ``IndexHierarchy.from_product()``

Added ``index_constructor`` argument to ``Index.level_add()``, ``IndexHierarchy.level_add()``, and  ``Frame.relabel_level_add()``.

Added ``index_constructor``, ``columns_constructor`` arguments to ``Frame.relabel()``.

Added ``Series.to_frame_he()``.

Added ``index``, ``index_constructor``, ``columns``, ``columns_constructor`` arguments to ``Series.to_frame()``, ``Series.to_frame_go()``, ``Series.to_frame_he()``.

Improvement to ``Frame.from_concat`` to avoid creating one-element indices from ``Series`` when an index is provided along the appropriate axis.

Added ``index_constructor`` argument to ``Series.relabel()``.

Added ``index_constructor`` argument to ``Series.from_concat()``.

Added ``index`` argument to ``Series.from_pandas()``.

Added ``index`` and ``columns`` argument to ``Frame.from_pandas()``.

Improvement to ``Index`` initialization to raise ``ErrorInitIndex`` if given a single string as ``labels``.

Set operations on labels of different ``datetime64`` units now raise an ``Exception``.


0.8.33
----------

Performance enhancements to ``Frame.from_npy`` and ``Frame.from_npz``.


0.8.32
----------

Added ``Frame.to_pickle()``, ``Frame.from_pickle()``.

Added ``index_constructor``, ``columns_constructor`` to ``Frame.from_concat``.

Fixed issue in ``Frame.insert_after()``, ``Frame.insert_before()``,  ``Series.insert_after()``, ``Series.insert_before()`` with negative ``ILoc`` labels.


0.8.31
----------

Added ``Frame.from_npy_mmap``; removed ``memory_map`` option from ``Frame.from_npy``.


0.8.30
----------

Performance enhancements to ``Frame.from_npy`` and ``Frame.from_npz``.


0.8.29
----------

Added ``consolidate_blocks`` Boolean parameter to ``Frame.to_npz()`` and ``Frame.to_npy``.


0.8.28
----------

Added ``Frame.to_npy()``, ``Frame.from_npy()`` with a ``memory_map`` option.

Improvements to ``Frame.to_npz()`` to support large files and buffered writes.

Performance enhancements to all ``_StoreZip`` subclasses through usage of ``WeakValueDictionary`` caching.

Added ``IndexHiearchy.relabel_at_depth()``.

Added support for string slicing and selection with ``Series.via_str[]`` and ``Frame.via_str[]``.


0.8.27
----------

Reimplemented ``Frame.to_npz()``, ``Frame.from_npz()``, removing support for object arrays (and pickles) and improving performance.

Added ``Bus.to_zip_npz()``, ``Bus.from_zip_npz()``, ``Quilt.to_zip_npz()``, ``Quilt.from_zip_npz()``, ``Batch.to_zip_npz()``, ``Batch.from_zip_npz()`` and ``Yarn.to_zip_npz()``.

Implemented ``Series.fillfalsy_forward()``, ``Series.fillfalsy_backward()``, ``Series.fillfalsy_leading()``, ``Series.fillfalsy_trailing()``.

Implemented ``Frame.fillfalsy_forward()``, ``Frame.fillfalsy_backward()``, ``Frame.fillfalsy_leading()``, ``Frame.fillfalsy_trailing()``.

Added ``Quilt.equals()``.

``Frame.from_pandas()`` now supports zero-sized DataFrame.

Fixed issue in ``Frame.set_index()`` where ``column`` is passed as ``None``.

Removed ``TypeBlocks._block_slices``.


0.8.26
----------

``Frame.to_pandas()`` now creates ``pd.RangeIndex`` for ``IndexAutoFactory``-created indices.

Performance enhancements to ``Frame.from_concat()``.


0.8.25
----------

Corrected issue extracting containers stored in ``Series``.


0.8.24
----------

Improved dtype resoltion on ``Frame`` methods that reduce dimensionality.


0.8.23
----------

Corrected issue where summing a ``Frame`` of Booleans along axis 0 resulted in Booleans instead of integers.


0.8.22
----------

Performance enhancements to ``Frame.iter_group()`` and ``Frame.iter_group_items()``.


0.8.21
----------

Added ``Frame.to_npz()``, ``Frame.from_npz()``.

Performance enhancements to ``Frame.iter_group()`` and ``Frame.iter_group_items()``.

Performance enhancements to ``Frame.pivot()``.

Added ``drop`` parameter to ``Frame.iter_group()`` and ``Frame.iter_group_items()``.

Introduction of ``TypeBlocks._block_slices`` as lazily derived and persistently stored.

Fixed issue with ``Frame.from_overlay`` when called with ``FrameGO``.

Added ``index_constructor`` argument to ``apply``, ``apply_pool``, ``map_any``, ``map_fill``, ``map_all``.


0.8.20
----------

Added ``dtypes`` parameter to ``Frame.from_pandas()``.

Added ``index_constructors``, ``columns_constructors`` to the following interfaces: ``Frame.from_sql()``, ``Frame.from_structured_array()``, ``Frame.from_delimited()``, ``Frame.from_csv()``, ``Frame.from_clipboard``, ``Frame.from_tsv()``, ``Frame.from_xlsx()``, ``Frame.from_sqlite()``, ``Frame.from_hdf5()``, ``Frame.from_arrrow()``, ``Frame.from_parquet()``.

``StoreConfig`` now exposes ``index_constructors`` and ``columns_constructors`` arguments.

Incorrectly formed ``Batch`` iterables will now, upon iteration, raise a ``BatchIterableInvalid`` exception.

Added ``Quilt.sample()``.

``all()`` and ``any()`` on ``Series`` and ``Frame`` no longer raise when NA values are present and ``skipna`` is ``False``.

Performance enhancements to ``Bus`` loading routines when using ``max_persist`` by refactoring internal architecture of ``Bus`` to no longer hold a reference to a ``Series`` but instead use a mutable array.


0.8.19
----------

Optimization of ``Bus.items()``, ``Bus.values``,  ``Bus.iter_element()``, and ``Bus.iter_element_items()`` when ``max_persist`` is greater than one.

Added ``Yarn.iter_element()``, ``Yarn.iter_element_items()``.

Added ``Yarn.drop[]``

Added ``Yarn.reindex()``, ``Yarn.relabel_flat()``, ``Yarn.relabel_level_add()``, ``Yarn.relabel_level_drop()``, ``Yarn.rehierarch()``.

Added ``Bus.unpersist()``, ``Yarn.unpersist()``, and ``Quilt.unpersist()``.

Improvements to standard string representation of ``Quilt``.

Added ``is_month_start()``, ``is_month_end()``, ``is_year_start()``, ``is_year_end()``, ``is_quarter_start()``, ``is_quarter_end()`` to ``via_dt`` interfaces.

Added ``hour``, ``minute``, ``second`` properties to ``via_dt`` interfaces.

Improved implementation of ``weekday()``, added ``quarter()`` to ``via_dt`` interfaces.

Fixed issue when using ``iter_window_*`` methods on two-dimensional containers where the opposite axis is not a default index constructor.

Fixed issue when selecting rows from ``Frame`` with 0-length columns.


0.8.18
----------

Implementation of ``Yarn()``, a container that presents numerous ``Bus`` as a uniform, 1D interface.

Fixed issue in ``Frame.astype[]`` when selecting targets with a Boolean ``Series`` or arrays.

Fixed unnecessary type coercion in the ``Frame`` returned by ``Frame.drop_duplicated()``.

Improved handling of reindexing and lookups between datetime64 and date / datetime objects.

``Frame.equals()``, ``Series.equals()``, ``Index.equals()``, ``IndexHiearchy.equals()`` and all related routines now distinguish by ``datetime64`` unit in evaluating basic equality.


0.8.17
----------

Extended ``Series.count()`` and ``Frame.count()`` with ``skipfalsy`` and ``unique`` parameters.

Added ``Series.isfalsy()``, ``Series.notfalsy()``, ``Series.dropfalsy()``, ``Series.fillfalsy()``.

Added ``Frame.isfalsy()``, ``Frame.notfalsy()``, ``Frame.dropfalsy()``, ``Frame.fillfalsy()``.

Exposed ``isna_element()`` via ``arraykit`` on root namespace.

Added ``Bus.from_concat()``.

Added ``Bus.to_series()``.

``Bus.reindex()``, ``Bus.relabel()``, ``Bus.relabel_flat()``, ``Bus.relabel_level_add()``, ``Bus.relabel_level_drop()``, ``Bus.rehierarch()`` now, if necessary, load all contents from the associated ``Store`` and return a ``Bus`` without a ``Store`` association.

Added ``index_constructor`` argument to ``Series.from_concat_items()``.

Added ``index_constructor``, ``columns_constructor`` arguments to ``Frame.from_concat_items()``.

Introduced ``IndexDefaultConstructorFactory`` to permit specifying index ``name`` attributes with default index constructors.


0.8.16
----------

Added ``Frame.to_series()``.

``Frame.sort_values()``, ``Frame.sort_index()``, ``Frame.sort_columns``, ``Series.sort_index()``, and ``IndexHierarchy.sort()`` now accept ``ascending`` as an iterable of Booleans to specify value per vector.

``FrameGO.via_fill_value()`` now supports providng a fill value in ``__setitem__()`` assignment.

``IndexAutoFactory`` can now be instantiated with a ``size`` parameter to pre-set the size of an auto-index, such as when used to initialize a ``FrameGO``.


0.8.15
----------

Added support for loading containers into specialized VisiData ``Sheet`` and  ``IndexSheet`` subclasses; added ``to_visidata()`` exporter to all containers.

Added ``StyleConfig`` class for configuring display characteristics. Added default ``StyleConfigCSS`` for improved default HTML presentation.

Added ``Series.rank_ordinal``, ``Series.rank_dense``, ``Series.rank_mean``, ``Series.rank_min``, ``Series.rank_max``.

Added ``Frame.rank_ordinal``, ``Frame.rank_dense``, ``Frame.rank_mean``, ``Frame.rank_min``, ``Frame.rank_max``.

Fixed issue in ``Series.from_element()`` and ``Frame.from_element()`` that would broadcast some iterables instead of treat them as an element.

Extended ``Frame.unset_index()`` to support unsetting ``IndexHierarchy``.


0.8.14
----------

Added ``index_continuation_token`` and ``columns_continuation_token`` to ``Frame.from_delimited()`` and related methods.

Added ``via_re()`` interfaces to ``Index``, ``IndexHierarchy``, ``Series``, ``Frame``.

Updated ``arraykit`` to 0.1.8


0.8.13
----------

Integration with ``arraykit``; replacement of numerous utility methods with ``arraykit`` implementations.

Added ``via_fill_value()`` interface to ``Series`` and ``Frame``.


0.8.12
----------

Performance enhancements to ``Quilt.iter_series().apply()``, ``Quilt.iter_tuple().apply()``, ``Quilt.iter_array().apply()``.


0.8.11
----------

Fixed issue when supplying ``dtype`` arguments to ``apply`` methods with string dtypes.

Added ``parameters`` argument to ``Frame.from_sql`` to perform SQL parameter substitution.

In group-by operations where the group key is a hashable, the returned ``Index.name`` will be set to that key.

Performance enhancements to ``Bus.iter_element().apply()`` and `Bus.iter_element_items().apply()``.


0.8.10
----------

Performance enhancements to ``Index`` initialization.

Performance enhancements to ``Series.iter_element().apply()``, ``Series.iter_element().map_any()``, ``Series.iter_element().map_all()``, and ``Series.iter_element().map_fill()``.

Performance enhancements to ``Frame.iter_series().apply()``, ``Frame.iter_tuple().apply()``, ``Frame.iter_array().apply()``.


0.8.9
----------

Performance enhancements to ``Series.dropna()``.

``Series.relabel()`` and ``Frame.relabel()`` now raise if given a ``set`` or ``frozenset``.

Fixed issue in ``Frame.assign.loc[]`` when using a Boolean array as a column selector.


0.8.8
----------

Added ``Frame.cov()``, ``Series.cov()``, and ``Batch.cov()``.

Performance enhancements to ``loc`` selections by element.


0.8.7
----------

Implemented support for multiprocessing Frame writing from ``StoreZip`` subclasses used by ``Bus``, ``Batch``, and ``Quilt``.

Enabled ``write_max_workers`` and ``write_chunksize`` in ``StoreConfig``.

Added py.typed file to package.

Improved exceptions raised when attempting to write to a file at an invalid path.

Improved handling of reading files with columns but no data with ``Frame.from_delimited``.


0.8.6
----------

``Frame.rename`` now accepts optional arguments for ``index`` and ``columns`` renaming.

``Series.rename`` now accepts an optional argument for ``index`` renaming.

Added ``Frame.relabel_shift_in()`` and ``Frame.relabel_shift_out()``.

Fixed issue where ``Frame.dropna()`` fails on single-columns ``Frame``.

Extended ``IndexHierarchy.level_drop`` to perform corresponding drops on ``name`` when ``name`` is an appropriately sized tuple.

Extended ``Frame.set_index`` to support creating a 1D index of tuples when more than one column is selected.


0.8.5
----------

``Frame.from_sql`` now properly applies ``dtypes`` to columns used by ``index_depth`` selections.

Added ``Index.unique`` and ``IndexHierarchy.unique``, both taking a ``depth_level`` specifier for selecting one or more depths.

Fixed issue with ``Frame.bloc`` selections that result in a zero-sized ``Series``.


0.8.4
----------

Refined ``Frame.bloc`` selections to reduce type coercion.

Improved ``Frame.assign.bloc`` when assigning with ``Series`` and ``Frame``.


0.8.3
----------

Added ``iloc_searchsorted()`` and ``loc_searchsorted()`` to ``Index``, ``IndexDatetime``, and ``IndexHierarchy``.

Added ``ddof`` parameter to all containers that expose ``std`` and ``var``.

Fixed issue with ``Frame.assign`` where there was a dependency on the order of column labels given in selection.

Improved handling for NumPy Boolean types stored in SQLite DBs via ``StoreSQLite`` interfaces.

Improved `loc_to_iloc()` methods to raise for missing keys in `Index` created where `loc_is_iloc`.


0.8.2
----------

Added ``Series.iloc_searchsorted()`` and ``Series.loc_searchsorted()``.

Interfaces of ``Frame.to_delimited()``, ``Frame.to_csv()``, ``Frame.to_tsv()``, and ``Frame.to_clipboard()`` are extended with parameters for control of quoting and escaping delimiters and other characters. The standard library's ``csv`` module is now used for writing.


0.8.1
----------

API change: ``Frame.from_element_loc_items()`` renamed ``Frame.from_element_items``; ``Frame.from_element_iloc_items`` is removed.

``Frame.assign`` now returns a ``FrameAssign`` instance with an ``apply`` method to permit using the assignment target, after function application, as the assignment value.

``Series.assign`` now returns a ``SeriesAssign`` instance with an ``apply`` method to permit using the assignment target, after function application, as the assignment value.

``IndexDatetime`` subclasses now properly assign ``name`` attrs from an `Index` given as an initializer.

``Series.items()`` now returns labels of ``IndexHierarchy`` as tuples instead of ``np.ndarray``.

Added ``Batch.apply_except`` and ``Batch.apply_items_except`` to permit ignore exceptions on function application to contained Frames.

Added ``Batch.unique()``.

``Batch`` now supports operations on ``Frame`` that return an ``np.ndarray``.

Added ``Quilt.from_items()`` and ``Quilt.from_frames()``.

``Bus.sort_index()`` and ``Bus.sort_values()`` now return a ``Bus`` instance.

Improvements to ``Bus.items()``, ``Bus.values`` for optimal ``Store`` reads when ``max_persist`` is None.

Implemented ``Bus.rename()`` to return a ``Bus`` instance.

Implemented ``Bus.drop[]`` to return a ``Bus`` instance.

Implemented ``Bus.reindex()``, ``Bus.relabel()``, ``Bus.relabel_flat()``, ``Bus.relabel_level_add()``, ``Bus.relabel_level_drop()``, ``Bus.rehierarch()``.

Implemented ``Bus.roll()``, ``Bus.shift()``.


0.8.0
----------

API change: ``Frame.sort_values()`` now has a ``label`` positional argument that replaces the former ``key`` positional argument.

API change: ``Frame.sort_values()`` now requires multiple labels to be provided as a list to permit distinguishing selection of single tuple labels.

API change: ``iter_labels.apply()`` on ``Index`` and ``IndexHierarchy`` now returns an np.ndarray rather than a ``Series``.

API change: ``iter_tuple`` and ``iter_tuple_items`` interfaces now require ``axis`` to be kwarg-only.

API change: ``iter_tuple``, ``iter_tuple_items`` methods now require an explicit ``tuple`` as constructor if fields are invalid NamedTuple attrs.

API change: ``iter_array``, ``iter_array_items``, ``iter_series``, and ``iter_series_items`` now require ``axis`` to be kwarg-only.

Added ``key`` argument for sort pre-processing to ``Frame.sort_values()``.


0.7.15
----------

Added ``key`` argument for sort pre-processing to ``Index.sort()``, ``IndexHierarchy.sort()``, ``Series.sort_index()``, ``Series.sort_values()``, ``Frame.sort_index()``, ``Frame.sort_columns``

Implemented support for multiprocessing Frame loading from ``StoreZip`` subclasses used by ``Bus``, ``Batch``, and ``Quilt``.

Added ``read_max_workers``, ``read_chunksize``, ``write_max_workers``, ``write_chunksize`` to ``StoreConfig``.

Added ``include_index_name``, ``include_columns_name`` parameters to ``Frame.to_arrow``

Added ``include_index_name``, ``include_columns_name`` parameters to ``Frame.to_parquet``

Added ``index_name_depth_level``, ``columns_name_depth_level`` parameters to ``Frame.from_arrow``

Added ``index_name_depth_level``, ``columns_name_depth_level`` parameters to ``Frame.from_parquet``

Fixed issue where non-optimal dtype would be used for new columns added in reindexing.


0.7.14
----------

Added immutable, hashable containers ``SeriesHE`` and ``FrameHE``.

Implemented ``read_many`` for all ``Store`` subclasses; ``Bus`` now uses these interfaces for significantly faster reads of multi-``Frame`` selections.

Improved handling of connection object given to ``Frame.from_sql``.

Improved type-preservation and performance when assigning ``Frame`` into ``Frame``.

Added ``Bus.from_items()`` constructor.


0.7.13
----------

Improved handling for using ``Frame.iter_group`` on zero-sized ``Frame``.

``Series`` can now be used as arguments to ``dtypes`` in ``Frame`` constructors.

Added ``via_dt.strptime`` and ``via_dt.strpdate`` for parsing strings to Python ``date``, ``datetime`` objects, respectively.


0.7.12
----------

``Bus`` indices are no longer required to be string typed.

``StoreConfig`` adds ``label_encoder``, ``label_decoder`` parameters for translating hashables to strings and strings to hashables when writing to / from ``Store`` formats.

``Frame.from_sql`` now supports a ``columns_select`` parameter.

``StoreConfig`` now supports a ``columns_select`` parameter; ``columns_select`` parameters from ``StoreConfig`` are now used in ``StoreZipParquet``, ``StoreSQLite``.

Extended ``via_str.startswith()`` and ``via_str.endswith()`` functions to support passing an iterable of strings to match.

Improved ``IndexHierarchy.loc_to_iloc`` to support Boolean array selections.


0.7.11
----------

Corrected issue in ``Frame.iter_series`` due to recent optimization.


0.7.10
----------

Improvements to ``Quilt`` extraction routines.


0.7.9
----------

Improved handling of invalid file paths given to constructors.

Improved implementations of ``Bus.items()``, ``Bus.values``, and ``Bus.equals()`` that deliver proper results when `max_persist` is active.

Implementation of ``Quilt``, a container that presents the contents of a ``Bus`` as either vertically or horizontally stacked ``Frame``.

Implemented ``__deepcopy__()`` on all containers.


0.7.8
----------

``Frame.iter_tuple_items()`` now exposes a ``constructor`` argument to control creation of axis containers.

Added ``Batch.apply_items``.

Added ``Frame.count``, ``Series.count``, ``Batch.count``.

Added ``Frame.sample``, ``Series.sample``, ``Index.sample``, ``IndexHierarchy.sample``, ``Batch.sample``.

Added ``Frame.via_T`` and ``IndexHierarchy.via_T`` accessors for opposite axis binary operator application of 1D operands.


0.7.7
----------

``IndexHierarchy.iter_label`` now defaults to iterating full depth labels.

``Batch.__repr__()`` is no longer a display that exhausts the stored generator.

``Frame.iter_tuple()`` now exposes a ``constructor`` argument to control creation of axis containers.


0.7.6
----------

Fixed issue in using ``Frame.extend`` with zero-length ``Frame``.


0.7.5
----------

Implemented ``Frame.isin`` on ``TypeBlocks``.

Implemented ``Frame.clip`` on ``TypeBlocks``.


0.7.4
----------

``Series.from_element`` now works correctly with tuples

``Batch`` element handling now avoids diagonal formations; ``Batch.apply()`` now handles elements correctly

``dtypes`` parameters can now be provided with ``dict_values`` instances.

``Frame.to_parquet``, ``Frame.to_arrow`` now convert ``np.datetime64`` units to nanosecond if not supported by PyArrow.


0.7.3
----------

``Bus`` now exposes ``max_persist`` parameter to define the maximum number of loaded ``Frame`` retained by the ``Bus``.

Added ``len()`` to ``via_str`` interfaces.

``Frame.iter_element`` now takes an ``axis`` argument to determine element order, where 0 is row major, 1 is column major.

Silenced ``NaturalNameWarning`` via ``tables`` in ``StoreHDF5``.

``StoreSQLite`` will now re-write, rather than update, a file path where an SQLite DB already exists.

Improved handling for iterating zero-sized ``Frame``.

Improved type detection when performing operations on ``Frame.iter_element`` iterators.

``Frame.shift()`` ``file_value`` parameter is now key-word argument only.

``Frame.roll()`` ``include_index``, ``include_columns`` is now key-word argument only.


0.7.2
----------

Extended application of binary equality operators to permit comparison with arrays of single elements.


0.7.1
----------

Refined application of binary equality operators to permit comparison with strings or elements that are not sequences.


0.7.0
----------

API change: ``__bool__`` of all containers now raises a ValueError.

API change: ``IndexHierarchy.iter_label`` now iterates over realized labels.

API change: ``IndexBase.union``, ``IndexBase.intersection`` no longer automatically unpack ``values`` from ``ContainerOperand`` subclasses.

API change: Container operands used with binary equality operators will raise if sizes are not equivalent.

API change: ``Frame.from_xlsx``, as well as ``StoreConfig`` now set ``trim_nadir`` to ``False`` by default.

API change: ``Series.relabel_add_level`` to ``Series.relabel_level_add``, ``Series.relabel_drop_level`` to ``Series.relabel_level_drop``, ``Frame.relabel_add_level`` to ``Frame.relabel_level_add``, ``Frame.relabel_drop_level`` to ``Frame.relabel_level_drop``, ``Index.add_level`` to ``Index.level_add``, ``IndexHierarchy.add_level`` to ``IndexHierarchy.level_add``, ``IndexHierarchy.drop_level`` to ``IndexHierarchy.level_drop``.


0.6.38
----------

``Frame.dtype`` interface now takes ``TDtypesSpecifier``, permitting setting ``dtype`` by mapping, iterable, or single value.

``dtypes`` can be given as a single ``TDtypeSpecifier`` for specifying ``dtype`` of all columns.

``Series`` of ``Frame`` can now be created without specifying ``dtype`` arguments.

``Frame`` now supports usage as a ``weakref``.

``Frame.from_parquet`` now raises when ``columns_select`` names columns not found in the file.


0.6.37
----------

Fixed issue in implementation of ``trim_nadir`` when reading XLSX files.


0.6.36
----------

Fixed issue in ``Frame.from_pandas`` when the columns have mixed types including integers.

Improved ``dtype`` preservation in zero-sized ``Series`` extraction from ``Frame``.

Added ``trim_nadir`` parameter to ``StoreConfig`` and ``Frame.from_xlsx``: permits removing all-None trailing rows and columns resulting from XLSX styles being applied to empty cells.


0.6.35
----------

Added a ``name`` parameter to ``Series.from_pandas`` and ``Frame.from_pandas``.

Added ``Frame.from_msgpack`` and ``Frame.to_msgpack``.

Refactored ``Bus`` and ``Batch`` to use the mixin class ``StoreClientMixin`` to share exporters and constructors.

Added ``StoreClientMixin.to_zip_parquet`` and ``StoreClientMixin.from_zip_parquet``.

Performance improvements to ``Frame.to_pandas`` when a ``Frame`` has unified ``TypeBlocks``.


0.6.34
----------

Updated all delimited text output formats to include a final line termination.

``Frame.from_overlay`` now takes optional ``index`` and ``columns`` arguments; ``Series.from_overlay`` now takes an optional ``index`` argument.

Improvements to union/intersection index formation in ``Frame.from_overlay`` and ``Series.from_overlay``.


0.6.33
----------

Performance improvements to ``Frame.pivot``.

``Frame.from_xlsx`` now exposes ``skip_header`` and ``skip_footer`` parameters.


0.6.32
----------

Added ``Frame.from_overlay``, ``Series.from_overlay`` constructors.

Added support for ``dataclass`` as records in ``Frame.from_records`` and ``Frame.from_records_items``.

Additional delegated ``Frame`` methods added to ``Batch``.


0.6.31
----------

Fixed issue when loading pickled containers where Boolean selection would not be properly identified.


0.6.30
----------

Added ``via_dt.fromisoformat()`` to all containers, supporting creation of date/datetime objects from ISO 8601 strings.

``Batch.to_frame`` now returns a `Frame` with an `IndexHierarchy` if all ``Batch`` operations retain one or more ``Frame``.

``Batch`` interface extended with core ``Frame`` methods.

Restored parameter name in ``Series.relabel`` to be ``index``.

Support for writing date, datetime, and np.datetime64 via `Frame.to_xlsx`.

Exposed ``store_filter`` parameter in ``Frame.from_xlsx``,``Frame.to_xlsx``.

Removed  ``format_index``, ``format_columns`` attributes from ``StoreConfig``.


0.6.29
----------

Fixed issue in ``Series.drop`` when the ``Series`` has an ``IndexHierarchy``.

Calling ``Frame.from_series`` with something other than a ``Series`` will now raise.

Calling ``Index.from_pandas``, ``Series.from_pandas``, and ``Frame.from_pandas`` now raise when given a non-Pandas object.

``StoreConfig`` given to ``Bus.to_xlsx``, ``Bus.to_sqlite``, and ``Bus.to_hdf5`` are now properly used.


0.6.28
----------

Introduced the ``Batch``, a lazy, parallel processor of groups of ``Frame``.

``Index`` and ``IndexHierarchy`` ``intersection()`` and ``union()`` now accept ``*args``, performing the set operation iteratively on all arguments.

Revised default aggregation function to ``Frame.pivot``.

Fixed issue in writing SQLite stores from ``Frame`` labelled with strings containing hyphens.

Added `include_index_name`, `include_columns_name` to ``Frame.to_delimited``.

Added `include_index_name`, `include_columns_name` to ``StoreConfig`` and ``Frame.to_xlsx`` interfaces.

Added `index_name_depth_level` and `columns_name_depth_level` to `Frame.from_delimited` and related methods.

Added `index_name_depth_level`, `columns_name_depth_level` to ``StoreConfig`` and ``Frame.from_xlsx`` interfaces.


0.6.27
----------

Improved implementation of ``Frame.pivot``.


0.6.26
----------

Removed class-level documentation injection, permitting better static analysis.

Corrected issue in appending tuples to an empty ``IndexGO``.


0.6.25
----------

Added ``Frame.from_clipboard()`` and ``Frame.to_clipboard()``.

Added ``Frame.pivot_stack()`` and ``Frame.pivot_unstack()``.


0.6.24
----------

Fixed flaw in difference operations on ``IndexDatetime`` subclasses of equivalent indices.


0.6.23
----------

``Frame.from_parquet`` and ``Frame.from_arrow`` now accept a ``dtypes`` argument.

All ``PathLike`` path objects now accepted wherever ``Path`` objects were previously.

Added ``fillna`` methods to ``Index``, ``IndexHierarchy``.

Added to ``StoreFilter`` the following parameters: ``value_format_float_positional``, ``value_format_float_scientific``, ``value_format_complex_positional``, ``value_format_complex_scientific``.

``Index`` and ``IndexHierarchy`` will reuse instances for set operations on equivalent indices.

Added ``IndexHierarchy.from_names`` constructor for creating zero-length ``IndexHierarchy``.

Refinements to ``IndexHierarchy`` to support grow-only mutation from zero length.


0.6.22
----------

Fixed flaw in ``IndexLevel`` for handling of zero-length levels.

Fixed flaw in ``TypeBlocks.iloc`` that caused an undesirable reference cycle.


0.6.21
----------

``IndexHierarchy`` set operations will now delegate ``Index`` types when they are equivalent between operands at corresponding depth levels.

``Frame.from_concat`` now delegates returned index input index name, type, ``IndexHierarchy`` contained types, if aligned on all indices per axis.

Fixed issue when calling ``relabel_add_level()`` from a ``FrameGO``.


0.6.20
----------

Extended functionality of ``HLoc`` selections in ``IndexHierarchy`` to properly handle selection lists, Boolean arrays, and nested ``ILoc`` selections.

Corrected issue in ``Frame.from_concat`` whereby, when given inputs with ``IndexHierarchy``, ``IndexHierarchy`` were not returned.


0.6.19
----------

Extended ``name`` propagation to applications of binary operators where an operand is a scalar.

Binary operators now work with ``Frame`` and same-shaped NumPy arrays.


0.6.18
----------

Extended support for step arguments in ``loc`` interfaces.

Implemented ``Frame.join_left``, ``Frame.join_right``, ``Frame.join_inner``, and ``Frame.join_outer``.

Implemented ``Frame.insert_before``, ``Frame.insert_after``.

Implemented ``Series.insert_before``, ``Series.insert_after``.

``IndexHierarchy.from_labels`` now enforces all labels to have the same depth.

Fixed issue where, when passing an array to ``Frame.from_records``, the ``name`` parameter is not passed to the constructor.


0.6.17
----------

Implemented ``equals()`` methods on all containers.

Added defensive check against assigning a Pandas Series to a FrameGO as an unlabeled iterator.

Added proper handling of types multiple-inherited from ``str`` (or any other type) and ``Enum``.

Implemented support for operator overloading of addition and multiplication on string dtypes.


0.6.16
----------

Implemented ``via_str`` and ``via_dt`` accesors on all ``ContainerOperand``.

When writing to XLSX, the shape of the ``Frame`` is validated to fit within the limits of XLSX sheets.


0.6.15
----------

Added support for ``round()`` on ``Frame``.

Added ``name`` parameter to all methods of ``IterNodeDelegate`` that produce a new container, including ``map_any()``, ``map_fill()``, ``map_all()``, ``apply()``, and ``apply_pool()``.

Support for ``include_index`` and ``include_columns`` in ``DisplayConfig`` instances and ``Display`` output.

Performance improvements to iterating tuples from ``IndexHierarchy``.

Performance improvements for ``IndexHierarchy`` transformations, including adding or dropping levels and rehierarch.


0.6.14
----------

Added explicit handling for binary operators applied to differently-sized ``IndexHierarchy``.


0.6.13
----------

Refined behavior of ``Frame.from_concat_items`` when given tuples as labels; implemented support for tuples as labels in ``IndexLevels.values_at_depth``.


0.6.12
----------

Refined behavior of ``names`` attribute on ``IndexBase`` to ensure that an appropriately sized iterable of labels is always returned.


0.6.11
----------

Added ``IndexHour`` and ``IndexHourGO`` indices.

Added ``IndexMicrosecond`` and ``IndexMicrosecondGO`` indices.

Added support for ``round()`` on ``Series``.

``Index.astype`` now returns specialized ``datetime64`` ``Index`` objects when given an appropriate dtype.

``IndexHierarchy.astype`` now produces an ``IndexHierarchy`` with specialized ``datetime64`` ``Index`` objects when given an appropriate dtype.

Added ``IndexLevels.dtypes_at_depth()`` and ``IndexLevels.dtype_per_depth()`` to capture resolved dtypes per depth.

Added ``IndexLevels.values_at_depth()`` to capture resolved typed arrays per depth.

Updated ``IndexHierarchy.display()`` to display proper types per depth.

Refactored ``IndexLevel`` to lazily cache depth and length attributes.

Refactored ``IndexHierarchy`` to store a ``TypeBlocks`` instance instead of 2D array, permitting reuse of ``TypeBlocks`` functionality, columnar type preservation, and immutable array reuse.

Fixed flaw in ``IndexHierarchy.label_widths_at_depth``.

Fixed flaw in ``Frame.from_records`` and related routines whereby a ``NamedTuple`` in an iterable of length 1 was converted to a single-row, two-dimensional array.

Fixed flaw in ``Frame`` function application on iterators for some ``Index`` type configurations.

API documentation now shows full signatures for all functions.


0.6.10
----------

Improvements to ``interface`` display, including in inclusion of function arguments and new "Assignment" category; improvements to API documentation.

Fixed issue in not handling mismatched size between index and values on ``Series`` initialization.

Fixed issue creating a datetime64 ``Index`` from another datetime64 ``Index`` when their dtypes differ.

Fixed an issue when passing an immutable ``Index`` as ``columns`` in ``FrameGO.reindex``.


0.6.9
----------

``Series`` default constructor now efficiently handles ``Series`` given as ``values``.

``Frame`` default constructor now efficiently handles ``Frame`` given as ``data``.

``AutoMap`` now serves as the core mapping structure for all ``Index`` object, offering better performance, immutability, and internal uniqueness checks.


0.6.8
----------

Fixed issue in using ``relabel()`` on columns in ``FrameGO``.

Fixed issue in using ``Frame.drop`` with ``IndexHierarchy`` on either axis.

Unified ``to_frame`` and ``to_frame_go`` interfaces on ``Frame``, ``FrameGO``, and ``IndexHierarchy``.

Enabled ``include_index``, ``include_columns`` parameters for ``Frame.to_parquet``.

Added ``columns_select`` parameter to ``Frame.from_parquet``.

Updated requirements: pyarrow==0.16.0

Refined ``Frame.from_arrow`` usage of ChunkedArray, disabling ``date_as_object``, enabling ``self_destruct``, and improving handling of NumPy array extraction.

Added ``STATIC`` attribute to ``ContainerBase`` and all subclasses.


0.6.7
----------

Fixed issue in assigning a column to a ``FrameGO`` from a generator that raises an exception.


0.6.6
----------

Added ``difference`` method to all ``Index`` subclasses.

Added ``index_constructor`` and ``columns_constructor`` parameters to ``Frame.from_pandas``; ``index_constructor`` added to ``Series.from_pandas``.


0.6.5
----------

Refined ``IndexBase.from_pandas``.


0.6.4
----------

Fixed issue introduced into ``Frame.iter_group`` and ``Frame.iter_group_items`` when selecting a single column with an object dytpe.

Fixed mapping lookups to use single-argument tuples in ``map_any_iter_items`` and ``map_fill_iter_items`` and related methods.


0.6.3
----------

Improvements to ``any`` and ``all`` methods on all containers when using ``skipna=True`` and NAs are presernt; now, a ``TypeError`` will now be raised when NAs are found and ``skipna=False``.

When converting from Pandas 1.0 extension dtypes, proper NumPy types are used if no ``pd.NA`` are present; if ``pd.NA`` are present, they are replaced with ``np.nan`` in the resulting object array.


0.6.2
----------

``Frame.sort_values`` now accepts multiple labels given as any iterable.

``loc`` selection on ``Series`` or ``Frame`` with ``IndexAutoFactory``-style indices now treat the slice stop as inclusive.

Removed creation of internal mapping object for ``IndexAutoFactory`` indices, or where ``Index`` are created where ``loc_is_iloc``.

Improved induction of dtype for labels array stored in ``Index``.


0.6.1
----------

The ``bloc`` and ``assign.bloc`` selectors on ``Frame`` now use ``[]`` instead of ``()``, aligning the interface with other selectors.

Added ``IndexNanosecond`` and ``IndexNanosecondGO`` indices.

All ``iter_*`` interfaces now explictly define arguments.

``Frame.fillna()`` and ``Series.fillna()`` now accept ``Frame`` and ``Series``, respectively, as arguments.

``Series.sort_index``, ``Series.sort_values``, ``Frame.sort_index``, ``Frame.sort_columns``, and ``Frame.sort_values`` now retain index/columns name after sorting.

Renamed ``Series.iter_group_index()``, ``Series.iter_group_index_items()``, ``Frame.iter_group_index()``, ``Frame.iter_group_index_items()`` to ``Series.iter_group_labels()``, ``Series.iter_group_labels_items()``, ``Frame.iter_group_labels()``, ``Frame.iter_group_labels_items()``

Fixed issue in ``Frame`` display where, when at or one less than the count of ``display_rows``, would display different numbers of rows for the ``Index`` and the body of the ``Frame``.

Zero-sized ``Frame`` now return zero-sized ``Series`` from selection where possible.


0.6.0
----------

Removed deprecated ``Frame`` and ``Series`` non-specialized constructor usage; removed support for providing mapping types to ``apply``.

Improved support for using tuples in ``Frame.__getitem__`` and ``FrameGO.__setitem__`` with ``IndexHierarchy`` and ``Index`` with tuple labels.


0.5.13
----------

Made ``Frame.clip``, ``Frame.duplicated``, ``Frame.drop_duplicated`` key-word argument only. Made ``Series.clip``, ``Series.duplicated``, ``Series.drop_duplicated`` key-word argument only.

``Frame.iter_series`` now sets the ``name`` attribute of the Series from the appropriate index.

Added ``Index.head()``, ``Index.tail()``, ``IndexHierarchy.head()``, ``IndexHierarchy.tail()``.

``Frame.from_records`` and related routines now do full type induction per column; all type induction on untyped iterables now examines all values.


0.5.12
----------

All ``Index`` subclasses now use ``PositionsAllocator`` to share immutable positions arrays, increasing ``Index`` performance.

Fixed issue in using ``FrameGO.relabel`` with a non grow-only ``IndexBase``.

``IndexHiearchy.from_labels`` now accepts a ``reorder_for_hierarchy`` Boolean option to reorder labels for hierarchical formation.

``FrameGO.from_xlsx``, ``FrameGO.from_hdf5``, ``FrameGO.from_sqlite`` now return the ``FrameGO`` instances. Updated all ``Store.read`` methods to accept a ``containter_type`` arguement.

Added ``consolidate_blocks`` parameter to ``StoreConfig``.

Added ``consolidate_blocks`` parameter to ``Frame.from_xlsx``, ``Frame.from_hdf5``, ``Frame.from_sqlite``, ``Frame.from_pandas``.

Implemented ``IndexYearGO``, ``IndexYearMonthGO``, ``IndexDateGO``, ``IndexMinuteGO``, ``IndexSecondGO``, ``IndexMillisecondGO`` grow-only, derived classes of `np.datetime64` indices.

Added ``Frame`` constructors: ``Frame.from_series``, ``Frame.from_element``, ``Frame.from_elements``. Deprecated creating ``Frame`` from an untyped iterable or element.

Added ``Series`` constructors: ``Series.from_element``. Deprecated creating ``Series`` from an element with the default intializer.

Added `index_constructor`, `columns_constructor` arguement to `Frame.from_items`, `Frame.from_dict`.

NP-style methods on ``Series`` and ``Frame`` no longer accept arbitrary keywork arguments.

Removed ``keys()`` and ``items()`` methods from ``Index`` and ``IndexHierarch``; default iterators from ``IndexHierarchy`` now iterate tuples instead of arrays.

Added to ``IterNodeDelegate`` the following methods for applying mapping types to iterators: ``map_all``, ``map_any``, and ``map_fill``. Generator versions are also made available: ``map_all_iter``, ``map_all_iter_items``, ``map_any_iter``, ``map_any_iter_items``, ``map_fill_iter``, ``map_fill_iter_items``.


0.5.11
----------

Fixed issue in ``Frame.assign`` when assigning iterables into a single column.


0.5.10
----------

Improvements to ``Frame.assign`` to handle unordered column selectors and preserve columnar types not affected by assignment.

Restored application of default column and index formattng in ``StoreXLSX``.


0.5.9
----------

Fixed issue in ``__slots__`` usage of derived Containers.

Implemented ``StoreConfig`` and ``StoreConfigMap`` classes, and updated all ``Store`` and ``Bus`` interfaces to use them.

Implemented tracking of Store file modification times, and implemented raising exceptions for any unexpected file modifications.

Improved handling of reading XLSX files with trailing all-empty rows resulting from style formatting across empty data.

Improved HDF5 reading so as to reduce memory overhead.


0.5.8
----------

Fixed issue in ``Frame.sort_values()`` when ``axis=0`` and underlying block structure is homogenous.

Improved performance of ``Frame.iter_group`` and related methods.

Fixed issue raised when calling built-in ``help()`` on SF containers.

Improved passing of index ``names`` in ``IndexHierarchy.to_pandas``.

Improved propagation of ``name`` in methods of ``Index`` and ``IndexHierarchy``.


0.5.7
----------

``StoreFilter`` added to the public namespace.

``names`` argument added to ``Frame.unset_index``.

Improved handling of ``ILoc`` usage within ``loc`` calls.

Improved input and output from/to XLSX.


0.5.6
----------

``Frame.from_concat``, ``Series.from_concat`` now accept empty iterables.

``Frame.iter_group.apply`` and related routines now handle producing a `Series` from a multi-column group selection.


0.5.5
----------

``Index`` objects based on ``np.datetime64`` now accept Python ``datetime.date`` objects in ``loc`` expressions.

Fixed index formation when using ``apply`` on ``Frame.iter_group`` and ``Frame.iter_group_items`` (and related interfaces) when the ``Frame`` has an ``IndexHierarchy``.

Fixed issue in a ``Frame.to_frame_go()`` not creating a fully decoupled ``Index`` for columns in the returned ``Frame``.

0.5.4
----------

``Index`` objects based on ``np.datetime64`` now return empty Series when a partial ``loc`` selection does not match any values found in the ``Index``.


0.5.3
----------

``Frame.set_index_hiearchy`` passes on ``name`` to returned ``Frame``.

``Index`` objects based on ``np.datetime64`` now accept Python ``datetime.datetime`` objects in ``loc`` expressions.

Exposed ``interface`` attribute on ``ContainerBase`` subclasses.


0.5.2
----------

Refinements to ``Series.isin()``, ``Frame.isin()``, ``Index.isin()``, and ``IndexHierarchy.isin()`` to better identify cases of unique elements.

Added ``IndexMinute`` datetime index subclass.

0.5.1
----------

Implemented handling in ``Frame.from_delimited`` for column-only files.

``Frame.iter_tuple`` and ``Frame.iter_tuple_items`` will return ``tuple`` instead of ``NamedTuple`` if fields are not valid identifiers.

``Frame.from_records`` now supports empty records if ``columns`` is provided.

``Frame.from_concat`` now implements better type preservation in vertical concatenation of arrays.


0.5.0
-----------

Introduced the ``Bus``, a ``Series``-like container of mulitple ``Frame``, supporting lazily reading from and writing to XLSX, SQLite, and HDF5 data stores, as well as zipped pickles and delimited files.

Added ``interface`` attribute to all containers, providing a hierarchical presentation of all interfaces.

Added ``display_tall()`` and ``display_wide()`` convenience methods to all containers.

Added ``label_widths_at_depth()`` on ``Index`` and ``IndexHierarchy``.

Added ``Series.from_concat_items()`` and ``Frame.from_concat_items()``.

Added ``Frame.to_xarray()``.

Added ``Frame.to_xlsx()``, ``Frame.from_xlsx()``.

Added ``Frame.to_sqlite()``, ``Frame.from_sqlite()``.

Added ``Frame.to_hdf5()``, ``Frame.from_hdf5()``.

Added ``Frame.to_rst()``.

Added ``Frame.to_markdown()``.

Added ``Frame.to_latex()``.

The interface of ``Frame.from_delimited`` (as well as ``Frame.from_csv`` and ``Frame.from_tsv``) has been updated to conform to the common usage of ``index_depth`` and ``columns_depth``. IndexHierarchy is now supported when ``index_depth`` or ``columns_depth`` is greater than one. The former parameter ``index_column`` is renamed ``index_column_first``.

Added ``IndexHierarchy.from_index_items`` and ``IndexHierarchy.from_labels_delimited``.

Added ``IndexBase.names`` attribute to provide normalized names equal in length to depth.

The ``DisplayConfig`` parameter ``type_show`` now, if ``False``, hides, native class types used as headers. This is the default display for all specialized string output via ``Frame.to_html``, ``Frame.to_rst``, ``Frame.to_markdown``, ``Frame.to_latex``, as well as Jupyter display methods.

Added ``Frame.unset_index()``.

Added ``Frame.pivot()``.

Added ``Frame.iter_window``, ``Frame.iter_window_items``, ``Frame.iter_window_array``, ``Frame.iter_window_array_items``.

Added ``Series.iter_window``, ``Series.iter_window_items``, ``Series.iter_window_array``, ``Series.iter_window_array_items``.

Added ``Frame.bloc`` and ``Frmae.assign.bloc``

Added ``IndexHierarchy.rehierarch``, ``Series.rehierarch``, and ``Frame.rehierarch``.

Defined ``__bool__`` for all containers, where the result is determined based on if the underlying NumPy array has ``size`` greater than zero.

Improved ``Frame.to_pandas()`` to preserve columnar types.

``Frame.set_index_hierarchy`` now accepts a ``reorder_for_hierarchy`` argument, reordering the rows to support hierarchability.

Added ``Frame.from_dict_records`` and ``Frame.from_dict_records_items``; when given records, the union of all keys is used to derive columns.


0.4.3
-----------

Fixed issues in ``FrameGO`` setitem and using binary operators between ``Frame`` and ``FrameGO``.

0.4.2
-----------

Corrected flaw in axis 1 statistical operations with ``Frame`` constructed from mixed sized ``TypeBlocks``.

Added ``Series.loc_min``, ``Series.loc_max``, ``Series.iloc_min``, ``Series.iloc_max``.

Added ``Frame.loc_min``, ``Frame.loc_max``, ``Frame.iloc_min``, ``Frame.iloc_max``,


0.4.1
-----------

``iter_element().apply`` now properly preserves index and column types.

Using ``Frame.from_records`` with an empty iterable or iterator will deliver a ``ErrorInitFrame``.

Matrix multiplication implemented for ``Index``, ``Series``, and ``Frame``.

Added ``Frame.from_records_items`` constructor.

Improved dtype selection in ``FrameGO`` set item and related functions.

``IndexHierarchy.from_labels`` now accepts an ``index_constructors`` argument.

``Frame.set_index_hierarchy`` now accepts an ``index_constructors`` argument.

``IndexHierarhcy.from_product() now attempts to use ``name`` of provided indicies for the ``IndexHierarchy`` name, when all names are non-None.

Added ``IndexHierarchy.dtypes`` and ``IndexHierarchy.index_types``, returning ``Series`` indexed by ``name`` when possible.


0.4.0
-----------

Improved handling for special cases ``Series`` initialization, including initialization from iterables of lists.

The ``Series`` initializer no longer accepts dictionaries; ``Series.from_dict`` is added for explicit creation from mappings.

``IndexAutoFactory`` suport removed from ``Series.reindex`` and ``Frame.reindex`` and added to ``Series.relabel`` and ``Frame.relabel``.

The following ``Series`` and ``Frame`` methods are renamed: ``reindex_flat``, ``reindex_add_level``, and ``reindex_drop_level`` are now ``relabel_flat``, ``relabel_add_level``, and ``relabel_level_drop``.

Implemented ``Frame.from_sql`` constructor.


0.3.9
-----------

``IndexAutoFactory`` introduced to consolidate creation of auto-incremented integer indices, and provide a single token to force auto-incremented integer indices in other contexts where ``index`` arguments are taken.

``IndexAutoFactory`` support implemented for the ``index`` argument in ``Series.from_concat`` and ``Series.reindex``.

``IndexAutoFactory`` support implemented for the ``index`` and ``columns`` argument in ``Frame.from_concat`` and ``Frame.reindex``.

Added new ``DisplyaConfig`` parameters to format floating-point values: ``value_format_float_positional``, ``value_format_float_scientific``,  ``value_format_complex_positional``, ``value_format_complex_scientific``,

Set default ``value_format_float_scientific`` and ``value_format_complex_scientific`` to avoid truncation of scientific notation in output displays.


0.3.8
-----------

All duplicate-handling functions now support heterogenously typed object arrays with unsortable (but hashable) types.

Operations on all indices now preserve order when indices are equal.

Functions with the ``skipna`` argument now properly skip ``None`` in ``Frames`` with built with object arrays.

``Frame.to_csv`` now uses the argument name `delimiter` instead of `sep`, aligning with the usage in ``Frame.from_csv``.


0.3.7
------------

Completed implementation of ``Frame.fillna_forward``, ``Frame.fillna_backward``, ``Frame.fillna_leading``, ``Frame.fillna_trailing``.

Fixed issue exposed in FrameGO.sort_values() due to NumPy integers being used for selection.

``IndexHierarchy.sort()``, ``IndexHierarchy.isin()``, ``IndexHierarchy.roll()`` now implemented.

``Series.sort_index()`` now properly propagates ``IndexBase`` subclasses.

``Frame.sort_index()`` and ``Frame.sort_columns()`` now properly propagate ``IndexBase`` subclasses.

All containers now derive from ``ContainerOperand``, simplyfying inheritance and ``ContainerOperandMeta`` application.

``Index`` objects based on ``np.datetime64`` now accept ``np.datetime64`` objects in ``loc`` expressions.

All construction from Python iterables now better handle array creation from diverse Python objects.


0.3.6
------------

``Frame.to_frame_go`` now properly handles ``IndexHierarchy`` columns.

Improved creation of ``IndexHierarchy`` from other ``IndexHierarchy`` or ``IndexHierarchyGO``.

``Frame`` initializer now exposes ``index_constructor`` and ``columns_constructor`` arguments.

``Frame.from_records`` now efficiently uses ``dict_view`` objects containing row records.

``Frame`` now supports shapes of all zero and non-zero combinations of index and column lengths; ``Frame`` construction will raise an exception if attempting to set a value in an unfillable Frame shape.

``Frame``, ``Series``, ``Index``, and ``IndexHierarchy`` all have improved implementations of ``cumprod`` and ``cumsum`` methods.


0.3.5
------------

Improved type handling of ``np.datetime64`` typed columns in ``Frame``.

Added ``median`` method to all ``MetaOperatorDelegate`` classes, inlcuding ``Series``, ``Index``, and ``Frame``.

``Frame`` and ``Series`` sort methods now propagate ``name`` attributes.

``Index.from_pandas()`` now correctly collects ``name`` / ``names`` attributes from Pandas indexes.

Implemented ``Series.fillna_forward``, ``Series.fillna_backward``, ``Series.fillna_leading``, ``Series.fillna_trailing``.

Fixed flaw in dropping columns from a ``Frame`` (via ``Frame.set_index`` or the ``Frame.drop`` interface), whereby sometimes (depending on ``TypeBlocks`` structure) the drop would not be executed.

``Index`` objects based on ``np.datetime64`` now limit ``__init__`` arguments only to those relevant for those derived classes.

``Index`` objects based on ``np.datetime64`` now support transformations from both ``datetime.timedelta`` as well as ``np.timedelta64``.

Index objects based on ``np.datetime64`` now support selection with slices with ``np.datetime64`` units different than those used in the ``Index``.


0.3.4
-------------

Added ``dtypes`` argument to all relevant ``Frame`` constructors; ``dtypes`` can now be specified with a dictionary.

Deprecated instantiating a ``Frame`` from ``dict``; added ``Frame.from_dict`` for explicit ``Frame`` creation from a ``dict``.


0.3.3
--------------

Improvements to all ``datetime64`` based indicies: direct creation from labels now properly parses values into ``datetime64``, and ``loc``-style lookups now handle partial matches on lower-resolution datetimes. Added ``IndexSecond`` and ``IndexMillisecond`` Index classes.

Index can now be constructed directly from an ``IndexHierarchy`` (resulting in an Index of tuples)

Improvements to application of ellipsis when normalizing width in ``Display`` string representations.

``Frame.values`` now always returns a 2D NumPy array.

``Series.iloc``, when a non-mulitple selection is given, now returns a single element, not a ``Series``.


0.3.2
-----------

``IndexHierarchy.level_drop()`` and related methods have been updated such that negative integers drop innermost levels, and postive integers drop outermost levels. This is an API breaking change.

Fixed missing handling for all-missing in ``Series.dropna``.

Improved ``loc`` and ``HLoc`` usage on Series with ``IndexHierarchy`` to insure a Series is returned when a multiple selection is used.

``IndexHierarchy.from_labels()`` now returns proper error message for invalid tree forms.


0.3.1
----------

Implemented Series.iter_group_index(), Series.iter_group_index_items(), Frame.iter_group_index(), Frame.iter_group_index_items() for producing iterators (and targets of function application) based on groupings of the index; particularly useful for IndexHierarhcy.

Implemented Series.from_concat; improved Frame.from_concat in concatenating indices with diverse types. Frame.from_concat() now accepts Series.

Added ``Index.iter_label()`` and ``IndexHierarchy.iter_label()``, for variable depth label iteration, particularly useful for IndexHierarchy.

Improved initializer behavior of IndexDate, IndexYearMonth, IndexYear to apply expected dtype when creating arrays from non-array initializers, allowing conversion of string date representations to proper date types.

Added ``Index.to_pandas`` and specialized methods on ``IndexDate`` and derived classes. Added ``IndexHierarchy.to_pandas``.

Added support for ``Series`` as an argument to ``FrameGO.extend()``.

Added ``Series.to_frame()`` and ``Series.to_frame_go()``.

The ``name`` attribute is now implemented for all containers; all constructors now take a ``name`` argument, and a ``rename`` method is available. Extracting columns, rows, and setting indices on ``Frame`` all propagate name attributes appropriately.

The default ``Series`` display has been updated to show the "<Series>" label above the index, consistent with the presentation of ``Frame``.

The ``Frame.from_records()`` method has been extended to support explicitly passing dtypes per column, which permits avoiding type discovery through observing the first record or relying on NumPy's type discovery in array creation.

The ``Frame.from_concat()`` constructor now handles hierarchical indices correctly.


0.3.0
---------

The ``Index.keys()`` method now returns the underlying KeysView from the Index's dictionary.

All primary containers (i.e., Series, Frame, and Index) now display HTML tables in Jupyter Notebooks. This is implemented via the ``_repr_html_()`` methods.

All primary containers now feature a ``to_html()`` method.

All primary containers now feature a ``to_html_datatables()`` method, which authors a complete HTML file with DataTables/JavaScript-powered table viewing, sorting, and searching.

StaticFrame's display infrastructure now permits individually coloring types by category, as well as different display formats for supporting HTML output.

StaticFrame's display infrastructure now shows hierarchical indices, used for either indices or columns, in the same display grid used for other display components.

The ``DisplayConfig`` class has been expanded to permit definition of colors, specified in hexadecimal integers or string codes, for all type categories, as well as independent settings for type delimiters, and a new setting for ``display_format``.

The following ``DisplayFormats`` have been created and implemented: ``terminal``, ``html_datatables``, ``html_table``, and ``html_pre``.

