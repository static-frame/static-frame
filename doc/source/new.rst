

What is New in Static Frame
===============================


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

The ``DisplayConfig`` parameter ``type_show`` now, if False, hides, native class types used as headers. This is the default display for all specialized string output via ``Frame.to_html``, ``Frame.to_rst``, ``Frame.to_markdown``, ``Frame.to_latex``, as well as Jupyter display methods.

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

The following ``Series`` and ``Frame`` methods are renamed: ``reindex_flat``, ``reindex_add_level``, and ``reindex_drop_level`` are now ``relabel_flat``, ``relabel_add_level``, and ``relabel_drop_level``.

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

``IndexHierarchy.drop_level()`` and related methods have been updated such that negative integers drop innermost levels, and postive integers drop outermost levels. This is an API breaking change.

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

