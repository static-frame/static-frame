

What is New in Static Frame
===============================


0.3.7-dev
------------

``IndexHierarchy.sort()``, ``IndexHierarchy.isin()``, ``IndexHierarchy.roll()`` now implemented.

``Series.sort_index()`` now properly propagates ``IndexBase`` subclasses.

``Frame.sort_index()`` and ``Frame.sort_columns()`` now properly propagate ``IndexBase`` subclasses.

All containers now derive from ``ContainerBase``, simplyfying inheritance and ``ContainerMeta`` application.

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

