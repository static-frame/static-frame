

What is New in Static Frame
===============================


0.3.1-dev
----------

Implemented Series.iter_group_index() and Series.iter_group_index_items(), for producing iterators (and targets of function application) based on groupings of the index; particularly useful for IndexHierarhcy.

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

