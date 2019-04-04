

What New in Static Frame
===============================


0.3.1
----------

The ``Frame.from_records()`` method has been extended to support explicitly passing dtypes per column, which permits avoiding type discovery through observing the first record or relying on NumPy's type discovery in array creation.



0.3.0
---------

All primary containers (i.e., Series, Frame, and Index) now display HTML tables in Jupyter Notebooks. This is implemented via the ``_repr_html_()`` methods.

All primary containers now feature a ``to_html()`` method.

All primary containers now feature a ``to_html_datatables()`` method, which authors a complete HTML file with DataTables/JavaScript-powered table viewing, sorting, and searching.

StaticFrame's display infrastructure now permits individually coloring types by category, as well as different display formats for supporting HTML output.

StaticFrame's display infrastructure now shows hierarchical indices, used for either indices or columns, in the same display grid used for other display components.

The ``DisplayConfig`` class has been expanded to permit definition of colors, specified in hexadecimal integers or string codes, for all type categories, as well as independent settings for type delimiters, and a new setting for ``display_format``.

The following ``DisplayFormats`` have been created nad implemented: ``terminal``, ``html_datatables``, ``html_table``, and ``html_pre``.

