'''
Storage for common doc strings and templates shared in non-related classes and methods.
'''


import typing as tp

from static_frame.core.util import AnyCallable

#-------------------------------------------------------------------------------
# common strings
#NOTE: for kwargs, it is sometimes useful to only define the string, not the variable name, as in some contexts different variable names are use same conceptual entity.
INDEX_INITIALIZER = '''An iterable of unique, hashable values, or another ``Index`` or ``IndexHierarchy``, to be used as the labels of the index.'''

LOC_SELECTOR = '''A loc selector, either a label, a list of labels, a slice of labels, or a Boolean array.'''

ILOC_SELECTOR = '''An iloc selector, either an index, a list of indices, a slice of indices, or a Boolean array.'''


#-------------------------------------------------------------------------------
# full parameter definitions

AXIS = '''axis: Integer specifying axis, where 0 is rows and 1 is columns. Axis 0 is set by default.'''

CHUNKSIZE = 'chunksize: Units of work per executor, as passed to the Thread- or ProcessPoolExecutor.'

COLUMNS_CONSTRUCTOR = "columns_constructor: Optional class or constructor function to create the :obj:`Index` applied to the columns."

CONSOLIDATE_BLOCKS = 'consolidate_blocks: Optionally consolidate adjacent same-typed columns into contiguous arrays.'

DEEPCOPY_FROM_BUS = 'deepcopy_from_bus: Boolean to determine if containers are deep-copied from the contained :obj:`Bus` during extraction. Set to ``True`` to avoid holding references from the :obj:`Bus`.'

DTYPE_SPECIFIER = '''dtype: A value suitable for specyfying a NumPy dtype, such as a Python type (float), NumPy array protocol strings ('f8'), or a dtype instance.'''

DTYPES = "dtypes: Optionally provide an iterable of dtypes, equal in length to the length of each row, or a mapping by column name. If a dtype is given as None, NumPy's default type determination will be used."

FP = 'fp: A string file path or :obj:`Path` instance.'

INDEX_CONSTRUCTOR = "index_constructor: Optional class or constructor function to create the :obj:`Index` applied to the rows."

MAX_PERSIST = 'max_persist: When loading :obj:`Frame` from a :obj:`Store`, optionally define the maximum number of :obj:`Frame` to remain in the :obj:`Bus`, regardless of the size of the :obj:`Bus`. If more than ``max_persist`` number of :obj:`Frame` are loaded, least-recently loaded :obj:`Frame` will be replaced by ``FrameDeferred``. A ``max_persist`` of 1, for example, permits reading one :obj:`Frame` at a time without ever holding in memory more than 1 :obj:`Frame`.'

MAX_WORKERS = 'max_workers: Number of parallel executors, as passed to the Thread- or ProcessPoolExecutor; ``None`` defaults to the max number of machine processes.'

NAME = 'name: A hashable object to label the container.'

OWN_COLUMNS = '''own_columns: Flag the passed columns as ownable by this :obj:`static_frame.{class_name}`. Primarily used by internal clients.'''

OWN_DATA = '''own_data: Flag the data values as ownable by this :obj:`static_frame.{class_name}`. Primarily used by internal clients.'''

OWN_INDEX = '''own_index: Flag the passed index as ownable by this :obj:`static_frame.{class_name}`. Primarily used by internal clients.'''

RETAIN_LABELS = 'retain_labels: Boolean to determine if, along the axis of virtual concatentation, if component :obj:`Frame` labels should be used to form the outer depth of an :obj:`IndexHierarchy`. This is required to be ``True`` if component :obj:`Frame` labels are not globally unique along the axis of concatenation.'

STORE = 'store: A :obj:`Store` subclass.'

STORE_CONFIG_MAP = 'config: A :obj:`StoreConfig`, or a mapping of label ot :obj:`StoreConfig`'

USE_THREADS = 'use_threads: Use the ThreadPoolExecutor instead of the ProcessPoolExecutor.'

class DOC_TEMPLATE:

    #---------------------------------------------------------------------------
    # complete or partial function doc strings

    to_html = '''
    Return an HTML table representation of this :obj:`{class_name}` using standard TABLE, TR, and TD tags. This is not a complete HTML page.

    Args:
        config: Optional :obj:`DisplayConfig` instance.

    Returns:
        :obj:`str`
    '''

    to_html_datatables = '''
    Return a complete HTML representation of this :obj:`{class_name}` using the DataTables JS library for table naviagation and search. The page links to CDNs for JS resources, and thus will not fully render without an internet connection.

    Args:
        fp: optional file path to write; if not provided, a temporary file will be created. Note: the caller is responsible for deleting this file.
        show: if True, the file will be opened with a webbrowser.
        config: Optional :obj:`DisplayConfig` instance.

    Returns:
        :obj:`str`, absolute file path to the file written.
    '''

    name = '''
    A hashable label attached to this container.

    Returns:
        :obj:`Hashable`
    '''

    interface = '''
    A :obj:`Frame` documenting the interface of this class.
    '''

    label_widths_at_depth = '''
    A generator of pairs, where each pair is the label and the count of that label found at the depth specified by  ``depth_level``.

    Args:
        depth_level: a depth level, starting from zero.
    '''

    clip = '''Apply a clip operation to this :obj:`{class_name}`. Note that clip operations can be applied to object types, but cannot be applied to non-numerical objects (e.g., strings, None)'''


    values_2d = '''
    A 2D NumPy array of all values in the :obj:`{class_name}`. As this is a single array, heterogenous columnar types might be coerced to a compatible type.
    '''

    values_1d = '''
    A 1D NumPy array of the values in the :obj:`{class_name}`. This array will have the same dtype as the container.
    '''


    #---------------------------------------------------------------------------
    # dict entries

    apply = dict(
            doc='Apply a function to each value.',
            func='func: A function that takes a value.',
            name=NAME,
            dtype=DTYPE_SPECIFIER,
            max_workers=MAX_WORKERS,
            chunksize=CHUNKSIZE,
            use_threads=USE_THREADS,
            )

    argminmax = dict(
            skipna='skipna: if True, NaN or None values will be ignored; if False, a found NaN will propagate.',
            axis='axis: Axis upon which to evaluate contiguous missing values, where 0 is vertically (between row values) and 1 is horizontally (between column values).'
    )

    astype = dict(
            dtype=DTYPE_SPECIFIER
            )


    batch_constructor = dict(
            args = f'''
        Args:
            {FP}
            {STORE_CONFIG_MAP}
            {MAX_WORKERS}
            {CHUNKSIZE}
            {USE_THREADS}
            '''
            )

    batch_init = dict(
            args = f'''
        Args:
            {NAME}
            {STORE_CONFIG_MAP}
            {MAX_WORKERS}
            {CHUNKSIZE}
            {USE_THREADS}
            '''
            )

    bus_constructor = dict(
            args = f'''
        Args:
            {FP}
            {STORE_CONFIG_MAP}
            {MAX_PERSIST}
            '''
            )

    bus_init = dict(
            args = f'''
        Args:
            {STORE}
            {STORE_CONFIG_MAP}
            {MAX_PERSIST}
            '''
            )

    container_init = dict(
            index='''index: Optional index initializer. If provided in addition to data values, lengths must be compatible.''',
            columns='''columns: Optional column initializer. If provided in addition to data values, lengths must be compatible.
            ''',
            own_index=OWN_INDEX,
            own_data=OWN_DATA,
            own_columns=OWN_COLUMNS
            )

    constructor_frame = dict(
            dtypes=DTYPES,
            name=NAME,
            consolidate_blocks=CONSOLIDATE_BLOCKS
    )
    delimited = dict(
            doc='Given a file path or file-like object, write the :obj:`Frame` as delimited text.',
            fp='A file path, PathLib instance, or file-like object.',
            delimiter='delimiter: Character to be used for delimiterarating elements.',
            include_index='include_index: If True, the index will be written.',
            include_index_name='include_index_name: If including columns, populate the row above the index with the index ``name``. Cannot be True if ``include_columns_name`` is ``True``.',
            include_columns='include_columns: If ``True``, the columns will be written.',
            include_columns_name='include_columns_name: If including index, populate the column to the left of the columns with the columns ``name``. Cannot be True if ``include_index_name`` is True.',
            encoding='encoding: Encoding type to be used when opening the file.',
            line_terminator='line_terminator: The string used to terminate lines.',
            quote_char='quote_char: A one-character string used to quote fields containing special characters, such as the ``delimiter`` or ``quote_char``, or which contain new-line characters.',
            quote_double='quote_double: Controls how instances of quote_char appearing inside a field should themselves be quoted. When ``True``, the character is doubled. When ``False``, the ``escape_char`` is used as a prefix to the ``quote_char``. It defaults to True.',
            escape_char='escape_char: A one-character string used by the writer to escape the delimiter if quoting is set to QUOTE_NONE and the quotechar if quote_double is False.',
            quoting='quoting: Controls when quotes should be generated. It can take on any of the QUOTE_* constants from the standard library csv module.',
            store_filter='store_filter: A :obj:`StoreFilter` instance.',
            )
    duplicated = dict(
            exclude_first='exclude_first: Boolean to select if the first duplicated value is excluded.',
            exclude_last='exclude_last: Boolean to select if the last duplicated value is excluded.',
            axis=AXIS,
            )

    display = dict(
            doc='Return a :obj:`static_frame.Display`, capable of providing a string representation.',
            config='config: A :obj:`static_frame.DisplayConfig` instance. If not provided, the :obj:`static_frame.DisplayActive` will be used.'
            )

    equals = dict(
            doc='Return a :obj:`bool` from comparison to any other object.',
            compare_name="compare_name: Include equality of the container's name (and all composed containers) in the comparison.",
            compare_dtype="compare_dtype: Include equality of the container's dtype (and all composed containers) in the comparison.",
            compare_class="compare_class: Include equality of the container's class (and all composed containers) in the comparison.",
            skipna="skipna: If True, comparisons between missing values are equal.",
            )

    from_any = dict(
            fp=FP,
            index_depth='index_depth: integer specification of how many columns to use in forming the index. A value of 0 will select none; a value greater than 1 will create an :obj:`IndexHierarchy`.',
            columns_depth='columns_depth: integer specification of how many rows to use in forming the columns. A value of 0 will select none; a value greater than 1 will create an :obj:`IndexHierarchy`.',
            columns_select='columns_select: An optional iterable of column names to load.',
            consolidate_blocks=CONSOLIDATE_BLOCKS,
            name=NAME,
            dtypes=DTYPES,
            )


    from_pandas = dict(
            own_data='''own_data: If True, the underlying NumPy data array will be made immutable and used without a copy.''',
            own_index='''own_index: If True, the underlying NumPy index label array will be made immutable and used without a copy.''',
            own_columns='''own_columns: If True, the underlying NumPy column label array will be made immutable and used without a copy.''',
            columns_constructor=COLUMNS_CONSTRUCTOR,
            index_constructor=INDEX_CONSTRUCTOR,
            consolidate_blocks=CONSOLIDATE_BLOCKS,
            )


    fillna = dict(
            limit='limit: Set the maximum count of missing values (NaN or None) to be filled per contiguous region of missing vlaues. A value of 0 is equivalent to no limit.',
            value='value: Value to be used to replace missing values (NaN or None).',
            axis='axis: Axis upon which to evaluate contiguous missing values, where 0 is vertically (between row values) and 1 is horizontally (between column values).'
            )

    head = dict(
            doc='''Return a :obj:`{class_name}` consisting only of the top elements as specified by ``count``.
            ''',
            count='''count: Number of elements to be returned from the top of the :obj:`{class_name}`''',
            )

    index_init = dict(
            args = f'''
        Args:
            labels: {INDEX_INITIALIZER}
            {NAME}
            loc_is_iloc: Optimization when a contiguous integer index is provided as labels. Generally only set by internal clients.
            {DTYPE_SPECIFIER}'''
            )

    index_date_time_init = dict(
            args = f'''
        Args:
            labels: Iterable of hashable values to be used as the index labels. If strings, NumPy datetime conversions will be applied.
            {NAME}
            '''
            )

    insert = dict(
            key_before="key: Label before which the new container will be inserted.",
            key_after="key: Label after which the new container will be inserted.",
            container="container: Container to be inserted.",
            fill_value='fill_value: A value to be used to fill space after reindexing the new container.'
            )

    join = dict(
            left_depth_level="left_depth_level: Specify one or more left index depths to include in the join predicate.",
            left_columns="left_columns: Specify one or more left columns to include in the join predicate.",
            right_depth_level="right_depth_level: Specify one or more right index depths to include in the join predicate.",
            right_columns="right_columns: Specify one or more right columns to include in the join predicate.",
            left_template="left_template: Provide a format string for naming left columns in the joined result.",
            right_template="right_template: Provide a format string for naming right columns in the joined result.",
            fill_value='fill_value: A value to be used to fill space created in the join.',
            composite_index='composite_index: If True, an index of tuples will be returned, formed from the left index label and the right index label; if False, an index of matching labels, if unique, will be returned.',
            composite_index_fill_value='composite_index_fill_value: Value to be used when forming a composite index when a label is missing.'
            )

    mloc = dict(
            doc_int='The memory location, represented as an integer, of the underlying NumPy array.',
            doc_array='The memory locations, represented as an array of integers, of the underlying NumPy arrays.',
            )

    map_any = dict(
            doc='Apply a mapping; for values not in the mapping, the value is returned.',
            mapping='mapping: A mapping type, such as a dictionary or Series.',
            dtype=DTYPE_SPECIFIER,
            )

    map_fill = dict(
            doc = 'Apply a mapping; for values not in the mapping, the ``fill_value`` is returned.',
            mapping = 'mapping: A mapping type, such as a dictionary or Series.',
            fill_value = 'fill_value: Value to be returned if the values is not a key in the mapping.',
            dtype=DTYPE_SPECIFIER
            )

    map_all = dict(
            doc = 'Apply a mapping; for values not in the mapping, an Exception is raised.',
            mapping='mapping: A mapping type, such as a dictionary or Series.',
            dtype=DTYPE_SPECIFIER
            )


    quilt_constructor = dict(
            args = f'''
        Args:
            {FP}
            {STORE_CONFIG_MAP}
            axis: Integer specifying axis of virtual concatenation, where 0 is vertically (stacking rows) and 1 is horizontally (extending columns).
            {RETAIN_LABELS}
            {DEEPCOPY_FROM_BUS}
            {MAX_PERSIST}
            '''
            )

    quilt_init = dict(
            args = f'''
        Args:
            bus: :obj:`Bus` of :obj:`Frame` to be used for virtual concatenation.
            axis: Integer specifying axis of virtual concatenation, where 0 is vertically (stacking rows) and 1 is horizontally (extending columns).
            {RETAIN_LABELS}
            {DEEPCOPY_FROM_BUS}
            '''
            )

    reindex = dict(
            doc='''Return a new :obj:`{class_name}` with labels defined by the provided index. The size and ordering of the data is determined by the newly provided index, where data will continue to be aligned under labels found in both the new and the old index. Labels found only in the new index will be filled with ``fill_value``.
            ''',
            index_initializer=INDEX_INITIALIZER,
            fill_value='''fill_value: A value to be used to fill space created by a new index that has values not found in the previous index.''',
            own_index=OWN_INDEX,
            own_columns=OWN_COLUMNS
            )

    relabel = dict(
            doc ='''Return a new :obj:`{class_name}` with transformed labels on the index. The size and ordering of the data is never changed in a relabeling operation. The resulting index must be unique.
            ''',
            count='''A positive integer drops that many outer-most levels; a negative integer drops that many inner-most levels.''',
            level='''A hashable value to be used as a new root level, extending or creating an ``IndexHierarchy``''',
            relabel_input='''One of the following types, used to create a new ``Index`` with the same size as the previous index. (a) A mapping (as a dictionary or ``Series``), used to lookup and transform the labels in the previous index. Previous labels not found in the mapping will be reused. (b) A function, returning a hashable, that is applied to each label in the previous index. (c) The ``IndexAutoFactory`` type, to apply an auto-incremented integer index. (d) An index initializer, i.e., either an iterable of hashables or an ``Index`` instance.'''
            )

    relabel_flat = dict(
            doc='''Return a new :obj:`{class_name}`, where an ``IndexHierarchy`` (if defined) is replaced with a flat, one-dimension index of tuples.
            ''',
            )

    relabel_level_add = dict(
            doc='''Return a new :obj:`{class_name}`, adding a new root level to an existing ``IndexHierarchy``, or creating an ``IndexHierarchy`` if one is not yet defined.
            ''',
            level='''A hashable value to be used as a new root level, extending or creating an ``IndexHierarchy``''',
            )

    relabel_level_drop = dict(
            doc='''Return a new :obj:`{class_name}`, dropping one or more levels from a either the root or the leaves of an ``IndexHierarchy``. The resulting index must be unique.
            ''',
            count='''A positive integer drops that many outer-most (root) levels; a negative integer drops that many inner-most (leaf)levels.''',
            )
    sample = dict(
            doc='''Randomly (optionally made deterministic with a fixed seed) extract items from the container to return a subset of the container.''',
            count='''Number of elements to select.''',
            index='''Number of labels to select from the index.''',
            columns='''Number of labels to select from the columns.''',
            seed='''Initial state of random selection.''',
            )
    searchsorted = dict(
            doc='Given a sorted :obj:`Series`, return the {label_type} position(s) at which insertion in ``values`` would retain sort order.',
            values='values: a single value, or iterable of values.',
            side_left='side_left: If True, the index of the first suitable location found is given, else return the last such index. If matching an existing value, `side_left==True` will return that position, `side_left==Right` will return the next position (or the length).',
            fill_value='fill_value: A value to be used to fill the label beyond the last label.',
            )
    selector = dict(
            key_loc=LOC_SELECTOR,
            key_iloc=ILOC_SELECTOR,
            )
    sort = dict(
            ascending='If True, sort in ascending order; if False, sort in descending order.',
            kind='Name of the sort algorithm as passed to NumPy.',
            key='A function that is used to pre-process the selected columns or rows and derive new values to sort by.'
            )

    store_client_exporter = dict(
            args = f'''
        Args:
            {FP}
            {STORE_CONFIG_MAP}
            '''
            )

    tail = dict(
            doc='''Return a :obj:`{class_name}` consisting only of the bottom elements as specified by ``count``.
            ''',
            count='''count: Number of elements to be returned from the bottom of the :obj:`{class_name}`''',
            )


    ufunc_skipna = dict(
            args = '''
        Args:
            axis: Axis, defaulting to axis 0.
            skipna: Skip missing (NaN) values, defaulting to True.
            '''
            )
    window = dict(
            args = f'''
        Args:
            size: Elements per window, given as an integer greater than 0.
            {AXIS}
            step: Element shift per window, given as an integer greater than 0. Determines the step size between windows. A step of 1 shifts each window 1 element; a step equal to the ``size`` will result in non-overlapping windows.
            window_sized: if True, windows with fewer elements than ``size`` are skipped.
            window_func: Array processor of window values, executed before function application (if used): can be used for applying a weighting function to each window.
            window_valid: Function that, given an array window, returns True if the window is valid; invalid windows are skipped.
            label_shift: A shift, relative to the right-most element contained in the window, to derive the label to be paired with the window. For example, to label each window with the label found at the start of the window, ``label_shift`` can be set to one less than ``size``.
            start_shift: A shift to determine the first element where window collection begins.
            size_increment: A value to be added to ``size`` with each window after the first, so as to, in combination with setting ``step`` to 0, permit iterating over expanding windows.
            '''
            )

# NOTE: F here should replace AnyCallable below
F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])

def doc_inject(*,
        selector: tp.Optional[str] = None,
        **kwargs: object
        ) -> tp.Callable[[AnyCallable], AnyCallable]:
    '''
    Args:
        selector: optionally specify name of doc template dictionary to use; if not provided, the name of the function will be used.
    '''
    def decorator(f: AnyCallable) -> AnyCallable:

        assert f.__doc__ is not None, f'{f} must have a docstring!'

        nonlocal selector
        selector = f.__name__ if selector is None else selector
        # get doc string, template with decorator args, then template existing doc string
        doc_src = getattr(DOC_TEMPLATE, selector)
        if isinstance(doc_src, str):
            doc = doc_src.format(**kwargs)
            f.__doc__ = f.__doc__.format(doc)
        else: # assume it is a dictionary
            # try to format each value
            doc_dict = {k: v.format(**kwargs) for k, v in doc_src.items()}
            f.__doc__ = f.__doc__.format(**doc_dict)

        return f

    return decorator
