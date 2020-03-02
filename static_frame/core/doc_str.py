'''
Storage for common doc strings and templates shared in non-related classes and methods.
'''


import typing as tp

from static_frame.core.util import AnyCallable

#NOTE: for kwargs, it is sometimes useful to only define the string, not the variable name, as in some contexts different variable names are use same conceptual entity.

OWN_INDEX = '''own_index: Flag the passed index as ownable by this :obj:`static_frame.{class_name}`. Primarily used by internal clients.'''

OWN_DATA = '''own_data: Flag the data values as ownable by this :obj:`static_frame.{class_name}`. Primarily used by internal clients.'''

OWN_COLUMNS = '''own_columns: Flag the passed columns as ownable by this :obj:`static_frame.{class_name}`. Primarily used by internal clients.'''

INDEX_INITIALIZER = '''An iterable of unique, hashable values, or another ``Index`` or ``IndexHierarchy``, to be used as the labels of the index.'''

LOC_SELECTOR = '''A loc selector, either a label, a list of labels, a slice of labels, or a Boolean array.'''

ILOC_SELECTOR = '''An iloc selector, either an index, a list of indicces, a slice of indices, or a Boolean array.'''

DTYPE_SPECIFIER = '''dtype: A value suitable for specyfying a NumPy dtype, such as a Python type (float), NumPy array protocol strings ('f8'), or a dtype instance.'''

AXIS = '''axis: Integer specifying axis, where 0 is rows and 1 is columns. Axis 0 is set by default.'''

class DOC_TEMPLATE:

    #---------------------------------------------------------------------------
    # functions

    to_html = '''
    Return an HTML table representation of this :obj:`static_frame.{class_name}` using standard TABLE, TR, and TD tags. This is not a complete HTML page.

    Args:
        config: Optional :obj:`static_frame.DisplayConfig` instance.
    '''

    to_html_datatables = '''
    Return a complete HTML representation of this :obj:`static_frame.{class_name}` using the DataTables JS library for table naviagation and search. The page links to CDNs for JS resources, and thus will not fully render without an internet connection.

    Args:
        fp: optional file path to write; if not provided, a temporary file will be created. Note: the caller is responsible for deleting this file.
        show: if True, the file will be opened with a webbrowser.
        config: Optional :obj:`static_frame.DisplayConfig` instance.

    Returns:
        Absolute file path to the file written.
    '''

    clip = '''Apply a clip opertion to this :obj:`static_frame.{class_name}`. Note that clip operations can be applied to object types, but cannot be applied to non-numerical objects (e.g., strings, None)'''

    ufunc_skipna = '''{header}

    Args:
        axis: Axis, defaulting to axis 0.
        skipna: Skip missing (NaN) values, defaulting to True.
    '''

    label_widths_at_depth = '''
    A generator of pairs, where each pair is the label and the count of that label found at the depth specified by  ``depth_level``.

    Args:
        depth_level: a depth level, starting from zero.
    '''

    interface = '''
    A :obj:`static_frame.Frame` documenting the interface of this class.
    '''

    name = '''
    A hashable label attached to this container.
    '''

    #---------------------------------------------------------------------------
    # dict entries
    selector = dict(
            key_loc=LOC_SELECTOR,
            key_iloc=ILOC_SELECTOR,
            )

    reindex = dict(
            doc='''Return a new :obj:`static_frame.{class_name}` with labels defined by the provided index. The size and ordering of the data is determined by the newly provided index, where data will continue to be aligned under labels found in both the new and the old index. Labels found only in the new index will be filled with ``fill_value``.
            ''',
            index_initializer=INDEX_INITIALIZER,
            fill_value='''fill_value: A value to be used to fill space created by a new index that has values not found in the previous index.''',
            own_index=OWN_INDEX,
            own_columns=OWN_COLUMNS
            )

    relabel = dict(
            doc ='''Return a new :obj:`static_frame.{class_name}` with transformed labels on the index. The size and ordering of the data is never chagned in a relabeling operation. The resulting index must be unique.
            ''',
            count='''A positive integer drops that many outer-most levels; a negative integer drops that many inner-most levels.''',
            level='''A hashable value to be used as a new root level, extending or creating an ``IndexHierarchy``''',
            relabel_input='''One of the following types, used to create a new ``Index`` with the same size as the previous index. (a) A mapping (as a dictionary or ``Series``), used to lookup and transform the labels in the previous index. Previous labels not found in the mapping will be reused. (b) A function, returning a hashable, that is applied to each label in the previous index. (c) The ``IndexAutoFactory`` type, to apply an auto-incremented integer index. (d) An index initializer, i.e., either an iterable of hashables or an ``Index`` instance.'''
            )

    relabel_flat = dict(
            doc='''Return a new :obj:`static_frame.{class_name}`, where an ``IndexHierarchy`` (if defined) is replaced with a flat, one-dimension index of tuples.
            ''',
            )

    relabel_add_level = dict(
            doc='''Return a new :obj:`static_frame.{class_name}`, adding a new root level to an existing ``IndexHierarchy``, or creating an ``IndexHierarchy`` if one is not yet defined.
            ''',
            level='''A hashable value to be used as a new root level, extending or creating an ``IndexHierarchy``''',
            )

    relabel_drop_level = dict(
            doc='''Return a new :obj:`static_frame.{class_name}`, dropping one or more levels from a either the root or the leaves of an ``IndexHierarchy``. The resulting index must be unique.
            ''',
            count='''A positive integer drops that many outer-most (root) levels; a negative integer drops that many inner-most (leaf)levels.''',
            )

    head = dict(
            doc='''Return a :obj:`static_frame.{class_name}` consisting only of the top elements as specified by ``count``.
            ''',
            count='''count: Number of elements to be returned from the top of the :obj:`static_frame.{class_name}`''',
            )

    tail = dict(
            doc='''Return a :obj:`static_frame.{class_name}` consisting only of the bottom elements as specified by ``count``.
            ''',
            count='''count: Number of elements to be returned from the bottom of the :obj:`static_frame.{class_name}`''',
            )


    index_init = dict(
            args = f'''
        Args:
            labels: {INDEX_INITIALIZER}
            name: A hashable object to name the Index.
            loc_is_iloc: Optimization when a contiguous integer index is provided as labels. Generally only set by internal clients.
            {DTYPE_SPECIFIER}'''
            )

    index_date_time_init = dict(
            args = '''
        Args:
            labels: Iterable of hashable values to be used as the index labels. If strings, NumPy datetime conversions will be applied.
            name: A hashable object to name the Index.
            '''
            )

    from_pandas = dict(
            own_data='''own_data: If True, the underlying NumPy data array will be made immutable and used without a copy.''',
            own_index='''own_index: If True, the underlying NumPy index label array will be made immutable and used without a copy.''',
            own_columns='''own_columns: If True, the underlying NumPy column label array will be made immutable and used without a copy.''',
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
            dtypes='''dtypes: Optionally provide an iterable of dtypes, equal in length to the length of each row, or a mapping by column name. If a dtype is given as None, NumPy's default type determination will be used.
            ''',
            name='name: A hashable object to name the Frame.',
            consolidate_blocks='consolidate_blocks: Optionally consolidate adjacent same-typed columns into contiguous arrays.'
    )

    fillna = dict(
            limit='limit: Set the maximum count of missing values (NaN or None) to be filled per contiguous region of missing vlaues. A value of 0 is equivalent to no limit.',
            value='value: Value to be used to replace missing values (NaN or None).',
            axis='axis: Axis upon which to evaluate contiguous missing values, where 0 is vertically (between row values) and 1 is horizontally (between column values).'
    )


    argminmax = dict(
            skipna='skipna: if True, NaN or None values will be ignored; if False, a found NaN will propagate.',
            axis='axis: Axis upon which to evaluate contiguous missing values, where 0 is vertically (between row values) and 1 is horizontally (between column values).'
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

    apply = dict(
            doc='Apply a function to each value.',
            func='func: A function that takes a value.',
            dtype=DTYPE_SPECIFIER
            )

    astype = dict(
            dtype=DTYPE_SPECIFIER
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
