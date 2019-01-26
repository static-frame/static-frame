
Assignment / Dropping / Masking
=======================================================

:py:class:`Series` and :py:class:`Frame` provide interface attributes for exposing assignment-like operations, dropping data, and producing masks. Each interface attribute exposes a root ``__getitem__`` interface, as well as ``__getitem__`` interfaces on ``loc`` and ``iloc`` attributes, exposing the full range selection approaches.


Assignment
---------------------------

The assign-to-copy interfaces permit expressive assignment to new containers with the same flexibility as Pandas and NumPy. As all underlying data is immutable, the caller will not be mutated. With ``Frame`` objects, the minimum amount of data will be copied to the new ``Frame``, depending on the type of assignment and the organization of the underlying ``TypeBlocks``.


Series
.................

.. py:method:: Series.assign[key](value)
.. py:method:: Series.assign.loc[key](value)
.. py:method:: Series.assign.iloc[key](value)

    Replace the values specified by the ``key`` with ``value``.

    :param key: A selector, either a label, a list of labels, a slice of labels, or a Boolean array. The root ``__getitem__`` takes loc labels, ``loc`` takes loc labels, and ``iloc`` takes integer indices.

    :param value: The value to be assigned. Can be a single value, an iterable of values, or a ``Series``.

    :returns: :py:class:`static_frame.Series`


.. literalinclude:: api.py
   :language: python
   :start-after: start_series_assign_a
   :end-before: end_series_assign_a


Frame
.................

.. py:method:: Frame.assign[key](value)
.. py:method:: Frame.assign.loc[key](value)
.. py:method:: Frame.assign.iloc[key](value)

    Replace the values specified by the ``key`` with ``value``.

    :param key: A selector, either a label, a list of labels, a slice of labels, or a Boolean array. The root ``__getitem__`` takes loc labels, ``loc`` takes loc labels, and ``iloc`` takes integer indices. The root ``__getitem__`` interface is a column selector; ``loc`` and ``iloc`` interfaces accept one or two arguments, for either row selection or row and column selection (respectively).

    :param value: The value to be assigned. Can be a single value, an iterable of values, a ``Series``, or a ``Frame``.

    :returns: :py:class:`static_frame.Frame`


.. literalinclude:: api.py
   :language: python
   :start-after: start_frame_assign_a
   :end-before: end_frame_assign_a




Dropping Data
--------------------------------

While data from a ``Series`` or ``Frame`` can be excluded through common selection interfaces, in some cases it is more efficient and readable to specify what to drop rather than what to keep. The drop interface return new containers, efficiently removing the values specified by the key. For ``Frame``, removal of rows and columns can happen simultaneously.


Series
.................

.. py:method:: Series.drop[key]
.. py:method:: Series.drop.loc[key]
.. py:method:: Series.drop.iloc[key]

    Remove the values specified by the ``key``.

    :param key: A selector, either a label, a list of labels, a slice of labels, or a Boolean array. The root ``__getitem__`` takes loc labels, ``loc`` takes loc labels, and ``iloc`` takes integer indices.

    :returns: :py:class:`static_frame.Series`


.. literalinclude:: api.py
   :language: python
   :start-after: start_series_drop_a
   :end-before: end_series_drop_a




Frame
.................

.. py:method:: Frame.drop[key]
.. py:method:: Frame.drop.loc[key]
.. py:method:: Frame.drop.iloc[key]

    Remove the values specified by the ``key``.

    :param key: A selector, either a label, a list of labels, a slice of labels, or a Boolean array. The root ``__getitem__`` takes loc labels, ``loc`` takes loc labels, and ``iloc`` takes integer indices. The root ``__getitem__`` interface is a column selector; ``loc`` and ``iloc`` interfaces accept one or two arguments, for either row selection or row and column selection (respectively).

    :returns: :py:class:`static_frame.Frame`



.. literalinclude:: api.py
   :language: python
   :start-after: start_frame_drop_a
   :end-before: end_frame_drop_a




Masking Data
------------------------------

While Boolean ``Series`` and ``Frame`` can be created directly or with comparison operators (or functions like ``isin()``), in some cases it is desirable to directly specify a mask through the common selection idioms.


Series
.................

.. py:method:: Series.mask[key]
.. py:method:: Series.mask.loc[key]
.. py:method:: Series.mask.iloc[key]

    Mask (set to ``True``) the values specified by the key and return a Boolean ``Series``.

    :param key: A selector, either a label, a list of labels, a slice of labels, or a Boolean array. The root ``__getitem__`` takes loc labels, ``loc`` takes loc labels, and ``iloc`` takes integer indices.

    :returns: :py:class:`static_frame.Series`


Frame
.................

.. py:method:: Frame.mask[key]
.. py:method:: Frame.mask.loc[key]
.. py:method:: Frame.mask.iloc[key]

    Mask (set to ``True``) the values specified by the key and return a Boolean ``Frame``.

    :param key: A selector, either a label, a list of labels, a slice of labels, or a Boolean array. The root ``__getitem__`` takes loc labels, ``loc`` takes loc labels, and ``iloc`` takes integer indices. The root ``__getitem__`` interface is a column selector; ``loc`` and ``iloc`` interfaces accept one or two arguments, for either row selection or row and column selection (respectively).

    :returns: :py:class:`static_frame.Frame`




Creating a Masked Array
----------------------------

NumPy masked arrays permit blocking out problematic data (i.e., NaNs) while maintaining compatibility with nearly all NumPy operations.

https://docs.scipy.org/doc/numpy/reference/maskedarray.generic.html


Series
.................

.. py:method:: Series.masked_array[key]
.. py:method:: Series.masked_array.loc[key]
.. py:method:: Series.masked_array.iloc[key]

    Mask (set to ``True``) the values specified by the key and return a NumPy ``MaskedArray``.

    :param key: A selector, either a label, a list of labels, a slice of labels, or a Boolean array. The root ``__getitem__`` takes loc labels, ``loc`` takes loc labels, and ``iloc`` takes integer indices.

    :returns: :py:class:`np.ma.MaskedArray`


Frame
.................

.. py:method:: Frame.masked_array[key]
.. py:method:: Frame.masked_array.loc[key]
.. py:method:: Frame.masked_array.iloc[key]

    Mask (set to ``True``) the values specified by the key and return a NumPy ``MaskedArray``.

    :param key: A selector, either a label, a list of labels, a slice of labels, or a Boolean array. The root ``__getitem__`` takes loc labels, ``loc`` takes loc labels, and ``iloc`` takes integer indices. The root ``__getitem__`` interface is a column selector; ``loc`` and ``iloc`` interfaces accept one or two arguments, for either row selection or row and column selection (respectively).

    :returns: :py:class:`np.ma.MaskedArray`

