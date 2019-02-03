Selection
=======================================================

Data selection permits returning views of data contained within a container.

The two-dimensional :py:class:`Frame` exposes three primary means of data selection: a root ``__getitem__`` interface, as well as ``__getitem__`` interfaces on ``loc`` and ``iloc`` attributes. While the one-dimensional :py:class:`Series` provides the same interface, the root and ``loc`` ``__getitem__`` are identical./

As much as possible, slices or views of underlying data will be returned from selection operations. As underlying data is immutable, there is no risk of undesirable side-effects from returning views of underlying data.






Series
---------------------------

.. py:method:: Series[key]
.. py:method:: Series.loc[key]
.. py:method:: Series.iloc[key]


    Return the values specified by ``key``.

    :param key: A selector, either a label, a list of labels, a slice of labels, or a Boolean array. The root ``__getitem__`` takes loc labels, ``loc`` takes loc labels, and ``iloc`` takes integer indices.

    :returns: :py:class:`static_frame.Series`


.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_series_selection_a
   :end-before: end_series_selection_a



Frame
---------------------------

.. py:method:: Frame[key]
.. py:method:: Frame.loc[key]
.. py:method:: Frame.iloc[key]

    Return the values specified by ``key``.

    :param key: A selector, either a label, a list of labels, a slice of labels, or a Boolean array. The root ``__getitem__`` takes loc labels, ``loc`` takes loc labels, and ``iloc`` takes integer indices. The root ``__getitem__`` interface is a column selector; ``loc`` and ``iloc`` interfaces accept one or two arguments, for either row selection or row and column selection (respectively).

    :returns: :py:class:`static_frame.Frame`

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_frame_selection_a
   :end-before: end_frame_selection_a



Selection Modifiers
=======================================================

StaticFrame permits using selection modifiers in ``loc`` selectors. These modifiers permit encapsulating, per axis, a different kind of selection.

.. py:method:: ILoc[key]

    A wrapper for embedding ``iloc`` specificiations within a single axis argument of a ``loc`` selection.


.. py:method:: HLoc[key]

    A wrapper for embedding hierarchical specificiations for :py:class:`static_frame.IndexHierarchy` within a single axis argument of a ``loc`` selection.


