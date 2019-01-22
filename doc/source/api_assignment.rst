
Assignment / Dropping / Masking
=======================================================

:py:class:`Series` and :py:class:`Frame` provide assign-to-copy interfaces, permitting expressive assignment to new containers with the same flexibility as Pandas and NumPy.

In addition, :py:class:`Series` and :py:class:`Frame` expose the same interfaces for dropping data and producing Boolean masks.


Series
---------


Assignment
...............


.. py:method:: Series.assign[key](value)

    Given a ``key``, replace the values specified with ``value``.

    Args:
        key: A column selector, either a loc label, a list of labels, a slice of labels, or a Boolean array.

.. py:method:: Series.assign.loc[key](value)

    Given a loc ``key``, replace the values specified with ``value``.

.. py:method:: Series.assign.iloc[key](value)

    Given an iloc ``key``, replace the values specified with ``value``.


Dropping Data
....................


.. py:method:: Series.drop[key]

    Given a ``key``, remove the values specified by the key.

.. py:method:: Series.drop.loc[key]

    Given a loc ``key``, remove the values specified by the key.

.. py:method:: Series.drop.iloc[key]

    Given an iloc ``key``, remove the values specified by the key.


Masking Data
....................


.. py:method:: Series.mask[key] -> Series

    Given a ``key``, mask (set to ``True``) the values specified by the key.

.. py:method:: Series.mask.loc[key]

    Given a loc ``key``, mask (set to ``True``) the values specified by the key.

.. py:method:: Series.mask.iloc[key]

    Given an iloc ``key``, mask (set to ``True``) the values specified by the key.


Creating a Masked Array
..........................

.. py:method:: Series.masked_array[key] -> Series

    Given a ``key``, mask (set to ``True``) the values specified by the key and return a NumPy ``MaskedArray``.

.. py:method:: Series.masked_array.loc[key]

    Given a loc ``key``, mask (set to ``True``) the values specified by the key and return a NumPy ``MaskedArray``.

.. py:method:: Series.masked_array.iloc[key]

    Given an iloc ``key``, mask (set to ``True``) the values specified by the key and return a NumPy ``MaskedArray``.




Frame
---------

Assignment
..................

.. py:method:: Frame.assign[key](value)

    Given a ``key``, replace the values specified with ``value``.

.. py:method:: Frame.assign.loc[key](value)

    Given a loc ``key``, replace the values specified with ``value``.

.. py:method:: Frame.assign.iloc[key](value)

    Given an iloc ``key``, replace the values specified with ``value``.


Dropping Data
..................

.. py:method:: Frame.drop[key]

    Given a ``key``, remove the values specified by the key.

.. py:method:: Frame.drop.loc[key]

    Given a loc ``key``, remove the values specified by the key.

.. py:method:: Frame.drop.iloc[key]

    Given an iloc ``key``, remove the values specified by the key.


Masking Data
..................

.. py:method:: Frame.mask[key]

    Given a ``key``, mask (set to ``True``) the values specified by the key.

.. py:method:: Frame.mask.loc[key]

    Given a loc ``key``, mask (set to ``True``) the values specified by the key.

.. py:method:: Frame.mask.iloc[key]

    Given an iloc ``key``, mask (set to ``True``) the values specified by the key.


Creating a Masked Array
..........................

.. py:method:: Frame.masked_array[key] -> Frame

    Given a ``key``, mask (set to ``True``) the values specified by the key and return a NumPy ``MaskedArray``.

.. py:method:: Frame.masked_array.loc[key]

    Given a loc ``key``, mask (set to ``True``) the values specified by the key and return a NumPy ``MaskedArray``.

.. py:method:: Frame.masked_array.iloc[key]

    Given an iloc ``key``, mask (set to ``True``) the values specified by the key and return a NumPy ``MaskedArray``.

