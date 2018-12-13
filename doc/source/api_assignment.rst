
Assignment
===============================

:py:class:`Series` and :py:class:`Frame` provide assign-to-copy interfaces, permitting immutable assignment to new containers with the same flexibility as Pandas and NumPy.


Series
---------

.. py:method:: Series.assign[key](value)

    Given a key, replace the values specified by the key with value.

.. py:method:: Series.assign.loc[key](value)

    Given a loc key, replace the values specified by the key with value.

.. py:method:: Series.assign.iloc[key](value)

    Given a iloc key, replace the values specified by the key with value.


Frame
---------

.. py:method:: Frame.assign[key](value)

    Given a key, replace the values specified by the key with value.

.. py:method:: Frame.assign.loc[key](value)

    Given a loc key, replace the values specified by the key with value.

.. py:method:: Frame.assign.iloc[key](value)

    Given a iloc key, replace the values specified by the key with value.


