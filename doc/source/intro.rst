

About StaticFrame
*******************

StaticFrame is an alternative dataframe library built on an immutable data model. StaticFrame is not a drop-in replacement for Pandas. While some conventions and API components are directly borrowed from Pandas, some are completely different, either by necessity (due to the immutable data model) or by choice (offering more uniform, less redundant, and more explicit interfaces). As StaticFrame does not support in-place mutation, architectures that made significant use of mutability in Pandas will require refactoring.

Please assist in development by reporting bugs or requesting features. We are a welcoming community and appreciate all feedback! Visit `GitHub Issues <https://github.com/InvestmentSystems/static-frame/issues>`_. To get started contributing to StaticFrame, see :ref:`contributing`.


About Immutability
***********************************

The :obj:`Series` and :obj:`Frame` store data in immutable NumPy arrays. Once created, array values cannot be changed. StaticFrame manages NumPy arrays, setting the ``ndarray.flags.writeable`` attribute to False on all managed and returned NumPy arrays.

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_immutability
   :end-before: end_immutability


To mutate values in a ``Series`` or ``Frame``, a copy must be made. Convenient functional interfaces to assign to a copy are provided, using conventions familiar to NumPy and Pandas users.


.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_assign
   :end-before: end_assign


Immutable data has the overwhelming benefit of providing the confidence that a client of a ``Series`` or ``Frame`` cannot mutate its data. This removes the need for many unnecessary, defensive copies, and forces clients to only make copies when absolutely necessary.

There is no guarantee that using immutable data will produce correct code or more resilient and robust libraries. It is true, however, that using immutable data removes countless opportunities for introducing flaws in data processing routines and libraries.


History
***********************************

The ideas behind StaticFrame developed out of years of work with Pandas and related tabular data structures by the Investment Systems team at Research Affiliates, LLC. In May of 2017 Christopher Ariza proposed the basic model to the Investment Systems team and began implementation. The first public release was in May 2018.


Media
********************************

Articles
..........

- 2022: `One Fill Value Is Not Enough: Preserving Columnar Types When Reindexing DataFrames <https://dev.to/flexatone/one-fill-value-is-not-enough-preserving-columnar-types-when-reindexing-dataframes-2jdj>`_
- 2022: `StaticFrame from the Ground Up: Getting Started with Immutable DataFrames <https://mybinder.org/v2/gh/InvestmentSystems/static-frame-ftgu/default?urlpath=tree/index.ipynb>`_
- 2022: `Using Higher-Order Containers to Efficiently Process 7,163 (or More) DataFrames <https://towardsdatascience.com/using-higher-order-containers-to-efficiently-process-7-163-or-more-dataframes-964da8b0c679>`_
- 2020: `Ten Reasons to Use StaticFrame instead of Pandas <https://dev.to/flexatone/ten-reasons-to-use-staticframe-instead-of-pandas-4aad>`_


Presentations
..................

- PyCon US 2022: "Employing NumPy's NPY Format for Faster-Than-Parquet DataFrame Serialization": https://youtu.be/HLH5AwF-jx4
- PyData Global 2021: "Why Datetimes Need Units: Avoiding a Y2262 Problem & Harnessing the Power of NumPy's datetime64": https://www.youtube.com/watch?v=jdnr7sgxCQI
- PyData LA 2019: "The Best Defense is not a Defensive Copy" (lightning talk starting at 18:25): https://youtu.be/_WXMs8o9Gdw?t=1105
- PyData LA 2019: "Fitting Many Dimensions into One The Promise of Hierarchical Indices for Data Beyond Two Dimensions": https://youtu.be/xX8tXSNDpmE
- PyCon US 2019: "A Less Kind, Less Gentle DataFrame" (lightning talk starting at 53:00): https://pyvideo.org/pycon-us-2019/friday-lightning-talksbreak-pycon-2019.html
- Talk Python to Me, interview: https://talkpython.fm/episodes/show/204/staticframe-like-pandas-but-safer
- PyData LA 2018: "StaticFrame: An Immutable Alternative to Pandas": https://pyvideo.org/pydata-la-2018/staticframe-an-immutable-alternative-to-pandas.html


