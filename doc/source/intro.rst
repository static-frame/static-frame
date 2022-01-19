

About StaticFrame
*******************

StaticFrame is not a drop-in replacement for Pandas. While some conventions and API components are directly borrowed from Pandas, some are completely different, either by necessity (due to the immutable data model) or by choice (offering more uniform, less redundant, and more explicit interfaces). As StaticFrame does not support in-place mutation, architectures that made significant use of mutability in Pandas will require refactoring.

For more comparisons to Pandas, see `Ten Reasons to Use StaticFrame instead of Pandas <https://dev.to/flexatone/ten-reasons-to-use-staticframe-instead-of-pandas-4aad>`_.

For a concise overview of StaticFrame interfaces, start with :ref:`api-overview-Frame`.

StaticFrame does not aspire to be an all-in-one framework for all aspects of data processing and visualization. StaticFrame focuses on providing efficient and powerful data structures with consistent, clear, and stable interfaces.

StaticFrame targets comparable or better performance than Pandas. While this is already the case for many core operations, other operations are, for now, still more performant in Pandas (such as reading delimited text files via ``pd.read_csv``). StaticFrame provides easy conversion to and from Pandas to bridge needed functionality or performance.

StaticFrame relies entirely on NumPy for types and numeric computation routines. NumPy offers desirable stability in performance and interface. For working with SciPy and related tools, StaticFrame exposes easy access to NumPy arrays, conversion to and from Pandas and Arrow, and support for reading from and writing to a wide variety of storage formats.

Please assist in development by reporting bugs or requesting features. We are a welcoming community and appreciate all feedback! Visit `GitHub Issues <https://github.com/InvestmentSystems/static-frame/issues>`_. To get started contributing to StaticFrame, see :ref:`contributing`.


Immutability
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


Presentations
***********************************

The following presentations and interviews describe StaticFrame in greater depth.

- PyData Global 2021: "Why Datetimes Need Units: Avoiding a Y2262 Problem & Harnessing the Power of NumPy's datetime64": https://zoom.us/rec/share/MhHxZLi-SMkU3Sewhv7MKLWhgS0y0T7E7xFqAWfukUNdUGtFJFcHxJf8g2r_dTqq.cBJaD2SZP5P7eLI9?startTime=1635534301000
- PyData LA 2019: "The Best Defense is not a Defensive Copy" (lightning talk starting at 18:25): https://youtu.be/_WXMs8o9Gdw
- PyData LA 2019: "Fitting Many Dimensions into One The Promise of Hierarchical Indices for Data Beyond Two Dimensions": https://youtu.be/xX8tXSNDpmE
- PyCon US 2019: "A Less Kind, Less Gentle DataFrame" (lightning talk starting at 53:00): https://pyvideo.org/pycon-us-2019/friday-lightning-talksbreak-pycon-2019.html
- Talk Python to Me, interview: https://talkpython.fm/episodes/show/204/staticframe-like-pandas-but-safer
- PyData LA 2018: "StaticFrame: An Immutable Alternative to Pandas": https://pyvideo.org/pydata-la-2018/staticframe-an-immutable-alternative-to-pandas.html


Contributors
***********************************

These members of the Investment Systems team have contributed greatly to the design of StaticFrame:

- Brandt Bucher
- Charles Burkland
- Guru Devanla
- John Hawk
- Adam Kay
- Mark LeMoine
- Myrl Marmarelis
- Tom Rutherford
- Yu Tomita
- Quang Vu

Thanks also for additional contributions from GitHub users:

https://github.com/InvestmentSystems/static-frame/graphs/contributors

