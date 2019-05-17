

About StaticFrame
*******************

The StaticFrame library consists of the Series and Frame, immutable data structures for one- and two-dimensional calculations with self-aligning, labelled axes. StaticFrame offers an alternative to Pandas. While many interfaces for data extraction and manipulation are similar to Pandas, StaticFrame deviates from Pandas in many ways: all data is immutable, and all indices must be unique; all vector processing uses NumPy, and the full range of NumPy data types is preserved; the implementation is concise and lightweight; consistent naming and interfaces are used throughout; and flexible approaches to iteration and function application, with built-in options for parallelization, are provided.


.. admonition:: Alpha Release
    :class: Warning

    The current release of StaticFrame is an alpha release, meaning that interfaces may change, functionality may be incomplete, and there may be significant flaws. Please assist in development by reporting any bugs or request any missing features.

    https://github.com/InvestmentSystems/static-frame/issues


StaticFrame is not a drop-in replacement for Pandas. While some conventions and API components are directly borrowed from Pandas, some are completely different, either by necessity (due to the immutable data model) or by choice (offering more uniform, less redundant, and more explicit interfaces). Further, as StaticFrame does not support in-place mutation, architectures that made significant use of mutability in Pandas will require refactoring.


StaticFrame is lightweight. It has few dependencies (Pandas is not a dependency). The core library is less than 10,000 lines of code, less than 5% the size of the Pandas code base [#]_.

StaticFrame does not aspire to be an all-in-one framework for all aspects of data processing and visualization. StaticFrame focuses on providing efficient and powerful data structures with a consistent, clear, and stable interfaces.

StaticFrame does not implement its own types or numeric computation routines, relying entirely on NumPy. NumPy offers desirable stability in performance and interface. For working with SciPy and related tools, StaticFrame exposes easy access to NumPy arrays.



Immutability
***********************************

The :py:class:`static_frame.Series` and :py:class:`static_frame.Frame` store data in immutable NumPy arrays. Once created, array values cannot be changed. StaticFrame manages NumPy arrays, setting the ``ndarray.flags.writeable`` attribute to False on all managed and returned NumPy arrays.

.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_immutability
   :end-before: end_immutability


To mutate values in a ``Series`` or ``Frame``, a copy must be made. Convenient functional interfaces to assign to a copy are provided, using conventions familiar to NumPy and Pandas users.


.. literalinclude:: ../../static_frame/test/unit/test_doc.py
   :language: python
   :start-after: start_assign
   :end-before: end_assign


Immutable data has the overwhelming benefit of providing the confidence that a client of a ``Series`` or ``Frame`` cannot mutate its data. This removes the need for many unnecessary copies, and forces clients to only make copies when absolutely necessary.

There is no guarantee that using immutable data will produce correct code or more resilient and robust libraries. It is true, however, that using immutable data removes countless opportunities for introducing flaws in data processing routines and libraries.



History
***********************************

The ideas behind StaticFrame developed out of years of work with Pandas and related tabular data structures by the Investment Systems team at Research Affiliates, LLC. In May of 2017 Christopher Ariza proposed the basic model to the Investment Systems team and began implementation. The first public release was in May 2018.


Presentations
***********************************

The following presentations and interviews describe StaticFrame in greater depth.

- PyData LA 2018: https://pyvideo.org/pydata-la-2018/staticframe-an-immutable-alternative-to-pandas.html
- PyCon US 2019, lightning talk (starting at 53:00): https://pyvideo.org/pycon-us-2019/friday-lightning-talksbreak-pycon-2019.html
- Talk Python to Me, interview: https://talkpython.fm/episodes/show/204/staticframe-like-pandas-but-safer



Contributors
***********************************

These members of the Investment Systems team have contributed greatly to the design of StaticFrame:

- Brandt Bucher
- Guru Devanla
- John Hawk
- Adam Kay
- Mark LeMoine
- Myrl Marmarelis
- Tom Rutherford
- Yu Tomita
- Quang Vu

Thanks also for additional contributions from GitHub pull requests.

https://github.com/InvestmentSystems/static-frame/graphs/contributors


.. [#] The Pandas 2.0 Design Docs state that the Pandas codebase has over 200,000 lines of code: https://pandas-dev.github.io/pandas2/goals.html


