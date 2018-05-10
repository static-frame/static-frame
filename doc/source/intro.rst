

Introduction
*******************

The StaticFrame library consists of the Series and Frame, immutable data structures for one- and two-dimensional calculations with self-aligning, labelled axis. StaticFrame offers an alternative to Pandas. While many interfaces for data extraction and manipulation are similar to Pandas, StaticFrame deviates from Pandas in many ways: all data is immutable, and all indices must be unique; all vector processing uses NumPy, and the full range of NumPy data types is preserved; the implementation is concise and lightweight; consistent naming and interfaces are used throughout; and flexible approaches to iteration and function application, with built-in options for parallelization, are provided.


.. admonition:: Alpha Release
    :class: Warning

    The current release of StaticFrame is an alpha release, meaning that interfaces may change, functionality may be incomplete, and there may be significant flaws. Please assist in development by reporting any bugs or request any missing features.

    https://github.com/InvestmentSystems/static-frame/issues



What StaticFrame Is and Is Not
=================================

StaticFrame is not a drop-in replacment for Pandas. While some conventions and API components are directly borrowed from Pandas, some are completely different, either by necessity (due the immutable data model) or by choice (to offer more uniform, less redundant, and more explicit interfaces). Further, as StaticFrame does not support in-place mutation, approaches used in Pandas will have to be refactored for StaticFrame. Certain conveniences in Pandas are not supported for the sake of having "one ... obvious way to do it" [#]_ or to achieve greater consistency.

StaticFrame requires modern Python (3.5+) and modern NumPy (1.14.1+). There are no plans to support older versions. Modern features of Python, such as type-hints, are used throughout the code, and provide benefits for users using type-hint-aware IDEs.

StaticFrame is lightweight. It has few dependencies (Pandas is not a dependency). The entire library is less than 6,000 lines of code, less than 3% the size of the Pandas code base [#]_.

StaticFrame does not aspire to be an all-in-one framework for all aspects of data processing and visualization. StaticFrame focuses on providing efficient and powerful data structures.

StaticFrame does not implement its own types or numeric computation routines, relying entirely on NumPy. NumPy offers desirable stability in performance and interface. For working with SciPy and related tools, StaticFrame exposes easy access to NumPy arrays.



Immutability
===============================

The :py:class:`Series` and :py:class:`Frame` store data in immutable NumPy arrays. Once created, array values cannot be changed. StaticFrame manages NumPy arrays, setting the ``ndarray.flags.writeable`` attribute to False on all managed and returned NumPy arrays.

.. literalinclude:: intro.py
   :language: python
   :start-after: start_immutability
   :end-before: end_immutability


To mutate values in a :py:class:`Series` or :py:class:`Frame`, a copy must be made. Convenient functional interfaces to assign to a copy are provided, using conventions familiar to NumPy and Pandas users.


.. literalinclude:: intro.py
   :language: python
   :start-after: start_assign
   :end-before: end_assign


Immutable data has the overwhelming benefit of providing the confidence that a client of a :py:class:`Series` or :py:class:`Frame` cannot mutate its data. This removes the need for many unneccesary copies, and forces clients to only make copies when absolutely necessary.

There is no guarantee that using immutable data will produce correct code or more resilliant and robust libraries. It is true, however, that using immutable data removes countless opportunites for introducing serious flaws in data processing routines and libraries.



History of StaticFrame
============================

The ideas behind StaticFrame developed out of years of work with Pandas and related tabular data structures by the Investment Systems team at Research Affiliates, LLC. In May of 2017 Christopher Ariza proposed the basic model to the Investment Systems team and began implementation. The first public release was in May 2018.


Contributors
============================

These members of the Investment Systems team have contributed greatly to the design of StaticFrame:

- Guru Devanla
- John Hawk
- Adam Kay
- Mark LeMoine
- Tom Rutherford
- Yu Tomita
- Quang Vu


.. [#] The Zen of Python: https://www.python.org/dev/peps/pep-0020/

.. [#] The Pandas 2.0 Design Docs state that the Pandas codebase has over 200,000 lines of code: https://pandas-dev.github.io/pandas2/goals.html


