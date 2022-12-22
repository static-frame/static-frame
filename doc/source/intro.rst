

About StaticFrame
*******************

StaticFrame is an alternative DataFrame library built on an immutable data model. An immutable data model reduces opportunities for error and offers very significant performance advantages for some operations. (Read more about the benefits of immutability `here <https://static-frame.readthedocs.io/en/latest/articles/no_copy.html>`_.)

StaticFrame is not a drop-in replacement for Pandas. While some conventions and API components are directly borrowed from Pandas, some are completely different, either by necessity (due to the immutable data model) or by choice (offering more uniform, less redundant, and more explicit interfaces). (Read more about differences between Pandas and StaticFrame `here <https://static-frame.readthedocs.io/en/latest/articles/upgrade.html>`_.)

An interactive Jupyter notebook, with a guided introduction to using StaticFrame, is available `here <https://mybinder.org/v2/gh/static-frame/static-frame-ftgu/default?urlpath=tree/index.ipynb>`_.

StaticFrame's API documentation features thousands of code examples, covering nearly every end-point in the API. For each component, an "overview" and a "detail" section is provided. The overview lists one interface and its signature per line. (See the overview of ``Series`` constructors `here <https://static-frame.readthedocs.io/en/latest/api_overview/series-constructor.html#api-overview-series-constructor>`_.) The detail section provides full documentation as well as extensive code examples. (See the detail of ``Frame`` methods `here <https://static-frame.readthedocs.io/en/latest/api_detail/frame-method.html#api-detail-frame-method>`_.)

Please assist in development by reporting bugs or requesting features. We are a welcoming community and appreciate all feedback! Visit `GitHub Issues <https://github.com/static-frame/static-frame/issues>`_. To get started contributing to StaticFrame, see :ref:`contributing`.

The ideas behind StaticFrame developed out of years of work with Pandas and related tabular data structures by the Investment Systems team at Research Affiliates, LLC. In May of 2017 Christopher Ariza proposed the basic model to the Investment Systems team and began implementation. The first public release was in May 2018.


Media
********************************

Articles
..........

- 2022: `The Performance Advantage of No-Copy DataFrame Operations <https://towardsdatascience.com/the-performance-advantage-of-no-copy-dataframe-operations-7bf8c565c9a0>`_
- 2022: `One Fill Value Is Not Enough: Preserving Columnar Types When Reindexing DataFrames <https://dev.to/flexatone/one-fill-value-is-not-enough-preserving-columnar-types-when-reindexing-dataframes-2jdj>`_
- 2022: `StaticFrame from the Ground Up: Getting Started with Immutable DataFrames <https://mybinder.org/v2/gh/static-frame/static-frame-ftgu/default?urlpath=tree/index.ipynb>`_
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


