# Double DataFrame Row Processing Performance with Free-Threaded Python
# Free-Threaded Python & Immutable DataFrames Deliver 2x Performance


Function application to each row of a DataFrame is a an ubiquitous operation.

<!-- Representing each row with a Series, expressive operations can be defined to reduce the DataFrame to Series.  -->

Such processing can be expensive, particularly with heterogenous columnar types that might have to be cast into a new array for each row.

<!-- Sometimes row-wise function application can be done more efficiently as column-wise operations, though not always. -->

But such processing is embarrassingly parallelizable: each row, or chunks of rows, can be processed independently, in parallel.

However, in Python, this was not practical. Multi-threaded function application avoids copying data, but is throttled by the Global Interpreter Lock (GIL). Alternatively, multi-processed function application, while effective when the unit of work is very time-consuming, is generally inefficient for row-wise function application. Subprocess interpreter overhead, as well as the cost of copying data into each subprocesses, overwhelms the performance benefits in most cases.

Python now offers a better option: with the new, "experimental free-threading build" of Python 3.13, true multi-threaded concurrency of CPU-bound operations is now possible. Without the cost of multi-processing, light-weight multi-threading can now deliver improved performance. For row-wise function application, free-threaded Python can process a DataFrame in half the time.


```python
Python 3.13.5 (main, Jun 11 2025, 15:36:57) [Clang 16.0.0 (clang-1600.0.26.6)]
>>> import numpy as np; import static_frame as sf
>>> f = sf.Frame(np.arange(1_000_000).reshape(1000, 1000))
>>> func = lambda s: s.loc[(s % 2) == 0].sum()

>>> %timeit f.iter_series(axis=1).apply(func)
17.7 ms ± 144 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

>>> %timeit f.iter_series(axis=1).apply_pool(func, use_threads=True, max_workers=4)
39.9 ms ± 354 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```



```python
Python 3.13.5 experimental free-threading build (main, Jun 11 2025, 15:36:57) [Clang 16.0.0 (clang-1600.0.26.6)] on darwin
>>> import numpy as np; import static_frame as sf
>>> f = sf.Frame(np.arange(1_000_000).reshape(1000, 1000))
>>> func = lambda s: s.loc[(s % 2) == 0].sum()

>>> %timeit f.iter_series(axis=1).apply(func)
21.3 ms ± 77.1 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)

>>> %timeit f.iter_series(axis=1).apply_pool(func, use_threads=True, max_workers=4)
7.89 ms ± 60.1 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```


<!-- Built on an immutable data model, already exposing interfaces for parallel function application, and now offering free-threaded compatible wheel dependencies, StaticFrame is ready now to take advantage of concurrency. -->


The risk of data races are a material problem with multi-threaded operations; a DataFrame built on an immutable data model, such as StaticFrame, removes these concerns.

<!-- Finally, mutable DataFrames, such as those provided by Pandas, expose opportunities for data races. -->

