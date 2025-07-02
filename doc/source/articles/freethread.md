# Double DataFrame Row Processing Performance with Free-Threaded Python
# Free-Threaded Python with Immutable DataFrames Deliver Significant Performance Improvements



Function application to each row of a DataFrame is a an ubiquitous operation.Such processing can be expensive (particularly with heterogenous columnar types) but is embarrassingly parallelizable: each row, or chunks of rows, can be processed independently, in parallel.

Until recently, exploiting this opportunity was not possible with Python. Multi-threaded function application on a DataFrame avoids copying data, but is throttled by the Global Interpreter Lock (GIL). Alternatively, multi-processed function application, while effective when the unit of work is very time-consuming, is inefficient in many common applications. Subprocess interpreter overhead, as well as the cost of copying data into each subprocesses, can overwhelm performance benefits.

Python now offers a solution: with the new, "experimental free-threading build" of Python 3.13, true multi-threaded concurrency of CPU-bound operations is possible. Without the cost of multi-processing, light-weight multi-threading can now deliver improved performance.

The benefits are extraordinary. StaticFrame, leveraging free-threaded Python, can process a DataFrame in less than half the time. For example, when using Python 3.13t (the "t" denotes the free-threading variant) to process a 1000 by 1000 `Frame` of uniform integers, the duration falls from 21.3 ms to 7.89 ms:

```python
# Python 3.13.5 experimental free-threading build (main, Jun 11 2025, 15:36:57) [Clang 16.0.0 (clang-1600.0.26.6)] on darwin
>>> import numpy as np; import static_frame as sf

>>> f = sf.Frame(np.arange(1_000_000).reshape(1000, 1000))
>>> func = lambda s: s.loc[(s % 2) == 0].sum()

>>> %timeit f.iter_series(axis=1).apply(func)
21.3 ms ± 77.1 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)

>>> %timeit f.iter_series(axis=1).apply_pool(func, use_threads=True, max_workers=4)
7.89 ms ± 60.1 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

As shown below, these results are generally consistent across a wide range of DataFrame shapes and compositions, and the out-performance scales positively with size.

When using standard Python, using threads actually degrades performance: the same operation takes almost 40 ms:

```python
# Python 3.13.5 (main, Jun 11 2025, 15:36:57) [Clang 16.0.0 (clang-1600.0.26.6)]
>>> import numpy as np; import static_frame as sf

>>> f = sf.Frame(np.arange(1_000_000).reshape(1000, 1000))
>>> func = lambda s: s.loc[(s % 2) == 0].sum()

>>> %timeit f.iter_series(axis=1).apply(func)
17.7 ms ± 144 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

>>> %timeit f.iter_series(axis=1).apply_pool(func, use_threads=True, max_workers=4)
39.9 ms ± 354 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

Unlocking such performance improvements simply by using the free-threaded variant of Python offers tremendous potential.

There are trade-offs: as shown above, the single-threaded base-line operation is slower under 3.13t (21.3 ms compared to 17.7 ms). There is known performance overhead in free-threaded Python that is an active area of CPython development; improvements are expected in 3.14t.

Further, while many C-extension packages like NumPy are now offering 3.13t pre-compiled binary wheels, the risk of thread contention or data races still exist. A DataFrame built on an immutable data model, such as StaticFrame, removes many of these concerns. StaticFrame has long leveraged immutable NumPy arrays, as well standard library `ThreadPoolExecutor` interfaces, laying the foundation for immediate adoption of free-threaded Python.


## Extended DataFrame Performance Tests

Evaluating performance characteristics of a complex data structure like a DataFrame requires multiple tests. The following performance tests perform row-wise function application on nine different DataFrame types, testing all combinations of three different shapes and three different levels of type homogeneity. For the same number of elements (1e6), three shapes are tested: tall (10,000 by 100), square (1,000 by 1,000), and wide (100 by 10,0000). To vary type homogeneity, three categories of synthetic data are defined: columnar (no adjacent columns have the same type), mixed (groups of four adjacent columns share the same type), and uniform (all columns are the same type). StaticFrame permits adjacent columns of the same type to be represented by two-dimensional NumPy arrays, reducing the cost of forming rows.

The same function as used above, where even numbers are identified and then summed, is used here: `lambda s: s.loc[(s % 2) == 0].sum()`. While a more efficient implementation is possible using the NumPy array directly, this function better replicates common applications where many intermediate `Series` are created.


### Multi-threaded Function Application with Python 3.13t

As shown below, the results of the isolated test in 3.13t are consistent across all DataFrame types tested: processing time is reduced at least by half. The optimal number of threads (the `max_workers` parameter) is smaller for tall versus wide DataFrames, presumably because the smaller rows are processed faster and do not componsate for additional thread overhead.

![threads-ftp-1e6-macos](https://raw.githubusercontent.com/static-frame/static-frame/master/doc/source/articles/freethread/threads-ftp-1e6-macos.png)


![process-ftp-1e8-macos](https://raw.githubusercontent.com/static-frame/static-frame/master/doc/source/articles/freethread/process-ftp-1e8-macos.png)



### Multi-threaded Function Application with Python 3.13

![threads-np-1e6-maco](https://raw.githubusercontent.com/static-frame/static-frame/master/doc/source/articles/freethread/threads-np-1e6-macos.png)



### Multi-processed Function Application with Python 3.13

![process-np-1e6-macos](https://raw.githubusercontent.com/static-frame/static-frame/master/doc/source/articles/freethread/process-np-1e6-macos.png)





## The Status of Free-Threaded Python

PEP 703 (https://peps.python.org/pep-0703/), "Making the Global Interpreter Lock Optional in CPython", was accepted by the Python Steering Council in July of 2023 with the guidance that, in the first phase (for Python 3.13) it is marked experimental and non-default; in the second phase, it becomes non-experimental and officially supported; in the third phase, it becomes the default Python implementation.

After significant CPython developement and support by critical packages like NumPy, PEP 779 (https://peps.python.org/pep-0779/), "Criteria for supported status for free-threaded Python" was accepted by the Python Steering Council in June of 2025. In Python 3.14, free-threaded Python will enter the second phase: non-experimental and officially supported.

While it is not certain when free-threaded Python will become the default, it is clear that a trajectory is set. The time is right for investing in free-threaded support and applications.


##  Conclusion




<!-- Built on an immutable data model, already exposing interfaces for parallel function application, and now offering free-threaded compatible wheel dependencies, StaticFrame is ready now to take advantage of concurrency. -->



<!-- Finally, mutable DataFrames, such as those provided by Pandas, expose opportunities for data races. -->




<!-- Representing each row with a Series, expressive operations can be defined to reduce the DataFrame to Series.  -->

<!-- Sometimes row-wise function application can be done more efficiently as column-wise operations, though not always. -->


