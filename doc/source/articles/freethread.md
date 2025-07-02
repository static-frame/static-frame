
# The Extraordinary Performance Now Possible with Immutable DataFrames in Free-Threaded Python

### How StaticFrame and Python 3.13t unlock dramatic multi-threaded processing improvements

<!--
The Extraordinary Performance Now Possible Processing Immutable DataFrames in Free-Threaded Python
Double DataFrame Row Processing Performance with Free-Threaded Python
Free-Threaded Python with Immutable DataFrames Deliver Significant Performance Improvements
-->


Applying a function to each row of a DataFrame is a common operation. Such processing is embarrassingly parallelizable: with multi-core CPUs, each row can be processed independently, in parallel.

Until recently, exploiting this opportunity was not possible with Python. Multi-threaded function application is throttled by the Global Interpreter Lock (GIL). Alternatively, multi-processed function application, while effective when the unit of work large, is inefficient in many common applications. Subprocess interpreter overhead, as well as the cost of copying data into each subprocesses, can overwhelm performance benefits.

Python now offers a solution: with the new, "experimental free-threading build" of Python 3.13, true multi-threaded concurrency of CPU-bound operations is possible.

The performance benefits are extraordinary. [StaticFrame](https://github.com/static-frame/static-frame), leveraging free-threaded Python, can process a DataFrame in less than half the time of single-threaded processing.

For example, when using Python 3.13t (the "t" denotes the free-threaded variant) to process a 1000 by 1000 `Frame` of uniform integers, the duration falls from 21.3 ms to 7.89 ms:

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

Row-wise function application in StaticFrame uses the `iter_series(axis=1)` interface followed by either `apply()` (for single-threaded application) or `apply_pool` for multi-threaded (where `use_threads=True`) or multi-processed application.

The benefits of using free-threaded Python are consistent across a wide range of DataFrame shapes and compositions, in both MacOS and Linux, and the out-performance scales with size.

When using standard Python bound by the GIL, multi-threaded processing of CPU-bound processes often degrades performance. As shown below, the same operation goes from 17.7 ms with a single thread to almost 40 ms with multi-threading:

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

There are trade-offs: as apparent in these examples, single-threaded processing is slower on 3.13t (21.3 ms compared to 17.7 ms). There is known performance overhead in free-threaded Python; this is an active area of CPython development and improvements are expected in 3.14t and beyond.

Further, while many C-extension packages like NumPy are already offering pre-compiled binary wheels for 3.13t, the risk of thread contention or data races still exist. StaticFrame's immutable data model removes many of these concerns. Now that all core dependencies have 3.13t wheels available, StaticFrame 3.2 is ready for free-threaded Python.

<!-- StaticFrame has long leveraged immutable NumPy arrays, as well as the standard library `ThreadPoolExecutor` interfaces;  -->


## Extended DataFrame Performance Tests

Evaluating performance characteristics of a complex data structure like a DataFrame requires multiple tests. The following performance tests perform row-wise function application on nine different DataFrame types, testing all combinations of three different shapes and three different levels of type homogeneity. For a million (1e6) elements, three shapes are tested: tall (10,000 by 100), square (1,000 by 1,000), and wide (100 by 10,0000). To vary type homogeneity, three categories of synthetic data are defined: columnar (no adjacent columns have the same type), mixed (groups of four adjacent columns share the same type), and uniform (all columns are the same type). StaticFrame permits adjacent columns of the same type to be represented by two-dimensional NumPy arrays, reducing the cost of forming rows.

The same function as used above, where even numbers are identified and then summed, is used here: `lambda s: s.loc[(s % 2) == 0].sum()`. While a more efficient implementation is possible using the NumPy array directly, this function better replicates common applications where many intermediate `Series` are created.

Figure legends make clear configuration variations: when `use_threads=True`, multi-threading is used, when `use_threads=False`, multi-processing is used. As StaticFrame uses the `ThreadPoolExecutor` and `ProcessPoolExecutor` interfaces, the `max_workers` parameter defines the maximum number of threads or processes used. A `chunksize` parameter is also available but not varied in this study.

### Multi-threaded Function Application with Python 3.13t

As shown below, the performance benefits of multi-threaded processing in 3.13t are consistent across all DataFrame types tested: processing time is reduced at least by half. The optimal number of threads (the `max_workers` parameter) is smaller for tall versus wide DataFrames, presumably because the smaller rows are processed faster and do not componsate for additional thread overhead. The best out-performance is almost an order-of-magnitude improvement, from 26.4 ms to 3.5 ms, with a uniform wide DataFrame.

![Multi-threaded (3.13t, 1e6, MacOS)](https://raw.githubusercontent.com/static-frame/static-frame/1083/free-thread-perf/doc/source/articles/freethread/threads-ftp-1e6-macos.png)


Scaling to DataFrames of 100 million elements (1e8), out-performance improves modestly.

![Multi-threaded (3.13t, 1e8, MacOS)](https://raw.githubusercontent.com/static-frame/static-frame/1083/free-thread-perf/doc/source/articles/freethread/process-ftp-1e8-macos.png)


Multi-threading and processing overhead can vary greatly between platforms. In all cases, the out-performance of using free-threaded Python was consistent between MacOS and Linux, though MacOS did show marginally greater benefits. The process of 100 million elements on Linux shows similar benefits:

![Multi-threaded (3.13t, 1e8, Linux)](https://raw.githubusercontent.com/static-frame/static-frame/1083/free-thread-perf/doc/source/articles/freethread/process-ftp-1e8-linux.png)



### Multi-threaded Function Application with Python 3.13

Prior to free-threaded Python, CPU-bound concurrency with threads resulted in degraded performance. This is made clear below, where the same tests are conducted in standard Python 3.13. The one exception where performance is not degraded is again wit uniform wide DataFrames: the per row extraction cost is fast enough to still deliver a benefit.

![Multi-threaded (3.13, 1e8, MacOS)](https://raw.githubusercontent.com/static-frame/static-frame/1083/free-thread-perf/doc/source/articles/freethread/threads-np-1e6-macos.png)


### Multi-processed Function Application with Python 3.13

Prior to free-threaded Python, multi-processing was the only option for CPU-bound concurrency. However, as mentioned above, multi-processing only delivered benefits if the amount of per-process work was substantial enough to offset the high cost of the per-process interpreter and copying data between processes.

As shown here, multi-processing row-wise function application degrades performance. Each unit of work is too small to make up for multi-processing overhead.

![Multi-processed (3.13, 1e6, MacOS)](https://raw.githubusercontent.com/static-frame/static-frame/1083/free-thread-perf/doc/source/articles/freethread/process-np-1e6-macos.png)



## The Status of Free-Threaded Python

[PEP 703](https://peps.python.org/pep-0703), "Making the Global Interpreter Lock Optional in CPython", was accepted by the Python Steering Council in July of 2023 with the guidance that, in the first phase (for Python 3.13) it is marked experimental and non-default; in the second phase, it becomes non-experimental and officially supported; in the third phase, it becomes the default Python implementation.

After significant CPython development, and support by critical packages like NumPy, [PEP 779](https://peps.python.org/pep-0779), "Criteria for supported status for free-threaded Python" was accepted by the Python Steering Council in June of 2025. In Python 3.14, free-threaded Python will enter the second phase: non-experimental and officially supported.

While it is not certain when free-threaded Python will become the default, it is clear that a trajectory is set. Now is the time to invest in free-threaded support and applications.


##  Conclusion

Row-wise function application is just the beginning: group-by operations, windowed function application, and many other operations on immutable DataFrames are similarly embarrassingly parallelizable and very likely to show similar benefits.

The work to make CPython faster has had success: some state that Python 3.14 is 20% to 40% faster than Python 3.10, those performance benefits have been realized for many working with DataFrames, where performance is largely bound within C-extensions (be it NumPy, Arrow, or other libraries). As shown here, free-threaded Python permits more, from 50% to 90% reduction in time. The combined results of these improvements will be significant.




<!-- Built on an immutable data model, already exposing interfaces for parallel function application, and now offering free-threaded compatible wheel dependencies, StaticFrame is ready now to take advantage of concurrency. -->



<!-- Finally, mutable DataFrames, such as those provided by Pandas, expose opportunities for data races. -->




<!-- Representing each row with a Series, expressive operations can be defined to reduce the DataFrame to Series.  -->

<!-- Sometimes row-wise function application can be done more efficiently as column-wise operations, though not always. -->


