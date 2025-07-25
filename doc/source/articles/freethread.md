
# Liberating Performance with Immutable DataFrames in Free-Threaded Python

### How StaticFrame and Python 3.13t Enable True Thread-Based Concurrency


Applying a function to each row of a DataFrame is a common operation. These operations are embarrassingly parallel: each row can be processed independently. With a multi-core CPU, many rows can be processed at once.

Until recently, exploiting this opportunity in Python was not possible. Multi-threaded function application, being CPU-bound, was throttled by the Global Interpreter Lock (GIL).

Python now offers a solution: with the "experimental free-threading build" of Python 3.13, the GIL is removed, and true multi-threaded concurrency of CPU-bound operations is possible.

The performance benefits are extraordinary. Leveraging free-threaded Python, [StaticFrame](https://github.com/static-frame/static-frame) 3.2 can perform row-wise function application on a DataFrame at least twice as fast as single-threaded execution.

For example, for each row of a square DataFrame of one-million integers, we can calculate the sum of all even values with `lambda s: s.loc[s % 2 == 0].sum()`. When using Python 3.13t (the "t" denotes the free-threaded variant), the duration (measured with `ipython` `%timeit`) drops by more than 60%, from 21.3 ms to 7.89 ms:

```python
# Python 3.13.5 experimental free-threading build (main, Jun 11 2025, 15:36:57) [Clang 16.0.0 (clang-1600.0.26.6)] on darwin
>>> import numpy as np; import static_frame as sf

>>> f = sf.Frame(np.arange(1_000_000).reshape(1000, 1000))
>>> func = lambda s: s.loc[s % 2 == 0].sum()

>>> %timeit f.iter_series(axis=1).apply(func)
21.3 ms ± 77.1 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)

>>> %timeit f.iter_series(axis=1).apply_pool(func, use_threads=True, max_workers=4)
7.89 ms ± 60.1 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

Row-wise function application in StaticFrame uses the `iter_series(axis=1)` interface followed by either `apply()` (for single-threaded application) or `apply_pool()` for multi-threaded (`use_threads=True`) or multi-processed (`use_threads=False`) application.

The benefits of using free-threaded Python are robust: the outperformance is consistent across a wide range of DataFrame shapes and compositions, is proportional in both MacOS and Linux, and positively scales with DataFrame size.

When using standard Python with the GIL enabled, multi-threaded processing of CPU-bound processes often degrades performance. As shown below, the duration of the same operation in standard Python increases from 17.7 ms with a single thread to almost 40 ms with multi-threading:

```python
# Python 3.13.5 (main, Jun 11 2025, 15:36:57) [Clang 16.0.0 (clang-1600.0.26.6)]
>>> import numpy as np; import static_frame as sf

>>> f = sf.Frame(np.arange(1_000_000).reshape(1000, 1000))
>>> func = lambda s: s.loc[s % 2 == 0].sum()

>>> %timeit f.iter_series(axis=1).apply(func)
17.7 ms ± 144 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

>>> %timeit f.iter_series(axis=1).apply_pool(func, use_threads=True, max_workers=4)
39.9 ms ± 354 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

There are trade-offs when using free-threaded Python: as apparent in these examples, single-threaded processing is slower (21.3 ms on 3.13t compared to 17.7 ms on 3.13). Free-threaded Python, in general, incurs performance overhead. This is an active area of CPython development and improvements are expected in 3.14t and beyond.

Further, while many C-extension packages like NumPy now offer pre-compiled binary wheels for 3.13t, risks such as thread contention or data races still exists.

StaticFrame avoids these risks by enforcing immutability: thread safety is implicit, eliminating the need for locks or defensive copies. StaticFrame does this by using immutable NumPy arrays (with `flags.writeable` set to `False`) and forbidding in-place mutation.

## Extended DataFrame Performance Tests

Evaluating performance characteristics of a complex data structure like a DataFrame requires testing many types of DataFrames. The following performance panels perform row-wise function application on nine different DataFrame types, testing all combinations of three shapes and three levels of type homogeneity.

For a fixed number of elements (e.g., 1 million), three shapes are tested: tall (10,000 by 100), square (1,000 by 1,000), and wide (100 by 10,0000). To vary type homogeneity, three categories of synthetic data are defined: columnar (no adjacent columns have the same type), mixed (groups of four adjacent columns share the same type), and uniform (all columns are the same type). StaticFrame permits adjacent columns of the same type to be represented as two-dimensional NumPy arrays, reducing the costs of column transversal and row formation. At the uniform extreme, an entire DataFrame can be represented by one two-dimensional array. Synthetic data is produced with the [frame-fixtures](https://github.com/static-frame/frame-fixtures) package.

The same function is used: `lambda s: s.loc[s % 2 == 0].sum()`. While a more efficient implementation is possible using NumPy directly, this function approximates common applications where many intermediate `Series` are created.

Figure legends document concurrency configuration. When `use_threads=True`, multi-threading is used; when `use_threads=False`, multi-processing is used. StaticFrame uses the `ThreadPoolExecutor` and `ProcessPoolExecutor` interfaces from the standard library and exposes their parameters: the `max_workers` parameter defines the maximum number of threads or processes used. A `chunksize` parameter is also available, but is not varied in this study.


### Multi-Threaded Function Application with Free-Threaded Python 3.13t

As shown below, the performance benefits of multi-threaded processing in 3.13t are consistent across all DataFrame types tested: processing time is reduced by at least 50%, and in some cases by over 80%. The optimal number of threads (the `max_workers` parameter) is smaller for tall DataFrames, as the quicker processing of smaller rows means that additional thread overhead actually degrades performance.


![Multi-threaded (3.13t, 1e6, MacOS)](https://raw.githubusercontent.com/static-frame/static-frame/master/doc/source/articles/freethread/threads-ftp-1e6-macos.png)


Scaling to DataFrames of 100 million elements (1e8), outperformance improves. Processing time is reduced by over 70% for all but two DataFrame types.

![Multi-threaded (3.13t, 1e8, MacOS)](https://raw.githubusercontent.com/static-frame/static-frame/master/doc/source/articles/freethread/threads-ftp-1e8-macos.png)


The overhead of multi-threading can vary greatly between platforms. In all cases, the outperformance of using free-threaded Python is proportionally consistent between MacOS and Linux, though MacOS shows marginally greater benefits. The processing of 100 million elements on Linux shows similar relative outperformance:

![Multi-threaded (3.13t, 1e8, Linux)](https://raw.githubusercontent.com/static-frame/static-frame/master/doc/source/articles/freethread/threads-ftp-1e8-linux.png)


Surprisingly, even small DataFrame's of only ten-thousand elements (1e4) can benefit from multi-threaded processing in 3.13t. While no benefit is found for wide DataFrames, the processing time of tall and square DataFrames can be reduced in half.

![Multi-threaded (3.13t, 1e4, MacOS)](https://raw.githubusercontent.com/static-frame/static-frame/master/doc/source/articles/freethread/threads-ftp-1e4-macos.png)



### Multi-Threaded Function Application with Standard Python 3.13

Prior to free-threaded Python, multi-threaded processing of CPU-bound applications resulted in degraded performance. This is made clear below, where the same tests are conducted with standard Python 3.13.

![Multi-threaded (3.13, 1e8, Linux)](https://raw.githubusercontent.com/static-frame/static-frame/master/doc/source/articles/freethread/threads-np-1e6-linux.png)


### Multi-Processed Function Application with Standard Python 3.13

Prior to free-threaded Python, multi-processing was the only option for CPU-bound concurrency. Multi-processing, however, only delivered benefits if the amount of per-process work was sufficient to offset the high cost of creating an interpreter per process and copying data between processes.

As shown here, multi-processing row-wise function application significantly degrades performance, process time increasing from two to ten times the single-threaded duration. Each unit of work is too small to make up for multi-processing overhead.

![Multi-processed (3.13, 1e6, MacOS)](https://raw.githubusercontent.com/static-frame/static-frame/master/doc/source/articles/freethread/process-np-1e6-macos.png)



## The Status of Free-Threaded Python

[PEP 703](https://peps.python.org/pep-0703), "Making the Global Interpreter Lock Optional in CPython", was accepted by the Python Steering Council in July of 2023 with the guidance that, in the first phase (for Python 3.13) it is experimental and non-default; in the second phase, it becomes non-experimental and officially supported; in the third phase, it becomes the default Python implementation.

After significant CPython development, and support by critical packages like NumPy, [PEP 779](https://peps.python.org/pep-0779), "Criteria for supported status for free-threaded Python" was accepted by the Python Steering Council in June of 2025. In Python 3.14, free-threaded Python will enter the second phase: non-experimental and officially supported. While it is not yet certain when free-threaded Python will become the default, it is clear that a trajectory is set.


##  Conclusion

Row-wise function application is just the beginning: group-by operations, windowed function application, and many other operations on immutable DataFrames are similarly well-suited to concurrent execution and are likely to show comparable performance gains.

The work to make CPython faster has had success: Python 3.14 is said to be 20% to 40% faster than Python 3.10. Unfortunately, those performance benefits have not been realized for many working with DataFrames, where performance is largely bound within C-extensions (be it NumPy, Arrow, or other libraries).

As shown here, free-threaded Python enables efficient parallel execution using low-cost, memory-efficient threads, delivering a 50% to 90% reduction in processing time, even when performance is primarily bound in C-extension libraries like NumPy. With the ability to safely share immutable data structures across threads, opportunities for substantial performance improvements are now abundant.


