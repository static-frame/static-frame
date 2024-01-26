

<!-- # Read and Write DataFrames Faster than Parquet with StaticFrame NPZ Serialization -->

# Serialize DataFrames Two to Ten Times Faster than Parquet with StaticFrame NPZ


The Apache Parquet format provides an efficient binary representation of columnar table data, as seen with widespread use in Apache Hadoop and Spark, AWS Athena and Glue, and Pandas DataFrame serialization. While Parquet offers interoperability across many systems with performance superior to text formats (such as CSV or JSON), it is as much as ten times slower than NPZ, an alternative DataFrame serialization format introduced in StaticFrame [https://github.com/static-frame/static-frame].

StaticFrame (an open-source DataFrame library of which I am an author) builds upon NumPy NPY and NPZ formats to offer this high-performance alternative to Parquet. The NPY format (a binary encoding of array data) and the NPZ format (zipped bundles of NPY files) were defined in the first NumPy Enhancement Proposal in 2007 [https://numpy.org/neps/nep-0001-npy-format.html]. By extending the NPZ format with specialized JSON metadata, StaticFrame provides a complete DataFrame serialization format that supports all NumPy dtypes.

<!-- As NPY is still the "default" binary representation of NumPy array data, underlying array data in StaticFrame NPZs are directly readable within NumPy. -->


## The Challenge of Serializing DataFrames

DataFrames are not just collections of columnar data with string column labels, such as found in relational databases. In addition to columnar data, DataFrames have labelled rows and columns, and those row and column labels can be of any type or (with hierarchical labels) many types. Further, it is common to store metadata with a ``name`` attribute, either on the DataFrame or on the axis labels.

As Parquet was originally designed just to store collections of columnar data, the full range of DataFrame characteristics is not directly supported. Pandas supplies this additional information by adding JSON metadata into the Parquet file.

As Parquet was designed to support a minimal selection of types, the full range of NumPy dtypes is not directly supported. For example, Parquet does not natively support unsigned integers or any datetime or timedelta types.

While Python pickles are capable of efficiently serializing DataFrames and NumPy arrays, they are only suitable for short-term caches from trusted sources. While Pickles are fast, they can become invalid due to code changes and are insecure to load from untrusted sources.

Another alternative to Parquet, originating in the Arrow project, is Feather. While Feather supports all Arrow types and succeeds in being faster than Parquet, it is still at least two times slower at reading DataFrames than NPZ.

Parquet and Feather support compression to reduce file size on disk. Parquet defaults to using "snappy" compression, while Feather default to "lz4". As the NPZ format prioritizes performance, it does not yet support compression. As will be shown below, NPZ outperforms both compressed and uncompressed Parquet and Feather files. File size comparisons will be provided below.


## DataFrame Serialization Performance Comparisons

First, read and write performance will be examined; second, the details of encoding a DataFrame with NPY and NPZ will be described.

Numerous publications offer DataFrame performance comparisons by testing just one or two data sets[e.g. https://ursalabs.org/blog/2020-feather-v2/]. This is insufficient, as both the shape of the DataFrame and the degree of columnar type heterogeneity make a significant difference in performance.

To avoid this deficiency, I present nine performance results across two dimensions of synthetic fixtures: shape (tall, square, and wide) and columnar heterogeneity (columnar, mixed, and uniform). Shape variations alter the distribution of elements between tall (e.g., 10,000 rows and 100 columns), square (e.g., 1,000 rows and columns), and wide (e.g., 100 rows and 10,000 columns) geometries. Columnar heterogeneity variations alter the diversity of types between columnar (no adjacent columns have the same type), mixed (some adjacent columns have the same type), and uniform (all columns have the same type).

The `frame-fixtures` library defines a domain-specific language to create deterministic but randomly-generated DataFrames for testing; the nine variations of DataFrames are generated with this tool.

To demonstrate the interfaces evaluated, the following IPython session performs a basic write performance test using `%time`. StaticFrame NPZ writes this square, homogenous DataFrame almost six times faster than Parquet.

```python
>>> import numpy as np
>>> import static_frame as sf
>>> import pandas as pd

>>> array = np.random.random_sample((10_000, 10_000))

>>> f1 = sf.Frame(array)
>>> %time f1.to_npz('/tmp/frame.npz')
CPU times: user 715 ms, sys: 394 ms, total: 1.11 s
Wall time: 1.23 s

>>> df1 = pd.DataFrame(array)
>>> %time df1.to_parquet('/tmp/df.parquet', compression=None)
CPU times: user 5.87 s, sys: 790 ms, total: 6.66 s
Wall time: 6.81 s
```

Plotted performance tests, as shown below, extend this basic approach by using ``frame-fiuxtures`` for systematic variation of shape and type heterogeneity, and average results over ten iterations.


### Read Performance

As data is generally read more often then it is written, read performance is a priority. As shown for all nine DataFrames of one million (1e+06) elements, NPZ significantly outperforms Parquet and Feather with every fixture. NPZ read performance is nearly ten times faster than compressed Parquet. For example, with the Uniform Tall fixture, compressed Parquet reading is 21 ms compared to 1.5 ms with NPZ.

The chart below shows processing time, where lower bars correspond to faster performance.

![Read performance 1e6](serialize/serialize-read-linux-1e6.png "1e6 Read Performance")


This impressive NPZ performance is retained with scale. Moving to 100 million (1e+08) elements, NPZ continues to be at least twice as fast as Parquet and Feather, regardless of if compression is used.

![Read performance 1e6](serialize/serialize-read-linux-1e8.png "1e8 Read Performance")



### Write Performance

As data is generally read more often then it is written, write performance is secondary. Nonetheless, NPZ outperforms Parquet (both compressed and uncompressed) in all write scenarios. For example, with the Uniform Square fixture, compressed Parquet writing is 200 ms compared to 18.3 ms with NPZ.


![Write performance 1e6](serialize/serialize-write-linux-1e6.png "1e6 Write Performance")


As with read performance, NPZ write performance is retained with scale. Moving to 100 million (1e+08) elements, NPZ continues to be at least twice as fast as Parquet, regardless of of if compression is used or not. Feather write performance (both compressed and uncompressed) out-performs NPZ in a few scenarios, but NPZ write performance is generally comparable to Feather (and in a few scenarios faster).


![Write performance 1e6](serialize/serialize-write-linux-1e8.png "1e8 Write Performance")


### File Size

As shown below for 1e6 and 1e8 element DataFrames, uncompressed NPZ is generally equal in size on disk to uncompressed Feather. This size is almost always smaller than uncompressed Parquet.

![Size 1e6](serialize/perf-space-1e6.png "1e6 Write Performance")

![Size 1e8](serialize/perf-space-1e8.png "1e8 Write Performance")


## Encoding a DataFrame in NPY and NPZ

StaticFrame stores DataFrame data as a collection of 1D and 2D NumPy arrays. Arrays represent columnar values as well as variable-depth index and column labels. The metadata stores index and column types as well as ``name`` attributes.

DataFrame data can be represented by the following diagram, which isolates arrays, array types, component types, and component names.

![NPY Encoding](serialize/frame.png "NPY Encoding")

Those comoponents can be mapped, by color, to elements of text-based representation of a ``Frame``.

![NPY Encoding](serialize/frame-repr-overlay.png "NPY Encoding")


### Encoding an Array in NPY

An NPY stores an n-dimensional NumPy array as a binary file with six components: a "magic" prefix, a version number, a header length and header (where the header is a string representation of a Python dictionary), and padding followed buy raw raw array byte data. These components are shown below with a sample NPY file representing a three-element binary array.

![NPY Encoding](serialize/npy-components.png "NPY Encoding")

As an NPY file can encode any n-dimensional array, large 2D arrays can be loaded from contiguous byte data, providing excellent performance when multiple columns are represented by single array.

### Encoding Arrays in NPZ

The NPZ format is simply an (often uncompressed) ZIP file containing one or more NPY files.

As shown below, the values arrays, as well as index and column label arrays, are stored as NPY files inside a StaticFrame NPZ:

![NPZ Array Storage](serialize/frame-to-npz.png "NPZ Array Storage")

While NPY files can encode all NumPy array types, additional data is needed to build a DataFrame from those arrays. For example, indices have types independent of their underlying array. This information is stored in the ``__meta__.json`` file included in NPZ zip files.

![NPZ Metadata Storage](serialize/frame-to-meta.png "NPZ Metadata Storage")



# Conclusion

The performance of DataFrame serialization is critical to many applications. While Parquet has widespread support, its generality compromises type specificity and, as shown with StaticFrame NPZ, performance.
