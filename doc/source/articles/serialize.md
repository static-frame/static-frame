

# Read and Write DataFrames Faster than Parquet with StaticFrame's NPZ Serialization

# Serialize DataFrames Two to Ten Times Faster than Parquet with StaticFrame NPZ

<!--


-->


The Apache Parquet format provides an efficient binary representation of tables of columnar data, as seen with widespread use in Apache Hadoop and Spark, as well as AWS Athena and Glue. Parquet is also often used with Pandas to serialize DataFrame data, though the translation is imperfect: Parquet does not directly support the full range of index and column labels used with DataFrames, nor all NumPy dtypes. Building on NumPy's NPY and NPZ formats, StaticFrame (an open-source DataFrame library of which I am an author) offers complete, high-performance binary serialization. Using StaticFrame NPZs, large DataFrames can be read and written two to ten times faster than Pandas Parquet, all while retaining support for all types of index labels, column labels, and NumPy dtypes.


## The Challenge of Serializing DataFrames

DataFrames are not just tables of columnar data, like those found in relational databases. In addition to columnar data, DataFrames have labelled rows and columns, and those row and column labels can be of any type or hierarchical types. Further, it is common to store metadata with the ``name`` attribute, either on the DataFrame or on the axis labels. Common serialization formats, including Parquet, do not support the full range of DataFrame characteristics, making serialization challenging.

Python pickles are capable of complete serialization of DataFrames, but are only suitable for short-term caches from trusted sources. While Pickles are fast, they can become invalid due to code changes and are widely regarded as insecure to load from untrusted sources.

An alternative to Parquet, originating in the PyArrow project, is Feather. While Feather succeeds in being faster than Parquet at serializing DataFrames in Pandas, it is still slower than NPZ encoding with StaticFrame.


## Origins of NPY and NPZ Encoding

The first NumPy Enhancement Proposal (NEP), in 2007, defined the NPY format (a binary encoding of array data and metadata) and the NPZ format (zipped bundles of NPY files). By reusing the NPY format and extending the NPZ format with specialized JSON metadata, StaticFrame can completely encode and decode DataFrames significantly faster than Pandas Parquet.


## DataFrame Serialization Performance Comparisons

Read / write performance will be examined before diving into the technical details of encoding a DataFrame using the NPY and NPZ formats.

Numerous publications offer DataFrame performance comparisons by testing just a single DataFrame. This is insufficient, as both the shape of the DataFrame, as well as the degree of columnar type heterogeneity, can make a significant difference in performance. To avoid this problem, I present nine performance results across two dimensions: shape (tall, square, and wide) and columnar heterogeneity (columnar, mixed, and uniform). Shape variations alter the distribution of the same number of elements between tall (e.g., 10,000 rows and 100 columns), square (e.g., 1,000 rows and columns), and wide (e.g., 100 rows and 10,000 columns) geometries. Columnar heterogeneity variations alter the diversity of types between columnar (no adjacent columns have the same type), mixed (some adjacent columns have the same type), and uniform (all columns have the same type).


### Read Performance

As data is generally read far more often then it is written, read performance is a priority. As shown for all nine DataFrames of one million elements, NPZ significantly outperforms Parquet and Feather. The chart below shows processing time, where lower bars correspond to faster performance.



### Write Performance





## Encoding a DataFrame in NPY and NPZ

A StaticFrame DataFrame can be thought of as a collection of 1D and 2D arrays and metadata. The arrays represent columnar values as well as variable-depth index and column labels. The metadata represents index and column types as well as ``name`` attributes values.



### Encoding Arrays in NPY



### Encoding DataFrame Metadata in JSON



# Size on Disk



# Conclusion


