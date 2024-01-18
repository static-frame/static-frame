

# Read and Write DataFrames Faster than Parquet with StaticFrame's NPZ Serialization

# Serialize DataFrames Two to Ten Times Faster than Parquet with StaticFrame NPZ


The Apache Parquet format provides an efficient binary representation of columnar table data, with widespread use in Apache Hadoop and Spark, AWS Athena and Glue, and to serialize DataFrames in Pandas. While Parquet offers interoperability across many systems with performance far superior to text formats such as CSV or JSON, it is far from the fastest format to serialize a DataFrame.

Building on NumPy's NPY and NPZ formats, StaticFrame (an open-source DataFrame library of which I am an author) offers a high-performance alternative to Parquet. Using StaticFrame NPZ, DataFrames can be read and written two to ten times faster than Pandas Parquet.


## The Challenge of Serializing DataFrames

DataFrames are not just tables of columnar data, like those found in relational databases. In addition to columnar data, DataFrames have labelled rows and columns, and those row and column labels can be of any type or hierarchical types. Further, it is common to store metadata with the ``name`` attribute, either on the DataFrame or on the axis labels.

Parquet was originally designed to store tables of columnar data, not this full range of DataFrame characteristics. Pandas supplies this additional information by adding JSON metadata into the Parquet file.

Python pickles are capable of complete serialization of DataFrames, but are only suitable for short-term caches from trusted sources. While Pickles are fast, they can become invalid due to code changes and are widely regarded as insecure to load from untrusted sources.

An alternative to Parquet, originating in the PyArrow project, is Feather. While Feather succeeds in being faster than Parquet at serializing Pandas DataFrames, it is still slower than StaticFrame NPZ.


## Origins of NPY and NPZ Encoding

The first NumPy Enhancement Proposal (NEP), in 2007, defined the NPY format (a binary encoding of array data and metadata) and the NPZ format (zipped bundles of NPY files). By reusing the NPY format and extending the NPZ format with specialized JSON metadata, StaticFrame provides a complete DataFrame serialization format that can be read by common ZIP tools and directly supports all NumPy types.


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


