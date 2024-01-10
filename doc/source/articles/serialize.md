

# Read and Write DataFrames Faster than Parquet with StaticFrame's NPZ Serialization

<!--


-->


The Apache Parquet format provides an efficient binary representation of tables of columnar data, as seen with widespread support in tools like Apache Hadoop and Spark, as well as AWS Athena and Glue. Parquet is also often used with Pandas to serialize DataFrame data in Python, though the translation is imperfect: Parquet does not directly support the full range of index and column labels used with DataFrames, nor all NumPy dtypes. Building on NumPy's NPY and NPZ formats, StaticFrame offers superior binary serialization, reading and writing large DataFrames at best an order of magnitude faster than Pandas Parquet, all while retaining full support for all types of index labels, column labels, and NumPy dtypes.


## The Challenge of Serializing DataFrames

DataFrames are not just tables of columnar data, like those found in relational databases. In addition to columnar data, DataFrames have labelled rows and columns, and those row and column labels can be of any type, or hierarchical types. Further, it is common to store metadata with the ``name`` attribute, either on the table or on the axis labels. These additional features are not commonly encoded in formats like CSV, TSV, Excel, HDF5, and Parquet, all of which are built on the model of a table of columnar data with string column labels.

As Pickles completely serialize complex Python objects, they offer an alternative. While using Pickles to serialize DataFrames is extraordinarily fast (as will be shown below), Pickles can become invalid due to code changes and are widely regarded as insecure to load from untrusted sources.


## Origins of NPY and NPZ Encoding

The first NumPy Enhancement Proposal (NEP), in 2007, defined the NPY format (a binary encoding of array data and metadata) and the NPZ format (zipped bundles of NPY files). By reusing the NPY format and extending the NPZ format with specialized JSON metadata, StaticFrame can completely encode and decode DataFrames materially faster than Pandas Parquet.

## DataFrame Performance Comparisons

Before describing how a DataFrame is encoded using the NPY and NPZ formats, lets look at the read / write performance.

Countless publications offer DataFrame performance comparisons using just a single DataFrame. This is insufficient and misleading, as both the shape of the DataFrame, as well as the degree of columnar type heterogeneity, make a huge difference in performance characteristics.

To avoid this problem, I present nine performance results across these two dimensions: shape (tall, square, and wide) and columnar heterogeneity (columnar, mixed, and uniform).

### Write Performance of StaticFrame NPZ versus Pandas Parquet



### Read Performance of StaticFrame NPZ versus Pandas Parquet



## Encoding a DataFrame in NPY and NPZ

A StaticFrame DataFrame can be thought of as a collection of 1D and 2D arrays and metadata. The arrays represent columnar values as well as variable-depth index and column labels. The metadata represents index and column types as well as ``name`` attributes values.



### Encoding Arrays in NPY



### Encoding DataFrame Metadata in JSON



# Size on Disk



# Conclusion

