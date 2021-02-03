

# The Bus, Batch, & Quilt: Abstractions for Working with Collections of DataFrames


It is common in DataFrame processing routines to work with collections of tables. Examples include a multi-year dataset with a single table per year, historical stock data with a table per stock, or data from multiple sheets in an XLSX file. This presentation introduces three novel containers for working with such collections of DataFrames: the Bus, Batch, and Quilt.

While hierarchical indices can be used to bring multiple tables into a single DataFrame, this may not be practical for performance: efficient data handling may require having only a few tables loaded at a time, and hierarchical indices incur overhead. In other situations, heterogenous schemas may not permit any reasonable combination into a single table.

The Bus, Batch, and Quilt provide distinct abstractions for different usage contexts. Each offers significant performance opportunities for flexible memory usage and parallel processing. As a family of multi-frame containers, they define three fundamental ways of working with collections of DataFrames.

All three containers provide identical interfaces for reading from, and writing to, multi-table storage formats such as XLSX, SQLite, HDF5, or zipped containers of pickle, Parquet, or delimited-text files. This uniformity permits sharing the same data store for different usage contexts.

These tools evolved from the context of my work: processing financial data and modelling investment systems. There, data sets are naturally partitioned by date or characteristic. For historical simulations, the data needed can be large. The Bus, Batch, and Quilt have provided convenient and efficient tools for this domain, without the overhead of moving to multi-machine architectures like Dask.

While these containers are implemented in the Python StaticFrame package (a Pandas alternative that offers immutable DataFrames), the abstractions are useful for application in any DataFrame or table processing library. StaticFrame calls DataFrames simply "Frames," and that convention will be used here.


## Sample Data


To be transparent and concise, code examples will begin with a small dataset of five tables: the Kaggle "World Happiness Report". To demonstrate an example at scale, code examples will conclude with a dataset of over seven thousand tables from the Kaggle "Huge Stock Market Dataset".

The "World Happiness Report" dataset consists of five tables: a table for each of years 2015 to 2019, each table defining for each country various characteristics. The file "archive.zip" is available at https://www.kaggle.com/unsdsn/world-happiness.

After creating a ``StoreConfig`` instance to specify table and archive characteristics, the Bus can directly load the zip. A critical feature of the Bus is demonstrated by this example: the Bus is lazy, only loading data from the zip when that data is accessed. The container is initialized with placeholder objects named ``FrameDeferred``.

    >>> import static_frame as sf
    >>> config = sf.StoreConfig(index_depth=1, label_decoder=int, label_encoder=str)
    >>> bus = sf.Bus.from_zip_csv('archive.zip', config=config)
    >>> bus
    <Bus>
    <Index>
    2015    <FrameDeferred>
    2016    <FrameDeferred>
    2017    <FrameDeferred>
    2018    <FrameDeferred>
    2019    <FrameDeferred>
    <int64> <object>


While representing the same kind of data, the component tables in this archive have inconsistent columns. We can normalize each table to four columns and assign an index with the following Frame-processing function:

    def normalize(f: sf.Frame) -> sf.Frame:
        fields = ['country', 'score', 'rank', 'gdp']

        def relabel(label):
            for field in fields:
                if field in label.lower():
                    return field
            return label

        return f.relabel(columns=relabel)[fields].set_index(fields[0], drop=True)

Already, we need a tool to load and process each Frame in the zip one at a time: this is what the Batch is designed to do. After reloading the zipped data, the ``apply`` method can be used to call ``normalize`` on each Frame, storing the results in a new Bus.

    >>> bus = sf.Batch.from_zip_csv('archive.zip', config=config).apply(normalize).to_bus()


While zip archives of CSV files are convenient, they are far less efficient than zip archives of parquet or pickle files.


## Relation to Other Containers


The Bus can be thought of as a Series (or an ordered dictionary) of Frames, permitting random access by label. When reading from a file store, Frames are loaded lazily: only when a Frame is accessed is it loaded into memory, and the Bus can be configured to only hold strong references to a limited number of Frames defined by the ``max_persist`` argument. This permits limiting the total memory loaded by the Bus.

As the Bus supports reading from and writing to XLSX and HDF5 (as well as other formats), it provides the functionality of the Pandas ``ExcelWriter`` and ``HDFStore`` interfaces, but with a more general and consistent interface.

The Batch can be thought of as a generator-like iterator of pairs of label and Frame. Pairs are only iterated when creating a composite Frame with the ``to_frame()`` method, or using dictionary-like methods such as ``keys()``, ``items()``, or ``values``. The Batch exposes nearly the entire Frame interface; method calls (and even operator applications) are composed and deferred in a newly returned Batch, composing operations upon the stored iterator.

The Batch is related to the Pandas ``DataFrameGroupBy`` and ``Rolling`` objects: interfaces that, after configuring a group-by or rolling window iterable, expose function application on those groups or windows. The Batch generalizes this functionality, supporting those cases as well as any iterator of labels and Frames.

The Quilt can be thought of as a Frame built from many smaller Frames, and aligned either vertically or horizontally. This larger frame is not eagerly concatenated; rather, Frames are accessed from a contained Bus as needed, providing a lazy, "virtual concatenation" of tables along an axis. The Quilt exposes a subset of the Frame interface. Contained Frames are read from a Bus which can be configured with the ``max_persist`` argument to limit the total number of Frames held in memory. Such explicit memory management can permit doing operations on a virtual Frame that might not be possible to load into memory.


## Characteristics

These abstractions can be compared in terms of shape, interface, and iteration characteristics. The Bus and Batch are one-dimensional collections of Frames; the Quilt presents a single, two-dimensional Frame. While the shape of the Bus and the (iterated) Batch is the number of Frames, the shape of the Quilt depends on its contained Frames and its axis of orientation. While the Bus exposes a Series-like interface, the Batch and Quilt expose a Frame-like interface, operating on individual Frames or the virtually concatenated Frame, respectively. While each container is iterable, only the Batch is an iterator: its length cannot be known until iteration is completed.

**For n Frame of shape (x, y)**
|                      |Bus    |Batch |Quilt             |
|----------------------|-------|------|------------------|
|ndim                  |1      |1     |2                 |
|shape                 |(n,)   |(n,)  |(xn, y) or (x, yn)|
|Approximate Interface |Series |Frame |Frame             |
|Iterable              |True   |True  |True              |
|Iterator              |False  |True  |False             |




## Initialization


Any iterable of pairs of label and Frame can be used to construct these containers. An ``items()`` method, such as the one found on the Bus or a dictionary, provides such an iterable.

As the Batch is a generator-like iterator, it can only be iterated once before it is exhausted. For this reason a fresh Batch will be created for each example below. The following example demonstrates that, just like a generator, a Batch can only be iterated once:

    >>> batch = sf.Batch(bus.items())
    >>> len(tuple(batch))
    5
    >>> len(tuple(batch))
    0

A Quilt can be initialized with a Bus instance, and requires specification of which axis to orient on, either vertically (axis 0) or horizontally (axis 1). Additionally, a Quilt must define a Boolean value for ``retain_labels``: if True, Frame labels are retained as the outer labels in a hierarchical index along the axis of virtual concatenation. If ``retain_labels`` is False, all labels must be unique.

    >>> quilt = sf.Quilt(bus, axis=0, retain_labels=True)
    >>> [f.shape for f in bus.values]
    [(158, 3), (157, 3), (155, 3), (156, 3), (156, 3)]
    >>> quilt.shape
    (782, 3)


The Bus, Batch, and Quilt all share the same file-based constructors and exporters, such as ``from_zip_csv()`` (shown above) or ``from_xlsx()``; each constructor has a corresponding exporter, e.g., ``to_zip_csv()`` or ``to_xlsx()``, respectively, permitting round-strip reading and writing, or conversion from on format to another. The following table summarize the file-based constructors and exporters available.

**File-Based Constructors & Exporters**
|Constructor      |Exporter      |
|-----------------|--------------|
|from_zip_tsv     |to_zip_tsv    |
|from_zip_csv     |to_zip_csv    |
|from_zip_pickle  |to_zip_pickle |
|from_zip_parquet |to_zip_parquet|
|from_xlsx        |to_xlsx       |
|from_sqlite      |to_sqlite     |
|from_hdf5        |to_hdf5       |



## Cross-Container Comparisons


Performing the same method on each container will illustrate their differences. The ``head(2)`` method, for example, returns different results with each container. The Bus, behaving as a Series, returns a new Bus consisting of the first two Frames:

    >>> bus.head(2)
    <Bus>
    <Index>
    2015    Frame
    2016    Frame
    <int64> <object>


The Batch operates on each Frame, in this case calling ``head(2)`` on each contained Frame and returning, after combination with ``to_frame()``, the top two rows from each Frame in the Bus.

    >>> sf.Batch(bus.items()).head(2).to_frame()
    <Frame>
    <Index>                      score            rank    gdp              <<U29>
    <IndexHierarchy>
    2015             Switzerland 7.587            1       1.39651
    2015             Iceland     7.561            2       1.30232
    2016             Denmark     7.526            1       1.44178
    2016             Switzerland 7.509            2       1.52733
    2017             Norway      7.53700017929077 1       1.61646318435669
    2017             Denmark     7.52199983596802 2       1.48238301277161
    2018             Finland     7.632            1       1.305
    2018             Norway      7.594            2       1.456
    2019             Finland     7.769            1       1.34
    2019             Denmark     7.6              2       1.383
    <int64>          <<U24>      <float64>        <int64> <float64>


The Quilt represents the contained Frames as if they were a single, contiguous Frame. Calling ``head(2)`` returns the first two rows of that virtual frame, labelled with a hierarchical index whose outer label is the Frame's label.

    >> quilt.head(2)
    <Frame: 2015>
    <Index>                               score     rank    gdp       <<U29>
    <IndexHierarchy: country>
    2015                      Switzerland 7.587     1       1.39651
    2015                      Iceland     7.561     2       1.30232
    <int64>                   <<U24>      <float64> <int64> <float64>


The ``head`` method is a pre-configured type of row selector. The full range of ``loc`` and ``iloc`` selection interfaces is supported in all containers, though specialized within each containers' dimensional context. With the Bus, the index is formed from Frame labels. Using ``loc``, we can select one or more of the contained Frames.

    >>> bus.loc[2017].tail()
    <Frame: 2017>
    <Index>                  score            rank    gdp               <<U29>
    <Index: country>
    Rwanda                   3.47099995613098 151     0.368745893239975
    Syria                    3.46199989318848 152     0.777153134346008
    Tanzania                 3.34899997711182 153     0.511135876178741
    Burundi                  2.90499997138977 154     0.091622568666935
    Central African Republic 2.69300007820129 155     0.0
    <<U24>                   <float64>        <int64> <float64>


If we want to select from each Frame individually, we can iterate through each Frame, do a selection, and then concatenate these into a new Frame:

    >>> sf.Frame.from_concat_items(((label, f.loc['Tanzania', ['score', 'rank']]) for label, f in bus.items()))
    <Frame>
    <Index>                   score            rank      <<U29>
    <IndexHierarchy>
    2015             Tanzania 3.781            146.0
    2016             Tanzania 3.666            149.0
    2017             Tanzania 3.34899997711182 153.0
    2018             Tanzania 3.303            153.0
    2019             Tanzania 3.231            153.0
    <int64>          <<U8>    <float64>        <float64>


As already seen, the Batch is desigend for operating on each Frame and concatenating the results. Additionally, the Batch offers a more compact interface than that shown above using the Bus. We can select row and column values from within each contained Frame and bring the resultd together under their individual Frame labels.

    >>> sf.Batch(bus.items()).loc['Tanzania', ['score', 'rank']].to_frame()
    <Frame>
    <Index> score            rank      <<U29>
    <Index>
    2015    3.781            146.0
    2016    3.666            149.0
    2017    3.34899997711182 153.0
    2018    3.303            153.0
    2019    3.231            153.0
    <int64> <float64>        <float64>


As a virtual concatenation of Frames, the Quilt permits selection as if from a single Frame. As shown below, a hierarchical selection on the inner label "Tanzania" brings together records for that country for all years.

    >>> quilt.loc[sf.HLoc[:, 'Tanzania'], ['score', 'rank']]
    <Frame>
    <Index>                            score            rank    <<U29>
    <IndexHierarchy: country>
    2015                      Tanzania 3.781            146
    2016                      Tanzania 3.666            149
    2017                      Tanzania 3.34899997711182 153
    2018                      Tanzania 3.303            153
    2019                      Tanzania 3.231            153
    <int64>                   <<U8>    <float64>        <int64>


These last examples demonstrate that, in some cases, the same operation can be done with the Bus, Batch, and Quilt. The difference is in the abstraction and the interface.


## Further Usage



Examples of iteration and function application demonstrate additional functionality of these containers. Just as with a Series, we can apply a function to each Frame with ``iter_element().apply()``.

    >>> bus.iter_element().apply(lambda f: 'Laos' in f.index)
    <Series>
    <Index>
    2015     True
    2016     True
    2017     False
    2018     True
    2019     True
    <int64>  <bool>


While explicit iteration and function application are available with a Bus, the Batch is specialized for this purpose, implicitly performing actions on each contained Frame. In the example below, a Batch is used to, per year, get the label of the maximum value of two fields.

    >>> sf.Batch(bus.items())[['score','gdp']].loc_max().to_frame()
    <Frame>
    <Index> score       gdp                  <<U29>
    <Index>
    2015    Switzerland Qatar
    2016    Denmark     Qatar
    2017    Norway      Qatar
    2018    Finland     United Arab Emirates
    2019    Finland     Qatar
    <int64> <<U24>      <<U24>


A similar example selects two countries from each table, sorts them by columns, and concatenates them into a Frame.

    >>> sf.Batch(bus.items()).loc[['Norway','Finland']].sort_values('rank').to_frame()
    <Frame>
    <Index>                  score            rank    gdp              <<U29>
    <IndexHierarchy>
    2015             Norway  7.522            4       1.459
    2015             Finland 7.406            6       1.29025
    2016             Norway  7.498            4       1.57744
    2016             Finland 7.413            5       1.40598
    2017             Norway  7.53700017929077 1       1.61646318435669
    2017             Finland 7.4689998626709  5       1.44357192516327
    2018             Finland 7.632            1       1.305
    2018             Norway  7.594            2       1.456
    2019             Finland 7.769            1       1.34
    2019             Norway  7.554            3       1.488
    <int64>          <<U24>  <float64>        <int64> <float64>


While per-Frame operations are idiomatic to the Batch, the Quilt permits making observations across the totality of Frames. For example, to find the maximum GDP per capita over all years, a Boolean array selecting the maximum value can be given to a ``loc`` selection.

    >>> quilt.loc[quilt['gdp'] == quilt['gdp'].max()]
    <Frame: 2018>
    <Index>                                        score     rank    gdp       <<U28>
    <IndexHierarchy: country>
    2018                      United Arab Emirates 6.774     20      2.096
    <int64>                   <<U24>               <float64> <int64> <float64>


All of the Frame's row- and column-wise iteration and function application routines, including group and window operations, are supported on the Quilt. By using a Quilt with a Bus with ``max_persist`` set to 1, for example, all rows of the Quilt can be iterated without keeping more than one Frame in memory. The following example extracts top-ranked rows from a Quilt. With an appropriately configured Bus, this can be done across large datasets with minimal memory overhead.


    >>> rows = quilt.iter_series(axis=1).apply_iter(lambda s: s if s['rank'] <= 2 else None)
    >>> sf.Frame.from_concat(r for r in rows if r is not None)
    <Frame>
    <Index>               score            rank      gdp              <<U29>
    <Index>
    (2015, 'Switzerland') 7.587            1.0       1.39651
    (2015, 'Iceland')     7.561            2.0       1.30232
    (2016, 'Denmark')     7.526            1.0       1.44178
    (2016, 'Switzerland') 7.509            2.0       1.52733
    (2017, 'Norway')      7.53700017929077 1.0       1.61646318435669
    (2017, 'Denmark')     7.52199983596802 2.0       1.48238301277161
    (2018, 'Finland')     7.632            1.0       1.305
    (2018, 'Norway')      7.594            2.0       1.456
    (2019, 'Finland')     7.769            1.0       1.34
    (2019, 'Denmark')     7.6              2.0       1.383
    <object>              <float64>        <float64> <float64>


## Larger Data


The previous dataset, while compact, does not demonstrate some the advantages of working with larger collections of tables.

The "Huge Stock Market Dataset" dataset is a collection of over seven thousand tables, each table a time series of characteristics for a US stock. The file "archive.zip" is available at https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs

This zip archive contains subdirectories and thus cannot be directly read from a Bus. After opening the archive, we can read from the contained "Stocks" directory and create a zip pickle of the stock data for fast reading in subsequent examples. As some files are empty, we must also filter out files with no size.

    >>> import os
    >>> d = 'archive/Stocks'
    >>> fps = ((fn, os.path.join(d, fn)) for fn in os.listdir(d))
    >>> items = ((fn.replace('.us.txt', ''), sf.Frame.from_csv(fp, index_depth=1)) for fn, fp in fps if os.path.getsize(fp))
    >>> sf.Batch(items).to_zip_pickle('stocks.zip')


Creating a Bus from this new store loads zero Frames, and thus provides fast access to a subset of the data without loading anything more than is requested.

    >>> bus = sf.Bus.from_zip_pickle('stocks.zip')
    >>> bus.shape
    (7163,)
    >>> bus.status['loaded'].sum()
    0

Accessing a single Frame loads only one Frame.

    >>> bus['ibm'].tail(2)
    <Frame>
    <Index>    Open      High      Low       Close     Volume  OpenInt <<U7>
    <Index>
    2017-11-09 149.93    151.8     149.86    150.3     4776388 0
    2017-11-10 150.65    150.89    149.14    149.16    4306433 0
    <<U10>     <float64> <float64> <float64> <float64> <int64> <int64>


Extracting multiple Frames produces a new Bus that reads from the same store.

    >>> bus[['aapl', 'msft', 'goog']]
    <Bus>
    <Index>
    aapl    Frame
    msft    Frame
    goog    Frame
    <<U9>   <object>
    >>> bus.status['loaded'].sum()
    4


With a Batch we can perform operations on those Frames. The ``apply()`` method can be used to multiply volume and close price; we then extract the most recent two values:

    >>> sf.Batch(bus[['aapl', 'msft', 'goog']].items()).apply(lambda f: f['Close'] * f['Volume']).iloc[-2:].to_frame()
    <Frame>
    <Index> 2017-11-09         2017-11-10         <<U10>
    <Index>
    aapl    5175673321.5       4389543386.98
    msft    1780638040.5600002 1626767764.8700001
    goog    1283539710.3       740903319.18
    <<U4>   <float64>          <float64>


To make observations across the entire data set, we can pass the Bus to a Quilt. Here, a null slice is used to force loading all Frames at once to improve Quilt performance. The shape shows a Quilt of almost 15 million rows.

    >>> quilt = sf.Quilt(bus[:], retain_labels=True)
    >>> quilt.shape
    (14887665, 6)


With this interface we can calculate the total volume of almost seven thousand securities for a single day,

    >>> quilt.loc[sf.HLoc[:, '2017-11-10'], 'Volume'].sum()
    5520175355


A much more expensive operation provides the ticker and date of the security with the highest volume across all securities:

    >>> quilt.iloc[quilt['Volume'].iloc_max()]
    <Series: ('bac', '2012-03-07')>
    <Index>
    Open                            7.4073
    High                            7.6065
    Low                             7.3694
    Close                           7.6065
    Volume                          2423735131.0
    OpenInt                         0.0
    <<U7>                           <float64>


## Conclusion


While related tools for working with collections of Frame exist, the Bus, Batch, and Quilt provide well-defined abstractions that cover common needs in working with collections of tables. Combined with lazy loading and lazy execution, as well as support for a variety of multi-table storage formats, these tools provide valuable resources for DataFrame processing.

