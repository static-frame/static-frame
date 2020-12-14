

import typing as tp

import numpy as np

from static_frame.core.frame import Frame
from static_frame.core.batch import Batch

import static_frame as sf

# def func_b(frame: Frame) -> Frame:
#     for row in frame.iter_series():
#         pass
#         # if row[10] > 1000: # type: ignore
#     return frame


# def main() -> None:

#     f1 = Frame(np.arange(100000000).reshape(1000000, 100), name='a')
#     f2 = Frame(np.arange(100000000).reshape(1000000, 100), name='b')
#     f3 = Frame(np.arange(100000000).reshape(1000000, 100), name='c')
#     f4 = Frame(np.arange(100000000).reshape(1000000, 100), name='d')
#     f5 = Frame(np.arange(100000000).reshape(1000000, 100), name='e')
#     f6 = Frame(np.arange(100000000).reshape(1000000, 100), name='f')
#     f7 = Frame(np.arange(100000000).reshape(1000000, 100), name='g')
#     f8 = Frame(np.arange(100000000).reshape(1000000, 100), name='h')



#     @timer #type: ignore
#     def a1() -> None:
#         batch1 = Batch.from_frames((f1, f2, f3, f4, f5, f6, f7, f8))
#         batch2 = (batch1 * 100).sum()
#         _ = tuple(batch2.items())

#     @timer #type: ignore
#     def a2() -> None:
#         batch1 = Batch.from_frames((f1, f2, f3, f4, f5, f6, f7, f8), max_workers=6, use_threads=True)
#         batch2 = (batch1 * 100).sum()
#         post = dict(batch2.items())

#     a1()
#     a2()

#     @timer #type: ignore
#     def b1() -> None:
#         batch1 = Batch.from_frames((f1, f2, f3, f4, f5, f6, f7, f8))
#         batch2 = batch1.apply(func_b)
#         _ = tuple(batch2.items())

#     @timer #type: ignore
#     def b2() -> None:
#         batch1 = Batch.from_frames((f1, f2, f3, f4, f5, f6, f7, f8), max_workers=8, use_threads=False, chunksize=2)
#         batch2 = batch1.apply(func_b)
#         _ = tuple(batch2.items())

#     # b1()
    # b2()


def bus_batch_streaming() -> None:

    TypeIterFrameItems = tp.Iterator[tp.Tuple[str, sf.Frame]]

    # do a bunch of processing on large Frames while maintaining single frame overhead

    def origin_data() -> TypeIterFrameItems:
        for label, i in ((chr(i), i) for i in range(65, 75)): # A, B, ...
            f = sf.Frame(np.arange(100000).reshape(1000, 100) * i, name=label)
            yield label, f

    # a StoreConfig is bundle of read/write config in a single object
    config = sf.StoreConfig(include_index=True,
            include_columns=True,
            index_depth=1,
            columns_depth=1)

    # incrementally generate and write data one Frame at at ime
    sf.Batch(origin_data()).to_zip_parquet('/tmp/pq-stg-01.zip', config=config)

    # we can read the same data store into Bus
    # settting max_persist to 1 keeps only 1 frame in memory
    src_bus = sf.Bus.from_zip_parquet('/tmp/pq-stg-01.zip', max_persist=1, config=config)

    def proc_data(bus: sf.Bus) -> TypeIterFrameItems:
        for label in bus.keys():
            f = bus[label]
            f_post = f * .00001
            yield label, f_post

    # incrementally read, process data, and write data
    # note that we do not necessarily need to write this intermediate step; we might chain a number of processors together
    sf.Batch(proc_data(src_bus)).to_zip_parquet('/tmp/pq-stg-02.zip', config=config)

    # a function that reads through the derived data and produces single in-memory result
    def derive_characteristic(bus: sf.Bus) -> sf.Series:
        def gen() -> tp.Iterator[tp.Tuple[str, float]]:
            for label in bus.keys():
                f = bus[label]
                yield label, f.mean().mean()

        return sf.Series.from_items(gen())

    post_bus = sf.Bus.from_zip_parquet('/tmp/pq-stg-02.zip', max_persist=1, config=config)
    print(derive_characteristic(post_bus))

    # <Series>
    # <Index>
    # A        32.499674999999996
    # B        32.99967
    # C        33.49966500000001
    # D        33.999660000000006
    # E        34.499655000000004
    # F        34.99965
    # G        35.49964500000001
    # H        35.99964000000001
    # I        36.499635000000005
    # J        36.999629999999996
    # <<U1>    <float64>

    # if we have family of functions that process items pairs that do not need random access, we can chain them together without writing to an intermediary

    def proc_data_alt(items: TypeIterFrameItems) -> TypeIterFrameItems:
        for label, f in items:
            f_post = f * .00001
            yield label, f_post

    def derive_characteristic_alt(items: TypeIterFrameItems) -> sf.Series:
        def gen() -> tp.Iterator[tp.Tuple[str, float]]:
            for label, f in items:
                yield label, f.mean().mean()
        return sf.Series.from_items(gen())

    # now we can use a Batch to get sequential processing, and start with the origin data set
    batch = sf.Batch.from_zip_parquet('/tmp/pq-stg-01.zip', config=config)
    print(derive_characteristic_alt(proc_data_alt(batch.items())))

    # <Series>
    # <Index>
    # A        32.499674999999996
    # B        32.99967
    # C        33.49966500000001
    # D        33.999660000000006
    # E        34.499655000000004
    # F        34.99965
    # G        35.49964500000001
    # H        35.99964000000001
    # I        36.499635000000005
    # J        36.999629999999996
    # <<U1>    <float64>


def bus_batch_demo() -> None:
    # field definitions
    # https://www.ndbc.noaa.gov/measdes.shtml

    # full data set
    # https://www.ndbc.noaa.gov/view_text_file.php?filename=46222h2018.txt.gz&dir=data/historical/stdmet/


    # multi-table storage: XLSX, HDF5, SQLite, zipped bundles
    # how can we easily read, write, and process these

    # Batch and the Bus: two series-like containers of Frames
    # Frames can have different shapes and types: not a Panel
    # Bus: random access to Frame, writing to multi-table formats, lazy loading
    # Batch: lazy, sequential processor of Frame, with support for reading/writing to multi-table formats

    sp2018 = sf.Frame.from_tsv('/tmp/san_pedro.txt')

    sp2018 = sf.Frame.from_tsv('/tmp/san_pedro.txt', dtypes={'datetime': 'datetime64[m]'})

    sp2018.shape
    # (16824, 5)

    sp2018 = sp2018.set_index('datetime', index_constructor=sf.IndexMinute, drop=True)

    # sp2018.loc['2018-01'].shape
    # (1419, 4)

    yms = sf.IndexYearMonth.from_year_month_range('2018-01', '2018-12')

    sp_per_month = [sp2018.loc[ym].rename(str(ym)) for ym in yms]

    [f.shape for f in sp_per_month]
    # [(1419, 4), (1295, 4), (1460, 4), (1411, 4), (1431, 4), (1418, 4), (1471, 4), (1160, 4), (1426, 4), (1471, 4), (1420, 4), (1442, 4)]

    sp_per_month[0].to_xlsx('/tmp/sp_2018_01.xlsx')


    # [TALK] introduce Bus: how to create a multi-sheet workbook
    # pandas: pd.ExcelWriter and pd.HDFStore

    sf.Bus.interface # from frames


    b1 = sf.Bus.from_frames(sp_per_month)
    b1.to_xlsx('/tmp/sp_2018.xlsx')


    # [TALK] to do this in a single line:
    # but note that this will full-create the Bus before writing

    sf.Bus.from_frames(sp2018.loc[ym].rename(str(ym)) for ym in yms).to_xlsx('/tmp/sp_2018.xlsx')


    # [TALK] what other formats can we export to
    # show interface, constructors and exporters
    b1.interace

    b1.to_sqlite('/tmp/sp_2018.sqlite')


    # [TALK] value of parquet and pickle

    b1.to_zip_pickle('/tmp/sp_2018.zip')


    # [TALK] Bus is Series like: can select using loc, iloc, etc
    # limit in that the index has to be strings: need to use as labels in output formats

    b1['2018-08']
    b1['2018-08':] #type: ignore
    b1.iloc[[0, 5, 10]]
    b1[b1.index.isin(('2018-01', '2018-09'))]


    # [TALK] What about reading: we can lazily read from these output formats
    sf.Bus.interface


    b2 = sf.Bus.from_zip_pickle('/tmp/sp_2018.zip')

    # notice that we have FrameDeferred
    # when access Frames, the simply get loaded as the are selected:

    b2['2018-08']
    b2 # show only one Frame loaded

    b2.iloc[[0, 5, 10]]
    b2 # show we have loaded additional frames


    # what if you have 600 Frames and you do not want the stored frames to all load
    b3 = sf.Bus.from_zip_pickle('/tmp/sp_2018.zip', max_persist=2)

    for label in b3.index:
        print(b3[label].shape)
        print(b3)


    #---------------------------------------------------------------------------
    # there are other situations where we need to deal with multiple Frames at as as single unit

    import pandas as pd
    df = pd.read_csv('/tmp/san_pedro.txt', sep='\t')
    df['year_mo'] = df['datetime'].apply(lambda d: d[:7])

    # what happens when we do a group-by in Pandas

    df.groupby('year_mo')
    # <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7fb3263e7090>

    # can iterate
    [k for k, v in df.groupby('year_mo')]
# ['2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06', '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12']

    # can take the max of each group in one line
    df.groupby('year_mo')['WVHT'].max()

    # how does this work: some operations are deferred, some result in containers
    df.groupby('year_mo')['WVHT']
    # <pandas.core.groupby.generic.SeriesGroupBy object at 0x7fb32e1b7fd0>

    # but what if we want to do the same thing with Frames we created without a groupby, or by some other means?


    #---------------------------------------------------------------------------
    # the Batch

    batch1 = sf.Batch.from_frames(sp_per_month)

    # can call to_frame to realize all data with a hierarchical index:
    sf.Batch.from_frames(sp_per_month).to_frame()


    # can do the same thing as pandas, but have to explicitly call to_frame() to get a result
    batch1['WVHT'].max().to_frame()


    # Batch are like generators: one-time use, one-time iteration. If we look at batch1 again, we see it is empty
    batch1

    # the reason we have to explicitly call to_frame as that there is no limit on how many operations you can do on the Batch; all are deferred until final iteration or `to_frame` call
    # here, we convert meters to feet before printing getting result

    (sf.Batch.from_frames(sp_per_month)['WVHT'].max() * 3.28084).to_frame()


    # what if we want to do this on a group by?
    # SF's iter_group is simply an iterator of groups, we can feed it into a Batch


    # lets create a fresh FrameGO and add a year mo column
    sp2018 = sf.FrameGO.from_tsv('/tmp/san_pedro.txt', dtypes={'datetime': 'datetime64[m]'})
    sp2018 = sp2018.set_index('datetime', index_constructor=sf.IndexMinute, drop=False)
    sp2018['year_mo'] = sp2018['datetime'].astype('datetime64[M]') #type: ignore

    sp2018.iter_group_items('year_mo')

    # we can feed the iterator of pairs of label, frame to Batch, then process
    sf.Batch(sp2018.iter_group_items('year_mo'))['WVHT'].max().to_frame() #type: ignore


    # any time we have iterators of pairs of label, Frame, we can use a Batch
    # for example, what if we want to iterate on windows

    tuple(sp2018.iter_window_items(size=100))[0]

    # We can feed those same pairs into a Batch and get the rolling mean

    sf.Batch(sp2018.iter_window_items(size=100))['WVHT'].mean().to_frame()

    # Again, we can convert meters to feet:

    (sf.Batch(sp2018.iter_window_items(size=100))['WVHT'].mean() * 3.28084).to_frame()

    # These operations are pretty fast, but we can potential optimize them by using multi threading or multi processing

    (sf.Batch(sp2018.iter_window_items(size=100), max_workers=6, use_threads=True)['WVHT'].mean() * 3.28084).to_frame()


    # there is also a generic apply method to perform arbitrary functions
    sf.Batch(sp2018.iter_window_items(size=100, step=100)).apply(lambda f: f.loc[f['DPD'].loc_max(), ['WVHT', 'DPD']]).to_frame()


    # what if we want to write read or write from a multi-table format
    # because the Batch is a lazy sequential processor, this is actually a pipeline that is processing one table at time
    # memory overhead is one table at a time


    sf.Batch.from_xlsx('/tmp/sp_2018.xlsx').apply(lambda f: f.assign['WVHT'](f['WVHT'] * 3.28084)).to_xlsx('/tmp/sp_2018_ft.xlsx')




if __name__ == '__main__':
    bus_batch_streaming()