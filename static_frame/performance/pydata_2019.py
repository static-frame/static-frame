


# https://www.ocregister.com/2018/12/17/big-surf-possible-flooding-and-erosion-as-massive-swell-hits-the-coast-this-week/

# field definitions
# https://www.ndbc.noaa.gov/measdes.shtml

# full data set
# https://www.ndbc.noaa.gov/view_text_file.php?filename=46222h2018.txt.gz&dir=data/historical/stdmet/


# Fitting Many Dimensions into One: The Promise of Hierarchical Indices for Data Beyond Two Dimensions

# -----------------------------------------------
import typing as tp
import os
import pickle
# import datetime
from urllib import request
from typing import NamedTuple
import datetime
import functools


import numpy as np
import static_frame as sf
import pandas as pd

from static_frame.core.util import array2d_to_tuples

from static_frame.performance.perf_test import PerfTest

#-------------------------------------------------------------------------------
#

class Buoy(NamedTuple):
    station_id: int
    name: str

BUOYS = (
    Buoy(46222, 'San Pedro'),
    Buoy(46253, 'San Pedro South'),
    Buoy(46221, 'Santa Monica Bay'),
)

def cache_buoy(prefix, active=True):
    def decorator(func):
        def wrapper(cls, buoy, year):
            fp = f'/tmp/{prefix}-{buoy.station_id}-{year}.p'
            load_source = True
            if active and os.path.exists(fp):
                with open(fp, 'rb') as f:
                    try:
                        post = pickle.load(f)
                        load_source = False
                    except ModuleNotFoundError:
                        pass
            if load_source:
                post = func(cls, buoy, year)
                with open(fp, 'wb') as f:
                    pickle.dump(post, f)
            return post
        return wrapper
    return decorator


class BuoyLoader:

    FIELD_DATETIME = 'datetime'
    FIELD_STATION_ID = 'station_id'
    FIELD_WAVE_HEIGHT = 'WVHT'
    FIELD_WAVE_PERIOD = 'DPD'	# Dominant wave period
    FIELD_WAVE_DIRECTION = 'MWD'

    COMPASS = ('N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW')
    URL_TEMPLATE = 'https://www.ndbc.noaa.gov/view_text_file.php?filename={station_id}h{year}.txt.gz&dir=data/historical/stdmet/'

    @classmethod
    def buoy_record(cls,
            line: str,
            line_number: int,
            station_id: int
            ) -> tp.Sequence[str]:
        timestamp = []

        def gen() -> tp.Iterator[tp.Union[int, str]]:
            yield cls.FIELD_STATION_ID if not line_number else station_id
            cell_pos = -1 # increment before usage
            for cell in line.split(' '):
                cell = cell.strip()
                if cell:
                    cell_pos += 1
                    if cell_pos < 5:
                        timestamp.append(cell)
                    elif cell_pos == 5:
                        yield cls.FIELD_DATETIME if not line_number else '{}-{}-{}T{}:{}'.format(*timestamp)
                        yield cell
                    else:
                        yield cell

        return tuple(gen())

    @classmethod
    def buoy_to_records(cls,
            buoy: Buoy,
            year: int,
            ) -> tp.Iterator[tp.Sequence[str]]:

        url = cls.URL_TEMPLATE.format(station_id=buoy.station_id, year=year)

        with request.urlopen(url) as response:
            raw = response.read().decode('utf-8')

        line_pos = -1 # increment before usage to allow skipped lines
        for line in raw.split('\n'):
            if not line.strip():
                continue
            line_pos += 1
            yield cls.buoy_record(line, line_pos, station_id=buoy.station_id)


    @classmethod
    def degree_to_compass(cls, degrees: np.array):
        indices = np.floor(((degrees + 12.25) % 360) / 22.5).astype(int)
        return np.array([cls.COMPASS[i] for i in indices])

    @classmethod
    @cache_buoy('sf')
    def buoy_to_sf(cls, buoy: Buoy, year: int) -> sf.Frame:
        '''
        Return a simple Frame presentation without an index.
        '''

        records = cls.buoy_to_records(buoy, year=year)
        columns = next(records)
        units = next(records)

        # can pass dtypes here, but doing it below consolidates blocks
        dtypes = {
                cls.FIELD_WAVE_HEIGHT: float,
                cls.FIELD_WAVE_PERIOD: float,
                cls.FIELD_DATETIME: np.datetime64}
        f = sf.Frame.from_records(records, columns=columns, dtypes=dtypes)

        direction = cls.degree_to_compass(f[cls.FIELD_WAVE_DIRECTION].astype(int).values)

        f = f[[cls.FIELD_STATION_ID,
                cls.FIELD_DATETIME,
                cls.FIELD_WAVE_HEIGHT,
                cls.FIELD_WAVE_PERIOD]].to_frame_go()

        f[cls.FIELD_WAVE_DIRECTION] = direction

        return f.to_frame()

    @classmethod
    @cache_buoy('df')
    def buoy_to_pd(cls, buoy: Buoy, year: int):

        records = cls.buoy_to_records(buoy, year=year)
        columns = next(records)
        units = next(records)

        df = pd.DataFrame.from_records(records, columns=columns)

        direction = cls.degree_to_compass(df[cls.FIELD_WAVE_DIRECTION].astype(int).values)

        df = df[[cls.FIELD_STATION_ID,
                cls.FIELD_DATETIME,
                cls.FIELD_WAVE_HEIGHT,
                cls.FIELD_WAVE_PERIOD]]

        df[cls.FIELD_WAVE_DIRECTION] = direction

        return df.astype({
                cls.FIELD_WAVE_HEIGHT: float,
                cls.FIELD_WAVE_PERIOD: float,
                cls.FIELD_DATETIME: np.datetime64})

    @classmethod
    def buoy_to_np(cls):
        pass

    @classmethod
    def buoy_to_xarray(cls):
        pass

#-------------------------------------------------------------------------------
# here we build different representations of the same data using different hierarchies

# things to do:
# 1. select all observations at a particular date/time
# 2. get the max values for each buoy
# 3. filter vlaues within targets to see if we find correspondence on the target date


class BuoySingleYear2D:

    @staticmethod
    def to_sf(year: int = 2018) -> sf.Frame:

        frames = []
        for buoy in BUOYS:
            f = BuoyLoader.buoy_to_sf(buoy, year)
            frames.append(f)

        f = sf.Frame.from_concat(frames,
                index=sf.IndexAutoFactory,
                )
        f = f.set_index_hierarchy(('station_id', 'datetime'),
                index_constructors=(sf.Index, sf.IndexMinute),
                drop=True)
        return f

    @staticmethod
    def to_pd(year: int = 2018):

        dfs = []
        for buoy in BUOYS:
            f = BuoyLoader.buoy_to_pd(buoy, year)
            dfs.append(f)

        df = pd.concat(dfs)

        return df.set_index(['station_id', 'datetime'])

        # this sets to datetime, but does not a
    @staticmethod
    def to_pd_dict(year: int = 2018):

        dfs = {}
        for buoy in BUOYS:
            df = BuoyLoader.buoy_to_pd(buoy, year)
            df = df.set_index('datetime')[['DPD', 'WVHT', 'MWD']]
            dfs[buoy.station_id] = df

        return dfs


    @staticmethod
    def to_pd_panel_a(year: int = 2018):
        dfs = {}
        for buoy in BUOYS:
            df = BuoyLoader.buoy_to_pd(buoy, year)
            df = df.set_index('datetime')[['DPD', 'WVHT', 'MWD']]
            dfs[buoy.station_id] = df

        return pd.Panel(dfs)

        # reindexes and adds NaN
        # ipdb> panel[:, '2018-03-01T18:58']
        #       46222  46253  46221
        # DPD   12.50    NaN    NaN
        # WVHT   0.69    NaN    NaN

    @staticmethod
    def to_pd_panel_b(year: int = 2018):
        dfs = {}
        for buoy in BUOYS:
            df = BuoyLoader.buoy_to_pd(buoy, year)
            # remove station id, set datetime as indexnns
            # df = df.set_index('datetime')[['DPD', 'WVHT']]
            df = df.set_index('datetime')[['DPD', 'WVHT']]
            dfs[buoy.station_id] = df

        return pd.Panel(dfs)


    @classmethod
    def to_xarray(cls, year: int = 2018):
        return cls.to_pd_panel(year=year).to_xarray()


    @classmethod
    def to_np(cls, year: int = 2018):
        arrays = []

        frames = [BuoyLoader.buoy_to_sf(buoy, year) for buoy in BUOYS]

        # NOTE: take union and inssert NaNs
        date_intersect = functools.reduce(lambda x, y: x & y, (set(f['datetime'].values) for f in frames))

        frames_aligned = []
        for f in frames:
            frames_aligned.append(f.loc[f['datetime'].isin(date_intersect)])

        station_ids = {}
        for idx, f in enumerate(frames_aligned):
            datetime = {d: x for x, d in enumerate(f['datetime'].values)} # store the last one
            station_id = f.loc[sf.ILoc[0], 'station_id']
            arrays.append(f[['DPD', 'WVHT']].values)
            station_ids[station_id] = idx

        # dome data found only in one
        # numpy.datetime64('2018-06-25T14:28'), numpy.datetime64('2018-01-21T13:58')
        # numpy.datetime64('2018-03-01T18:58'), numpy.datetime64('2018-10-10T20:00')
        # numpy.datetime64('2018-10-29T01:00

        # In : len(date_intersect)
        # 16465

        indices = {'station_id': station_ids, 'datetime':  datetime, 'attr': {'DPD':0, 'WVHT':1}}
        return np.array(arrays), indices

    @classmethod
    def process_sf(cls) -> None:

        fsf = cls.to_sf()

        #-----------------------------------------------------------------------
        # creating IndexHierarchy



        #-----------------------------------------------------------------------
        # getting values out

        # iterating parts of the index
        part = tuple(fsf.index.iter_label(0))

        # getting all the values at an array
        part = fsf.index.values_at_depth(0)

        # convert to a frame
        part = fsf.index.to_frame()

        # rehiearch (make dates outer, station id inner); note that changing the hiearchy order forces a reordering
        part = fsf.rehierarch((1, 0))

        # to tuples
        part = fsf.relabel_flat(index=True)

        # adding a level
        part = fsf.relabel_add_level(index='A')

        # dropping a level
        #         ipdb> fsf.relabel_drop_level(index=1)
        # *** static_frame.core.exception.ErrorInitIndex: labels (50350) have non-unique values (17034)
        # ipdb>

        #-----------------------------------------------------------------------
        # show different types of selection

        # getting a sample from a partial match
        part = fsf.loc[sf.HLoc[:, '2018-12-18T07'], 'DPD']

        # select based on partial time
        post1 = fsf.loc[sf.HLoc[:, '2018-12-18T20']]

        # getting a slice
        part = fsf.loc[sf.HLoc[:, '2018-12-18T07':'2018-12-18T08'], 'DPD']

        # dsicrete selection (this works on Pandas)
        part = fsf.loc[sf.HLoc[:, ['2018-12-18T20:00', '2018-12-18T20:30']], 'DPD']

        # can show iloc
        part = fsf.loc[sf.HLoc[:, ['2018-12-18T20:00', '2018-12-18T20:30']], sf.ILoc[-1]]



        # show getting labels with iter_label (unique values)
        # show converting to a Frame



        #-----------------------------------------------------------------------
        # doing some analysis

        # find max for givne day
        fsf.loc[sf.HLoc[:, '2018-12-18']].max()


        max_dpd = [fsf.loc[sf.HLoc[station_id], 'DPD'].loc_max() for station_id in fsf.index.iter_label(0)]
        max_wvht = [fsf.loc[sf.HLoc[station_id], 'WVHT'].loc_max() for station_id in fsf.index.iter_label(0)]

        # get the peaks of the two fields, but this does not get us to the date
        peaks = fsf.loc[fsf.index.isin(max_dpd + max_wvht)]

        # use 2 to get 1.731622836825856
        threshold_wvht = fsf.loc[:, 'WVHT'].mean() + (fsf.loc[:, 'WVHT'].std() * 2)

        # use 1 to get 15.889409302831822
        threshold_dpd = fsf.loc[:, 'DPD'].mean() + fsf.loc[:, 'DPD'].std()

        # this isolates the relevant days; but does not get 46253
        # 2 and 18 gets all
        targets = fsf.loc[(fsf.loc[:, 'WVHT'] > threshold_wvht) & (fsf.loc[:, 'DPD'] > threshold_dpd)]


        targets = targets.to_frame_go()
        targets['date'] = [d.date() for d in targets.index.values_at_depth(1)]


        targets['station_id'] = targets.index.values_at_depth(0)
        targets.iter_group(['date', 'station_id']).apply(len)

        # targets.iter_group('date').apply(lambda x: len(x))
        peaks_per_day = targets.iter_group('date').apply(len)
        print(peaks_per_day)

        def gen():
            for date, date_frame in targets.iter_group_items('date'):
                for station_id, station in date_frame.iter_group_index_items(0):
                    yield date, station_id, len(station)

        post = sf.Frame.from_records(gen(), columns=('date', 'station_id', 'count'))
        post = post.set_index_hierarchy(('date', 'station_id'),
                drop=True,
                index_constructors=(sf.IndexDate, sf.Index))

        print(post)

# <Frame: buos_single_year>
# <Index>                                       WVHT      DPD       MWD   <<U10>
# <IndexHierarchy>
# 46253                     2018-12-18 13:00:00 2.03      18.18     WSW
# 46253                     2018-12-18 13:30:00 2.01      18.18     W
# 46253                     2018-12-18 14:30:00 2.11      18.18     WSW
# <object>                  <object>            <float64> <float64> <<U3>


    @classmethod
    def process_pd_panel(cls):
        panel = cls.to_pd_panel()

#         ipdb> panel
# <class 'pandas.core.panel.Panel'>
# Dimensions: 3 (items) x 17034 (major_axis) x 2 (minor_axis)
# Items axis: 46222 to 46221
# Major_axis axis: 2018-01-01 00:00:00 to 2018-12-31 23:30:00
# Minor_axis axis: DPD to WVHT

        # can select two buoys be creating a new panel
        p2 = panel[[46222, 46221]]
        # ipdb> p2.shape
        # (2, 17034, 2)

        # all buoy data for 2018-12-18, partial selection working
        p3 = panel[:, '2018-12-18', 'WVHT']

        # ipdb> panel[:, '2018-12-18', ['WVHT', 'DPD']].mean()
        #           46222      46253      46221
        # WVHT   2.339787   1.737917   2.392917
        # DPD   16.888936  15.658125  16.677917

        # ipdb> panel[:, '2018-12-18', 'DPD'].mean()
        # 46222    16.888936
        # 46253    15.658125
        # 46221    16.677917
        # dtype: float64

        # import ipdb; ipdb.set_trace()

    @classmethod
    def process_np(cls):
        a1, indices = cls.to_np()

        # ipdb> a1.shape
        # (3, 16465, 2)
        # ipdb> indices['datetime'][np.datetime64('2018-12-18T20:30')]
        # 15848
        # ipdb> a1[:, indices['datetime'][np.datetime64('2018-12-18T20:30')], 1]
        # array([1.85, 1.97, 2.04])

        # import ipdb; ipdb.set_trace()


    @classmethod
    def process_pd_multi_index(cls):

        df = cls.to_pd()
        # selecting records for a single time across all buoys
        # df.loc[(slice(None), ['2018-12-18T20:00', '2018-12-18T20:30']), 'DPD']

        part = df.loc[pd.IndexSlice[:, ['2018-12-18T20:00', '2018-12-18T20:30']], 'DPD']
        # note that part has the whole original index, and that its display is terrible
        # show:       remove_unused_levels

        # these do not work
        # pdb> df.loc[pd.IndexSlice[:, '2018-12-18'], 'DPD']
        # *** pandas.errors.UnsortedIndexError: 'MultiIndex slicing requires the index to be lexsorted: slicing on levels [1], lexsort depth 0'

        # this works in 25.3

        # ipdb> fpd.loc[pd.IndexSlice[46221, np.datetime64('2018-12-09T07:30')], 'DPD']
        # 15.38

        # ipdb> fpd.loc[pd.IndexSlice[46221, np.datetime64('2018-12-09T07:30'):], 'DPD']
        # *** pandas.errors.UnsortedIndexError: 'MultiIndex slicing requires the index to be lexsorted: slicing on levels [1], lexsort depth 0'

        # delivers a single values when matching on the whole day: (SF gives full day?)
        # ipdb> fpd.loc[pd.IndexSlice[46221, datetime.date(2018,12,7)], 'DPD']
        # 11.76


        big = df.loc[(df.loc[:, 'WVHT'] > 2.1) & (df.loc[:, 'DPD'] > 18)]

        # import ipdb; ipdb.set_trace()


#-------------------------------------------------------------------------------
class BuoySingleYear1D:
    '''
    Fit all the data into a Series
    '''

    FIELD_ATTR = 'attr'

    @staticmethod
    def to_sf(year: int = 2018) -> sf.Frame:

        labels = []
        values = []

        for buoy in BUOYS:
            f = BuoyLoader.buoy_to_sf(buoy, year)
            for row in f.iter_series(1):
                for attr in (
                        BuoyLoader.FIELD_WAVE_HEIGHT,
                        BuoyLoader.FIELD_WAVE_PERIOD):
                    label = (row[BuoyLoader.FIELD_STATION_ID],
                            row[BuoyLoader.FIELD_DATETIME],
                            attr)
                    labels.append(label)
                    values.append(row[attr])

        index = sf.IndexHierarchy.from_labels(labels,
                index_constructors=(sf.Index, sf.IndexMinute, sf.Index))

        return sf.Series(values, index=index)



    @staticmethod
    def to_pd(year: int = 2018) -> sf.Frame:
        import pandas as pd

        labels = []
        values = []

        for buoy in BUOYS:
            df = BuoyLoader.buoy_to_pd(buoy, year)
            for _, row in df.iterrows():
                for attr in (
                        BuoyLoader.FIELD_WAVE_HEIGHT,
                        BuoyLoader.FIELD_WAVE_PERIOD):
                    label = (row[BuoyLoader.FIELD_STATION_ID],
                            row[BuoyLoader.FIELD_DATETIME],
                            attr)
                    labels.append(label)
                    values.append(row[attr])

        # display of this index is terrible
        index = pd.MultiIndex.from_tuples(labels)
        return pd.Series(values, index=index)



    @classmethod
    def process_sf(cls):

        ssf = cls.to_sf()

        # betting a two observations of both metrics at the same hour
        post = ssf[sf.HLoc[:, '2018-12-18T07', ['DPD', 'WVHT']]]
        # import ipdb; ipdb.set_trace()


        # get one field from two buoys
        post = ssf[sf.HLoc[[46222, 46221], :, 'DPD']]

        # get partial date matching:
        post = ssf[sf.HLoc[46222, '2018-12-18', 'DPD']]

        # can use a datetime object
        post = ssf[sf.HLoc[46222, datetime.datetime(2018, 12, 18), 'DPD']]


    @classmethod
    def process_pd(cls):
        import pandas as pd

        spd = cls.to_pd()

        # SHOW: does not do a hierarchical selection
        # ipdb> spd[46211]
        # 6.25

        # SHOw: cannot do two at a time
#         ipdb> spd[pd.IndexSlice[[46222, 46221], :, 'DPD']]
        # *** TypeError: '[46222, 46221]' is an invalid key


        # this works
        post = spd[pd.IndexSlice[46222, '2018-12-18T20:00', 'DPD']]


#-------------------------------------------------------------------------------
# performance tests

class SampleData:

    _store: tp.Dict[str, tp.Any] = {}


    @classmethod
    def create(cls) -> None:
        fsf = BuoyLoader.buoy_to_sf(BUOYS[0], 2018)
        fpd = BuoyLoader.buoy_to_pd(BUOYS[0], 2018)


        cls._store['array_datetime'] = fsf['datetime'].values
        cls._store['array_station_id'] = tuple(b.station_id for b in BUOYS)
        cls._store['array_attr'] = ('WVHT', 'DPD', 'MWD')

        cls._store['sf_index_datetime'] = sf.IndexMinute(cls.get('array_datetime'))
        cls._store['sf_index_station_id'] = sf.Index(cls.get('array_station_id'))
        cls._store['sf_index_attr'] = sf.Index(cls.get('array_attr'))

        cls._store['pd_index_datetime'] = pd.Index(cls.get('array_datetime'))
        cls._store['pd_index_station_id'] = pd.Index(cls.get('array_station_id'))
        cls._store['pd_index_attr'] = pd.Index(cls.get('array_attr'))

        cls._store['sf_index_2D'] = sf.IndexHierarchy.from_product(
                cls.get('array_datetime'),
                cls.get('array_station_id')
                )
        cls._store['sf_index_3D'] = sf.IndexHierarchy.from_product(
                cls.get('array_datetime'),
                cls.get('array_station_id'),
                cls.get('array_attr')
                )

        cls._store['tuple_index_2D'] = tuple(array2d_to_tuples(cls.get('sf_index_2D').values))
        cls._store['tuple_index_3D'] = tuple(array2d_to_tuples(cls.get('sf_index_3D').values))


    @classmethod
    def get(cls, key: str) -> tp.Any:
        return cls._store[key]


class IndexCreation_from_product_2D(PerfTest):
    NUMBER = 10

    @classmethod
    def pd(cls) -> None:
        labels0 = SampleData.get('pd_index_datetime')
        labels1 = SampleData.get('pd_index_station_id')
        ih = pd.MultiIndex.from_product((labels0, labels1))
        assert ih.shape[0] == 50472

    @classmethod
    def sf(cls) -> None:
        labels0 = SampleData.get('sf_index_datetime')
        labels1 = SampleData.get('sf_index_station_id')

        ih = sf.IndexHierarchy.from_product(labels0, labels1)
        assert ih.shape[0] == 50472


class IndexCreation_from_product_3D(PerfTest):
    NUMBER = 10

    @classmethod
    def pd(cls) -> None:
        labels0 = SampleData.get('pd_index_datetime')
        labels1 = SampleData.get('pd_index_station_id')
        labels2 = SampleData.get('pd_index_attr')
        ih = pd.MultiIndex.from_product((labels0, labels1, labels2))
        assert ih.shape[0] == 151416

    @classmethod
    def sf(cls) -> None:
        labels0 = SampleData.get('sf_index_datetime')
        labels1 = SampleData.get('sf_index_station_id')
        labels2 = SampleData.get('sf_index_attr')
        ih = sf.IndexHierarchy.from_product(labels0, labels1, labels2)
        assert ih.shape[0] == 151416

class IndexCreation_from_labels_2D(PerfTest):
    NUMBER = 10

    @classmethod
    def pd(cls) -> None:
        ih = pd.MultiIndex.from_tuples(tuple(SampleData.get('tuple_index_2D')))

    @classmethod
    def sf(cls) -> None:
        ih = sf.IndexHierarchy.from_labels(SampleData.get('tuple_index_2D'))


class IndexCreation_from_labels_3D(PerfTest):
    NUMBER = 10

    @classmethod
    def pd(cls) -> None:
        ih = pd.MultiIndex.from_tuples(tuple(SampleData.get('tuple_index_3D')))

    @classmethod
    def sf(cls) -> None:
        ih = sf.IndexHierarchy.from_labels(SampleData.get('tuple_index_3D'))







if __name__ == '__main__':

    # fsf = BuoyLoader.buoy_to_sf(BUOYS[0], 2018)
    fpd = BuoyLoader.buoy_to_pd(BUOYS[0], 2018)

    #-----------------------------------------------------------
    dfs = BuoySingleYear2D.to_pd_dict()
    # ipdb> {station_id: df.loc['2018-12-17', 'WVHT'].mean() for station_id, df in dfs.items()}
    # {46222: 1.5556249999999998, 46253: 1.2519148936170212, 46221: 1.6397916666666665}

    # ipdb> pd.DataFrame.from_records(([station_id,] + df[['WVHT', 'DPD']].mean().tolist() for station_id, df in dfs.items()), columns=('station_id', 'WVHT', 'DPD'))
    #    station_id      WVHT        DPD
    # 0       46222  0.970099  11.813405
    # 1       46253  0.927338  12.479156
    # 2       46221  1.012615  12.883838


    pna = BuoySingleYear2D.to_pd_panel_a()
    # <class 'pandas.core.panel.Panel'>
    # Dimensions: 3 (items) x 17034 (major_axis) x 3 (minor_axis)
    # Items axis: 46222 to 46221
    # Major_axis axis: 2018-01-01 00:00:00 to 2018-12-31 23:30:00
    # Minor_axis axis: DPD to MWD

    # pna[:, '2018-12-17', ['WVHT', 'DPD']].mean()
    #           46222      46253      46221
    # WVHT   1.555625   1.251915   1.639792
    # DPD   14.927500  14.190851  14.811667

    # ipdb> pna[46222, :, 'DPD'].values
    # array([11.76, 10.53, 11.11, ..., 15.38, 15.38, 15.38], dtype=object)

    # ipdb> pna[46222].dtypes
    # DPD     object
    # WVHT    object
    # MWD     object
    # dtype: object

    # ipdb> pna.values.shape
    # (3, 17034, 3)

    pnb = BuoySingleYear2D.to_pd_panel_b()

    # ipdb> pnb
    # <class 'pandas.core.panel.Panel'>
    # Dimensions: 3 (items) x 17034 (major_axis) x 2 (minor_axis)
    # Items axis: 46222 to 46221
    # Major_axis axis: 2018-01-01 00:00:00 to 2018-12-31 23:30:00
    # Minor_axis axis: DPD to WVHT

    fpd = BuoySingleYear2D.to_pd()

    # ipdb> fpd.loc[pd.IndexSlice[46221, datetime.datetime(2018, 12, 16, 10, 30)], 'WVHT']
    #1.47

# ipdb> fpd.loc[pd.IndexSlice[46221, '2018-12-17'], 'WVHT']
    # *** pandas.errors.UnsortedIndexError: 'MultiIndex slicing requires the index to be lexsorted: slicing on levels [1], lexsort depth 0'

    # >>> fpd.sort_index(inplace=True)

    # ipdb> fpd.loc[pd.IndexSlice[46221, '2018-12-17'], 'WVHT'].head()
    # station_id  datetime
    # 46221       2018-12-17 00:00:00    1.43
    #             2018-12-17 00:30:00    1.32
    #             2018-12-17 01:00:00    1.38
    #             2018-12-17 01:30:00    1.42
    #             2018-12-17 02:00:00    1.36
    # Name: WVHT, dtype: float64

    # ipdb> fpd.loc[pd.IndexSlice[:, '2018-12-17'], ['WVHT', 'DPD']].mean()
    # WVHT     1.484056
    # DPD     14.646503
    # dtype: float64
    # ipdb> fpd.loc[pd.IndexSlice[:, '2018-12-18'], ['WVHT', 'DPD']].mean()
    # WVHT     2.155594
    # DPD     16.404965
    # dtype: float64



    fsf = BuoySingleYear2D.to_sf()

    BuoySingleYear2D.process_sf()



    # BuoySingleYear2D.process_np()
    # BuoySingleYear2D.process_pd_panel()


    # ssf = BuoySingleYear1D.to_sf()
    # spd = BuoySingleYear1D.to_pd()

    # df = BuoySingleYear2D.process_pd_multi_index()


































# example of pandas series that supports partial matching
# In : s1 = pd.Series(range(60), index=pd.date_range('1999-12', freq='D', periods=60))

# s1 = pd.Series(range(120), pd.MultiIndex.from_product((('a', 'b'), pd.date_range('1999-12', freq='D', periods=60))))
# # this does not work
# s1[pd.IndexSlice['a', '2000']]


# partial date string matching
# https://github.com/pandas-dev/pandas/issues/25165


# class BuoyMultiYear:

#     @staticmethod
#     def to_sf(years: tp.Iterable[int] = (2017, 2017, 2018)) -> sf.Frame:

#         frames = []
#         for buoy in BUOYS:
#             for year in years:
#                 f = BuoyLoader.buoy_to_sf(buoy, year).to_frame_go()
#                 f['year'] = year
#                 f['month'] = [int(x.split('-')[1]) for x in f['datetime'].values]
#                 # import ipdb; ipdb.set_trace()
#                 frames.append(f)

#         f = sf.Frame.from_concat(frames,
#                 axis=0,
#                 index=sf.IndexAutoFactory,
#                 name='buos_multi_year'
#                 )
#         # NOTE: this fails unexpectedly
#         f = f.set_index_hierarchy(('station_id', 'datetime'),
#                 index_constructors=(sf.Index, sf.IndexMinute),
#                 drop=True)
#         return f

