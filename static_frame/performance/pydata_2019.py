


# https://www.ocregister.com/2018/12/17/big-surf-possible-flooding-and-erosion-as-massive-swell-hits-the-coast-this-week/

# The long-period, west-northwest swell started showing early Monday. At some stand-out Southern California beaches, waves were expected to reach up to 12 feet. Waves are even bigger at Central Coast, up to 20 feet, and largest in Northern California, with waves upwards of 50 feet at some spots.

# 2018-12-17


# station 46222

# field definitions
# https://www.ndbc.noaa.gov/measdes.shtml

# WVHT 	Significant wave height (meters) is calculated as the average of the highest one-third of all of the wave heights during the 20-minute sampling period. See the Wave Measurements section.
# DPD 	Dominant wave period (seconds) is the period with the maximum wave energy. See the Wave Measurements section.
# APD 	Average wave period (seconds) of all waves during the 20-minute period. See the Wave Measurements section.


# https://www.ndbc.noaa.gov/histsearch.php?station=46222&year=2018&f1=wvht&t1a=lt&v1a=4&t1b=&v1b=&c1=&f2=&t2a=&v2a=&t2b=&v2b=&c2=&f3=&t3a=&v3a=&t3b=&v3b=

# full data set
# https://www.ndbc.noaa.gov/view_text_file.php?filename=46222h2018.txt.gz&dir=data/historical/stdmet/


# sample data from peak

# 2018 12 17 18 00 999 99.0 99.0  2.54 18.18 13.44 268 9999.0 999.0  16.7 999.0 99.0 99.00
# 2018 12 17 18 30 999 99.0 99.0  2.33 20.00 13.40 267 9999.0 999.0  16.8 999.0 99.0 99.00
# 2018 12 17 19 00 999 99.0 99.0  2.44 20.00 13.35 273 9999.0 999.0  16.8 999.0 99.0 99.00
# 2018 12 17 19 30 999 99.0 99.0  2.55 15.38 13.39 267 9999.0 999.0  16.8 999.0 99.0 99.00
# 2018 12 17 20 00 999 99.0 99.0  2.48 18.18 13.22 270 9999.0 999.0  16.9 999.0 99.0 99.00
# 2018 12 17 20 30 999 99.0 99.0  2.54 18.18 13.25 268 9999.0 999.0  17.1 999.0 99.0 99.00

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

# stations


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

    COMPASS = ('N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW')


    URL_TEMPLATE = 'https://www.ndbc.noaa.gov/view_text_file.php?filename={station_id}h{year}.txt.gz&dir=data/historical/stdmet/'

    @classmethod
    def buoy_record(cls,
            line: str,
            line_number: int,
            station_id: int
            ) -> tp.Sequence[str]:
        timestamp = []

        def gen() -> tp.Iterator[tp.Union[int, str]]:
            if line_number == 0:
                yield cls.FIELD_STATION_ID
            else:
                yield station_id # always put first
            cell_pos = -1 # increment before usage
            for cell in line.split(' '):
                cell = cell.strip()
                if cell:
                    cell_pos += 1
                    if cell_pos < 5:
                        timestamp.append(cell)
                    elif cell_pos == 5:
                        if line_number == 0:
                            yield cls.FIELD_DATETIME
                        else:
                            yield '{}-{}-{}T{}:{}'.format(*timestamp)
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
        # import ipdb; ipdb.set_trace()
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
        import pandas as pd

        records = cls.buoy_to_records(buoy, year=year)
        columns = next(records)
        units = next(records)

        df = pd.DataFrame.from_records(records, columns=columns)

        df = df[[cls.FIELD_STATION_ID,
                cls.FIELD_DATETIME,
                cls.FIELD_WAVE_HEIGHT,
                cls.FIELD_WAVE_PERIOD]]

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
                name='buos_single_year'
                )
        f = f.set_index_hierarchy(('station_id', 'datetime'),
                index_constructors=(sf.Index, sf.IndexMinute),
                drop=True)
        return f

    @staticmethod
    def to_pd(year: int = 2018):
        import pandas as pd

        dfs = []
        for buoy in BUOYS:
            f = BuoyLoader.buoy_to_pd(buoy, year)
            dfs.append(f)

        df = pd.concat(dfs)

        # try setting dtype to datetime, with pd.to_datetime, before index creation
        df = df.set_index(['station_id', 'datetime'])

        # this sets to datetime, but does not allow slicing
        # df.index = df.index.set_levels([df.index.levels[0], pd.to_datetime(df.index.levels[1])])
        return df

    @staticmethod
    def to_pd_panel(year: int = 2018):
        import pandas as pd

        dfs = {}
        for buoy in BUOYS:
            df = BuoyLoader.buoy_to_pd(buoy, year)
            # remove station id, set datetime as indexnns
            df = df.set_index('datetime')[['DPD', 'WVHT']]
            dfs[buoy.station_id] = df

        p = pd.Panel(dfs)
        return p

        # reindexes and adds NaN
        # ipdb> panel[:, '2018-03-01T18:58']
        #       46222  46253  46221
        # DPD   12.50    NaN    NaN
        # WVHT   0.69    NaN    NaN


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

        f = cls.to_sf()

        #-----------------------------------------------------------------------
        # creating IndexHierarchy



        #-----------------------------------------------------------------------
        # getting values out

        # iterating parts of the index
        part = tuple(f.index.iter_label(0))

        # getting all the values at an array
        part = f.index.values_at_depth(0)

        # convert to a frame
        part = f.index.to_frame()

        # rehiearch (make dates outer, station id inner); note that changing the hiearchy order forces a reordering
        part = f.rehierarch((1, 0))

        # to tuples
        part = f.relabel_flat(index=True)

        # adding a level
        part = f.relabel_add_level(index='A')

        # dropping a level
        #         ipdb> f.relabel_drop_level(index=1)
        # *** static_frame.core.exception.ErrorInitIndex: labels (50350) have non-unique values (17034)
        # ipdb>

        #-----------------------------------------------------------------------
        # show different types of selection

        # getting a sample from a partial match
        part = f.loc[sf.HLoc[:, '2018-12-18T07'], 'DPD']

        # select based on partial time
        post1 = f.loc[sf.HLoc[:, '2018-12-18T20']]

        # getting a slice
        part = f.loc[sf.HLoc[:, '2018-12-18T07':'2018-12-18T08'], 'DPD']

        # dsicrete selection (this works on Pandas)
        part = f.loc[sf.HLoc[:, ['2018-12-18T20:00', '2018-12-18T20:30']], 'DPD']

        # can show iloc
        part = f.loc[sf.HLoc[:, ['2018-12-18T20:00', '2018-12-18T20:30']], sf.ILoc[-1]]



        # show getting labels with iter_label (unique values)
        # show converting to a Frame



        #-----------------------------------------------------------------------
        # doing some analysis

        # find max for givne day
        f.loc[sf.HLoc[:, '2018-12-18']].max()


        max_dpd = [f.loc[sf.HLoc[station_id], 'DPD'].loc_max() for station_id in f.index.iter_label(0)]
        max_wvht = [f.loc[sf.HLoc[station_id], 'WVHT'].loc_max() for station_id in f.index.iter_label(0)]

        # get the peaks of the two fields, but this does not get us to the date
        peaks = f.loc[f.index.isin(max_dpd + max_wvht)]

        # use 2 to get 1.731622836825856
        wvht_threshold = f.loc[:, 'WVHT'].mean() + (f.loc[:, 'WVHT'].std() * 2)

        # use 1 to get 15.889409302831822
        dpd_threshold = f.loc[:, 'DPD'].mean() + f.loc[:, 'DPD'].std()

        # this isolates the relevant days; but does not get 46253
        # 2 and 18 gets all
        targets = f.loc[(f.loc[:, 'WVHT'] > wvht_threshold) & (f.loc[:, 'DPD'] > dpd_threshold)]
        targets = targets.to_frame_go()
        targets['date'] = [d.date() for d in targets.index.values_at_depth(1)]

        # targets.iter_group('date').apply(lambda x: len(x))
        peaks_per_day = targets.iter_group('date').apply(lambda x: len(x))
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
        import pandas as pd


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



if __name__ == '__main__':

    # fsf = BuoyLoader.buoy_to_sf(BUOYS[0], 2018)
    # fpd = BuoyLoader.buoy_to_pd(BUOYS[0], 2018)

    fsf = BuoySingleYear2D.to_sf()
    fpd = BuoySingleYear2D.to_pd()

    BuoySingleYear2D.process_sf()

    # BuoySingleYear2D.process_np()
    # BuoySingleYear2D.process_pd_panel()


    # ssf = BuoySingleYear1D.to_sf()
    # spd = BuoySingleYear1D.to_pd()

    # df = BuoySingleYear2D.process_pd_multi_index()
    # import ipdb; ipdb.set_trace()


































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

