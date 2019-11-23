


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


import numpy as np


import static_frame as sf

# stations


class Buoy(NamedTuple):
    station_id: int
    name: str
    # filename: str

BUOYS = (
    Buoy(46222, 'San Pedro'),
    Buoy(46253, 'San Pedro South'),
    Buoy(46221, 'Santa Monica Bay'),
)


def cache_buoy(prefix):
    def decorator(func):
        def wrapper(cls, buoy, year):
            fp = f'/tmp/{prefix}-{buoy.station_id}-{year}.p'
            if os.path.exists(fp):
                with open(fp, 'rb') as f:
                    post = pickle.load(f)
            else:
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

    URL_TEMPLATE = 'https://www.ndbc.noaa.gov/view_text_file.php?filename={station_id}h{year}.txt.gz&dir=data/historical/stdmet/'

    @classmethod
    def buoy_record(cls,
            line: str,
            count: int,
            station_id: int
            ) -> tp.Sequence[str]:
        timestamp = []

        def gen() -> tp.Iterator[tp.Union[int, str]]:
            if count == 0:
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
                        if count == 0:
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
    @cache_buoy('sf')
    def buoy_to_sf(cls, buoy: Buoy, year: int) -> sf.Frame:
        '''
        Return a simple Frame presentation without an index.
        '''

        records = cls.buoy_to_records(buoy, year=year)
        columns = next(records)
        units = next(records)

        # can pass dtypes here, but doing it below consolidates blocks
        f = sf.Frame.from_records(records, columns=columns)

        f = f[[cls.FIELD_STATION_ID,
                cls.FIELD_DATETIME,
                cls.FIELD_WAVE_HEIGHT,
                cls.FIELD_WAVE_PERIOD]] # wave height, dominant period

        return f.astype[[cls.FIELD_WAVE_HEIGHT, cls.FIELD_WAVE_PERIOD]](float)

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

        return df.astype({cls.FIELD_WAVE_HEIGHT: float, cls.FIELD_WAVE_PERIOD: float})

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
    def to_pd(year: int = 2018) -> sf.Frame:
        import pandas as pd

        dfs = []
        for buoy in BUOYS:
            f = BuoyLoader.buoy_to_pd(buoy, year)
            dfs.append(f)

        df = pd.concat(dfs)

        # try setting dtype to datetime, with pd.to_datetime, before index creation
        df = df.astype({'datetime':np.datetime64})
        df = df.set_index(['station_id', 'datetime'])

        # this sets to datetime, but does not allow slicing
        # df.index = df.index.set_levels([df.index.levels[0], pd.to_datetime(df.index.levels[1])])
        return df


    @classmethod
    def process_sf(cls) -> None:

        f = cls.to_sf()

        # dsicrete selection
        part = f.loc[sf.HLoc[:, ['2018-12-18T20:00', '2018-12-18T20:30']], 'DPD']
        # can show iloc

        # show getting labels with iter_label (unique values)
        # show converting to a Frame


        # select based on partial time
        post1 = f.loc[sf.HLoc[:, '2018-12-18T20']]

        # find max for givne day
        f.loc[sf.HLoc[:, '2018-12-18']].max()


        max_dpd = [f.loc[sf.HLoc[station_id], 'DPD'].loc_max() for station_id in f.index.iter_label(0)]
        max_wvht = [f.loc[sf.HLoc[station_id], 'WVHT'].loc_max() for station_id in f.index.iter_label(0)]

        # get the peaks of the two fields
        peaks = f.loc[f.index.isin(max_dpd + max_wvht)]
        # this isolates the relevant days; might need to change buoys
        big = f.loc[(f.loc[:, 'WVHT'] > 2.1) & (f.loc[:, 'DPD'] > 18)]

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
        # df.loc[pd.IndexSlice[:, datipdb> df.loc[pd.IndexSlice[:, '2018-12-18'], 'DPD']
        # *** pandas.errors.UnsortedIndexError: 'MultiIndex slicing requires the index to be lexsorted: slicing on levels [1], lexsort depth 0'
        # etime.date(2018,12,18), 'DPD']]

        big = df.loc[(df.loc[:, 'WVHT'] > 2.1) & (df.loc[:, 'DPD'] > 18)]

        import ipdb; ipdb.set_trace()


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

    # f = BuoyLoader.buoy_to_sf(BUOYS[0], 2018)
    # df = BuoyLoader.buoy_to_pd(BUOYS[0], 2018)
    # f = BuoySingleYear2D.to_sf()
    # df = BuoySingleYear2D.to_pd()

    ssf = BuoySingleYear1D.to_sf()
    ssf[sf.HLoc[46222, datetime.date(2018, 12, 18), 'DPD']]
    # spd = BuoySingleYear1D.to_pd()

    # df = BuoySingleYear2D.process_pd_multi_index()
    import ipdb; ipdb.set_trace()






































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

