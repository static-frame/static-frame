
import timeit
from datetime import date
from datetime import datetime
from datetime import timedelta
import calendar

import typing as tp
from static_frame.core.display_color import HexColor

import numpy as np
dt64 = np.datetime64

import static_frame as sf


def get_date_strings():
    # get 30 years of daily data, 10957 observations
    post = sf.IndexDate.from_date_range('1990-01-01', '2019-12-31')
    return post.values.astype(str).tolist()


class DirectFromString:
    NUMBER = 500
    def __init__(self):
        self.date_str = get_date_strings()

    def dt(self):
        date_str = self.date_str
        array = np.empty(len(date_str), dtype=object)
        for i, msg in enumerate(date_str):
            array[i] = date.fromisoformat(msg)
        return array

    def dt64(self):
        return np.array(self.date_str, dtype='datetime64[D]')
        # return np.array(self.date_str, dtype=np.datetime64)


class ParseFromString:
    NUMBER = 50

    def __init__(self):
        self.date_str = get_date_strings()

    def dt(self):
        date_str = self.date_str
        array = np.empty(len(date_str), dtype=object)
        for i, msg in enumerate(date_str):
            array[i] = datetime.strptime(msg, '%Y-%m-%d').date()
        return array

    def dt64(self):
        return np.array(self.date_str, dtype=np.datetime64)

class ShiftDay:
    NUMBER = 500

    def __init__(self):
        runner = DirectFromString()
        self.array_dt64 = runner.dt64()
        self.array_dt = runner.dt()

    def dt(self):
        td = timedelta(days=1)
        np.array([d + td for d in self.array_dt], dtype=object)

    def dt64(self):
        self.array_dt64 + 1

class DeriveYear:
    NUMBER = 100

    def __init__(self):
        runner = DirectFromString()
        self.array_dt64 = runner.dt64()
        self.array_dt = runner.dt()

    def dt(self):
        np.array([x.year for x in self.array_dt], dtype=int)

    def dt64(self):
        self.array_dt64.astype('datetime64[Y]')

class TrueOnMonday:
    NUMBER = 100

    def __init__(self):
        runner = DirectFromString()
        self.array_dt64 = runner.dt64()
        self.array_dt = runner.dt()

    def dt(self):
        np.array([x.weekday() == 0 for x in self.array_dt], dtype=bool)

    def dt64(self):
        np.array(
            ((self.array_dt64.astype(int) + 3) % 7) == 0,
            dtype=int,
            )

class TrueOnEOM:
    NUMBER = 100

    def __init__(self):
        runner = DirectFromString()
        self.array_dt64 = runner.dt64()
        self.array_dt = runner.dt()

    def dt(self):

        np.array([d.day == calendar.monthrange(d.year, d.month)[1] for d in self.array_dt], dtype=bool)

    def dt64(self):
        np.array(
            (self.array_dt64.astype('datetime64[M]') + 1).astype('datetime64[D]') - 1 == self.array_dt64,
            dtype=bool,
            )


class TrueLessThan:
    NUMBER = 400

    def __init__(self):
        runner = DirectFromString()
        self.array_dt64 = runner.dt64()
        self.boundary_dt64 = np.datetime64('2005-06-18')
        self.array_dt = runner.dt()
        self.boundary_dt = date(2005, 6, 18)


    def dt(self):
        self.array_dt < self.boundary_dt

    def dt64(self):
        self.array_dt64 < self.boundary_dt64


class ConvertToStr:
    NUMBER = 100

    def __init__(self):
        runner = DirectFromString()
        self.array_dt64 = runner.dt64()
        self.array_dt = runner.dt()

    def dt(self):
        np.array([str(d) for d in self.array_dt], dtype=object)

    def dt64(self):
        self.array_dt64.astype(str)



def get_format():

    name_root_last = None
    name_root_count = 0

    def format(key: tp.Tuple[tp.Any, str], v: object) -> str:
        nonlocal name_root_last
        nonlocal name_root_count

        if isinstance(v, float):
            if np.isnan(v):
                return ''
            return str(round(v, 4))
        if isinstance(v, (bool, np.bool_)):
            if v:
                return HexColor.format_terminal('green', str(v))
            return HexColor.format_terminal('orange', str(v))

        return str(v)

    return format



def run_test():
    records = []
    for cls in (
            DirectFromString,
            ParseFromString,
            ShiftDay,
            DeriveYear,
            TrueOnMonday,
            TrueOnEOM,
            TrueLessThan,
            ConvertToStr,
            ):
        runner = cls()
        record = [cls.__name__, cls.NUMBER]
        for func_name in ('dt', 'dt64'):
            result = timeit.timeit(
                    f'runner.{func_name}()',
                    globals=locals(),
                    number=cls.NUMBER)
            record.append(result)
        records.append(record)

    f = sf.FrameGO.from_records(records,
            columns=('name', 'number', 'dt', 'dt64')
            )

    f['dt64/dt'] = f['dt64'] / f['dt']
    f['dt/dt64'] = f['dt'] / f['dt64']

    f['dt64-faster'] = f['dt/dt64'] > 1

    display = f.iter_element_items().apply(get_format())

    config = sf.DisplayConfig(
            cell_max_width_leftmost=np.inf,
            cell_max_width=np.inf,
            type_show=False,
            display_rows=200,
            include_index=False,
            )
    print(display.display(config))
    # import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    run_test()


    # print(len(get_date_strings()))


    # ptest = ArrayCreationFromString()
    # post = ptest.dt()
    # post = ptest.dt64()

    # import ipdb; ipdb.set_trace()



