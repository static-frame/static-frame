

import os
import timeit
import tempfile
import typing as tp
import pickle

import numpy as np
import frame_fixtures as ff
import static_frame as sf
from static_frame.core.display_color import HexColor



FF_wide = 's(10,10_000)|v(int,int,bool,float,float)|i(I,str)|c(I,int)'
FF_tall = 's(10_000,10)|v(int,int,bool,float,float)|i(I,str)|c(I,int)'

class FileIOTest:
    NUMBER = 4
    SUFFIX = '.tmp'

    def __init__(self, fixture: str):
        self.fixture = ff.parse(fixture)
        _, self.fp = tempfile.mkstemp(suffix=self.SUFFIX)

    def __del__(self) -> None:
        os.unlink(self.fp)

    def run(self):
        raise NotImplementedError()



class FileReadParquet(FileIOTest):
    SUFFIX = '.parquet'

    def __init__(self, fixture: str):
        super().__init__(fixture)
        self.fixture.to_parquet(self.fp)

    def run(self):
        _ = sf.Frame.from_parquet(self.fp)

class FileWriteParquet(FileIOTest):
    SUFFIX = '.parquet'

    def run(self):
        self.fixture.to_parquet(self.fp)



class FileReadNPZ(FileIOTest):
    SUFFIX = '.npz'

    # NOTE: must write a file with NPZ

    def __init__(self, fixture: str):
        super().__init__(fixture)
        self.fixture.to_npz(self.fp)

    def run(self):
        _ = sf.Frame.from_npz(self.fp)

class FileWriteNPZ(FileIOTest):
    SUFFIX = '.npz'

    def run(self):
        self.fixture.to_npz(self.fp)



# class FileWriteNPZCompressed(FileIOTest):

#     def run(self):
#         self.fixture.to_npz(self.fp, compress=True)


class FileWritePickle(FileIOTest):
    SUFFIX = '.pickle'

    def __init__(self, fixture):
        super().__init__(fixture)
        self.file = open(self.fp, 'wb')

    def run(self):
        pickle.dump(self.fixture, self.file)

    def __del__(self) -> None:
        self.file.close()
        os.unlink(self.fp)



#-------------------------------------------------------------------------------
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
    for label, fixture in (('wide', FF_wide), ('tall', FF_tall)):
        for cls in (
                FileWriteParquet,
                FileWriteNPZ,
                # FileWriteNPZCompressed,
                FileWritePickle,
                FileReadParquet,
                FileReadNPZ,
                ):
            runner = cls(fixture)
            record = [cls.__name__, cls.NUMBER, label]
            result = timeit.timeit(
                    f'runner.run()',
                    globals=locals(),
                    number=cls.NUMBER)
            record.append(result)
            records.append(record)

    f = sf.FrameGO.from_records(records,
            columns=('name', 'number', 'fixture', 'time')
            )

    # f['dt64/dt'] = f['dt64'] / f['dt']
    # f['dt/dt64'] = f['dt'] / f['dt64']

    # f['dt64-faster'] = f['dt/dt64'] > 1

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

