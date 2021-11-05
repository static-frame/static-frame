

import os
import timeit
import tempfile
import typing as tp
import pickle

import numpy as np
import frame_fixtures as ff
import static_frame as sf
from static_frame.core.display_color import HexColor


FF_wide = 's(10,1_000)|v(int,int,bool,float,float)|i(I,str)|c(I,int)'
FF_tall = 's(1_000,10)|v(int,int,bool,float,float)|i(I,str)|c(I,int)'
FF_square = 's(100,100)|v(float)|i(I,str)|c(I,int)'



class FileIOTest:
    NUMBER = 4
    SUFFIX = '.zip'

    def __init__(self, fixtures: tp.Iterable[str]):
        # TODO: store fixtures
        frames = []
        for i, fixture in enumerate(fixtures):
            frames.append(ff.parse(fixture).rename(str(i)))
        self.fixtures = tuple(frames)

        _, self.fp = tempfile.mkstemp(suffix=self.SUFFIX)

    def __del__(self) -> None:
        os.unlink(self.fp)

    def __call__(self):
        raise NotImplementedError()


class FileReadBusParquet(FileIOTest):
    def __init__(self, fixtures: tp.Iterable[str]):
        super().__init__(fixtures)
        b = sf.Bus.from_frames(self.fixtures)
        b.to_zip_parquet(self.fp)
        self.config = sf.StoreConfig(index_depth=1, columns_depth=1)

    def __call__(self):
        b = sf.Bus.from_zip_parquet(self.fp, config=self.config)
        tuple(b.items())

class FileWriteBusParquet(FileIOTest):
    def __init__(self, fixtures: tp.Iterable[str]):
        super().__init__(fixtures)
        self.b = sf.Bus.from_frames(self.fixtures)

    def __call__(self):
        self.b.to_zip_parquet(self.fp)




class FileReadBusNPZ(FileIOTest):
    def __init__(self, fixtures: tp.Iterable[str]):
        super().__init__(fixtures)
        b = sf.Bus.from_frames(self.fixtures)
        b.to_zip_npz(self.fp)

    def __call__(self):
        b = sf.Bus.from_zip_npz(self.fp)
        tuple(b.items())

class FileWriteBusNPZ(FileIOTest):
    def __init__(self, fixtures: tp.Iterable[str]):
        super().__init__(fixtures)
        self.b = sf.Bus.from_frames(self.fixtures)

    def __call__(self):
        self.b.to_zip_npz(self.fp)



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
    for label, fixture in (
            ('wide', FF_wide),
            ('tall', FF_tall),
            ('square', FF_square),
            ):
    # for label, fixture in (('square', FF_square),):
        fixtures = [fixture] * 100
        for cls in (
                FileWriteBusParquet,
                FileWriteBusNPZ,
                # FileWritePickle,
                FileReadBusParquet,
                FileReadBusNPZ,
                # FileReadPickle,
                ):
            runner = cls(fixtures)
            record = [cls.__name__, cls.NUMBER, label]
            result = timeit.timeit(
                    f'runner()',
                    globals=locals(),
                    number=cls.NUMBER)
            record.append(result)
            records.append(record)

    f = sf.FrameGO.from_records(records,
            columns=('name', 'number', 'fixture', 'time')
            )

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


