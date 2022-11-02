import tempfile
import typing as tp
from io import BytesIO
from io import StringIO
from pathlib import Path
from urllib import request


def URL(
        url: str,
        encoding: tp.Optional[str] = 'utf-8',
        in_memory: bool = True,
        buffer_size: int = 8192,
        ) -> tp.Union[Path, StringIO, BytesIO]:
    '''
    Args:
        encoding: Defaults to UTF-8; if None, binary data is collected.
        in_memory: if True, data is loaded into memory; if False, a temporary file is written.
    '''
    # TODO: support unzipping files

    with request.urlopen(url) as response:
        if in_memory:
            if encoding:
                return StringIO(response.read().decode(encoding))
            else:
                return BytesIO(response.read())

        # not in-memory, write a file
        with tempfile.NamedTemporaryFile(mode='w' if encoding else 'wb',
                suffix=None,
                delete=False,
                ) as f:
            fp = f.name

            if encoding:
                extract = lambda: response.read(buffer_size).decode(encoding)
            else:
                extract = lambda: response.read(buffer_size)

            while True:
                b = extract()
                if b:
                    f.write(b)
                else:
                    break
            return Path(fp)



if __name__ == '__main__':
    import ipdb; ipdb.set_trace()
