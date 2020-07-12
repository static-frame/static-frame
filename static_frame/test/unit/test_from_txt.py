from io import StringIO
from pathlib import Path

import pytest

import static_frame as sf


def test_frame_from_txt() -> None:
    # header, mixed types, no index

    s1 = StringIO('count,score,color\n1,1.3,red\n3,5.2,green\n100,3.4,blue\n4,9.0,black')

    f1 = sf.Frame.from_txt(s1, delimiter=',')

    post = f1.iloc[:, :2].sum(axis=0)
    assert post.to_pairs() == (('count', 108.0), ('score', 18.9))
    assert f1.shape == (4, 3)

    assert (f1.dtypes.iter_element().apply(str).to_pairs() ==
            (('count', 'int64'), ('score', 'float64'), ('color', '<U5')))


def test_from_tsv(tmpdir) -> None:
    infp = Path('/var/folders/w6/kz3x68k54sbg64gdkn2ph9nm0000gn/T/static_frame.performance.from_text_file-of-tomrutherford/r1000c5.csv')
    # infp = Path(tmpdir) / 'infp.txt'
    # simple_frame = sf.Frame.from_records(
        # [(1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 2, 3)],
        # columns=['11', '22', '33'],
    # )
    # simple_frame.to_tsv(infp, include_index=False)

    with open(infp, 'r') as f:
        f1 = sf.Frame.from_txt(f, delimiter='\t')

    assert 0


def test_frame_from_txt_file(tmpdir) -> None:
    tmpfile = Path(tmpdir) / 'temp.txt'
    with open(tmpfile, 'w') as file:
        file.write('\n'.join(('index|A|B', 'a|True|20.2', 'b|False|85.3')))

    with pytest.raises(sf.ErrorInitFrame):
        f = sf.Frame.from_txt(tmpfile, index_depth=1, delimiter='|', skip_header=-1)

    f = sf.Frame.from_txt(tmpfile, index_depth=1, delimiter='|')
    assert (f.to_pairs(0) ==
            (('A', (('a', True), ('b', False))), ('B', (('a', 20.2), ('b', 85.3)))))
