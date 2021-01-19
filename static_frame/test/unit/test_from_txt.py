from io import StringIO
from pathlib import Path
import functools
import operator

import pytest
import numpy as np

import static_frame as sf


@pytest.fixture
def unique_tsv() -> StringIO:
    '''A tsv file where every cell is unique'''
    column_count = 10
    row_count = 5

    i = 0
    data = []
    for r in range(row_count):
        row = []
        for c in range(column_count):
            row.append(str(i))
            i += 1
        data.append('\t'.join(row))

    return StringIO('\n'.join(data))


def test_frame_from_txt() -> None:
    # header, mixed types, no index

    s1 = StringIO('count,score,color\n1,1.3,red\n3,5.2,green\n100,3.4,blue\n4,9.0,black')

    f1 = sf.Frame.from_delimited_no_guess(s1, delimiter=',')

    assert f1.shape == (4, 3)
    assert f1.iloc[:, :2].to_pairs(0) == (
        ('count', ((0, '1'), (1, '3'), (2, '100'), (3, '4'))),
        ('score', ((0, '1.3'), (1, '5.2'), (2, '3.4'), (3, '9.0'))))


def test_frame_from_txt_file(tmpdir) -> None:
    # header, mixed types, no index

    fp = tmpdir / 'input.tsv'

    with open(fp, 'w') as f:
        f.write('count,score,color\n1,1.3,red\n3,5.2,green\n100,3.4,blue\n4,9.0,black')

    f1 = sf.Frame.from_delimited_no_guess(str(fp), delimiter=',')

    assert f1.shape == (4, 3)
    assert f1.iloc[:, :2].to_pairs(0) == (
        ('count', ((0, '1'), (1, '3'), (2, '100'), (3, '4'))),
        ('score', ((0, '1.3'), (1, '5.2'), (2, '3.4'), (3, '9.0'))))


def test_from_tsv(tmpdir) -> None:
    infp = Path('/var/folders/w6/kz3x68k54sbg64gdkn2ph9nm0000gn/T/static_frame.performance.from_text_file-of-tomrutherford/r1000c5.tsv')
    # infp = Path(tmpdir) / 'infp.txt'
    # simple_frame = sf.Frame.from_records(
        # [(1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 2, 3)],
        # columns=['11', '22', '33'],
    # )
    # simple_frame.to_tsv(infp, include_index=False)

    with open(infp, 'r') as f:
        f1 = sf.Frame.from_delimited_no_guess(f, delimiter='\t')

    assert 0


def test_frame_from_txt_file_2(tmpdir) -> None:
    tmpfile = Path(tmpdir) / 'temp.txt'
    with open(tmpfile, 'w') as file:
        file.write('\n'.join(('index|A|B', 'a|True|20.2', 'b|False|85.3')))

    with pytest.raises(sf.ErrorInitFrame):
        f = sf.Frame.from_delimited_no_guess(tmpfile, index_depth=1, delimiter='|', skip_header=-1)

    f = sf.Frame.from_delimited_no_guess(tmpfile, index_depth=1, delimiter='|')
    assert (f.to_pairs(0) ==
            (('A', (('a', True), ('b', False))), ('B', (('a', 20.2), ('b', 85.3)))))


def test_depth(unique_tsv: StringIO):
    f = sf.Frame.from_delimited_no_guess(unique_tsv, delimiter='\t', index_depth=1)
    assert f.index.values.tolist() == ['10', '20', '30', '40']
    e = np.array([['11', '12'],
           ['21', '22']], dtype=object)
    assert (f.iloc[:2, :2].values == e).all()
    assert f.columns.values.tolist() == ['1', '2', '3', '4', '5', '6', '7', '8', '9']

    unique_tsv.seek(0)
    f = sf.Frame.from_delimited_no_guess(unique_tsv, delimiter='\t', columns_depth=0)
    e = np.array([['0', '1'],
                  ['10', '11']], dtype=object)
    assert (f.iloc[:2, :2].values == e).all()
    assert f.columns.values.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    unique_tsv.seek(0)
    f = sf.Frame.from_delimited_no_guess(unique_tsv, delimiter='\t', columns_depth=2, index_depth=2)
    e = np.array([['22', '23'],
                  ['32', '33']], dtype=object)
    assert (f.iloc[:2, :2].values == e).all()
    assert f.columns.shape == (8, 2)
    assert f.index.shape == (3, 2)


def test_converters(unique_tsv: StringIO):
    # Converters are needed for dtypes to be useful.
    def is_11(element):
        return element == '11'
    f = sf.Frame.from_delimited_no_guess(
        unique_tsv,
        delimiter='\t',
        index_depth=1,
        converters={
            '1': is_11,
            '3': lambda x: x + '0',
            '2': lambda x: '3' in x,
        },
        dtypes={
            '3': int,
        }
    )

    assert f['3'].equals(sf.Series([130, 230, 330, 430], index=f.index))
    assert f['1'].equals(sf.Series([True, False, False, False], index=f.index))
    assert f['2'].equals(sf.Series([False, False, True, False], index=f.index))
