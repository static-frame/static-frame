
import unittest
import typing as tp

import numpy as np
import static_frame as sf

from static_frame.test.test_case import TestCase

CHARACTERS_REFERENCE = dict((
        ('count', len),
        ('count<0', lambda x: len(x.loc[x < 0])),
        ('count>0', lambda x: len(x.loc[x > 0])),
        ('count_unique', lambda x: len(x.dropna().unique())),
        ('min', lambda x: x.min()),
        ('max', lambda x: x.max()),
        ('sum', lambda x: x.sum()),
        ('mean', lambda x: x.mean()),
        ('median', lambda x: x.median()),
        ('nanfill', lambda x: x.isna().sum() / len(x)),
        ))

class FieldStatConfig:

    CHARACTERS = tuple((k, v) for k, v in CHARACTERS_REFERENCE.items())

    CHARACTERS_AGGREGATE = tuple((k, v)
            for k, v in CHARACTERS_REFERENCE.items()
            if not (k.startswith('count') or k.startswith('nan')))

    GROUP_BY = 'group_by'
    CATEGORY_A_FIELDS: tp.Iterable[tp.Tuple[str, tp.Any]] = (('a1', None), ('a2', None))
    CATEGORY_B_FIELDS: tp.Iterable[tp.Tuple[str, tp.Any]] = (('b1', None), ('b2', None))

    # store pairs of lable, config attr
    CATEGORY_LABELS = (
            ('category_a', 'CATEGORY_A_FIELDS'),
            ('category_b', 'CATEGORY_B_FIELDS'),
            )



def process(
        frame: sf.Frame,
        config: tp.Type[FieldStatConfig] = FieldStatConfig
        ) -> sf.Frame:
    '''
    Perform statistical analysis, returning a new Frame.
    '''
    # print(frame.columns.display(sf.DisplayConfigs.UNBOUnND))

    def observations(
            fields: tp.Iterable[tp.Tuple[str, tp.Any]],
            ) -> tp.Iterator[sf.Series]:
        for k, group in frame.iter_group_items(config.GROUP_BY):

            def gen() -> tp.Iterator[tp.Tuple[tp.Tuple[str, str], sf.Series]]:
                for field, converter in fields:
                    # the key here becomes columns
                    converter = converter if converter else lambda x: x
                    for label, func in config.CHARACTERS:
                        yield (field, label), func(converter(group[field])) # type: ignore

            # name will become row (index) identifier
            yield sf.Series.from_items(gen(),
                    name=k,
                    index_constructor=sf.IndexHierarchy.from_labels)

    def field_categories() -> tp.Iterator[sf.Frame]:

        for label, attr in config.CATEGORY_LABELS:
            # for each category, get all fields, and process each character, for each field, into one Frame
            fields = getattr(config, attr)


            post = sf.Frame.from_concat(observations(fields))

            # import ipdb; ipdb.set_trace()

            # create more rows with axis config.CHARACTERS over the groups
            def gen() -> tp.Iterator[sf.Frame]:
                for label, func in config.CHARACTERS_AGGREGATE:
                    yield func(post).rename(label) # type: ignore

            post = sf.Frame.from_concat((post, sf.Frame.from_concat(gen())))

            # round float, rotate table, and new index for this level
            # NOTe: after iter_lement apply, we are loosing hierarchucal index
            yield post.iter_element().apply(lambda e: round(e, 3)).T.relabel_level_add(label)

    # vertically stack all Frame for each category
    return sf.Frame.from_concat(field_categories()) # type: ignore



class TestUnit(TestCase):


    def test_process(self) -> None:

        a1 = np.array([.1, -.1, np.nan, .2])
        a2 = np.array([.4, np.nan, .7, -.5])

        records = (
            a1,
            a1 * .2,
            a1 * -.5,
            a1 * 0,
            np.flip(a1),
            np.flip(a1) * .2,
            a2,
            a2 * .2,
            a2 * -.5,
            a2 * 0,
            np.flip(a2),
            np.flip(a2) * .2,
        )

        f = sf.FrameGO.from_records(records, columns=('a1', 'a2', 'b1', 'b2'))
        f['group_by'] = ['x'] * (len(f) // 2) + ['y'] * (len(f) // 2)


        post = process(f)

        self.assertEqual(post.shape, (40, 7))
        self.assertEqual(post.index.depth, 3)

        self.assertEqual(post.iloc[0].to_pairs(),
            (('x', 6.0), ('y', 6.0), ('min', 6.0), ('max', 6.0), ('sum', 12.0), ('mean', 6.0), ('median', 6.0))
            )

        self.assertEqual(post['x'].to_pairs(),
            ((('category_a', 'a1', 'count'), 6.0), (('category_a', 'a1', 'count<0'), 1.0), (('category_a', 'a1', 'count>0'), 4.0), (('category_a', 'a1', 'count_unique'), 6.0), (('category_a', 'a1', 'min'), -0.05), (('category_a', 'a1', 'max'), 0.2), (('category_a', 'a1', 'sum'), 0.31), (('category_a', 'a1', 'mean'), 0.052), (('category_a', 'a1', 'median'), 0.03), (('category_a', 'a1', 'nanfill'), 0.0), (('category_a', 'a2', 'count'), 6.0), (('category_a', 'a2', 'count<0'), 2.0), (('category_a', 'a2', 'count>0'), 1.0), (('category_a', 'a2', 'count_unique'), 4.0), (('category_a', 'a2', 'min'), -0.1), (('category_a', 'a2', 'max'), 0.05), (('category_a', 'a2', 'sum'), -0.07), (('category_a', 'a2', 'mean'), -0.018), (('category_a', 'a2', 'median'), -0.01), (('category_a', 'a2', 'nanfill'), 0.333), (('category_b', 'b1', 'count'), 6.0), (('category_b', 'b1', 'count<0'), 2.0), (('category_b', 'b1', 'count>0'), 0.0), (('category_b', 'b1', 'count_unique'), 2.0), (('category_b', 'b1', 'min'), -0.1), (('category_b', 'b1', 'max'), -0.02), (('category_b', 'b1', 'sum'), -0.12), (('category_b', 'b1', 'mean'), -0.06), (('category_b', 'b1', 'median'), -0.06), (('category_b', 'b1', 'nanfill'), 0.667), (('category_b', 'b2', 'count'), 6.0), (('category_b', 'b2', 'count<0'), 1.0), (('category_b', 'b2', 'count>0'), 4.0), (('category_b', 'b2', 'count_unique'), 6.0), (('category_b', 'b2', 'min'), -0.1), (('category_b', 'b2', 'max'), 0.2), (('category_b', 'b2', 'sum'), 0.26), (('category_b', 'b2', 'mean'), 0.043), (('category_b', 'b2', 'median'), 0.03), (('category_b', 'b2', 'nanfill'), 0.0))
        )

        self.assertAlmostEqualValues(post.sum().values.tolist(),
            [58.466999999999985, 61.121, 48.06699999999999, 71.52100000000004, 119.587, 59.794000000000004, 59.794000000000004])

if __name__ == '__main__':
    unittest.main()
