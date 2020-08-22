








import unittest

import numpy as np

nan = np.nan

from static_frame.core.frame import Frame
from static_frame.core.batch import Batch
from static_frame.test.test_case import TestCase
from static_frame.core.index_auto import IndexAutoFactory
from static_frame.core.display import DisplayConfig


class TestUnit(TestCase):



    def test_batch_a(self) -> None:

        f1 = Frame.from_dict(
                {'a':[1,49,2,3], 'b':[2,4,381, 6], 'group': ['x', 'x','z','z']},
                index=('r', 's', 't', 'u'))

        b1 = Batch(f1.iter_group_items('group'))

        b2 = b1 * 3

        post = tuple(b2.items())
        self.assertEqual(post[0][1].to_pairs(0), #type: ignore
                (('a', (('r', 3), ('s', 147))), ('b', (('r', 6), ('s', 12))), ('group', (('r', 'xxx'), ('s', 'xxx')))),
                )

    def test_batch_b(self) -> None:

        f1 = Frame.from_dict(
                {'a':[1,49,2,3], 'b':[2,4,381, 6], 'group': ['x', 'x','z','z']},
                index=('r', 's', 't', 'u'))

        b1 = Batch(f1.iter_group_items('group'))
        self.assertEqual(b1['b'].sum().to_frame().to_pairs(0),
                (('b', (('x', 6), ('z', 387))),)
                )


    def test_batch_c(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('x', 'q'),
                name='f3')

        b1 = Batch.from_frames((f1, f2, f3))

        self.assertEqual(b1.shapes.to_pairs(),
                (('f1', (2, 2)), ('f2', (3, 2)), ('f3', (2, 2)))
                )


        self.assertEqual(b1.loc['x'].to_frame(fill_value=0).to_pairs(0),
                (('a', (('f1', 1), ('f2', 0), ('f3', 0))), ('b', (('f1', 3), ('f2', 4), ('f3', 50))), ('c', (('f1', 0), ('f2', 1), ('f3', 0))), ('d', (('f1', 0), ('f2', 0), ('f3', 10))))
                )

        self.assertEqual(b1.loc['x'].to_frame(fill_value=0, axis=1).to_pairs(0),
                (('f1', (('a', 1), ('b', 3), ('c', 0), ('d', 0))), ('f2', (('a', 0), ('b', 4), ('c', 1), ('d', 0))), ('f3', (('a', 0), ('b', 50), ('c', 0), ('d', 10))))
                )
        # import ipdb; ipdb.set_trace()



    def test_batch_d(self) -> None:

        f1 = Frame.from_dict({'a':[1,2,3], 'b':[2,4,6], 'group': ['x','z','z']})

        f2 = Batch(f1.iter_group_items('group'))['b'].sum()
        self.assertEqual(f2.to_frame().to_pairs(0),
                (('b', (('x', 2), ('z', 10))),)
                )

    def test_batch_e(self) -> None:

        f1 = Frame.from_dict({'a':[1,2,3], 'b':[2,4,6], 'group': ['x','z','z']})

        gi = f1.iter_group_items('group')
        f2 = Batch(gi)[['a', 'b']].sum().to_frame()
        self.assertEqual(f2.to_pairs(0),
                (('a', (('x', 1), ('z', 5))), ('b', (('x', 2), ('z', 10))))
                )

        gi = f1.iter_group_items('group')
        f3 = Frame.from_concat((-Batch(gi)[['a', 'b']]).values)
        self.assertEqual(f3.to_pairs(0),
                (('a', ((0, -1), (1, -2), (2, -3))), ('b', ((0, -2), (1, -4), (2, -6)))))


    def test_batch_f(self) -> None:

        f1 = Frame.from_dict({'a':[1,2,3], 'b':[2,4,6], 'group': ['x','z','z']})

        f2 = Batch(f1.iter_group_items('group')).loc[:, 'b'].sum().to_frame()
        self.assertEqual(f2.to_pairs(0),
                (('b', (('x', 2), ('z', 10))),))

    #---------------------------------------------------------------------------
    def test_batch_display_a(self) -> None:

        dc = DisplayConfig.from_default(type_color=False)
        f1 = Frame.from_dict({'a':[1,2,3], 'b':[2,4,6], 'group': ['x','z','z']})

        gi = f1.iter_group_items('group')
        d1 = Batch(gi)[['a', 'b']].display(dc)

        self.assertEqual(d1.to_rows(),
            ['<Batch>', '<Index>',
            'x       <Frame>',
            'z       <Frame>',
            '<<U1>   <object>'
            ])

    #---------------------------------------------------------------------------
    def test_batch_shapes_a(self) -> None:

        dc = DisplayConfig.from_default(type_color=False)
        f1 = Frame.from_dict({'a':[1,2,3], 'b':[2,4,6], 'group': ['x','z','z']})

        b1 = Batch(f1.iter_group_items('group'))[['a', 'b']]

        self.assertEqual(b1.shapes.to_pairs(),
                (('x', (1, 2)), ('z', (2, 2)))
                )
        # import ipdb; ipdb.set_trace()


    #---------------------------------------------------------------------------
    def test_batch_apply_a(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('x', 'q'),
                name='f3')

        b1 = Batch.from_frames((f1, f2, f3))
        b2 = b1.apply(lambda x: x.via_str.replace('4', '_'))

        self.assertEqual(
                Frame.from_concat(b2.values, index=IndexAutoFactory, fill_value='').to_pairs(0),
                (('a', ((0, '1'), (1, '2'), (2, ''), (3, ''), (4, ''), (5, ''), (6, ''))), ('b', ((0, '3'), (1, '_'), (2, '_'), (3, '5'), (4, '6'), (5, '50'), (6, '60'))), ('c', ((0, ''), (1, ''), (2, '1'), (3, '2'), (4, '3'), (5, ''), (6, ''))), ('d', ((0, ''), (1, ''), (2, ''), (3, ''), (4, ''), (5, '10'), (6, '20')))))


    def test_batch_apply_b(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('x', 'q'),
                name='f3')

        b1 = Batch.from_frames((f1, f2, f3), use_threads=True, max_workers=8)
        b2 = b1.apply(lambda x: x.shape)
        self.assertEqual(dict(b2.items()),
                {'f1': (2, 2), 'f2': (3, 2), 'f3': (2, 2)}
)


    #---------------------------------------------------------------------------
    def test_batch_name_a(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')

        b1 = Batch.from_frames((f1, f2), name='foo')
        self.assertEqual(b1.name, 'foo')

        b2 = b1.rename('bar')
        self.assertEqual(b2.name, 'bar')
        self.assertEqual(tuple(b2.keys()), ('f1', 'f2'))


    #---------------------------------------------------------------------------
    def test_batch_ufunc_shape_a(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')

        b1 = Batch.from_frames((f1, f2), name='foo').cumsum()

        f1 = Frame.from_concat_items(b1.items(), fill_value=0)

        self.assertEqual(f1.to_pairs(0),
                (('a', ((('f1', 'x'), 1), (('f1', 'y'), 3), (('f2', 'x'), 0), (('f2', 'y'), 0), (('f2', 'z'), 0))), ('b', ((('f1', 'x'), 3), (('f1', 'y'), 7), (('f2', 'x'), 4), (('f2', 'y'), 9), (('f2', 'z'), 15))), ('c', ((('f1', 'x'), 0), (('f1', 'y'), 0), (('f2', 'x'), 1), (('f2', 'y'), 3), (('f2', 'z'), 6))))
                )

    #---------------------------------------------------------------------------
    def test_batch_iter_a(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')

        b1 = Batch.from_frames((f1, f2), name='foo').cumsum()
        self.assertEqual(list(b1), ['f1', 'f2'])


    #---------------------------------------------------------------------------
    def test_batch_to_bus_a(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')

        batch1 = Batch.from_frames((f1, f2))
        bus1 = batch1.to_bus()

        self.assertEqual(Frame.from_concat_items(bus1.items(), fill_value=0).to_pairs(0),
                (('a', ((('f1', 'x'), 1), (('f1', 'y'), 2), (('f2', 'x'), 0), (('f2', 'y'), 0), (('f2', 'z'), 0))), ('b', ((('f1', 'x'), 3), (('f1', 'y'), 4), (('f2', 'x'), 4), (('f2', 'y'), 5), (('f2', 'z'), 6))), ('c', ((('f1', 'x'), 0), (('f1', 'y'), 0), (('f2', 'x'), 1), (('f2', 'y'), 2), (('f2', 'z'), 3))))
                )

    #---------------------------------------------------------------------------
    def test_batch_iloc_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')

        b1 = Batch.from_frames((f1, f2))
        b2 = b1.iloc[1, 1]
        post = list(s.values.tolist() for s in b2.values)
        self.assertEqual(post, [[4], [5]])


    def test_batch_iloc_b(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')

        b1 = Batch.from_frames((f1, f2), max_workers=8, use_threads=True)
        b2 = b1.iloc[1, 1]
        post = list(s.values.tolist() for s in b2.values)
        self.assertEqual(post, [[4], [5]])


    #---------------------------------------------------------------------------
    def test_batch_bloc_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(10,20,0), b=(30,40,50)),
                index=('x', 'y', 'z'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')

        b1 = Batch.from_frames((f1, f2))
        b2 = b1.bloc[f2 >= 2]
        post = list(s.values.tolist() for s in b2.values)
        self.assertEqual(post, [[30, 40, 50], [4, 2, 5, 3, 6]])



if __name__ == '__main__':
    unittest.main()


