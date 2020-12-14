import unittest

import numpy as np
import frame_fixtures as ff

nan = np.nan

from static_frame.core.frame import Frame
from static_frame.core.batch import Batch
from static_frame.test.test_case import TestCase
from static_frame.core.index_auto import IndexAutoFactory
from static_frame.core.display_config import DisplayConfig
from static_frame.test.test_case import temp_file
from static_frame.core.store import StoreConfig


class TestUnit(TestCase):

    def test_batch_slotted_a(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='foo')

        b1 = Batch.from_frames((f1,))

        with self.assertRaises(AttributeError):
            b1.g = 30 # type: ignore #pylint: disable=E0237
        with self.assertRaises(AttributeError):
            b1.__dict__ #pylint: disable=W0104


    def test_batch_a(self) -> None:

        f1 = Frame.from_dict(
                {'a':[1,49,2,3], 'b':[2,4,381, 6], 'group': ['x', 'x','z','z']},
                index=('r', 's', 't', 'u'))

        b1 = Batch(f1.iter_group_items('group'))

        b2 = b1 * 3

        post = tuple(b2.items())
        self.assertEqual(post[0][1].to_pairs(0),
                (('a', (('r', 3), ('s', 147))), ('b', (('r', 6), ('s', 12))), ('group', (('r', 'xxx'), ('s', 'xxx')))),
                )

    def test_batch_b(self) -> None:

        f1 = Frame.from_dict(
                {'a':[1,49,2,3], 'b':[2,4,381, 6], 'group': ['x', 'x','z','z']},
                index=('r', 's', 't', 'u'))

        b1 = Batch(f1.iter_group_items('group'))
        self.assertEqual(b1['b'].sum().to_frame().to_pairs(0),
                ((None, (('x', 6), ('z', 387))),)
                )


    def test_batch_c1(self) -> None:

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

    def test_batch_c2(self) -> None:

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

        self.assertEqual(b1.loc['x'].to_frame(fill_value=0).to_pairs(0),
                (('a', (('f1', 1), ('f2', 0), ('f3', 0))), ('b', (('f1', 3), ('f2', 4), ('f3', 50))), ('c', (('f1', 0), ('f2', 1), ('f3', 0))), ('d', (('f1', 0), ('f2', 0), ('f3', 10))))
                )

    def test_batch_c3(self) -> None:

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

        self.assertEqual(b1.loc['x'].to_frame(fill_value=0, axis=1).to_pairs(0),
                (('f1', (('a', 1), ('b', 3), ('c', 0), ('d', 0))), ('f2', (('a', 0), ('b', 4), ('c', 1), ('d', 0))), ('f3', (('a', 0), ('b', 50), ('c', 0), ('d', 10))))
                )


    def test_batch_d(self) -> None:

        f1 = Frame.from_dict({'a':[1,2,3], 'b':[2,4,6], 'group': ['x','z','z']})

        f2 = Batch(f1.iter_group_items('group'))['b'].sum().to_frame()
        self.assertEqual(f2.to_pairs(0),
                ((None, (('x', 2), ('z', 10))),)
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
                ((None, (('x', 2), ('z', 10))),))


    def test_batch_g(self) -> None:
        f1 = Frame(np.arange(6).reshape(2,3), index=(('a', 'b')), columns=(('x', 'y', 'z')), name='f1')
        f2 = Frame(np.arange(6).reshape(2,3) * 30.5, index=(('a', 'b')), columns=(('x', 'y', 'z')), name='f2')

        # this results in two rows. one column labelled None
        f3 = Batch.from_frames((f1, f2)).sum().sum().to_frame()
        self.assertEqual(f3.to_pairs(0),
                ((None, (('f1', 15.0), ('f2', 457.5))),))

        f4 = Batch.from_frames((f1, f2)).apply(lambda f: f.iloc[0, 0]).to_frame()
        self.assertEqual(f4.to_pairs(0),
                ((None, (('f1', 0.0), ('f2', 0.0))),))


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

    def test_batch_repr_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        b1 = Batch.from_frames((f1, f2))
        self.assertTrue(repr(b1).startswith('<Batch at '))

        b2 = b1.rename('foo')
        self.assertTrue(repr(b2).startswith('<Batch: foo at '))

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

        b1 = Batch.from_frames((f1, f2, f3), use_threads=True, max_workers=8).apply(lambda x: x.shape)
        self.assertEqual(b1.to_frame().to_pairs(0),
                ((None, (('f1', (2, 2)), ('f2', (3, 2)), ('f3', (2, 2)))),)
                )

        f2 = Frame(np.arange(4).reshape(2, 2), name='f2')
        post = Batch.from_frames((f1, f2)).apply(lambda f: f.iloc[1, 1]).to_frame(fill_value=0.0)

        self.assertEqual(
                post.to_pairs(0),
                ((None, (('f1', 4), ('f2', 3))),)
                )

    #---------------------------------------------------------------------------
    def test_batch_apply_items_a(self) -> None:

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

        b1 = Batch.from_frames((f1, f2, f3)).apply_items(
                lambda k, x: (k, x['b'].mean()))

        self.assertEqual(b1.to_frame().to_pairs(0),
                ((None, (('f1', ('f1', 3.5)), ('f2', ('f2', 5.0)), ('f3', ('f3', 55.0)))),)
)
        b2 = Batch.from_frames((f1, f2, f3), use_threads=True, max_workers=8).apply_items(
                lambda k, x: (k, x['b'].mean()))

        self.assertEqual(b2.to_frame().to_pairs(0),
                ((None, (('f1', ('f1', 3.5)), ('f2', ('f2', 5.0)), ('f3', ('f3', 55.0)))),)
                )

    def test_batch_apply_items_b(self) -> None:

        f1 = ff.parse('s(20,4)|v(bool,bool,int,float)|c(I,str)|i(I,str)')

        b1 = Batch(f1.iter_group_items(['zZbu', 'ztsv'])).apply_items(lambda k, f: f.iloc[:1] if k != (True, True) else f.iloc[:3]).to_frame()

        self.assertEqual(b1.to_pairs(0),
            (('zZbu', ((((False, False), 'zZbu'), False), (((False, True), 'zr4u'), False), (((True, False), 'zkuW'), True), (((True, True), 'zIA5'), True), (((True, True), 'zGDJ'), True), (((True, True), 'zo2Q'), True))), ('ztsv', ((((False, False), 'zZbu'), False), (((False, True), 'zr4u'), True), (((True, False), 'zkuW'), False), (((True, True), 'zIA5'), True), (((True, True), 'zGDJ'), True), (((True, True), 'zo2Q'), True))), ('zUvW', ((((False, False), 'zZbu'), -3648), (((False, True), 'zr4u'), 197228), (((True, False), 'zkuW'), 54020), (((True, True), 'zIA5'), 194224), (((True, True), 'zGDJ'), 172133), (((True, True), 'zo2Q'), -88017))), ('zkuW', ((((False, False), 'zZbu'), 1080.4), (((False, True), 'zr4u'), 3884.48), (((True, False), 'zkuW'), 3338.48), (((True, True), 'zIA5'), -1760.34), (((True, True), 'zGDJ'), 1857.34), (((True, True), 'zo2Q'), 268.96))))
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


    #---------------------------------------------------------------------------
    def test_batch_to_frame_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(10,20,0), b=(30,40,50)),
                index=('x', 'y', 'z'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')

        b1 = Batch.from_frames((f1, f2))
        f3 = b1.loc['y':].to_frame(fill_value=0) #type: ignore
        self.assertEqual(f3.to_pairs(0),
                (('a', ((('f1', 'y'), 20), (('f1', 'z'), 0), (('f2', 'y'), 0), (('f2', 'z'), 0))), ('b', ((('f1', 'y'), 40), (('f1', 'z'), 50), (('f2', 'y'), 5), (('f2', 'z'), 6))), ('c', ((('f1', 'y'), 0), (('f1', 'z'), 0), (('f2', 'y'), 2), (('f2', 'z'), 3)))))

    def test_batch_to_frame_b(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(10,20,0), b=(30,40,50)),
                index=('x', 'y', 'z'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')

        b1 = Batch.from_frames((f1, f2))
        f3 = b1.loc['y':].to_frame(fill_value=0, index=IndexAutoFactory) #type: ignore

        self.assertEqual(f3.to_pairs(0),
                (('a', ((0, 20), (1, 0), (2, 0), (3, 0))), ('b', ((0, 40), (1, 50), (2, 5), (3, 6))), ('c', ((0, 0), (1, 0), (2, 2), (3, 3)))))


    def test_batch_to_frame_c(self) -> None:
        f1 = ff.parse('s(20,4)|v(bool,bool,int,float)|c(I,str)|i(I,str)')

        f2 = Batch(f1.iter_group_items(['zZbu', 'ztsv'])).apply(lambda f: f.iloc[0]).to_frame(index=IndexAutoFactory)
        self.assertEqual(f2.to_pairs(0),
                (('zZbu', ((0, False), (1, False), (2, True), (3, True))), ('ztsv', ((0, False), (1, True), (2, False), (3, True))), ('zUvW', ((0, -3648), (1, 197228), (2, 54020), (3, 194224))), ('zkuW', ((0, 1080.4), (1, 3884.48), (2, 3338.48), (3, -1760.34))))
                )

        f3 = Batch(f1.iter_group_items(['zZbu', 'ztsv'])).apply(lambda f: f.iloc[:2]).to_frame()
        self.assertEqual(f3.to_pairs(0),
                (('zZbu', ((((False, False), 'zZbu'), False), (((False, False), 'ztsv'), False), (((False, True), 'zr4u'), False), (((False, True), 'zmhG'), False), (((True, False), 'zkuW'), True), (((True, False), 'z2Oo'), True), (((True, True), 'zIA5'), True), (((True, True), 'zGDJ'), True))), ('ztsv', ((((False, False), 'zZbu'), False), (((False, False), 'ztsv'), False), (((False, True), 'zr4u'), True), (((False, True), 'zmhG'), True), (((True, False), 'zkuW'), False), (((True, False), 'z2Oo'), False), (((True, True), 'zIA5'), True), (((True, True), 'zGDJ'), True))), ('zUvW', ((((False, False), 'zZbu'), -3648), (((False, False), 'ztsv'), 91301), (((False, True), 'zr4u'), 197228), (((False, True), 'zmhG'), 96520), (((True, False), 'zkuW'), 54020), (((True, False), 'z2Oo'), 35021), (((True, True), 'zIA5'), 194224), (((True, True), 'zGDJ'), 172133))), ('zkuW', ((((False, False), 'zZbu'), 1080.4), (((False, False), 'ztsv'), 2580.34), (((False, True), 'zr4u'), 3884.48), (((False, True), 'zmhG'), 1699.34), (((True, False), 'zkuW'), 3338.48), (((True, False), 'z2Oo'), 3944.56), (((True, True), 'zIA5'), -1760.34), (((True, True), 'zGDJ'), 1857.34))))
                )




    #---------------------------------------------------------------------------
    def test_batch_drop_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(10,20,0), b=(30,40,50)),
                index=('x', 'y', 'z'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')

        b1 = Batch.from_frames((f1, f2))
        f3 = b1.drop.iloc[1, 1].to_frame(fill_value=0)

        self.assertEqual(f3.to_pairs(0),
                (('a', ((('f1', 'x'), 10), (('f1', 'z'), 0), (('f2', 'x'), 0), (('f2', 'z'), 0))), ('c', ((('f1', 'x'), 0), (('f1', 'z'), 0), (('f2', 'x'), 1), (('f2', 'z'), 3))))
                )


    #---------------------------------------------------------------------------
    def test_batch_sort_index_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(10,20,0), b=(30,40,50)),
                index=('z', 'y', 'x'),
                name='f1')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('y', 'z', 'x'),
                name='f2')

        f3 = Batch.from_frames((f1, f2)).sort_index().to_frame()
        self.assertEqual(f3.to_pairs(0),
                (('a', ((('f1', 'x'), 0), (('f1', 'y'), 20), (('f1', 'z'), 10), (('f2', 'x'), 3), (('f2', 'y'), 1), (('f2', 'z'), 2))), ('b', ((('f1', 'x'), 50), (('f1', 'y'), 40), (('f1', 'z'), 30), (('f2', 'x'), 6), (('f2', 'y'), 4), (('f2', 'z'), 5)))))


    #---------------------------------------------------------------------------
    def test_batch_sort_columns_a(self) -> None:
        f1 = Frame.from_dict(
                dict(b=(10,20,0), a=(30,40,50)),
                index=('z', 'y', 'x'),
                name='f1')
        f2 = Frame.from_dict(
                dict(b=(1,2,3), a=(4,5,6)),
                index=('y', 'z', 'x'),
                name='f2')

        f3 = Batch.from_frames((f1, f2)).sort_columns().to_frame()
        self.assertEqual(f3.to_pairs(0),
                (('a', ((('f1', 'z'), 30), (('f1', 'y'), 40), (('f1', 'x'), 50), (('f2', 'y'), 4), (('f2', 'z'), 5), (('f2', 'x'), 6))), ('b', ((('f1', 'z'), 10), (('f1', 'y'), 20), (('f1', 'x'), 0), (('f2', 'y'), 1), (('f2', 'z'), 2), (('f2', 'x'), 3)))))


    #---------------------------------------------------------------------------
    def test_batch_sort_values_a(self) -> None:
        f1 = Frame.from_dict(
                dict(b=(50,20,0), a=(30,40,50)),
                index=('z', 'y', 'x'),
                name='f1')
        f2 = Frame.from_dict(
                dict(b=(3,2,1), a=(4,5,6)),
                index=('y', 'z', 'x'),
                name='f2')

        f3 = Batch.from_frames((f1, f2)).sort_values('b').to_frame()
        self.assertEqual(f3.to_pairs(0),
                (('b', ((('f1', 'x'), 0), (('f1', 'y'), 20), (('f1', 'z'), 50), (('f2', 'x'), 1), (('f2', 'z'), 2), (('f2', 'y'), 3))), ('a', ((('f1', 'x'), 50), (('f1', 'y'), 40), (('f1', 'z'), 30), (('f2', 'x'), 6), (('f2', 'z'), 5), (('f2', 'y'), 4)))))



    #---------------------------------------------------------------------------
    def test_batch_isin_a(self) -> None:
        f1 = Frame.from_dict(
                dict(b=(10,20,0), a=(30,40,50)),
                index=('z', 'y', 'x'),
                name='f1')
        f2 = Frame.from_dict(
                dict(b=(1,20,3), a=(4,50,6)),
                index=('y', 'z', 'x'),
                name='f2')

        f3 = Batch.from_frames((f1, f2)).isin((20, 50)).to_frame()

        self.assertEqual(f3.to_pairs(0),
                (('b', ((('f1', 'z'), False), (('f1', 'y'), True), (('f1', 'x'), False), (('f2', 'y'), False), (('f2', 'z'), True), (('f2', 'x'), False))), ('a', ((('f1', 'z'), False), (('f1', 'y'), False), (('f1', 'x'), True), (('f2', 'y'), False), (('f2', 'z'), True), (('f2', 'x'), False))))
                )

    #---------------------------------------------------------------------------
    def test_batch_clip_a(self) -> None:
        f1 = Frame.from_dict(
                dict(b=(10,20,0), a=(30,40,50)),
                index=('z', 'y', 'x'),
                name='f1')
        f2 = Frame.from_dict(
                dict(b=(1,20,3), a=(4,50,6)),
                index=('y', 'z', 'x'),
                name='f2')

        f3 = Batch.from_frames((f1, f2)).clip(upper=22, lower=20).to_frame()
        self.assertEqual(f3.to_pairs(0),
                (('b', ((('f1', 'z'), 20), (('f1', 'y'), 20), (('f1', 'x'), 20), (('f2', 'y'), 20), (('f2', 'z'), 20), (('f2', 'x'), 20))), ('a', ((('f1', 'z'), 22), (('f1', 'y'), 22), (('f1', 'x'), 22), (('f2', 'y'), 20), (('f2', 'z'), 22), (('f2', 'x'), 20))))
                )
        # import ipdb; ipdb.set_trace()

    #---------------------------------------------------------------------------
    def test_batch_T_a(self) -> None:
        f1 = Frame.from_dict(
                dict(b=(10,20,0), a=(30,40,50)),
                index=('z', 'y', 'x'),
                name='f1')
        f2 = Frame.from_dict(
                dict(b=(1,20,3), a=(4,50,6)),
                index=('y', 'z', 'x'),
                name='f2')

        f3 = Batch.from_frames((f1, f2)).T.to_frame()
        # import ipdb; ipdb.set_trace()
        self.assertEqual(f3.to_pairs(0),
                (('x', ((('f1', 'b'), 0), (('f1', 'a'), 50), (('f2', 'b'), 3), (('f2', 'a'), 6))), ('y', ((('f1', 'b'), 20), (('f1', 'a'), 40), (('f2', 'b'), 1), (('f2', 'a'), 4))), ('z', ((('f1', 'b'), 10), (('f1', 'a'), 30), (('f2', 'b'), 20), (('f2', 'a'), 50))))
        )
    #---------------------------------------------------------------------------
    def test_batch_transpose_a(self) -> None:
        f1 = Frame.from_dict(
                dict(b=(10,20,0), a=(30,40,50)),
                index=('z', 'y', 'x'),
                name='f1')
        f2 = Frame.from_dict(
                dict(b=(1,20,3), a=(4,50,6)),
                index=('y', 'z', 'x'),
                name='f2')

        f3 = Batch.from_frames((f1, f2)).transpose().to_frame()
        self.assertEqual(f3.to_pairs(0),
                (('x', ((('f1', 'b'), 0), (('f1', 'a'), 50), (('f2', 'b'), 3), (('f2', 'a'), 6))), ('y', ((('f1', 'b'), 20), (('f1', 'a'), 40), (('f2', 'b'), 1), (('f2', 'a'), 4))), ('z', ((('f1', 'b'), 10), (('f1', 'a'), 30), (('f2', 'b'), 20), (('f2', 'a'), 50))))
        )

    #---------------------------------------------------------------------------
    def test_batch_duplicated_a(self) -> None:
        f1 = Frame.from_dict(
                dict(b=(20,20,0), a=(20,20,50)),
                index=('z', 'y', 'x'),
                name='f1')
        f2 = Frame.from_dict(
                dict(b=(1,20,1), a=(1,50,1)),
                index=('y', 'z', 'x'),
                name='f2')

        f3 = Batch.from_frames((f1, f2)).duplicated().to_frame()

        self.assertEqual(f3.to_pairs(0),
                (('x', (('f1', False), ('f2', True))), ('y', (('f1', True), ('f2', True))), ('z', (('f1', True), ('f2', False))))
                )

    #---------------------------------------------------------------------------
    def test_batch_drop_duplicated_a(self) -> None:
        f1 = Frame.from_dict(
                dict(b=(20,20,0), a=(20,20,50)),
                index=('z', 'y', 'x'),
                name='f1')
        f2 = Frame.from_dict(
                dict(b=(1,20,1), a=(1,50,1)),
                index=('y', 'z', 'x'),
                name='f2')

        f3 = Batch.from_frames((f1, f2)).drop_duplicated().to_frame()

        self.assertEqual(f3.to_pairs(0),
                (('b', ((('f1', 'x'), 0), (('f2', 'z'), 20))), ('a', ((('f1', 'x'), 50), (('f2', 'z'), 50))))
                )

    #---------------------------------------------------------------------------
    def test_batch_round_a(self) -> None:
        f1 = Frame.from_dict(
                dict(b=(20, 20.234, 0), a=(20.234, 20.234, 50.828)),
                index=('z', 'y', 'x'),
                name='f1')
        f2 = Frame.from_dict(
                dict(b=(1, 20.234, 1.043), a=(1.043, 50.828, 1.043)),
                index=('y', 'z', 'x'),
                name='f2')

        f3 = round(Batch.from_frames((f1, f2)), 1).to_frame() #type: ignore
        self.assertEqual(f3.to_pairs(0),
                (('b', ((('f1', 'z'), 20.0), (('f1', 'y'), 20.2), (('f1', 'x'), 0.0), (('f2', 'y'), 1.0), (('f2', 'z'), 20.2), (('f2', 'x'), 1.0))), ('a', ((('f1', 'z'), 20.2), (('f1', 'y'), 20.2), (('f1', 'x'), 50.8), (('f2', 'y'), 1.0), (('f2', 'z'), 50.8), (('f2', 'x'), 1.0))))
                )

    #---------------------------------------------------------------------------
    def test_batch_roll_a(self) -> None:
        f1 = Frame.from_dict(
                dict(b=(20,20,0), a=(20,20,50)),
                index=('z', 'y', 'x'),
                name='f1')
        f2 = Frame.from_dict(
                dict(b=(1,20,1), a=(1,50,1)),
                index=('y', 'z', 'x'),
                name='f2')

        f3 = Batch.from_frames((f1, f2)).roll(index=1, columns=-1).to_frame()

        self.assertEqual(f3.to_pairs(0),
                (('b', ((('f1', 'z'), 50), (('f1', 'y'), 20), (('f1', 'x'), 20), (('f2', 'y'), 1), (('f2', 'z'), 1), (('f2', 'x'), 50))), ('a', ((('f1', 'z'), 0), (('f1', 'y'), 20), (('f1', 'x'), 20), (('f2', 'y'), 1), (('f2', 'z'), 1), (('f2', 'x'), 20))))
                )

    #---------------------------------------------------------------------------
    def test_batch_shift_a(self) -> None:
        f1 = Frame.from_dict(
                dict(b=(20,20,0), a=(20,20,50)),
                index=('z', 'y', 'x'),
                name='f1')
        f2 = Frame.from_dict(
                dict(b=(1,20,1), a=(1,50,1)),
                index=('y', 'z', 'x'),
                name='f2')

        f3 = Batch.from_frames((f1, f2)).shift(index=1, columns=-1, fill_value=0).to_frame()

        self.assertEqual(f3.to_pairs(0),
                (('b', ((('f1', 'z'), 0), (('f1', 'y'), 20), (('f1', 'x'), 20), (('f2', 'y'), 0), (('f2', 'z'), 1), (('f2', 'x'), 50))), ('a', ((('f1', 'z'), 0), (('f1', 'y'), 0), (('f1', 'x'), 0), (('f2', 'y'), 0), (('f2', 'z'), 0), (('f2', 'x'), 0))))
                )


    #---------------------------------------------------------------------------
    def test_batch_head_a(self) -> None:
        f1 = Frame.from_dict(
                dict(b=(20,20,0), a=(20,20,50)),
                index=('z', 'y', 'x'),
                name='f1')
        f2 = Frame.from_dict(
                dict(b=(1,20,1), a=(1,50,1)),
                index=('y', 'z', 'x'),
                name='f2')

        f3 = Batch.from_frames((f1, f2)).head(1).to_frame()
        self.assertEqual(f3.to_pairs(0),
            (('b', ((('f1', 'z'), 20), (('f2', 'y'), 1))), ('a', ((('f1', 'z'), 20), (('f2', 'y'), 1))))
            )

    #---------------------------------------------------------------------------
    def test_batch_tail_a(self) -> None:
        f1 = Frame.from_dict(
                dict(b=(20,20,0), a=(20,20,50)),
                index=('z', 'y', 'x'),
                name='f1')
        f2 = Frame.from_dict(
                dict(b=(1,20,1), a=(1,50,1)),
                index=('y', 'z', 'x'),
                name='f2')

        f3 = Batch.from_frames((f1, f2)).tail(1).to_frame()
        self.assertEqual(f3.to_pairs(0),
            (('b', ((('f1', 'x'), 0), (('f2', 'x'), 1))), ('a', ((('f1', 'x'), 50), (('f2', 'x'), 1))))
            )

    #---------------------------------------------------------------------------
    def test_batch_loc_min_a(self) -> None:
        f1 = Frame.from_dict(
                dict(b=(20,20,0), a=(20,20,50)),
                index=('z', 'y', 'x'),
                name='f1')
        f2 = Frame.from_dict(
                dict(b=(1,20,1), a=(1,50,1)),
                index=('y', 'z', 'x'),
                name='f2')

        f3 = Batch.from_frames((f1, f2)).loc_min().to_frame()
        self.assertEqual(f3.to_pairs(0),
            (('b', (('f1', 'x'), ('f2', 'y'))), ('a', (('f1', 'z'), ('f2', 'y'))))
            )

    #---------------------------------------------------------------------------
    def test_batch_iloc_min_a(self) -> None:
        f1 = Frame.from_dict(
                dict(b=(20,20,0), a=(20,20,50)),
                index=('z', 'y', 'x'),
                name='f1')
        f2 = Frame.from_dict(
                dict(b=(1,20,1), a=(1,50,1)),
                index=('y', 'z', 'x'),
                name='f2')

        f3 = Batch.from_frames((f1, f2)).iloc_min().to_frame()
        self.assertEqual(f3.to_pairs(0),
            (('b', (('f1', 2), ('f2', 0))), ('a', (('f1', 0), ('f2', 0))))
            )

    #---------------------------------------------------------------------------
    def test_batch_loc_max_a(self) -> None:
        f1 = Frame.from_dict(
                dict(b=(20,20,0), a=(20,20,50)),
                index=('z', 'y', 'x'),
                name='f1')
        f2 = Frame.from_dict(
                dict(b=(1,20,1), a=(1,50,1)),
                index=('y', 'z', 'x'),
                name='f2')

        f3 = Batch.from_frames((f1, f2)).loc_max().to_frame()
        self.assertEqual(f3.to_pairs(0),
            (('b', (('f1', 'z'), ('f2', 'z'))), ('a', (('f1', 'x'), ('f2', 'z'))))
            )

    #---------------------------------------------------------------------------
    def test_batch_iloc_max_a(self) -> None:
        f1 = Frame.from_dict(
                dict(b=(20,20,0), a=(20,20,50)),
                index=('z', 'y', 'x'),
                name='f1')
        f2 = Frame.from_dict(
                dict(b=(1,20,1), a=(1,50,1)),
                index=('y', 'z', 'x'),
                name='f2')

        f3 = Batch.from_frames((f1, f2)).iloc_max().to_frame()
        self.assertEqual(f3.to_pairs(0),
            (('b', (('f1', 0), ('f2', 1))), ('a', (('f1', 2), ('f2', 1))))
            )

    #---------------------------------------------------------------------------
    def test_batch_count_a(self) -> None:
        f1 = Frame.from_dict(
                dict(b=(20,20,0), a=(20,20,np.nan)),
                index=('z', 'y', 'x'),
                name='f1')
        f2 = Frame.from_dict(
                dict(b=(1,np.nan,1), a=(1,50,1)),
                index=('y', 'z', 'x'),
                name='f2')

        self.assertEqual(
                Batch.from_frames((f1, f2)).count(axis=0).to_frame().to_pairs(0),
            (('b', (('f1', 3), ('f2', 2))), ('a', (('f1', 2), ('f2', 3)))))

        self.assertEqual(
            Batch.from_frames((f1, f2)).count(axis=1).to_frame().to_pairs(0),
            (('x', (('f1', 1), ('f2', 2))), ('y', (('f1', 2), ('f2', 2))), ('z', (('f1', 2), ('f2', 1))))
            )

    #---------------------------------------------------------------------------
    def test_batch_to_zip_pickle_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(a=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        config = StoreConfig(
                index_depth=1,
                columns_depth=1,
                include_columns=True,
                include_index=True
                )

        b1 = Batch.from_frames((f1, f2, f3))

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp, config=config)
            b2 = Batch.from_zip_pickle(fp, config=config)
            frames = dict(b2.items())

        for frame in (f1, f2, f3):
            # parquet brings in characters as objects, thus forcing different dtypes
            self.assertEqualFrames(frame, frames[frame.name], compare_dtype=False)

    #---------------------------------------------------------------------------
    def test_batch_to_xlsx_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(a=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        config = StoreConfig(
                index_depth=1,
                columns_depth=1,
                include_columns=True,
                include_index=True
                )

        b1 = Batch.from_frames((f1, f2, f3))

        with temp_file('.xlsx') as fp:
            b1.to_xlsx(fp)
            b2 = Batch.from_xlsx(fp, config=config)
            frames = dict(b2.items())

        for frame in (f1, f2, f3):
            # parquet brings in characters as objects, thus forcing different dtypes
            self.assertEqualFrames(frame, frames[frame.name], compare_dtype=False)


    def test_batch_to_zip_parquet_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')

        config = StoreConfig(
                index_depth=1,
                columns_depth=1,
                include_columns=True,
                include_index=True
                )

        b1 = Batch.from_frames((f1, f2), config=config)

        with temp_file('.xlsx') as fp:
            b1.to_xlsx(fp)
            b2 = (Batch.from_xlsx(fp, config=config) * 20).sum()

            self.assertEqual(b2.to_frame().to_pairs(0),
                (('a', (('f1', 60), ('f2', 120))), ('b', (('f1', 140), ('f2', 300)))))


    #---------------------------------------------------------------------------
    def test_batch_sample_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')

        self.assertEqual(
                Batch.from_frames((f1, f2)).sample(1, 1, seed=22).to_frame().to_pairs(0),
                (('a', ((('f1', 'x'), 1), (('f2', 'z'), 3))),)
                )

if __name__ == '__main__':
    unittest.main()


