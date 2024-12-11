from __future__ import annotations

import datetime
import time

import frame_fixtures as ff
import numpy as np

from static_frame.core.batch import Batch
from static_frame.core.batch import normalize_container
from static_frame.core.display_config import DisplayConfig
from static_frame.core.exception import BatchIterableInvalid
from static_frame.core.exception import ErrorInitFrame
from static_frame.core.exception import StoreLabelNonUnique
from static_frame.core.frame import Frame
from static_frame.core.index_auto import IndexAutoFactory
from static_frame.core.index_datetime import IndexDate
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.node_dt import InterfaceBatchDatetime
from static_frame.core.quilt import Quilt
from static_frame.core.series import Series
from static_frame.core.store_config import StoreConfig
from static_frame.core.util import TLabel
from static_frame.test.test_case import TestCase
from static_frame.test.test_case import skip_no_hdf5
from static_frame.test.test_case import temp_file

nan = np.nan

def func1(f: Frame) -> Frame:
    return f.loc['q']


def func2(label: TLabel, f: Frame) -> Frame:
    return f.loc['q']

class TestUnit(TestCase):

    def test_normalize_container_a(self) -> None:
        post = normalize_container(np.arange(8).reshape(2, 4))
        self.assertEqual(post.to_pairs(),
                ((0, ((0, 0), (1, 4))), (1, ((0, 1), (1, 5))), (2, ((0, 2), (1, 6))), (3, ((0, 3), (1, 7))))
                )

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

    def test_batch_h(self) -> None:

        frame = ff.parse('s(10,3)')
        with self.assertRaises(BatchIterableInvalid):
            Batch(frame.iter_window(size=3)).std(ddof=1).to_frame()

    def test_batch_i(self) -> None:

        f1 = ff.parse('s(3,2)|v(bool)|c(I,str)|i(I,int)')
        f2 = ff.parse('s(3,5)|v(bool)|c(I,str)|i(I,int)')

        post = Batch.from_frames((f1, f2)).drop['zZbu']
        self.assertEqual(
            [list(v.columns) for _, v in post.items()],
            [['ztsv'], ['ztsv', 'zUvW', 'zkuW', 'zmVj']]
            )

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
        self.assertEqual(repr(b1), ('<Batch max_workers=None>'))

        b2 = Batch.from_frames((f1, f2), max_workers=3)
        self.assertEqual(repr(b2), ('<Batch max_workers=3>'))

    #---------------------------------------------------------------------------

    def test_batch_shapes_a(self) -> None:

        dc = DisplayConfig.from_default(type_color=False)
        f1 = Frame.from_dict({'a':[1,2,3], 'b':[2,4,6], 'group': ['x','z','z']})

        b1 = Batch(f1.iter_group_items('group'))[['a', 'b']]

        self.assertEqual(b1.shapes.to_pairs(),
                (('x', (1, 2)), ('z', (2, 2)))
                )

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

    def test_batch_apply_except_a(self) -> None:

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

        post = Batch.from_frames((f1, f2, f3)
                ).apply_except(lambda f: f.loc['q'], KeyError).to_frame()
        self.assertEqual(post.to_pairs(),
                (('d', (('f3', 20),)), ('b', (('f3', 60),))))

    def test_batch_apply_except_b(self) -> None:

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

        post = Batch.from_frames((f1, f2, f3), max_workers=3
                ).apply_except(func1, KeyError).to_frame()
        self.assertEqual(post.to_pairs(),
                (('d', (('f3', 20),)), ('b', (('f3', 60),))))

        with self.assertRaises(NotImplementedError):
            _ = Batch.from_frames((f1, f2, f3), max_workers=3, chunksize=2,
                    ).apply_except(func1, KeyError).to_frame()

    def test_batch_apply_except_c(self) -> None:

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

        post = Batch.from_frames((f1, f2, f3)
                ).apply_items_except(lambda label, f: f.loc['q'], KeyError).to_frame()
        self.assertEqual(post.to_pairs(),
                (('d', (('f3', 20),)), ('b', (('f3', 60),))))

    def test_batch_apply_except_d(self) -> None:

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

        post = Batch.from_frames((f1, f2, f3), max_workers=3
                ).apply_items_except(func2, KeyError).to_frame()
        self.assertEqual(post.to_pairs(),
                (('d', (('f3', 20),)), ('b', (('f3', 60),))))

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
        self.assertTrue(repr(b1).startswith('<Batch: foo'))

        b2 = b1.rename('bar') # this rename contained Frame
        self.assertEqual(tuple(f.name for f in b2.values), ('bar', 'bar'))

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

    def test_batch_to_bus_b(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(False, True), b=(True, True)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(False, True, False), b=(False, False, True)),
                index=('x', 'y', 'z'),
                name='f2')

        batch1 = Batch.from_frames((f1, f2))
        post = batch1._to_signature_bytes(include_name=False)
        self.assertEqual(post, b'BusIndexf\x00\x00\x001\x00\x00\x00f\x00\x00\x002\x00\x00\x00FrameIndexx\x00\x00\x00y\x00\x00\x00Indexa\x00\x00\x00b\x00\x00\x00\x00\x01\x01\x01FrameIndexx\x00\x00\x00y\x00\x00\x00z\x00\x00\x00Indexc\x00\x00\x00b\x00\x00\x00\x00\x01\x00\x00\x00\x01')

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
        post = list(b2.values)
        self.assertEqual(post, [4, 5])

        b3 = Batch.from_frames((f1, f2))

        b4 = b3.iloc[1, 1].via_container
        post = list(s.values.tolist() for s in b4.values)
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
        b2 = b1.iloc[1, 1].via_container
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
        post = list(s.to_pairs() for s in b2.values)

        self.assertEqual(post,
            [((('x', 'b'), 30), (('y', 'b'), 40), (('z', 'b'), 50)),
             ((('x', 'b'), 4),
              (('y', 'c'), 2),
              (('y', 'b'), 5),
              (('z', 'c'), 3),
              (('z', 'b'), 6))])

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

    def test_batch_to_frame_d(self) -> None:
        f1 = ff.parse('s(12,2)|v(bool)|i(ID,dtD)')
        f2 = Batch(f1.iter_window_items(size=8)).std().to_frame(index_constructor=IndexDate)

        self.assertEqual(f2.index.__class__, IndexDate)
        self.assertEqual(round(f2, 1).to_pairs(),
                ((0, ((np.datetime64('2427-01-09'), 0.5), (np.datetime64('2304-09-13'), 0.5), (np.datetime64('2509-12-29'), 0.5), (np.datetime64('2258-03-21'), 0.5), (np.datetime64('2298-04-20'), 0.5))), (1, ((np.datetime64('2427-01-09'), 0.0), (np.datetime64('2304-09-13'), 0.3), (np.datetime64('2509-12-29'), 0.3), (np.datetime64('2258-03-21'), 0.3), (np.datetime64('2298-04-20'), 0.4))))
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

    def test_batch_drop_b(self) -> None:

        f1 = ff.parse('s(3,2)|v(bool)|c(I,str)|i(I,int)')
        f2 = ff.parse('s(3,5)|v(bool)|c(I,str)|i(I,int)').rename('b')

        post = Batch.from_frames((f1, f2)).drop['zZbu']
        self.assertEqual(
            [list(v.columns) for _, v in post.items()],
            [['ztsv'], ['ztsv', 'zUvW', 'zkuW', 'zmVj']]
            )

    def test_batch_drop_c(self) -> None:

        f1 = ff.parse('s(3,2)|v(bool)|c(I,str)|i(I,int)')
        f2 = ff.parse('s(3,5)|v(bool)|c(I,str)|i(I,int)').rename('b')

        post = Batch.from_frames((f1, f2)).drop.loc[-3648:, 'zZbu']
        self.assertEqual(
            [list(v.columns) for _, v in post.items()], #type: ignore
            [['ztsv'], ['ztsv', 'zUvW', 'zkuW', 'zmVj']]
            )

        post = Batch.from_frames((f1, f2)).drop.loc[-3648:, 'zZbu']
        self.assertEqual(
            [list(v.index) for _, v in post.items()],
            [[34715], [34715]]
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

        f3 = round(Batch.from_frames((f1, f2)), 1).to_frame()
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
    def test_batch_isna(self) -> None: # also tests `notna()`
        f0 = ff.parse('v(str,str,bool,float)|s(9,4)').assign[3]([None, 100.0, 632.23, None, 12.5, 51526.002, None, None, 0.231])
        actual1 = Batch.from_frames((
            f0.iloc[0:3].rename('1'),
            f0.iloc[3:6].rename('2'),
            f0.iloc[6:9].rename('3'),
        )).isna().to_frame()[3].values.tolist()
        actual2 = Batch.from_frames((
            f0.iloc[0:3].rename('1'),
            f0.iloc[3:6].rename('2'),
            f0.iloc[6:9].rename('3'),
        )).notna().to_frame()[3].values
        actual2 = np.invert(actual2).tolist()

        expected = [True, False, False, True, False, False, True, True, False]
        self.assertEqual(actual1, actual2)
        self.assertEqual(actual2, expected)

    def test_batch_dropna(self) -> None:
        f0 = ff.parse('v(str,str,bool,float)|s(9,4)').assign[3]([None, 100.0, 632.23, None, 12.5, 51526.002, None, None, 0.231])
        f = Batch.from_frames((
            f0.iloc[0:3].rename('1'),
            f0.iloc[3:6].rename('2'),
            f0.iloc[6:9].rename('3'),
        )).dropna(condition=np.any).to_frame()

        actual_index = f.index.values[:,-1].astype(int).tolist()
        expected_index = [1,2,4,5,8]
        self.assertEqual(actual_index, expected_index)

    #---------------------------------------------------------------------------
    def test_batch_isfalsy(self) -> None: # also tests `notfalsy()`
        f0 = ff.parse('v(str,str,bool,float)|s(9,4)')
        actual1 = Batch.from_frames((
            f0.iloc[0:3].rename('1'),
            f0.iloc[6:9].rename('3'),
            f0.iloc[3:6].rename('2'),
        )).isfalsy().to_frame()[2].values.tolist()
        actual2 = Batch.from_frames((
            f0.iloc[0:3].rename('1'),
            f0.iloc[6:9].rename('3'),
            f0.iloc[3:6].rename('2'),
        )).notfalsy().to_frame()[2].values
        actual2 = np.invert(actual2).tolist()

        expected = [False, True, True, False, False, False, False, True, True]
        self.assertEqual(actual1, actual2)
        self.assertEqual(actual2, expected)

    def test_batch_fillna_a(self) -> None:
        f0 = ff.parse('v(str,str,bool,int)|s(9,4)').assign.loc[4:,0]('')
        f1 = Batch.from_frames((
            f0.iloc[0:3].rename('1'),
            f0.iloc[3:6].rename('2'),
            f0.iloc[6:9].rename('3'),
        )).fillfalsy('filled').to_frame()
        expected = ['zjZQ', 'zO5l', 'zEdH', 'zB7E', 'filled', 'filled', 'filled', 'filled', 'filled']
        actual = f1[0].values.tolist()
        self.assertEqual(expected, actual)

    def test_batch_fillna_b(self) -> None:
        f0 = ff.parse('v(str,str,bool,int)|s(9,4)').assign[3]([None, 100, 632, None, 12, 51526, None, None, 231])
        f1 = Batch.from_frames((
            f0.iloc[0:3].rename('1'),
            f0.iloc[3:6].rename('2'),
            f0.iloc[6:9].rename('3'),
        )).fillna(0).to_frame()
        expected = [0, 100, 632, 0, 12, 51526, 0, 0, 231]
        actual = f1[3].values.tolist()
        self.assertEqual(expected, actual)

    def test_batch_dropfalsy(self) -> None:
        f0 = ff.parse('v(str,str,bool,float)|s(9,4)')
        f4 = Batch.from_frames((
            f0.iloc[0:3].rename('1'),
            f0.iloc[3:6].rename('2'),
            f0.iloc[6:9].rename('3'),
        )).dropfalsy(condition=np.any).to_frame()
        actual_index = f4.index.values[:,-1].astype(int).tolist()
        expected_index = [0,3,6,7,8]
        self.assertEqual(actual_index, expected_index)

    #---------------------------------------------------------------------------
    def test_batch_fillna_leading(self) -> None:
        f0 = ff.parse('v(int,str,float,str)|s(9,4)')
        f0 = f0.assign[0]([None if i in (0,3,6) else x for i,x in enumerate(f0[0].values)])
        f = Batch.from_frames((
            f0.iloc[0:3].rename('1'),
            f0.iloc[3:6].rename('2'),
            f0.iloc[6:9].rename('3'),
        )).fillna_leading(value=123456789).to_frame()

        expected = [123456789, 92867, 84967, 123456789, 175579, 58768, 123456789, 170440, 32395]
        actual = f[0].values.astype(int).tolist()
        self.assertEqual(expected, actual)

    def test_batch_fillna_trailing(self) -> None:
        f0 = ff.parse('v(int,str,float,str)|s(9,4)')
        f0 = f0.assign[0]([None if i in (2,5,8) else x for i,x in enumerate(f0[0].values)])
        f = Batch.from_frames((
            f0.iloc[0:3].rename('1'),
            f0.iloc[3:6].rename('2'),
            f0.iloc[6:9].rename('3'),
        )).fillna_trailing(value=123456789).to_frame()

        expected = [-88017, 92867, 123456789, 13448, 175579, 123456789, 146284, 170440, 123456789]
        actual = f[0].values.astype(int).tolist()
        self.assertEqual(expected, actual)

    def test_batch_fillna_forward(self) -> None:
        f0 = ff.parse('v(int,str,bool,str)|s(9,4)').assign[0]([1,None,3,4,None,6,7,None,9])
        f = Batch.from_frames((
            f0.iloc[0:3].rename('1'),
            f0.iloc[3:6].rename('2'),
            f0.iloc[6:9].rename('3'),
        )).fillna_forward().to_frame()

        expected = [1,1,3,4,4,6,7,7,9]
        actual = f[0].values.astype(int).tolist()
        self.assertEqual(expected, actual)

    def test_batch_fillna_backward(self) -> None:
        f0 = ff.parse('v(int,str,bool,str)|s(9,4)').assign[0]([1,None,3,4,None,6,7,None,9])
        f = Batch.from_frames((
            f0.iloc[0:3].rename('1'),
            f0.iloc[3:6].rename('2'),
            f0.iloc[6:9].rename('3'),
        )).fillna_backward().to_frame()

        expected = [1,3,3,4,6,6,7,9,9]
        actual = f[0].values.astype(int).tolist()
        self.assertEqual(expected, actual)

    #---------------------------------------------------------------------------
    def test_batch_fillfalsy_leading(self) -> None:
        f0 = ff.parse('v(str,str,float,str)|s(9,4)')
        f0 = f0.assign[0](['' if i in (0,3,6) else x for i,x in enumerate(f0[0].values)])
        f = Batch.from_frames((
            f0.iloc[0:3].rename('1'),
            f0.iloc[3:6].rename('2'),
            f0.iloc[6:9].rename('3'),
        )).fillfalsy_leading(value='--leading--').to_frame()

        expected = ['--leading--', 'zO5l', 'zEdH', '--leading--', 'zwIp', 'zDVQ', '--leading--', 'zyT8', 'zS6w']
        actual = f[0].values.tolist()
        self.assertEqual(expected, actual)

    def test_batch_fillfalsy_trailing(self) -> None:
        f0 = ff.parse('v(str,str,float,int)|s(9,4)')
        f0 = f0.assign[0](['' if i in (2,5,8) else x for i,x in enumerate(f0[0].values)])
        f = Batch.from_frames((
            f0.iloc[0:3].rename('1'),
            f0.iloc[3:6].rename('2'),
            f0.iloc[6:9].rename('3'),
        )).fillfalsy_trailing(value='--trailing--').to_frame()

        expected = ['zjZQ', 'zO5l', '--trailing--', 'zB7E', 'zwIp', '--trailing--', 'z5hI', 'zyT8', '--trailing--']
        actual = f[0].values.tolist()
        self.assertEqual(expected, actual)

    def test_batch_fillfalsy_forward(self) -> None:
        f0 = ff.parse('v(int,str,bool,str)|s(9,4)').assign[0]([1,0,3,4,0,6,7,0,9])
        f = Batch.from_frames((
            f0.iloc[0:3].rename('1'),
            f0.iloc[3:6].rename('2'),
            f0.iloc[6:9].rename('3'),
        )).fillfalsy_forward().to_frame()

        expected = [1,1,3,4,4,6,7,7,9]
        self.assertEqual(expected, f[0].values.tolist())

    def test_batch_fillfalsy_backward(self) -> None:
        f0 = ff.parse('v(int,str,bool,str)|s(9,4)').assign[0]([1,0,3,4,0,6,7,0,9])
        f = Batch.from_frames((
            f0.iloc[0:3].rename('1'),
            f0.iloc[3:6].rename('2'),
            f0.iloc[6:9].rename('3'),
        )).fillfalsy_backward().to_frame()

        expected = [1,3,3,4,6,6,7,9,9]
        self.assertEqual(expected, f[0].values.tolist())

    #---------------------------------------------------------------------------
    def test_batch_unset_index(self) -> None:
        f0 = ff.parse('v(int,str,bool,str)|s(9,4)')
        f = Batch.from_frames((
            f0.iloc[0:2].rename('1'),
            f0.iloc[2:5].rename('2'),
            f0.iloc[5:9].rename('3'),
        )).unset_index().to_frame()
        actual = f.index.values[:,-1].astype(int).tolist()
        expected = [*range(2), *range(3), *range(4)]
        self.assertEqual(actual, expected)

    #---------------------------------------------------------------------------
    def test_batch_reindex(self) -> None:
        f0 = ff.parse('v(int,str,bool,str)|s(9,4)')
        f = list(Batch.from_frames((
            f0.iloc[0:2].rename('1'),
            f0.iloc[2:5].rename('2'),
            f0.iloc[5:9].rename('3'),
        )).reindex(index=list(range(9))).items())
        a = [(x[1].shape) for x in f]
        self.assertTrue(a.count(a[0]) == 3)

    #---------------------------------------------------------------------------
    def test_batch_relabel_a(self) -> None:
        f0 = ff.parse('v(int,str,bool,str)|s(9,4)')
        f1 = Batch.from_frames((
            f0.iloc[0:3].rename('1'),
            f0.iloc[3:6].rename('2'),
            f0.iloc[6:9].rename('3'),
        )).relabel(columns={2:'two'}).to_frame()
        actual = f1.columns.values.tolist()
        expected = [0, 1, 'two', 3]
        self.assertEqual(expected, actual)

    def test_batch_relabel_b(self) -> None:
        f0 = ff.parse('v(int,str,bool,str)|s(9,4)')
        with self.assertRaises(ErrorInitFrame):
            f1 = Batch.from_frames((
                f0.iloc[0:2].rename('1'),
                f0.iloc[2:8].rename('2'),
            )).relabel(index=('a', 'b')).to_frame()

    #---------------------------------------------------------------------------
    def test_batch_relabel_level_add_drop(self) -> None:
        f0 = ff.parse('v(int,str,bool,str)|s(9,4)')

        f123 = list(Batch.from_frames((
                f0.iloc[0:3].rename('1'),
                f0.iloc[3:6].rename('2'),
                f0.iloc[6:9].rename('3'),
                )).relabel_level_add('removeme').items())
        expected = [['removeme', i] for i in range(9)]
        actual = [*f123[0][1].index.values.tolist(),
                *f123[1][1].index.values.tolist(),
                *f123[2][1].index.values.tolist()]
        self.assertEqual(expected, actual)

        f123 = list(Batch.from_frames(f for _, f in f123).relabel_level_drop(index=1).items())
        expected = list(range(9))
        actual = [*f123[0][1].index.values.tolist(), *f123[1][1].index.values.tolist(), *f123[2][1].index.values.tolist()]
        self.assertEqual(expected, actual)


    def test_batch_relabel_shift_flat(self) -> None:
        f0 = ff.parse('v(int,str,bool,str)|s(9,4)')
        f123 = list(Batch.from_frames((
            f0.iloc[0:3].rename('1'),
            f0.iloc[3:6].rename('2'),
            f0.iloc[6:9].rename('3'),
        )).relabel_shift_in(2, axis=0).relabel_flat(index=1).items())
        expected = [(i,bool(x)) for i,x in enumerate([1,0,0,1,0,0,1,1,1])]
        actual = [*f123[0][1].index.values.tolist(), *f123[1][1].index.values.tolist(), *f123[2][1].index.values.tolist()]
        self.assertEqual(expected, actual)

    #---------------------------------------------------------------------------
    def test_batch_rank_dense(self) -> None:
        i = [1,4,7,2,5,8,3,6,9]
        f0 = Frame.from_items(
            (
                ('i',i),
                ('b',(b%2==0 for b in i))
            )
        )
        f123 = (f0.iloc[0:3], f0.iloc[3:6], f0.iloc[6:9])
        b123 = Batch.from_frames(f123).rank_dense()
        for i,(_, f) in enumerate(b123.items()): # type: ignore
            self.assertEqual(f.values.tolist(), f123[i].rank_dense().values.tolist()) # type: ignore

    def test_batch_rank_max(self) -> None:
        i = [1,4,7,2,5,8,3,6,9]
        f0 = Frame.from_items(
            (
                ('i',i),
                ('b',(b%2==0 for b in i))
            )
        )
        f123 = (f0.iloc[0:3],f0.iloc[3:6],f0.iloc[6:9])
        b123 = Batch.from_frames(f123).rank_max()
        for i,(_, f) in enumerate(b123.items()): # type: ignore
            self.assertEqual(f.values.tolist(), f123[i].rank_max().values.tolist()) # type: ignore

    def test_batch_rank_mean(self) -> None:
        i = [1,4,7,2,5,8,3,6,9]
        f0 = Frame.from_items(
            (
                ('i',i),
                ('b',(b%2==0 for b in i))
            )
        )
        f123 = (f0.iloc[0:3],f0.iloc[3:6],f0.iloc[6:9])
        b123 = Batch.from_frames(f123).rank_mean()
        for i,(_, f) in enumerate(b123.items()): # type: ignore
            self.assertEqual(f.values.tolist(), f123[i].rank_mean().values.tolist()) # type: ignore

    def test_batch_rank_min(self) -> None:
        i = [1,4,7,2,5,8,3,6,9]
        f0 = Frame.from_items(
            (
                ('i',i),
                ('b',(b%2==0 for b in i))
            )
        )
        f123 = (f0.iloc[0:3],f0.iloc[3:6],f0.iloc[6:9])
        b123 = Batch.from_frames(f123).rank_min()
        for i,(_, f) in enumerate(b123.items()): # type: ignore
            self.assertEqual(f.values.tolist(), f123[i].rank_min().values.tolist()) # type: ignore

    def test_batch_rank_ordinal(self) -> None:
        i = [1,4,7,2,5,8,3,6,9]
        f0 = Frame.from_items(
            (
                ('i',i),
                ('b',(b%2==0 for b in i))
            )
        )
        f123 = (f0.iloc[0:3],f0.iloc[3:6],f0.iloc[6:9])
        b123 = Batch.from_frames(f123).rank_ordinal()
        for i,(_, f) in enumerate(b123.items()): # type: ignore
            self.assertEqual(f.values.tolist(), f123[i].rank_ordinal().values.tolist()) # type: ignore

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

    def test_batch_cov_a(self) -> None:
        f1 = Frame.from_dict(
                dict(b=(1,2,3), a=(4,5,6)),
                index=('z', 'y', 'x'),
                name='f1')
        f2 = Frame.from_dict(
                dict(b=(1,10,100), a=(1,2,3)),
                index=('y', 'z', 'x'),
                name='f2')

        f3 = Batch.from_frames((f1, f2)).cov().to_frame()
        self.assertEqual(f3.to_pairs(),
                (('b', ((('f1', 'b'), 1.0), (('f1', 'a'), 1.0), (('f2', 'b'), 2997.0), (('f2', 'a'), 49.5))), ('a', ((('f1', 'b'), 1.0), (('f1', 'a'), 1.0), (('f2', 'b'), 49.5), (('f2', 'a'), 1.0)))))

        f4 = Batch.from_frames((f1, f2)).cov(axis=0).to_frame()
        self.assertEqual( f4.to_pairs(),
                (('x', ((('f1', 'z'), 4.5), (('f1', 'y'), 4.5), (('f1', 'x'), 4.5), (('f2', 'y'), 0.0), (('f2', 'z'), 388.0), (('f2', 'x'), 4704.5))), ('y', ((('f1', 'z'), 4.5), (('f1', 'y'), 4.5), (('f1', 'x'), 4.5), (('f2', 'y'), 0.0), (('f2', 'z'), 0.0), (('f2', 'x'), 0.0))), ('z', ((('f1', 'z'), 4.5), (('f1', 'y'), 4.5), (('f1', 'x'), 4.5), (('f2', 'y'), 0.0), (('f2', 'z'), 32.0), (('f2', 'x'), 388.0)))))


    #---------------------------------------------------------------------------

    def test_batch_corr_a(self) -> None:
        f1 = Frame.from_dict(
                dict(b=(1,2,3), a=(4,5,6)),
                index=('z', 'y', 'x'),
                name='f1')
        f2 = Frame.from_dict(
                dict(b=(1,10,100), a=(1,2,3)),
                index=('y', 'z', 'x'),
                name='f2')

        f3 = Batch.from_frames((f1, f2)).corr().to_frame()
        self.assertEqual(round(f3, 6).to_pairs(),
                (('b', ((('f1', 'b'), 1.0), (('f1', 'a'), 1.0), (('f2', 'b'), 1.0), (('f2', 'a'), 0.904194))), ('a', ((('f1', 'b'), 1.0), (('f1', 'a'), 1.0), (('f2', 'b'), 0.904194), (('f2', 'a'), 1.0))))
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

    def test_batch_count_b(self) -> None:
        s = Series((1, 1, 1, 3, 3, 8, 8, 8, 8))
        post = Batch(s.iter_group_items()).count().to_series()
        self.assertEqual(post.to_pairs(), ((1, 3), (3, 2), (8, 4)))




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

    def test_batch_to_zip_pickle_b(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f')

        b1 = Batch.from_frames((f1, f2))

        with temp_file('.zip') as fp:
            with self.assertRaises(StoreLabelNonUnique):
                b1.to_zip_pickle(fp)


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

    #---------------------------------------------------------------------------

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

        with temp_file('.zip') as fp:
            b1.to_zip_parquet(fp)
            b2 = Batch.from_zip_parquet(fp, config=config)
            frames = dict(b2.items())

        for frame in (f1, f2):
            # parquet brings in characters as objects, thus forcing different dtypes
            self.assertEqualFrames(frame, frames[frame.name], compare_dtype=False)

    #---------------------------------------------------------------------------

    def test_batch_from_zip_tsv_a(self) -> None:
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

        with temp_file('.zip') as fp:
            b1.to_zip_tsv(fp)
            b2 = Batch.from_zip_tsv(fp, config=config)
            frames = dict(b2.items())

        for frame in (f1, f2):
            # parquet brings in characters as objects, thus forcing different dtypes
            self.assertEqualFrames(frame, frames[frame.name], compare_dtype=False)

    def test_batch_from_zip_csv_a(self) -> None:
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

        with temp_file('.zip') as fp:
            b1.to_zip_csv(fp)
            b2 = Batch.from_zip_csv(fp, config=config)
            frames = dict(b2.items())

        for frame in (f1, f2):
            # parquet brings in characters as objects, thus forcing different dtypes
            self.assertEqualFrames(frame, frames[frame.name], compare_dtype=False)

    #---------------------------------------------------------------------------

    def test_batch_to_sqlite_a(self) -> None:
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

        with temp_file('.sqlite') as fp:
            b1.to_sqlite(fp)
            b2 = Batch.from_sqlite(fp, config=config)
            frames = dict(b2.items())

        for frame in (f1, f2):
            # brings in characters as objects, thus forcing different dtypes
            self.assertEqualFrames(frame, frames[frame.name], compare_dtype=False)

    #---------------------------------------------------------------------------

    def test_batch_to_duckdb_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1',
                dtypes=np.int64,
                )
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2',
                dtypes=np.int64,
                )

        config = StoreConfig(
                index_depth=1,
                columns_depth=1,
                include_columns=True,
                include_index=True
                )

        b1 = Batch.from_frames((f1, f2), config=config)

        with temp_file('.db') as fp:
            b1.to_duckdb(fp)
            b2 = Batch.from_duckdb(fp, config=config)
            frames = dict(b2.items())

        for frame in (f1, f2):
            self.assertEqualFrames(frame, frames[frame.name], compare_dtype=False)


    #---------------------------------------------------------------------------

    @skip_no_hdf5
    def test_batch_to_hdf5_a(self) -> None:
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

        with temp_file('.hdf5') as fp:
            b1.to_hdf5(fp)
            b2 = Batch.from_hdf5(fp, config=config)
            frames = dict(b2.items())

        for frame in (f1, f2):
            # brings in characters as objects, thus forcing different dtypes
            self.assertEqualFrames(frame, frames[frame.name], compare_dtype=False)

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

    #---------------------------------------------------------------------------

    def test_batch_apply_array_a(self) -> None:

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

        post = Batch.from_frames((f1, f2, f3)).unique().to_frame(axis=1, fill_value=None)
        self.assertEqual(post.to_pairs(0),
                (('f1', ((0, 1), (1, 2), (2, 3), (3, 4), (4, None), (5, None))), ('f2', ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6))), ('f3', ((0, 10), (1, 20), (2, 50), (3, 60), (4, None), (5, None))))
                )

    #---------------------------------------------------------------------------
    def test_batch_to_series_a(self) -> None:
        frame = Frame.from_records([["A", 1.0], ["B", 2.0], ["C", 3.0], ["C", 4.0]], columns=tuple("AB"))
        s = Batch(frame.iter_group_items("A"))["B"].sum().to_series()
        self.assertEqual(s.to_pairs(),
                (('A', 1.0), ('B', 2.0), ('C', 7.0))
                )

    #---------------------------------------------------------------------------
    def test_batch_to_npz(self) -> None:

        f1 = ff.parse('s(3,2)|v(bool)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(3,5)|v(bool)|c(I,str)|i(I,int)').rename('b')

        b1 = Batch.from_frames((f1, f2))
        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)
            b2 = Batch.from_zip_npz(fp)
            frames = dict(b2.items())

            self.assertTrue(frames['a'].equals(f1, compare_name=True, compare_dtype=True, compare_class=True))

    #---------------------------------------------------------------------------
    def test_batch_to_npy(self) -> None:

        f1 = ff.parse('s(3,2)|v(bool)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(3,5)|v(bool)|c(I,str)|i(I,int)').rename('b')

        b1 = Batch.from_frames((f1, f2))
        with temp_file('.zip') as fp:
            b1.to_zip_npy(fp)
            b2 = Batch.from_zip_npy(fp)
            frames = dict(b2.items())

            self.assertTrue(frames['a'].equals(f1, compare_name=True, compare_dtype=True, compare_class=True))


    #---------------------------------------------------------------------------
    def test_batch_via_values_a(self) -> None:
        f1 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('b')
        post = Batch.from_frames((f1, f2)).via_values.apply(np.cos).to_frame()
        self.assertEqual(round(post, 2).to_pairs(),
                (('zZbu', ((('a', 0), -0.54), (('a', 1), 0.05), (('b', 0), -0.54), (('b', 1), 0.05))), ('ztsv', ((('a', 0), -0.96), (('a', 1), -0.54), (('b', 0), -0.96), (('b', 1), -0.54))), ('zUvW', ((('a', 0), -0.82), (('a', 1), 1.0), (('b', 0), -0.82), (('b', 1), 1.0))))
                )

    def test_batch_via_values_b(self) -> None:
        f1 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('b')
        post = np.sin(Batch.from_frames((f1, f2)).via_values).to_frame()
        self.assertEqual(round(post, 2).to_pairs(),
                (('zZbu', ((('a', 0), -0.84), (('a', 1), 1.0), (('b', 0), -0.84), (('b', 1), 1.0))), ('ztsv', ((('a', 0), 0.28), (('a', 1), -0.84), (('b', 0), 0.28), (('b', 1), -0.84))), ('zUvW', ((('a', 0), 0.57), (('a', 1), 0.03), (('b', 0), 0.57), (('b', 1), 0.03))))
                )

    def test_batch_via_values_c(self) -> None:
        f1 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,3)|v(float)|c(I,str)').rename('b')
        post = Batch.from_frames((f1, f2)).apply(lambda s: np.sum(s.values)).to_series()

        self.assertEqual(post.to_pairs(),
            (('a', 213543.0), ('b', 3424.54))
            )

    def test_batch_via_values_d(self) -> None:
        f1 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('a') % 3
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('b') % 3
        post = np.power(Batch.from_frames((f1, f2)).via_values(dtype=float), 2).to_frame() # type: ignore
        self.assertEqual([dt.kind for dt in post.dtypes.values], ['f', 'f', 'f'])
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), 0.0), (('a', 1), 4.0), (('b', 0), 0.0), (('b', 1), 4.0))), ('ztsv', ((('a', 0), 4.0), (('a', 1), 0.0), (('b', 0), 4.0), (('b', 1), 0.0))), ('zUvW', ((('a', 0), 0.0), (('a', 1), 4.0), (('b', 0), 0.0), (('b', 1), 4.0)))))

        # import ipdb; ipdb.set_trace()


    #---------------------------------------------------------------------------
    def test_batch_via_str_getitem(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str[-1].to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), 'Q'), (('a', -3648), 'l'), (('b', 34715), 'Q'), (('b', -3648), 'l'))), ('ztsv', ((('a', 34715), 'i'), (('a', -3648), 'C'), (('b', 34715), 'i'), (('b', -3648), 'C'))), ('zUvW', ((('a', 34715), 'v'), (('a', -3648), 'W'), (('b', 34715), 'v'), (('b', -3648), 'W'))))
                )

    def test_batch_via_str_capitalize(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.capitalize().to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), 'Zjzq'), (('a', -3648), 'Zo5l'), (('b', 34715), 'Zjzq'), (('b', -3648), 'Zo5l'))), ('ztsv', ((('a', 34715), 'Zaji'), (('a', -3648), 'Zjnc'), (('b', 34715), 'Zaji'), (('b', -3648), 'Zjnc'))), ('zUvW', ((('a', 34715), 'Ztsv'), (('a', -3648), 'Zuvw'), (('b', 34715), 'Ztsv'), (('b', -3648), 'Zuvw')))))

    def test_batch_via_str_center(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.center(8, '-').to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), '--zjZQ--'), (('a', -3648), '--zO5l--'), (('b', 34715), '--zjZQ--'), (('b', -3648), '--zO5l--'))), ('ztsv', ((('a', 34715), '--zaji--'), (('a', -3648), '--zJnC--'), (('b', 34715), '--zaji--'), (('b', -3648), '--zJnC--'))), ('zUvW', ((('a', 34715), '--ztsv--'), (('a', -3648), '--zUvW--'), (('b', 34715), '--ztsv--'), (('b', -3648), '--zUvW--')))))

    def test_batch_via_str_contains(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.contains('zU').to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), False), (('a', -3648), False), (('b', 34715), False), (('b', -3648), False))), ('ztsv', ((('a', 34715), False), (('a', -3648), False), (('b', 34715), False), (('b', -3648), False))), ('zUvW', ((('a', 34715), False), (('a', -3648), True), (('b', 34715), False), (('b', -3648), True))))
                )

    def test_batch_via_str_count(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.count('z').to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), 1), (('a', -3648), 1), (('b', 34715), 1), (('b', -3648), 1))), ('ztsv', ((('a', 34715), 1), (('a', -3648), 1), (('b', 34715), 1), (('b', -3648), 1))), ('zUvW', ((('a', 34715), 1), (('a', -3648), 1), (('b', 34715), 1), (('b', -3648), 1))))
                )

    def test_batch_via_str_decode(self) -> None:
        f1 = ff.parse('s(2,3)|v(object)|c(I,str)|i(I,int)').rename('a').astype(bytes)
        f2 = ff.parse('s(2,3)|v(object)|c(I,str)|i(I,int)').rename('b').astype(bytes)
        post = Batch.from_frames((f1, f2)).via_str.decode('utf-8').to_frame()
        self.assertEqual([dt.kind for dt in post.dtypes.values], ['U', 'U', 'U'])

    def test_batch_via_str_encode(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a').astype(object)
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b').astype(object)
        post = Batch.from_frames((f1, f2)).via_str.encode('ascii').to_frame()
        self.assertEqual([dt.kind for dt in post.dtypes.values], ['S', 'S', 'S'])

    def test_batch_via_str_endswith(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.endswith('v').to_frame()

        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), False), (('a', -3648), False), (('b', 34715), False), (('b', -3648), False))), ('ztsv', ((('a', 34715), False), (('a', -3648), False), (('b', 34715), False), (('b', -3648), False))), ('zUvW', ((('a', 34715), True), (('a', -3648), False), (('b', 34715), True), (('b', -3648), False))))
                )

    def test_batch_via_str_find(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.find('v').to_frame()

        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), -1), (('a', -3648), -1), (('b', 34715), -1), (('b', -3648), -1))), ('ztsv', ((('a', 34715), -1), (('a', -3648), -1), (('b', 34715), -1), (('b', -3648), -1))), ('zUvW', ((('a', 34715), 3), (('a', -3648), 2), (('b', 34715), 3), (('b', -3648), 2)))))

    def test_batch_via_str_format(self) -> None:
        f1 = ff.parse('s(2,3)|v(int)|c(I,str)|i(I,int)').rename('a') / 3
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)|i(I,int)').rename('b') / 3
        post = Batch.from_frames((f1, f2)).via_str.format('{:.3}').to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), '-2.93e+04'), (('a', -3648), '3.1e+04'), (('b', 34715), '-2.93e+04'), (('b', -3648), '3.1e+04'))), ('ztsv', ((('a', 34715), '5.41e+04'), (('a', -3648), '-1.37e+04'), (('b', 34715), '5.41e+04'), (('b', -3648), '-1.37e+04'))), ('zUvW', ((('a', 34715), '-1.22e+03'), (('a', -3648), '3.04e+04'), (('b', 34715), '-1.22e+03'), (('b', -3648), '3.04e+04'))))
                )

    def test_batch_via_str_index(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')

        with self.assertRaises(ValueError):
            post = Batch.from_frames((f1, f2)).via_str.index('v').to_frame()

    def test_batch_via_str_isalnum(self) -> None:
        f1 = ff.parse('s(2,3)|v(int)|c(I,str)|i(I,int)').rename('a').astype(str)
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)|i(I,int)').rename('b').astype(str)
        post = Batch.from_frames((f1, f2)).via_str.isalnum().to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), False), (('a', -3648), True), (('b', 34715), False), (('b', -3648), True))), ('ztsv', ((('a', 34715), True), (('a', -3648), False), (('b', 34715), True), (('b', -3648), False))), ('zUvW', ((('a', 34715), False), (('a', -3648), True), (('b', 34715), False), (('b', -3648), True)))))

    def test_batch_via_str_isalpha(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.isalpha().to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), True), (('a', -3648), False), (('b', 34715), True), (('b', -3648), False))), ('ztsv', ((('a', 34715), True), (('a', -3648), True), (('b', 34715), True), (('b', -3648), True))), ('zUvW', ((('a', 34715), True), (('a', -3648), True), (('b', 34715), True), (('b', -3648), True)))))

    def test_batch_via_str_isdecimal(self) -> None:
        f1 = ff.parse('s(2,3)|v(int)|c(I,str)|i(I,int)').rename('a').astype(str)
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)|i(I,int)').rename('b').astype(str)
        post = Batch.from_frames((f1, f2)).via_str.isdecimal().to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), False), (('a', -3648), True), (('b', 34715), False), (('b', -3648), True))), ('ztsv', ((('a', 34715), True), (('a', -3648), False), (('b', 34715), True), (('b', -3648), False))), ('zUvW', ((('a', 34715), False), (('a', -3648), True), (('b', 34715), False), (('b', -3648), True))))
                )

    def test_batch_via_str_isdigit(self) -> None:
        f1 = ff.parse('s(2,3)|v(int)|c(I,str)|i(I,int)').rename('a').astype(str)
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)|i(I,int)').rename('b').astype(str)
        post = Batch.from_frames((f1, f2)).via_str.isdigit().to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), False), (('a', -3648), True), (('b', 34715), False), (('b', -3648), True))), ('ztsv', ((('a', 34715), True), (('a', -3648), False), (('b', 34715), True), (('b', -3648), False))), ('zUvW', ((('a', 34715), False), (('a', -3648), True), (('b', 34715), False), (('b', -3648), True))))
                )

    def test_batch_via_str_islower(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.islower().to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), False), (('a', -3648), False), (('b', 34715), False), (('b', -3648), False))), ('ztsv', ((('a', 34715), True), (('a', -3648), False), (('b', 34715), True), (('b', -3648), False))), ('zUvW', ((('a', 34715), True), (('a', -3648), False), (('b', 34715), True), (('b', -3648), False))))
                )

    def test_batch_via_str_isnumeric(self) -> None:
        f1 = ff.parse('s(2,3)|v(int)|c(I,str)|i(I,int)').rename('a').astype(str)
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)|i(I,int)').rename('b').astype(str)
        post = Batch.from_frames((f1, f2)).via_str.isnumeric().to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), False), (('a', -3648), True), (('b', 34715), False), (('b', -3648), True))), ('ztsv', ((('a', 34715), True), (('a', -3648), False), (('b', 34715), True), (('b', -3648), False))), ('zUvW', ((('a', 34715), False), (('a', -3648), True), (('b', 34715), False), (('b', -3648), True))))
                )

    def test_batch_via_str_isspace(self) -> None:
        f1 = ff.parse('s(2,3)|v(int)|c(I,str)|i(I,int)'
                ).iter_element().apply(lambda e: ' ' if e % 2 else e).rename('a')
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)|i(I,int)').rename('b').astype(str)
        post = Batch.from_frames((f1, f2)).via_str.isspace().to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), True), (('a', -3648), True), (('b', 34715), False), (('b', -3648), False))), ('ztsv', ((('a', 34715), True), (('a', -3648), True), (('b', 34715), False), (('b', -3648), False))), ('zUvW', ((('a', 34715), False), (('a', -3648), True), (('b', 34715), False), (('b', -3648), False))))
                )

    def test_batch_via_str_istitle(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.istitle().to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), False), (('a', -3648), False), (('b', 34715), False), (('b', -3648), False))), ('ztsv', ((('a', 34715), False), (('a', -3648), False), (('b', 34715), False), (('b', -3648), False))), ('zUvW', ((('a', 34715), False), (('a', -3648), False), (('b', 34715), False), (('b', -3648), False))))
                )

    def test_batch_via_str_isupper(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.isupper().to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), False), (('a', -3648), False), (('b', 34715), False), (('b', -3648), False))), ('ztsv', ((('a', 34715), False), (('a', -3648), False), (('b', 34715), False), (('b', -3648), False))), ('zUvW', ((('a', 34715), False), (('a', -3648), False), (('b', 34715), False), (('b', -3648), False))))
                )

    def test_batch_via_str_len(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.len().to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), 4), (('a', -3648), 4), (('b', 34715), 4), (('b', -3648), 4))), ('ztsv', ((('a', 34715), 4), (('a', -3648), 4), (('b', 34715), 4), (('b', -3648), 4))), ('zUvW', ((('a', 34715), 4), (('a', -3648), 4), (('b', 34715), 4), (('b', -3648), 4)))))

    def test_batch_via_str_ljust(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.ljust(6, '-').to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), 'zjZQ--'), (('a', -3648), 'zO5l--'), (('b', 34715), 'zjZQ--'), (('b', -3648), 'zO5l--'))), ('ztsv', ((('a', 34715), 'zaji--'), (('a', -3648), 'zJnC--'), (('b', 34715), 'zaji--'), (('b', -3648), 'zJnC--'))), ('zUvW', ((('a', 34715), 'ztsv--'), (('a', -3648), 'zUvW--'), (('b', 34715), 'ztsv--'), (('b', -3648), 'zUvW--'))))
                )

    def test_batch_via_str_lower(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.lower().to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), 'zjzq'), (('a', -3648), 'zo5l'), (('b', 34715), 'zjzq'), (('b', -3648), 'zo5l'))), ('ztsv', ((('a', 34715), 'zaji'), (('a', -3648), 'zjnc'), (('b', 34715), 'zaji'), (('b', -3648), 'zjnc'))), ('zUvW', ((('a', 34715), 'ztsv'), (('a', -3648), 'zuvw'), (('b', 34715), 'ztsv'), (('b', -3648), 'zuvw'))))
                )

    def test_batch_via_str_lstrip(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.lstrip('z').to_frame()

        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), 'jZQ'), (('a', -3648), 'O5l'), (('b', 34715), 'jZQ'), (('b', -3648), 'O5l'))), ('ztsv', ((('a', 34715), 'aji'), (('a', -3648), 'JnC'), (('b', 34715), 'aji'), (('b', -3648), 'JnC'))), ('zUvW', ((('a', 34715), 'tsv'), (('a', -3648), 'UvW'), (('b', 34715), 'tsv'), (('b', -3648), 'UvW'))))
                )

    def test_batch_via_str_partition(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.partition('n').to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), ('zjZQ', '', '')), (('a', -3648), ('zO5l', '', '')), (('b', 34715), ('zjZQ', '', '')), (('b', -3648), ('zO5l', '', '')))), ('ztsv', ((('a', 34715), ('zaji', '', '')), (('a', -3648), ('zJ', 'n', 'C')), (('b', 34715), ('zaji', '', '')), (('b', -3648), ('zJ', 'n', 'C')))), ('zUvW', ((('a', 34715), ('ztsv', '', '')), (('a', -3648), ('zUvW', '', '')), (('b', 34715), ('ztsv', '', '')), (('b', -3648), ('zUvW', '', '')))))
                )

    def test_batch_via_str_replace(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.replace('z', '9').to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), '9jZQ'), (('a', -3648), '9O5l'), (('b', 34715), '9jZQ'), (('b', -3648), '9O5l'))), ('ztsv', ((('a', 34715), '9aji'), (('a', -3648), '9JnC'), (('b', 34715), '9aji'), (('b', -3648), '9JnC'))), ('zUvW', ((('a', 34715), '9tsv'), (('a', -3648), '9UvW'), (('b', 34715), '9tsv'), (('b', -3648), '9UvW'))))
                )

    def test_batch_via_str_rfind(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.rfind('C').to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), -1), (('a', -3648), -1), (('b', 34715), -1), (('b', -3648), -1))), ('ztsv', ((('a', 34715), -1), (('a', -3648), 3), (('b', 34715), -1), (('b', -3648), 3))), ('zUvW', ((('a', 34715), -1), (('a', -3648), -1), (('b', 34715), -1), (('b', -3648), -1))))
                )

    def test_batch_via_str_rindex(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        with self.assertRaises(ValueError):
            _ = Batch.from_frames((f1, f2)).via_str.rindex('C').to_frame()

    def test_batch_via_str_rjust(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.rjust(6, '-').to_frame()

        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), '--zjZQ'), (('a', -3648), '--zO5l'), (('b', 34715), '--zjZQ'), (('b', -3648), '--zO5l'))), ('ztsv', ((('a', 34715), '--zaji'), (('a', -3648), '--zJnC'), (('b', 34715), '--zaji'), (('b', -3648), '--zJnC'))), ('zUvW', ((('a', 34715), '--ztsv'), (('a', -3648), '--zUvW'), (('b', 34715), '--ztsv'), (('b', -3648), '--zUvW'))))
                )

    def test_batch_via_str_rpartition(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.rpartition('j').to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), ('z', 'j', 'ZQ')), (('a', -3648), ('', '', 'zO5l')), (('b', 34715), ('z', 'j', 'ZQ')), (('b', -3648), ('', '', 'zO5l')))), ('ztsv', ((('a', 34715), ('za', 'j', 'i')), (('a', -3648), ('', '', 'zJnC')), (('b', 34715), ('za', 'j', 'i')), (('b', -3648), ('', '', 'zJnC')))), ('zUvW', ((('a', 34715), ('', '', 'ztsv')), (('a', -3648), ('', '', 'zUvW')), (('b', 34715), ('', '', 'ztsv')), (('b', -3648), ('', '', 'zUvW')))))
                )

    def test_batch_via_str_rsplit(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.rsplit('j').to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), ('z', 'ZQ')), (('a', -3648), ('zO5l',)), (('b', 34715), ('z', 'ZQ')), (('b', -3648), ('zO5l',)))), ('ztsv', ((('a', 34715), ('za', 'i')), (('a', -3648), ('zJnC',)), (('b', 34715), ('za', 'i')), (('b', -3648), ('zJnC',)))), ('zUvW', ((('a', 34715), ('ztsv',)), (('a', -3648), ('zUvW',)), (('b', 34715), ('ztsv',)), (('b', -3648), ('zUvW',)))))
                )

    def test_batch_via_str_rstrip(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.rstrip('Q').to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), 'zjZ'), (('a', -3648), 'zO5l'), (('b', 34715), 'zjZ'), (('b', -3648), 'zO5l'))), ('ztsv', ((('a', 34715), 'zaji'), (('a', -3648), 'zJnC'), (('b', 34715), 'zaji'), (('b', -3648), 'zJnC'))), ('zUvW', ((('a', 34715), 'ztsv'), (('a', -3648), 'zUvW'), (('b', 34715), 'ztsv'), (('b', -3648), 'zUvW'))))
                )

    def test_batch_via_str_split(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.split('j').to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), ('z', 'ZQ')), (('a', -3648), ('zO5l',)), (('b', 34715), ('z', 'ZQ')), (('b', -3648), ('zO5l',)))), ('ztsv', ((('a', 34715), ('za', 'i')), (('a', -3648), ('zJnC',)), (('b', 34715), ('za', 'i')), (('b', -3648), ('zJnC',)))), ('zUvW', ((('a', 34715), ('ztsv',)), (('a', -3648), ('zUvW',)), (('b', 34715), ('ztsv',)), (('b', -3648), ('zUvW',)))))
                )

    def test_batch_via_str_startswith(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.startswith('z').to_frame()

        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), True), (('a', -3648), True), (('b', 34715), True), (('b', -3648), True))), ('ztsv', ((('a', 34715), True), (('a', -3648), True), (('b', 34715), True), (('b', -3648), True))), ('zUvW', ((('a', 34715), True), (('a', -3648), True), (('b', 34715), True), (('b', -3648), True))))
                )

    def test_batch_via_str_strip(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.strip('z').to_frame()

        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), 'jZQ'), (('a', -3648), 'O5l'), (('b', 34715), 'jZQ'), (('b', -3648), 'O5l'))), ('ztsv', ((('a', 34715), 'aji'), (('a', -3648), 'JnC'), (('b', 34715), 'aji'), (('b', -3648), 'JnC'))), ('zUvW', ((('a', 34715), 'tsv'), (('a', -3648), 'UvW'), (('b', 34715), 'tsv'), (('b', -3648), 'UvW'))))
                )

    def test_batch_via_str_swapcase(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.swapcase().to_frame()

        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), 'ZJzq'), (('a', -3648), 'Zo5L'), (('b', 34715), 'ZJzq'), (('b', -3648), 'Zo5L'))), ('ztsv', ((('a', 34715), 'ZAJI'), (('a', -3648), 'ZjNc'), (('b', 34715), 'ZAJI'), (('b', -3648), 'ZjNc'))), ('zUvW', ((('a', 34715), 'ZTSV'), (('a', -3648), 'ZuVw'), (('b', 34715), 'ZTSV'), (('b', -3648), 'ZuVw'))))
                )

    def test_batch_via_str_title(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.title().to_frame()

        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), 'Zjzq'), (('a', -3648), 'Zo5L'), (('b', 34715), 'Zjzq'), (('b', -3648), 'Zo5L'))), ('ztsv', ((('a', 34715), 'Zaji'), (('a', -3648), 'Zjnc'), (('b', 34715), 'Zaji'), (('b', -3648), 'Zjnc'))), ('zUvW', ((('a', 34715), 'Ztsv'), (('a', -3648), 'Zuvw'), (('b', 34715), 'Ztsv'), (('b', -3648), 'Zuvw'))))
                )

    def test_batch_via_str_upper(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.upper().to_frame()

        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), 'ZJZQ'), (('a', -3648), 'ZO5L'), (('b', 34715), 'ZJZQ'), (('b', -3648), 'ZO5L'))), ('ztsv', ((('a', 34715), 'ZAJI'), (('a', -3648), 'ZJNC'), (('b', 34715), 'ZAJI'), (('b', -3648), 'ZJNC'))), ('zUvW', ((('a', 34715), 'ZTSV'), (('a', -3648), 'ZUVW'), (('b', 34715), 'ZTSV'), (('b', -3648), 'ZUVW')))))

    def test_batch_via_str_zfill(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2)).via_str.zfill(6).to_frame()

        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 34715), '00zjZQ'), (('a', -3648), '00zO5l'), (('b', 34715), '00zjZQ'), (('b', -3648), '00zO5l'))), ('ztsv', ((('a', 34715), '00zaji'), (('a', -3648), '00zJnC'), (('b', 34715), '00zaji'), (('b', -3648), '00zJnC'))), ('zUvW', ((('a', 34715), '00ztsv'), (('a', -3648), '00zUvW'), (('b', 34715), '00ztsv'), (('b', -3648), '00zUvW'))))
                )

    #---------------------------------------------------------------------------
    def test_batch_via_fill_value_loc(self) -> None:

        f1 = ff.parse('s(2,2)|v(str)').rename('a')
        f2 = ff.parse('s(2,2)|v(str)').rename('b')
        post = Batch.from_frames((f1, f2)).via_fill_value('').loc[[0, 2]].to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), 'zjZQ'), (('a', 2), ''), (('b', 0), 'zjZQ'), (('b', 2), ''))), (1, ((('a', 0), 'zaji'), (('a', 2), ''), (('b', 0), 'zaji'), (('b', 2), '')))))

    def test_batch_via_fill_value_getitem(self) -> None:

        f1 = ff.parse('s(2,2)|v(str)').rename('a')
        f2 = ff.parse('s(2,2)|v(str)').rename('b')
        post1 = Batch.from_frames((f1, f2)).via_fill_value('')[[0, 2]].to_frame()

        self.assertEqual(post1.to_pairs(),
                ((0, ((('a', 0), 'zjZQ'), (('a', 1), 'zO5l'), (('b', 0), 'zjZQ'), (('b', 1), 'zO5l'))), (2, ((('a', 0), ''), (('a', 1), ''), (('b', 0), ''), (('b', 1), '')))))
        self.assertEqual(
                tuple(s.values.tolist() for s in Batch.from_frames((f1, f2)).via_fill_value('*')[2].values),
                (['*', '*'], ['*', '*'])
                )

    def test_batch_via_fill_value_add(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').rename('a')
        f2 = ff.parse('s(2,2)|v(int)').rename('b')
        f3 = ff.parse('s(1,2)|v(int)|c(I,str)').relabel(columns=(1,4))
        post1 = (Batch.from_frames((f1, f2)).via_fill_value(0) + f3).to_frame()

        self.assertEqual(post1.to_pairs(),
                ((0, ((('a', 0), -88017), (('a', 1), 92867), (('b', 0), -88017), (('b', 1), 92867))), (1, ((('a', 0), 74180), (('a', 1), -41157), (('b', 0), 74180), (('b', 1), -41157))), (4, ((('a', 0), 162197), (('a', 1), 0), (('b', 0), 162197), (('b', 1), 0))))
                )

    def test_batch_via_fill_value_sub(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').rename('a')
        f2 = ff.parse('s(2,2)|v(int)').rename('b')
        f3 = ff.parse('s(1,2)|v(int)|c(I,str)').relabel(columns=(1,4))
        post1 = (Batch.from_frames((f1, f2)).via_fill_value(0) - f3).to_frame()

        self.assertEqual(post1.to_pairs(),
                ((0, ((('a', 0), -88017), (('a', 1), 92867), (('b', 0), -88017), (('b', 1), 92867))), (1, ((('a', 0), 250214), (('a', 1), -41157), (('b', 0), 250214), (('b', 1), -41157))), (4, ((('a', 0), -162197), (('a', 1), 0), (('b', 0), -162197), (('b', 1), 0))))
                )

    def test_batch_via_fill_value_mul(self) -> None:
        f1 = ff.parse('s(2,2)|v(int64)').rename('a')
        f2 = ff.parse('s(2,2)|v(int64)').rename('b')
        f3 = ff.parse('s(1,2)|v(int64)|c(I,str)').relabel(columns=(1,4))
        post1 = (Batch.from_frames((f1, f2)).via_fill_value(0) * f3).to_frame()

        self.assertEqual(post1.to_pairs(),
                ((0, ((('a', 0), 0), (('a', 1), 0), (('b', 0), 0), (('b', 1), 0))), (1, ((('a', 0), -14276093349), (('a', 1), 0), (('b', 0), -14276093349), (('b', 1), 0))), (4, ((('a', 0), 0), (('a', 1), 0), (('b', 0), 0), (('b', 1), 0))))
                )

    def test_batch_via_fill_value_truediv(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').rename('a')
        f2 = ff.parse('s(2,2)|v(int)').rename('b')
        f3 = ff.parse('s(2,3)|v(int)')
        post1 = (Batch.from_frames((f1, f2)).via_fill_value(0) / f3).to_frame()
        self.assertEqual(post1.to_pairs(),
            ((0, ((('a', 0), 1.0), (('a', 1), 1.0), (('b', 0), 1.0), (('b', 1), 1.0))), (1, ((('a', 0), 1.0), (('a', 1), 1.0), (('b', 0), 1.0), (('b', 1), 1.0))), (2, ((('a', 0), -0.0), (('a', 1), 0.0), (('b', 0), -0.0), (('b', 1), 0.0))))
            )

    def test_batch_via_fill_value_floordiv(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').rename('a')
        f2 = ff.parse('s(2,2)|v(int)').rename('b')
        f3 = ff.parse('s(2,3)|v(int)')
        post1 = (Batch.from_frames((f1, f2)).via_fill_value(0) // f3).to_frame()
        self.assertEqual(post1.to_pairs(),
            ((0, ((('a', 0), 1), (('a', 1), 1), (('b', 0), 1), (('b', 1), 1))), (1, ((('a', 0), 1), (('a', 1), 1), (('b', 0), 1), (('b', 1), 1))), (2, ((('a', 0), 0), (('a', 1), 0), (('b', 0), 0), (('b', 1), 0))))
            )

    def test_batch_via_fill_value_mod(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').rename('a')
        f2 = ff.parse('s(2,2)|v(int)').rename('b')
        f3 = ff.parse('s(2,3)|v(int)') % 3 + 1
        post1 = (Batch.from_frames((f1, f2)).via_fill_value(0) % f3).to_frame()
        self.assertEqual(post1.to_pairs(),
            ((0, ((('a', 0), 0), (('a', 1), 2), (('b', 0), 0), (('b', 1), 2))), (1, ((('a', 0), 2), (('a', 1), 0), (('b', 0), 2), (('b', 1), 0))), (2, ((('a', 0), 0), (('a', 1), 0), (('b', 0), 0), (('b', 1), 0))))
            )

    def test_batch_via_fill_value_pow(self) -> None:
        f1 = ff.parse('s(2,2)|v(int64)').rename('a')
        f2 = ff.parse('s(2,2)|v(int64)').rename('b')
        f3 = ff.parse('s(2,3)|v(int64)') % 2 + 1
        post1 = (Batch.from_frames((f1, f2)).via_fill_value(0) ** f3).to_frame()

        self.assertEqual(post1.to_pairs(),
            ((0, ((('a', 0), 7746992289), (('a', 1), 8624279689), (('b', 0), 7746992289), (('b', 1), 8624279689))), (1, ((('a', 0), 26307866809), (('a', 1), 1693898649), (('b', 0), 26307866809), (('b', 1), 1693898649))), (2, ((('a', 0), 0), (('a', 1), 0), (('b', 0), 0), (('b', 1), 0))))
            )

    def test_batch_via_fill_value_lshift(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').rename('a')
        f2 = ff.parse('s(2,2)|v(int)').rename('b')
        f3 = ff.parse('s(2,3)|v(int)') % 2 + 1
        post1 = (Batch.from_frames((f1, f2)).via_fill_value(0) << f3).to_frame()

        self.assertEqual(post1.to_pairs(),
            ((0, ((('a', 0), -352068), (('a', 1), 371468), (('b', 0), -352068), (('b', 1), 371468))), (1, ((('a', 0), 648788), (('a', 1), -164628), (('b', 0), 648788), (('b', 1), -164628))), (2, ((('a', 0), 0), (('a', 1), 0), (('b', 0), 0), (('b', 1), 0))))
            )

    def test_batch_via_fill_value_rshift(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').rename('a')
        f2 = ff.parse('s(2,2)|v(int)').rename('b')
        f3 = ff.parse('s(2,3)|v(int)') % 2 + 1
        post1 = (Batch.from_frames((f1, f2)).via_fill_value(0) >> f3).to_frame()
        self.assertEqual(post1.to_pairs(),
            ((0, ((('a', 0), -22005), (('a', 1), 23216), (('b', 0), -22005), (('b', 1), 23216))), (1, ((('a', 0), 40549), (('a', 1), -10290), (('b', 0), 40549), (('b', 1), -10290))), (2, ((('a', 0), 0), (('a', 1), 0), (('b', 0), 0), (('b', 1), 0))))
            )

    def test_batch_via_fill_value_and(self) -> None:
        f1 = ff.parse('s(2,2)|v(bool)').rename('a')
        f2 = ~ff.parse('s(2,2)|v(bool)').rename('b')
        f3 = ~ff.parse('s(2,3)|v(bool)')
        post1 = (Batch.from_frames((f1, f2)).via_fill_value(False) & f3).to_frame()
        self.assertEqual(post1.to_pairs(),
            ((0, ((('a', 0), False), (('a', 1), False), (('b', 0), True), (('b', 1), True))), (1, ((('a', 0), False), (('a', 1), False), (('b', 0), True), (('b', 1), True))), (2, ((('a', 0), False), (('a', 1), False), (('b', 0), False), (('b', 1), False)))))

    def test_batch_via_fill_value_xor(self) -> None:
        f1 = ff.parse('s(2,2)|v(bool)').rename('a')
        f2 = ~ff.parse('s(2,2)|v(bool)').rename('b')
        f3 = ~ff.parse('s(2,3)|v(bool)')
        post1 = (Batch.from_frames((f1, f2)).via_fill_value(False) ^ f3).to_frame()
        self.assertEqual(post1.to_pairs(),
            ((0, ((('a', 0), True), (('a', 1), True), (('b', 0), False), (('b', 1), False))), (1, ((('a', 0), True), (('a', 1), True), (('b', 0), False), (('b', 1), False))), (2, ((('a', 0), False), (('a', 1), True), (('b', 0), False), (('b', 1), True))))
            )

    def test_batch_via_fill_value_or(self) -> None:
        f1 = ff.parse('s(2,2)|v(bool)').rename('a')
        f2 = ~ff.parse('s(2,2)|v(bool)').rename('b')
        f3 = ~ff.parse('s(2,3)|v(bool)')
        post1 = (Batch.from_frames((f1, f2)).via_fill_value(False) | f3).to_frame()
        self.assertEqual(post1.to_pairs(),
            ((0, ((('a', 0), True), (('a', 1), True), (('b', 0), True), (('b', 1), True))), (1, ((('a', 0), True), (('a', 1), True), (('b', 0), True), (('b', 1), True))), (2, ((('a', 0), False), (('a', 1), True), (('b', 0), False), (('b', 1), True))))
            )

    def test_batch_via_fill_value_lt(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').rename('a')
        f2 = -ff.parse('s(2,2)|v(int)').rename('b')
        f3 = ff.parse('s(2,3)|v(int)') * 2
        post1 = (Batch.from_frames((f1, f2)).via_fill_value(0) < f3).to_frame()

        self.assertEqual(post1.to_pairs(),
            ((0, ((('a', 0), False), (('a', 1), True), (('b', 0), False), (('b', 1), True))), (1, ((('a', 0), True), (('a', 1), False), (('b', 0), True), (('b', 1), False))), (2, ((('a', 0), False), (('a', 1), True), (('b', 0), False), (('b', 1), True))))
            )

    def test_batch_via_fill_value_le(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').rename('a')
        f2 = -ff.parse('s(2,2)|v(int)').rename('b')
        f3 = ff.parse('s(2,3)|v(int)')
        post1 = (Batch.from_frames((f1, f2)).via_fill_value(0) <= f3).to_frame()

        self.assertEqual(post1.to_pairs(),
            ((0, ((('a', 0), True), (('a', 1), True), (('b', 0), False), (('b', 1), True))), (1, ((('a', 0), True), (('a', 1), True), (('b', 0), True), (('b', 1), False))), (2, ((('a', 0), False), (('a', 1), True), (('b', 0), False), (('b', 1), True))))
            )

    def test_batch_via_fill_value_eq(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').rename('a')
        f2 = -ff.parse('s(2,2)|v(int)').rename('b')
        f3 = ff.parse('s(2,3)|v(int)')
        post1 = (Batch.from_frames((f1, f2)).via_fill_value(0) == f3).to_frame()
        self.assertEqual(post1.to_pairs(),
            ((0, ((('a', 0), True), (('a', 1), True), (('b', 0), False), (('b', 1), False))), (1, ((('a', 0), True), (('a', 1), True), (('b', 0), False), (('b', 1), False))), (2, ((('a', 0), False), (('a', 1), False), (('b', 0), False), (('b', 1), False)))))

    def test_batch_via_fill_value_ne(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').rename('a')
        f2 = -ff.parse('s(2,2)|v(int)').rename('b')
        f3 = ff.parse('s(2,3)|v(int)')
        post1 = (Batch.from_frames((f1, f2)).via_fill_value(0) != f3).to_frame()
        self.assertEqual( post1.to_pairs(),
            ((0, ((('a', 0), False), (('a', 1), False), (('b', 0), True), (('b', 1), True))), (1, ((('a', 0), False), (('a', 1), False), (('b', 0), True), (('b', 1), True))), (2, ((('a', 0), True), (('a', 1), True), (('b', 0), True), (('b', 1), True))))
            )

    def test_batch_via_fill_value_gt(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').rename('a')
        f2 = -ff.parse('s(2,2)|v(int)').rename('b')
        f3 = ff.parse('s(2,3)|v(int)') * 2
        post1 = (Batch.from_frames((f1, f2)).via_fill_value(0) > f3).to_frame()
        self.assertEqual(post1.to_pairs(),
            ((0, ((('a', 0), True), (('a', 1), False), (('b', 0), True), (('b', 1), False))), (1, ((('a', 0), False), (('a', 1), True), (('b', 0), False), (('b', 1), True))), (2, ((('a', 0), True), (('a', 1), False), (('b', 0), True), (('b', 1), False))))
            )

    def test_batch_via_fill_value_ge(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').rename('a')
        f2 = -ff.parse('s(2,2)|v(int)').rename('b')
        f3 = ff.parse('s(2,3)|v(int)') * 2
        post1 = (Batch.from_frames((f1, f2)).via_fill_value(0) >= f3).to_frame()
        self.assertEqual(post1.to_pairs(),
            ((0, ((('a', 0), True), (('a', 1), False), (('b', 0), True), (('b', 1), False))), (1, ((('a', 0), False), (('a', 1), True), (('b', 0), False), (('b', 1), True))), (2, ((('a', 0), True), (('a', 1), False), (('b', 0), True), (('b', 1), False))))
            )

    def test_batch_via_fill_value_radd(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').rename('a')
        f2 = ff.parse('s(2,2)|v(int)').rename('b')
        f3 = ff.parse('s(1,2)|v(int)|c(I,str)').relabel(columns=(1,4))
        post1 = (f3 + Batch.from_frames((f1, f2)).via_fill_value(0)).to_frame()
        self.assertEqual(post1.to_pairs(),
                ((0, ((('a', 0), -88017), (('a', 1), 92867), (('b', 0), -88017), (('b', 1), 92867))), (1, ((('a', 0), 74180), (('a', 1), -41157), (('b', 0), 74180), (('b', 1), -41157))), (4, ((('a', 0), 162197), (('a', 1), 0), (('b', 0), 162197), (('b', 1), 0))))
                )

    def test_batch_via_fill_value_rsub(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').rename('a')
        f2 = ff.parse('s(2,2)|v(int)').rename('b')
        f3 = ff.parse('s(1,2)|v(int)|c(I,str)').relabel(columns=(1,4))
        post1 = (f3 - Batch.from_frames((f1, f2)).via_fill_value(0)).to_frame()
        self.assertEqual(post1.to_pairs(),
                ((0, ((('a', 0), 88017), (('a', 1), -92867), (('b', 0), 88017), (('b', 1), -92867))), (1, ((('a', 0), -250214), (('a', 1), 41157), (('b', 0), -250214), (('b', 1), 41157))), (4, ((('a', 0), 162197), (('a', 1), 0), (('b', 0), 162197), (('b', 1), 0))))
                )

    def test_batch_via_fill_value_rmul(self) -> None:
        f1 = ff.parse('s(2,2)|v(int64)').rename('a')
        f2 = ff.parse('s(2,2)|v(int64)').rename('b')
        f3 = ff.parse('s(1,2)|v(int64)|c(I,str)').relabel(columns=(1,4))
        post1 = (f3 * Batch.from_frames((f1, f2)).via_fill_value(0)).to_frame()
        self.assertEqual(post1.to_pairs(),
            ((0, ((('a', 0), 0), (('a', 1), 0), (('b', 0), 0), (('b', 1), 0))), (1, ((('a', 0), -14276093349), (('a', 1), 0), (('b', 0), -14276093349), (('b', 1), 0))), (4, ((('a', 0), 0), (('a', 1), 0), (('b', 0), 0), (('b', 1), 0))))
            )

    def test_batch_via_fill_value_rtruediv(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').rename('a')
        f2 = ff.parse('s(2,2)|v(int)').rename('b')
        f3 = ff.parse('s(2,3)|v(int)')
        post1 = (f3 / Batch.from_frames((f1, f2)).via_fill_value(1)).to_frame()
        self.assertEqual(post1.to_pairs(),
            ((0, ((('a', 0), 1.0), (('a', 1), 1.0), (('b', 0), 1.0), (('b', 1), 1.0))), (1, ((('a', 0), 1.0), (('a', 1), 1.0), (('b', 0), 1.0), (('b', 1), 1.0))), (2, ((('a', 0), -3648.0), (('a', 1), 91301.0), (('b', 0), -3648.0), (('b', 1), 91301.0))))
            )

    def test_batch_via_fill_value_rfloordiv(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').rename('a')
        f2 = ff.parse('s(2,2)|v(int)').rename('b')
        f3 = ff.parse('s(2,3)|v(int)')
        post1 = (f3 // Batch.from_frames((f1, f2)).via_fill_value(1)).to_frame()
        self.assertEqual(post1.to_pairs(),
            ((0, ((('a', 0), 1), (('a', 1), 1), (('b', 0), 1), (('b', 1), 1))), (1, ((('a', 0), 1), (('a', 1), 1), (('b', 0), 1), (('b', 1), 1))), (2, ((('a', 0), -3648), (('a', 1), 91301), (('b', 0), -3648), (('b', 1), 91301))))
            )



    #---------------------------------------------------------------------------
    def test_batch_via_re_search(self) -> None:

        f1 = ff.parse('s(2,2)|v(int)').rename('a').astype(str)
        f2 = ff.parse('s(2,2)|v(int)').rename('b').astype(str)
        post = Batch.from_frames((f1, f2)).via_re('[23]').search(2).to_frame()
        self.assertEqual(post.to_pairs(),
            ((0, ((('a', 0), False), (('a', 1), False), (('b', 0), False), (('b', 1), False))), (1, ((('a', 0), True), (('a', 1), False), (('b', 0), True), (('b', 1), False))))
            )

    def test_batch_via_re_match(self) -> None:

        f1 = ff.parse('s(2,2)|v(int)').rename('a').astype(str)
        f2 = ff.parse('s(2,2)|v(int)').rename('b').astype(str)
        post = Batch.from_frames((f1, f2)).via_re('928').match().to_frame()
        self.assertEqual(post.to_pairs(),
            ((0, ((('a', 0), False), (('a', 1), True), (('b', 0), False), (('b', 1), True))), (1, ((('a', 0), False), (('a', 1), False), (('b', 0), False), (('b', 1), False))))
            )

    def test_batch_via_re_fullmatch(self) -> None:

        f1 = ff.parse('s(2,2)|v(int)').rename('a').astype(str)
        f2 = ff.parse('s(2,2)|v(int)').rename('b').astype(str)
        post = Batch.from_frames((f1, f2)).via_re('162197').fullmatch().to_frame()
        self.assertEqual(post.to_pairs(),
            ((0, ((('a', 0), False), (('a', 1), False), (('b', 0), False), (('b', 1), False))), (1, ((('a', 0), True), (('a', 1), False), (('b', 0), True), (('b', 1), False))))
            )

    def test_batch_via_re_split(self) -> None:

        f1 = ff.parse('s(2,2)|v(int)').rename('a').astype(str)
        f2 = ff.parse('s(2,2)|v(int)').rename('b').astype(str)
        post = Batch.from_frames((f1, f2)).via_re('8').split(1).to_frame()
        self.assertEqual(post.to_pairs(),
            ((0, ((('a', 0), ('-', '8017')), (('a', 1), ('92', '67')), (('b', 0), ('-', '8017')), (('b', 1), ('92', '67')))), (1, ((('a', 0), ('162197',)), (('a', 1), ('-41157',)), (('b', 0), ('162197',)), (('b', 1), ('-41157',)))))
            )

    def test_batch_via_re_findall(self) -> None:

        f1 = ff.parse('s(2,2)|v(int)').rename('a').astype(str)
        f2 = ff.parse('s(2,2)|v(int)').rename('b').astype(str)
        post = Batch.from_frames((f1, f2)).via_re('8').findall(1).to_frame()
        self.assertEqual(post.to_pairs(),
            ((0, ((('a', 0), ('8', '8')), (('a', 1), ('8',)), (('b', 0), ('8', '8')), (('b', 1), ('8',)))), (1, ((('a', 0), ()), (('a', 1), ()), (('b', 0), ()), (('b', 1), ()))))
            )

    def test_batch_via_re_sub(self) -> None:

        f1 = ff.parse('s(2,2)|v(int)').rename('a').astype(str)
        f2 = ff.parse('s(2,2)|v(int)').rename('b').astype(str)
        post = Batch.from_frames((f1, f2)).via_re('8').sub('--', 2).to_frame()
        self.assertEqual(post.to_pairs(),
            ((0, ((('a', 0), '-----017'), (('a', 1), '92--67'), (('b', 0), '-----017'), (('b', 1), '92--67'))), (1, ((('a', 0), '162197'), (('a', 1), '-41157'), (('b', 0), '162197'), (('b', 1), '-41157'))))
            )

    def test_batch_via_re_subn(self) -> None:

        f1 = ff.parse('s(2,2)|v(int)').rename('a').astype(str)
        f2 = ff.parse('s(2,2)|v(int)').rename('b').astype(str)
        post = Batch.from_frames((f1, f2)).via_re('8').subn('--').to_frame()
        self.assertEqual(post.to_pairs(),
            ((0, ((('a', 0), ('-----017', 2)), (('a', 1), ('92--67', 1)), (('b', 0), ('-----017', 2)), (('b', 1), ('92--67', 1)))), (1, ((('a', 0), ('162197', 0)), (('a', 1), ('-41157', 0)), (('b', 0), ('162197', 0)), (('b', 1), ('-41157', 0)))))
            )

    #---------------------------------------------------------------------------
    def test_batch_via_dt_year(self) -> None:

        f1 = ff.parse('s(2,2)|v(dtD)').rename('a')
        f2 = ff.parse('s(2,2)|v(dtD)').rename('b')
        post = Batch.from_frames((f1, f2)).via_dt.year.to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), 2210), (('a', 1), 2224), (('b', 0), 2210), (('b', 1), 2224))), (1, ((('a', 0), 2414), (('a', 1), 2082), (('b', 0), 2414), (('b', 1), 2082))))
                )

    def test_batch_via_dt_month(self) -> None:

        f1 = ff.parse('s(2,2)|v(dtD)').rename('a')
        f2 = ff.parse('s(2,2)|v(dtD)').rename('b')
        post = Batch.from_frames((f1, f2)).via_dt.month.to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), 12), (('a', 1), 4), (('b', 0), 12), (('b', 1), 4))), (1, ((('a', 0), 1), (('a', 1), 9), (('b', 0), 1), (('b', 1), 9))))
                )

    def test_batch_via_dt_year_month(self) -> None:

        f1 = ff.parse('s(2,2)|v(dtD)').rename('a')
        f2 = ff.parse('s(2,2)|v(dtD)').rename('b')
        post = Batch.from_frames((f1, f2)).via_dt.year_month.to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), '2210-12'), (('a', 1), '2224-04'), (('b', 0), '2210-12'), (('b', 1), '2224-04'))), (1, ((('a', 0), '2414-01'), (('a', 1), '2082-09'), (('b', 0), '2414-01'), (('b', 1), '2082-09'))))
                )

    def test_batch_via_dt_year_quarter(self) -> None:

        f1 = ff.parse('s(2,2)|v(dtD)').rename('a')
        f2 = ff.parse('s(2,2)|v(dtD)').rename('b')
        post = Batch.from_frames((f1, f2)).via_dt.year_quarter.to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), '2210-Q4'), (('a', 1), '2224-Q2'), (('b', 0), '2210-Q4'), (('b', 1), '2224-Q2'))), (1, ((('a', 0), '2414-Q1'), (('a', 1), '2082-Q3'), (('b', 0), '2414-Q1'), (('b', 1), '2082-Q3'))))
                )

    def test_batch_via_dt_day(self) -> None:

        f1 = ff.parse('s(2,2)|v(dtD)').rename('a')
        f2 = ff.parse('s(2,2)|v(dtD)').rename('b')
        post = Batch.from_frames((f1, f2)).via_dt.day.to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), 26), (('a', 1), 6), (('b', 0), 26), (('b', 1), 6))), (1, ((('a', 0), 30), (('a', 1), 7), (('b', 0), 30), (('b', 1), 7))))
                )

    def test_batch_via_dt_hour(self) -> None:

        f1 = ff.parse('s(2,2)|v(dts)').rename('a')
        f2 = ff.parse('s(2,2)|v(dts)').rename('b')
        post = Batch.from_frames((f1, f2)).via_dt.hour.to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), 0), (('a', 1), 1), (('b', 0), 0), (('b', 1), 1))), (1, ((('a', 0), 21), (('a', 1), 11), (('b', 0), 21), (('b', 1), 11))))
                )

    def test_batch_via_dt_minute(self) -> None:

        f1 = ff.parse('s(2,2)|v(dts)').rename('a')
        f2 = ff.parse('s(2,2)|v(dts)').rename('b')
        post = Batch.from_frames((f1, f2)).via_dt.minute.to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), 26), (('a', 1), 47), (('b', 0), 26), (('b', 1), 47))), (1, ((('a', 0), 3), (('a', 1), 25), (('b', 0), 3), (('b', 1), 25))))
                )

    def test_batch_via_dt_second(self) -> None:

        f1 = ff.parse('s(2,2)|v(dts)').rename('a')
        f2 = ff.parse('s(2,2)|v(dts)').rename('b')
        post = Batch.from_frames((f1, f2)).via_dt.second.to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), 57), (('a', 1), 47), (('b', 0), 57), (('b', 1), 47))), (1, ((('a', 0), 17), (('a', 1), 57), (('b', 0), 17), (('b', 1), 57))))
                )

    def test_batch_via_dt_weekday(self) -> None:

        f1 = ff.parse('s(2,2)|v(dts)').rename('a')
        f2 = ff.parse('s(2,2)|v(dts)').rename('b')
        post = Batch.from_frames((f1, f2)).via_dt.weekday().to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), 4), (('a', 1), 4), (('b', 0), 4), (('b', 1), 4))), (1, ((('a', 0), 4), (('a', 1), 3), (('b', 0), 4), (('b', 1), 3))))
                )

    def test_batch_via_dt_quarter(self) -> None:

        f1 = ff.parse('s(2,2)|v(dts)').rename('a')
        f2 = ff.parse('s(2,2)|v(dts)').rename('b')
        post = Batch.from_frames((f1, f2)).via_dt.quarter().to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), 1), (('a', 1), 1), (('b', 0), 1), (('b', 1), 1))), (1, ((('a', 0), 1), (('a', 1), 1), (('b', 0), 1), (('b', 1), 1))))
                )

    def test_batch_via_dt_is_month_end(self) -> None:

        f1 = ff.parse('s(2,2)|v(dts)').rename('a')
        f2 = ff.parse('s(2,2)|v(dts)').rename('b')
        post = Batch.from_frames((f1, f2)).via_dt.is_month_end().to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), False), (('a', 1), False), (('b', 0), False), (('b', 1), False))), (1, ((('a', 0), False), (('a', 1), False), (('b', 0), False), (('b', 1), False))))
                )

    def test_batch_via_dt_is_month_start(self) -> None:

        f1 = ff.parse('s(2,2)|v(dts)').rename('a')
        f2 = ff.parse('s(2,2)|v(dts)').rename('b')
        post = Batch.from_frames((f1, f2)).via_dt.is_month_start().to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), False), (('a', 1), False), (('b', 0), False), (('b', 1), False))), (1, ((('a', 0), False), (('a', 1), True), (('b', 0), False), (('b', 1), True))))
                )

    def test_batch_via_dt_is_year_end(self) -> None:

        f1 = ff.parse('s(2,2)|v(dts)').rename('a')
        f2 = ff.parse('s(2,2)|v(dts)').rename('b')
        post = Batch.from_frames((f1, f2)).via_dt.is_year_end().to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), False), (('a', 1), False), (('b', 0), False), (('b', 1), False))), (1, ((('a', 0), False), (('a', 1), False), (('b', 0), False), (('b', 1), False))))
                )

    def test_batch_via_dt_is_year_start(self) -> None:

        f1 = ff.parse('s(2,2)|v(dts)').rename('a')
        f2 = ff.parse('s(2,2)|v(dts)').rename('b')
        post = Batch.from_frames((f1, f2)).via_dt.is_year_start().to_frame()
        self.assertEqual( post.to_pairs(),
                ((0, ((('a', 0), False), (('a', 1), False), (('b', 0), False), (('b', 1), False))), (1, ((('a', 0), False), (('a', 1), True), (('b', 0), False), (('b', 1), True)))))

    def test_batch_via_dt_is_quarter_end(self) -> None:

        f1 = ff.parse('s(2,2)|v(dts)').rename('a')
        f2 = ff.parse('s(2,2)|v(dts)').rename('b')
        post = Batch.from_frames((f1, f2)).via_dt.is_quarter_end().to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), False), (('a', 1), False), (('b', 0), False), (('b', 1), False))), (1, ((('a', 0), False), (('a', 1), False), (('b', 0), False), (('b', 1), False)))))

    def test_batch_via_dt_is_quarter_start(self) -> None:

        f1 = ff.parse('s(2,2)|v(dts)').rename('a')
        f2 = ff.parse('s(2,2)|v(dts)').rename('b')
        post = Batch.from_frames((f1, f2)).via_dt.is_quarter_start().to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), False), (('a', 1), False), (('b', 0), False), (('b', 1), False))), (1, ((('a', 0), False), (('a', 1), True), (('b', 0), False), (('b', 1), True))))
                )

    def test_batch_via_dt_timetuple(self) -> None:

        f1 = ff.parse('s(2,2)|v(dts)').rename('a')
        f2 = ff.parse('s(2,2)|v(dts)').rename('b')
        post = Batch.from_frames((f1, f2)).via_dt.timetuple().to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), time.struct_time((1970, 1, 2, 0, 26, 57, 4, 2, -1))), (('a', 1), time.struct_time((1970, 1, 2, 1, 47, 47, 4, 2, -1))), (('b', 0), time.struct_time((1970, 1, 2, 0, 26, 57, 4, 2, -1))), (('b', 1), time.struct_time((1970, 1, 2, 1, 47, 47, 4, 2, -1))))), (1, ((('a', 0), time.struct_time((1970, 1, 2, 21, 3, 17, 4, 2, -1))), (('a', 1), time.struct_time((1970, 1, 1, 11, 25,57, 3, 1, -1))), (('b', 0), time.struct_time((1970, 1, 2, 21, 3, 17, 4, 2, -1))), (('b', 1), time.struct_time((1970, 1, 1, 11, 25, 57, 3, 1, -1))))))
                )

    def test_batch_via_dt_isoformat(self) -> None:

        f1 = ff.parse('s(2,2)|v(dtD)').rename('a')
        f2 = ff.parse('s(2,2)|v(dtD)').rename('b')
        post = Batch.from_frames((f1, f2)).via_dt.isoformat().to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), '2210-12-26'), (('a', 1), '2224-04-06'), (('b', 0), '2210-12-26'), (('b', 1), '2224-04-06'))), (1, ((('a', 0), '2414-01-30'), (('a', 1), '2082-09-07'), (('b', 0), '2414-01-30'), (('b', 1), '2082-09-07'))))
                )

    def test_batch_via_dt_fromisoformat(self) -> None:

        f1 = ff.parse('s(2,2)|v(dtD)').rename('a').astype(str)
        f2 = ff.parse('s(2,2)|v(dtD)').rename('b').astype(str)
        post = Batch.from_frames((f1, f2)).via_dt.fromisoformat().to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), datetime.date(2210, 12, 26)), (('a', 1), datetime.date(2224, 4, 6)), (('b', 0), datetime.date(2210, 12, 26)), (('b', 1), datetime.date(2224, 4, 6)))), (1, ((('a', 0), datetime.date(2414, 1, 30)), (('a', 1), datetime.date(2082, 9, 7)), (('b', 0), datetime.date(2414, 1, 30)), (('b', 1), datetime.date(2082, 9, 7)))))
                )

    def test_batch_via_dt_strftime(self) -> None:

        f1 = ff.parse('s(2,2)|v(dtD)').rename('a')
        f2 = ff.parse('s(2,2)|v(dtD)').rename('b')
        post = Batch.from_frames((f1, f2)).via_dt.strftime('%Y**%m').to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), '2210**12'), (('a', 1), '2224**04'), (('b', 0), '2210**12'), (('b', 1), '2224**04'))), (1, ((('a', 0), '2414**01'), (('a', 1), '2082**09'), (('b', 0), '2414**01'), (('b', 1), '2082**09'))))
                )

    def test_batch_via_dt_strptime(self) -> None:

        f1 = ff.parse('s(2,2)|v(dtD)').rename('a').astype(str)
        f2 = ff.parse('s(2,2)|v(dtD)').rename('b').astype(str)
        post = Batch.from_frames((f1, f2)).via_dt.strptime('%Y-%m-%d').to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), datetime.datetime(2210, 12, 26, 0, 0)), (('a', 1), datetime.datetime(2224, 4, 6, 0, 0)), (('b', 0), datetime.datetime(2210, 12, 26, 0, 0)), (('b', 1), datetime.datetime(2224, 4, 6, 0, 0)))), (1, ((('a', 0), datetime.datetime(2414, 1, 30, 0, 0)), (('a', 1), datetime.datetime(2082, 9, 7, 0, 0)), (('b', 0), datetime.datetime(2414, 1, 30, 0, 0)), (('b', 1), datetime.datetime(2082, 9, 7, 0, 0)))))
                )

    def test_batch_via_dt_strpdate(self) -> None:

        f1 = ff.parse('s(2,2)|v(dtD)').rename('a').astype(str)
        f2 = ff.parse('s(2,2)|v(dtD)').rename('b').astype(str)
        post = Batch.from_frames((f1, f2)).via_dt.strpdate('%Y-%m-%d').to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), datetime.date(2210, 12, 26)), (('a', 1), datetime.date(2224, 4, 6)), (('b', 0), datetime.date(2210, 12, 26)), (('b', 1), datetime.date(2224, 4, 6)))), (1, ((('a', 0), datetime.date(2414, 1, 30)), (('a', 1), datetime.date(2082, 9, 7)), (('b', 0), datetime.date(2414, 1, 30)), (('b', 1), datetime.date(2082, 9, 7)))))
                )

    def test_batch_via_dt_call(self) -> None:
        f1 = ff.parse('s(2,2)|v(dtD)').rename('a')
        f2 = ff.parse('s(2,2)|v(dtD)').rename('b')

        f1 = f1.assign.iloc[0, 0](np.datetime64("NAT"))
        f2 = f2.assign.iloc[1, 1](np.datetime64("NAT"))

        via_dt = Batch.from_frames((f1, f2)).via_dt

        # Not really expected behavior, but it is supported
        for _ in range(10):
            via_dt = via_dt(fill_value="x")
            self.assertIsInstance(via_dt, InterfaceBatchDatetime)

        post = via_dt.year.to_frame()
        self.assertEqual(post.to_pairs(),
                ((0, ((('a', 0), 'x'), (('a', 1), 2224), (('b', 0), 2210), (('b', 1), 2224))),
                (1, ((('a', 0), 2414), (('a', 1), 2082), (('b', 0), 2414), (('b', 1), 'x'))))
                )

    #---------------------------------------------------------------------------
    def test_batch_via_transpose_add(self) -> None:

        f1 = ff.parse('s(2,2)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,2)|v(int)|c(I,str)').rename('b')
        post = (Batch.from_frames((f1, f2)).via_T + Series((87017, 0))).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), -1000), (('a', 1), 92867), (('b', 0), -1000), (('b', 1), 92867))), ('ztsv', ((('a', 0), 249214), (('a', 1), -41157), (('b', 0), 249214), (('b', 1), -41157)))))

    def test_batch_via_transpose_sub(self) -> None:

        f1 = ff.parse('s(2,2)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,2)|v(int)|c(I,str)').rename('b')
        post = (Batch.from_frames((f1, f2)).via_T - Series((162197, 0))).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), -250214), (('a', 1), 92867), (('b', 0), -250214), (('b', 1), 92867))), ('ztsv', ((('a', 0), 0), (('a', 1), -41157), (('b', 0), 0), (('b', 1), -41157))))
                )

    def test_batch_via_transpose_mul(self) -> None:

        f1 = ff.parse('s(2,2)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,2)|v(int)|c(I,str)').rename('b')
        post = (Batch.from_frames((f1, f2)).via_T * Series((1, 0))).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), -88017), (('a', 1), 0), (('b', 0), -88017), (('b', 1), 0))), ('ztsv', ((('a', 0), 162197), (('a', 1), 0), (('b', 0), 162197), (('b', 1), 0))))
                )

    def test_batch_via_transpose_truediv(self) -> None:

        f1 = ff.parse('s(2,2)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,2)|v(int)|c(I,str)').rename('b')
        post = (Batch.from_frames((f1, f2)).via_T / Series((2, 1))).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), -44008.5), (('a', 1), 92867.0), (('b', 0), -44008.5), (('b', 1), 92867.0))), ('ztsv', ((('a', 0), 81098.5), (('a', 1), -41157.0), (('b', 0), 81098.5), (('b', 1), -41157.0))))
                )

    def test_batch_via_transpose_floordiv(self) -> None:

        f1 = ff.parse('s(2,2)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,2)|v(int)|c(I,str)').rename('b')
        post = (Batch.from_frames((f1, f2)).via_T // Series((3, 1))).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), -29339), (('a', 1), 92867), (('b', 0), -29339), (('b', 1), 92867))), ('ztsv', ((('a', 0), 54065), (('a', 1), -41157), (('b', 0), 54065), (('b', 1), -41157))))
                )

    def test_batch_via_transpose_mod(self) -> None:

        f1 = ff.parse('s(2,2)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,2)|v(int)|c(I,str)').rename('b')
        post = (Batch.from_frames((f1, f2)).via_T % Series((2, 4))).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), 1), (('a', 1), 3), (('b', 0), 1), (('b', 1), 3))), ('ztsv', ((('a', 0), 1), (('a', 1), 3), (('b', 0), 1), (('b', 1), 3))))
                )

    def test_batch_via_transpose_pow(self) -> None:

        f1 = ff.parse('s(2,2)|v(int64)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,2)|v(int64)|c(I,str)').rename('b')
        post = (Batch.from_frames((f1, f2)).via_T ** Series((2, 1))).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), 7746992289), (('a', 1), 92867), (('b', 0), 7746992289), (('b', 1), 92867))), ('ztsv', ((('a', 0), 26307866809), (('a', 1), -41157), (('b', 0), 26307866809), (('b', 1), -41157))))
                )

    def test_batch_via_transpose_lshift(self) -> None:

        f1 = ff.parse('s(2,2)|v(int64)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,2)|v(int64)|c(I,str)').rename('b')
        post = (Batch.from_frames((f1, f2)).via_T << Series((2, 0))).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), -352068), (('a', 1), 92867), (('b', 0), -352068), (('b', 1), 92867))), ('ztsv', ((('a', 0), 648788), (('a', 1), -41157), (('b', 0), 648788), (('b', 1), -41157))))
                )

    def test_batch_via_transpose_rshift(self) -> None:

        f1 = ff.parse('s(2,2)|v(int64)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,2)|v(int64)|c(I,str)').rename('b')
        post = (Batch.from_frames((f1, f2)).via_T >> Series((2, 0))).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), -22005), (('a', 1), 92867), (('b', 0), -22005), (('b', 1), 92867))), ('ztsv', ((('a', 0), 40549), (('a', 1), -41157), (('b', 0), 40549), (('b', 1), -41157))))
                )


    def test_batch_via_transpose_and(self) -> None:

        f1 = ff.parse('s(2,2)|v(bool)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,2)|v(bool)|c(I,str)').rename('b')
        post = (Batch.from_frames((f1, f2)).via_T & Series((True, False))).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), False), (('a', 1), False), (('b', 0), False), (('b', 1), False))), ('ztsv', ((('a', 0), False), (('a', 1), False), (('b', 0), False), (('b', 1), False)))))

    def test_batch_via_transpose_xor(self) -> None:

        f1 = ff.parse('s(2,2)|v(bool)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,2)|v(bool)|c(I,str)').rename('b')
        post = (Batch.from_frames((f1, f2)).via_T ^ Series((True, False))).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), True), (('a', 1), False), (('b', 0), True), (('b', 1), False))), ('ztsv', ((('a', 0), True), (('a', 1), False), (('b', 0), True), (('b', 1), False))))
                )

    def test_batch_via_transpose_or(self) -> None:

        f1 = ff.parse('s(2,2)|v(bool)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,2)|v(bool)|c(I,str)').rename('b')
        post = (Batch.from_frames((f1, f2)).via_T | Series((True, False))).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), True), (('a', 1), False), (('b', 0), True), (('b', 1), False))), ('ztsv', ((('a', 0), True), (('a', 1), False), (('b', 0), True), (('b', 1), False))))
                )

    def test_batch_via_transpose_lt(self) -> None:

        f1 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('b')
        post = (Batch.from_frames((f1, f2)).via_T < Series((0, 10000))).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), True), (('a', 1), False), (('b', 0), True), (('b', 1), False))), ('ztsv', ((('a', 0), False), (('a', 1), True), (('b', 0), False), (('b', 1), True))), ('zUvW', ((('a', 0), True), (('a', 1), False), (('b', 0), True), (('b', 1), False))))
                )

    def test_batch_via_transpose_le(self) -> None:

        f1 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('b')
        post = (Batch.from_frames((f1, f2)).via_T <= Series((0, 92867))).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), True), (('a', 1), True), (('b', 0), True), (('b', 1), True))), ('ztsv', ((('a', 0), False), (('a', 1), True), (('b', 0), False), (('b', 1), True))), ('zUvW', ((('a', 0), True), (('a', 1), True), (('b', 0), True), (('b', 1), True))))
                )

    def test_batch_via_transpose_eq(self) -> None:

        f1 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('b')
        post = (Batch.from_frames((f1, f2)).via_T == Series((0, 92867))).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), False), (('a', 1), True), (('b', 0), False), (('b', 1), True))), ('ztsv', ((('a', 0), False), (('a', 1), False), (('b', 0), False), (('b', 1), False))), ('zUvW', ((('a', 0), False), (('a', 1), False), (('b', 0), False), (('b', 1), False))))
                )

    def test_batch_via_transpose_ne(self) -> None:

        f1 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('b')
        post = (Batch.from_frames((f1, f2)).via_T != Series((0, 92867))).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), True), (('a', 1), False), (('b', 0), True), (('b', 1), False))), ('ztsv', ((('a', 0), True), (('a', 1), True), (('b', 0), True), (('b', 1), True))), ('zUvW', ((('a', 0), True), (('a', 1), True), (('b', 0), True), (('b', 1), True))))
                )


    def test_batch_via_transpose_gt(self) -> None:

        f1 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('b')
        post = (Batch.from_frames((f1, f2)).via_T > Series((0, 92867))).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), False), (('a', 1), False), (('b', 0), False), (('b', 1), False))), ('ztsv', ((('a', 0), True), (('a', 1), False), (('b', 0), True), (('b', 1), False))), ('zUvW', ((('a', 0), False), (('a', 1), False), (('b', 0), False), (('b', 1), False))))
                )

    def test_batch_via_transpose_ge(self) -> None:

        f1 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('b')
        post = (Batch.from_frames((f1, f2)).via_T >= Series((0, 92867))).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), False), (('a', 1), True), (('b', 0), False), (('b', 1), True))), ('ztsv', ((('a', 0), True), (('a', 1), False), (('b', 0), True), (('b', 1), False))), ('zUvW', ((('a', 0), False), (('a', 1), False), (('b', 0), False), (('b', 1), False))))
                )

    def test_batch_via_transpose_radd(self) -> None:

        f1 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('b')
        post = (Series((0, 92867)) + Batch.from_frames((f1, f2)).via_T).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), -88017), (('a', 1), 185734), (('b', 0), -88017), (('b', 1), 185734))), ('ztsv', ((('a', 0), 162197), (('a', 1), 51710), (('b', 0), 162197), (('b', 1), 51710))), ('zUvW', ((('a', 0), -3648), (('a', 1), 184168), (('b', 0), -3648), (('b', 1), 184168))))
                )

    def test_batch_via_transpose_rsub(self) -> None:

        f1 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('b')
        post = (Series((0, 92867)) - Batch.from_frames((f1, f2)).via_T).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), 88017), (('a', 1), 0), (('b', 0), 88017), (('b', 1), 0))), ('ztsv', ((('a', 0), -162197), (('a', 1), 134024), (('b', 0), -162197), (('b', 1), 134024))), ('zUvW', ((('a', 0), 3648), (('a', 1), 1566), (('b', 0), 3648), (('b', 1), 1566))))
                )

    def test_batch_via_transpose_rmul(self) -> None:

        f1 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('b')
        post = (Series((0, 1)) * Batch.from_frames((f1, f2)).via_T).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), 0), (('a', 1), 92867), (('b', 0), 0), (('b', 1), 92867))), ('ztsv', ((('a', 0), 0), (('a', 1), -41157), (('b', 0), 0), (('b', 1), -41157))), ('zUvW', ((('a', 0), 0), (('a', 1), 91301), (('b', 0), 0), (('b', 1), 91301))))
                )

    def test_batch_via_transpose_rtruediv(self) -> None:

        f1 = ff.parse('s(2,3)|v(int64)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,3)|v(int64)|c(I,str)').rename('b')
        post = (Series((92867, 10000)) / Batch.from_frames((f1, f2)).via_T).to_frame()
        self.assertEqual(round(post, 2).to_pairs(),
                (('zZbu', ((('a', 0), -1.06), (('a', 1), 0.11), (('b', 0), -1.06), (('b', 1), 0.11))), ('ztsv', ((('a', 0), 0.57), (('a', 1), -0.24), (('b', 0), 0.57), (('b', 1), -0.24))), ('zUvW', ((('a', 0), -25.46), (('a', 1), 0.11), (('b', 0), -25.46), (('b', 1), 0.11))))
                )

    def test_batch_via_transpose_rfloordiv(self) -> None:

        f1 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('b')
        post = (Series((1, 2)) // Batch.from_frames((f1, f2)).via_T).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), -1), (('a', 1), 0), (('b', 0), -1), (('b', 1), 0))), ('ztsv', ((('a', 0), 0), (('a', 1), -1), (('b', 0), 0), (('b', 1), -1))), ('zUvW', ((('a', 0), -1), (('a', 1), 0), (('b', 0), -1), (('b', 1), 0))))
                )

    def test_batch_via_transpose_via_fill_value(self) -> None:

        f1 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('b')
        post = (Batch.from_frames((f1, f2)).via_T.via_fill_value(0) * Series((1,))).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), -88017), (('a', 1), 0), (('b', 0), -88017), (('b', 1), 0))), ('ztsv', ((('a', 0), 162197), (('a', 1), 0), (('b', 0), 162197), (('b', 1), 0))), ('zUvW', ((('a', 0), -3648), (('a', 1), 0), (('b', 0), -3648), (('b', 1), 0))))
                )

    def test_batch_via_fill_value_via_transpose(self) -> None:

        f1 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('b')
        post = (Batch.from_frames((f1, f2)).via_fill_value(0).via_T * Series((1,))).to_frame()
        self.assertEqual(post.to_pairs(),
                (('zZbu', ((('a', 0), -88017), (('a', 1), 0), (('b', 0), -88017), (('b', 1), 0))), ('ztsv', ((('a', 0), 162197), (('a', 1), 0), (('b', 0), 162197), (('b', 1), 0))), ('zUvW', ((('a', 0), -3648), (('a', 1), 0), (('b', 0), -3648), (('b', 1), 0))))
                )

    #---------------------------------------------------------------------------

    def test_batch_astype_a(self) -> None:

        f1 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('b')
        post = Batch.from_frames((f1, f2)).astype(str).to_frame()
        self.assertEqual(
                [dt.kind for dt in post.dtypes.values],
                ['U', 'U', 'U']
                )

    def test_batch_astype_getitem_a(self) -> None:

        f1 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('a')
        f2 = ff.parse('s(2,3)|v(int)|c(I,str)').rename('b')

        col = f1.columns[0]
        batch = Batch.from_frames((f1, f2))
        batch = batch.astype[[col]]
        self.assertNotIsInstance(batch, Batch)
        batch = batch(object)
        self.assertIsInstance(batch, Batch)

        post = batch.to_frame()[[col]]

        expected = Frame.from_concat(
                (f1[[col]], f2[[col]]),
                index=IndexHierarchy.from_product(('a', 'b'), f1.index),
                )
        self.assertTrue(post.equals(expected))
        self.assertFalse(post.equals(expected, compare_dtype=True))

    #---------------------------------------------------------------------------

    def test_bus_to_quilt_a(self) -> None:
        f1 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(2,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        b1 = Batch.from_frames((f1, f2)).iloc[:, 2].to_bus()

        self.assertEqual(
                Quilt(b1, retain_labels=True).to_frame().to_pairs(),
                (('zUvW', ((('a', 34715), 'ztsv'), (('a', -3648), 'zUvW'), (('b', 34715), 'ztsv'), (('b', -3648), 'zUvW'))),)
                )

    #---------------------------------------------------------------------------

    def test_batch_reduce_a(self) -> None:
        f1 = ff.parse('s(3,3)|v(str)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(3,3)|v(str)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2))

        f3 = post.reduce.from_func(lambda f: f.iloc[1:, 1:]).to_frame()
        self.assertEqual(f3.to_pairs(),
                (('ztsv', ((('a', -3648), 'zJnC'), (('a', 91301), 'zDdR'), (('b', -3648), 'zJnC'), (('b', 91301), 'zDdR'))), ('zUvW', ((('a', -3648), 'zUvW'), (('a', 91301), 'zkuW'), (('b', -3648), 'zUvW'), (('b', 91301), 'zkuW'))))
                )

    def test_batch_reduce_b(self) -> None:
        f1 = ff.parse('s(3,3)|v(int64)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(3,3)|v(int64)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2))

        f3 = (post.reduce.from_map_func(np.min) * 100).to_frame()
        self.assertEqual(f3.to_pairs(),
                (('zZbu', ((('a', 'a'), -8801700), (('b', 'b'), -8801700))), ('ztsv', ((('a', 'a'), -4115700), (('b', 'b'), -4115700))), ('zUvW', ((('a', 'a'), -364800), (('b', 'b'), -364800))))
                )

    def test_batch_reduce_c(self) -> None:
        f1 = ff.parse('s(3,3)|v(int64)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(3,3)|v(int64)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2))

        f3 = post.reduce.from_label_map({'ztsv': np.min, 'zZbu': np.max}).to_frame()
        self.assertEqual(f3.to_pairs(),
                (('ztsv', ((('a', 'a'), -41157), (('b', 'b'), -41157))), ('zZbu', ((('a', 'a'), 92867), (('b', 'b'), 92867))))
                )

    def test_batch_reduce_d(self) -> None:
        f1 = ff.parse('s(3,3)|v(int64)|c(I,str)|i(I,int)').rename('a')
        f2 = ff.parse('s(3,3)|v(int64)|c(I,str)|i(I,int)').rename('b')
        post = Batch.from_frames((f1, f2))

        f3 = post.reduce.from_label_pair_map({
                ('ztsv', 'z-min'): np.min,
                ('zZbu', 'z-max'): np.max,
                }).to_frame()
        self.assertEqual(f3.to_pairs(),
                (('z-min', ((('a', 'a'), -41157), (('b', 'b'), -41157))), ('z-max', ((('a', 'a'), 92867), (('b', 'b'), 92867))))
                )

if __name__ == '__main__':
    import unittest
    unittest.main()
