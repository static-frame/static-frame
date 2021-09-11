import unittest
from itertools import product
# from io import StringIO
import numpy as np

from static_frame.core.store import Store
from static_frame.core.store import StoreConfig
from static_frame.core.store import StoreConfigHE
from static_frame.core.store import StoreConfigMap
from static_frame.core.frame import Frame

from static_frame.test.test_case import TestCase
# from static_frame.test.test_case import temp_file
from static_frame.core.exception import ErrorInitStoreConfig
from static_frame.core.exception import StoreParameterConflict



class TestUnit(TestCase):


    #---------------------------------------------------------------------------

    def test_store_init_a(self) -> None:

        class StoreDerived(Store):
            _EXT = frozenset(('.txt',))

        st = StoreDerived(fp='foo.txt')
        self.assertTrue(np.isnan(st._last_modified))

    def test_store_config_map_init_a(self) -> None:
        maps = {'a': StoreConfig(index_depth=2),
                'b': StoreConfig(index_depth=3, label_encoder=str)}

        with self.assertRaises(ErrorInitStoreConfig):
            sc1m = StoreConfigMap.from_initializer(maps)

    def test_store_config_map_init_b(self) -> None:
        maps = {'a': StoreConfig(index_depth=2, label_encoder=str),
                'b': StoreConfig(index_depth=3, label_encoder=str)}
        default = StoreConfig(label_encoder=str)

        sc1m = StoreConfigMap(maps, default=default)
        self.assertEqual(sc1m.default.label_encoder, str)

    def test_store_config_map_init_c(self) -> None:
        maps1 = {'a': StoreConfig(read_max_workers=2),
                'b': StoreConfig(read_max_workers=3)}

        default = StoreConfig(read_max_workers=2)

        with self.assertRaises(ErrorInitStoreConfig):
            StoreConfigMap(maps1, default=default) # Config has conflicting info

        maps2 = {'a': StoreConfig(read_max_workers=2),
                'b': StoreConfig(read_max_workers=2)}

        with self.assertRaises(ErrorInitStoreConfig):
            StoreConfigMap(maps2) # Default is None

        sc1m = StoreConfigMap(maps2, default=default)
        self.assertEqual(sc1m.default.read_max_workers, 2)

    def test_store_config_map_init_d(self) -> None:
        maps1 = {'a': StoreConfig(read_chunksize=2),
                'b': StoreConfig(read_chunksize=3)}

        default = StoreConfig(read_chunksize=2)

        with self.assertRaises(ErrorInitStoreConfig):
            StoreConfigMap(maps1, default=default) # Config has conflicting info

        maps2 = {'a': StoreConfig(read_chunksize=2),
                'b': StoreConfig(read_chunksize=2)}

        with self.assertRaises(ErrorInitStoreConfig):
            StoreConfigMap(maps2) # Default is 1

        sc1m = StoreConfigMap(maps2, default=default)
        self.assertEqual(sc1m.default.read_chunksize, 2)


    def test_store_config_map_init_e(self) -> None:
        maps1 = {'a': StoreConfig(write_max_workers=2),
                'b': StoreConfig(write_max_workers=3)}

        default = StoreConfig(write_max_workers=2)

        with self.assertRaises(ErrorInitStoreConfig):
            StoreConfigMap(maps1, default=default) # Config has conflicting info

        maps2 = {'a': StoreConfig(write_max_workers=2),
                'b': StoreConfig(write_max_workers=2)}

        with self.assertRaises(ErrorInitStoreConfig):
            StoreConfigMap(maps2) # Default is None

        sc1m = StoreConfigMap(maps2, default=default)
        self.assertEqual(sc1m.default.write_max_workers, 2)

    def test_store_config_map_init_f(self) -> None:
        maps1 = {'a': StoreConfig(write_chunksize=2),
                'b': StoreConfig(write_chunksize=3)}

        default = StoreConfig(write_chunksize=2)

        with self.assertRaises(ErrorInitStoreConfig):
            StoreConfigMap(maps1, default=default) # Config has conflicting info

        maps2 = {'a': StoreConfig(write_chunksize=2),
                'b': StoreConfig(write_chunksize=2)}

        with self.assertRaises(ErrorInitStoreConfig):
            StoreConfigMap(maps2) # Default is 1

        sc1m = StoreConfigMap(maps2, default=default)
        self.assertEqual(sc1m.default.write_chunksize, 2)

    #---------------------------------------------------------------------------
    def test_store_config_he_a(self) -> None:
        he_kwargs = dict(
                index_depth=1,
                columns_depth=1,
                consolidate_blocks=True,
                skip_header=1,
                skip_footer=1,
                trim_nadir=True,
                include_index=True,
                include_index_name=True,
                include_columns=True,
                include_columns_name=True,
                merge_hierarchical_labels=True,
                read_max_workers=1,
                read_chunksize=1,
                write_max_workers=1,
                write_chunksize=1,
        )

        kwargs = dict(**he_kwargs,
                label_encoder=lambda x: x,
                label_decoder=lambda x: x,
        )

        for (depth_levels, columns_select, dtypes) in product(
            (None, 1, [1, 2], (1, 2)),
            (None, ['a'], ('a',)),
            (None, 'int', int, np.int64, [int], (int,), {'a': int}),
        ):
            config = StoreConfig(**kwargs, # type: ignore [arg-type]
                    index_name_depth_level=depth_levels,
                    columns_name_depth_level=depth_levels,
                    columns_select=columns_select,
                    dtypes=dtypes,
            )

            config_he = StoreConfigHE(**he_kwargs, # type: ignore [arg-type]
                    index_name_depth_level=depth_levels,
                    columns_name_depth_level=depth_levels,
                    columns_select=columns_select,
                    dtypes=dtypes,
            )
            self.assertNotEqual(config_he, config)
            self.assertEqual(config_he, config.to_store_config_he())
            self.assertTrue(isinstance(hash(config_he), int))


    def test_store_config_he_b(self) -> None:

        config1 = StoreConfigHE(index_depth=1)
        config2 = StoreConfigHE(index_depth=2)
        self.assertNotEqual(config1, config2)

        self.assertTrue(config1 is not None)
        self.assertFalse(config1 == None) #pylint: disable=C0121

        config1 = StoreConfigHE(columns_select=['a'])
        config2 = StoreConfigHE(columns_select=('b',))
        self.assertNotEqual(config1, config2)

        config1 = StoreConfigHE(dtypes=str)
        config2 = StoreConfigHE(dtypes=int)
        self.assertNotEqual(config1, config2)

        config1 = StoreConfigHE(dtypes=str)
        config2 = StoreConfigHE(dtypes=dict(a=int))
        self.assertNotEqual(config1, config2)

        config1 = StoreConfigHE(index_name_depth_level=None)
        config2 = StoreConfigHE(index_name_depth_level=(1, 2))
        self.assertNotEqual(config1, config2)

    def test_store_config_not_hashable(self) -> None:
        with self.assertRaises(NotImplementedError):
            hash(StoreConfig())


    #---------------------------------------------------------------------------
    def test_store_config_map_a(self) -> None:

        sc1 = StoreConfig(index_depth=3, columns_depth=3)
        sc1m = StoreConfigMap.from_config(sc1)
        self.assertEqual(sc1m['a'].index_depth, 3)
        self.assertEqual(sc1m['b'].index_depth, 3)

        sc2 = StoreConfig(include_index=False)
        sc2m = StoreConfigMap.from_config(sc2)
        self.assertEqual(sc2m['a'].include_index, False)
        self.assertEqual(sc2m['b'].include_index, False)


    def test_store_config_map_b(self) -> None:

        maps = {'a': StoreConfig(index_depth=2),
                'b': StoreConfig(index_depth=3)}
        sc1m = StoreConfigMap(maps)
        self.assertEqual(sc1m['a'].index_depth, 2)
        self.assertEqual(sc1m['b'].index_depth, 3)
        self.assertEqual(sc1m['c'].index_depth, 0)

    def test_store_config_map_c(self) -> None:
        sc1 = StoreConfig(index_depth=3, columns_depth=3)
        maps = {'a': StoreConfig(index_depth=2),
                'b': StoreConfig(index_depth=3)}
        sc1m = StoreConfigMap(maps)

        sc2m = StoreConfigMap.from_initializer(sc1)
        self.assertEqual(sc2m['a'].index_depth, 3)

        sc3m = StoreConfigMap.from_initializer(sc1m)
        self.assertEqual(sc3m['a'].index_depth, 2)
        self.assertEqual(sc3m['b'].index_depth, 3)

        sc4m = StoreConfigMap.from_initializer(maps)
        self.assertEqual(sc4m['a'].index_depth, 2)
        self.assertEqual(sc4m['b'].index_depth, 3)


    def test_store_config_map_d(self) -> None:
        with self.assertRaises(ErrorInitStoreConfig):
            _ = StoreConfigMap({'a': object()}) #type: ignore

        with self.assertRaises(ErrorInitStoreConfig):
            _ = StoreConfigMap(default=object()) #type: ignore


    def test_store_get_field_names_and_dtypes_a(self) -> None:

        f1 = Frame.from_records((('a', True, None),), index=(('a',)), columns=(('x', 'y', 'z')))

        field_names, dtypes = Store.get_field_names_and_dtypes(frame=f1,
                include_index=False,
                include_index_name=True,
                include_columns=False,
                include_columns_name=False,
                )
        self.assertEqual(field_names, range(0, 3))
        self.assertEqual(dtypes,
                [np.dtype('<U1'), np.dtype('bool'), np.dtype('O')])

    def test_store_get_field_names_and_dtypes_b(self) -> None:

        f1 = Frame.from_records((('a', True, None),), index=(('a',)), columns=(('x', 'y', 'z')))

        field_names, dtypes = Store.get_field_names_and_dtypes(frame=f1,
                include_index=False,
                include_index_name=True,
                include_columns=True,
                include_columns_name=False,
                )

        self.assertEqual(field_names.tolist(), ['x', 'y', 'z']) #type: ignore
        self.assertEqual(dtypes,
                [np.dtype('<U1'), np.dtype('bool'), np.dtype('O')])


    def test_store_get_field_names_and_dtypes_c(self) -> None:

        f1 = Frame.from_records((('a', True, None),), index=(('a',)), columns=(('x', 'y', 'z'))).rename(columns='foo')

        with self.assertRaises(StoreParameterConflict):
            field_names, dtypes = Store.get_field_names_and_dtypes(frame=f1,
                    include_index=False,
                    include_index_name=True,
                    include_columns=True,
                    include_columns_name=True,
                    )

        field_names, dtypes = Store.get_field_names_and_dtypes(frame=f1,
                include_index=True,
                include_index_name=False,
                include_columns=True,
                include_columns_name=True,
                )
        self.assertEqual(field_names, ['foo', 'x', 'y', 'z'])
        self.assertTrue(len(field_names) == len(dtypes))


    def test_store_get_field_names_and_dtypes_d(self) -> None:

        from static_frame.core.index_hierarchy import IndexHierarchy
        columns = IndexHierarchy.from_labels(((1, 'a'), (1, 'b'), (2, 'c')), name=('foo', 'bar'))
        f1 = Frame.from_records((('a', True, None),), index=(('a',)), columns=columns)

        field_names, dtypes = Store.get_field_names_and_dtypes(frame=f1,
                include_index=True,
                include_index_name=False,
                include_columns=True,
                include_columns_name=True,
                )
        self.assertEqual(field_names, [('foo', 'bar'), "[1 'a']", "[1 'b']", "[2 'c']"])
        self.assertTrue(len(field_names) == len(dtypes))


        field_names, dtypes = Store.get_field_names_and_dtypes(frame=f1,
                include_index=True,
                include_index_name=False,
                include_columns=True,
                include_columns_name=True,
                force_brackets=True,
                )
        self.assertEqual(field_names, ["['foo' 'bar']", "[1 'a']", "[1 'b']", "[2 'c']"])

        with self.assertRaises(StoreParameterConflict):
            field_names, dtypes = Store.get_field_names_and_dtypes(frame=f1,
                    include_index=True,
                    include_index_name=False,
                    include_columns=True,
                    include_columns_name=False,
                    )

        with self.assertRaises(StoreParameterConflict):
            field_names, dtypes = Store.get_field_names_and_dtypes(frame=f1,
                    include_index=False,
                    include_index_name=False,
                    include_columns=True,
                    include_columns_name=True,
                    )

    #---------------------------------------------------------------------------

    def test_store_config_map_get_default_a(self) -> None:
        maps = {'a': StoreConfig(index_depth=2),
                'b': StoreConfig(index_depth=3)}

        sc1m = StoreConfigMap.from_initializer(maps)
        self.assertTrue(sc1m.default == StoreConfigMap._DEFAULT)

    #---------------------------------------------------------------------------


if __name__ == '__main__':
    unittest.main()
