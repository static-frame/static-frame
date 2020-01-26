import unittest
# from io import StringIO


from static_frame.core.store import StoreConfig
from static_frame.core.store import StoreConfigMap


from static_frame.test.test_case import TestCase
# from static_frame.test.test_case import temp_file
from static_frame.core.exception import ErrorInitStoreConfig



class TestUnit(TestCase):


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
            _ = StoreConfigMap({'a': object()}) # type: ignore

        with self.assertRaises(ErrorInitStoreConfig):
            _ = StoreConfigMap(default=object()) # type: ignore


if __name__ == '__main__':
    unittest.main()
