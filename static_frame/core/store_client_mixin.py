
import typing as tp


from static_frame.core.doc_str import doc_inject
from static_frame.core.store import StoreConfigMapInitializer
from static_frame.core.store_hdf5 import StoreHDF5
from static_frame.core.store_sqlite import StoreSQLite
from static_frame.core.store_xlsx import StoreXLSX
from static_frame.core.store_zip import StoreZipCSV
from static_frame.core.store_zip import StoreZipParquet
from static_frame.core.store_zip import StoreZipPickle
from static_frame.core.store_zip import StoreZipTSV
from static_frame.core.util import PathSpecifier
from static_frame.core.store import StoreConfigMap



# NOTE: wanted this to inherit from tp.Generic[T], such that values returned from constructors would be known, but this breaks in 3.6 with: metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its bases

class StoreClientMixin:
    '''
    Mixin class for multi-table IO via Store interfaces.
    '''
    # NOTE: assignment of metaclass above is only needed for Python 3.6

    __slots__ = ()

    _config: StoreConfigMap
    _from_store: tp.Callable[..., tp.Any]
    _items_store: tp.Callable[..., tp.Iterator[tp.Tuple[tp.Hashable, tp.Any]]]

    #---------------------------------------------------------------------------
    # constructors by data format

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_zip_tsv(cls,
            fp: PathSpecifier,
            config: StoreConfigMapInitializer = None,
            **kwargs: tp.Any,
            ) -> 'StoreClientMixin':
        '''
        Given a file path to zipped TSV :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreZipTSV(fp)
        return cls._from_store(store, #type: ignore
                config=config,
                **kwargs,
                )

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_zip_csv(cls,
            fp: PathSpecifier,
            config: StoreConfigMapInitializer = None,
            **kwargs: tp.Any,
            ) -> 'StoreClientMixin':
        '''
        Given a file path to zipped CSV :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreZipCSV(fp)
        return cls._from_store(store, #type: ignore
                config=config,
                **kwargs,
                )

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_zip_pickle(cls,
            fp: PathSpecifier,
            config: StoreConfigMapInitializer = None,
            **kwargs: tp.Any,
            ) -> 'StoreClientMixin':
        '''
        Given a file path to zipped pickle :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreZipPickle(fp)
        return cls._from_store(store, #type: ignore
                config=config,
                **kwargs,
                )


    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_zip_parquet(cls,
            fp: PathSpecifier,
            config: StoreConfigMapInitializer = None,
            **kwargs: tp.Any,
            ) -> 'StoreClientMixin':
        '''
        Given a file path to zipped parquet :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreZipParquet(fp)
        return cls._from_store(store, #type: ignore
                config=config,
                **kwargs,
                )


    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_xlsx(cls,
            fp: PathSpecifier,
            config: StoreConfigMapInitializer = None,
            **kwargs: tp.Any,
            ) -> 'StoreClientMixin':
        '''
        Given a file path to an XLSX :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        # how to pass configuration for multiple sheets?
        store = StoreXLSX(fp)
        return cls._from_store(store, #type: ignore
                config=config,
                **kwargs,
                )


    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_sqlite(cls,
            fp: PathSpecifier,
            config: StoreConfigMapInitializer = None,
            **kwargs: tp.Any,
            ) -> 'StoreClientMixin':
        '''
        Given a file path to an SQLite :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreSQLite(fp)
        return cls._from_store(store, #type: ignore
                config=config,
                **kwargs,
                )


    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_hdf5(cls,
            fp: PathSpecifier,
            config: StoreConfigMapInitializer = None,
            **kwargs: tp.Any,
            ) -> 'StoreClientMixin':
        '''
        Given a file path to a HDF5 :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreHDF5(fp)
        return cls._from_store(store, #type: ignore
                config=config,
                **kwargs,
                )


    #---------------------------------------------------------------------------
    # exporters

    @doc_inject(selector='bus_exporter')
    def to_zip_tsv(self,
            fp: PathSpecifier,
            config: StoreConfigMapInitializer = None,
            ) -> None:
        '''
        Write the complete :obj:`Bus` as a zipped archive of TSV files.

        {args}
        '''
        store = StoreZipTSV(fp)
        config = config if not config is None else self._config
        store.write(self._items_store(), config=config)

    @doc_inject(selector='bus_exporter')
    def to_zip_csv(self,
            fp: PathSpecifier,
            config: StoreConfigMapInitializer = None
            ) -> None:
        '''
        Write the complete :obj:`Bus` as a zipped archive of CSV files.

        {args}
        '''
        store = StoreZipCSV(fp)
        config = config if not config is None else self._config
        store.write(self._items_store(), config=config)

    @doc_inject(selector='bus_exporter')
    def to_zip_pickle(self,
            fp: PathSpecifier,
            config: StoreConfigMapInitializer = None
            ) -> None:
        '''
        Write the complete :obj:`Bus` as a zipped archive of pickles.

        {args}
        '''
        store = StoreZipPickle(fp)
        # config must be None for pickels, will raise otherwise
        store.write(self._items_store(), config=config)

    @doc_inject(selector='bus_exporter')
    def to_zip_parquet(self,
            fp: PathSpecifier,
            config: StoreConfigMapInitializer = None
            ) -> None:
        '''
        Write the complete :obj:`Bus` as a zipped archive of parquet files.

        {args}
        '''
        store = StoreZipParquet(fp)
        config = config if not config is None else self._config
        store.write(self._items_store(), config=config)

    @doc_inject(selector='bus_exporter')
    def to_xlsx(self,
            fp: PathSpecifier,
            config: StoreConfigMapInitializer = None
            ) -> None:
        '''
        Write the complete :obj:`Bus` as a XLSX workbook.

        {args}
        '''
        store = StoreXLSX(fp)
        config = config if not config is None else self._config
        store.write(self._items_store(), config=config)

    @doc_inject(selector='bus_exporter')
    def to_sqlite(self,
            fp: PathSpecifier,
            config: StoreConfigMapInitializer = None
            ) -> None:
        '''
        Write the complete :obj:`Bus` as an SQLite database file.

        {args}
        '''
        store = StoreSQLite(fp)
        config = config if not config is None else self._config
        store.write(self._items_store(), config=config)

    @doc_inject(selector='bus_exporter')
    def to_hdf5(self,
            fp: PathSpecifier,
            config: StoreConfigMapInitializer = None
            ) -> None:
        '''
        Write the complete :obj:`Bus` as an HDF5 table.

        {args}
        '''
        store = StoreHDF5(fp)
        config = config if not config is None else self._config
        store.write(self._items_store(), config=config)
