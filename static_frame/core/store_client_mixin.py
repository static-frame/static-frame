
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


if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame # pylint: disable=W0611 #pragma: no cover


T = tp.TypeVar('T')

class StoreClientMixin(tp.Generic[T]):
    '''
    Mixin class for multi-table IO via Store interfaces.
    '''
    __slots__ = ()

    _config: StoreConfigMap
    items: tp.Callable[..., tp.Iterator[tp.Tuple[tp.Hashable, tp.Any]]]
    _from_store: tp.Callable[..., T]

    #---------------------------------------------------------------------------
    # constructors by data format

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_zip_tsv(cls: tp.Type['StoreClientMixin[T]'],
            fp: PathSpecifier,
            config: StoreConfigMapInitializer = None,
            ) -> T:
        '''
        Given a file path to zipped TSV :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreZipTSV(fp)
        return tp.cast(T, cls._from_store(store, config))

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_zip_csv(cls: tp.Type['StoreClientMixin[T]'],
            fp: PathSpecifier,
            config: StoreConfigMapInitializer = None
            ) -> T:
        '''
        Given a file path to zipped CSV :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreZipCSV(fp)
        return tp.cast(T, cls._from_store(store, config))

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_zip_pickle(cls,
            fp: PathSpecifier,
            config: StoreConfigMapInitializer = None
            ) -> T:
        '''
        Given a file path to zipped pickle :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreZipPickle(fp)
        return tp.cast(T, cls._from_store(store, config))


    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_zip_parquet(cls,
            fp: PathSpecifier,
            config: StoreConfigMapInitializer = None
            ) -> T:
        '''
        Given a file path to zipped parquet :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreZipParquet(fp)
        return tp.cast(T, cls._from_store(store, config))


    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_xlsx(cls,
            fp: PathSpecifier,
            config: StoreConfigMapInitializer = None
            ) -> T:
        '''
        Given a file path to an XLSX :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        # how to pass configuration for multiple sheets?
        store = StoreXLSX(fp)
        return tp.cast(T, cls._from_store(store, config))


    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_sqlite(cls,
            fp: PathSpecifier,
            config: StoreConfigMapInitializer = None
            ) -> T:
        '''
        Given a file path to an SQLite :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreSQLite(fp)
        return tp.cast(T, cls._from_store(store, config))


    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_hdf5(cls,
            fp: PathSpecifier,
            config: StoreConfigMapInitializer = None
            ) -> T:
        '''
        Given a file path to a HDF5 :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreHDF5(fp)
        return tp.cast(T, cls._from_store(store, config))


    #---------------------------------------------------------------------------
    # exporters

    @doc_inject(selector='bus_exporter')
    def to_zip_tsv(self,
            fp: PathSpecifier,
            config: StoreConfigMapInitializer = None
            ) -> None:
        '''
        Write the complete :obj:`Bus` as a zipped archive of TSV files.

        {args}
        '''
        store = StoreZipTSV(fp)
        config = config if not config is None else self._config
        store.write(self.items(), config=config)

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
        store.write(self.items(), config=config)

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
        store.write(self.items(), config=config)

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
        store.write(self.items(), config=config)

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
        store.write(self.items(), config=config)

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
        store.write(self.items(), config=config)

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
        store.write(self.items(), config=config)
