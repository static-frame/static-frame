from __future__ import annotations

import zipfile

import typing_extensions as tp

from static_frame.core.doc_str import doc_inject
from static_frame.core.store_sqlite import StoreSQLite
from static_frame.core.store_xlsx import StoreXLSX
from static_frame.core.store_zip import (
    StoreZipCSV,
    StoreZipNPY,
    StoreZipNPZ,
    StoreZipParquet,
    StoreZipPickle,
    StoreZipTSV,
)

if tp.TYPE_CHECKING:
    from static_frame.core.store import Store
    from static_frame.core.store_config import (
        StoreConfigCSV,
        StoreConfigNPY,
        StoreConfigNPZ,
        StoreConfigParquet,
        StoreConfigPickle,
        StoreConfigSQLite,
        StoreConfigTSV,
        StoreConfigXLSX,
        TVStoreConfig,
        TVStoreConfigMapInitializer,
    )
    from static_frame.core.util import (
        TLabel,
        TPathSpecifier,
    )


# NOTE: wanted this to inherit from tp.Generic[T], such that values returned from constructors would be known, but this breaks in 3.6 with: metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its bases


class StoreClientMixin:
    """
    Mixin class for multi-table exporting via Store interfaces.
    """

    # NOTE: assignment of metaclass above is only needed for Python 3.6

    __slots__ = ()

    _store: Store[tp.Any] | None
    _from_store: tp.Callable[..., tp.Any]
    _items_store: tp.Callable[..., tp.Iterator[tuple[TLabel, tp.Any]]]

    def _resolve_config(
        self,
        config: TVStoreConfigMapInitializer[TVStoreConfig],
    ) -> TVStoreConfigMapInitializer[TVStoreConfig]:
        if config is not None:
            return config

        store: Store[tp.Any] | None = None

        if hasattr(self, '_bus'):  # this is Quilt
            store = self._bus._store  # pyright: ignore

        elif hasattr(self, '_store'):  # this is Bus
            store = self._store

        return None if store is None else store._config

    # ---------------------------------------------------------------------------
    # exporters

    @doc_inject(selector='store_client_exporter')
    def to_zip_tsv(
        self,
        fp: TPathSpecifier,
        /,
        *,
        config: TVStoreConfigMapInitializer[StoreConfigTSV] = None,
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> None:
        """
        Write the complete :obj:`Bus` as a zipped archive of TSV files.

        {args}
        """
        store = StoreZipTSV(fp, config=self._resolve_config(config))
        store.write(self._items_store(), compression=compression)

    @doc_inject(selector='store_client_exporter')
    def to_zip_csv(
        self,
        fp: TPathSpecifier,
        /,
        *,
        config: TVStoreConfigMapInitializer[StoreConfigCSV] = None,
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> None:
        """
        Write the complete :obj:`Bus` as a zipped archive of CSV files.

        {args}
        """
        store = StoreZipCSV(fp, config=self._resolve_config(config))
        store.write(self._items_store(), compression=compression)

    @doc_inject(selector='store_client_exporter')
    def to_zip_pickle(
        self,
        fp: TPathSpecifier,
        /,
        *,
        config: TVStoreConfigMapInitializer[StoreConfigPickle] = None,
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> None:
        """
        Write the complete :obj:`Bus` as a zipped archive of pickles.

        {args}
        """
        store = StoreZipPickle(fp, config=self._resolve_config(config))
        store.write(self._items_store(), compression=compression)

    @doc_inject(selector='store_client_exporter')
    def to_zip_npz(
        self,
        fp: TPathSpecifier,
        /,
        *,
        config: TVStoreConfigMapInitializer[StoreConfigNPZ] = None,
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> None:
        """
        Write the complete :obj:`Bus` as a zipped archive of NPZ files.

        {args}
        """
        store = StoreZipNPZ(fp, config=self._resolve_config(config))
        store.write(self._items_store(), compression=compression)

    @doc_inject(selector='store_client_exporter')
    def to_zip_npy(
        self,
        fp: TPathSpecifier,
        /,
        *,
        config: TVStoreConfigMapInitializer[StoreConfigNPY] = None,
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> None:
        """
        Write the complete :obj:`Bus` as a zipped archive of NPY files.

        {args}
        """
        store = StoreZipNPY(fp, config=self._resolve_config(config))
        store.write(self._items_store(), compression=compression)

    @doc_inject(selector='store_client_exporter')
    def to_zip_parquet(
        self,
        fp: TPathSpecifier,
        /,
        *,
        config: TVStoreConfigMapInitializer[StoreConfigParquet] = None,
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> None:
        """
        Write the complete :obj:`Bus` as a zipped archive of parquet files.

        {args}
        """
        store = StoreZipParquet(fp, config=self._resolve_config(config))
        store.write(self._items_store(), compression=compression)

    @doc_inject(selector='store_client_exporter')
    def to_xlsx(
        self,
        fp: TPathSpecifier,
        /,
        *,
        config: TVStoreConfigMapInitializer[StoreConfigXLSX] = None,
    ) -> None:
        """
        Write the complete :obj:`Bus` as a XLSX workbook.

        {args}
        """
        store = StoreXLSX(fp, config=self._resolve_config(config))
        store.write(self._items_store())

    @doc_inject(selector='store_client_exporter')
    def to_sqlite(
        self,
        fp: TPathSpecifier,
        /,
        *,
        config: TVStoreConfigMapInitializer[StoreConfigSQLite] = None,
    ) -> None:
        """
        Write the complete :obj:`Bus` as an SQLite database file.

        {args}
        """
        store = StoreSQLite(fp, config=self._resolve_config(config))
        store.write(self._items_store())
