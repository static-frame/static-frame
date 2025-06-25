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
    from static_frame.core.store_config import (
        StoreConfigMap,  # pragma: no cover
        StoreConfigMapInitializer,
    )  # pragma: no cover
    from static_frame.core.util import (
        TLabel,  # pragma: no cover
        TPathSpecifier,  # pragma: no cover
    )


# NOTE: wanted this to inherit from tp.Generic[T], such that values returned from constructors would be known, but this breaks in 3.6 with: metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its bases


class StoreClientMixin:
    """
    Mixin class for multi-table exporting via Store interfaces.
    """

    # NOTE: assignment of metaclass above is only needed for Python 3.6

    __slots__ = ()

    _config: StoreConfigMap
    _from_store: tp.Callable[..., tp.Any]
    _items_store: tp.Callable[..., tp.Iterator[tp.Tuple[TLabel, tp.Any]]]

    def _filter_config(
        self,
        config: StoreConfigMapInitializer,
    ) -> StoreConfigMapInitializer:
        if config is not None:
            return config
        if hasattr(self, '_bus'):  # this is Quilt
            return self._bus._config  # type: ignore
        # Yarn does not have a _config attr
        return getattr(self, '_config', None)

    # ---------------------------------------------------------------------------
    # exporters

    @doc_inject(selector='store_client_exporter')
    def to_zip_tsv(
        self,
        fp: TPathSpecifier,
        /,
        *,
        config: StoreConfigMapInitializer = None,
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> None:
        """
        Write the complete :obj:`Bus` as a zipped archive of TSV files.

        {args}
        """
        store = StoreZipTSV(fp)
        config = self._filter_config(config)
        store.write(self._items_store(), config=config, compression=compression)

    @doc_inject(selector='store_client_exporter')
    def to_zip_csv(
        self,
        fp: TPathSpecifier,
        /,
        *,
        config: StoreConfigMapInitializer = None,
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> None:
        """
        Write the complete :obj:`Bus` as a zipped archive of CSV files.

        {args}
        """
        store = StoreZipCSV(fp)
        config = self._filter_config(config)
        store.write(self._items_store(), config=config, compression=compression)

    @doc_inject(selector='store_client_exporter')
    def to_zip_pickle(
        self,
        fp: TPathSpecifier,
        /,
        *,
        config: StoreConfigMapInitializer = None,
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> None:
        """
        Write the complete :obj:`Bus` as a zipped archive of pickles.

        {args}
        """
        store = StoreZipPickle(fp)
        config = self._filter_config(config)
        store.write(self._items_store(), config=config, compression=compression)

    @doc_inject(selector='store_client_exporter')
    def to_zip_npz(
        self,
        fp: TPathSpecifier,
        /,
        *,
        config: StoreConfigMapInitializer = None,
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> None:
        """
        Write the complete :obj:`Bus` as a zipped archive of NPZ files.

        {args}
        """
        store = StoreZipNPZ(fp)
        config = self._filter_config(config)
        store.write(self._items_store(), config=config, compression=compression)

    @doc_inject(selector='store_client_exporter')
    def to_zip_npy(
        self,
        fp: TPathSpecifier,
        /,
        *,
        config: StoreConfigMapInitializer = None,
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> None:
        """
        Write the complete :obj:`Bus` as a zipped archive of NPY files.

        {args}
        """
        store = StoreZipNPY(fp)
        config = self._filter_config(config)
        store.write(self._items_store(), config=config, compression=compression)

    @doc_inject(selector='store_client_exporter')
    def to_zip_parquet(
        self,
        fp: TPathSpecifier,
        /,
        *,
        config: StoreConfigMapInitializer = None,
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> None:
        """
        Write the complete :obj:`Bus` as a zipped archive of parquet files.

        {args}
        """
        store = StoreZipParquet(fp)
        config = self._filter_config(config)
        store.write(self._items_store(), config=config, compression=compression)

    @doc_inject(selector='store_client_exporter')
    def to_xlsx(
        self,
        fp: TPathSpecifier,
        /,
        *,
        config: StoreConfigMapInitializer = None,
    ) -> None:
        """
        Write the complete :obj:`Bus` as a XLSX workbook.

        {args}
        """
        store = StoreXLSX(fp)
        config = self._filter_config(config)
        store.write(self._items_store(), config=config)

    @doc_inject(selector='store_client_exporter')
    def to_sqlite(
        self,
        fp: TPathSpecifier,
        /,
        *,
        config: StoreConfigMapInitializer = None,
    ) -> None:
        """
        Write the complete :obj:`Bus` as an SQLite database file.

        {args}
        """
        store = StoreSQLite(fp)
        config = self._filter_config(config)
        store.write(self._items_store(), config=config)
