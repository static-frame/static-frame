from __future__ import annotations

from functools import partial

import numpy as np
import typing_extensions as tp

from static_frame.core.container_util import ContainerMap
from static_frame.core.index_base import IndexBase
from static_frame.core.util import JSONTranslator
from static_frame.core.util import TIndexCtor
from static_frame.core.util import TName

if tp.TYPE_CHECKING:
    from static_frame.core.generic_aliases import TFrameAny  # pragma: no cover
    TDtypeAny = np.dtype[tp.Any] #pragma: no cover

class NPYLabel:
    KEY_NAMES = '__names__'
    KEY_DEPTHS = '__depths__'
    KEY_TYPES = '__types__'
    KEY_TYPES_INDEX = '__types_index__'
    KEY_TYPES_COLUMNS = '__types_columns__'
    FILE_TEMPLATE_VALUES_INDEX = '__values_index_{}__.npy'
    FILE_TEMPLATE_VALUES_COLUMNS = '__values_columns_{}__.npy'
    FILE_TEMPLATE_BLOCKS = '__blocks_{}__.npy'


class JSONMeta:
    '''Metadata for JSON encodings.
    '''
    KEY_NAMES = NPYLabel.KEY_NAMES
    KEY_DEPTHS = NPYLabel.KEY_DEPTHS
    KEY_TYPES = NPYLabel.KEY_TYPES
    KEY_TYPES_INDEX = NPYLabel.KEY_TYPES_INDEX
    KEY_TYPES_COLUMNS = NPYLabel.KEY_TYPES_COLUMNS
    KEY_DTYPES = '__dtypes__'
    KEY_DTYPES_INDEX = '__dtypes_index__'
    KEY_DTYPES_COLUMNS = '__dtypes_columns__'

    @staticmethod
    def _dtype_to_str(dt: TDtypeAny) -> str:
        '''Normalize all dtype strings as platform native
        '''
        dts = dt.str
        if dts[0] == '|':
            return dts
        return '=' + dts[1:]

    @classmethod
    def _index_to_dtype_str(cls, index: IndexBase) -> tp.List[str]:
        if index.depth == 1:
            return [cls._dtype_to_str(index.dtype)] # type: ignore[attr-defined]
        return [cls._dtype_to_str(dt) for dt in index.dtypes.values] # type: ignore[attr-defined]

    @classmethod
    def to_dict(cls, f: TFrameAny) -> tp.Dict[str, tp.Any]:
        '''Generic routine to extract an JSON-encodable metadata bundle.
        '''
        # NOTE: need to store dtypes per index, per values; introduce new metadata label, use dtype.str to get string encoding

        md = {}
        md[cls.KEY_NAMES] = [
                JSONTranslator.encode_element(f._name),
                JSONTranslator.encode_element(f._index._name),
                JSONTranslator.encode_element(f._columns._name),
                ]

        md[cls.KEY_DTYPES] = [cls._dtype_to_str(dt) for dt in f.dtypes.values]
        md[cls.KEY_DTYPES_INDEX] = cls._index_to_dtype_str(f.index)
        md[cls.KEY_DTYPES_COLUMNS] = cls._index_to_dtype_str(f.columns)

        md[cls.KEY_TYPES] = [
                f._index.__class__.__name__,
                f._columns.__class__.__name__,
                ]

        for labels, key in (
                (f.index, cls.KEY_TYPES_INDEX),
                (f.columns, cls.KEY_TYPES_COLUMNS),
                ):
            if labels.depth > 1:
                md[key] = [cls.__name__ for cls in labels.index_types.values]

        md[cls.KEY_DEPTHS] = [
                f._blocks._index.shape[1], # count of columns
                f._index.depth,
                f._columns.depth]

        return md



    @staticmethod
    def _build_index_ctor(
            depth: int,
            cls_index: tp.Type['IndexBase'],
            name: TName,
            cls_components: tp.List[str] | None,
            dtypes: tp.List[str],
            ) -> TIndexCtor:

        from static_frame.core.index import Index
        from static_frame.core.index_datetime import IndexDatetime
        from static_frame.core.index_hierarchy import IndexHierarchy

        if depth == 1:
            if issubclass(cls_index, IndexDatetime):
                # do not provide dtype if a datetime64 index subclass
                return partial(cls_index, name=name)
            return partial(cls_index, name=name, dtype=dtypes[0]) # type: ignore

        assert cls_components is not None
        assert len(cls_components) == len(dtypes) # if depth > 1, must be provided

        index_constructors: tp.List[tp.Callable[..., Index[tp.Any]]] = []
        for cls_name, dt in zip(cls_components, dtypes):
            cls = ContainerMap.str_to_cls(cls_name)
            if issubclass(cls, IndexDatetime):
                index_constructors.append(cls)
            else:
                index_constructors.append(partial(cls, dtype=dt)) # type: ignore

        return partial(IndexHierarchy.from_labels,
                name=name,
                index_constructors=index_constructors,
                )



    @staticmethod
    def _get_cls(name: str, ctor_static: bool) -> tp.Type[IndexBase]:
        cls = ContainerMap.str_to_cls(name)
        # if containing Frame static does not match this class, update
        if ctor_static != cls.STATIC:
            if ctor_static:
                return cls._IMMUTABLE_CONSTRUCTOR #type: ignore
            return cls._MUTABLE_CONSTRUCTOR #type: ignore
        return cls #type: ignore

    @classmethod
    def from_dict_to_ctors(cls,
            md: tp.Dict[str, tp.Any],
            ctor_static: bool,
            ) -> tp.Tuple[TIndexCtor, TIndexCtor]:

        names = md[NPYLabel.KEY_NAMES]
        name_index = JSONTranslator.decode_element(names[1])
        name_columns = JSONTranslator.decode_element(names[2])

        types = md[NPYLabel.KEY_TYPES]
        cls_index: tp.Type[IndexBase] = cls._get_cls(types[0], True)
        cls_columns: tp.Type[IndexBase] = cls._get_cls(types[1], ctor_static)

        _, depth_index, depth_columns = md[NPYLabel.KEY_DEPTHS]

        index_ctor = cls._build_index_ctor(
                depth_index,
                cls_index,
                name_index,
                md.get(JSONMeta.KEY_TYPES_INDEX),
                md.get(JSONMeta.KEY_DTYPES_INDEX), # type: ignore
                )

        columns_ctor = cls._build_index_ctor(
                depth_columns,
                cls_columns,
                name_columns,
                md.get(JSONMeta.KEY_TYPES_COLUMNS),
                md.get(JSONMeta.KEY_DTYPES_COLUMNS), # type: ignore
                )

        return index_ctor, columns_ctor


