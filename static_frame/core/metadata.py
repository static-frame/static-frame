from __future__ import annotations

import typing_extensions as tp

from static_frame.core.util import JSONTranslator

if tp.TYPE_CHECKING:
    from static_frame.core.generic_aliases import TFrameAny

class NPYLabel:
    KEY_NAMES = '__names__'
    KEY_TYPES = '__types__'
    KEY_DEPTHS = '__depths__'
    KEY_TYPES_INDEX = '__types_index__'
    KEY_TYPES_COLUMNS = '__types_columns__'
    FILE_TEMPLATE_VALUES_INDEX = '__values_index_{}__.npy'
    FILE_TEMPLATE_VALUES_COLUMNS = '__values_columns_{}__.npy'
    FILE_TEMPLATE_BLOCKS = '__blocks_{}__.npy'


class JSONMeta:
    '''Metadata for JSON encodings.
    '''
    KEY_NAMES = NPYLabel.KEY_NAMES
    KEY_TYPES = NPYLabel.KEY_TYPES
    KEY_DEPTHS = NPYLabel.KEY_DEPTHS
    KEY_TYPES_INDEX = NPYLabel.KEY_TYPES_INDEX
    KEY_TYPES_COLUMNS = NPYLabel.KEY_TYPES_COLUMNS

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
                len(f._blocks._blocks),
                f._index.depth,
                f._columns.depth]

        return md
