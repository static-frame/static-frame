import typing as tp

import numpy as np

from static_frame.core.display import Display
from static_frame.core.display_config import DisplayConfig
from static_frame.core.doc_str import doc_inject

from static_frame.core.hloc import HLoc

from static_frame.core.index import Index
from static_frame.core.index import IndexGO
from static_frame.core.index_base import IndexBase
from static_frame.core.index_level import IndexLevel
from static_frame.core.index_level import IndexLevelGO
from static_frame.core.index_level import TreeNodeT
from static_frame.core.index_auto import RelabelInput

from static_frame.core.node_dt import InterfaceDatetime
from static_frame.core.node_iter import IterNodeDepthLevel
from static_frame.core.node_selector import InterfaceAsType
from static_frame.core.node_selector import InterfaceGetItem
from static_frame.core.node_selector import TContainer
from static_frame.core.node_str import InterfaceString
from static_frame.core.node_transpose import InterfaceTranspose
from static_frame.core.node_re import InterfaceRe
from static_frame.core.type_blocks import TypeBlocks

from static_frame.core.util import DEFAULT_SORT_KIND
from static_frame.core.util import DepthLevelSpecifier
from static_frame.core.util import DtypeSpecifier
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import IndexConstructor
from static_frame.core.util import IndexConstructors
from static_frame.core.util import IndexInitializer
from static_frame.core.util import intersect2d
from static_frame.core.util import NAME_DEFAULT
from static_frame.core.util import NameType
from static_frame.core.util import setdiff2d
from static_frame.core.util import UFunc
from static_frame.core.util import union2d
from static_frame.core.util import CONTINUATION_TOKEN_INACTIVE
from static_frame.core.util import BoolOrBools

from static_frame.core.style_config import StyleConfig


if tp.TYPE_CHECKING:
    from pandas import DataFrame #pylint: disable=W0611 #pragma: no cover
    from static_frame.core.frame import Frame #pylint: disable=W0611,C0412 #pragma: no cover
    from static_frame.core.frame import FrameGO #pylint: disable=W0611,C0412 #pragma: no cover
    from static_frame.core.series import Series #pylint: disable=W0611,C0412 #pragma: no cover

IH = tp.TypeVar('IH', bound='IndexHierarchy')


#-------------------------------------------------------------------------------
class IndexHierarchy(IndexBase):
    '''A hierarchy of :obj:`Index` objects, defined as a strict tree of uniform depth across all branches.'''

    __slots__ = (
            '_levels',
            '_blocks',
            '_recache',
            '_name',
            )
    _levels: IndexLevel
    _blocks: TypeBlocks # should be tp.Optional[TypeBlocks] but many typing changes required
    _recache: bool
    _name: NameType

    # Temporary type overrides, until indices are generic.
    # __getitem__: tp.Callable[['IndexHierarchy', tp.Hashable], tp.Tuple[tp.Hashable, ...]]

    # _IMMUTABLE_CONSTRUCTOR is None from IndexBase
    # _MUTABLE_CONSTRUCTOR will be defined after IndexHierarhcyGO defined

    _INDEX_CONSTRUCTOR = Index
    _LEVEL_CONSTRUCTOR = IndexLevel

    _UFUNC_UNION = union2d
    _UFUNC_INTERSECTION = intersect2d
    _UFUNC_DIFFERENCE = setdiff2d
    _NDIM: int = 2

    #---------------------------------------------------------------------------
    # constructors

    @classmethod
    def from_product(cls: tp.Type[IH],
            *levels: IndexInitializer,
            name: NameType = None,
            index_constructors: tp.Optional[IndexConstructors] = None,
            ) -> IH:
        '''
        Given groups of iterables, return an ``IndexHierarchy`` made of the product of a values in those groups, where the first group is the top-most hierarchy.

        Args:
            *levels: index initializers (or Index instances) for each level
            name:
            index_consructors:

        Returns:
            :obj:`static_frame.IndexHierarchy`

        '''
        raise NotImplementedError()

    @classmethod
    def from_tree(cls: tp.Type[IH],
            tree: TreeNodeT,
            *,
            name: NameType = None,
            index_constructors: tp.Optional[IndexConstructors] = None,
            ) -> IH:
        '''
        Convert into a ``IndexHierarchy`` a dictionary defining keys to either iterables or nested dictionaries of the same.

        Returns:
            :obj:`static_frame.IndexHierarchy`
        '''
        raise NotImplementedError()

    @classmethod
    def from_labels(cls: tp.Type[IH],
            labels: tp.Iterable[tp.Sequence[tp.Hashable]],
            *,
            name: NameType = None,
            reorder_for_hierarchy: bool = False,
            index_constructors: tp.Optional[IndexConstructors] = None,
            depth_reference: tp.Optional[int] = None,
            continuation_token: tp.Union[tp.Hashable, None] = CONTINUATION_TOKEN_INACTIVE
            ) -> IH:
        '''
        Construct an ``IndexHierarhcy`` from an iterable of labels, where each label is tuple defining the component labels for all hierarchies.

        Args:
            labels: an iterator or generator of tuples.
            *,
            name:
            reorder_for_hierarchy: reorder the labels to produce a hierarchable Index, assuming hierarchability is possible.
            index_constructors:
            depth_reference:
            continuation_token: a Hashable that will be used as a token to identify when a value in a label should use the previously encountered value at the same depth.

        Returns:
            :obj:`static_frame.IndexHierarchy`
        '''
        raise NotImplementedError()

    @classmethod
    def from_index_items(cls: tp.Type[IH],
            items: tp.Iterable[tp.Tuple[tp.Hashable, Index]],
            *,
            index_constructor: tp.Optional[IndexConstructor] = None,
            name: NameType = None,
            ) -> IH:
        '''
        Given an iterable of pairs of label, :obj:`Index`, produce an :obj:`IndexHierarchy` where the labels are depth 0, the indices are depth 1.

        Args:
            items: iterable of pairs of label, :obj:`Index`.
            index_constructor: Optionally provide index constructor for outermost index.
        '''
        raise NotImplementedError()

    @classmethod
    def from_labels_delimited(cls: tp.Type[IH],
            labels: tp.Iterable[str],
            *,
            delimiter: str = ' ',
            name: NameType = None,
            index_constructors: tp.Optional[IndexConstructors] = None,
            ) -> IH:
        '''
        Construct an :obj:`IndexHierarchy` from an iterable of labels, where each label is string defining the component labels for all hierarchies using a string delimiter. All components after splitting the string by the delimited will be literal evaled to produce proper types; thus, strings must be quoted.

        Args:
            labels: an iterator or generator of tuples.

        Returns:
            :obj:`static_frame.IndexHierarchy`
        '''
        raise NotImplementedError()

    @classmethod
    def from_names(cls: tp.Type[IH],
            names: tp.Iterable[tp.Hashable]
            ) -> IH:
        '''
        Construct a zero-length :obj:`IndexHierarchy` from an iterable of ``names``, where the length of ``names`` defines the zero-length depth.

        Args:
            names: Iterable of hashable names per depth.
        '''
        raise NotImplementedError()

    @classmethod
    def _from_type_blocks(cls: tp.Type[IH],
            blocks: TypeBlocks,
            *,
            name: NameType = None,
            index_constructors: tp.Optional[IndexConstructors] = None,
            own_blocks: bool = False,
            ) -> IH:
        '''
        Construct an :obj:`IndexHierarchy` from a :obj:`TypeBlocks` instance.

        Args:
            blocks: a TypeBlocks instance

        Returns:
            :obj:`IndexHierarchy`
        '''
        raise NotImplementedError()

    # NOTE: could have a _from_fields (or similar) that takes a sequence of column iterables/arrays

    #---------------------------------------------------------------------------
    def __init__(self,
            levels: tp.Union[IndexLevel, 'IndexHierarchy'],
            *,
            name: NameType = NAME_DEFAULT,
            blocks: tp.Optional[TypeBlocks] = None,
            own_blocks: bool = False,
            ):
        '''
        Initializer.

        Args:
            levels: :obj:`IndexLevels` instance, or, optionally, an :obj`IndexHierarchy` to be used to construct a new :obj`IndexHierarchy`.
        '''
        raise NotImplementedError()

    #---------------------------------------------------------------------------
    def __deepcopy__(self: IH, memo: tp.Dict[int, tp.Any]) -> IH:
        raise NotImplementedError()

    def __copy__(self: IH) -> IH:
        '''
        Return a shallow copy of this IndexHierarchy.
        '''
        raise NotImplementedError()

    def copy(self: IH) -> IH:
        '''
        Return a shallow copy of this IndexHierarchy.
        '''
        return self.__copy__()

    #---------------------------------------------------------------------------
    # name interface

    def rename(self: IH, name: NameType) -> IH:
        '''
        Return a new Frame with an updated name attribute.
        '''
        raise NotImplementedError()

    #---------------------------------------------------------------------------
    # interfaces

    @property
    def loc(self) -> InterfaceGetItem['IndexHierarchy']:
        raise NotImplementedError()

    @property
    def iloc(self) -> InterfaceGetItem['IndexHierarchy']:
        raise NotImplementedError()

    def _iter_label(self,
            depth_level: tp.Optional[DepthLevelSpecifier] = None,
            ) -> tp.Iterator[tp.Hashable]:
        raise NotImplementedError()

    def _iter_label_items(self,
            depth_level: tp.Optional[DepthLevelSpecifier] = None,
            ) -> tp.Iterator[tp.Tuple[int, tp.Hashable]]:
        '''This function is not directly called in iter_label or related routines, fulfills the expectations of the IterNodeDepthLevel interface.
        '''
        raise NotImplementedError()

    @property
    def iter_label(self) -> IterNodeDepthLevel[tp.Any]:
        raise NotImplementedError()

    # NOTE: Index implements drop property

    @property #type: ignore
    @doc_inject(select='astype')
    def astype(self) -> InterfaceAsType[TContainer]:
        '''
        Retype one or more depths. Can be used as as function to retype the entire ``IndexHierarchy``; alternatively, a ``__getitem__`` interface permits retyping selected depths.

        Args:
            {dtype}
        '''
        raise NotImplementedError()

    #---------------------------------------------------------------------------
    @property
    def via_str(self) -> InterfaceString[np.ndarray]:
        '''
        Interface for applying string methods to elements in this container.
        '''
        raise NotImplementedError()

    @property
    def via_dt(self) -> InterfaceDatetime[np.ndarray]:
        '''
        Interface for applying datetime properties and methods to elements in this container.
        '''
        raise NotImplementedError()

    @property
    def via_T(self) -> InterfaceTranspose['IndexHierarchy']:
        '''
        Interface for using binary operators with one-dimensional sequences, where the opperand is applied column-wise.
        '''
        raise NotImplementedError()

    def via_re(self,
            pattern: str,
            flags: int = 0,
            ) -> InterfaceRe[np.ndarray]:
        '''
        Interface for applying regular expressions to elements in this container.
        '''
        raise NotImplementedError()

    #---------------------------------------------------------------------------

    def _update_array_cache(self) -> None:
        raise NotImplementedError()

    #---------------------------------------------------------------------------

    @property # type: ignore
    @doc_inject()
    def mloc(self) -> int:
        '''{doc_int}
        '''
        raise NotImplementedError()

    @property
    def dtypes(self) -> 'Series':
        '''
        Return a Series of dytpes for each index depth.

        Returns:
            :obj:`static_frame.Series`
        '''
        raise NotImplementedError()

    @property
    def shape(self) -> tp.Tuple[int, ...]:
        '''
        Return a tuple describing the shape of the underlying NumPy array.

        Returns:
            :obj:`tp.Tuple[int]`
        '''
        raise NotImplementedError()

    @property
    def ndim(self) -> int:
        '''
        Return the number of dimensions.

        Returns:
            :obj:`int`
        '''
        return self._NDIM

    @property
    def size(self) -> int:
        '''
        Return the size of the underlying NumPy array.

        Returns:
            :obj:`int`
        '''
        raise NotImplementedError()

    @property
    def nbytes(self) -> int:
        '''
        Return the total bytes of the underlying NumPy array.

        Returns:
            :obj:`int`
        '''
        raise NotImplementedError()

    #---------------------------------------------------------------------------

    def __len__(self) -> int:
        raise NotImplementedError()

    @doc_inject()
    def display(self,
            config: tp.Optional[DisplayConfig] = None,
            *,
            style_config: tp.Optional[StyleConfig] = None,
            ) -> Display:
        '''{doc}

        Args:
            {config}
        '''
        raise NotImplementedError()

    #---------------------------------------------------------------------------
    # set operations

    def _ufunc_set(self,
            func: tp.Callable[[np.ndarray, np.ndarray, bool], np.ndarray],
            other: tp.Union['IndexBase', tp.Iterable[tp.Hashable]]
            ) -> 'IndexHierarchy':
        '''
        Utility function for preparing and collecting values for Indices to produce a new Index.
        '''
        raise NotImplementedError()

    #---------------------------------------------------------------------------
    def _drop_iloc(self, key: GetItemKeyType) -> 'IndexHierarchy':
        '''Create a new index after removing the values specified by the loc key.
        '''
        raise NotImplementedError()

    def _drop_loc(self, key: GetItemKeyType) -> 'IndexHierarchy':
        '''Create a new index after removing the values specified by the loc key.
        '''
        raise NotImplementedError()

    #---------------------------------------------------------------------------

    @property #type: ignore
    @doc_inject(selector='values_2d', class_name='IndexHierarchy')
    def values(self) -> np.ndarray:
        '''
        {}
        '''
        raise NotImplementedError()

    @property
    def positions(self) -> np.ndarray:
        '''Return the immutable positions array.
        '''
        raise NotImplementedError()

    @property
    def depth(self) -> int: #type: ignore
        raise NotImplementedError()

    def values_at_depth(self,
            depth_level: DepthLevelSpecifier = 0
            ) -> np.ndarray:
        '''
        Return an NP array for the ``depth_level`` specified.

        Args:
            depth_level: a single depth level, or iterable depth of depth levels.
        '''
        raise NotImplementedError()

    @doc_inject()
    def label_widths_at_depth(self,
            depth_level: DepthLevelSpecifier = 0
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, int]]:
        '''{}'''
        raise NotImplementedError()

    @property
    def index_types(self) -> 'Series':
        '''
        Return a Series of Index classes for each index depth.

        Returns:
            :obj:`Series`
        '''
        raise NotImplementedError()

    #---------------------------------------------------------------------------
    def relabel(self, mapper: RelabelInput) -> 'IndexHierarchy':
        '''
        Return a new IndexHierarchy with labels replaced by the callable or mapping; order will be retained. If a mapping is used, the mapping should map tuple representation of labels, and need not map all origin keys.
        '''
        raise NotImplementedError()

    def relabel_at_depth(self,
            mapper: RelabelInput,
            depth_level: DepthLevelSpecifier = 0
            ) -> "IndexHierarchy":
        '''
        Return a new :obj:`IndexHierarchy` after applying `mapper` to a level or each individual level specified by `depth_level`.

        `mapper` can be a callable, mapping, or iterable.
            - If a callable, it must accept a single value, and return a single value.
            - If a mapping, it must map a single value to a single value.
            - If a iterable, it must be the same length as `self`.

        This call:

        >>> index.relabel_at_depth(mapper, depth_level=[0, 1, 2])

        is equivalent to:

        >>> for level in [0, 1, 2]:
        >>>     index = index.relabel_at_depth(mapper, depth_level=level)

        albeit more efficient.
        '''
        raise NotImplementedError()

    def rehierarch(self: IH,
            depth_map: tp.Sequence[int]
            ) -> IH:
        '''
        Return a new :obj:`IndexHierarchy` that conforms to the new depth assignments given be `depth_map`.
        '''
        raise NotImplementedError()

    #---------------------------------------------------------------------------

    def _loc_to_iloc(self,
            key: tp.Union[GetItemKeyType, HLoc]
            ) -> GetItemKeyType:
        '''
        Given iterable of GetItemKeyTypes, apply to each level of levels.
        '''
        raise NotImplementedError()

    def loc_to_iloc(self,
            key: tp.Union[GetItemKeyType, HLoc]
            ) -> GetItemKeyType:
        '''Given a label (loc) style key (either a label, a list of labels, a slice, or a Boolean selection), return the index position (iloc) style key. Keys that are not found will raise a KeyError or a sf.LocInvalid error.

        Args:
            key: a label key.
        '''
        raise NotImplementedError()

    def _extract_iloc(self,
            key: GetItemKeyType,
            ) -> tp.Union['IndexHierarchy', tp.Tuple[tp.Hashable]]:
        '''Extract a new index given an iloc key
        '''
        raise NotImplementedError()

    def _extract_loc(self,
            key: GetItemKeyType
            ) -> tp.Union['IndexHierarchy', tp.Tuple[tp.Hashable]]:
        raise NotImplementedError()

    def __getitem__(self, #pylint: disable=E0102
            key: GetItemKeyType
            ) -> tp.Union['IndexHierarchy', tp.Tuple[tp.Hashable]]:
        '''Extract a new index given an iloc key.
        '''
        raise NotImplementedError()

    #---------------------------------------------------------------------------
    def _extract_getitem_astype(self, key: GetItemKeyType) -> 'IndexHierarchyAsType':
        '''Given an iloc key (using integer positions for columns) return a configured IndexHierarchyAsType instance.
        '''
        raise NotImplementedError()

    #---------------------------------------------------------------------------
    # operators

    def _ufunc_unary_operator(self,
            operator: UFunc
            ) -> np.ndarray:
        '''Always return an NP array.
        '''
        raise NotImplementedError()

    def _ufunc_binary_operator(self, *,
            operator: UFunc,
            other: tp.Any,
            axis: int = 0,
            fill_value: object = np.nan,
            ) -> np.ndarray:
        '''
        Binary operators applied to an index always return an NP array. This deviates from Pandas, where some operations (multiplying an int index by an int) result in a new Index, while other operations result in a np.array (using == on two Index).
        '''
        raise NotImplementedError()

    def _ufunc_axis_skipna(self, *,
            axis: int,
            skipna: bool,
            ufunc: UFunc,
            ufunc_skipna: UFunc,
            composable: bool,
            dtypes: tp.Tuple[np.dtype, ...],
            size_one_unity: bool
            ) -> np.ndarray:
        '''
        Returns:
            immutable NumPy array.
        '''
        raise NotImplementedError()

    # _ufunc_shape_skipna defined in IndexBase

    #---------------------------------------------------------------------------
    # dictionary-like interface

    # NOTE: we intentionally exclude keys(), items(), and get() from Index classes, as they return inconsistent result when thought of as a dictionary

    def __iter__(self) -> tp.Iterator[tp.Tuple[tp.Hashable, ...]]:
        '''Iterate over labels.
        '''
        raise NotImplementedError()

    def __reversed__(self) -> tp.Iterator[tp.Tuple[tp.Hashable, ...]]:
        '''
        Returns a reverse iterator on the index labels.
        '''
        raise NotImplementedError()

    def __contains__(self, #type: ignore
            value: tp.Tuple[tp.Hashable]
            ) -> bool:
        '''Determine if a leaf loc is contained in this Index.
        '''
        raise NotImplementedError()

    #---------------------------------------------------------------------------
    # utility functions

    def unique(self,
            depth_level: DepthLevelSpecifier = 0
            ) -> np.ndarray:
        '''
        Return a NumPy array of unique values.

        Args:
            depth_level: Specify a single depth or multiple depths in an iterable.

        Returns:
            :obj:`numpy.ndarray`
        '''
        raise NotImplementedError()

    @doc_inject()
    def equals(self,
            other: tp.Any,
            *,
            compare_name: bool = False,
            compare_dtype: bool = False,
            compare_class: bool = False,
            skipna: bool = True,
            ) -> bool:
        '''
        {doc}

        Args:
            {compare_name}
            {compare_dtype}
            {compare_class}
            {skipna}
        '''
        raise NotImplementedError()

    @doc_inject(selector='sort')
    def sort(self: IH,
            *,
            ascending: BoolOrBools = True,
            kind: str = DEFAULT_SORT_KIND,
            key: tp.Optional[tp.Callable[['IndexHierarchy'], tp.Union[np.ndarray, 'IndexHierarchy']]] = None,
            ) -> IH:
        '''Return a new Index with the labels sorted.

        Args:
            {ascendings}
            {kind}
            {key}
        '''
        raise NotImplementedError()

    def isin(self, other: tp.Iterable[tp.Iterable[tp.Hashable]]) -> np.ndarray:
        '''
        Return a Boolean array showing True where one or more of the passed in iterable of labels is found in the index.
        '''
        raise NotImplementedError()

    def roll(self, shift: int) -> 'IndexHierarchy':
        '''Return an :obj:`IndexHierarchy` with values rotated forward and wrapped around (with a positive shift) or backward and wrapped around (with a negative shift).
        '''
        raise NotImplementedError()

    @doc_inject(selector='fillna')
    def fillna(self, value: tp.Any) -> 'IndexHierarchy':
        '''Return an :obj:`IndexHierarchy` after replacing NA (NaN or None) with the supplied value.

        Args:
            {value}
        '''
        raise NotImplementedError()

    def _sample_and_key(self,
            count: int = 1,
            *,
            seed: tp.Optional[int] = None,
            ) -> tp.Tuple['IndexHierarchy', np.ndarray]:
        raise NotImplementedError()

    @doc_inject(selector='searchsorted', label_type='iloc (integer)')
    def iloc_searchsorted(self,
            values: tp.Any,
            *,
            side_left: bool = True,
            ) -> tp.Union[tp.Hashable, tp.Iterable[tp.Hashable]]:
        '''
        {doc}

        Args:
            {values}
            {side_left}
        '''
        raise NotImplementedError()

    @doc_inject(selector='searchsorted', label_type='loc (label)')
    def loc_searchsorted(self,
            values: tp.Any,
            *,
            side_left: bool = True,
            fill_value: tp.Any = np.nan,
            ) -> tp.Union[tp.Hashable, tp.Iterable[tp.Hashable]]:
        '''
        {doc}

        Args:
            {values}
            {side_left}
            {fill_value}
        '''
        raise NotImplementedError()

    #---------------------------------------------------------------------------
    # export

    def _to_frame(self,
            constructor: tp.Type['Frame']
            ) -> 'Frame':

        raise NotImplementedError()

    def to_frame(self) -> 'Frame':
        '''
        Return :obj:`Frame` version of this :obj:`IndexHiearchy`.
        '''
        from static_frame import Frame
        return self._to_frame(Frame)

    def to_frame_go(self) -> 'FrameGO':
        '''
        Return a :obj:`FrameGO` version of this :obj:`IndexHierarchy`.
        '''
        from static_frame import FrameGO
        return self._to_frame(FrameGO) #type: ignore

    def to_pandas(self) -> 'DataFrame':
        '''Return a Pandas MultiIndex.
        '''
        raise NotImplementedError()

    def to_tree(self) -> TreeNodeT:
        '''Returns the tree representation of an IndexHierarchy
        '''
        raise NotImplementedError()

    def flat(self) -> IndexBase:
        '''Return a flat, one-dimensional index of tuples for each level.
        '''
        raise NotImplementedError()

    def level_add(self: IH,
            level: tp.Hashable,
            *,
            index_constructor: IndexConstructor = None,
            ) -> IH:
        '''Return an IndexHierarchy with a new root (outer) level added.
        '''
        raise NotImplementedError()

    def level_drop(self,
            count: int = 1,
            ) -> tp.Union[Index, 'IndexHierarchy']:
        '''Return an IndexHierarchy with one or more leaf levels removed. This might change the size of the resulting index if the resulting levels are not unique.

        Args:
            count: A positive value is the number of depths to remove from the root (outer) side of the hierarchy; a negative value is the number of depths to remove from the leaf (inner) side of the hierarchy.
        '''
        raise NotImplementedError()


class IndexHierarchyGO(IndexHierarchy):
    '''
    A hierarchy of :obj:`static_frame.Index` objects that permits mutation only in the addition of new hierarchies or labels.
    '''

    STATIC = False

    _IMMUTABLE_CONSTRUCTOR = IndexHierarchy
    _LEVEL_CONSTRUCTOR = IndexLevelGO
    _INDEX_CONSTRUCTOR = IndexGO

    __slots__ = (
            '_levels', # IndexLevel
            '_blocks',
            '_recache',
            '_name'
            )

    _levels: IndexLevelGO

    def append(self, value: tp.Sequence[tp.Hashable]) -> None:
        '''
        Append a single label to this index.
        '''
        raise NotImplementedError()

    def extend(self, other: IndexHierarchy) -> None:
        '''
        Extend this IndexHiearchy in-place
        '''
        raise NotImplementedError()

    def __copy__(self: IH) -> IH:
        '''
        Return a shallow copy of this IndexHierarchy.
        '''
        raise NotImplementedError()

# update class attr on Index after class initialization
IndexHierarchy._MUTABLE_CONSTRUCTOR = IndexHierarchyGO


class IndexHierarchyAsType:

    __slots__ = ('container', 'key',)

    def __init__(self,
            container: IndexHierarchy,
            key: GetItemKeyType
            ) -> None:
        self.container = container
        self.key = key

    def __call__(self, dtype: DtypeSpecifier) -> IndexHierarchy:
        raise NotImplementedError()