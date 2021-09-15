import typing as tp

import numpy as np


from static_frame.core.util import array2d_to_tuples
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import intersect1d
from static_frame.core.util import intersect2d

if tp.TYPE_CHECKING:

    from static_frame.core.index import Index  # pylint: disable = W0611 #pragma: no cover


class IndexCorrespondence:
    '''
    All iloc data necessary for reindexing.
    '''

    __slots__ = (
            'has_common',
            'is_subset',
            'iloc_src',
            'iloc_dst',
            'size',
            )

    has_common: bool
    is_subset: bool
    iloc_src: GetItemKeyType
    iloc_dst: GetItemKeyType
    size: int

    @classmethod
    def from_correspondence(cls,
            src_index: 'Index',
            dst_index: 'Index') -> 'IndexCorrespondence':
        '''
        Return an IndexCorrespondence instance from the correspondence of two Index or IndexHierarchy objects.

        This is called in all reindexing operations to get the iloc postions for remapping values.
        '''
        mixed_depth = False
        if src_index.depth == dst_index.depth:
            depth = src_index.depth
        else:
            # if dimensions are mixed, the only way there can be a match is if the 1D index is of object type (so it can hold a tuple); otherwise, there can be no matches;
            if src_index.depth == 1 and src_index.values.dtype.kind == 'O':
                depth = dst_index.depth
                mixed_depth = True
            elif dst_index.depth == 1 and dst_index.values.dtype.kind == 'O':
                depth = src_index.depth
                mixed_depth = True
            else:
                depth = 0

        # need to use lower level array methods go get intersection, rather than Index methods, as need arrays, not Index objects
        if depth == 1:
            # NOTE: this can fail in some cases: comparing two object arrays with NaNs and strings.
            common_labels = intersect1d(
                    src_index.values,
                    dst_index.values,
                    assume_unique=True
                    )
            has_common = len(common_labels) > 0
            assert not mixed_depth
        elif depth > 1:
            # NOTE: calling .values will convert dt64 to objects
            common_labels = intersect2d(
                    src_index.values,
                    dst_index.values,
                    assume_unique=True
                    )
            if mixed_depth:
                # when mixed, on the 1D index we have to use loc_to_iloc with tuples
                common_labels = list(array2d_to_tuples(common_labels))
            has_common = len(common_labels) > 0
        else:
            has_common = False

        size = len(dst_index.values)
        # either a reordering or a subset
        if has_common:
            if len(common_labels) == len(dst_index):
                # use new index to retain order
                values_dst = dst_index.values
                if values_dst.dtype == DTYPE_BOOL:
                    # if the index values are a Boolean array, loc_to_iloc will try to do a Boolean selection, which is incorrect. Using a list avoids this problem.
                    iloc_src = src_index._loc_to_iloc(values_dst.tolist())
                else:
                    iloc_src = src_index._loc_to_iloc(values_dst)
                iloc_dst = np.arange(size)
                return cls(has_common=has_common,
                        is_subset=True,
                        iloc_src=iloc_src,
                        iloc_dst=iloc_dst,
                        size=size
                        )

            # these will be equal sized
            # NOTE: if this fails, it means that our common labels are not common, likely due to a type conversions
            iloc_src = src_index._loc_to_iloc(common_labels)
            iloc_dst = dst_index._loc_to_iloc(common_labels)

            return cls(has_common=has_common,
                    is_subset=False,
                    iloc_src=iloc_src,
                    iloc_dst=iloc_dst,
                    size=size)

        # if no common values, nothing to transfer from src to dst
        return cls(has_common=has_common,
                is_subset=False,
                iloc_src=None,
                iloc_dst=None,
                size=size)


    def __init__(self,
            has_common: bool,
            is_subset: bool,
            iloc_src: GetItemKeyType,
            iloc_dst: GetItemKeyType,
            size: int) -> None:
        '''
        Args:
            has_common: True if any of the indices align
            is_subset: True if the destination is a reordering or subset
            iloc_src: An iterable of iloc values to be taken from the source
            iloc_dst: An iterable of iloc values to be written to
            size: The size of the destination.
        '''
        self.has_common = has_common
        self.is_subset = is_subset
        self.iloc_src = iloc_src
        self.iloc_dst = iloc_dst
        self.size = size

    def iloc_src_fancy(self) -> tp.List[tp.List[int]]:
        '''
        Convert an iloc iterable of integers into one that is combitable with fancy indexing.
        '''
        return [[x] for x in self.iloc_src] #type: ignore
