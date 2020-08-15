from enum import Enum
import typing as tp

import numpy as np

from static_frame.core.util import GetItemKeyType
from static_frame.core.util import UFunc
from static_frame.core.container import ContainerOperand
from static_frame.core.display import DisplayConfig
from static_frame.core.display import Display


if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.series import Series # pylint: disable=W0611 #pragma: no cover

FrameOrSeries = tp.TypeVar('FrameOrSeries', 'Frame', 'Series')


class BatchSelector(Enum):
    ILoc = 1
    Loc = 2 # do not use index class on container
    GetItem = 3
    BLoc = 4

class BatchProcessor(ContainerOperand):

    __slots__ = (
            '_key',
            '_selector',
            '_container_items',
            '_constructor',
            )

    def __init__(self,
            key: GetItemKeyType,
            selector: BatchSelector,
            container_items: tp.Iterable[tp.Tuple[tp.Hashable, 'Frame']],
            constructor: tp.Type[ContainerOperand],
            ):
        self._key = key
        self._selector = selector
        self._container_items = container_items
        self._constructor = constructor



    def _extract(self, frame: 'Frame') -> FrameOrSeries:
        if self._selector is BatchSelector.ILoc:
            return frame._extract_iloc(self._key)
        elif self._selector is BatchSelector.Loc:
            return frame._extract_loc(self._key)
        elif self._selector is BatchSelector.GetItem:
            return frame.__getitem__(self._key)
        elif self._selector is BatchSelector.BLoc:
            return frame._extract_bloc(self._key)
        raise NotImplementedError(f'{self._selector} not handled')


    def display(self,
            config: tp.Optional[DisplayConfig] = None
            ) -> Display:
        from static_frame.core.series import Series

        # realize generator
        if not hasattr(self._container_items, '__len__'):
            self._container_items = tuple(self._container_items)

        items = ((label, self._extract(f).shape) for label, f in self._container_items)

        return Series.from_items(items,
                name=self.__class__.__name__
                ).display(config=config)

    #---------------------------------------------------------------------------
    # operators

    def _ufunc_unary_operator(self,
            operator: UFunc
            ) -> np.ndarray:
        '''Cannot reduce dimensionality, so ignore labels.
        '''
        def gen():
            for _, frame in self._container_items:
                yield self._extract(frame)._ufunc_unary_operator(
                        operator=operator,
                        )
        return self._constructor.from_concat(gen())


    def _ufunc_binary_operator(self, *,
            operator: UFunc,
            other: tp.Any,
            ) -> np.ndarray:
        '''
        Binary operators applied to an index always return an NP array. This deviates from Pandas, where some operations (multipling an int index by an int) result in a new Index, while other operations result in a np.array (using == on two Index).
        '''

        # # NOTE: might use TypeBlocks._ufunc_binary_operator
        # values = self._blocks.values

        # other_is_array = False
        # if isinstance(other, Index):
        #     # if this is a 1D index, must rotate labels before using an operator
        #     other = other.values.reshape((len(other), 1)) # operate on labels to labels
        #     other_is_array = True
        # elif isinstance(other, IndexHierarchy):
        #     # already 2D
        #     other = other.values # operate on labels to labels
        #     other_is_array = True
        # elif isinstance(other, np.ndarray):
        #     other_is_array = True

        # if operator.__name__ == 'matmul':
        #     return matmul(values, other)
        # elif operator.__name__ == 'rmatmul':
        #     return matmul(other, values)

        # return apply_binary_operator(
        #         values=values,
        #         other=other,
        #         other_is_array=other_is_array,
        #         operator=operator,
        #         )

    def _ufunc_axis_skipna(self, *,
            axis: int,
            skipna: bool,
            ufunc: UFunc,
            ufunc_skipna: UFunc,
            composable: bool,
            dtypes: tp.Tuple[np.dtype, ...],
            size_one_unity: bool
            ) -> 'Frame':

        from static_frame.core.frame import Frame
        from static_frame.core.series import Series
        labels = []
        def gen():
            for label, frame in self._container_items:
                labels.append(label)
                extracted = self._extract(frame)
                part = extracted._ufunc_axis_skipna(axis=axis,
                        skipna=skipna,
                        ufunc=ufunc,
                        ufunc_skipna=ufunc_skipna,
                        composable=composable,
                        dtypes=dtypes,
                        size_one_unity=size_one_unity,
                        )
                # part might be an element
                if not isinstance(part, (Frame, Series)):
                    # promote to a Series to permit concatenation
                    yield Series.from_element(part, index=(extracted.name,))
                else:
                    yield part

        return self._constructor.from_concat(gen(), index=labels)

    def _ufunc_shape_skipna(self, *,
            axis: int,
            skipna: bool,
            ufunc: UFunc,
            ufunc_skipna: UFunc,
            composable: bool,
            dtypes: tp.Tuple[np.dtype, ...],
            size_one_unity: bool
            ) -> 'Frame':
        pass



    # def sum(self):
    #     return 'foo'
