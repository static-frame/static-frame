
import typing as tp
import numpy as np
from numpy import char as npc

from static_frame.core.util import DT64_YEAR
from static_frame.core.util import DT64_MONTH
from static_frame.core.util import DT64_DAY
from static_frame.core.util import DTYPE_INT_DEFAULT
from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import DTYPE_DATETIME_KIND
from static_frame.core.util import EMPTY_TUPLE

if tp.TYPE_CHECKING:

    from static_frame.core.frame import Frame  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index import Index  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchy  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.series import Series  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.type_blocks import TypeBlocks  #pylint: disable = W0611 #pragma: no cover

# only ContainerOperand subclasses
TContainer = tp.TypeVar('TContainer', 'Index', 'IndexHierarchy', 'Series', 'Frame', 'TypeBlocks')

BlocksType = tp.Iterable[np.ndarray]
ToContainerType = tp.Callable[[np.ndarray], TContainer]


class InterfaceDatetime(tp.Generic[TContainer]):

    __slots__ = (
        '_blocks', # function that returns iterable of arrays
        '_blocks_to_container', # partialed function that will return a new container
        )

    DT64_EXCLUDE_YEAR = (DT64_YEAR,)
    DT64_EXCLUDE_YEAR_MONTH = (DT64_YEAR, DT64_MONTH)

    def __init__(self,
            blocks: BlocksType,
            func_to_container: ToContainerType[TContainer]
            ) -> None:
        self._blocks: BlocksType = blocks
        self._blocks_to_container: ToContainerType[TContainer] = func_to_container

    @staticmethod
    def _validate_dtype(
            dtype: np.dtype,
            exclude: tp.Iterable[np.dtype] = EMPTY_TUPLE,
            ) -> None:
        if ((dtype.kind == DTYPE_DATETIME_KIND
                or dtype == DTYPE_OBJECT)
                and dtype not in exclude
                ):
            return
        raise RuntimeError(f'invalid dtype ({dtype}) for date operation')

    @staticmethod
    def _array_from_dt_attr(
            array: np.ndarray,
            attr_name: str,
            dtype: np.dtype
            ) -> np.array:
        '''
        Handle element-wise attribute acesss on arrays of Python date/datetime objects.
        '''
        if array.ndim == 1:
            post = np.fromiter(
                    (getattr(d, attr_name) for d in array),
                    count=len(array),
                    dtype=dtype,
                    )
        else:
            post = np.empty(shape=array.shape, dtype=dtype)
            for iloc, e in np.ndenumerate(array):
                post[iloc] = getattr(e, attr_name)
        return post

    @staticmethod
    def _array_from_dt_method(
            array: np.ndarray,
            method_name: str,
            args: tp.Tuple[tp.Any, ...],
            dtype: np.dtype
            ) -> np.array:
        '''
        Handle element-wise method calling on arrays of Python date/datetime objects.
        '''
        if array.ndim == 1:
            post = np.fromiter(
                    (getattr(d, method_name)(*args) for d in array),
                    count=len(array),
                    dtype=dtype,
                    )
        else:
            post = np.empty(shape=array.shape, dtype=dtype)
            for iloc, e in np.ndenumerate(array):
                post[iloc] = getattr(e, method_name)(*args)

        return post

    #---------------------------------------------------------------------------

    @property
    def year(self) -> TContainer:
        def blocks() -> tp.Iterator[np.ndarray]:
            for block in self._blocks:
                self._validate_dtype(block.dtype)
                array = block.astype(DT64_YEAR)
                array.flags.writeable = False
                yield array

        return self._blocks_to_container(blocks())

    @property
    def month(self) -> TContainer:

        def blocks() -> tp.Iterator[np.ndarray]:
            for block in self._blocks:
                self._validate_dtype(block.dtype,
                        exclude=self.DT64_EXCLUDE_YEAR
                        )
                if block.dtype.kind == DTYPE_DATETIME_KIND:
                    array = block.astype(DT64_MONTH).astype(int) % 12 + 1
                else: # must be object type
                    array = self._array_from_dt_attr(
                            block,
                            'month',
                            DTYPE_INT_DEFAULT)
                array.flags.writeable = False
                yield array

        return self._blocks_to_container(blocks())

    @property
    def day(self) -> TContainer:
        pass

    def weekday(self) -> TContainer:

        def blocks() -> tp.Iterator[np.ndarray]:
            for block in self._blocks:
                self._validate_dtype(block.dtype,
                        exclude=self.DT64_EXCLUDE_YEAR_MONTH
                        )
                if block.dtype.kind == DTYPE_DATETIME_KIND:
                    if block.dtype != DT64_DAY:
                        # go to day first, then object
                        block = block.astype(DT64_DAY)
                    block = block.astype(DTYPE_OBJECT)
                # all object arrays by this point

                array = self._array_from_dt_method(
                        block,
                        'weekday',
                        EMPTY_TUPLE,
                        DTYPE_INT_DEFAULT
                        )
                array.flags.writeable = False
                yield array

        return self._blocks_to_container(blocks())