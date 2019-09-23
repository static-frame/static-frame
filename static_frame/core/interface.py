'''
Tools for documenting the SF interface.
'''

# from enum import Enum
from collections import namedtuple
import typing as tp

import numpy as np

from static_frame.core.frame import Frame
from static_frame.core.container import ContainerBase
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.index_datetime import IndexDate
from static_frame.core.index_datetime import IndexYearMonth
from static_frame.core.index_datetime import IndexYear
from static_frame.core.index_hierarchy import IndexHierarchy

# from static_frame.core.iter_node import IterNode
from static_frame.core.iter_node import IterNodeDelegate

# from static_frame.core.container import ContainerMeta
from static_frame.core.container import _UFUNC_BINARY_OPERATORS
from static_frame.core.container import _RIGHT_OPERATOR_MAP
from static_frame.core.container import _UFUNC_UNARY_OPERATORS

# from static_frame.core.util import InterfaceSelection1D # used on index.drop
from static_frame.core.selector_node import InterfaceSelection2D
from static_frame.core.selector_node import InterfaceAsType
from static_frame.core.selector_node import InterfaceGetItem


Interface = namedtuple('Interface', ('cls', 'group', 'name', 'doc'))


class InterfaceGroup:
    Attribute = 'attribute'
    OperatorBinary = 'operator_binary'
    OperatorUnary = 'operator_unary'
    Method = 'method'
    DictLike = 'dict_like'
    Iterator = 'iterator'
    Selector = 'selector'
    Constructor = 'constructor'
    Exporter = 'exporter'


class InterfaceSummary:

    DOC_CHARS = 40

    EXCLUDE_PRIVATE = {
        '__class__',
        '__class_getitem__',
        '__annotations__',
        '__doc__',
        '__delattr__',
        '__dir__',
        '__dict__',
        '__format__',
        '__getattribute__',
        '__hash__',
        '__init_sbclass__',
        '__lshift__',
        '__module__',
        '__init_subclass__',
        '__new__',
        '__setattr__',
        '__setstate__',
        '__slots__',
        '__subclasshook__',
        '__weakref__',
        '__reduce__',
        '__reduce_ex__',
        '__sizeof__',
        }

    DICT_LIKE = {'keys', 'values', 'items', '__contains__', '__iter__', '__reversed__'}
    ATTR_ITER_NODE = ('apply', 'apply_iter', 'apply_iter_items', 'apply_pool')

    GETITEM = '__getitem__'

    SELECTOR_ROOT = {'__getitem__', 'iloc', 'loc'}
    # SELECTOR_COMPOUND = {'drop', 'mask', 'masked_array', 'assign'}
    ATTR_SELECTOR_NODE = ('__getitem__', 'iloc', 'loc',)

    _CLS_TO_INSTANCE_CACHE: tp.Dict[int, int] = {}



    # astype is a normal function in Series, is a selector in Frame

    @classmethod
    def is_public(cls, field: str) -> bool:
        if field.startswith('_') and not field.startswith('__'):
            return False
        if field in cls.EXCLUDE_PRIVATE:
            return False
        return True

    @classmethod
    def scrub_doc(cls, doc: tp.Optional[str]) -> str:
        if not doc:
            return ''
        doc = doc.replace('``', '')
        # split and join removes contiguous whitespace
        return ' '.join(doc.split())[:cls.DOC_CHARS]


    @classmethod
    def get_instance(cls, target: tp.Type[ContainerBase]) -> ContainerBase:
        '''
        Get a sample instance from any ContainerBase; cache to only create one per life of process.
        '''
        if target not in cls._CLS_TO_INSTANCE_CACHE:
            if target is TypeBlocks:
                instance = target.from_blocks(np.array((0,)))
            elif issubclass(target, IndexHierarchy):
                instance = target.from_labels(((0,0),))
            elif target in (IndexYearMonth, IndexYear, IndexDate):
                instance = target(np.array((0,), dtype='datetime64[s]'))
            else:
                instance = target((0,))
            cls._CLS_TO_INSTANCE_CACHE[target] = instance
        return cls._CLS_TO_INSTANCE_CACHE[target]

    @classmethod
    def name_obj_iter(cls, target: tp.Any):
        instance = cls.get_instance(target=target)

        for name_attr in dir(target.__class__): # get metaclass
            if name_attr == 'interface':
                yield name_attr, getattr(target.__class__, name_attr)

        for name_attr in dir(target):
            if not cls.is_public(name_attr):
                continue
            yield name_attr, getattr(instance, name_attr)

    @classmethod
    def interrogate(cls, target: tp.Type[ContainerBase]) -> tp.Iterator[Interface]:

        for name_attr, obj in sorted(cls.name_obj_iter(target)):

            doc = ''
            if hasattr(obj, '__doc__'):
                doc = cls.scrub_doc(obj.__doc__)

            if hasattr(obj, '__name__'):
                name = obj.__name__
            else: # some attributes yield objects like arrays, Series, or Frame
                name = name_attr

            cls_name = target.__name__


            if name in cls.DICT_LIKE:
                yield Interface(cls_name, InterfaceGroup.DictLike, name, doc)
            elif name == 'astype':
                yield Interface(cls_name, InterfaceGroup.Method, name, doc)
                if isinstance(obj, InterfaceAsType): # an InterfaceAsType
                    display = f'{name}[]'
                    doc = cls.scrub_doc(getattr(InterfaceAsType, cls.GETITEM).__doc__)
                    yield Interface(cls_name, InterfaceGroup.Method, display, doc)

            elif name.startswith('from_') or name == '__init__':
                display = f'{name}()'
                yield Interface(cls_name, InterfaceGroup.Constructor, display, doc)
            elif name.startswith('to_'):
                display = f'{name}()'
                yield Interface(cls_name, InterfaceGroup.Exporter, display, doc)
            elif name.startswith('iter_'):
                display = f'{name}(axis)'
                yield Interface(cls_name, InterfaceGroup.Iterator, display, doc)
                for field in cls.ATTR_ITER_NODE:
                    display = f'{name}(axis).{field}()'
                    doc = cls.scrub_doc(getattr(IterNodeDelegate, field).__doc__)
                    yield Interface(cls_name, InterfaceGroup.Iterator, display, doc)

            elif isinstance(obj, InterfaceGetItem) or name == cls.GETITEM:
                display = f'{name}[]' if name != cls.GETITEM else '[]'
                yield Interface(cls_name, InterfaceGroup.Selector, display, doc)

            elif isinstance(obj, InterfaceSelection2D):
                for field in cls.ATTR_SELECTOR_NODE:
                    display = f'{name}.{field}[]' if name != cls.GETITEM else f'{name}[]'
                    doc = cls.scrub_doc(getattr(InterfaceSelection2D, field).__doc__)
                    yield Interface(cls_name, InterfaceGroup.Selector, display, doc)

            elif callable(obj):
                if name_attr in _UFUNC_UNARY_OPERATORS:
                    yield Interface(cls_name, InterfaceGroup.OperatorUnary, name, doc)
                elif name_attr in _UFUNC_BINARY_OPERATORS or name_attr in _RIGHT_OPERATOR_MAP:
                    yield Interface(cls_name, InterfaceGroup.OperatorBinary, name, doc)
                else:
                    yield Interface(cls_name, InterfaceGroup.Method, name, doc)
            else:
                yield Interface(cls_name, InterfaceGroup.Attribute, name, doc)

    @classmethod
    def to_frame(cls, target: tp.Any) -> Frame:
        f = Frame.from_records(cls.interrogate(target))
        f = f.sort_values(('cls', 'group', 'name'))
        f = f.set_index('name', drop=True)
        return f

