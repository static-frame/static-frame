'''
Tools for documenting the SF interface.
'''

# from enum import Enum
from collections import namedtuple
import typing as tp

from static_frame.core.series import Series
from static_frame.core.frame import Frame
# from static_frame.core.iter_node import IterNode
from static_frame.core.iter_node import IterNodeDelegate

from static_frame.core.display import DisplayConfigs
# from static_frame.core.container import ContainerMeta
from static_frame.core.container import _UFUNC_BINARY_OPERATORS
from static_frame.core.container import _RIGHT_OPERATOR_MAP
from static_frame.core.container import _UFUNC_UNARY_OPERATORS

# from static_frame.core.util import InterfaceSelection1D # used on index.drop
from static_frame.core.util import InterfaceSelection2D
from static_frame.core.util import InterfaceAsType


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

    SELECTOR_ROOT = {'__getitem__', 'iloc', 'loc'}
    SELECTOR_COMPOUND = {'drop', 'mask', 'masked_array', 'assign'}
    ATTR_SELECTOR_NODE = ('__getitem__', 'iloc', 'loc',)

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
        return doc.strip().replace('\n', ' ')[:cls.DOC_CHARS]

    @classmethod
    def interrogate(cls, target: tp.Any) -> tp.Iterator[Interface]:

        for name_attr in sorted(dir(target)):
            if not cls.is_public(name_attr):
                continue

            # this gets object off the class, not an instance
            obj = getattr(target, name_attr)

            doc = ''
            if hasattr(obj, '__doc__'):
                doc = cls.scrub_doc(obj.__doc__)

            if hasattr(obj, '__name__'):
                name = obj.__name__
            else:
                name = name_attr

            cls_name = target.__name__


            if name in cls.DICT_LIKE:
                yield Interface(cls_name, InterfaceGroup.DictLike, name, doc)
            elif name == 'astype':
                yield Interface(cls_name, InterfaceGroup.Method, name, doc)
                if isinstance(obj, property): # an InterfaceAsType
                    field = '__getitem__'
                    display = f'{name}.{field}'
                    doc = cls.scrub_doc(getattr(InterfaceAsType, field).__doc__)
                    yield Interface(cls_name, InterfaceGroup.Method, display, doc)

            elif name.startswith('from_') or name == '__init__':
                yield Interface(cls_name, InterfaceGroup.Constructor, name, doc)
            elif name.startswith('to_'):
                yield Interface(cls_name, InterfaceGroup.Exporter, name, doc)
            elif name.startswith('iter_'):
                yield Interface(cls_name, InterfaceGroup.Iterator, name, doc)
                for field in cls.ATTR_ITER_NODE:
                    display = f'{name}(axis).{field}'
                    doc = cls.scrub_doc(getattr(IterNodeDelegate, field).__doc__)
                    yield Interface(cls_name, InterfaceGroup.Iterator, display, doc)
            elif name in cls.SELECTOR_ROOT:
                yield Interface(cls_name, InterfaceGroup.Selector, name, doc)
            elif name in cls.SELECTOR_COMPOUND:
                for field in cls.ATTR_SELECTOR_NODE:
                    display = f'{name}.{field}'
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

if __name__ == '__main__':

    print(InterfaceSummary.to_frame(Frame).display(DisplayConfigs.UNBOUND))