'''
Tools for documenting the SF interface.
'''

# from enum import Enum
from collections import namedtuple
import typing as tp

from static_frame.core.series import Series
from static_frame.core.frame import Frame
from static_frame.core.iter_node import IterNode
from static_frame.core.iter_node import IterNodeDelegate

from static_frame.core.display import DisplayConfigs
# from static_frame.core.container import ContainerMeta
from static_frame.core.container import _UFUNC_BINARY_OPERATORS
from static_frame.core.container import _RIGHT_OPERATOR_MAP

from static_frame.core.container import _UFUNC_UNARY_OPERATORS


Interface = namedtuple('Interface', ('cls', 'group', 'name', 'doc'))


class InterfaceGroup:
    Attribute = 'attribute'
    OperatorBinary = 'operator_binary'
    OperatorUnary = 'operator_unary'
    Method = 'method'
    DictLike = 'method_dict_like'
    Iterator = 'iterator'
    Selector = 'selector'


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
    '__init__',
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
    }

DICT_LIKE = {'keys', 'values', 'items'}

ATTR_ITER_NODE = ('apply', 'apply_iter', 'apply_iter_items', 'apply_pool')
ATTR_SELECTOR = ('__getitem__', 'iloc', 'loc')


def is_public(field: str) -> bool:
    if field.startswith('_') and not field.startswith('__'):
        return False
    if field in EXCLUDE_PRIVATE:
        return False
    return True

def scrub_doc(doc: tp.Optional[str]) -> str:
    if not doc:
        return ''
    return doc.strip().replace('\n', ' ')[:DOC_CHARS]

def interrogate(cls) -> tp.Iterator[Interface]:

    for name_attr in sorted(dir(cls)):
        if not is_public(name_attr):
            continue

        # this gets object off the class, not an instance
        obj = getattr(cls, name_attr)

        doc = ''
        if hasattr(obj, '__doc__'):
            doc = scrub_doc(obj.__doc__)

        if hasattr(obj, '__name__'):
            name = obj.__name__
        else:
            name = name_attr

        cls_name = cls.__name__


        if name in DICT_LIKE:
            yield Interface(cls_name, InterfaceGroup.DictLike, name, doc)
        elif name.startswith('iter_'):
            for field in ATTR_ITER_NODE:
                display = f'{name}(axis).{field}'
                doc = scrub_doc(getattr(IterNodeDelegate, field).__doc__)
                yield Interface(cls_name, InterfaceGroup.Iterator, display, doc)

        elif callable(obj):
            if name_attr in _UFUNC_UNARY_OPERATORS:
                yield Interface(cls_name, InterfaceGroup.OperatorUnary, name, doc)
            elif name_attr in _UFUNC_BINARY_OPERATORS or name_attr in _RIGHT_OPERATOR_MAP:
                yield Interface(cls_name, InterfaceGroup.OperatorBinary, name, doc)
            else:
                yield Interface(cls_name, InterfaceGroup.Method, name, doc)
        else:
            yield Interface(cls_name, InterfaceGroup.Attribute, name, doc)


if __name__ == '__main__':


    f = Frame.from_records(interrogate(Series))

    f = f.sort_values(('cls', 'group', 'name'))
    print(f.display(DisplayConfigs.UNBOUND))