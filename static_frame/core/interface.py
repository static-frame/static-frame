'''
Tools for documenting the SF interface.
'''

# from enum import Enum
from collections import namedtuple
import typing as tp

from static_frame.core.series import Series
from static_frame.core.frame import Frame
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


DOC_CHARS = 40

EXCLUDE_PRIVATE = {
    '__class__',
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


def interrogate(cls) -> tp.Iterator[Interface]:

    for name_attr in sorted(dir(cls)):
        if name_attr.startswith('_') and not name_attr.startswith('__'):
            continue
        if name_attr in EXCLUDE_PRIVATE:
            continue

        obj = getattr(cls, name_attr)

        doc = ''
        if hasattr(obj, '__doc__'):
            if obj.__doc__:
                doc = obj.__doc__.strip().replace('\n', ' ')[:DOC_CHARS]

        if hasattr(obj, '__name__'):
            name = obj.__name__
        else:
            name = name_attr


        if callable(obj):
            if name_attr in _UFUNC_UNARY_OPERATORS:
                yield Interface(cls.__name__, InterfaceGroup.OperatorUnary, name, doc)
            elif name_attr in _UFUNC_BINARY_OPERATORS or name_attr in _RIGHT_OPERATOR_MAP   :
                yield Interface(cls.__name__, InterfaceGroup.OperatorBinary, name, doc)
            else:
                yield Interface(cls.__name__, InterfaceGroup.Method, name, doc)
        else:
            yield Interface(cls.__name__, InterfaceGroup.Attribute, name, doc)


if __name__ == '__main__':


    f = Frame.from_records(interrogate(Series))

    f = f.sort_values(('cls', 'group', 'name'))
    print(f.display(DisplayConfigs.UNBOUND))