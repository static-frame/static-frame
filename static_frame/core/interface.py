'''
Tools for documenting the SF interface.
'''

from enum import Enum
from collections import namedtuple
import typing as tp

from static_frame.core.series import Series
from static_frame.core.frame import Frame
from static_frame.core.display import DisplayConfigs
from static_frame.core.container import ContainerMeta

Interface = namedtuple('Interface', ('group', 'name', 'doc'))


class InterfaceGroup:
    Attribute = 'attribute'
    Method = 'method'
    DictLike = 'dict_like'


DOC_CHARS = 40

EXCLUDE_PRIVATE = {
    '__class__',
    '__annotations__',
    '__doc__',
    '__delattr__',
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

    for name in sorted(dir(cls)):
        if name.startswith('_') and not name.startswith('__'):
            continue
        if name in EXCLUDE_PRIVATE:
            continue

        obj = getattr(cls, name)

        doc = ''
        if hasattr(obj, '__doc__'):
            if obj.__doc__:
                doc = obj.__doc__.strip().replace('\n', ' ')[:DOC_CHARS]

        if hasattr(obj, '__name__'):
            name = obj.__name__


        if callable(obj):
            yield Interface(InterfaceGroup.Method, name, doc)
        # print(name)
        else:
            yield Interface(InterfaceGroup.Attribute, name, doc)


if __name__ == '__main__':


    f = Frame.from_records(interrogate(Series))
    print(f.display(DisplayConfigs.UNBOUND))