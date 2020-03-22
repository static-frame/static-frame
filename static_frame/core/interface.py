'''
Tools for documenting the SF interface.
'''
import typing as tp

import numpy as np

from static_frame.core.frame import Frame
from static_frame.core.bus import Bus

from static_frame.core.util import _DT64_S

from static_frame.core.container import ContainerBase
from static_frame.core.container import ContainerMeta
from static_frame.core.container import ContainerOperand

from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.index_base import IndexBase

from static_frame.core.index_datetime import IndexDate
from static_frame.core.index_datetime import IndexYearMonth
from static_frame.core.index_datetime import IndexYear

from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.display import Display

from static_frame.core.iter_node import IterNodeDelegate
from static_frame.core.iter_node import IterNodeNoArg
from static_frame.core.iter_node import IterNodeAxis
from static_frame.core.iter_node import IterNodeGroup
from static_frame.core.iter_node import IterNodeGroupAxis
from static_frame.core.iter_node import IterNodeDepthLevel
from static_frame.core.iter_node import IterNodeDepthLevelAxis
from static_frame.core.iter_node import IterNodeWindow

from static_frame.core.container import _UFUNC_BINARY_OPERATORS
from static_frame.core.container import _RIGHT_OPERATOR_MAP
from static_frame.core.container import _UFUNC_UNARY_OPERATORS

from static_frame.core.selector_node import InterfaceSelection2D
from static_frame.core.selector_node import InterfaceAssign2D

from static_frame.core.selector_node import InterfaceAsType
from static_frame.core.selector_node import InterfaceGetItem


class InterfaceGroup:
    Attribute = 'Attribute'
    Constructor = 'Constructor'
    DictLike = 'Dictionary-Like'
    Display = 'Display'
    Exporter = 'Exporter'
    Iterator = 'Iterator'
    Method = 'Method'
    OperatorBinary = 'Operator Binary'
    OperatorUnary = 'Operator Unary'
    Selector = 'Selector'


class Interface(tp.NamedTuple):
    cls: tp.Type[ContainerBase]
    group: InterfaceGroup
    signature: str
    doc: str
    reference: str = '' # a qualified name as a string for doc gen
    use_signature: bool = False
    reference_is_attr: bool = False
    reference_end_point: str = ''
    reference_end_point_is_attr: bool = False
    signature_no_args: str = ''


class InterfaceSummary:

    DOC_CHARS = 100

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
        '__setitem__',
        '__slots__',
        '__slotnames__',
        '__subclasshook__',
        '__weakref__',
        '__reduce__',
        '__reduce_ex__',
        '__sizeof__',
        }

    DICT_LIKE = {
        'get',
        'keys',
        'values',
        'items',
        '__contains__',
        '__iter__',
        '__reversed__'
        }

    DISPLAY = {
        'display',
        'display_tall',
        'display_wide',
        '__repr__',
        '__str__',
        'interface',
        }

    ATTR_ITER_NODE = (
        'apply',
        'apply_iter',
        'apply_iter_items',
        'apply_pool',
        'map_all',
        'map_all_iter',
        'map_all_iter_items',
        'map_any',
        'map_any_iter',
        'map_any_iter_items',
        'map_fill',
        'map_fill_iter',
        'map_fill_iter_items',
        )

    GETITEM = '__getitem__'

    # must all be members of InterfaceSelection2D
    ATTR_SELECTOR_NODE = ('__getitem__', 'iloc', 'loc',)
    ATTR_SELECTOR_NODE_ASSIGN = ('__getitem__', 'iloc', 'loc', 'bloc')
    ATTR_ASTYPE = ('__call__', '__getitem__')

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
        doc = doc.replace('`', '')
        doc = doc.replace(':py:meth:', '')
        doc = doc.replace(':obj:', '')
        doc = doc.replace('static_frame.', '')

        # split and join removes contiguous whitespace
        msg = ' '.join(doc.split())
        if len(msg) <= cls.DOC_CHARS:
            return msg
        return msg[:cls.DOC_CHARS].strip() + Display.ELLIPSIS


    @classmethod
    def get_instance(cls, target: tp.Type[ContainerBase]) -> ContainerBase:
        '''
        Get a sample instance from any ContainerBase; cache to only create one per life of process.
        '''
        if target not in cls._CLS_TO_INSTANCE_CACHE:
            if target is TypeBlocks:
                instance = target.from_blocks(np.array((0,)))
            elif target is Bus:
                f = Frame.from_elements((0,), name='frame')
                instance = target.from_frames((f,))
            elif issubclass(target, IndexHierarchy):
                instance = target.from_labels(((0,0),))
            elif issubclass(target, (IndexYearMonth, IndexYear, IndexDate)):
                instance = target(np.array((0,), dtype=_DT64_S))
            elif target in (ContainerOperand, ContainerBase, IndexBase):
                instance = target()
            elif issubclass(target, Frame):
                instance = target.from_elements((0,))
            else:
                instance = target((0,))
            cls._CLS_TO_INSTANCE_CACHE[target] = instance
        return cls._CLS_TO_INSTANCE_CACHE[target]

    @classmethod
    def name_obj_iter(cls, target: tp.Type[ContainerBase]):
        instance = cls.get_instance(target=target)

        for name_attr in dir(target.__class__): # get metaclass
            if name_attr == 'interface':
                # getting interface off of the class will recurse
                yield name_attr, None, ContainerBase.__class__.interface

        for name_attr in dir(target):
            if name_attr == 'interface':
                continue # skip, provided by class
            if not cls.is_public(name_attr):
                continue
            yield name_attr, getattr(instance, name_attr), getattr(target, name_attr)

    @classmethod
    def interrogate(cls,
            target: tp.Type[ContainerBase]
            ) -> tp.Iterator[Interface]:

        for name_attr, obj, obj_cls in sorted(cls.name_obj_iter(target)):
            # properties resdie on the class
            doc = ''
            reference = '' # reference attribute to use

            if isinstance(obj_cls, property):
                doc = cls.scrub_doc(obj_cls.__doc__)
            elif hasattr(obj, '__doc__'):
                doc = cls.scrub_doc(obj.__doc__)

            if hasattr(obj, '__name__'):
                name = obj.__name__
            else: # some attributes yield objects like arrays, Series, or Frame
                name = name_attr

            cls_name = target.__name__
            reference = f'{cls_name}.{name}'

            if name in cls.DICT_LIKE:
                signature = f'{name}()' if name != 'values' else name
                yield Interface(cls_name,
                        InterfaceGroup.DictLike,
                        signature,
                        doc,
                        reference,
                        signature_no_args=signature
                        )

            elif name in cls.DISPLAY:
                if name != 'interface':
                    signature = f'{name}()'
                    yield Interface(cls_name,
                            InterfaceGroup.Display,
                            signature,
                            doc,
                            reference,
                            signature_no_args=signature
                            )
                else: # interface attr
                    yield Interface(cls_name,
                            InterfaceGroup.Display,
                            name,
                            doc,
                            use_signature=True,
                            signature_no_args=name
                            )

            elif name == 'astype':
                # InterfaceAsType found on Frame, IndexHierarchy
                if isinstance(obj, InterfaceAsType):
                    for field in cls.ATTR_ASTYPE:
                        signature = f'{name}[]' if field == cls.GETITEM else f'{name}()'
                        reference_end_point = f'{InterfaceAsType.__name__}.{field}'
                        doc = cls.scrub_doc(getattr(InterfaceAsType, field).__doc__)
                        yield Interface(cls_name,
                                InterfaceGroup.Method,
                                signature,
                                doc,
                                reference,
                                use_signature=True,
                                reference_is_attr=True,
                                reference_end_point=reference_end_point,
                                signature_no_args=signature
                                )
                else: # Series, Index, astype is just a method
                    yield Interface(cls_name,
                            InterfaceGroup.Method,
                            name,
                            doc,
                            reference,
                            signature_no_args=signature
                            )
            elif name.startswith('from_') or name == '__init__':
                signature = f'{name}()'
                yield Interface(cls_name,
                        InterfaceGroup.Constructor,
                        signature,
                        doc,
                        reference,
                        signature_no_args=signature
                        )

            elif name.startswith('to_'):
                signature = f'{name}()'
                yield Interface(cls_name,
                        InterfaceGroup.Exporter,
                        signature,
                        doc,
                        reference,
                        signature_no_args=signature
                        )

            elif name.startswith('iter_'):
                # replace with inspect call
                reference_is_attr = True
                signature_no_args = f'{name}()'

                if isinstance(obj, IterNodeNoArg):
                    signature = f'{name}()'
                elif isinstance(obj, IterNodeAxis):
                    signature = f'{name}(axis)'
                elif isinstance(obj, IterNodeGroup):
                    signature = f'{name}()'
                elif isinstance(obj, IterNodeGroupAxis):
                    signature = f'{name}(key, axis)'
                elif isinstance(obj, IterNodeDepthLevel):
                    signature = f'{name}(depth_level)'
                elif isinstance(obj, IterNodeDepthLevelAxis):
                    signature = f'{name}(depth_level, axis)'
                elif isinstance(obj, IterNodeWindow):
                    signature = f'{name}(size, step, axis, ...)'
                else:
                    raise NotImplementedError() #pragma: no cover

                yield Interface(cls_name,
                        InterfaceGroup.Iterator,
                        signature,
                        doc,
                        reference,
                        use_signature=True,
                        reference_is_attr=True,
                        signature_no_args=signature_no_args,
                        )

                for field in cls.ATTR_ITER_NODE: # apply, map, etc
                    signature_sub = f'{signature}.{field}()'
                    signature_sub_no_args = f'{signature_no_args}.{field}()'

                    reference_end_point = f'{IterNodeDelegate.__name__}.{field}'
                    doc = cls.scrub_doc(getattr(IterNodeDelegate, field).__doc__)
                    yield Interface(cls_name,
                            InterfaceGroup.Iterator,
                            signature_sub,
                            doc,
                            reference,
                            use_signature=True,
                            reference_is_attr=True,
                            reference_end_point=reference_end_point,
                            signature_no_args=signature_sub_no_args
                            )

            elif isinstance(obj, InterfaceGetItem) or name == cls.GETITEM:
                if name != cls.GETITEM:
                    signature = f'{name}[]'
                    reference_is_attr = True
                else:
                    signature = f'[]'
                    reference_is_attr = False

                # signature = f'{name}[]' if name != cls.GETITEM else '[]'
                yield Interface(cls_name,
                        InterfaceGroup.Selector,
                        signature,
                        doc,
                        reference,
                        use_signature=True,
                        reference_is_attr=True,
                        signature_no_args=signature
                        )

            elif isinstance(obj, InterfaceSelection2D):
                for field in cls.ATTR_SELECTOR_NODE:
                    if field != cls.GETITEM:
                        signature = f'{name}.{field}[]'
                        reference_end_point_is_attr = True
                    else:
                        signature = f'{name}[]'
                        reference_end_point_is_attr = False

                    reference_end_point = f'{InterfaceSelection2D.__name__}.{field}'
                    doc = cls.scrub_doc(getattr(InterfaceSelection2D, field).__doc__)
                    yield Interface(cls_name,
                            InterfaceGroup.Selector,
                            signature,
                            doc,
                            reference,
                            use_signature=True,
                            reference_is_attr=True,
                            reference_end_point=reference_end_point,
                            reference_end_point_is_attr=reference_end_point_is_attr,
                            signature_no_args=signature
                            )

            elif isinstance(obj, InterfaceAssign2D):
                for field in cls.ATTR_SELECTOR_NODE_ASSIGN:
                    if field != cls.GETITEM:
                        signature = f'{name}.{field}[]'
                        reference_end_point_is_attr = True
                    else:
                        signature = f'{name}[]'
                        reference_end_point_is_attr = False

                    reference_end_point = f'{InterfaceAssign2D.__name__}.{field}'
                    doc = cls.scrub_doc(getattr(InterfaceAssign2D, field).__doc__)
                    yield Interface(cls_name,
                            InterfaceGroup.Selector,
                            signature,
                            doc,
                            reference,
                            use_signature=True,
                            reference_is_attr=True,
                            reference_end_point=reference_end_point,
                            reference_end_point_is_attr=reference_end_point_is_attr,
                            signature_no_args=signature
                            )

            elif callable(obj): # general methods
                signature = f'{name}()'
                if name_attr in _UFUNC_UNARY_OPERATORS:
                    yield Interface(cls_name,
                            InterfaceGroup.OperatorUnary,
                            signature,
                            doc,
                            reference,
                            signature_no_args=signature
                            )
                elif name_attr in _UFUNC_BINARY_OPERATORS or name_attr in _RIGHT_OPERATOR_MAP:
                    yield Interface(cls_name,
                            InterfaceGroup.OperatorBinary,
                            signature,
                            doc,
                            reference,
                            signature_no_args=signature
                            )
                else:
                    yield Interface(cls_name,
                            InterfaceGroup.Method,
                            signature,
                            doc,
                            reference,
                            signature_no_args=signature
                            )
            else: # attributes
                yield Interface(cls_name,
                        InterfaceGroup.Attribute,
                        name,
                        doc,
                        reference,
                        signature_no_args=name
                        )

    @classmethod
    def to_frame(cls,
            target: ContainerMeta,
            *,
            minimized: bool = True,
            ) -> Frame:
        '''
        Reduce to key fields.
        '''
        f = Frame.from_records(cls.interrogate(target), name=target.__name__)
        f = f.sort_values(('cls', 'group', 'signature'))
        f = f.set_index('signature', drop=True)
        if minimized:
            return f[['cls', 'group', 'doc']]
        return f


