import typing as tp

from static_frame.core.doc_str import doc_inject

if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame #pylint: disable=W0611 #pragma: no cover

class InterfaceMeta(type):
    '''Lowest level metaclass for providing interface property on class.
    '''

    @property #type: ignore
    @doc_inject()
    def interface(cls) -> 'Frame':
        '''{}'''
        from static_frame.core.interface import InterfaceSummary
        return InterfaceSummary.to_frame(cls) #type: ignore
