
from static_frame.core.doc_str import doc_inject


class InterfaceMeta(type):
    '''Lowest level metaclass for providing interface property on class.
    '''

    @property #type: ignore
    @doc_inject()
    def interface(cls) -> 'Frame':
        '''{}'''
        from static_frame.core.interface import InterfaceSummary
        return InterfaceSummary.to_frame(cls) #type: ignore
