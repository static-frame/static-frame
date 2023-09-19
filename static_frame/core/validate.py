import typing as tp

from static_frame.core.container import ContainerBase


class ValidationError(TypeError):
    def __init__(self, pairs: tp.Sequence[tp.Tuple[tp.Any, tp.Any]]) -> None:
        pass

def validate_pair(value: tp.Any, hint: tp.Any) -> None:
    if hint is tp.Any:
        return

    if isinstance(hint, ContainerBase):
        if isinstance(value, hint):
            return

    elif isinstance(hint, tp._GenericAlias):
        # have a generic container
        pass

    elif isinstance(hint, (tp._UnionGenericAlias, tp.UnionType)):
        # _UnionGenericAlias comes from tp.Union, tp.UnionType from | expressions
        # tp.Optional returns a _UnionGenericAlias
        pass


