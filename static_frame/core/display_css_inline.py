import typing as tp

CSSDict = tp.Dict[str, str]

class CSSInlineConfig(tp.NamedTuple):

    frame: tp.Callable[[], CSSDict]
    index: tp.Callable[[tp.Hashable], CSSDict]
    columns: tp.Callable[[tp.Hashable], CSSDict]
    values: tp.Callable[[tp.Tuple[int, int], tp.Any], CSSDict]



