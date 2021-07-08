import typing as tp

from static_frame.core.display_color import HexColor

CSSDict = tp.Dict[str, str]

class CSSInlineConfig(tp.NamedTuple):

    frame: tp.Callable[[], CSSDict]
    index: tp.Callable[[tp.Hashable], CSSDict]
    columns: tp.Callable[[tp.Hashable], CSSDict]
    values: tp.Callable[[tp.Tuple[int, int], tp.Any], CSSDict]

    def dict_to_style(self, css_dict: CSSDict) -> str:
        style = ';'.join(f'{k.replace("_", "-")}:{v}' for k, v in css_dict.items())
        return f'style="{style}"'


def values(labels) -> CSSDict:
    row, col = labels
    if row % 2:
        background = HexColor.get_html('darkslategrey')
    else:
        background = '#778899'
    return dict(
            background_color=background,
            font_weight='normal',
            padding='2px',
            )


CSSInlineConfigDefault = CSSInlineConfig(
    frame = lambda: dict(border='1px solid', border_collapse='collapse'),
    index = lambda label: dict(font_weight='bold'),
    columns = lambda label: dict(font_weight='bold'),
    values = values
)


