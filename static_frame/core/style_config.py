import typing as tp

from static_frame.core.display_color import HexColor


if tp.TYPE_CHECKING:
    from static_frame.core.container import ContainerOperand #pylint: disable=W0611 #pragma: no cover

CSSDict = tp.Dict[str, str]


class StyleConfig:
    __slots__ = ('container',)

    def __init__(self, container: tp.Optional['ContainerOperand'] = None):
        self.container: tp.Optional['ContainerOperand'] = container

    def frame(self) -> str:
        '''
        Frame- (or table-) level styling.
        '''
        return ''

    def apex(self,
            value: tp.Any,
            coordinates: tp.Tuple[int, int],
            ) -> tp.Tuple[str, str]:
        '''
        Args:
            coordinates: negative integers to describe the apex space (i.e., the upper left corner formed when index and columns are displayed.
        Returns:
            A pair of the value to display, and a style information appropriate to the format.
        '''
        return str(value), ''

    def values(self,
            value: tp.Any,
            coordinates: tp.Tuple[int, int],
            ) -> tp.Tuple[str, str]:
        '''
        Returns:
            A pair of the value to display, and a style information appropriate to the format.
        '''
        return str(value), ''

    def index(self,
            label: tp.Hashable,
            ) -> tp.Tuple[str, str]:
        '''
        Returns:
            A pair of the value to display, and a style information appropriate to the format.
        '''
        return str(label), ''

    def columns(self,
            label: tp.Hashable,
            ) -> tp.Tuple[str, str]:
        '''
        Returns:
            A pair of the value to display, and a style information appropriate to the format.
        '''
        return str(label), ''


# Create an empty instance serve as default sentinal
STYLE_CONFIG_DEFAULT = StyleConfig()


class StyleConfigCSS(StyleConfig):

    COLOR_LIGHT_GREY = HexColor.get_html(0xe0e0e0)
    COLOR_OFF_BLACK = HexColor.get_html(0x2b2a2a)
    COLOR_GREY = HexColor.get_html(0xd1d2d4)
    COLOR_DARK_GREY = HexColor.get_html(0x898b8e)
    COLOR_WHITE = HexColor.get_html(0xffffff)
    COLOR_OFF_WHITE = HexColor.get_html(0xf2f2f2)

    FONT_SIZE = '14px'

    CSS_COMMON = dict(
            font_size=FONT_SIZE,
            border_width='1px',
            border_color=COLOR_DARK_GREY,
            border_style='solid',
            color=COLOR_OFF_BLACK,
    )

    @staticmethod
    def _dict_to_style(css_dict: CSSDict) -> str:
        '''
        Return a style attribute string containing all CSS in the CSSDict.
        '''
        style = ';'.join(f'{k.replace("_", "-")}:{v}' for k, v in css_dict.items())
        # NOTE: keep leading space to separate from tag
        return f' style="{style}"'


    def frame(self) -> str:
        '''
        Frame- (or table-) level styling.
        '''
        css = dict(
                border_collapse='collapse',
                border_width='1px',
                border_color=self.COLOR_DARK_GREY,
                border_style='solid',
                )
        return self._dict_to_style(css)

    def apex(self,
            value: tp.Any,
            coordinates: tp.Tuple[int, int],
            ) -> tp.Tuple[str, str]:

        css = dict(
                background_color=self.COLOR_LIGHT_GREY,
                font_weight='normal',
                padding='2px',
                border_width='0px',
                )
        return str(value), self._dict_to_style(css)

    def values(self,
            value: tp.Any,
            coordinates: tp.Tuple[int, int],
            ) -> tp.Tuple[str, str]:
        row, _ = coordinates

        def get_bg(row: int) -> str:
            if (row % 2) == 1:
                return self.COLOR_OFF_WHITE
            return self.COLOR_WHITE

        css = dict(
                background_color=get_bg(row),
                font_weight='normal',
                padding='2px',
                **self.CSS_COMMON,
                )
        return str(value), self._dict_to_style(css)

    def index(self,
            label: tp.Hashable,
            ) -> tp.Tuple[str, str]:
        css = dict(
                background_color=self.COLOR_GREY,
                font_weight='bold',
                **self.CSS_COMMON,
                )
        return str(label), self._dict_to_style(css)

    def columns(self,
            label: tp.Hashable,
            ) -> tp.Tuple[str, str]:

        css = dict(
                background_color=self.COLOR_GREY,
                font_weight='bold',
                **self.CSS_COMMON,
                )
        return str(label), self._dict_to_style(css)




def style_config_css_factory(
        style_config: tp.Optional[StyleConfig],
        container: 'ContainerOperand',
        ) -> tp.Optional[StyleConfig]:
    # let user set style_config to None to disable styling
    if style_config is STYLE_CONFIG_DEFAULT:
        # if given the base class, get the derived CSS class
        return StyleConfigCSS(container)
    if style_config is None:
        return None
    return style_config


