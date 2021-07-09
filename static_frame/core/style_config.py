from re import X
import typing as tp

from static_frame.core.display_color import HexColor


if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame #pylint: disable=W0611 #pragma: no cover

CSSDict = tp.Dict[str, str]


class StyleConfig:
    __slots__ = ('container',)

    def __init__(self, container: tp.Optional['Frame'] = None):
        self.container: 'Frame' = container

    def frame(self) -> str:
        '''
        Frame- (or table-) level styling.
        '''
        pass

    def apex(self,
            coordinates: tp.Tuple[int, int],
            ) -> tp.Tuple[str, str]:
        '''
        Args:
            coordinates: negative integers to describe the apex space (i.e., the upper left corner formed when index and columns are displayed.
        Returns:
            A pair of the value to display, and a style information appropriate to the format.
        '''
        raise NotImplementedError()

    def values(self,
            coordinates: tp.Tuple[int, int],
            ) -> tp.Tuple[str, str]:
        '''
        Returns:
            A pair of the value to display, and a style information appropriate to the format.
        '''
        raise NotImplementedError()

    def index(self,
            label: tp.Hashable,
            ) -> tp.Tuple[str, str]:
        '''
        Returns:
            A pair of the value to display, and a style information appropriate to the format.
        '''
        raise NotImplementedError()

    def columns(self,
            label: tp.Hashable,
            ) -> tp.Tuple[str, str]:
        '''
        Returns:
            A pair of the value to display, and a style information appropriate to the format.
        '''
        raise NotImplementedError()


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

        def get_bg(row):
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

