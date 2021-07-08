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
        css = dict(border='1px solid', border_collapse='collapse'
        return self._dict_to_style(css)

    def values(self,
            value: tp.Any,
            coordinates: tp.Tuple[int, int],
            ) -> tp.Tuple[str, str]:
        row, _ = coordinates
        if row % 2:
            background = HexColor.get_html('darkslategrey')
        else:
            background = '#778899'

        css = dict(
                background_color=background,
                font_weight='normal',
                padding='2px',
                )
        return str(value), self._dict_to_style(css)

    def index(self,
            label: tp.Hashable,
            ) -> tp.Tuple[str, str]:
        css = dict(font_weight='bold')
        return str(label), self._dict_to_style(css)

    def columns(self,
            label: tp.Hashable,
            ) -> tp.Tuple[str, str]:
        css = dict(font_weight='bold')
        return str(label), self._dict_to_style(css)

