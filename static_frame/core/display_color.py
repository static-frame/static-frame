
import typing as tp


#-------------------------------------------------------------------------------
# https://www.w3.org/TR/css-color-3/#svg-color

_COLOR_NAME_X11 = {
    'aliceblue': 0xf0f8ff,
    'antiquewhite': 0xfaebd7,
    'aqua': 0xffff,
    'aquamarine': 0x7fffd4,
    'azure': 0xf0ffff,
    'beige': 0xf5f5dc,
    'bisque': 0xffe4c4,
    'black': 0x0,
    'blanchedalmond': 0xffebcd,
    'blue': 0xff,
    'blueviolet': 0x8a2be2,
    'brown': 0xa52a2a,
    'burlywood': 0xdeb887,
    'cadetblue': 0x5f9ea0,
    'chartreuse': 0x7fff00,
    'chocolate': 0xd2691e,
    'coral': 0xff7f50,
    'cornflowerblue': 0x6495ed,
    'cornsilk': 0xfff8dc,
    'crimson': 0xdc143c,
    'cyan': 0xffff,
    'darkblue': 0x8b,
    'darkcyan': 0x8b8b,
    'darkgoldenrod': 0xb8860b,
    'darkgray': 0xa9a9a9,
    'darkgreen': 0x6400,
    'darkgrey': 0xa9a9a9,
    'darkkhaki': 0xbdb76b,
    'darkmagenta': 0x8b008b,
    'darkolivegreen': 0x556b2f,
    'darkorange': 0xff8c00,
    'darkorchid': 0x9932cc,
    'darkred': 0x8b0000,
    'darksalmon': 0xe9967a,
    'darkseagreen': 0x8fbc8f,
    'darkslateblue': 0x483d8b,
    'darkslategray': 0x2f4f4f,
    'darkslategrey': 0x2f4f4f,
    'darkturquoise': 0xced1,
    'darkviolet': 0x9400d3,
    'deeppink': 0xff1493,
    'deepskyblue': 0xbfff,
    'dimgray': 0x696969,
    'dimgrey': 0x696969,
    'dodgerblue': 0x1e90ff,
    'firebrick': 0xb22222,
    'floralwhite': 0xfffaf0,
    'forestgreen': 0x228b22,
    'fuchsia': 0xff00ff,
    'gainsboro': 0xdcdcdc,
    'ghostwhite': 0xf8f8ff,
    'gold': 0xffd700,
    'goldenrod': 0xdaa520,
    'gray': 0x808080,
    'green': 0x8000,
    'greenyellow': 0xadff2f,
    'grey': 0x808080,
    'honeydew': 0xf0fff0,
    'hotpink': 0xff69b4,
    'indianred': 0xcd5c5c,
    'indigo': 0x4b0082,
    'ivory': 0xfffff0,
    'khaki': 0xf0e68c,
    'lavender': 0xe6e6fa,
    'lavenderblush': 0xfff0f5,
    'lawngreen': 0x7cfc00,
    'lemonchiffon': 0xfffacd,
    'lightblue': 0xadd8e6,
    'lightcoral': 0xf08080,
    'lightcyan': 0xe0ffff,
    'lightgoldenrodyellow': 0xfafad2,
    'lightgray': 0xd3d3d3,
    'lightgreen': 0x90ee90,
    'lightgrey': 0xd3d3d3,
    'lightpink': 0xffb6c1,
    'lightsalmon': 0xffa07a,
    'lightseagreen': 0x20b2aa,
    'lightskyblue': 0x87cefa,
    'lightslategray': 0x778899,
    'lightslategrey': 0x778899,
    'lightsteelblue': 0xb0c4de,
    'lightyellow': 0xffffe0,
    'lime': 0xff00,
    'limegreen': 0x32cd32,
    'linen': 0xfaf0e6,
    'magenta': 0xff00ff,
    'maroon': 0x800000,
    'mediumaquamarine': 0x66cdaa,
    'mediumblue': 0xcd,
    'mediumorchid': 0xba55d3,
    'mediumpurple': 0x9370db,
    'mediumseagreen': 0x3cb371,
    'mediumslateblue': 0x7b68ee,
    'mediumspringgreen': 0xfa9a,
    'mediumturquoise': 0x48d1cc,
    'mediumvioletred': 0xc71585,
    'midnightblue': 0x191970,
    'mintcream': 0xf5fffa,
    'mistyrose': 0xffe4e1,
    'moccasin': 0xffe4b5,
    'navajowhite': 0xffdead,
    'navy': 0x80,
    'oldlace': 0xfdf5e6,
    'olive': 0x808000,
    'olivedrab': 0x6b8e23,
    'orange': 0xffa500,
    'orangered': 0xff4500,
    'orchid': 0xda70d6,
    'palegoldenrod': 0xeee8aa,
    'palegreen': 0x98fb98,
    'paleturquoise': 0xafeeee,
    'palevioletred': 0xdb7093,
    'papayawhip': 0xffefd5,
    'peachpuff': 0xffdab9,
    'peru': 0xcd853f,
    'pink': 0xffc0cb,
    'plum': 0xdda0dd,
    'powderblue': 0xb0e0e6,
    'purple': 0x800080,
    'red': 0xff0000,
    'rosybrown': 0xbc8f8f,
    'royalblue': 0x4169e1,
    'saddlebrown': 0x8b4513,
    'salmon': 0xfa8072,
    'sandybrown': 0xf4a460,
    'seagreen': 0x2e8b57,
    'seashell': 0xfff5ee,
    'sienna': 0xa0522d,
    'silver': 0xc0c0c0,
    'skyblue': 0x87ceeb,
    'slateblue': 0x6a5acd,
    'slategray': 0x708090,
    'slategrey': 0x708090,
    'snow': 0xfffafa,
    'springgreen': 0xff7f,
    'steelblue': 0x4682b4,
    'tan': 0xd2b48c,
    'teal': 0x8080,
    'thistle': 0xd8bfd8,
    'tomato': 0xff6347,
    'turquoise': 0x40e0d0,
    'violet': 0xee82ee,
    'webgray': 0x808080,
    'wheat': 0xf5deb3,
    'white': 0xffffff,
    'whitesmoke': 0xf5f5f5,
    'yellow': 0xffff00,
    'yellowgreen': 0x9acd32,
}



#-------------------------------------------------------------------------------

# Based largely on broadinstitute/xtermcolor
# https://github.com/broadinstitute/xtermcolor
# Copyright (C) 2012 The Broad Institute

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:

class HexColor:

    _ANSI_TO_HEX = None
    _HEX_TO_ANSI_CACHE: tp.Dict[int, int] = {}

    @staticmethod
    def _rgb(color: int) -> tp.Tuple[int, int, int]:
        return ((color >> 16) & 0xff, (color >> 8) & 0xff, color & 0xff)

    @classmethod
    def _diff(cls, color1: int, color2: int) -> int:
        (r1, g1, b1) = cls._rgb(color1)
        (r2, g2, b2) = cls._rgb(color2)
        return abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)

    @staticmethod
    def _get_ansi_to_hex_map() -> tp.Dict[int, int]:
        '''
        Called once (lazzily) to get the ANSI to hex color mapping. This will return a dictionary mapping integers 0 to 255 to corresponding hex values (as integers).
        '''
        primary = (
            0x000000,
            0x800000,
            0x008000,
            0x808000,
            0x000080,
            0x800080,
            0x008080,
            0xc0c0c0
            )

        bright = (
            0x808080,
            0xff0000,
            0x00ff00,
            0xffff00,
            0x0000ff,
            0xff00ff,
            0x00ffff,
            0xffffff
            )

        colors = {}

        for index, color in enumerate(primary + bright):
            colors[index] = color

        intensities = (0x00, 0x5F, 0x87, 0xAF, 0xD7, 0xFF)

        c = 16
        for i in intensities:
            color = i << 16
            for j in intensities:
                color &= ~(0xff << 8)
                color |= j << 8
                for k in intensities:
                    color &= ~0xff
                    color |= k
                    colors[c] = color
                    c += 1

        grayscale_start = 0x08
        grayscale_end = 0xf8
        grayscale_step = 10
        c = 232
        for hex_int in range(grayscale_start, grayscale_end, grayscale_step):
            color = (hex_int << 16) | (hex_int << 8) | hex_int
            colors[c] = color
            c += 1

        return colors

    @staticmethod
    def _hex_str_to_int(hex_color: str) -> int:
        '''
        Convert string hex representations, color names, to hex int.
        '''
        hex_str = hex_color.strip().lower()
        if hex_str.startswith('#'):
            hex_str = hex_str[1:]
        elif hex_str.startswith('0x'):
            hex_str = hex_str[2:]
        else: # will raise key error
            return _COLOR_NAME_X11[hex_str]
        return int(hex_str, 16)

    @classmethod
    def _to_ansi(cls, hex_color: tp.Union[int, str]) -> int:
        '''
        Find the nearest ANSI color given the hex value, encoded either as string or integer.
        '''

        # normalize hex colors as integers pre cache
        if isinstance(hex_color, str):
            hex_color = cls._hex_str_to_int(hex_color)

        if hex_color not in cls._HEX_TO_ANSI_CACHE:

            if not cls._ANSI_TO_HEX:
                cls._ANSI_TO_HEX = cls._get_ansi_to_hex_map()

            diffs = {}
            for ansi, rgb in cls._ANSI_TO_HEX.items():
                # for all ansi, find distance, store as key
                diffs[cls._diff(rgb, hex_color)] = ansi

            ansi = diffs[min(diffs.keys())]
            cls._HEX_TO_ANSI_CACHE[hex_color] = ansi

        return cls._HEX_TO_ANSI_CACHE[hex_color]

    @classmethod
    def format_terminal(cls,
            hex_color: tp.Union[int, str],
            text: str) -> str:
        '''
        Given a hex color and text, return a string formatted for ANSI colors
        '''
        ansi = cls._to_ansi(hex_color)
        return '\033[38;5;{ansi:d}m{text:s}\033[0m'.format(
                ansi=ansi,
                text=text)

    @classmethod
    def get_html(cls,
            hex_color: tp.Union[int, str],
            ) -> str:
        '''
        Given a hex color and text, return an html color string
        '''
        if isinstance(hex_color, str):
            hex_int = cls._hex_str_to_int(hex_color)
        else:
            hex_int = hex_color
        return '#' + format(hex_int, 'x')

    @classmethod
    def format_html(cls,
            hex_color: tp.Union[int, str],
            text: str,
            ) -> str:
        '''
        Given a hex color and text, return a string formatted for ANSI colors
        '''
        color = cls.get_html(hex_color)
        return '<span style="color: {color}">{text}</span>'.format(
                color=color,
                text=text)

