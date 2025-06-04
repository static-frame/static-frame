from __future__ import annotations

import typing_extensions as tp

# -------------------------------------------------------------------------------
# https://www.w3.org/TR/css-color-3/#svg-color

_COLOR_NAME_X11 = {
    'aliceblue': 0xF0F8FF,
    'antiquewhite': 0xFAEBD7,
    'aqua': 0xFFFF,
    'aquamarine': 0x7FFFD4,
    'azure': 0xF0FFFF,
    'beige': 0xF5F5DC,
    'bisque': 0xFFE4C4,
    'black': 0x0,
    'blanchedalmond': 0xFFEBCD,
    'blue': 0xFF,
    'blueviolet': 0x8A2BE2,
    'brown': 0xA52A2A,
    'burlywood': 0xDEB887,
    'cadetblue': 0x5F9EA0,
    'chartreuse': 0x7FFF00,
    'chocolate': 0xD2691E,
    'coral': 0xFF7F50,
    'cornflowerblue': 0x6495ED,
    'cornsilk': 0xFFF8DC,
    'crimson': 0xDC143C,
    'cyan': 0xFFFF,
    'darkblue': 0x8B,
    'darkcyan': 0x8B8B,
    'darkgoldenrod': 0xB8860B,
    'darkgray': 0xA9A9A9,
    'darkgreen': 0x6400,
    'darkgrey': 0xA9A9A9,
    'darkkhaki': 0xBDB76B,
    'darkmagenta': 0x8B008B,
    'darkolivegreen': 0x556B2F,
    'darkorange': 0xFF8C00,
    'darkorchid': 0x9932CC,
    'darkred': 0x8B0000,
    'darksalmon': 0xE9967A,
    'darkseagreen': 0x8FBC8F,
    'darkslateblue': 0x483D8B,
    'darkslategray': 0x2F4F4F,
    'darkslategrey': 0x2F4F4F,
    'darkturquoise': 0xCED1,
    'darkviolet': 0x9400D3,
    'deeppink': 0xFF1493,
    'deepskyblue': 0xBFFF,
    'dimgray': 0x696969,
    'dimgrey': 0x696969,
    'dodgerblue': 0x1E90FF,
    'firebrick': 0xB22222,
    'floralwhite': 0xFFFAF0,
    'forestgreen': 0x228B22,
    'fuchsia': 0xFF00FF,
    'gainsboro': 0xDCDCDC,
    'ghostwhite': 0xF8F8FF,
    'gold': 0xFFD700,
    'goldenrod': 0xDAA520,
    'gray': 0x808080,
    'green': 0x8000,
    'greenyellow': 0xADFF2F,
    'grey': 0x808080,
    'honeydew': 0xF0FFF0,
    'hotpink': 0xFF69B4,
    'indianred': 0xCD5C5C,
    'indigo': 0x4B0082,
    'ivory': 0xFFFFF0,
    'khaki': 0xF0E68C,
    'lavender': 0xE6E6FA,
    'lavenderblush': 0xFFF0F5,
    'lawngreen': 0x7CFC00,
    'lemonchiffon': 0xFFFACD,
    'lightblue': 0xADD8E6,
    'lightcoral': 0xF08080,
    'lightcyan': 0xE0FFFF,
    'lightgoldenrodyellow': 0xFAFAD2,
    'lightgray': 0xD3D3D3,
    'lightgreen': 0x90EE90,
    'lightgrey': 0xD3D3D3,
    'lightpink': 0xFFB6C1,
    'lightsalmon': 0xFFA07A,
    'lightseagreen': 0x20B2AA,
    'lightskyblue': 0x87CEFA,
    'lightslategray': 0x778899,
    'lightslategrey': 0x778899,
    'lightsteelblue': 0xB0C4DE,
    'lightyellow': 0xFFFFE0,
    'lime': 0xFF00,
    'limegreen': 0x32CD32,
    'linen': 0xFAF0E6,
    'magenta': 0xFF00FF,
    'maroon': 0x800000,
    'mediumaquamarine': 0x66CDAA,
    'mediumblue': 0xCD,
    'mediumorchid': 0xBA55D3,
    'mediumpurple': 0x9370DB,
    'mediumseagreen': 0x3CB371,
    'mediumslateblue': 0x7B68EE,
    'mediumspringgreen': 0xFA9A,
    'mediumturquoise': 0x48D1CC,
    'mediumvioletred': 0xC71585,
    'midnightblue': 0x191970,
    'mintcream': 0xF5FFFA,
    'mistyrose': 0xFFE4E1,
    'moccasin': 0xFFE4B5,
    'navajowhite': 0xFFDEAD,
    'navy': 0x80,
    'oldlace': 0xFDF5E6,
    'olive': 0x808000,
    'olivedrab': 0x6B8E23,
    'orange': 0xFFA500,
    'orangered': 0xFF4500,
    'orchid': 0xDA70D6,
    'palegoldenrod': 0xEEE8AA,
    'palegreen': 0x98FB98,
    'paleturquoise': 0xAFEEEE,
    'palevioletred': 0xDB7093,
    'papayawhip': 0xFFEFD5,
    'peachpuff': 0xFFDAB9,
    'peru': 0xCD853F,
    'pink': 0xFFC0CB,
    'plum': 0xDDA0DD,
    'powderblue': 0xB0E0E6,
    'purple': 0x800080,
    'red': 0xFF0000,
    'rosybrown': 0xBC8F8F,
    'royalblue': 0x4169E1,
    'saddlebrown': 0x8B4513,
    'salmon': 0xFA8072,
    'sandybrown': 0xF4A460,
    'seagreen': 0x2E8B57,
    'seashell': 0xFFF5EE,
    'sienna': 0xA0522D,
    'silver': 0xC0C0C0,
    'skyblue': 0x87CEEB,
    'slateblue': 0x6A5ACD,
    'slategray': 0x708090,
    'slategrey': 0x708090,
    'snow': 0xFFFAFA,
    'springgreen': 0xFF7F,
    'steelblue': 0x4682B4,
    'tan': 0xD2B48C,
    'teal': 0x8080,
    'thistle': 0xD8BFD8,
    'tomato': 0xFF6347,
    'turquoise': 0x40E0D0,
    'violet': 0xEE82EE,
    'webgray': 0x808080,
    'wheat': 0xF5DEB3,
    'white': 0xFFFFFF,
    'whitesmoke': 0xF5F5F5,
    'yellow': 0xFFFF00,
    'yellowgreen': 0x9ACD32,
}


# -------------------------------------------------------------------------------

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
        return ((color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF)

    @classmethod
    def _diff(cls, color1: int, color2: int) -> int:
        (r1, g1, b1) = cls._rgb(color1)
        (r2, g2, b2) = cls._rgb(color2)
        return abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)

    @staticmethod
    def _get_ansi_to_hex_map() -> tp.Dict[int, int]:
        """
        Called once (lazily) to get the ANSI to hex color mapping. This will return a dictionary mapping integers 0 to 255 to corresponding hex values (as integers).
        """
        primary = (
            0x000000,
            0x800000,
            0x008000,
            0x808000,
            0x000080,
            0x800080,
            0x008080,
            0xC0C0C0,
        )

        bright = (
            0x808080,
            0xFF0000,
            0x00FF00,
            0xFFFF00,
            0x0000FF,
            0xFF00FF,
            0x00FFFF,
            0xFFFFFF,
        )

        colors = {}

        for index, color in enumerate(primary + bright):
            colors[index] = color

        intensities = (0x00, 0x5F, 0x87, 0xAF, 0xD7, 0xFF)

        c = 16
        for i in intensities:
            color = i << 16
            for j in intensities:
                color &= ~(0xFF << 8)
                color |= j << 8
                for k in intensities:
                    color &= ~0xFF
                    color |= k
                    colors[c] = color
                    c += 1

        grayscale_start = 0x08
        grayscale_end = 0xF8
        grayscale_step = 10
        c = 232
        for hex_int in range(grayscale_start, grayscale_end, grayscale_step):
            color = (hex_int << 16) | (hex_int << 8) | hex_int
            colors[c] = color
            c += 1

        return colors

    @staticmethod
    def _hex_str_to_int(hex_color: str) -> int:
        """
        Convert string hex representations, color names, to hex int.
        """
        hex_str = hex_color.strip().lower()
        if hex_str.startswith('#'):
            hex_str = hex_str[1:]
        elif hex_str.startswith('0x'):
            hex_str = hex_str[2:]
        else:  # will raise key error
            return _COLOR_NAME_X11[hex_str]
        return int(hex_str, 16)

    @classmethod
    def _to_ansi(cls, hex_color: tp.Union[int, str]) -> int:
        """
        Find the nearest ANSI color given the hex value, encoded either as string or integer.
        """

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
    def format_terminal(cls, hex_color: tp.Union[int, str], text: str) -> str:
        """
        Given a hex color and text, return a string formatted for ANSI colors
        """
        ansi = cls._to_ansi(hex_color)
        return '\033[38;5;{ansi:d}m{text:s}\033[0m'.format(ansi=ansi, text=text)

    @classmethod
    def get_html(
        cls,
        hex_color: tp.Union[int, str],
    ) -> str:
        """
        Given a hex color and text, return an html color string
        """
        if isinstance(hex_color, str):
            hex_int = cls._hex_str_to_int(hex_color)
        else:
            hex_int = hex_color
        return '#' + format(hex_int, 'x')

    @classmethod
    def format_html(
        cls,
        hex_color: tp.Union[int, str],
        text: str,
    ) -> str:
        """
        Given a hex color and text, return a string formatted for ANSI colors
        """
        color = cls.get_html(hex_color)
        return '<span style="color: {color}">{text}</span>'.format(color=color, text=text)
