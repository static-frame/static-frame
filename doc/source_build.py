import os
# import typing as tp

from doc.source.conf import get_jinja_contexts


HEADER = '.. NOTE: auto-generated file, do not edit'


def get_rst_import(group: str, name: str) -> str:
    '''
    This approach works when we can import macros in this location. This does not yet work with ReadTheDocs.
    '''
    return f'''
.. jinja:: ctx

    {{% import 'doc/source/macros.jinja' as macros %}}

    {{{{ macros.{group}(examples_defined=examples_defined, *interface['{name}']) }}}}

'''

def get_rst_embed(group: str, name: str) -> str:
    '''
    This approach does not require importing the macro, and works with ReadTheDocs.
    '''
    doc_dir = os.path.abspath(os.path.dirname(__file__))
    fp = os.path.join(doc_dir, 'source', 'macros.jinja')

    with open(fp) as f:
        macro_lines = f.readlines()

    lines = [HEADER]
    lines.append('''
.. jinja:: ctx
''')

    for line in macro_lines:
        lines.append('    ' + line.rstrip())

    lines.append(f'''
    {{{{ {group}(examples_defined=examples_defined, *interface['{name}']) }}}}
''')
    return '\n'.join(lines)


def source_build() -> None:

    doc_dir = os.path.abspath(os.path.dirname(__file__))

    contexts = get_jinja_contexts()

    for group in ('api_detail', 'api_overview'):
        group_dir = os.path.join(doc_dir, 'source', group)

        for name, cls, frame in contexts['interface'].values():

            name_chars = []
            last_is_upper = False
            for i, char in enumerate(name):
                if char.isupper():
                    if i != 0 and not last_is_upper:
                        name_chars.append('_')
                    name_chars.append(char.lower())
                    last_is_upper = True
                else:
                    name_chars.append(char)
                    last_is_upper = False

            file_name = ''.join(name_chars) + '.rst'
            fp = os.path.join(group_dir, file_name)

            if not os.path.exists(fp):
                raise RuntimeError(f'must create and add RST file {fp}')

            print(fp)
            rst = get_rst_embed(group, name)
            # rst = get_rst_import(group, name)

            # print(rst)
            with open(fp, 'w') as f:
                f.write(rst)



if __name__ == '__main__':

    source_build()