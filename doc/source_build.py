import os
import sys
# import typing as tp



HEADER = '.. NOTE: auto-generated file, do not edit'


def get_rst_import(group: str, name: str, ig_tag: str) -> str:
    '''
    Args:
        group: either detail or overview
        name: class Name to docuemnt
        ig_tag: the interface group to document as tag string
    '''
    return f'''
.. jinja:: ctx

    {{% import 'macros.jinja' as macros %}}

    {{{{ macros.{group}(examples_defined=examples_defined, *interface[('{name}', '{ig_tag}')]) }}}}

'''

# def get_rst_embed(group: str, name: str) -> str:
#     doc_dir = os.path.abspath(os.path.dirname(__file__))
#     fp = os.path.join(doc_dir, 'source', 'macros.jinja')

#     with open(fp) as f:
#         macro_lines = f.readlines()

#     lines = [HEADER]
#     lines.append('''
# .. jinja:: ctx
# ''')

#     for line in macro_lines:
#         lines.append('    ' + line.rstrip())

#     lines.append(f'''
#     {{{{ {group}(examples_defined=examples_defined, *interface['{name}']) }}}}
# ''')
#     return '\n'.join(lines)



def name_to_snake_case(name: str):
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
    return ''.join(name_chars)

def source_build() -> None:
    from doc.source.conf import get_jinja_contexts


    doc_dir = os.path.abspath(os.path.dirname(__file__))

    contexts = get_jinja_contexts()

    for group in ('api_overview', 'api_detail'):
        source_dir = os.path.join(doc_dir, 'source')
        group_dir = os.path.join(source_dir, group)

        for name, cls, ig, ig_tag, frame in contexts['interface'].values():
            if len(frame) == 0:
                continue

            file_name = f'{name_to_snake_case(name)}-{ig_tag}.rst'
            fp = os.path.join(group_dir, file_name)

            # if not os.path.exists(fp):
            #     raise RuntimeError(f'must create and add RST file {fp}')

            print(fp.replace(source_dir + '/', '   '))
            rst = get_rst_import(group, name, ig_tag)

            # print(rst)
            with open(fp, 'w') as f:
                f.write(rst)



if __name__ == '__main__':
    doc_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(os.path.dirname(doc_dir)))
    source_build()


