import os
import sys
# import typing as tp
from collections import deque

HEADER = '.. NOTE: auto-generated file, do not edit'

def get_rst_import_api(macro_name: str, cls_name: str, ig_tag: str) -> str:
    '''
    Called once per cls interface group.

    Args:
        macro_name: either detail or overview
        cls_name: class Name to docuemnt
        ig_tag: the interface group to document as tag string
    '''
    # NOTE: we call the macro with `examples_defined`, `toc` as a kwargs, and then star-expand other args from the interface object mapping

    return f'''
.. jinja:: ctx

    {{% import 'macros.jinja' as macros %}}

    {{{{ macros.{macro_name}(examples_defined=examples_defined, toc=toc, *interface['{cls_name}']['{ig_tag}']) }}}}

'''

def get_rst_import_toc(doc_group, cls_name: str) -> str:
    return f'''
.. jinja:: ctx

    {{% import 'macros.jinja' as macros %}}

    {{{{ macros.{doc_group}_toc('{cls_name}', toc, interface_group_doc) }}}}

'''

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

    toc_dir_public = {}
    toc_dir_hidden = {}

    api_groups = ('api_overview', 'api_detail')

    # groups are also the names of the macros
    for group in api_groups:
        source_dir = os.path.join(doc_dir, 'source')
        group_dir = os.path.join(source_dir, group)

        toc_dir_public[group] = []
        toc_dir_hidden[group] = []

        for cls_name, records in contexts['interface'].items():

            file_name = f'{name_to_snake_case(cls_name)}.rst'
            fp = os.path.join(group_dir, file_name)
            rst = get_rst_import_toc(group, cls_name)
            with open(fp, 'w') as f:
                f.write(rst)

            toc_dir_public[group].append(fp.replace(source_dir + '/', '   '))

            for _, ig, ig_tag, frame in records.values():
                if len(frame) == 0:
                    continue

                file_name = f'{name_to_snake_case(cls_name)}-{ig_tag}.rst'
                fp = os.path.join(group_dir, file_name)

                rst = get_rst_import_api(group, cls_name, ig_tag)
                # print(rst)
                with open(fp, 'w') as f:
                    f.write(rst)

                toc_dir_hidden[group].append(fp.replace(source_dir + '/', '   '))

    # print to help create TOC in index.rst
    for group in api_groups:
        print(group)
        for name in toc_dir_public[group]:
            print(name)

    for group in api_groups:
        print(group)
        for name in toc_dir_hidden[group]:
            print(name)



if __name__ == '__main__':
    doc_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(os.path.dirname(doc_dir)))
    source_build()


