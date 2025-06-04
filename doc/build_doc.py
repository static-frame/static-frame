import os
import sys

from sphinx.cmd.build import main

if __name__ == '__main__':
    doc_dir = os.path.abspath(os.path.dirname(__file__))

    doctrees_dir = os.path.join(doc_dir, 'build', 'doctrees')
    source_dir = os.path.join(doc_dir, 'source')
    build_dir = os.path.join(doc_dir, 'build', 'html')

    sys.path.append(os.path.join(os.path.dirname(doc_dir)))

    args = ['-E', '-b', 'html', '-d', doctrees_dir, source_dir, build_dir]
    status = main(args)

    if not status:
        import webbrowser

        webbrowser.open(os.path.join(build_dir, 'index.html'))
