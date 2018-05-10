import os
import sphinx

if __name__ == '__main__':
    doc_dir = os.path.abspath(os.path.dirname(__file__))
    doctrees_dir = os.path.join(doc_dir, 'build', 'doctrees')
    source_dir = os.path.join(doc_dir, 'source')
    build_dir = os.path.join(doc_dir, 'build', 'html')

    args = ['sphinx', '-E', '-b', 'html',
            '-d', doctrees_dir,
            source_dir,
            build_dir]
    status = sphinx.main(args)

    import webbrowser
    webbrowser.open(os.path.join(build_dir, 'index.html'))
