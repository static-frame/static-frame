import os
import sphinx

from RALib.core import top_level_tool
from RALib.core.environment import EnvironmentFactory
env = EnvironmentFactory.from_tlt(top_level_tool.package)

if __name__ == '__main__':

    doc_dir = os.path.join(env.get_imdev_root_dir(),
            'package/public/static_frame/doc')

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
