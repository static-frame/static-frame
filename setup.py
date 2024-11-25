import typing as tp
from codecs import open
from os import path

from setuptools import setup

DESCRIPTION = 'Immutable and statically-typeable DataFrames with runtime type and data validation.'

ROOT_DIR_FP = path.abspath(path.dirname(__file__))

def get_long_description() -> str:
    with open(path.join(ROOT_DIR_FP, 'README.rst'), encoding='utf-8') as f:
        msg = []
        collect = False
        start = -1
        for i, line in enumerate(f):
            if line.startswith('static-frame'):
                start = i + 2 # skip this line and the next
            if i == start:
                collect = True
            if collect:
                msg.append(line)

    return ''.join(msg).strip()


def get_version() -> str:
    with open(path.join(ROOT_DIR_FP, 'static_frame', '__init__.py'),
            encoding='utf-8') as f:
        for l in f:
            if l.startswith('__version__'):
                if '#' in l:
                    l = l.split('#')[0].strip()
                return l.split('=')[-1].strip()[1:-1]
    raise ValueError('__version__ not found!')


def _get_requirements(file_name: str) -> tp.Iterator[str]:
    with open(path.join(ROOT_DIR_FP, file_name)) as f:
        for line in f:
            line = line.strip()
            if line:
                yield line

def get_install_requires() -> tp.Iterator[str]:
    yield from _get_requirements('requirements.txt')

def get_extras_require() -> tp.Dict[str, tp.List[str]]:
    # For now, have only one group that installs all extras; in the future, can create specialized groups if necessary.
    return {'extras': list(_get_requirements('requirements-extras.txt'))}


setup(
    name='static-frame',
    version=get_version(),
    description=DESCRIPTION,
    long_description=get_long_description(),
    python_requires='>=3.9',
    install_requires=list(get_install_requires()),
    extras_require=get_extras_require(),
    url='https://github.com/static-frame/static-frame',
    author='Christopher Ariza',
    license='MIT',
    package_data={'static_frame': ['py.typed']},

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Information Analysis',
            'License :: OSI Approved :: MIT License',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'Programming Language :: Python :: 3.13',
            'Typing :: Typed',
            ],

    keywords='staticframe pandas numpy immutable array',
    packages=[
            'static_frame',
            'static_frame.core',
            'static_frame.test', # needed for doc generation
            'static_frame.test.unit', # needed for doc generation
            ],
    )

