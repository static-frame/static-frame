# Always prefer setuptools over distutils
from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path

# https://packaging.python.org/distributing/
# to deploy:
# rm -r build
# rm -r dist
# python setup.py sdist
# python setup.py bdist_wheel
# twine upload dist/*

root_dir_fp = path.abspath(path.dirname(__file__))

def get_long_description():
    with open(path.join(root_dir_fp, 'README.rst'), encoding='utf-8') as f:
        return f.read()

def get_version():
    with open(path.join(root_dir_fp, 'static_frame', '__init__.py'), encoding='utf-8') as f:
        for l in f:
            if l.startswith('__version__'):
                return l.split('=')[-1].strip()[1:-1]

setup(
    name='static-frame',
    version=get_version(),
    description='Immutable structures for one- and two-dimensional calculations with labelled axis',
    long_description=get_long_description(),
    install_requires=['numpy>=1.14.2'],
    url='https://github.com/InvestmentSystems/static-frame',
    author='Christopher Ariza',
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Topic :: Software Development',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            ],

    keywords='staticframe pandas numpy immutable array',
    packages=[
            'static_frame',
            'static_frame.core'],
)
