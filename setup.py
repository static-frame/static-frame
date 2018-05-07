# Always prefer setuptools over distutils
from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path

# https://packaging.python.org/distributing/
# to deploy:
# rm -r dist
# python setup.py sdist
# python setup.py bdist_wheel
# twine upload dist/*

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='static-frame',
    version='0.1.0',

    description='Immutable structures for one- and two-dimensional calculations with labelled axis',
    long_description=long_description,

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
        ],

    keywords='staticframe pandas numpy immutable array',
    py_modules=['static_frame'], # no .py!

)
