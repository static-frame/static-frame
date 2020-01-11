import sys
import os

import invoke

#-------------------------------------------------------------------------------

@invoke.task
def clean(context):
    '''Clean doc and build artifacts
    '''
    context.run('rm -rf doc/build')
    context.run('rm -rf build')
    context.run('rm -rf dist')
    context.run('rm -rf *.egg-info')


@invoke.task(pre=(clean,))
def doc(context):
    '''Build docs
    '''
    context.run(f'{sys.executable} doc/doc_build.py')


#-------------------------------------------------------------------------------

@invoke.task
def test(context, unit=False, filename=None):
    '''Run tests
    '''
    if unit:
        fp = 'static_frame/test/unit'
    else:
        fp = 'static_frame/test'

    if filename:
        fp = os.path.join(fp, filename)

    cmd = f'pytest -s --color no --disable-pytest-warnings --tb=native {fp}'
    print(cmd)
    context.run(cmd)


@invoke.task
def mypy(context):
    '''Run mypy static analysis
    '''
    context.run('mypy --strict')

@invoke.task
def lint(context):
    '''Run pylint static analysis
    '''
    context.run('pylint static_frame')

@invoke.task(pre=(test, mypy, lint))
def integrate(context):
    '''Perform all continuous integration
    '''

#-------------------------------------------------------------------------------

@invoke.task(pre=(clean,))
def build(context):
    '''Build packages
    '''
    context.run(f'{sys.executable} setup.py sdist bdist_wheel')

@invoke.task(pre=(build,))
def release(context):
    context.run('twine upload dist/*')


