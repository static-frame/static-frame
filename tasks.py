
import invoke
import sys

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
def test(context, unit=False):
    '''Run tests
    '''
    if unit:
        path = 'static_frame/test/unit'
    else:
        path = 'static_frame/test'

    context.run(f'pytest -s --color no --disable-pytest-warnings --tb=native {path}')


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

@invoke.task(pre=(clean, build))
def release(context):
    context.run('twine upload dist/*')


