import sys
import typing_extensions as tp

from invoke import task # pyright: ignore

#-------------------------------------------------------------------------------
@task
def clean(context):
    '''Clean doc and build artifacts
    '''
    context.run('rm -rf coverage.xml')
    context.run('rm -rf htmlcov')
    context.run('rm -rf doc/build')
    context.run('rm -rf build')
    context.run('rm -rf dist')
    context.run('rm -rf *.egg-info')
    context.run('rm -rf .coverage')
    context.run('rm -rf .mypy_cache')
    context.run('rm -rf .pytest_cache')
    context.run('rm -rf .hypothesis')
    context.run('rm -rf .ipynb_checkpoints')
    context.run('rm -rf .ruff_cache')


@task()
def doc(context):
    '''Build docs
    '''
    context.run(f'{sys.executable} doc/doc_build.py')


@task
def performance(context):
    '''Run performance tests.
    '''
    # NOTE: we do not get to see incremental output when running this
    cmd = 'python static_frame/performance/main.py --performance "*"'
    context.run(cmd)


@task
def interface(context, container=None, doc=False):
    '''
    Optionally select a container type to discover what API endpoints have examples.
    '''
    import static_frame as sf
    from static_frame.core.container import ContainerBase

    def subclasses(cls) -> tp.Iterator[tp.Type]:
        if cls.__name__ not in ('IndexBase', 'IndexDatetime'):
            yield cls
        for sub in cls.__subclasses__():
            yield from subclasses(sub)

    if not container:
        def frames():
            for cls in sorted(subclasses(ContainerBase),
                    key=lambda cls: cls.__name__):
                yield cls.interface.unset_index()
        f = sf.Frame.from_concat(frames(), axis=0, index=sf.IndexAutoFactory)
    else:
        f = getattr(sf, container).interface

    if not doc:
        f = f.drop['doc']
    dc = sf.DisplayConfig(cell_max_width_leftmost=99, cell_max_width=60, display_rows=99999, display_columns=99)
    print(f.display(dc))


#-------------------------------------------------------------------------------

@task
def test(context,
        unit=False,
        cov=False,
        pty=False,
        ):
    '''Run tests.
    '''
    if unit:
        fp = 'static_frame/test/unit static_frame/test/typing'
    else:
        fp = 'static_frame/test'

    # cmd = f'pytest -s --disable-pytest-warnings --tb=native {fp}'
    cmd = f'pytest -s --tb=native {fp}'
    if cov:
        cmd += ' --cov=static_frame --cov-report=xml'

    print(cmd)
    context.run(cmd, pty=pty)


@task
def testex(context,
        cov=False,
        pty=False,
        ):
    '''Test example generation
    '''
    cmd = 'pytest -s --tb=native doc/test_example_gen.py'
    if cov:
        cmd += ' --cov=static_frame --cov-report=xml'

    print(cmd)
    context.run(cmd, pty=pty)


@task
def coverage(context):
    '''
    Perform code coverage, and open report HTML.
    '''
    cmd = 'pytest -s --cov=static_frame/core --cov-report html'
    print(cmd)
    context.run(cmd)
    import webbrowser
    webbrowser.open('htmlcov/index.html')


@task
def mypy(context,
        pty=False,
         ):
    '''Run mypy static analysis.
    '''
    context.run('mypy --strict', pty=pty)

@task
def pyright(context,
        pty=False,
         ):
    '''Run pyright static analysis.
    '''
    context.run('pyright', pty=pty)

@task
def testtyping(context,
        pty=False,
         ):
    '''Run mypy on targetted typing tests
    '''
    context.run('pytest -s --tb=native static_frame/test/typing')
    context.run('pyright static_frame/test/typing', pty=pty)
    context.run('mypy --strict static_frame/test/typing', pty=pty)


@task
def isort(context):
    '''Run isort as a check.
    '''
    context.run('isort static_frame doc --check')

@task
def lint(context):
    '''Run pylint static analysis.
    '''
    context.run('pylint -f colorized static_frame')

@task(pre=(mypy, pyright, lint, isort)) # pyright: ignore
def quality(context):
    '''Perform all quality checks.
    '''

@task
def format(context):
    '''Run mypy static analysis.
    '''
    context.run('isort static_frame doc')


#-------------------------------------------------------------------------------

@task(pre=(clean,)) # pyright: ignore
def build(context):
    '''Build packages
    '''
    context.run(f'{sys.executable} setup.py sdist bdist_wheel')

@task(pre=(build,), post=(clean,)) # pyright: ignore
def release(context):
    context.run('twine upload dist/*')


