import sys
import os
import typing as tp

import invoke

#-------------------------------------------------------------------------------

@invoke.task
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



@invoke.task()
def doc(context):
    '''Build docs
    '''
    context.run(f'{sys.executable} doc/doc_build.py')


@invoke.task
def performance(context):
    '''Run performance tests.
    '''
    # NOTE: we do not get to see incremental output when running this
    cmd = 'python static_frame/performance/main.py --performance "*"'
    context.run(cmd)


@invoke.task
def interface(context, container=None):
    '''
    Optionally select a container type to discover what API endpoints have examples.
    '''
    from static_frame.core.container import ContainerBase
    import static_frame as sf


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

    print(f.display_tall())

@invoke.task
def example(context, container=None):
    '''
    Discover API members that have a code example.
    '''
    from static_frame.core.display_color import HexColor
    from doc.source.conf import get_jinja_contexts

    contexts = get_jinja_contexts()
    defined = contexts['examples_defined']
    signatures = set()

    # discover all signatures; if it is defined, print in a darker color
    for name, cls, frame in contexts['interface'].values():
        for signature, row in frame.iter_tuple_items(axis=1):
            target = f'{name}-{row.signature_no_args}'
            signatures.add(target) # accumulate all signatures
            if container and name != container:
                continue
            if target in defined:
                print(HexColor.format_terminal(0x505050, target))
            else:
                print(target)

    for line in sorted(defined - signatures):
        print(HexColor.format_terminal(0x00ccff, line))


#-------------------------------------------------------------------------------

@invoke.task
def test(context,
        unit=False,
        cov=False,
        ):
    '''Run tests.
    '''
    if unit:
        fp = 'static_frame/test/unit'
    else:
        fp = 'static_frame/test'

    cmd = f'pytest -s --color no --disable-pytest-warnings --tb=native {fp}'

    if cov:
        cmd += ' --cov=static_frame --cov-report=xml'

    print(cmd)
    context.run(cmd)


@invoke.task
def coverage(context):
    '''
    Perform code coverage, and open report HTML.
    '''
    cmd = 'pytest -s --color no --disable-pytest-warnings --cov=static_frame/core --cov-report html'
    print(cmd)
    context.run(cmd)
    import webbrowser
    webbrowser.open('htmlcov/index.html')


@invoke.task
def mypy(context):
    '''Run mypy static analysis.
    '''
    context.run('mypy --strict')


@invoke.task
def lint(context):
    '''Run pylint static analysis.
    '''
    context.run('pylint -f colorized static_frame')

@invoke.task(pre=(mypy, lint))
def quality(context):
    '''Perform all quality checks.
    '''

#-------------------------------------------------------------------------------

@invoke.task(pre=(clean,))
def build(context):
    '''Build packages
    '''
    context.run(f'{sys.executable} setup.py sdist bdist_wheel')

@invoke.task(pre=(build,), post=(clean,))
def release(context):
    context.run('twine upload dist/*')


