import argparse
import os
import sys
import typing as tp

import nox

ARTIFACTS = (
    'coverage.xml',
    'htmlcov',
    'doc/build',
    'build',
    'dist',
    '*.egg-info',
    '.coverage',
    '.mypy_cache',
    '.pytest_cache',
    '.hypothesis',
    '.ipynb_checkpoints',
    '.ruff_cache',
)

# Make `nox` default to running tests if you just do `nox`
nox.options.sessions = ['test']


def do_clean(session: nox.Session) -> None:
    for artifact in sorted(ARTIFACTS):
        session.run('rm', '-rf', artifact, external=True)


def do_performance(session: nox.Session) -> None:
    # keep -v to see warnings; no build isolation to match your invoke cmd
    session.run(
        sys.executable,
        'static_frame/performance/main.py',
        '--performance',
        '*',
        external=True,
    )


# default is unit tests, for full do:
# nox -s test -- --full
# To show warnings:
# nox -s test --warnings
def do_test(session: nox.Session) -> None:
    unit = '--full' not in session.posargs
    cov = '--cov' in session.posargs
    warnings = '--warnings' in session.posargs

    fps = []
    fps.append('static_frame/test/unit')
    fps.append('static_frame/test/typing')
    if sys.version_info[:2] >= (3, 11):
        fps.append('static_frame/test/unit_forward')

    if not unit:
        fps.append('static_frame/test/integration')
        fps.append('static_frame/test/property')

    w_flag = '--disable-pytest-warnings'
    cmd = f'pytest -s --tb=native {w_flag if warnings else ""} {" ".join(fps)}'
    if cov:
        cmd += ' --cov=static_frame --cov-report=xml'

    session.run(
        *cmd.split(' '),
        external=True,
    )


def do_test_typing(session: nox.Session) -> None:
    """Run mypy on targetted typing tests"""
    for cmd in (
        'pytest -s --tb=native static_frame/test/typing',
        'pyright static_frame/test/typing',
        'mypy --strict static_frame/test/typing',
    ):
        session.run(*cmd.split(' '), external=True)


def do_test_ex(session: nox.Session) -> None:
    cov = '--cov' in session.posargs

    cmd = 'pytest -s --tb=native doc/test_example_gen.py'
    if cov:
        cmd += ' --cov=static_frame --cov-report=xml'

    session.run(*cmd.split(' '), external=True)


def do_lint(session: nox.Session) -> None:
    session.run(
        'ruff',
        'check',
        external=True,
    )


def do_mypy(session: nox.Session) -> None:
    session.run(
        'mypy',
        '--strict',
        external=True,
    )


def do_pyright(session: nox.Session) -> None:
    session.run(
        'pyright',
        external=True,
    )


def do_format(session: nox.Session) -> None:
    for cmd in ('ruff check --select I --fix', 'ruff format'):
        session.run(
            *cmd.split(' '),
            external=True,
        )


def do_format_check(session: nox.Session) -> None:
    for cmd in ('ruff check --select I', 'ruff format --check'):
        session.run(
            *cmd.split(' '),
            external=True,
        )


def do_coverage(session: nox.Session) -> None:
    session.run(
        *'pytest -s --cov=static_frame/core --cov-report html'.split(' '), external=True
    )
    import webbrowser

    webbrowser.open('htmlcov/index.html')


# NOTE: use `nox -s test` to launch a session


# nox -s interface -- --container Series
# nox -s interface -- --container Series --doc


@nox.session(python=False)  # use current environment
def interface(session):
    """
    Optionally select a container type to discover what API endpoints have examples.
    """
    sys.path.append(os.getcwd())

    import static_frame as sf
    from static_frame.core.container import ContainerBase

    parser = argparse.ArgumentParser()
    parser.add_argument('--container', type=str, default=None)
    parser.add_argument('--doc', action='store_true')
    options = parser.parse_known_args(session.posargs)[0]

    def subclasses(cls) -> tp.Iterator[tp.Type]:
        if cls.__name__ not in ('IndexBase', 'IndexDatetime'):
            yield cls
        for sub in cls.__subclasses__():
            yield from subclasses(sub)

    if not options.container:

        def frames():
            for cls in sorted(subclasses(ContainerBase), key=lambda cls: cls.__name__):
                yield cls.interface.unset_index()

        f = sf.Frame.from_concat(frames(), axis=0, index=sf.IndexAutoFactory)
    else:
        f = getattr(sf, options.container).interface

    if not options.doc:
        f = f.drop['doc']

    dc = sf.DisplayConfig(
        cell_max_width_leftmost=99,
        cell_max_width=60,
        display_rows=99999,
        display_columns=99,
    )
    print(f.display(dc))


@nox.session(python=False)  # use current environment
def clean(session):
    do_clean(session)


@nox.session(python=False)
def test(session):
    do_test(session)


@nox.session(python=False)
def test_typing(session):
    do_test_typing(session)


@nox.session(python=False)
def test_ex(session):
    do_test_ex(session)


@nox.session(python=False)
def lint(session):
    do_lint(session)


@nox.session(python=False)
def format(session):
    do_format(session)


@nox.session(python=False)
def mypy(session):
    do_mypy(session)


@nox.session(python=False)
def pyright(session):
    do_pyright(session)


@nox.session(python=False)
def quality(session):
    do_lint(session)
    do_format_check(session)
    do_mypy(session)
    do_pyright(session)


@nox.session(python=False)
def coverage(session):
    do_coverage(session)
