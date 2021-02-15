
Contributing
*******************

StaticFrame welcomes contributions to code and documentation via GitHub pull requests. For ideas on what to contribute, please see open issues on GitHub, particularly those marked "good first issue."

https://github.com/InvestmentSystems/static-frame/issues

If you have an idea for a new feature for which there is not already an issue, please create an issue first, before beginning development, so that it can be discussed.


Preparing an Environment
-------------------------------

To prepare a StaticFrame repository and environment, follow the following steps.

Clone the git repository (or your fork)::

    git clone https://github.com/InvestmentSystems/static-frame.git

Create a virtual environment with StaticFrame's requirements for testing::

    cd static-frame
    python3 -m venv .env-sf
    source .env-sf/bin/activate
    pip install -r requirements-test.txt



Running Tests & Static Analysis
-----------------------------------------

.. note::

    Running StaticFrame integration tests may clear your clipboard. This is an artifact of using Python's ``Tk`` for clipboard interaction in :obj:`Frame.to_clipboard()` and :obj:`Frame.from_clipboard()`.


PyTest can be used to run StaticFrame tests. Alternatively, Invoke (installed via ``requirements-test.txt``) can be used to run tests and static analysis. To run all test, enter the following::

    invoke test

When iterating on code, running fast unit tests is generally sufficient::

    invoke test -u

To run MyPy and Pylint static analysis, use the following::

    invoke mypy
    invoke lint

All tests and static analysis are run via GitHub Actions on pull requests, and all tests and static analysis must pass for a PR to be accepted.



