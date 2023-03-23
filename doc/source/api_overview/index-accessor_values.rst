
.. jinja:: ctx

    {% import 'macros.jinja' as macros %}

    {{ macros.api_overview(examples_defined=examples_defined, toc=toc, *interface['Index']['accessor_values']) }}

