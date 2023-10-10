
.. jinja:: ctx

    {% import 'macros.jinja' as macros %}

    {{ macros.api_detail(examples_defined=examples_defined, toc=toc, *interface['IndexHourGO']['accessor_type_clinic']) }}

