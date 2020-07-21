
.. jinja:: ctx

    {% import 'source/macros.jinja' as macros %}

    {{ macros.api_detail(*interface['Index'], examples_defined=examples_defined) }}

