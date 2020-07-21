
.. jinja:: ctx

    {% import 'macros.jinja' as macros %}

    {{ macros.api_detail(*interface['Series'], examples_defined=examples_defined) }}

