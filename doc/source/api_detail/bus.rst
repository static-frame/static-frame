
.. jinja:: ctx

    {% import 'macros.jinja' as macros %}

    {{ macros.api_detail(*interface['Bus'], examples_defined=examples_defined) }}

