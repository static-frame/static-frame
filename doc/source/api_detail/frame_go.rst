
.. jinja:: ctx

    {% import 'doc/source/macros.jinja' as macros %}

    {{ macros.api_detail(*interface['FrameGO'], examples_defined=examples_defined) }}

