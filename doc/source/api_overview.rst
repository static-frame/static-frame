.. _api-overview:

API Overview
===============================


For each container, the complete public API is presented below. Note that interface endpoints are expanded to show all interface sub components.

This is an overview for quick reference; for detailed documentation, see :ref:`api-detail`.


.. jinja:: ctx

    {% for name, cls, frame in interface %}

    {{ name }}
    -------------------------------------------------


    {% for group, frame_sub in frame.iter_group_items('group', axis=0) %}

    .. _api-overview-{{ name }}-{{ group }}:

    {{ name }}: {{ group }}
    ..........................................................

    API Detail: :ref:`api-detail-{{ name }}-{{ group }}`

    .. csv-table::
        :header-rows: 0

        {% for label, row in frame_sub.iter_tuple_items(axis=1) -%}
            {% if label == "[]" -%}
            ":obj:`static_frame.{{name}}{{label}}`", "{{row.doc}}"
            {% else -%}
            ":obj:`static_frame.{{name}}.{{label}}`", "{{row.doc}}"
            {% endif -%}
        {% endfor %}

    {% endfor %}

    {% endfor %}




